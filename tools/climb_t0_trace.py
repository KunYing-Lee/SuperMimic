"""Why do episodes from the clip's first frame end where they do? Rolls a policy
from frame 0 (deterministic, a few envs) and prints, per step, each limb's
distance and turn from the reference, whether the reference bears on it, and
which off-hold condition holds: a bearing limb past the cut, any limb past the
free cut, or turned past the turn cut.

usage: python tools/climb_t0_trace.py <clip> <hbold data> <checkpoint> [--steps n] [--start f]
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
parser = argparse.ArgumentParser()
parser.add_argument("clip")
parser.add_argument("data")
parser.add_argument("checkpoint")
parser.add_argument("--steps", type=int, default=45)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--policy", default="trained", choices=["trained", "refpd"])
args = parser.parse_args()
os.environ["HBOLD_DATA"] = args.data

import torch  # noqa: E402
from hbold.climb import task  # noqa: E402

from supermimic.climb.parts import climb_parts  # noqa: E402
from supermimic.climb.sim import ClimbSim, climb_physics  # noqa: E402
from supermimic.env import make_sim  # noqa: E402
from supermimic.mdp import OBS_DIM  # noqa: E402
from supermimic.networks import Policy  # noqa: E402

DEV = "cuda:0"
torch.manual_seed(0)
cfg = make_sim(DEV)
cfg.physics = climb_physics(args.clip)
kin, motion, mdp = climb_parts(args.clip, DEV)
mdp.hold_cut = task.hold_cut(10**9)
ck = torch.load(args.checkpoint, map_location=DEV, weights_only=False)
K, hist = ck["args"]["goal_frames"], ck["args"]["hist"]
pi = Policy(OBS_DIM, mdp.goal_dim(K), hist=hist).to(DEV)
pi.load_state_dict(ck["pi"])
env = ClimbSim(cfg, 4, motion, mdp, clip_id=args.clip, eval_mode=True)
pi.action_low, pi.action_high = env.action_bounds()
env.reset(torch.arange(4, device=DEV), torch.full((4,), args.start, device=DEV))
env.refresh()
o = mdp.add_noise(mdp.obs(env.state(), env.a_prev, env.xi, env.f))
obs = o.unsqueeze(1).repeat(1, hist, 1)
names = ("LH", "RH", "LF", "RF")
cut = torch.tensor([mdp.hold_cut] * 2 + [mdp.hold_cut * task.FOOT_SCALE] * 2, device=DEV)
print(f"cut {mdp.hold_cut} (feet x{task.FOOT_SCALE}), free {task.FREE_CUT}, turn {task.TURN_CUT}")
with torch.no_grad():
    for _ in range(args.steps):
        S = env.state()
        if args.policy == "trained":
            a = pi(obs, mdp.goal(env.f, K, S))
        else:
            a = (motion.q[env.f + 1] - env.default_q - env.xi[:, 1:30]) / env.alpha
        out = env.step(a)
        f = env.f[0]
        F = mdp.features(env.state())
        far, turn = mdp._limbs(F, env.f)
        bear = motion.bearing[env.f]
        cells = []
        for k, n in enumerate(names):
            why = "".join(c for c, on in (("B", bool(bear[0, k] and far[0, k] > cut[k])),
                                          ("F", bool(far[0, k] > task.FREE_CUT)),
                                          ("T", bool(turn[0, k] > task.TURN_CUT))) if on)
            cells.append(f"{n} {100 * far[0, k]:5.1f}cm {turn[0, k]:4.2f}r {'b' if bear[0, k] else ' '}{why:3s}")
        print(f"f {int(f):4d} | " + " | ".join(cells) + f" | off_s {env.aux[0, :4].tolist()} done {bool(out['done'][0])}")
        o = mdp.add_noise(mdp.obs(env.state(), env.a_prev, env.xi, env.f))
        obs = torch.cat((obs[:, 1:], o.unsqueeze(1)), 1)
        if out["done"][0]:
            break
