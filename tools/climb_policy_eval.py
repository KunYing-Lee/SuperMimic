"""Phase 4: a trained climb policy in the simulator, deterministic, from the clip's
first frame and from uniform starts. Reports how far episodes get (frames past
the start, the share passing PASS_FRAME and completing the clip), and which cut
ended them (the torso, off_hold, stayed: read back from the timers).

usage: python tools/climb_policy_eval.py <clip> <hbold data> <checkpoint> <out.json> [--envs N] [--off_hold_cut m]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
parser = argparse.ArgumentParser()
parser.add_argument("clip")
parser.add_argument("data")
parser.add_argument("checkpoint")
parser.add_argument("out")
parser.add_argument("--envs", type=int, default=256)
parser.add_argument("--off_hold_cut", type=float, default=0.05, help="the final curriculum stage")
parser.add_argument("--pass_frame", type=int, default=617, help="the right hand's bar 6 hold ends here")
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
mdp.hold_cut = args.off_hold_cut
ck = torch.load(args.checkpoint, map_location=DEV, weights_only=False)
K, hist = ck["args"]["goal_frames"], ck["args"]["hist"]
pi = Policy(OBS_DIM, mdp.goal_dim(K), hist=hist).to(DEV)
pi.load_state_dict(ck["pi"])
pi.eval()
N = args.envs
env = ClimbSim(cfg, N, motion, mdp, clip_id=args.clip, eval_mode=True)
pi.action_low, pi.action_high = env.action_bounds()
end = int(motion.clip_end[0])


@torch.no_grad()
def run(frames: torch.Tensor) -> dict:
    """Every env from its start frame until its first cut or the clip's end."""
    env.reset(torch.arange(N, device=DEV), frames)
    env.refresh()
    o = mdp.add_noise(mdp.obs(env.state(), env.a_prev, env.xi, env.f))
    obs = o.unsqueeze(1).repeat(1, hist, 1)
    alive = torch.ones(N, dtype=torch.bool, device=DEV)
    reached = frames.clone()
    cause = torch.full((N,), -1, device=DEV)  # 0 torso, 1 off_hold, 2 stayed, 3 clip end
    while alive.any():
        out = env.step(pi(obs, mdp.goal(env.f, K, env.state())))  # eval_mode: no auto-reset
        reached = torch.where(alive, env.f, reached)
        ended = alive & out["done"]
        off = (env.aux[:, :4] > task.HOLD_TOLERANCE).any(-1)
        stay = (env.aux[:, 4:] > task.STAY_TOLERANCE).any(-1)
        c = torch.where(out["term"], torch.where(off, 1, torch.where(stay, 2, 0)), 3)
        cause = torch.where(ended, c, cause)
        alive &= ~out["done"]
        env.f.clamp_(max=end - 1)  # an ended env steps on in eval mode; keep its frame inside the clip
        o = mdp.add_noise(mdp.obs(env.state(), env.a_prev, env.xi, env.f))
        obs = torch.cat((obs[:, 1:], o.unsqueeze(1)), 1)
    gained = (reached - frames).float()
    names = ["torso", "off_hold", "stayed", "clip_end"]
    return {
        "frames_gained_mean": float(gained.mean()), "frames_gained_median": float(gained.median()),
        "pass_frame_share": float((reached >= args.pass_frame).float().mean()),
        "complete_share": float((reached >= end).float().mean()),
        "cause": {n: int((cause == k).sum()) for k, n in enumerate(names)},
    }


res = {"checkpoint": args.checkpoint, "off_hold_cut": args.off_hold_cut, "envs": N,
       "from_t0": run(torch.zeros(N, dtype=torch.long, device=DEV)),
       "from_uniform": run(torch.randint(0, end - 1, (N,), device=DEV))}
print(json.dumps(res, indent=1))
with open(args.out, "w") as f:
    json.dump(res, f, indent=1)
