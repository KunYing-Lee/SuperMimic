"""Does the world model see the opening as the simulator plays it? Rolls a policy
from a start frame in the simulator (deterministic, eval context), then from
each recorded state replays the recorded actions through the world model
(open loop) and lets the policy act on the model's own states (closed loop, as
train.py imagines), and compares each limb's distance from its reference and
the task's cuts on the model's states with the simulator's.

usage: python tools/climb_t0_model_gap.py <clip> <hbold data> <checkpoint> <out.json>
       [--steps n] [--start f] [--horizon H] [--every k]
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
parser.add_argument("--steps", type=int, default=80)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--every", type=int, default=4, help="model rollouts from every k-th recorded state")
args = parser.parse_args()
os.environ["HBOLD_DATA"] = args.data

import torch  # noqa: E402
from hbold.climb import task  # noqa: E402

from supermimic.climb.parts import climb_parts  # noqa: E402
from supermimic.climb.sim import ClimbSim, climb_physics  # noqa: E402
from supermimic.env import make_sim  # noqa: E402
from supermimic.mdp import OBS_DIM, flatten_state, normalize_xi, unflatten_state  # noqa: E402
from supermimic.networks import Policy, WorldModel  # noqa: E402

DEV = "cuda:0"
torch.manual_seed(0)
cfg = make_sim(DEV)
cfg.physics = climb_physics(args.clip)
kin, motion, mdp = climb_parts(args.clip, DEV)
ck = torch.load(args.checkpoint, map_location=DEV, weights_only=False)
K, hist, H = ck["args"]["goal_frames"], ck["args"]["hist"], args.horizon
mdp.hold_cut = task.hold_cut(ck["sim_steps"] // ck["args"]["num_envs"])
pi = Policy(OBS_DIM, mdp.goal_dim(K), hist=hist).to(DEV)
pi.load_state_dict(ck["pi"])
wm = WorldModel(goal_dim=mdp.goal_dim(K), extra_dim=mdp.wm_extra_dim, contact_dim=mdp.contact_dim).to(DEV)
wm.load_state_dict(ck["wm"])
env = ClimbSim(cfg, 1, motion, mdp, clip_id=args.clip, eval_mode=True)
pi.action_low, pi.action_high = env.action_bounds()
env.reset(torch.zeros(1, dtype=torch.long, device=DEV), torch.full((1,), args.start, device=DEV))
env.refresh()


def far_of(S, f):
    return mdp._limbs(mdp.features(S), f)[0][0]


def step_obs(o_hist, S, a, f):
    o = mdp.add_noise(mdp.obs(S, a, env.xi, f))
    return torch.cat((o_hist[:, 1:], o.unsqueeze(1)), 1)


# The simulator's episode, every state, action, observation history and timer.
states, actions, hists, auxes, frames, cut_at = [], [], [], [], [], None
with torch.no_grad():
    o = mdp.add_noise(mdp.obs(env.state(), env.a_prev, env.xi, env.f))
    o_hist = o.unsqueeze(1).repeat(1, hist, 1)
    for t in range(args.steps + H):
        S = env.state()
        a = pi(o_hist, mdp.goal(env.f, K, S))
        states.append(flatten_state(S))
        actions.append(a)
        hists.append(o_hist)
        auxes.append(env.aux.clone())
        frames.append(env.f.clone())
        out = env.step(a)
        o_hist = step_obs(o_hist, env.state(), a, env.f)
        if bool(out["done"][0]):
            cut_at = t + 1
            break
T = len(states) - 1  # the last recorded state has a successor only if the episode ran on
xi_n = normalize_xi(env.xi)
names = ("LH", "RH", "LF", "RF")
real_far = torch.stack([far_of(unflatten_state(s), f) for s, f in zip(states, frames)])

rows = []
with torch.no_grad():
    for t0 in range(0, max(T - 1, 1), args.every):
        h = min(H, T - t0)
        # open loop: the recorded actions through the model
        S = unflatten_state(states[t0])
        open_far = []
        for j in range(h):
            S = wm.step(S, actions[t0 + j], xi_n, mdp.wm_features(S, f=frames[t0 + j]))
            open_far.append(far_of(S, frames[t0 + j] + 1))
        # closed loop: the policy on the model's own states, train.py's imagination
        S, oh, a_prev, f, aux = unflatten_state(states[t0]), hists[t0], actions[t0 - 1] if t0 else torch.zeros_like(actions[0]), frames[t0], auxes[t0]
        imag_far, imag_cut = [], None
        for j in range(H):
            a = pi(oh, mdp.goal(f, K, S))
            S = wm.step(S, a, xi_n, mdp.wm_features(S, f=f))
            f = f + 1
            F = mdp.features(S)
            imag_far.append(mdp._limbs(F, f)[0][0])
            aux, cut = mdp.terminate(S, f, aux, F=F)
            if imag_cut is None and bool(cut[0]):
                imag_cut = j + 1
            oh = step_obs(oh, S, a, f)
        idx = [j for j in (0, 7, H - 1) if j < h]
        rows.append({
            "t": t0, "frame": int(frames[t0]),
            "real_far_cm": {f"+{j + 1}": [round(100 * float(x), 1) for x in real_far[t0 + j + 1]] for j in idx},
            "open_far_cm": {f"+{j + 1}": [round(100 * float(x), 1) for x in open_far[j]] for j in idx},
            "imag_far_cm": {f"+{j + 1}": [round(100 * float(x), 1) for x in imag_far[j]] for j in (0, 7, H - 1)},
            "imag_cut_after": imag_cut, "real_cut_after": None if cut_at is None else cut_at - t0,
        })

res = {"checkpoint": args.checkpoint, "hold_cut": mdp.hold_cut, "start": args.start, "real_steps": cut_at or len(states),
       "limbs": names, "rows": rows}
for r in rows:
    print(f"t {r['t']:3d} f {r['frame']:4d} | real cut in {r['real_cut_after']}, imagined cut in {r['imag_cut_after']}")
    for k in r["real_far_cm"]:
        print(f"    {k:>4s} real {r['real_far_cm'][k]}  open {r['open_far_cm'][k]}  imag {r['imag_far_cm'].get(k)}")
with open(args.out, "w") as stream:
    json.dump(res, stream, indent=1)
