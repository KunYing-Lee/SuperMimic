"""Phase 3 action-ranking audit: does the world model order actions as the
simulator does?

From held-out start states (tools/climb_collect.py shards), K action sequences
of H steps run side by side in one simulator batch (cloned states, so batch
composition is the same for all) and through the world model (open loop): the
recorded sequence and K-1 copies with Gaussian noise added. Each gets its
H-step discounted return under the task's own reward and cuts (timers from
zero at the start). Reported: over every start and every pair of sequences,
the share whose return difference the model gets the sign of; and over the
perturbations, the predicted against the real change from the recorded one.

usage: python tools/climb_rank_audit.py <clip> <hbold data> <data dir> <wm checkpoint> <out.json>
       [--starts M] [--sequences K] [--horizon H] [--noise s] [--source prefix]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
parser = argparse.ArgumentParser()
parser.add_argument("clip")
parser.add_argument("data")
parser.add_argument("shards")
parser.add_argument("checkpoint")
parser.add_argument("out")
parser.add_argument("--starts", type=int, default=256)
parser.add_argument("--sequences", type=int, default=8)
parser.add_argument("--horizon", type=int, default=8)
parser.add_argument("--noise", type=float, default=0.2)
parser.add_argument("--source", default="bm_s0_")
parser.add_argument("--gamma", type=float, default=0.99)
args = parser.parse_args()
os.environ["HBOLD_DATA"] = args.data

import torch  # noqa: E402

from supermimic.climb.parts import climb_parts  # noqa: E402
from supermimic.climb.sim import ClimbSim, climb_physics  # noqa: E402
from supermimic.env import make_sim  # noqa: E402
from supermimic.mdp import normalize_xi, unflatten_state  # noqa: E402
from supermimic.networks import WorldModel  # noqa: E402

DEV = "cuda:0"
torch.manual_seed(0)
M, K, H = args.starts, args.sequences, args.horizon
cfg = make_sim(DEV)
cfg.physics = climb_physics(args.clip)
kin, motion, mdp = climb_parts(args.clip, DEV)
ck = torch.load(args.checkpoint, map_location=DEV)
geom = ck["args"]["variant"] == "geom"
wm = WorldModel(goal_dim=1, extra_dim=mdp.wm_extra_dim if geom else 0, contact_dim=mdp.contact_dim).to(DEV)
wm.load_state_dict(ck["wm"])
wm.eval()

# Starts: the held-out shard's states whose episode runs on for H steps.
shard = torch.load(sorted(glob.glob(os.path.join(args.shards, f"{args.source}*.pt")))[-1])
T, N = shard["s"].shape[:2]
t = torch.randint(0, T - H, (8 * M,))
e = torch.randint(0, N, (8 * M,))
rows = t[:, None] + torch.arange(H)
ok = shard["valid"][rows, e[:, None]].all(-1) & ~shard["done"][rows[:, :-1], e[:, None]].any(-1)
ok &= shard["f"][t, e] + H <= int(motion.clip_end[0])  # the horizon stays inside the clip
t, e = t[ok][:M], e[ok][:M]
s0 = shard["s"][t, e].to(DEV)
xi = shard["xi"][t, e].to(DEV)
f0 = shard["f"][t, e].to(DEV)
base = shard["a"][t[:, None] + torch.arange(H), e[:, None]].to(DEV)  # (M, H, 29)

env = ClimbSim(cfg, M * K, motion, mdp, clip_id=args.clip)
low, high = env.action_bounds()
noise = args.noise * torch.randn(M, K, H, 29, device=DEV)
noise[:, 0] = 0.0
acts = torch.clamp(base[:, None] + noise, low, high).reshape(M * K, H, 29)  # start-major: start m, sequence k
rep = lambda x: x.repeat_interleave(K, 0)  # noqa: E731
S0 = tuple(rep(x) for x in unflatten_state(s0))
ids = torch.arange(M * K, device=DEV)


def discounted(r, cut):
    alive = torch.cumprod(1 - cut.float(), 1)
    c = torch.cat((torch.ones_like(alive[:, :1]), alive[:, :-1]), 1)
    disc = args.gamma ** torch.arange(H, device=DEV, dtype=torch.float32)
    return (disc * c * r).sum(1)


# The simulator.
env.set_state(ids, S0, rep(xi), rep(f0))
r_sim, c_sim = [], []
with torch.no_grad():
    for j in range(H):
        out = env.step(acts[:, j])
        r_sim.append(out["r"])
        c_sim.append(out["term"])
R_sim = discounted(torch.stack(r_sim, 1), torch.stack(c_sim, 1)).view(M, K)

# The world model, open loop, the same reward and cuts on its states.
r_wm, c_wm = [], []
with torch.no_grad():
    S, f, a_prev = S0, rep(f0), torch.zeros(M * K, 29, device=DEV)
    aux = torch.zeros(M * K, mdp.aux_dim, device=DEV)
    for j in range(H):
        extra = mdp.wm_features(S, f=f) if geom else S[0].new_zeros(M * K, 0)
        S = wm.step(S, acts[:, j], normalize_xi(rep(xi)), extra)
        f = f + 1
        r_wm.append(mdp.reward(S, f, acts[:, j], a_prev))
        aux, cut = mdp.terminate(S, f, aux)
        c_wm.append(cut)
        a_prev = acts[:, j]
R_wm = discounted(torch.stack(r_wm, 1), torch.stack(c_wm, 1)).view(M, K)

i, j = torch.triu_indices(K, K, 1)
d_sim, d_wm = R_sim[:, i] - R_sim[:, j], R_wm[:, i] - R_wm[:, j]
meaningful = d_sim.abs() > 1e-4  # pairs the simulator actually separates
agree = (torch.sign(d_sim) == torch.sign(d_wm))[meaningful]
delta_sim, delta_wm = R_sim[:, 1:] - R_sim[:, :1], R_wm[:, 1:] - R_wm[:, :1]
res = {
    "checkpoint": args.checkpoint, "variant": ck["args"]["variant"], "starts": int(len(t)), "sequences": K,
    "horizon": H, "noise": args.noise,
    "pair_sign_accuracy": float(agree.float().mean()), "pairs": int(meaningful.sum()),
    "perturbation_sign_accuracy": float((torch.sign(delta_sim) == torch.sign(delta_wm)).float().mean()),
    "return_sim_mean": float(R_sim.mean()), "return_wm_mean": float(R_wm.mean()),
    "return_bias": float((R_wm - R_sim).mean()), "return_mae": float((R_wm - R_sim).abs().mean()),
}
print(json.dumps(res, indent=1))
with open(args.out, "w") as fh:
    json.dump(res, fh, indent=1)
