"""Phase 4 launch gate: do policy updates the world model predicts to help, help?

From a checkpoint (policy and world model), batches of start states (task
resets at the sampler's frames). The imagined objective's gradient gives the
update direction; candidates step the policy along it by multiples of one
Adam-sized step (lr times the gradient's sign, its first-step magnitude).
For each candidate and the checkpoint itself, the H-step discounted return is
measured in imagination (the world model, train.py's rollout) and in the
simulator (cloned starts side by side in one batch, same observation noise),
so differences are paired. Reported: the share of candidates whose predicted
and real change from the checkpoint agree in sign, and imagined over real
return at the checkpoint.

usage: python tools/climb_update_audit.py <clip> <hbold data> <checkpoint> <out.json>
       [--batches B] [--starts M] [--horizon H] [--multiples ...]
"""

from __future__ import annotations

import argparse
import copy
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
parser.add_argument("--batches", type=int, default=16)
parser.add_argument("--starts", type=int, default=256)
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--multiples", type=float, nargs="+", default=[-4.0, -1.0, 1.0, 4.0])
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--off_hold_cut", type=float, default=None, help="default: the checkpoint's stage")
args = parser.parse_args()
os.environ["HBOLD_DATA"] = args.data

import torch  # noqa: E402
from hbold.climb import task  # noqa: E402

from supermimic.climb.parts import climb_parts  # noqa: E402
from supermimic.climb.sim import ClimbSim, climb_physics  # noqa: E402
from supermimic.env import make_sim  # noqa: E402
from supermimic.mdp import OBS_DIM, normalize_xi  # noqa: E402
from supermimic.networks import Policy, WorldModel  # noqa: E402

DEV = "cuda:0"
cfg = make_sim(DEV)
cfg.physics = climb_physics(args.clip)
kin, motion, mdp = climb_parts(args.clip, DEV)
ck = torch.load(args.checkpoint, map_location=DEV, weights_only=False)
K, hist, H, M = ck["args"]["goal_frames"], ck["args"]["hist"], args.horizon, args.starts
num_envs_train = ck["args"]["num_envs"]
mdp.hold_cut = args.off_hold_cut if args.off_hold_cut is not None else task.hold_cut(
    ck["sim_steps"] // num_envs_train)
pi = Policy(OBS_DIM, mdp.goal_dim(K), hist=hist).to(DEV)
pi.load_state_dict(ck["pi"])
wm = WorldModel(goal_dim=mdp.goal_dim(K), extra_dim=mdp.wm_extra_dim, contact_dim=mdp.contact_dim).to(DEV)
wm.load_state_dict(ck["wm"])
for p in wm.parameters():
    p.requires_grad_(False)
C = 1 + len(args.multiples)  # the checkpoint and each candidate
env = ClimbSim(cfg, M * C, motion, mdp, clip_id=args.clip)
low, high = env.action_bounds()
pi.action_low, pi.action_high = low, high
disc = args.gamma ** torch.arange(H, device=DEV, dtype=torch.float32)


def returns(r, cut):
    alive = torch.cumprod(1 - cut.float(), 1)
    c = torch.cat((torch.ones_like(alive[:, :1]), alive[:, :-1]), 1)
    return (disc * c * r).sum(1)


def imagined(policy, S, xi, f, seed):
    """train.py's imagination: the policy on noisy observations of model states."""
    torch.manual_seed(seed)
    xi_n = normalize_xi(xi)
    o = mdp.add_noise(mdp.obs(S, torch.zeros(len(f), 29, device=DEV), xi, f))
    obs, a_prev = o.unsqueeze(1).repeat(1, hist, 1), torch.zeros(len(f), 29, device=DEV)
    aux, feats, rs, cs = torch.zeros(len(f), mdp.aux_dim, device=DEV), mdp.features(S), [], []
    for _ in range(H):
        a = policy(obs, mdp.goal(f, K, S))
        S = wm.step(S, a, xi_n, mdp.wm_features(S, feats))
        f = f + 1
        feats = mdp.features(S)
        o = mdp.add_noise(mdp.obs(S, a, xi, f))
        obs = torch.cat((obs[:, 1:], o.unsqueeze(1)), 1)
        rs.append(mdp.reward(S, f, a, a_prev, F=feats))
        aux, cut = mdp.terminate(S, f, aux, F=feats)
        cs.append(cut)
        a_prev = a
    return returns(torch.stack(rs, 1), torch.stack(cs, 1))


@torch.no_grad()
def real(policies, S, xi, f, seed):
    """Each policy on its own copy of the starts, all in one simulator batch."""
    ids = torch.arange(M * C, device=DEV)
    rep = lambda x: x.repeat(C, *[1] * (x.dim() - 1))  # noqa: E731  policy-major copies
    env.set_state(ids, tuple(rep(x) for x in S), rep(xi), rep(f))
    torch.manual_seed(seed)
    o = mdp.add_noise(mdp.obs(env.state(), env.a_prev, env.xi, env.f))
    obs = o.unsqueeze(1).repeat(1, hist, 1)
    rs, cs = [], []
    for _ in range(H):
        g = mdp.goal(env.f, K, env.state())
        a = torch.cat([p(obs[k * M:(k + 1) * M], g[k * M:(k + 1) * M]) for k, p in enumerate(policies)])
        out = env.step(a)
        rs.append(out["r"])
        cs.append(out["term"])
        o = mdp.add_noise(mdp.obs(env.state(), env.a_prev, env.xi, env.f))
        obs = torch.cat((obs[:, 1:], o.unsqueeze(1)), 1)
    return returns(torch.stack(rs, 1), torch.stack(cs, 1)).view(C, M)


agree, total, ratios, rows = 0, 0, [], []
for b in range(args.batches):
    frames, _ = env.sample_frames(M)
    frames = frames.clamp(max=int(motion.clip_end[0]) - H)  # the horizon stays inside the clip
    env.reset(torch.arange(M, device=DEV), frames)
    env.refresh()
    S0 = tuple(x[:M].clone() for x in env.state())
    xi0, f0 = env.xi[:M].clone(), env.f[:M].clone()
    pi.zero_grad(set_to_none=True)
    J = imagined(pi, S0, xi0, f0, seed=b).mean()
    J.backward()
    candidates = []
    for m in args.multiples:
        cand = copy.deepcopy(pi)
        with torch.no_grad():
            for p, p0 in zip(cand.parameters(), pi.parameters()):
                p.add_(m * args.lr * torch.sign(p0.grad))  # ascent: + along the gradient
        candidates.append(cand)
    with torch.no_grad():
        J_im = torch.stack([imagined(p, S0, xi0, f0, seed=b).mean() for p in (pi, *candidates)])
    J_re = real([pi, *candidates], S0, xi0, f0, seed=b).mean(1)
    d_im, d_re = J_im[1:] - J_im[0], J_re[1:] - J_re[0]
    agree += int((torch.sign(d_im) == torch.sign(d_re)).sum())
    total += len(d_im)
    ratios.append(float(J_im[0] / J_re[0]))
    rows.append({"batch": b, "J_imagined": J_im.tolist(), "J_real": J_re.tolist()})
    print(json.dumps(rows[-1]), flush=True)
res = {"checkpoint": args.checkpoint, "horizon": H, "starts": M, "batches": args.batches,
       "multiples": args.multiples, "off_hold_cut": mdp.hold_cut,
       "update_sign_agreement": agree / total, "imagined_over_real_mean": sum(ratios) / len(ratios),
       "rows": rows}
print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
with open(args.out, "w") as f:
    json.dump(res, f, indent=1)
