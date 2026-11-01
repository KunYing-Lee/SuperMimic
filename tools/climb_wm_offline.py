"""Phase 3 gate: fit the climb's world model offline on collected transitions
(tools/climb_collect.py) and measure it on held-out ones, with or without the
route features. The model, loss and windows are train.py's.

Held out: the last shard of every source (mode and sigma). Metrics on held-out
open-loop rollouts (recorded actions, the model's own states):
  site error  each limb's site (FK of the predicted state against the true one)
              after 8 and 32 steps, by phase of that limb at the window's end:
              hanging (a hand the reference bears), landing / release (its
              bearing changed within LAND_W frames), feet (a bearing foot), free.
  contact     F1 of the contact head over the window's steps.
  cuts        the simulator's cuts (ClimbMDP.terminate, timers from zero at the
              window start) on predicted against true states: recall of the true
              cuts and the false-cut rate where the true state is not cut.

usage: python tools/climb_wm_offline.py <clip> <hbold data> <data dir> <out dir>
       [--variant geom|nogeom] [--updates U] [--batch B] [--seed k]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
parser = argparse.ArgumentParser()
parser.add_argument("clip")
parser.add_argument("data")
parser.add_argument("shards")
parser.add_argument("out")
parser.add_argument("--variant", default="geom", choices=["geom", "nogeom"])
parser.add_argument("--updates", type=int, default=50_000)
parser.add_argument("--batch", type=int, default=2048)
parser.add_argument("--window", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--eval_windows", type=int, default=8192)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--eval_only", action="store_true", help="evaluate the saved wm_<variant>.pt")
args = parser.parse_args()
os.environ["HBOLD_DATA"] = args.data

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from supermimic.buffer import ReplayBuffer  # noqa: E402
from supermimic.climb.parts import climb_parts  # noqa: E402
from supermimic.mdp import normalize_xi, unflatten_state  # noqa: E402
from supermimic.networks import WorldModel  # noqa: E402

DEV = "cuda:0"
LAND_W = 15  # frames (0.3 s): a limb's bearing changed this recently is landing or releasing
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
os.makedirs(args.out, exist_ok=True)
kin, motion, mdp = climb_parts(args.clip, DEV)
GEOM = args.variant == "geom"


def features(S, f):
    return mdp.wm_features(S, f=f) if GEOM else S[0].new_zeros(S[0].shape[0], 0)


# ---------------------------------------------------------------- data
files = sorted(glob.glob(os.path.join(args.shards, "*.pt")))
sources: dict[str, list[str]] = {}
for f in files:
    sources.setdefault(re.sub(r"_\d+\.pt$", "", os.path.basename(f)), []).append(f)
test_files = [fs[-1] for fs in sources.values()]
train_files = [f for f in files if f not in test_files]


def buffer_of(paths: list[str]) -> ReplayBuffer:
    """Shards side by side along the environment axis (windows never span two)."""
    shards = [torch.load(p) for p in paths]
    T = min(s["s"].shape[0] for s in shards)
    data = {k: torch.cat([s[k][:T] for s in shards], 1) for k in shards[0]}
    buf = ReplayBuffer(T, data["s"].shape[1], DEV, "cpu", extra={"contact": (4, torch.bool)})
    for k in buf.fields:
        if k in data:
            buf.data[k] = data[k].to(buf.data[k].dtype)
    buf.data["ep_step"] = torch.zeros_like(buf.data["ep_step"])
    # hbold's env does not end an episode at the clip's last frame: its reference
    # wraps and the robot is teleported, so a step out of the last frame is no
    # transition (tools/climb_collect.py now marks these itself).
    wrap = buf.data["f"] >= int(motion.clip_end[0])
    buf.data["valid"] &= ~wrap
    buf.data["done"] |= wrap
    buf.ptr, buf.size = 0, T
    buf.shard_envs = shards[0]["s"].shape[1]  # env columns per shard: column // this is the source
    return buf


t0 = time.time()
train, test = buffer_of(train_files), buffer_of(test_files)
print(f"train {train.T}x{train.N} from {len(train_files)} shards, test {test.T}x{test.N} from {test_files}; "
      f"{time.time() - t0:.0f} s", flush=True)

# ---------------------------------------------------------------- model
wm = WorldModel(goal_dim=1, extra_dim=mdp.wm_extra_dim if GEOM else 0, contact_dim=mdp.contact_dim).to(DEV)
opt = torch.optim.Adam(wm.parameters(), lr=args.lr)
with torch.no_grad():  # the normalizers, from training transitions as train.py's collection fits them
    for _ in range(100):
        w = train.sample_windows(4096, 1)
        ok = w["valid"][:, 0]
        S0, S1 = unflatten_state(w["s"][ok, 0]), unflatten_state(w["s1"][ok, 0])
        wm.in_norm.update(wm.inputs(S0, w["a"][ok, 0], normalize_xi(w["xi"][ok, 0]), features(S0, w["f"][ok, 0])))
        wm.out_norm.update(wm.targets(S0, S1))


def rollout(s, a, xi, f):
    """Open-loop: (states over the window, contact logits)."""
    S = unflatten_state(s[:, 0])
    preds, contacts = [], []
    for j in range(a.shape[1]):
        S, c = wm.predict(S, a[:, j], normalize_xi(xi[:, j]), features(S, f[:, j]))
        preds.append(S)
        contacts.append(c)
    return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6)), torch.stack(contacts, 1)


def loss_of(w):
    S_hat, c = rollout(w["s"], w["a"], w["xi"], w["f"])
    state_error = wm.state_error(S_hat, unflatten_state(w["s1"])).square().sum(-1)
    contact_error = F.binary_cross_entropy_with_logits(c, w["contact"].float(), reduction="none").sum(-1)
    m = w["mask"]
    return ((state_error + contact_error) * m).sum() / m.shape[0], (state_error[:, -1] * m[:, -1]).sum() / m[:, -1].sum()


step = torch.compile(loss_of)
log = []
ck_path = os.path.join(args.out, f"wm_{args.variant}.pt")
if args.eval_only:
    wm.load_state_dict(torch.load(ck_path, map_location=DEV)["wm"])
    args.updates = 0
for u in range(1, args.updates + 1):
    for g in opt.param_groups:
        g["lr"] = args.lr * (0.03 + 0.97 * 0.5 * (1 + torch.cos(torch.tensor(torch.pi * u / args.updates)).item()))
    loss, e_last = step(train.sample_windows(args.batch, args.window))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(wm.parameters(), 1.0, error_if_nonfinite=True)
    opt.step()
    if u % 500 == 0:
        log.append({"update": u, "loss": loss.item() / args.window, "err_last": e_last.item(),
                    "h": (time.time() - t0) / 3600})
        print(json.dumps(log[-1]), flush=True)
if not args.eval_only:
    torch.save({"wm": wm.state_dict(), "args": vars(args)}, ck_path)
wm.eval()


# ---------------------------------------------------------------- evaluation
def alive_windows(buf: ReplayBuffer, n: int, H: int):
    """Windows of H steps with no episode end before the last and every step valid
    (and, once replay_timers has run, each step's timers before it: ``aux``)."""
    out = []
    while sum(len(w["s"]) for w in out) < n:
        t0 = torch.randint(0, buf.T - H + 1, (4 * n,))
        env = torch.randint(0, buf.N, (4 * n,))
        rows = t0[:, None] + torch.arange(H)
        e = env[:, None].expand_as(rows)
        keys = ("s", "s1", "a", "xi", "done", "f", "valid", "contact", *(("aux",) if "aux" in buf.data else ()))
        w = {k: buf.data[k][rows, e].to(DEV) for k in keys}
        w["src"] = (env // buf.shard_envs).to(DEV)[:, None].expand_as(rows)
        keep = w["valid"].all(-1) & ~w["done"][:, :-1].any(-1)
        out.append({k: v[keep] for k, v in w.items()})
    return {k: torch.cat([w[k] for w in out])[:n] for k in out[0]}


def phases(f_end):
    """(n, 4) phase of each limb at frame f_end: 0 hanging, 1 landing, 2 release, 3 feet, 4 free."""
    b = motion.bearing
    now = b[f_end]
    before = b[(f_end - LAND_W).clamp(min=0)]
    changed = torch.stack([(b[(f_end - k).clamp(min=0)] != now) for k in range(1, LAND_W + 1)]).any(0)
    hand = torch.tensor([True, True, False, False], device=DEV)
    ph = torch.full_like(f_end[:, None].expand(-1, 4), 4)
    ph = torch.where(now & hand, 0, ph)
    ph = torch.where(now & ~hand, 3, ph)
    ph = torch.where(changed & now & ~before, 1, ph)
    ph = torch.where(changed & ~now & before, 2, ph)
    return ph


NAMES = ("hanging", "landing", "release", "feet", "free")


@torch.no_grad()
def site_errors(H: int) -> dict:
    out, total = {k: [] for k in NAMES}, []
    by_src = {os.path.basename(f)[:-3]: [] for f in test_files}
    w = alive_windows(test, args.eval_windows, H)
    for i in range(0, len(w["s"]), 1024):
        b = {k: v[i:i + 1024] for k, v in w.items()}
        S_hat, _ = rollout(b["s"], b["a"], b["xi"], b["f"])
        last = lambda S: tuple(x[:, -1] for x in S)  # noqa: E731
        p_hat = kin.site_poses(*last(S_hat))[0]
        p_true = kin.site_poses(*last(unflatten_state(b["s1"])))[0]
        err = (p_hat - p_true).norm(dim=-1)  # (n, 4)
        ph = phases(b["f"][:, -1] + 1)
        total.append(err.flatten())
        for k, name in enumerate(by_src):
            by_src[name].append(err[b["src"][:, -1] == k].flatten())
        for k, name in enumerate(NAMES):
            out[name].append(err[ph == k])
    res = {}
    for name, v in [("all", total), *out.items(), *((f"src:{k}", v) for k, v in by_src.items())]:
        v = torch.cat(v)
        res[name] = {"n": len(v), "median_m": float(v.median()) if len(v) else None,
                     "p90_m": float(v.quantile(0.9)) if len(v) else None}
    return res


@torch.no_grad()
def replay_timers(buf: ReplayBuffer) -> None:
    """Each transition's hold timers before it, rebuilt by stepping the task's cuts
    along every recorded trajectory from its episode's start (zero), so a window's
    cuts start where the simulator's were. A trajectory already running at the
    shard's first step starts from zero too (a little early)."""
    T, N = buf.T, buf.N
    aux = torch.zeros(T, N, mdp.aux_dim)
    for n0 in range(0, N, 8192):
        n = slice(n0, min(n0 + 8192, N))
        cur = torch.zeros(n.stop - n.start, mdp.aux_dim, device=DEV)
        for t in range(T):
            aux[t, n] = cur.cpu()
            S1 = unflatten_state(buf.data["s1"][t, n].to(DEV))
            f1 = (buf.data["f"][t, n].to(DEV) + 1).clamp(max=int(motion.clip_end[0]))  # wrap rows: invalid anyway
            cur, _ = mdp.terminate(S1, f1, cur)
            cur = torch.where(buf.data["done"][t, n].to(DEV)[:, None], 0.0, cur)
    buf.data["aux"] = aux


@torch.no_grad()
def contact_and_cuts(H: int = 8) -> dict:
    replay_timers(test)
    w = alive_windows(test, args.eval_windows, H)
    tp = fp = fn = 0
    cut_tp = cut_fp = cut_fn = cut_tn = 0
    for i in range(0, len(w["s"]), 1024):
        b = {k: v[i:i + 1024] for k, v in w.items()}
        S_hat, c = rollout(b["s"], b["a"], b["xi"], b["f"])
        pred, lab = c > 0, b["contact"]
        tp += int((pred & lab).sum()); fp += int((pred & ~lab).sum()); fn += int((~pred & lab).sum())
        S_true = unflatten_state(b["s1"])
        aux_p = aux_t = b["aux"][:, 0]
        for j in range(H):
            f1 = b["f"][:, j] + 1
            aux_p, cp = mdp.terminate(tuple(x[:, j] for x in S_hat), f1, aux_p)
            aux_t, ct = mdp.terminate(tuple(x[:, j] for x in S_true), f1, aux_t)
            cut_tp += int((cp & ct).sum()); cut_fp += int((cp & ~ct).sum())
            cut_fn += int((~cp & ct).sum()); cut_tn += int((~cp & ~ct).sum())
    f1_score = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"contact_f1": f1_score, "cut_recall": cut_tp / max(cut_tp + cut_fn, 1),
            "false_cut_rate": cut_fp / max(cut_fp + cut_tn, 1), "true_cuts": cut_tp + cut_fn}


res = {"variant": args.variant, "train_files": train_files, "test_files": test_files, "log": log,
       "sites_8": site_errors(8), "sites_32": site_errors(32), **contact_and_cuts(8)}
with open(os.path.join(args.out, f"eval_{args.variant}.json"), "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps({k: res[k] for k in ("sites_8", "sites_32", "contact_f1", "cut_recall", "false_cut_rate")},
                 indent=1), flush=True)
