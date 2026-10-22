"""SuperMimic training (Algorithm A.1): collect -> fit world model -> BPTT policy + critic update."""

from __future__ import annotations

import argparse
import math
import random
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--reset_noise", action=argparse.BooleanOptionalAction, default=False,
                    help="optional reset perturbations, disabled in the aligned configuration")
parser.add_argument("--max_episode_steps", type=int, default=None,
                    help="optional episode cap; default episodes end at failure or clip end")
parser.add_argument("--iterations", type=int, default=320_000)
parser.add_argument("--buffer_steps", type=int, default=1000)
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--goal_frames", type=int, default=10)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=10)
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--pi_batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--no_critic", action="store_true")
parser.add_argument("--warmup_steps", type=int, default=8, help="initial replay population; at least wm_window steps")
parser.add_argument("--wm_only_iters", type=int, default=0, help="paper Algorithm A.1 updates both models from iteration 0")
parser.add_argument("--lr_decay_iters", type=int, default=250_000)
parser.add_argument("--ckpt_every", type=int, default=2000)
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--motion_dir", type=str, default=os.path.join(REPO, "data", "lafan1_g1"))
parser.add_argument("--log_dir", type=str, default=os.path.join(REPO, "logs", "run"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--precision", default="tf32", choices=["tf32", "fp32"])
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--replay_device", default="cpu", help="exact float32 replay storage; samples move to --device")
parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--activation_checkpoint", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--headless", action="store_true", help="accepted for command compatibility; training is headless")
args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint

from supermimic.buffer import ReplayBuffer
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import CRITIC_EXTRA, INV_DIM, MDP, OBS_DIM, normalize_xi, unflatten_state
from supermimic.motion import MotionLib
from supermimic.networks import Critic, Policy, WorldModel
from supermimic.robot import MJCF_PATH, default_joint_pos_list
from supermimic.reproducibility import SCHEMA_VERSION, manifest, restore_rng, rng_state, validate_resume, write_json

if args.max_episode_steps is not None and args.max_episode_steps < 1:
    parser.error("max_episode_steps must be positive when specified")
if args.warmup_steps < args.wm_window or args.buffer_steps < args.wm_window:
    parser.error("warmup_steps and buffer_steps must each cover a world-model window")
if args.pi_batch < 1 or args.critic_minibatches < 1 or args.pi_batch * args.horizon % args.critic_minibatches:
    parser.error("critic minibatches must divide the complete imagined transition batch")
if args.utd <= 0 or args.log_every < 1 or args.ckpt_every < 1:
    parser.error("utd and logging/checkpoint intervals must be positive")
torch.backends.cuda.matmul.allow_tf32 = args.precision == "tf32"
torch.backends.cudnn.allow_tf32 = args.precision == "tf32"
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
dev = args.device
if not dev.startswith("cuda") or not torch.cuda.is_available():
    raise RuntimeError("mjlab training requires the selected CUDA device")
torch.cuda.set_device(dev)
sim_cfg = make_sim(dev, args)  # Before all CUDA tensor allocations.
os.makedirs(args.log_dir, exist_ok=True)
if not args.resume and any(os.path.exists(os.path.join(args.log_dir, name)) for name in ("manifest.json", "args.json", "train_log.csv", "last.pt")):
    raise FileExistsError("Fresh training requires a new run directory; use --resume for a saved run")

kin = G1Kinematics(dev)
motion = MotionLib.from_dir(args.motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
K, Hh, HZ = args.goal_frames, args.hist, args.horizon
GOAL_DIM = 73 * K
CRITIC_IN = INV_DIM + CRITIC_EXTRA + GOAL_DIM

run_manifest = manifest(REPO, args, motion, MJCF_PATH)
wm = WorldModel(goal_dim=GOAL_DIM).to(dev)
pi = Policy(OBS_DIM, GOAL_DIM, hist=Hh).to(dev)
critic = Critic(CRITIC_IN).to(dev)
opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
start_it, wall_offset, sim_steps, step_acc = 0, 0.0, 0, 0.0
resume_ck = None
if args.resume:
    resume_ck = torch.load(args.resume, map_location="cpu", weights_only=False)
    validate_resume(resume_ck, run_manifest)
    wm.load_state_dict(resume_ck["wm"])
    pi.load_state_dict(resume_ck["pi"])
    critic.load_state_dict(resume_ck["critic"])
    opt_wm.load_state_dict(resume_ck["opt_wm"])
    opt_pi.load_state_dict(resume_ck["opt_pi"])
    opt_v.load_state_dict(resume_ck["opt_v"])
    start_it, wall_offset, sim_steps = resume_ck["iteration"], resume_ck["wall_time"], resume_ck["sim_steps"]
    step_acc = resume_ck["step_acc"]
    if args.iterations <= start_it:
        raise ValueError("--iterations must exceed the saved iteration")


write_json(os.path.join(args.log_dir, "args.json"), vars(args))
write_json(os.path.join(args.log_dir, "manifest.json"), run_manifest)

def maybe_checkpoint(function, *inputs):
    if args.activation_checkpoint and torch.is_grad_enabled():
        return checkpoint(function, *inputs, use_reentrant=False, preserve_rng_state=True)
    return function(*inputs)


def _wm_step(S, a, xi, f1):
    predicted = wm.step(S, a, normalize_xi(xi))
    logits = wm.termination_logits(mdp.termination_features(predicted, f1, motion.goal(f1, K, predicted)))
    return predicted, logits


def _wm_rollout(s, a, xi, f):
    S = unflatten_state(s[:, 0])
    preds, logits = [], []
    for j in range(args.wm_window):
        S, logit = maybe_checkpoint(_wm_step, S, a[:, j], xi[:, j], f[:, j] + 1)
        preds.append(S)
        logits.append(logit)
    return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6)), torch.stack(logits, 1)


def _wm_err(S_hat, logits, s1, terminated, mask):
    # Eq. 1: raw squared state error plus BCE for the learned termination head.
    # Analytic observation/reward heads have no trainable regression loss.
    state_error = wm.state_error(S_hat, unflatten_state(s1)).square().sum(-1)
    termination_error = F.binary_cross_entropy_with_logits(logits, terminated.float(), reduction="none")
    loss = ((state_error + termination_error) * mask).sum() / s1.shape[0]
    return loss, state_error[:, -1], (termination_error * mask).sum() / mask.sum()


def _imag_step(S, o_hist, a_prev, f, xi, xi_n, g):
    a = pi(o_hist, g)
    S1 = wm.step(S, a, xi_n)
    f1 = f + 1
    o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    g1 = motion.goal(f1, K, S1)
    r = mdp.reward(S1, f1, a, a_prev)
    d = wm.termination_probability(mdp.termination_features(S1, f1, g1))
    return S1, a, o_hist, f1, g1, r, d


def _critic_input(S, f, xi_n, g):
    return torch.cat((mdp.critic_features(S, f, xi_n), g), -1)


compile_fn = torch.compile if args.compile else (lambda function: function)
wm_rollout = compile_fn(_wm_rollout)
wm_err = compile_fn(_wm_err)
imag_step = compile_fn(_imag_step)
critic_input_c = compile_fn(_critic_input)


def lr_at(it: int) -> float:
    if it >= args.lr_decay_iters:
        return args.lr * 0.03
    return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


# ------------------------------------------------------------------------------------------------
def wm_update():
    w = buffer.sample_windows(args.wm_batch, args.wm_window)
    S_hat, logits = wm_rollout(w["s"], w["a"], w["xi"], w["f"])
    loss, e_last, term_bce = wm_err(S_hat, logits, w["s1"], w["terminated"], w["mask"])
    opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(wm.parameters(), 1.0, error_if_nonfinite=True)
    opt_wm.step()
    m = w["mask"][:, -1]
    return loss.detach() / args.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1), term_bce.detach()


def sample_valid_starts(B: int):
    return buffer.sample_starts(B, Hh, replay_clip_end, HZ)


PROFILE = bool(int(os.environ.get("SM_PROFILE", "0")))


def policy_update():
    if PROFILE:
        torch.cuda.synchronize(); tq0 = time.time()
    s, h = sample_valid_starts(args.pi_batch)
    S = unflatten_state(s["s"])
    xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
    xi_n = normalize_xi(xi)
    with torch.no_grad():
        o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
    o_hist = mdp.add_noise(o_hist)
    set_requires_grad(wm, False)
    set_requires_grad(critic, False)
    B = args.pi_batch
    g0 = motion.goal(f, K, S)
    g = g0
    S0 = S
    Ss, Fs, Gs, Rs, Ds = [], [], [], [], []
    for t in range(HZ):
        S, a_prev, o_hist, f, g, r, d = maybe_checkpoint(imag_step, S, o_hist, a_prev, f, xi, xi_n, g)
        Ss.append(S), Fs.append(f), Gs.append(g), Rs.append(r), Ds.append(d)
    r, d = torch.stack(Rs), torch.stack(Ds)
    alive = torch.cumprod(1 - d, 0)  # c_{t+1}
    c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
    disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    J = (disc * c_t * r).sum(0)
    c = alive[-1]
    r_sum = (c_t * r).detach().sum(0)
    rews, dones = r.detach(), d.detach()
    if not args.no_critic:
        J = J + (args.gamma**HZ) * c * critic(_critic_input(S, f, xi_n, g))
    loss = -J.mean()
    if PROFILE:
        torch.cuda.synchronize(); tq1 = time.time()
    opt_pi.zero_grad(set_to_none=True)
    loss.backward()
    if PROFILE:
        torch.cuda.synchronize(); tq2 = time.time()
    gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0, error_if_nonfinite=True)
    nn.utils.clip_grad_value_(pi.parameters(), 0.1)
    opt_pi.step()
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
    v_loss = torch.zeros(())
    if not args.no_critic:
        with torch.no_grad():
            S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
            f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
            g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
            xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
            X = critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
        critic.norm.update(X)
        with torch.no_grad():
            V = critic(X)
            R = V[-1]
            targets = []
            for t in range(HZ - 1, -1, -1):
                R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                targets.append(R)
            targets = torch.stack(targets[::-1])  # (H, B)
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        mb = n // args.critic_minibatches
        for _ in range(args.critic_epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(args.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
                opt_v.step()
    if PROFILE:
        torch.cuda.synchronize()
        print(f"[prof] imagine fwd {1e3*(tq1-tq0):.0f} ms | bwd {1e3*(tq2-tq1):.0f} ms | critic {1e3*(time.time()-tq2):.0f} ms", flush=True)
    return loss.detach(), v_loss.detach(), (r_sum / HZ / 0.1).mean(), gn.detach(), c.mean().detach()


# ------------------------------------------------------------------------------------------------
env = TrackingSim(sim_cfg, args.num_envs, motion, mdp, gamma=args.gamma,
                  max_episode_steps=args.max_episode_steps, reset_noise=args.reset_noise)
buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, args.replay_device)
replay_clip_end = motion.clip_end[motion.clip_id].to(buffer.storage_device)
print(f"Replay: {buffer.nbytes / 2**30:.3f} GiB on {buffer.storage_device}; learner {dev}, {args.precision}", flush=True)
obs_hist = torch.zeros(args.num_envs, Hh, OBS_DIM, device=dev)

@torch.no_grad()
def collect(n_steps: int, stats: dict):
    global obs_hist, sim_steps
    for _ in range(n_steps):
        S = env.state()
        o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f))
        fresh = env.ep_step == 0
        obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
        g = motion.goal(env.f, K, S)
        pi.obs_norm.update(o)
        pi.goal_norm.update(g)
        a = pi(obs_hist, g)
        a = a + args.action_noise * torch.randn_like(a)
        out = env.step(a)
        buffer.add(out)
        S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
        wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"])))
        wm.out_norm.update(wm.targets(S0, S1))
        sim_steps += args.num_envs
        stats["r"].append(out["r"].mean() / 0.1)
        if "ep_len" in out:
            stats["ep_len"].append(out["ep_len"].mean())
            stats["ep_ret"].append(out["ep_ret_norm"].mean())
            stats["term"].append(out["term"].float().sum())
            stats["done"].append(out["done"].float().sum())

stats = {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}
if resume_ck is not None:
    buffer.load_state_dict(resume_ck["replay"])
    env.load_state_dict(resume_ck["env"])
    obs_hist.copy_(resume_ck["obs_hist"])
    restore_rng(resume_ck["rng"])
    del resume_ck
else:
    collect(args.warmup_steps, stats)
log_f = open(os.path.join(args.log_dir, "train_log.csv"), "a")
if start_it == 0:
    log_f.write("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr,termination_bce,gpu_peak_gib\n")
t_start = time.time() - wall_offset
t_log = time.time()
agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": [], "term_bce": []}
for it in range(start_it, args.iterations):
    lr = lr_at(it)
    for o in (opt_wm, opt_pi, opt_v):
        for gparam in o.param_groups:
            gparam["lr"] = lr
    step_acc += 1.0 / args.utd
    n = int(step_acc)
    step_acc -= n
    prof = PROFILE and it % 20 == 0 and it > args.wm_only_iters + 5
    if prof:
        torch.cuda.synchronize(); tp0 = time.time()
    collect(n, stats)
    if prof:
        torch.cuda.synchronize(); tp1 = time.time()
    for _ in range(args.wm_batches):
        l, e8, bce = wm_update()
    if prof:
        torch.cuda.synchronize(); tp2 = time.time()
    agg["wm"].append(l), agg["wm8"].append(e8), agg["term_bce"].append(bce)
    if it >= args.wm_only_iters:
        pl, vl, ir, gn, surv = policy_update()
        if prof:
            torch.cuda.synchronize(); tp3 = time.time()
            print(f"[prof] collect {1e3*(tp1-tp0):.0f} ms ({n} steps) | wm {1e3*(tp2-tp1):.0f} ms | policy+critic {1e3*(tp3-tp2):.0f} ms", flush=True)
        agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
    if (it + 1) % args.log_every == 0:
        mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
        ips = args.log_every / (time.time() - t_log)
        t_log = time.time()
        wall_h = (time.time() - t_start) / 3600
        term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
        row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
               mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), mean(stats["ep_len"]), mean(stats["ep_ret"]), term_frac, ips, lr, mean(agg["term_bce"]), torch.cuda.max_memory_allocated(dev) / 2**30]  # fmt: skip
        log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
        log_f.flush()
        print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} "
              f"| sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | {ips:.2f} it/s", flush=True)  # fmt: skip
        stats = {k: [] for k in stats}
        agg = {k: [] for k in agg}
    if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations:
        ck = {"schema_version": SCHEMA_VERSION, "manifest": run_manifest,
              "replay": buffer.state_dict(), "env": env.state_dict(), "obs_hist": obs_hist, "rng": rng_state(), "step_acc": step_acc,
              "wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
              "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": it + 1,
              "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
              "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  # fmt: skip
        last = os.path.join(args.log_dir, "last.pt")
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        small = {k: ck[k] for k in ("pi", "wm", "iteration", "wall_time", "sim_steps", "args", "saved_at", "manifest", "schema_version")}
        path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
        torch.save(small, path + ".tmp")
        os.replace(path + ".tmp", path)


log_f.close()
