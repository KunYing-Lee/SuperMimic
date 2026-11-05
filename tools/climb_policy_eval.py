"""A trained climb policy in the simulator, by humanoid_boldering's evaluation
protocol (README, "Evaluation"): the task's final cuts (off-hold 5 cm, torso,
stayed on), the policy's mean action with only the task's observation noise;
per clip, 256 episodes from its first frame and 1024 from frames drawn
uniformly over it, each to a cut or the clip's last frame. Reported per clip:
from start, the share completed, the median and max frame reached and each
episode's cut; uniform, the share completed and the median frames gained; with
the checkpoint's iteration, training hours and environment steps.

usage: python tools/climb_policy_eval.py <clip[,clip...]> <hbold data> <checkpoint> <out.json>
       [--envs 256] [--uniform_envs 1024] [--record traj.npz]
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
parser.add_argument("--envs", type=int, default=256, help="episodes from each clip's first frame")
parser.add_argument("--uniform_envs", type=int, default=1024, help="episodes from uniform frames per clip")
parser.add_argument("--off_hold_cut", type=float, default=0.05, help="the final curriculum stage")
parser.add_argument("--pass_frame", type=int, default=617, help="the right hand's bar 6 hold ends here")
parser.add_argument("--start_offset", type=int, default=0, help="from_t0 starts this many frames into each clip")
parser.add_argument("--no_stayed_cut", action="store_true", help="never cut on the stayed timers")
parser.add_argument("--no_uniform", action="store_true", help="from each clip's start only")
parser.add_argument("--record", default=None, help="npz: env 0's from-start trajectory per clip (root pose, joints)")
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
if args.no_stayed_cut:
    task.STAY_TOLERANCE = float("inf")  # ClimbMDP.terminate reads it at call time
ck = torch.load(args.checkpoint, map_location=DEV, weights_only=False)
K, hist = ck["args"]["goal_frames"], ck["args"]["hist"]
pi = Policy(OBS_DIM, mdp.goal_dim(K), hist=hist).to(DEV)
pi.load_state_dict(ck["pi"])
pi.eval()
N = max(args.envs, 0 if args.no_uniform else args.uniform_envs)  # one simulator for both: unused worlds idle
env = ClimbSim(cfg, N, motion, mdp, clip_id=args.clip, eval_mode=True)
pi.action_low, pi.action_high = env.action_bounds()
@torch.no_grad()
def run(frames: torch.Tensor, n: int, record: list | None = None) -> dict:
    """The first n envs from their start frames until each one's first cut or
    its clip's end (the rest of the batch idles along).
    record: filled with the furthest-reaching env's (root position, root
    quaternion, joint positions in model order, frame) per step."""
    end = motion.clip_end[motion.clip_id[frames]]
    first = motion.clip_start[motion.clip_id[frames]]
    env.reset(torch.arange(N, device=DEV), frames)
    env.refresh()
    o = mdp.add_noise(mdp.obs(env.state(), env.a_prev, env.xi, env.f))
    obs = o.unsqueeze(1).repeat(1, hist, 1)
    alive = torch.arange(N, device=DEV) < n
    reached = frames.clone()
    steps = []
    cause = torch.full((N,), -1, device=DEV)  # 0 torso, 1 off_hold, 2 stayed, 3 clip end
    while alive.any():
        if record is not None:
            d = env.robot.data
            steps.append(((d.root_link_pos_w - env._origin(slice(None), env.f)).cpu(), d.root_link_quat_w.cpu(),
                          d.joint_pos.cpu(), (env.f - first).cpu(), alive.cpu()))
        out = env.step(pi(obs, mdp.goal(env.f, K, env.state())))  # eval_mode: no auto-reset
        reached = torch.where(alive, env.f, reached)
        ended = alive & out["done"]
        off = (env.aux[:, :4] > task.HOLD_TOLERANCE).any(-1)
        stay = (env.aux[:, 4:] > task.STAY_TOLERANCE).any(-1)
        c = torch.where(out["term"], torch.where(off, 1, torch.where(stay, 2, 0)), 3)
        cause = torch.where(ended, c, cause)
        alive &= ~out["done"]
        env.f = torch.minimum(env.f, end - 1)  # an ended env steps on in eval mode; keep its frame inside the clip
        o = mdp.add_noise(mdp.obs(env.state(), env.a_prev, env.xi, env.f))
        obs = torch.cat((obs[:, 1:], o.unsqueeze(1)), 1)
    frames, reached, cause, end, first = frames[:n], reached[:n], cause[:n], end[:n], first[:n]
    gained = (reached - frames).float()
    if record is not None:
        k = int(gained.argmax())
        record.extend((p[k].numpy(), q[k].numpy(), j[k].numpy(), int(f[k])) for p, q, j, f, a in steps if a[k])
    names = ["torso", "off_hold", "stayed", "clip_end"]
    return {
        "frames_gained_mean": float(gained.mean()), "frames_gained_median": float(gained.median()),
        "pass_frame_share": float((reached - first >= args.pass_frame).float().mean()),
        "complete_share": float((reached >= end).float().mean()),
        "frames_gained_max": float(gained.max()),
        "frame_reached_median": float((reached - first).float().median()),
        "frame_reached_max": float((reached - first).max()), "episodes": n,
        "cause": {n: int((cause == k).sum()) for k, n in enumerate(names)},
        "reached_each": (reached - first).tolist(), "cause_each": [names[int(c)] for c in cause],
    }


recorded = {}


def clip_runs(c: int, name: str) -> dict:
    """Clip c from its first frame (plus --start_offset), and from uniform frames inside it."""
    a, b = int(motion.clip_start[c]), int(motion.clip_end[c])
    rec = [] if args.record else None
    out = {"from_t0": run(torch.full((N,), a + args.start_offset, dtype=torch.long, device=DEV), args.envs, rec)}
    if rec:
        recorded[name] = rec
    if not args.no_uniform:
        out["from_uniform"] = run(torch.randint(a, b - 1, (N,), device=DEV), args.uniform_envs)
    return out


per_clip = {name: clip_runs(c, name) for c, name in enumerate(args.clip.split(","))}
if args.record:
    import numpy as np

    np.savez(args.record, **{f"{k}_{field}": np.stack([r[i] for r in v]) for k, v in recorded.items()
                             for i, field in enumerate(("root_p", "root_quat", "q", "frame"))},
             joint_names=np.array(env.robot.joint_names))
res = {"checkpoint": args.checkpoint, "off_hold_cut": args.off_hold_cut, "envs": args.envs,
       "uniform_envs": args.uniform_envs, "stayed_cut": not args.no_stayed_cut, "start_offset": args.start_offset,
       "iteration": ck.get("iteration"), "train_hours": ck.get("wall_time", 0.0) / 3600,
       "sim_steps": ck.get("sim_steps"), "env_steps": ck.get("sim_steps", 0) // ck["args"]["num_envs"],
       **next(iter(per_clip.values())), "clips": per_clip}
print(json.dumps(res, indent=1))
with open(args.out, "w") as f:
    json.dump(res, f, indent=1)
