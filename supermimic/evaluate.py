"""Checkpoint evaluation (paper Appendix F.1) on all motions of the library.

Each motion plays once from its first frame with friction 1.0, no domain randomization, no
observation or action noise. A motion fails if the rollout triggers the training termination.
Normalized return is the undiscounted reward sum divided by its maximum attainable value.

Run as a watcher: ``python supermimic/evaluate.py --run_dir logs/<run> --watch``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=("g1", "k1"), default=None)
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--watch", action="store_true")
parser.add_argument("--ckpt", type=str, default=None, help="evaluate one checkpoint")
parser.add_argument(
    "--record", type=str, default=None, help="save trajectories (npz) for rendering"
)
parser.add_argument("--motion_dir", type=str, default=None)
parser.add_argument(
    "--every", type=int, default=1, help="evaluate every n-th checkpoint"
)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

import numpy as np
import torch

from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import RobotKinematics
from supermimic.mdp import MAX_STEP_REWARD, MDP
from supermimic.motion import MotionLib
from supermimic.networks import Policy
from supermimic.robot import default_joint_pos_list

torch.backends.cuda.matmul.allow_tf32 = True
dev = args.device
torch.cuda.set_device(dev)
sim_cfg = make_sim(dev, args)  # Before all CUDA tensor allocations.
with open(os.path.join(args.run_dir, "args.json")) as stream:
    run_args = json.load(stream)
args.robot = args.robot or run_args.get("robot", "g1")
args.motion_dir = args.motion_dir or run_args["motion_dir"]
kin = RobotKinematics(dev, robot=args.robot)
motion = MotionLib.from_dir(args.motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(args.robot), device=dev))
C = len(motion.names)
depth_pipeline = None
if motion.terrain is not None:
    from supermimic.depth import DepthPipeline

    depth_pipeline = DepthPipeline(noise_std=0, range_noise_scale=0)


@torch.no_grad()
def evaluate(env: TrackingSim, ck: dict, record: str | None = None) -> dict:
    a_ = ck["args"]
    K, Hh = a_["goal_frames"], a_["hist"]
    if a_.get("precision") not in ("tf32", "fp32"):
        raise ValueError("Checkpoint predates the mjlab/93-observation training schema")
    torch.backends.cuda.matmul.allow_tf32 = a_["precision"] == "tf32"
    torch.backends.cudnn.allow_tf32 = a_["precision"] == "tf32"
    if a_.get("robot", "g1") != args.robot:
        raise ValueError("Checkpoint and evaluation robot differ")
    pi = Policy(
        mdp.obs_dim,
        motion.goal_frame_dim * K,
        hist=Hh,
        num_joints=mdp.num_joints,
        depth_shape=(18, 32) if motion.terrain is not None else None,
    ).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    ids = torch.arange(C, device=dev)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(ids, motion.clip_start.clone())
    clip_end = motion.clip_end.clone()
    ret = torch.zeros(C, device=dev)
    steps = torch.zeros(C, device=dev)
    hist = None
    traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}
    while not bool(env.done_eval.all()):
        S = env.state()
        o = mdp.obs(S, env.a_prev, env.xi, env.f)
        hist = (
            o.unsqueeze(1).repeat(1, Hh, 1)
            if hist is None
            else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        )
        image = (
            depth_pipeline.clean(env.depth(S)) if depth_pipeline is not None else None
        )
        a = pi(hist, motion.goal(env.f, K, S), image)
        alive = ~env.done_eval.clone()
        if record:
            from supermimic.rotations import mat_to_quat

            (
                traj["root_p"].append(S[0].cpu()),
                traj["root_quat"].append(mat_to_quat(S[1]).cpu()),
            )
            (
                traj["q"].append(S[4].cpu()),
                traj["f"].append(env.f.cpu()),
                traj["alive"].append(alive.cpu()),
            )
        out = env.step(a)
        ret += torch.where(alive, out["r"], 0.0)
        steps += alive.float()
        env.f = torch.minimum(
            env.f, clip_end - 1
        )  # finished clips keep stepping; keep f+1 in range
    norm = ret / ((motion.clip_len - 1).float() * MAX_STEP_REWARD)
    res = {
        "norm_return": norm.mean().item(),
        "norm_return_framew": (
            ret.sum() / ((motion.clip_len - 1).sum() * MAX_STEP_REWARD)
        ).item(),
        "n_failed": int(env.failed_eval.sum().item()),
        "failed": [
            motion.names[i] for i in env.failed_eval.nonzero().flatten().tolist()
        ],
        "per_clip": {motion.names[i]: round(norm[i].item(), 4) for i in range(C)},
    }
    if motion.terrain is not None:
        assignments = motion.terrain.manifest["motions"]
        entries = [assignments[os.path.basename(file)] for file in motion.files]
        variants = [
            entry.get("variant", "unspecified")
            if isinstance(entry, dict)
            else "unspecified"
            for entry in entries
        ]
        res["terrain_groups"] = {}
        for variant in sorted(set(variants)):
            selected = torch.tensor([x == variant for x in variants], device=dev)
            res["terrain_groups"][variant] = {
                "clips": int(selected.sum()),
                "completed": int((~env.failed_eval[selected]).sum()),
                "norm_return": float(norm[selected].mean()),
            }
    if record:
        np.savez_compressed(record, robot=np.array(args.robot), terrain_manifest=np.array(json.dumps(motion.terrain.manifest) if motion.terrain is not None else "null"), names=np.array(motion.names), **{k: torch.stack(v).numpy() for k, v in traj.items()},
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                            clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy())  # fmt: skip
    return res


env = TrackingSim(sim_cfg, C, motion, mdp, eval_mode=True)
if args.ckpt:
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    res = evaluate(env, ck, args.record)
    json.dump(res, open(args.ckpt.replace(".pt", "_eval.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1))
    sys.exit(0)
out_csv = os.path.join(args.run_dir, "eval.csv")
done = set()
if os.path.exists(out_csv):
    done = {int(l.split(",")[0]) for l in open(out_csv).read().strip().splitlines()[1:]}
else:
    open(out_csv, "w").write(
        "iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n"
    )
while True:
    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))
    todo = [c for c in ckpts if int(os.path.basename(c)[5:-3]) not in done]
    todo = [c for index, c in enumerate(ckpts) if c in todo and index % args.every == 0]
    for c in todo:
        it = int(os.path.basename(c)[5:-3])
        ck = torch.load(c, map_location="cpu", weights_only=False)
        t0 = time.time()
        res = evaluate(env, ck)
        line = f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},{res['norm_return_framew']:.4f},{res['n_failed']},{'|'.join(res['failed'])}\n"
        open(out_csv, "a").write(line)
        json.dump(res, open(c.replace(".pt", "_eval.json"), "w"), indent=1)
        print(
            f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{C} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        done.add(it)
    if not args.watch:
        break
    time.sleep(60)
