"""Collect climbing transitions for the offline world-model gate, in SuperMimic's
replay format: s, s1 (state tuples flattened), a, xi, f, done, terminated, valid,
contact. Shards of (T, N) are written as torch files.

Modes:
  bm      a trained hbold PPO policy (rsl_rl checkpoint) in hbold's own env, plus
          Gaussian noise `sigma` on its actions. hbold's PD target is
          default + scale a - encoder_bias; SuperMimic's is default + offset + alpha a,
          with the same scale, so a is the policy's and the offset is -encoder_bias.
  refpd   ClimbSim from task starts, PD targets on the reference's next joints.
  random  ClimbSim, a smoothed random walk of actions inside the PD bounds.

usage: python tools/climb_collect.py <mode> <clip> <hbold data> <out dir> <steps> [--sigma s]
       [--checkpoint model.pt] [--num_envs N] [--shard_steps T] [--seed k]
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["bm", "refpd", "random"])
parser.add_argument("clip")
parser.add_argument("data")
parser.add_argument("out")
parser.add_argument("steps", type=int)
parser.add_argument("--sigma", type=float, default=0.0)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--shard_steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
os.environ["HBOLD_DATA"] = args.data

import torch  # noqa: E402

from supermimic.climb.parts import climb_parts  # noqa: E402
from supermimic.climb.sim import (  # noqa: E402
    CONTACT_MAXMATCH,
    PRESS_N,
    ClimbSim,
    climb_physics,
    hook_action_scale,
    pd_action_bounds,
)
from supermimic.env import make_sim  # noqa: E402
from supermimic.mdp import XI_DIM, flatten_state  # noqa: E402
from supermimic.robot import G1_JOINT_NAMES  # noqa: E402
from supermimic.rotations import matvec3, quat_to_mat  # noqa: E402

DEV = "cuda:0"
KEYS = ("s", "s1", "a", "xi", "f", "done", "terminated", "valid", "contact")
torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)


class Shards:
    def __init__(self, out: str, steps: int):
        self.out, self.steps, self.rows, self.k = out, steps, {k: [] for k in KEYS}, 0

    def add(self, row: dict) -> None:
        for k in KEYS:
            self.rows[k].append(row[k].detach().to("cpu"))
        if len(self.rows["s"]) == self.steps:
            self.flush()

    def flush(self) -> None:
        if not self.rows["s"]:
            return
        shard = {k: torch.stack(v) for k, v in self.rows.items()}
        path = os.path.join(self.out, f"{args.mode}_s{args.sigma:g}_{self.k:03d}.pt")
        torch.save(shard, path + ".tmp")
        os.replace(path + ".tmp", path)
        print(f"wrote {path}: {tuple(shard['s'].shape)}", flush=True)
        self.rows, self.k = {k: [] for k in KEYS}, self.k + 1


def collect_sim(shards: Shards) -> None:
    cfg = make_sim(DEV)
    cfg.physics = climb_physics(args.clip)
    _, motion, mdp = climb_parts(args.clip, DEV)
    env = ClimbSim(cfg, args.num_envs, motion, mdp, clip_id=args.clip)
    low, high = env.action_bounds()
    walk = torch.zeros(args.num_envs, 29, device=DEV)
    for _ in range(args.steps):
        if args.mode == "refpd":
            nxt = torch.minimum(env.f + 1, motion.clip_end[motion.clip_id[env.f]])
            a = (motion.q[nxt] - env.default_q - env.xi[:, 1:30]) / env.alpha
        else:
            walk = 0.9 * walk + 0.3 * torch.randn_like(walk)
            walk = torch.where((env.ep_step == 0)[:, None], 0.0, walk)
            a = walk
        a = torch.clamp(a + args.sigma * torch.randn_like(a), low, high)
        shards.add(env.step(a))


def collect_bm(shards: Shards) -> None:
    from dataclasses import asdict

    from hbold.tasks.goal.goal_env_cfg import contact_sensors
    from hbold.tasks.registry import register
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    from supermimic.climb.mdp import LIMB_BODIES

    register()
    task = f"Hbold-Motion-Tracking-{args.clip}"
    env_cfg = load_env_cfg(task)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.sensors = (*env_cfg.scene.sensors, *contact_sensors(LIMB_BODIES)[: len(LIMB_BODIES)])
    env_cfg.sim.contact_sensor_maxmatch = CONTACT_MAXMATCH
    env_cfg.auto_reset = False  # the terminal successor is recorded, then the env is reset here
    agent_cfg = load_rl_cfg(task)
    raw = ManagerBasedRlEnv(env_cfg, device=DEV)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    runner = load_runner_cls(task)(env, asdict(agent_cfg), device=DEV)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=DEV)
    policy = runner.get_inference_policy(device=DEV)
    robot = raw.scene["robot"]
    cmd = raw.command_manager.get_term("motion")
    names = list(robot.joint_names)
    sdk = torch.tensor([names.index(n) for n in G1_JOINT_NAMES], device=DEV)  # entity -> SDK order
    torso = int(robot.indexing.body_ids[robot.body_names.index("torso_link")])
    com0 = raw.sim.get_default_field("body_ipos")[torso].clone()
    motion_end = int(cmd.motion.time_step_total) - 1
    default = robot.data.default_joint_pos[0, sdk]
    alpha = hook_action_scale(DEV)
    low, high = pd_action_bounds(raw.sim.mj_model, default, alpha)
    sensors = [raw.scene[f"{limb}_route"] for limb in ("left_hand", "right_hand", "left_foot", "right_foot")]
    press = torch.tensor(PRESS_N, device=DEV)
    nan_term = "nan" in raw.termination_manager.active_terms

    def state():
        idx = robot.indexing
        q = raw.sim.data.qpos[:, idx.free_joint_q_adr]
        v = raw.sim.data.qvel[:, idx.free_joint_v_adr]
        R = quat_to_mat(q[:, 3:7])
        return (q[:, :3] - raw.scene.env_origins, R, v[:, :3].clone(), matvec3(R, v[:, 3:6]),
                robot.data.joint_pos[:, sdk].clone(), robot.data.joint_vel[:, sdk].clone())

    obs = env.get_observations()
    for _ in range(args.steps):
        S, f = state(), cmd.time_steps.clone()
        xi = torch.zeros(args.num_envs, XI_DIM, device=DEV)
        xi[:, 0] = 1.0
        xi[:, 1:30] = -robot.data.encoder_bias[:, sdk]
        xi[:, 30:33] = raw.sim.model.body_ipos[:, torso] - com0
        with torch.no_grad():
            a_ent = policy(obs)
        a = torch.clamp(a_ent[:, sdk] + args.sigma * torch.randn(args.num_envs, 29, device=DEV), low, high)
        a_ent = torch.empty_like(a_ent)
        a_ent[:, sdk] = a
        obs, _, dones, _ = env.step(a_ent)
        S1 = state()
        valid = torch.isfinite(flatten_state(S1)).all(-1)
        if nan_term:
            valid &= ~raw.termination_manager.get_term("nan")
        term = raw.termination_manager.terminated & valid
        contact = torch.stack([(sn.data.force.norm(dim=-1) * (sn.data.found > 0)).sum(-1) for sn in sensors], -1) > press
        valid &= f < motion_end  # hbold's reference wraps past its last frame: a teleport, not a transition
        done = dones.bool() | ~valid
        s1 = torch.where(valid[:, None], flatten_state(S1), flatten_state(S))
        shards.add({"s": flatten_state(S), "s1": s1, "a": a, "xi": xi, "f": f, "done": done,
                    "terminated": term, "valid": valid, "contact": contact})
        ids = done.nonzero().flatten()
        if len(ids):
            raw.reset(env_ids=ids)
            obs = env.get_observations()


shards = Shards(args.out, args.shard_steps)
if args.mode == "bm":
    collect_bm(shards)
else:
    collect_sim(shards)
shards.flush()
