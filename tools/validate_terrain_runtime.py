"""Validate robot/terrain identity, native resets and complete simulator restart."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import RobotKinematics
from supermimic.mdp import MDP, flatten_state, unflatten_state
from supermimic.motion import MotionLib
from supermimic.robot import default_joint_pos_list


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion-dir", required=True)
    p.add_argument("--robot", choices=["g1", "k1"], required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cfg = make_sim(a.device)
    kin = RobotKinematics(a.device, a.robot)
    motion = MotionLib.from_dir(a.motion_dir, kin, a.device)
    mdp = MDP(
        kin, motion, torch.tensor(default_joint_pos_list(a.robot), device=a.device)
    )
    env = TrackingSim(cfg, len(motion.names), motion, mdp)
    ids = torch.arange(env.N, device=a.device)
    env.reset(ids, motion.clip_start)
    env.refresh()
    S = env.state()
    errors = [
        float((x - y).abs().max())
        for x, y in zip(S, motion.ref_state(motion.clip_start))
    ]
    # Static tile offsets use FP32 world coordinates; inverse translation is
    # bounded by float32 precision at the measured coordinate magnitude.
    max_coordinate = float(env.sim.data.qpos[:, :3].abs().max())
    tolerance = 4 * torch.finfo(torch.float32).eps * max(1, max_coordinate)
    for index, (x, y) in enumerate(zip(S, motion.ref_state(motion.clip_start))):
        torch.testing.assert_close(
            x, y, rtol=1e-5, atol=tolerance if index == 0 else 1e-6
        )
    torch.testing.assert_close(
        flatten_state(unflatten_state(flatten_state(S))), flatten_state(S)
    )
    obs = mdp.obs(S, env.a_prev, env.xi, env.f)
    assert obs.shape == (env.N, 3 * mdp.num_joints + 6)
    assert torch.isfinite(obs).all()
    depth_min = depth_max = None
    if motion.terrain is not None:
        assigned = motion.terrain_ids[motion.clip_id[env.f]]
        torch.testing.assert_close(
            env.scene.env_origins, motion.terrain.world_origins[assigned]
        )
        depth = env.depth(S)
        assert depth.shape == (env.N, 1, 18, 32) and torch.isfinite(depth).all()
        depth_min, depth_max = float(depth.min()), float(depth.max())
    snapshot = env.state_dict()
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    action = torch.zeros(env.N, mdp.num_joints, device=a.device)
    first = env.step(action)
    env.load_state_dict(snapshot)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state_all(cuda_rng)
    second = env.step(action)
    keys = ("s", "s1", "a", "xi", "f", "r", "done", "terminated")
    if motion.terrain is not None:
        keys += ("depth", "depth1")
    for key in keys:
        torch.testing.assert_close(first[key], second[key], rtol=1e-5, atol=1e-6)
    for step in range(32):
        transition = env.step(action)
        assert all(torch.isfinite(transition[k]).all() for k in ("s", "s1", "r", "xi"))
        if motion.terrain is not None:
            torch.testing.assert_close(
                env.scene.env_origins,
                motion.terrain.world_origins[motion.terrain_ids[motion.clip_id[env.f]]],
            )
            assert (
                torch.isfinite(transition["depth"]).all()
                and torch.isfinite(transition["depth1"]).all()
            )
    result = {
        "status": "PASS",
        "robot": a.robot,
        "num_joints": mdp.num_joints,
        "clips": motion.names,
        "reset_component_max_errors": errors,
        "reset_fp32_translation_tolerance": tolerance,
        "depth_range_m": [depth_min, depth_max],
        "restart_one_step_equal": True,
        "native_steps": 32,
        "scope": "runtime identity, finite transitions and restart; not motion tracking acceptance",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
