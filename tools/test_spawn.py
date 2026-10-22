"""Spawn native mjlab G1s and measure physics-only control-step throughput."""

import argparse
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.env import build_scene, make_sim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    cfg = make_sim(args.device)
    scene, sim = build_scene(cfg, args.num_envs)
    robot = scene["robot"]
    robot.write_root_state_to_sim(robot.data.default_root_state)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.set_joint_position_target(robot.data.default_joint_pos)
    print("JOINTS", robot.joint_names)
    print("BODIES", robot.body_names)
    print("mass", float(sim.mj_model.body_mass.sum()))
    for repetition in range(3):
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            for _ in range(cfg.decimation):
                scene.write_data_to_sim()
                sim.step()
                scene.update(cfg.physics.mujoco.timestep)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        assert torch.isfinite(sim.data.qpos).all()
        print(f"run={repetition} env_steps_per_second={args.steps * args.num_envs / elapsed:.0f}")


if __name__ == "__main__":
    main()
