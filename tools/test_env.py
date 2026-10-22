"""Validate native mjlab state, FK, randomization, checkpoint, and stepping contracts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, flatten_state
from supermimic.motion import MotionLib
from supermimic.robot import TRACKED_BODIES, default_joint_pos_list
from supermimic.rotations import quat_to_mat


def fixture_motion(kin, device):
    """Known, nonzero motion to expose joint order and angular-frame errors."""
    count = 91
    t = np.arange(count) / 30
    raw = np.zeros((count, 36), dtype=np.float32)
    raw[:, 0] = 0.05 * t
    raw[:, 1] = 0.02 * t
    raw[:, 2] = 0.9
    yaw = 0.3 + 0.1 * t
    # Rz(yaw) Ry(-0.12) Rx(0.2): world and body angular velocities differ.
    cr, sr = np.cos(0.1), np.sin(0.1)
    cp, sp = np.cos(-0.06), np.sin(-0.06)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    raw[:, 3] = sr * cp * cy - cr * sp * sy
    raw[:, 4] = cr * sp * cy + sr * cp * sy
    raw[:, 5] = cr * cp * sy - sr * sp * cy
    raw[:, 6] = cr * cp * cy + sr * sp * sy
    raw[:, 7:] = np.array(default_joint_pos_list())[None] + 0.01 * np.sin(t[:, None] + np.arange(29)[None])
    with tempfile.TemporaryDirectory(prefix="supermimic-backend-test-") as tmp:
        path = Path(tmp) / "fixture.csv"
        np.savetxt(path, raw, delimiter=",")
        return MotionLib([str(path)], kin, device)


def assert_close(a, b, label, atol=3e-5):
    err = float((a - b).abs().max()) if a.numel() else 0.0
    torch.testing.assert_close(a, b, atol=atol, rtol=3e-5, msg=label)
    print(f"{label}: max_abs_error={err:.8g}")


def run_checks(device="cpu", num_envs=2, steps=3):
    torch.manual_seed(17)
    kin = G1Kinematics(device)
    motion = fixture_motion(kin, device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    env = TrackingSim(make_sim(device), num_envs, motion, mdp)
    assert env.max_episode_steps is None and not env.reset_noise
    ids = torch.arange(num_envs, device=device)
    frames = 10 + ids % 20
    env.reset(ids, frames)
    for label, actual, expected in zip(("p", "R", "v", "w", "q", "qd"), env.state(), motion.ref_state(frames)):
        assert_close(actual, expected, "reset " + label)
    env.refresh()
    body_ids = [env.robot.body_names.index(name) for name in TRACKED_BODIES]
    data = env.robot.data
    bp, bR, bv, bw = kin.forward(*env.state())
    assert_close(bp, data.body_link_pos_w[:, body_ids], "FK position")
    assert_close(bR, quat_to_mat(data.body_link_quat_w[:, body_ids]), "FK rotation")
    assert_close(bv, data.body_link_lin_vel_w[:, body_ids], "FK linear velocity")
    assert_close(bw, data.body_link_ang_vel_w[:, body_ids], "FK angular velocity")
    assert_close(env.sim.model.geom_friction[:, :, 0], env.xi[:, :1].expand(-1, env.sim.mj_model.ngeom), "friction context")
    assert_close(env.sim.model.body_ipos[:, env.torso_id], env._default_com + env.xi[:, 30:33], "CoM context")
    old_xi = env.xi.clone()
    env.reset(ids[:1], frames[:1])
    assert not torch.equal(env.xi[:1], old_xi[:1]), "Context must resample every episode"
    if num_envs > 1:
        assert_close(env.xi[1:], old_xi[1:], "unreset context")
        assert_close(flatten_state(env.state())[1:], flatten_state(motion.ref_state(frames))[1:], "unreset state")
    actions = torch.randn(num_envs, 29, device=device) * 0.03
    for _ in range(steps):
        result = env.step(actions)
        assert torch.isfinite(result["s1"]).all() and torch.isfinite(result["r"]).all()
    # Snapshot immediately before clip timeout so replay also exercises the
    # grouped sampler update, randomized reset and next episode state.
    near_end = motion.clip_end[torch.zeros(num_envs, dtype=torch.long, device=device)] - 3
    env.reset(ids, near_end)
    env.step(actions)
    env.step(actions)
    saved = env.state_dict()
    rng = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if device.startswith("cuda") else None
    first = env.step(actions)
    first_reset_state = flatten_state(env.state()).clone()
    first_xi = env.xi.clone()
    first_bins = env.bin_G.clone()
    assert first["done"].all(), "Replay probe must cross an episode boundary"
    env.load_state_dict(saved)
    torch.set_rng_state(rng)
    if rng_cuda is not None:
        torch.cuda.set_rng_state_all(rng_cuda)
    second = env.step(actions)
    assert_close(first["s1"], second["s1"], "checkpoint next transition", atol=1e-6)
    assert_close(first["r"], second["r"], "checkpoint next reward", atol=1e-6)
    assert_close(first_reset_state, flatten_state(env.state()), "checkpoint reset state", atol=1e-6)
    assert_close(first_xi, env.xi, "checkpoint reset context", atol=1e-6)
    assert_close(first_bins, env.bin_G, "checkpoint sampler return", atol=1e-6)
    print(f"PASS: mjlab backend contracts on {device}, {num_envs} environments, {steps} control steps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    run_checks(args.device, args.num_envs, args.steps)


if __name__ == "__main__":
    main()
