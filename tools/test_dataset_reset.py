"""Check every dataset clip's actual first frame survives clean RSI unchanged."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP
from supermimic.motion import MotionLib
from supermimic.robot import G1_JOINT_NAMES, default_joint_pos_list
from supermimic.rotations import quat_to_mat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion_dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/lafan1_g1")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    files = sorted(args.motion_dir.glob("*.csv"))
    rates_file = args.motion_dir / "motion_rates.json"
    rates = json.loads(rates_file.read_text()) if rates_file.exists() else 30.0
    if not files:
        raise ValueError("No motion files found")
    sim_cfg = make_sim(args.device)
    kin = G1Kinematics(args.device)
    first_frames = []
    # The first nine source frames retain the true first-frame state and its
    # one-sided velocity. This also covers 120 Hz inputs at 50 Hz control. Trimming avoids allocating full-library FK tensors
    # for a test whose acceptance target is the actual reset frame of each clip.
    with tempfile.TemporaryDirectory(prefix="supermimic-dataset-reset-") as tmp:
        trimmed = []
        for path in files:
            raw = np.loadtxt(path, delimiter=",", max_rows=9, dtype=np.float32)
            if raw.ndim != 2 or raw.shape[1] != 36 or len(raw) < 3:
                raise ValueError(f"Invalid source motion: {path}")
            first_frames.append(raw[0])
            dest = Path(tmp) / path.name
            np.savetxt(dest, raw, delimiter=",")
            trimmed.append(str(dest))
        motion = MotionLib(trimmed, kin, args.device, src_fps=rates)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=args.device))
    env = TrackingSim(sim_cfg, len(files), motion, mdp, eval_mode=True)
    ids = torch.arange(len(files), device=args.device)
    env.reset(ids, motion.clip_start)
    assert env.max_episode_steps is None and not env.reset_noise
    assert torch.equal(env.ep_horizon, motion.clip_len - 1)
    raw = torch.tensor(np.stack(first_frames), device=args.device)
    state = env.state()
    quat = torch.cat((raw[:, 6:7], raw[:, 3:6]), -1)
    torch.testing.assert_close(state[0], raw[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(state[1], quat_to_mat(quat), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(state[4], raw[:, 7:], rtol=0, atol=0)
    for actual, expected in zip(state, motion.ref_state(motion.clip_start)):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    excess = torch.maximum(kin.q_lo - raw[:, 7:], raw[:, 7:] - kin.q_hi).clamp_min(0)
    violations = [{"clip": files[i].stem, "joint": G1_JOINT_NAMES[j], "excess_rad": float(excess[i, j])}
                  for i, j in (excess > 1e-6).nonzero().tolist()]
    print(json.dumps({"clips": len(files), "default_reset_noise": env.reset_noise,
                      "default_max_episode_steps": env.max_episode_steps,
                      "max_joint_reset_error": float((state[4] - raw[:, 7:]).abs().max()),
                      "first_frame_limit_violations": violations,
                      "max_limit_excess_rad": float(excess.max()), "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
