"""Robot-dependent state/replay contracts, including visual restart data."""

import tempfile
import unittest
from pathlib import Path

import torch

from supermimic.buffer import ReplayBuffer
from supermimic.mdp import flatten_state, normalize_xi, unflatten_state
from supermimic.motion import MotionLib


class RobotContracts(unittest.TestCase):
    def test_incomplete_import_is_rejected_before_reading_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "INCOMPLETE").write_text("validation in progress")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                MotionLib.from_dir(tmp, None, "cpu")

    def test_state_layout_and_context_for_both_robots(self):
        for n in (22, 29):
            S = (
                torch.randn(3, 3),
                torch.eye(3).repeat(3, 1, 1),
                torch.randn(3, 3),
                torch.randn(3, 3),
                torch.randn(3, n),
                torch.randn(3, n),
            )
            flat = flatten_state(S)
            self.assertEqual(flat.shape, (3, 13 + 2 * n))
            for actual, expected in zip(unflatten_state(flat), S):
                torch.testing.assert_close(actual, expected)
            xi = torch.zeros(3, n + 4)
            xi[:, 0] = 0.75
            self.assertEqual(normalize_xi(xi).shape, xi.shape)
            torch.testing.assert_close(normalize_xi(xi), torch.zeros_like(xi))

    def test_k1_visual_replay_restores_depth_and_masks_terminal_pairs(self):
        buffer = ReplayBuffer(5, 2, "cpu", num_joints=22, depth_shape=(1, 18, 32))
        for t in range(5):
            row = {
                key: torch.zeros_like(value[0]) for key, value in buffer.data.items()
            }
            row["f"][:] = t
            row["ep_step"][:] = t
            row["depth"].fill_(t)
            row["depth1"].fill_(t + 1)
            row["done"][:] = t == 2
            buffer.add(row)
        restored = ReplayBuffer(5, 2, "cpu", num_joints=22, depth_shape=(1, 18, 32))
        restored.load_state_dict(buffer.state_dict())
        w = restored.sample_windows(7, 5)
        torch.testing.assert_close(w["depth1"], w["depth"] + 1)
        torch.testing.assert_close(
            w["mask"], torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]]).repeat(7, 1)
        )
        starts, hist = restored.sample_starts(8, 3)
        self.assertEqual(starts["depth"].shape, (8, 1, 18, 32))
        torch.testing.assert_close(starts["depth"][:, 0, 0, 0], starts["f"].float())
        self.assertEqual(hist["s"].shape, (8, 3, 57))


if __name__ == "__main__":
    unittest.main()
