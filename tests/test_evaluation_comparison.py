"""Acceptance boundaries for fixed-evaluation comparisons, independent of GPU training."""

import copy
import unittest

import numpy as np

from tools.compare_evaluations import compare


def fixture():
    data = {
        "names": np.array(["base", "added"]),
        "clip_start": np.array([0, 4]),
        "clip_len": np.array([4, 4]),
        "ref_root_p": np.zeros((8, 3)),
        "ref_root_R": np.tile(np.eye(3), (8, 1, 1)),
        "ref_q": np.zeros((8, 2)),
        "root_p": np.zeros((3, 2, 3)),
        "root_quat": np.zeros((3, 2, 4)),
        "q": np.zeros((3, 2, 2)),
        "alive": np.ones((3, 2), dtype=bool),
        "f": np.array([[0, 4], [1, 5], [2, 6]]),
    }
    result = {"per_clip": {"base": 0.9, "added": 0.8}, "failed": [], "n_failed": 0}
    return data, result


class ComparisonTests(unittest.TestCase):
    def test_common_prefix_does_not_reward_early_failure(self):
        base, br = fixture()
        new, nr = copy.deepcopy(base), copy.deepcopy(br)
        base["alive"][1:, 1] = False
        br.update(failed=["added"], n_failed=1)
        new["root_p"][:, 1, 0] = [1, 4, 10]
        rows, summary = compare(base, new, [br, nr], base_clips=1)
        self.assertEqual(rows[1]["common_xy_mean_candidate"], 1)
        self.assertEqual(rows[1]["xy_mean_candidate"], 5)
        self.assertEqual(summary["newly_completed"], ["added"])
        self.assertEqual(summary["groups"]["added"]["candidate"]["completed"], 1)

    def test_nonfinite_post_episode_is_reported_live_is_rejected(self):
        base, result = fixture()
        base["alive"][1:, 1] = False
        result.update(failed=["added"], n_failed=1)
        base["root_p"][1:, 1] = np.nan
        _, summary = compare(base, copy.deepcopy(base), [result, result])
        self.assertEqual(
            summary["trajectory_diagnostics"]["baseline"]["root_p"][
                "post_episode_nonfinite_frames"
            ],
            2,
        )
        base["root_p"][0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite live"):
            compare(base, base, [result, result])

    def test_reference_change_and_truncated_recording_rejected(self):
        base, result = fixture()
        new = copy.deepcopy(base)
        new["ref_q"][0, 0] = 0.01
        with self.assertRaisesRegex(ValueError, "Reference libraries differ"):
            compare(base, new, [result, result])
        new = copy.deepcopy(base)
        new["alive"][-1, 0] = False
        with self.assertRaisesRegex(ValueError, "Incomplete recording"):
            compare(base, new, [result, result])

    def test_frame_identity_and_episode_restart_rejected(self):
        base, result = fixture()
        new = copy.deepcopy(base)
        new["f"][0, 0] = 1
        with self.assertRaisesRegex(ValueError, "first frame"):
            compare(base, new, [result, result])
        new = copy.deepcopy(base)
        new["alive"][1, 0] = False
        with self.assertRaisesRegex(ValueError, "contiguous"):
            compare(base, new, [result, result])


if __name__ == "__main__":
    unittest.main()
