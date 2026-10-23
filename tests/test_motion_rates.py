"""Mixed-rate ingestion must retain physical time and the legacy 30 Hz result."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from supermimic.kinematics import G1Kinematics
from supermimic.motion import MotionLib


class MotionRateTests(unittest.TestCase):
    def test_rates_preserve_time_and_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            kin = G1Kinematics("cpu")
            rates = {}
            for fps in [30, 120]:
                x = np.zeros((fps + 1, 36))
                x[:, 0] = np.arange(fps + 1) / fps
                x[:, 2] = 0.8
                x[:, 6] = 1
                f = p / f"m{fps}.csv"
                np.savetxt(f, x, delimiter=",")
                rates[f.name] = fps
            (p / "motion_rates.json").write_text(json.dumps(rates))
            m = MotionLib.from_dir(str(p), kin, "cpu")
            self.assertEqual(m.clip_len.tolist(), [51, 51])
            for c in range(2):
                s = int(m.clip_start[c])
                torch.testing.assert_close(
                    m.root_p[s : s + 51, 0],
                    torch.arange(51) * 0.02,
                    atol=1e-7,
                    rtol=1e-6,
                )
                torch.testing.assert_close(
                    m.root_v[s : s + 51, 0], torch.ones(51), atol=1e-5, rtol=1e-5
                )
            old = MotionLib([str(p / "m30.csv")], kin, "cpu")
            new = MotionLib([str(p / "m30.csv")], kin, "cpu", src_fps={"m30.csv": 30.0})
            for key in [
                "root_p",
                "root_R",
                "root_v",
                "root_w",
                "q",
                "qd",
                "body_p",
                "body_R",
            ]:
                self.assertTrue(torch.equal(getattr(old, key), getattr(new, key)), key)
            (p / "motion_rates.json").write_text(json.dumps({"m30.csv": 30}))
            with self.assertRaises(ValueError):
                MotionLib.from_dir(str(p), kin, "cpu")


if __name__ == "__main__":
    unittest.main()
