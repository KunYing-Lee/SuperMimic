"""Contracts for source-preserving K1 conversion and constrained terrain pairing."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.prepare_k1_terrain_data import (
    SOURCE_JOINT_NAMES,
    convert_motion,
    load_retarget,
    resample_qpos,
    segment_ranges,
    sha256,
    terrain_from_bounds,
    validate_contacts,
)


class K1DataTest(unittest.TestCase):
    def test_retarget_accepts_explicit_resampling_and_rejects_root_changes(self):
        source = {
            "joint_pos": np.zeros((4, 22)),
            "body_pos_w": np.zeros((4, 23, 3)),
            "body_quat_w": np.zeros((4, 23, 4)),
            "fps": np.array([30]),
        }
        source["body_pos_w"][:, 0, 0] = np.arange(4) * 0.1
        source["body_quat_w"][..., 0] = 1
        csv, _ = convert_motion(source, SOURCE_JOINT_NAMES)
        resampled = resample_qpos(csv, 30, 100)
        derived = {
            "joint_pos": resampled[:, 7:],
            "body_pos_w": np.zeros((len(resampled), 23, 3)),
            "body_quat_w": np.zeros((len(resampled), 23, 4)),
            "fps": np.array([100]),
        }
        derived["body_pos_w"][:, 0] = resampled[:, :3]
        derived["body_quat_w"][..., 0] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            model = path / "robot.xml"
            model.write_text("model fingerprint")
            output = path / "clip.npz"
            np.savez(output, **derived)
            record = {
                "source_sha256": "source",
                "output_sha256": sha256(output),
                "robot_xml_sha256": sha256(model),
                "converged": True,
                "source_frame_start": 0,
                "source_frame_stop": 4,
                "target_fps": 100,
                "joint_names": SOURCE_JOINT_NAMES,
                "joint_velocity_limits_rad_s": [10.0] * 22,
            }
            metadata = path / "clip.json"
            metadata.write_text(json.dumps(record))
            result, _, _ = load_retarget(path, "clip", source, "source", model)
            self.assertEqual(len(result["joint_pos"]), 11)
            derived["body_pos_w"][5, 0, 2] += 0.01
            np.savez(output, **derived)
            record["output_sha256"] = sha256(output)
            metadata.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "fixed source root"):
                load_retarget(path, "clip", source, "source", model)

    def test_native_contact_validation_independently_rejects_intersection(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("MuJoCo is required for native contact verification")
        children = "".join(
            f'<body name="body{i}" pos="0 0 0.01"><joint name="{name}" type="hinge"/><geom type="sphere" size="0.001" mass="0.01" contype="0" conaffinity="0"/></body>'
            for i, name in enumerate(SOURCE_JOINT_NAMES)
        )
        xml = (
            '<mujoco><worldbody><body name="Trunk"><freejoint/><geom type="sphere" size="0.1" mass="1"/>'
            + children
            + "</body></worldbody></mujoco>"
        )
        poses = np.zeros((2, 29))
        poses[:, 2] = 1
        poses[:, 6] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.xml"
            path.write_text(xml)
            valid = validate_contacts(
                path,
                poses,
                SOURCE_JOINT_NAMES,
                [{"center": [0, 0, 0.1], "half_size": [0.1, 0.1, 0.1]}],
            )
            self.assertEqual(valid["min_obstacle_distance_m"], 0)
            with self.assertRaisesRegex(ValueError, "intersects"):
                validate_contacts(
                    path,
                    poses,
                    SOURCE_JOINT_NAMES,
                    [{"center": [0, 0, 1], "half_size": [0.2, 0.2, 0.2]}],
                )

    def test_systematic_segments_preserve_boundary_and_cover_tail(self):
        self.assertEqual(
            segment_ranges(902, 30, 10), [(0, 301), (300, 601), (600, 901), (900, 902)]
        )
        self.assertEqual(
            segment_ranges(902, 30, 10, 3), [(0, 301), (300, 601), (600, 901)]
        )
        self.assertEqual(segment_ranges(902, 30, 0), [(0, 902)])
        with self.assertRaisesRegex(ValueError, "integral"):
            segment_ranges(902, 30, 0.015)

    def test_named_joint_and_quaternion_permutations_only(self):
        q = np.arange(44).reshape(2, 22).astype(float)
        p = np.arange(138).reshape(2, 23, 3).astype(float)
        quat = np.zeros((2, 23, 4))
        quat[..., 0] = 1
        source = dict(joint_pos=q, body_pos_w=p, body_quat_w=quat, fps=[30])
        csv, fps = convert_motion(source, list(reversed(SOURCE_JOINT_NAMES)))
        np.testing.assert_array_equal(csv[:, :3], p[:, 0])
        np.testing.assert_array_equal(csv[:, 3:7], [[0, 0, 0, 1]] * 2)
        np.testing.assert_array_equal(csv[:, 7:], q[:, ::-1])
        self.assertEqual(fps, 30)
        with self.assertRaisesRegex(ValueError, "names"):
            convert_motion(source, ["unknown"] * 22)
        source["body_quat_w"][0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, "non-unit"):
            convert_motion(source, SOURCE_JOINT_NAMES)

    def test_slerp_keeps_endpoints_and_shortest_arc(self):
        csv = np.zeros((2, 29))
        csv[0, 6], csv[1, 6] = 1, -1
        csv[1, 0] = 2
        csv[1, 7:] = 1
        out = resample_qpos(csv, 1, 2)
        np.testing.assert_allclose(out[:, 0], [0, 1, 2])
        np.testing.assert_allclose(out[:, 7], [0, 0.5, 1])
        np.testing.assert_allclose(out[:, 6], 1)
        np.testing.assert_allclose(np.linalg.norm(out[:, 3:7], axis=1), 1)

    def test_jump_clearance_and_revisited_ground_cell(self):
        # Both root paths cross a cell, but any grounded revisit forbids a box.
        low = np.array(
            [[[0.05, 0.05, 0.30]], [[1.05, 0.05, 0.30]], [[1.05, 0.05, 0.0]]]
        )
        high = low + [0.10, 0.10, 0.10]
        xy = np.array([[0.1, 0.1], [1.1, 0.1], [1.1, 0.1]])
        boxes, bounds, diagnostic = terrain_from_bounds(
            low,
            high,
            xy,
            cell_size=0.2,
            clearance=0.02,
            max_height=0.4,
            min_height=0.03,
        )
        self.assertEqual(len(boxes), 1)
        np.testing.assert_allclose(boxes[0]["center"], [0.1, 0.1, 0.14])
        np.testing.assert_allclose(boxes[0]["half_size"], [0.1, 0.1, 0.14])
        self.assertEqual(diagnostic["root_path_cells"], 2)
        self.assertLess(bounds[0][0], -1)
        self.assertGreater(bounds[1][0], 2)

    def test_all_body_geometry_constrains_obstacles(self):
        # Root is above ground, but a hand's collision bound blocks this cell.
        low = np.array([[[0.05, 0.05, 0.3], [0.06, 0.06, 0.01]]])
        high = low + [0.05, 0.05, 0.05]
        boxes, _, _ = terrain_from_bounds(
            low,
            high,
            np.array([[0.1, 0.1]]),
            cell_size=0.2,
            clearance=0.02,
            max_height=0.2,
            min_height=0.03,
        )
        self.assertEqual(boxes, [])


if __name__ == "__main__":
    unittest.main()
