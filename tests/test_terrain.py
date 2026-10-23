"""Geometric and differentiability contracts for terrain-conditioned imagination."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np
import torch

from supermimic.depth import CameraCfg, DepthPipeline, relative_pose
from supermimic.networks import DepthEnsemble, Policy, WorldModel
from supermimic.rotations import axis_angle_to_mat, matmul3, matvec3
from supermimic.terrain import TerrainBank


def manifest():
    return {
        "schema_version": 1,
        "terrains": [
            {
                "id": 0,
                "name": "flat",
                "ground_height": 0.0,
                "bounds_xy": [[-2.0, -2.0], [3.0, 2.0]],
                "boxes": [],
            },
            {
                "id": 1,
                "name": "step",
                "ground_height": 0.0,
                "bounds_xy": [[-2.0, -2.0], [3.0, 2.0]],
                "boxes": [{"center": [1.0, 0.0, 0.2], "half_size": [0.2, 0.5, 0.2]}],
            },
        ],
    }


def state(batch=2, joints=22):
    return (
        torch.tensor([[0.0, 0.0, 0.7]]).repeat(batch, 1),
        torch.eye(3).repeat(batch, 1, 1),
        torch.zeros(batch, 3),
        torch.zeros(batch, 3),
        torch.zeros(batch, joints),
        torch.zeros(batch, joints),
    )


class TerrainGeometryTests(unittest.TestCase):
    def test_ground_plane_matches_native_one_sided_rays(self):
        terrain = TerrainBank(manifest())
        spec = mujoco.MjSpec()
        terrain.add_to_spec(spec)
        model = spec.compile()
        simulator = mujoco.MjData(model)
        mujoco.mj_forward(model, simulator)
        origins = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        directions = torch.tensor(
            [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
        )[:, None]
        actual = terrain.raycast(origins, directions, torch.zeros(4, dtype=torch.long))
        expected = torch.empty_like(actual)
        for index in range(4):
            distance = mujoco.mj_ray(
                model,
                simulator,
                (origins[index] + terrain.world_origins[0]).double().numpy(),
                directions[index, 0].double().numpy(),
                None,
                True,
                -1,
                np.empty(1, np.int32),
            )
            expected[index] = distance if distance >= 0 else torch.inf
        torch.testing.assert_close(actual, expected)

    def test_scan_and_depth_follow_foreign_tiles_after_global_drift(self):
        data = manifest()
        data["terrains"][1]["ground_height"] = 0.15
        data["terrains"][1]["boxes"][0]["center"][2] += 0.15
        terrain = TerrainBank(data)
        spec = mujoco.MjSpec()
        terrain.add_to_spec(spec)
        model = spec.compile()
        simulator = mujoco.MjData(model)
        mujoco.mj_forward(model, simulator)
        # Motion zero remains assigned, but its physical position is now above
        # motion one's box. No bounds reset or artificial termination intervenes.
        p = (
            torch.tensor([[1.0, 0.0, 1.15]])
            + terrain.world_origins[1]
            - terrain.world_origins[0]
        )
        ids = torch.tensor([0])
        ray = torch.tensor([[[0.0, 0.0, -1.0]]])
        depth = terrain.raycast(p, ray, ids)
        native = mujoco.mj_ray(
            model,
            simulator,
            (p[0] + terrain.world_origins[0]).double().numpy(),
            ray[0, 0].double().numpy(),
            None,
            True,
            -1,
            np.empty(1, np.int32),
        )
        torch.testing.assert_close(
            depth, torch.tensor([[native]]), atol=2e-6, rtol=1e-5
        )
        torch.testing.assert_close(depth, torch.tensor([[0.6]]), atol=2e-6, rtol=1e-5)
        scan = terrain.scan(p, torch.eye(3).unsqueeze(0), ids)
        torch.testing.assert_close(scan[:, 93], depth[:, 0], atol=2e-6, rtol=1e-5)
        self.assertTrue(
            torch.isnan(
                terrain.ground_height(torch.tensor([[[float("nan"), 0.0]]]), ids)
            ).all()
        )

    def test_native_mujoco_depth_agrees_on_all_tiles_and_rotated_camera(self):
        data = manifest()
        data["terrains"][1]["ground_height"] = 0.15
        data["terrains"][1]["boxes"][0]["center"][2] += 0.15
        terrain = TerrainBank(data)
        spec = mujoco.MjSpec()
        terrain.add_to_spec(spec)
        model = spec.compile()
        simulator = mujoco.MjData(model)
        mujoco.mj_forward(model, simulator)
        generator = torch.Generator().manual_seed(78)
        p = torch.rand(2, 3, generator=generator) * 0.5
        p[:, 2] += 1.0 + terrain.ground
        directions = torch.randn(2, 40, 3, generator=generator)
        directions[..., 2] = -0.5 - directions[..., 2].abs()
        directions = torch.nn.functional.normalize(directions, dim=-1)
        actual = terrain.raycast(p, directions, torch.arange(2))
        expected = torch.empty_like(actual)
        for tid in range(2):
            world_p = (p[tid] + terrain.world_origins[tid]).double().numpy()
            for ray in range(40):
                direction = directions[tid, ray].double().numpy()
                expected[tid, ray] = mujoco.mj_ray(
                    model,
                    simulator,
                    world_p,
                    direction,
                    None,
                    True,
                    -1,
                    np.empty(1, np.int32),
                )
        torch.testing.assert_close(actual, expected, atol=4e-6, rtol=1e-5)
        # Tiny blocks exercise the exact min reduction independently of padding.
        blocked = terrain.raycast(
            p, directions, torch.arange(2), max_pairs=4, box_chunk=1
        )
        torch.testing.assert_close(actual, blocked, atol=0, rtol=0)
        camera = CameraCfg()
        from supermimic.depth import camera_pose

        R = axis_angle_to_mat(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(0.7)).repeat(
            2, 1, 1
        )
        center_p, center_R = camera_pose(p, R, camera)
        # The pinhole central ray rotates with the root and fixed downward mount.
        central_direction = center_R[..., :, 0]
        central_depth = terrain.raycast(
            center_p, central_direction[:, None], torch.arange(2)
        )
        one_pixel = CameraCfg(height=1, width=1)
        torch.testing.assert_close(
            terrain.render_depth(p, R, torch.arange(2), one_pixel).flatten(),
            central_depth.flatten().clamp(camera.near, camera.far),
        )

    def test_exact_rays_cover_box_faces_parallel_rays_and_ground(self):
        terrain = TerrainBank(manifest())
        origins = torch.tensor(
            [
                [0.0, 0.0, 0.3],
                [0.0, 0.0, 0.6],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 0.8],
                [1.0, 0.0, 0.2],
            ]
        )
        directions = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, -0.5],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )[:, None]
        depth = terrain.raycast(
            origins, directions, torch.ones(5, dtype=torch.long)
        ).squeeze(-1)
        torch.testing.assert_close(depth, torch.tensor([0.8, 0.8, 0.6, torch.inf, 0.2]))
        flat = terrain.raycast(origins[2:3], directions[2:3], torch.tensor([0]))
        torch.testing.assert_close(flat, torch.ones(1, 1))

    def test_scan_is_bilinear_differentiable_and_uses_motion_terrain(self):
        terrain = TerrainBank(manifest())
        s = state()
        s[0][:, 0] = 1.0
        scan = terrain.scan(s[0], s[1], torch.tensor([0, 1]))
        self.assertEqual(scan.shape, (2, 187))
        torch.testing.assert_close(scan[0], torch.full((187,), 0.7))
        self.assertAlmostEqual(scan[1, 93].item(), 0.3, places=5)
        ny, nx = terrain.heights.shape[-2:]
        x = terrain.origins[0, 0] + torch.arange(nx) * terrain.resolution
        y = terrain.origins[0, 1] + torch.arange(ny) * terrain.resolution
        terrain.heights[0] = 0.2 * x[None] + 0.3 * y[:, None]
        xy = torch.tensor([[[0.013, 0.017], [0.113, -0.117]]], requires_grad=True)
        height = terrain.ground_height(xy, torch.tensor([0]))
        torch.testing.assert_close(
            height, 0.2 * xy[..., 0] + 0.3 * xy[..., 1], atol=2e-7, rtol=1e-5
        )
        height.sum().backward()
        torch.testing.assert_close(
            xy.grad, torch.tensor([[[0.2, 0.3], [0.2, 0.3]]]), atol=2e-6, rtol=1e-5
        )
        outside = terrain.ground_height(
            torch.tensor([[[20.0, 20.0]]]), torch.tensor([0])
        )
        torch.testing.assert_close(outside, torch.zeros(1, 1))

    def test_pinhole_depth_is_axial_and_sensor_processing_is_shared(self):
        terrain = TerrainBank(manifest())
        camera = CameraCfg(
            height=1,
            width=1,
            position=(0.0, 0.0, 0.0),
            rotation_vector=(0.0, torch.pi / 2, 0.0),
        )
        s = state()
        raw = terrain.render_depth(s[0], s[1], torch.tensor([0, 1]), camera)
        torch.testing.assert_close(raw, torch.full((2, 1, 1, 1), 0.7))
        pipeline = DepthPipeline(camera, noise_std=0.0, range_noise_scale=0.0)
        clean = pipeline.clean(raw)
        torch.testing.assert_close(pipeline.meters_from_clean(clean), raw)
        torch.testing.assert_close(pipeline.noise(clean), clean)

    def test_manifest_rejects_overhangs_and_invalid_terrain_identity(self):
        data = manifest()
        data["terrains"][1]["boxes"][0]["center"][2] = 0.5
        with self.assertRaisesRegex(ValueError, "supported"):
            TerrainBank(data)
        data = manifest()
        data["terrains"][1]["id"] = 2
        with self.assertRaisesRegex(ValueError, "IDs"):
            TerrainBank(data)

    def test_relative_pose_is_invariant_to_global_rigid_transform(self):
        anchor, target = state(), list(state())
        target[0] = target[0] + torch.tensor([0.2, -0.1, 0.15])
        target[1] = axis_angle_to_mat(
            torch.tensor([0.0, 0.0, 1.0]), torch.tensor(0.4)
        ).repeat(2, 1, 1)
        expected = relative_pose(anchor, target)
        transform = axis_angle_to_mat(torch.tensor([0.0, 1.0, 0.0]), torch.tensor(0.7))
        moved = []
        for original in (anchor, target):
            result = list(original)
            result[0] = matvec3(transform, original[0]) + torch.tensor([2.0, 3.0, 4.0])
            result[1] = matmul3(transform, original[1])
            moved.append(result)
        torch.testing.assert_close(
            relative_pose(*moved), expected, atol=5e-7, rtol=1e-5
        )


class TerrainNetworkTests(unittest.TestCase):
    def test_joint_parameterized_world_model_and_visual_policy(self):
        for joints in (22, 29):
            s = state(joints=joints)
            wm = WorldModel(
                hidden=(16,),
                goal_dim=7,
                num_joints=joints,
                xi_dim=joints + 4,
                terrain_dim=187,
            )
            scan = torch.zeros(2, 187, requires_grad=True)
            action = torch.zeros(2, joints, requires_grad=True)
            context = torch.zeros(2, joints + 4)
            torch.nn.init.normal_(wm.net[-1].weight, std=0.01)
            predicted = wm.step(s, action, context, scan)
            self.assertEqual(predicted[4].shape, (2, joints))
            sum(value.square().sum() for value in predicted).backward()
            self.assertTrue(torch.isfinite(scan.grad).all())
            self.assertGreater(scan.grad.abs().sum().item(), 0.0)
            with self.assertRaisesRegex(ValueError, "height scan"):
                wm.step(s, action, context)
            pi = Policy(
                3 * joints + 6, 7, hist=2, num_joints=joints, depth_shape=(18, 32)
            )
            image = torch.rand(2, 1, 18, 32, requires_grad=True)
            result = pi(torch.zeros(2, 2, 3 * joints + 6), torch.zeros(2, 7), image)
            self.assertEqual(result.shape, (2, joints))
            result.square().sum().backward()
            self.assertGreater(image.grad.abs().sum().item(), 0.0)

    def test_bootstrap_depth_members_preserve_assignment_and_gradient(self):
        torch.manual_seed(45)
        ensemble = DepthEnsemble(num_joints=22, ensemble_size=3)
        anchor = torch.rand(3, 1, 18, 32)
        pose = relative_pose(state(3), state(3)).requires_grad_()
        inputs = (
            anchor,
            pose,
            torch.zeros(3, 22),
            torch.zeros(3, 6),
            torch.zeros(3, 187),
        )
        all_predictions = ensemble.forward_all(*inputs)
        assignments = torch.tensor([2, 0, 2])
        selected = ensemble(ensemble.groups(assignments), *inputs)
        torch.testing.assert_close(
            selected,
            all_predictions[assignments, torch.arange(3)],
            atol=2e-7,
            rtol=1e-5,
        )
        self.assertTrue(((selected > 0) & (selected < 1)).all())
        selected.sum().backward()
        self.assertTrue(torch.isfinite(pose.grad).all())
        self.assertGreater(pose.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
