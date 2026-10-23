"""Robot-profile contracts and independent MuJoCo checks of differentiable FK."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.kinematics import G1Kinematics, RobotKinematics
from supermimic.robot import G1, K1, action_scale, default_joint_pos_list, get_robot


class RobotProfileTests(unittest.TestCase):
    def test_profiles_are_explicit_and_preserve_g1_defaults(self):
        self.assertEqual(G1.num_joints, 29)
        self.assertEqual(K1.num_joints, 22)
        self.assertEqual(default_joint_pos_list(), list(G1.default_joint_positions))
        self.assertEqual(get_robot(K1), K1)
        self.assertEqual(K1.default_joint_pos["Left_Knee_Pitch"], 0.8)
        self.assertAlmostEqual(action_scale("Left_Knee_Pitch", K1), 0.35)
        with self.assertRaisesRegex(ValueError, "Unknown robot"):
            get_robot("unrecognized")
        with self.assertRaises(ValueError):
            replace(K1, joint_names=K1.joint_names[:-1])

    def _check_against_mujoco(self, spec):
        kin = RobotKinematics("cpu", spec)
        model = mujoco.MjModel.from_xml_path(spec.mjcf_path)
        data = mujoco.MjData(model)
        rng = np.random.default_rng(725)
        q = np.array(spec.default_joint_positions) + rng.uniform(
            -0.05, 0.05, spec.num_joints
        )
        qd = rng.uniform(-0.5, 0.5, spec.num_joints)
        data.qpos[:3] = [0.2, -0.3, 0.8]
        quat = rng.normal(size=4)
        data.qpos[3:7] = quat / np.linalg.norm(quat)
        data.qvel[:6] = rng.uniform(-0.5, 0.5, 6)
        for i, name in enumerate(spec.joint_names):
            joint = model.joint(name).id
            data.qpos[model.jnt_qposadr[joint]] = q[i]
            data.qvel[model.jnt_dofadr[joint]] = qd[i]
        mujoco.mj_forward(model, data)
        body_ids = [model.body(name).id for name in kin.body_names]
        root = model.body(spec.root_body).id
        velocities = []
        angular = []
        for body in body_ids:
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, body)
            velocities.append(jacp @ data.qvel)
            angular.append(jacr @ data.qvel)

        def tensor(x):
            return torch.tensor(np.asarray(x), dtype=torch.float32).unsqueeze(0)

        result = kin.forward(
            tensor(data.xpos[root]),
            tensor(data.xmat[root].reshape(3, 3)),
            tensor(velocities[0]),
            tensor(angular[0]),
            tensor(q),
            tensor(qd),
            all_bodies=True,
        )
        expected = (
            data.xpos[body_ids],
            data.xmat[body_ids].reshape(-1, 3, 3),
            velocities,
            angular,
        )
        for actual, wanted in zip(result, expected, strict=True):
            np.testing.assert_allclose(actual[0].numpy(), wanted, atol=1e-6, rtol=1e-6)
        q_tensor = tensor(q).requires_grad_()
        poses = kin.forward(
            tensor(data.xpos[root]),
            tensor(data.xmat[root].reshape(3, 3)),
            tensor(velocities[0]),
            tensor(angular[0]),
            q_tensor,
            tensor(qd),
        )
        (poses[0].square().sum() + poses[1].square().sum()).backward()
        self.assertTrue(torch.isfinite(q_tensor.grad).all())
        self.assertGreater(q_tensor.grad.abs().sum().item(), 0)

    def test_g1_fk_matches_mujoco_and_legacy_constructor(self):
        self._check_against_mujoco(G1)
        self.assertEqual(G1Kinematics("cpu").tracked_names, G1.tracked_bodies)

    @unittest.skipUnless(
        Path(K1.mjcf_path).is_file(),
        "Prepare the pinned K1 assets before K1 validation",
    )
    def test_k1_fk_matches_mujoco_including_fixed_hand_frames(self):
        self._check_against_mujoco(K1)

    def test_fixed_parent_nonzero_hinge_anchor_and_reference(self):
        xml = """<mujoco><worldbody><body name="root"><freejoint/>
          <geom type="sphere" size=".1" mass="1"/>
          <body name="fixed" pos=".2 -.1 .3" euler=".1 .2 -.3">
            <body name="hinge" pos=".1 .2 .3" euler="-.2 .1 .3">
              <joint name="hinge" type="hinge" axis="0 1 0" pos=".03 .04 .05"
                range="-1 1" ref=".17"/>
              <geom type="capsule" size=".03 .1" mass="1"/>
              <body name="tip" pos=".1 .2 .3"/>
            </body>
          </body>
        </body></worldbody><compiler angle="radian"/></mujoco>"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "anchored.xml"
            path.write_text(xml)
            profile = replace(
                G1,
                name="test",
                joint_names=("hinge",),
                tracked_bodies=("root", "fixed", "hinge", "tip"),
                root_body="root",
                torso_body="root",
                mjcf_path=str(path),
                motors=(G1.motors[0],),
                default_joint_positions=(0.2,),
            )
            self._check_against_mujoco(profile)


@unittest.skipUnless(
    Path(K1.mjcf_path).is_file(), "Prepare K1 assets before retarget validation"
)
class RetargetGeometryTests(unittest.TestCase):
    def _fixture(self):
        from tools.retarget_k1_morphology import ConstrainedRetargeter

        solver = ConstrainedRetargeter(K1.mjcf_path)
        model, data = solver.model, solver.data
        data.qpos[:3] = [0, 0, 1.0]
        data.qpos[3:7] = [1, 0, 0, 0]
        frames, rotations = [], []
        joints = np.zeros((2, 22), dtype=np.float32)
        joints[1, 0] = 0.5
        for q in joints:
            data.qpos[solver.qadr] = q
            mujoco.mj_forward(model, data)
            frames.append(data.xpos[solver.bodies].copy())
            rotations.append(data.xquat[solver.bodies].copy())
        source = {
            "fps": np.array([30]),
            "joint_pos": joints,
            "body_pos_w": np.array(frames),
            "body_quat_w": np.array(rotations),
        }
        return solver, source

    def test_retarget_objective_and_ground_jacobians(self):
        solver, source = self._fixture()
        solver.set_frame(source, 0, None, 30.0)
        # Avoid support-function ties (a perfectly horizontal box face or
        # cylinder axis has a subgradient, not a unique central derivative).
        q = np.random.default_rng(42).uniform(-0.05, 0.05, 22)
        _, gradient = solver.objective(q)
        step = 1e-5
        finite = np.array(
            [
                (
                    solver.objective(q + np.eye(22)[i] * step)[0]
                    - solver.objective(q - np.eye(22)[i] * step)[0]
                )
                / (2 * step)
                for i in range(22)
            ]
        )
        np.testing.assert_allclose(gradient, finite, atol=1e-8, rtol=1e-5)
        _, jacobian = solver.constraints(q, [])
        finite_ground = np.stack(
            [
                (
                    solver.constraints(q + np.eye(22)[i] * step, [])[0]
                    - solver.constraints(q - np.eye(22)[i] * step, [])[0]
                )
                / (2 * step)
                for i in range(22)
            ],
            axis=-1,
        )
        np.testing.assert_allclose(jacobian, finite_ground, atol=1e-7, rtol=1e-5)

    def test_native_contact_jacobian_for_penetrating_mesh_pair(self):
        solver, source = self._fixture()
        solver.set_frame(source, 0, None, 100.0)
        q = np.random.default_rng(13).uniform(-0.02, 0.02, 22)
        q[8], q[9] = -0.15, 0.15
        distances, jacobian = solver._contact_query(q, with_jacobian=True)
        pair = (
            solver.model.geom("left_foot_collision").id,
            solver.model.geom("right_foot_collision").id,
        )
        index = solver.pairs.index(pair)
        self.assertLess(distances[index], -0.01)
        # Validate the contact-normal projection independently against moving
        # witness points. Differencing the native contact search itself also
        # differentiates finite-precision feature selection, not just FK.
        contacts = [
            c
            for c in solver.data.contact[: solver.data.ncon]
            if set(c.geom) == set(pair)
        ]
        contact = min(contacts, key=lambda c: c.dist)
        normal = contact.frame[:3].copy()
        a, b = (int(solver.model.geom_bodyid[g]) for g in contact.geom)
        local_a = solver.data.xmat[a].reshape(3, 3).T @ (
            contact.pos - solver.data.xpos[a]
        )
        local_b = solver.data.xmat[b].reshape(3, 3).T @ (
            contact.pos - solver.data.xpos[b]
        )

        def witness_projection(angles):
            data = solver._forward(angles)
            pa = data.xpos[a] + data.xmat[a].reshape(3, 3) @ local_a
            pb = data.xpos[b] + data.xmat[b].reshape(3, 3) @ local_b
            return normal @ (pb - pa)

        step = 1e-5
        finite = np.array(
            [
                (
                    witness_projection(q + np.eye(22)[j] * step)
                    - witness_projection(q - np.eye(22)[j] * step)
                )
                / (2 * step)
                for j in range(22)
            ]
        )
        np.testing.assert_allclose(jacobian[index], finite, atol=1e-8, rtol=1e-6)

    def test_velocity_limits_constrain_the_optimization(self):
        solver, source = self._fixture()
        previous = source["joint_pos"][0].astype(float)
        solver.set_frame(source, 1, previous, 100.0)
        q, record = solver.solve(previous, 200, 1e-5)
        self.assertTrue(record["success"], record)
        self.assertTrue(
            np.all(np.abs(q - previous) * 100 <= solver.velocity_limits + 1e-6)
        )
        self.assertGreater(q[0], 0.01)
        self.assertLess(q[0], source["joint_pos"][1, 0])

    def test_resampling_preserves_duration_and_root_geometry(self):
        from tools.retarget_k1_morphology import resample_source

        _, source = self._fixture()
        for key in ("joint_pos", "body_pos_w", "body_quat_w"):
            source[key] = np.concatenate(
                [source[key], source[key][-1:].repeat(2, axis=0)]
            )
        selected, times = resample_source(source, 0, 4, 100)
        self.assertEqual(len(times), 11)
        self.assertEqual(times[-1], 0.1)
        np.testing.assert_allclose(
            selected["body_pos_w"][[0, -1]], source["body_pos_w"][[0, -1]], atol=1e-7
        )
        np.testing.assert_allclose(
            selected["body_quat_w"][:, 0], np.tile([1, 0, 0, 0], (11, 1))
        )


if __name__ == "__main__":
    unittest.main()
