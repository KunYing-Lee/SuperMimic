"""Algorithm invariants independent of the CUDA simulator."""

from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from supermimic.buffer import FIELDS, ReplayBuffer
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import INV_DIM, MDP, OBS_DIM, invariant_features, normalize_xi
from supermimic.motion import MotionLib
from supermimic.networks import WorldModel
from supermimic.robot import default_joint_pos_list
from supermimic.rotations import (
    axis_angle_to_mat,
    matmul3,
    matvec3,
    rot_angle_sq,
    so3_exp,
    yaw_mat,
)


def state(batch=3):
    return (
        torch.zeros(batch, 3),
        torch.eye(3).repeat(batch, 1, 1),
        torch.zeros(batch, 3),
        torch.zeros(batch, 3),
        torch.zeros(batch, 29),
        torch.zeros(batch, 29),
    )


def transition(frame, done=False):
    out = {}
    for key, (shape, dtype) in FIELDS.items():
        tail = (shape,) if isinstance(shape, int) else shape
        out[key] = torch.zeros((1,) + tail, dtype=dtype)
    out["s"][:] = frame
    out["s1"][:] = frame + 1
    out["f"][:] = frame
    out["ep_step"][:] = frame % 3
    out["done"][:] = done
    out["terminated"][:] = done
    return out


class ReplayTests(unittest.TestCase):
    def test_wrap_history_and_restart(self):
        replay = ReplayBuffer(5, 1, "cpu", "cpu")
        for t in range(8):
            replay.add(transition(t, done=t % 3 == 2))
        saved = copy.deepcopy(replay.state_dict())
        restored = ReplayBuffer(5, 1, "cpu", "cpu")
        restored.load_state_dict(saved)
        torch.manual_seed(19)
        rng = torch.get_rng_state()
        starts, history = replay.sample_starts(50, 4)
        torch.set_rng_state(rng)
        starts2, history2 = restored.sample_starts(50, 4)
        for key in starts:
            torch.testing.assert_close(starts[key], starts2[key], rtol=0, atol=0)
            torch.testing.assert_close(history[key], history2[key], rtol=0, atol=0)
        f = starts["f"]
        expected = torch.maximum(
            f[:, None] - torch.arange(3, -1, -1), (f - f % 3)[:, None]
        ).clamp(min=3)
        torch.testing.assert_close(history["f"], expected)

    def test_missing_history_at_replay_boundary_is_not_invented(self):
        replay = ReplayBuffer(3, 1, "cpu")
        for t in range(6):
            tr = transition(t)
            tr["ep_step"][:] = t
            replay.add(tr)
        starts, history = replay.sample_starts(20, 3)
        self.assertTrue((starts["f"] == 5).all())
        torch.testing.assert_close(
            history["f"], torch.tensor([[3, 4, 5]]).expand(20, -1)
        )

    def test_window_mask_includes_terminal_transition(self):
        replay = ReplayBuffer(3, 1, "cpu")
        for t in range(3):
            replay.add(transition(t, done=t == 0))
        sampled = replay.sample_windows(5, 3)
        torch.testing.assert_close(
            sampled["mask"], torch.tensor([[1.0, 0.0, 0.0]]).expand(5, -1)
        )
        self.assertTrue(sampled["terminated"][:, 0].all())

    def test_eligible_sampling_is_explicit_and_empty_set_fails(self):
        replay = ReplayBuffer(5, 1, "cpu")
        for t in range(5):
            replay.add(transition(t))
        end = torch.full((5,), 4)
        starts, _ = replay.sample_starts(100, 2, end, 3)
        self.assertTrue((starts["f"] <= 1).all())
        with self.assertRaisesRegex(ValueError, "no start state"):
            replay.sample_starts(1, 2, end, 8)


class RotationAndGoalTests(unittest.TestCase):
    def test_geometry_contractions_match_broadcast_products_and_gradients(self):
        generator = torch.Generator().manual_seed(14)
        a = torch.randn(
            2, 1, 3, 3, dtype=torch.float64, generator=generator, requires_grad=True
        )
        b = torch.randn(
            1, 4, 3, 3, dtype=torch.float64, generator=generator, requires_grad=True
        )
        v = torch.randn(
            1, 4, 3, dtype=torch.float64, generator=generator, requires_grad=True
        )
        torch.testing.assert_close(matmul3(a, b), a @ b, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(
            matvec3(a, v), (a @ v.unsqueeze(-1)).squeeze(-1), rtol=1e-14, atol=1e-14
        )
        self.assertTrue(torch.autograd.gradcheck(matmul3, (a, b)))
        self.assertTrue(torch.autograd.gradcheck(matvec3, (a, v)))

    def test_fp32_rotation_integration_preserves_geometry(self):
        # Exercise the same composition depth as the 32-step imagined rollout.
        # The reference computes every operation in float64 independently.
        generator = torch.Generator().manual_seed(26)
        increments = torch.randn(32, 8, 3, generator=generator) * 0.1
        result = torch.eye(3).repeat(8, 1, 1)
        reference = result.double()
        for increment in increments:
            result = matmul3(so3_exp(increment), result)
            reference = so3_exp(increment.double()) @ reference
        torch.testing.assert_close(result.double(), reference, rtol=3e-6, atol=1e-6)
        torch.testing.assert_close(
            matmul3(result.transpose(-1, -2), result),
            torch.eye(3).expand(8, -1, -1),
            rtol=0,
            atol=2e-6,
        )
        torch.testing.assert_close(
            torch.linalg.det(result), torch.ones(8), rtol=0, atol=2e-6
        )

    def test_squared_angle_has_exact_zero_and_correct_small_gradient(self):
        vectors = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0001, 0.0002, -0.0003]],
            dtype=torch.float64,
            requires_grad=True,
        )
        angles = rot_angle_sq(
            torch.eye(3, dtype=torch.float64).expand(2, -1, -1), so3_exp(vectors)
        )
        self.assertEqual(angles[0].item(), 0.0)
        torch.testing.assert_close(
            angles, vectors.square().sum(-1), rtol=1e-7, atol=1e-14
        )
        angles.sum().backward()
        self.assertTrue(torch.isfinite(vectors.grad).all())
        torch.testing.assert_close(
            vectors.grad, 2 * vectors.detach(), rtol=1e-6, atol=1e-12
        )

    def test_world_model_body_frame_represents_motion_at_vertical_pelvis(self):
        vertical = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        s = list(state(1))
        s[1] = vertical.unsqueeze(0)
        next_state = list(s)
        next_state[0] = torch.tensor([[0.3, -0.2, 0.1]])
        next_state[2] = torch.tensor([[0.6, 0.4, -0.1]])
        target = WorldModel.targets(s, next_state)
        torch.testing.assert_close(
            target[:, :3].square().sum(-1), next_state[0].square().sum(-1)
        )
        wm = WorldModel(hidden=(8,), goal_dim=4)
        wm.out_norm.mean.copy_(target[0])
        prediction = wm.step(s, torch.zeros(1, 29), torch.zeros(1, 33))
        for actual, desired in zip(prediction, next_state):
            torch.testing.assert_close(actual, desired, rtol=1e-6, atol=1e-7)
        features = invariant_features(s)
        self.assertEqual(features.shape[-1], INV_DIM)
        self.assertEqual(INV_DIM, 68)
        self.assertTrue(torch.isfinite(features).all())
        torch.testing.assert_close(
            WorldModel.state_error(next_state, s)[:, :3].square().sum(-1),
            next_state[0].square().sum(-1),
        )

    def test_heading_is_a_proper_rotation_without_epsilon_shrinkage(self):
        # The exact chart singularity has a conventional atan2 value, but no
        # finite derivative; it is not used in dynamics or goals.
        relative = torch.tensor(
            [
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
                [[1e-9, 0.0, 1.0], [1e-9, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            ]
        )
        headings = yaw_mat(relative)
        torch.testing.assert_close(
            headings.transpose(-1, -2) @ headings, torch.eye(3).expand(2, -1, -1)
        )
        torch.testing.assert_close(torch.linalg.det(headings), torch.ones(2))

    def test_robot_relative_goal_is_se2_invariant_and_observes_yaw(self):
        m = object.__new__(MotionLib)
        m.clip_id = torch.zeros(4, dtype=torch.long)
        m.clip_end = torch.tensor([3])
        m.root_p = torch.tensor(
            [[0.0, 0.0, 1.0], [0.1, 0.2, 1.0], [0.2, 0.4, 1.0], [0.3, 0.6, 1.0]]
        )
        m.root_R = torch.eye(3).repeat(4, 1, 1)
        m.root_v = torch.tensor([[1.0, 2.0, 0.0]]).repeat(4, 1)
        m.root_w = torch.tensor([[0.0, 0.0, 0.1]]).repeat(4, 1)
        m.q, m.qd = torch.zeros(4, 29), torch.zeros(4, 29)
        s = list(state(1))
        s[0] = torch.tensor([[0.01, 0.03, 0.95]])
        f = torch.tensor([0])
        goal = m.goal(f, 2, s)
        yaw = axis_angle_to_mat(torch.tensor([0.0, 0.0, 1.0]), torch.tensor(0.7))
        translated = torch.tensor([3.0, -4.0, 0.0])
        m2 = copy.deepcopy(m)
        m2.root_p = (yaw @ m.root_p.unsqueeze(-1)).squeeze(-1) + translated
        m2.root_R = yaw @ m.root_R
        m2.root_v = (yaw @ m.root_v.unsqueeze(-1)).squeeze(-1)
        m2.root_w = (yaw @ m.root_w.unsqueeze(-1)).squeeze(-1)
        s2 = list(s)
        s2[0] = (yaw @ s[0].unsqueeze(-1)).squeeze(-1) + translated
        s2[1] = yaw @ s[1]
        torch.testing.assert_close(m2.goal(f, 2, s2), goal, rtol=1e-5, atol=1e-6)
        s3 = list(s)
        s3[1] = yaw @ s[1]
        self.assertFalse(torch.allclose(m.goal(f, 2, s3), goal))
        self.assertEqual(OBS_DIM, 93)
        vertical = torch.tensor(
            [[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]], requires_grad=True
        )
        s4 = list(s)
        s4[1] = vertical
        vertical_goal = m.goal(f, 2, s4)
        self.assertTrue(torch.isfinite(vertical_goal).all())
        (vertical_goal * torch.arange(vertical_goal.numel())).sum().backward()
        self.assertTrue(torch.isfinite(vertical.grad).all())


class RecomputeTests(unittest.TestCase):
    def test_actual_imagination_step_preserves_gradients_with_frozen_world_model(self):
        # Load the actual function without running train.py's CUDA startup.
        # The fixture uses the same MDP/FK/goal implementations as the trainer.
        source_path = Path(__file__).parents[1] / "supermimic" / "train.py"
        tree = ast.parse(source_path.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_imag_step"
        )
        module = ast.Module(body=[function], type_ignores=[])
        kin = G1Kinematics("cpu")
        motion = object.__new__(MotionLib)
        frames, batch, K, history = 8, 2, 2, 3
        motion.clip_id = torch.zeros(frames, dtype=torch.long)
        motion.clip_end = torch.tensor([frames - 1])
        motion.root_p = torch.tensor([[0.0, 0.0, 0.8]]).repeat(frames, 1)
        motion.root_R = torch.eye(3).repeat(frames, 1, 1)
        motion.root_v = torch.zeros(frames, 3)
        motion.root_w = torch.zeros(frames, 3)
        default_q = torch.tensor(default_joint_pos_list())
        motion.q = default_q.repeat(frames, 1)
        motion.qd = torch.zeros(frames, 29)
        motion.body_p, motion.body_R, motion.body_v, motion.body_w = kin.forward(
            *motion.ref_state(torch.arange(frames))
        )
        mdp = MDP(kin, motion, default_q)
        torch.manual_seed(35)
        wm = WorldModel(hidden=(16, 16), goal_dim=73 * K)
        torch.nn.init.normal_(wm.net[-1].weight, std=0.001)
        for parameter in wm.parameters():
            parameter.requires_grad_(False)

        class TinyPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(history * OBS_DIM + 73 * K, 16),
                    torch.nn.Tanh(),
                    torch.nn.Linear(16, 29),
                )

            def forward(self, observations, goal, depth=None):
                return self.net(torch.cat((observations.flatten(1), goal), -1)) * 0.01

        pi = TinyPolicy()
        pi2 = copy.deepcopy(pi)
        wm_before = {key: value.clone() for key, value in wm.state_dict().items()}
        initial = motion.ref_state(torch.zeros(batch, dtype=torch.long))
        xi = torch.zeros(batch, 33)
        xi[:, 0] = 0.75
        xi_n = normalize_xi(xi)
        actions = torch.zeros(batch, 29)
        f0 = torch.zeros(batch, dtype=torch.long)
        observations = (
            mdp.obs(initial, actions, xi, f0).unsqueeze(1).repeat(1, history, 1)
        )

        def rollout(policy, recompute):
            namespace = {
                "pi": policy,
                "wm": wm,
                "motion": motion,
                "mdp": mdp,
                "K": K,
                "torch": torch,
                "depth_pipeline": None,
                "depth_model": None,
                "terrain_scan": lambda S, f: None,
            }
            exec(compile(module, str(source_path), "exec"), namespace)
            actual_step = namespace["_imag_step"]
            current, previous, hist, frame = initial, actions, observations, f0
            goal = motion.goal(frame, K, current)
            survival = torch.ones(batch)
            objective = torch.zeros(batch)
            for t in range(3):
                inputs = (
                    current,
                    hist,
                    previous,
                    frame,
                    xi,
                    xi_n,
                    goal,
                    None,
                    None,
                    None,
                    None,
                )
                result = (
                    checkpoint(
                        actual_step,
                        *inputs,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                    if recompute
                    else actual_step(*inputs)
                )
                current, previous, hist, frame, goal, reward, termination, depth = (
                    result
                )
                objective = objective + 0.99**t * survival * reward
                survival = survival * (1 - termination)
            loss = -objective.mean()
            loss.backward()
            return loss.detach(), [component.detach() for component in current]

        rng = torch.get_rng_state()
        direct, direct_state = rollout(pi, False)
        torch.set_rng_state(rng)
        recomputed, recomputed_state = rollout(pi2, True)
        torch.testing.assert_close(direct, recomputed, rtol=0, atol=0)
        for left, right in zip(direct_state, recomputed_state):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for left, right in zip(pi.parameters(), pi2.parameters()):
            self.assertTrue(torch.isfinite(left.grad).all())
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in pi.parameters()), 0.0)
        for key, value in wm.state_dict().items():
            torch.testing.assert_close(value, wm_before[key], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in wm.parameters()))

    def test_autoregressive_checkpoint_preserves_values_and_parameter_gradients(self):
        torch.manual_seed(4)
        wm = WorldModel(hidden=(32, 32), goal_dim=4)
        torch.nn.init.normal_(wm.net[-1].weight, std=0.01)
        wm2 = copy.deepcopy(wm)
        initial = state(2)
        actions = torch.randn(3, 2, 29)
        xi = torch.zeros(2, 33)

        def rollout(model, recompute):
            current = initial
            total = torch.zeros(())
            for a in actions:

                def step(S, action):
                    return model.step(S, action + torch.rand_like(action) * 0.01, xi)

                current = (
                    checkpoint(
                        step, current, a, use_reentrant=False, preserve_rng_state=True
                    )
                    if recompute
                    else step(current, a)
                )
                total = total + sum(value.square().mean() for value in current)
            total.backward()
            return total.detach()

        rng = torch.get_rng_state()
        loss = rollout(wm, False)
        torch.set_rng_state(rng)
        loss2 = rollout(wm2, True)
        torch.testing.assert_close(loss, loss2, rtol=0, atol=0)
        for p, q in zip(wm.parameters(), wm2.parameters()):
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
