"""The climb's simulator and MDP on a toy route (one bar, one facet) and a made-up
motion, on CPU: forward kinematics against the file and the simulator, the
task's state through a step, the cuts, the PD-target bounds, and the world
model's route features. Needs humanoid_boldering's hbold (skipped without it)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

try:
    import mujoco
    from hbold.asset_zoo.routes import kilter
except ImportError:  # pragma: no cover - the flat task does not need hbold
    kilter = None

FRAMES, ENVS, CLIP = 120, 4, "toy"


def _write_toy_clip(root: Path) -> None:
    """route.json and a motion.npz: the robot standing in front of the board,
    its joints swaying, the feet bearing for the first half."""
    from hbold.asset_zoo.robots.g1_hook import g1_hook_constants as g1

    clip = root / "kilter" / CLIP
    clip.mkdir(parents=True)
    bar = {"hold": "a", "pos": [0.3, 0.2, 1.4], "axis": [0.0, -1.0, 0.0], "normal": [-1.0, 0.0, 0.0]}
    facet = {"pos": [0.6, 0.0, 1.0], "rot": np.eye(3).tolist(), "half": [0.2, 1.0, 1.0]}
    size = {k: getattr(kilter.BAR, k) for k in ("radius", "half_length", "out", "post_radius", "facet_thickness")}
    doc = {"clip": CLIP, "scale": 1.0, "angle_deg": 0.0, "offset": [0.0, 0.0, 0.0], "bar": size,
           "facets": [facet], "bars": [bar], "meta": {"tool": "test"}}
    (clip / "route.json").write_text(json.dumps(doc))
    route = kilter.load(clip / "route.json")
    m = kilter.scene_spec(route).compile()
    d = mujoco.MjData(m)
    qpos = np.tile(m.qpos0, (FRAMES, 1))
    qpos[:, :3] = [-0.4, 0.0, 0.8]
    t = np.arange(FRAMES)[:, None] / 50.0
    qpos[:, 7:] += 0.2 * np.sin(np.pi * t + np.random.default_rng(0).uniform(0, 6.3, m.nq - 7))
    bodies = range(1, m.body("route").id)
    sites = [m.site(n).id for n in (*g1.SEAT_SITES, *g1.BALL_SITES)]
    pos, quat, point = (np.empty((FRAMES, len(bodies), 3)), np.empty((FRAMES, len(bodies), 4)),
                        np.empty((FRAMES, 4, 3)))
    for k, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        pos[k], quat[k], point[k] = d.xpos[1:len(bodies) + 1], d.xquat[1:len(bodies) + 1], d.site_xpos[sites]
    vel = np.gradient(pos, 1 / 50.0, axis=0)
    bearing = np.zeros((FRAMES, 4), bool)
    bearing[: FRAMES // 2, 2:] = True
    np.savez(
        clip / "motion.npz", fps=50, body_names=np.array([m.body(b).name for b in bodies]),
        joint_names=np.array([m.joint(j).name for j in range(1, m.njnt)]),
        limb_sites=np.array([*g1.SEAT_SITES, *g1.BALL_SITES]), joint_pos=qpos[:, 7:],
        joint_vel=np.gradient(qpos[:, 7:], 1 / 50.0, axis=0), body_pos_w=pos, body_quat_w=quat,
        body_lin_vel_w=vel, body_ang_vel_w=np.zeros_like(vel), hold_pos_w=point, bearing=bearing,
        limb=np.zeros(FRAMES, int), goal_pos_w=point[:, 0], goal_quat_w=np.tile([1.0, 0, 0, 0], (FRAMES, 1)),
        left_pos_w=point, land_pos_w=point,
    )


@unittest.skipIf(kilter is None, "hbold is not installed")
class ClimbTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        _write_toy_clip(root)
        import hbold.data

        hbold.data.DATA = root  # route_path/motion_path read it at call time
        from supermimic.climb.parts import climb_parts
        from supermimic.climb.sim import ClimbSim, climb_physics
        from supermimic.env import make_sim

        cfg = make_sim("cpu")
        cfg.physics = climb_physics(CLIP)
        cls.kin, cls.motion, cls.mdp = climb_parts(CLIP, "cpu")
        cls.env = ClimbSim(cfg, ENVS, cls.motion, cls.mdp, clip_id=CLIP)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_kinematics_reproduce_the_file_and_the_simulator(self):
        f = torch.arange(0, FRAMES - 1, 7)
        ref = self.motion.ref_state(f)
        torch.testing.assert_close(self.kin.forward(*ref)[0], self.motion.body_p[f], atol=1e-5, rtol=0)
        torch.testing.assert_close(self.kin.site_poses(*ref)[0], self.motion.hold_pos[f], atol=1e-5, rtol=0)
        self.env.reset(torch.arange(ENVS), torch.tensor([0, 10, 40, 90]))
        self.env.refresh()
        names = ("left_seat", "right_seat", "left_ball", "right_ball")
        sim = self.env.robot.data.site_pos_w[:, [self.env.robot.site_names.index(n) for n in names]]
        torch.testing.assert_close(self.kin.site_poses(*self.env.state())[0], sim, atol=1e-3, rtol=0)

    def test_a_step_carries_the_task_state_and_labels(self):
        self.env.reset(torch.arange(ENVS), torch.tensor([0, 10, 40, 90]))
        out = self.env.step(torch.zeros(ENVS, 29))
        self.assertEqual(out["aux"].shape, (ENVS, 8))
        self.assertEqual(out["contact"].shape, (ENVS, 4))
        self.assertTrue(out["valid"].all() and torch.isfinite(out["r"]).all())

    def test_a_torso_far_below_the_reference_is_cut(self):
        f = torch.tensor([5])
        p, R, v, w, q, qd = self.motion.ref_state(f)
        aux = torch.zeros(1, 8)
        _, cut = self.mdp.terminate((p, R, v, w, q, qd), f, aux)
        self.assertFalse(bool(cut))
        _, cut = self.mdp.terminate((p - torch.tensor([0.0, 0.0, 0.5]), R, v, w, q, qd), f, aux)
        self.assertTrue(bool(cut))

    def test_the_pd_bounds_hold_the_default_pose_and_are_finite(self):
        low, high = self.env.action_bounds()
        self.assertTrue(torch.isfinite(low).all() and torch.isfinite(high).all())
        self.assertTrue((low < 0).all() and (high > 0).all())

    def test_failure_rate_starts_follow_where_episodes_fail(self):
        from supermimic.climb.sim import ClimbSim

        env = object.__new__(ClimbSim)
        env.motion, env.device, env.start_sampler = self.motion, "cpu", "failure_rate"
        nb = self.motion.num_bins
        env.bin_visit = torch.ones(nb)
        env.bin_fail = torch.zeros(nb)
        env.bin_fail[1] = 1.0
        from supermimic.climb.sim import FAIL_KERNEL, FAIL_LAMBDA

        k = torch.tensor([FAIL_LAMBDA**i for i in range(FAIL_KERNEL)])
        env.fail_kernel = k / k.sum()
        frames, bins = env.sample_frames(4000)
        share = torch.bincount(bins, minlength=nb).float() / 4000
        self.assertGreater(float(share[1]), float(share[0]))  # the failing bin draws more than a flat share
        self.assertTrue(bool((frames <= self.motion.clip_end[0] - 1).all()))

    def test_the_route_features_are_differentiable_in_the_state(self):
        p, R, v, w, q, qd = (x.clone() for x in self.motion.ref_state(torch.tensor([3, 4])))
        p.requires_grad_(True)
        feats = self.mdp.wm_features((p, R, v, w, q, qd))
        self.assertEqual(feats.shape, (2, self.mdp.wm_extra_dim))
        feats.square().sum().backward()
        self.assertTrue(torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)


if __name__ == "__main__":
    unittest.main()
