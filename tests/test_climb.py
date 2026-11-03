"""The climb's simulator and MDP on a toy route (one bar, one facet) and a made-up
motion, on CPU: forward kinematics against the file and the simulator, the
task's state through a step, the cuts, the PD-target bounds, and the world
model's route features; and two clips on two routes in one simulator. Needs
humanoid_boldering's hbold (skipped without it)."""

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


def _write_toy_clip(root: Path, name: str = CLIP, bar_y: float = 0.2) -> None:
    """route.json and a motion.npz: the robot standing in front of the board,
    its joints swaying, the feet bearing for the first half. `bar_y`: where
    the route's one bar sits across the board."""
    from hbold.asset_zoo.robots.g1_hook import g1_hook_constants as g1

    clip = root / "kilter" / name
    clip.mkdir(parents=True)
    bar = {"hold": "a", "pos": [0.3, bar_y, 1.4], "axis": [0.0, -1.0, 0.0], "normal": [-1.0, 0.0, 0.0]}
    facet = {"pos": [0.6, 0.0, 1.0], "rot": np.eye(3).tolist(), "half": [0.2, 1.0, 1.0]}
    size = {k: getattr(kilter.BAR, k) for k in ("radius", "half_length", "out", "post_radius", "facet_thickness")}
    doc = {"clip": name, "scale": 1.0, "angle_deg": 0.0, "offset": [0.0, 0.0, 0.0], "bar": size,
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

    def test_the_soft_off_hold_cut_is_the_hard_one_with_a_gradient(self):
        from hbold.climb import limbs, task

        from supermimic.climb.mdp import soft_off_hold, soft_timed_out

        cut, dt = 0.05, 0.02
        far = torch.tensor([[0.02, 0.049, 0.051, 0.2], [0.0, 0.3, 0.098, 0.102], [0.36, 0.0, 0.0, 0.0]])
        turn = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.2], [0.0, 0.0, 0.0, 0.0]])
        bearing = torch.tensor([[True, True, True, True], [True, False, True, False], [False] * 4])
        hard = limbs.off_hold_step(torch.zeros(3, 4), far, turn, bearing, cut, task.FOOT_SCALE, task.FREE_CUT,
                                   task.TURN_CUT, dt) > 0
        timer = torch.tensor([[0.0, 0.48, 0.5, 0.52], [0.3, 0.6, 0.0, 0.0], [0.0] * 4])
        errs = [float((soft_off_hold(far, turn, bearing, cut, temp) - hard.float()).abs().max())
                for temp in (1e-2, 1e-3, 1e-5)]
        self.assertTrue(errs[0] > errs[1] > errs[2] and errs[2] < 1e-6)  # the soft tests tend to the hard ones
        torch.testing.assert_close(soft_timed_out(timer, task.HOLD_TOLERANCE, dt, 1e-5),
                                   limbs.timed_out(timer, task.HOLD_TOLERANCE).float(), atol=1e-6, rtol=0)

        # A foot the reference bears on its hold, 1.1 times its cut off it: its timer far from its
        # tolerance (no cut) or one step from it (cut). The values are the simulator's either way,
        # and either way the cut has a gradient in the state.
        f = torch.tensor([5])
        p, R, v, w, q, qd = self.motion.ref_state(f)
        foot = int(self.motion.bearing[f][0].nonzero()[0])
        self.assertGreaterEqual(foot, 2)
        self.addCleanup(setattr, self.mdp, "hold_cut", self.mdp.hold_cut)
        self.mdp.hold_cut = cut
        for timer_s, cut_now in ((0.0, False), (task.HOLD_TOLERANCE, True)):
            aux = torch.zeros(1, 8)
            aux[0, foot] = timer_s
            p_off = p.clone().requires_grad_(True)
            S = (p_off + torch.tensor([0.0, 0.0, 1.1 * task.FOOT_SCALE * cut]), R, v, w, q, qd)
            hard_aux, hard_cut = self.mdp.terminate(S, f, aux)
            soft_aux, soft_cut = self.mdp.terminate(S, f, aux, soft=(0.01, 0.1, True))
            self.assertEqual(bool(hard_cut), cut_now)
            self.assertGreater(float(hard_aux[0, foot]), 0)
            torch.testing.assert_close(soft_aux.detach(), hard_aux, atol=0, rtol=0)
            self.assertEqual(float(soft_cut.detach()), float(hard_cut))
            soft_cut.sum().backward()
            self.assertGreater(float(p_off.grad[0, 2]), 0)  # higher, further off: nearer the cut

    def test_the_soft_stayed_cut_tends_to_the_hard_one_with_a_gradient(self):
        from hbold.climb import limbs, task

        from supermimic.climb.mdp import soft_stayed

        left = torch.zeros(2, 4, 3)
        ref = torch.zeros(2, 4, 3)
        ref[:, :, 2] = 0.2  # the reference 20 cm gone from every hold
        site = torch.zeros(2, 4, 3)
        site[0, :, 0] = torch.tensor([0.02, 0.049, 0.051, 0.1])
        site[1, :, 0] = 0.03
        bearing = torch.tensor([[False] * 4, [False, True, False, True]])
        hard = limbs.stayed_step(torch.zeros(2, 4), site, left, ref, bearing, task.STAY_NEAR, task.STAY_GONE, 0.02) > 0
        errs = [float((soft_stayed(site, left, ref, bearing, task.STAY_NEAR, task.STAY_GONE, t, {}) - hard.float())
                      .abs().max()) for t in (1e-2, 1e-3, 1e-5)]
        self.assertTrue(errs[0] > errs[1] > errs[2] and errs[2] < 1e-6)
        site.requires_grad_(True)
        soft_stayed(site, left, ref, bearing, task.STAY_NEAR, task.STAY_GONE, 0.01, {}).sum().backward()
        self.assertLess(float(site.grad[0, 0, 0]), 0)  # further from the old hold: less stayed
        self.assertEqual(float(site.grad[1, 1].abs().sum()), 0.0)  # a bearing limb has no stayed test

    def test_the_pd_bounds_hold_the_default_pose_and_are_finite(self):
        low, high = self.env.action_bounds()
        self.assertTrue(torch.isfinite(low).all() and torch.isfinite(high).all())
        self.assertTrue((low < 0).all() and (high > 0).all())

    def test_failure_rate_starts_follow_where_episodes_fail(self):
        from supermimic.climb.sim import ClimbSim

        env = object.__new__(ClimbSim)
        env.motion, env.device, env.start_sampler, env.clip_start_share = self.motion, "cpu", "failure_rate", 0.0
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

    def test_a_share_of_starts_is_at_the_clip_start(self):
        from supermimic.climb.sim import ClimbSim

        env = object.__new__(ClimbSim)
        env.motion, env.device, env.start_sampler, env.clip_start_share = self.motion, "cpu", "failure_rate", 0.3
        nb = self.motion.num_bins
        env.bin_visit, env.bin_fail = torch.ones(nb), torch.zeros(nb)
        env.fail_kernel = torch.ones(1)
        frames, bins = env.sample_frames(4000)
        share = float((frames == 0).float().mean())
        self.assertGreater(share, 0.25)
        self.assertLess(share, 0.36)
        torch.testing.assert_close(bins, self.motion.frame_bin[frames])

    def test_the_route_features_are_differentiable_in_the_state(self):
        p, R, v, w, q, qd = (x.clone() for x in self.motion.ref_state(torch.tensor([3, 4])))
        p.requires_grad_(True)
        feats = self.mdp.wm_features((p, R, v, w, q, qd))
        self.assertEqual(feats.shape, (2, self.mdp.wm_extra_dim))
        feats.square().sum().backward()
        self.assertTrue(torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)


@unittest.skipIf(kilter is None, "hbold is not installed")
class TwoRouteTests(unittest.TestCase):
    """Clip "toy" on route 0 and "toy2" (its bar elsewhere) on route 1."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        _write_toy_clip(root)
        _write_toy_clip(root, "toy2", bar_y=-0.3)
        import hbold.data

        hbold.data.DATA = root
        from supermimic.climb.parts import climb_parts
        from supermimic.climb.sim import ClimbSim, climb_physics
        from supermimic.env import make_sim

        clips = f"{CLIP},toy2"
        cfg = make_sim("cpu")
        cfg.physics = climb_physics(clips)
        cls.kin, cls.motion, cls.mdp = climb_parts(clips, "cpu")
        cls.env = ClimbSim(cfg, ENVS, cls.motion, cls.mdp, clip_id=clips)
        cls.only2 = climb_parts("toy2", "cpu")[2]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_each_clip_plays_on_its_own_route_tile(self):
        from supermimic.climb.sim import ROUTE_TILE

        frames = torch.tensor([10, 40, FRAMES + 10, FRAMES + 40])
        self.env.reset(torch.arange(ENVS), frames)
        self.env.refresh()
        S = self.env.state()
        for x, ref in zip(S, self.motion.ref_state(frames)):
            torch.testing.assert_close(x, ref, atol=1e-5, rtol=0)
        y = self.env.robot.data.root_link_pos_w[:, 1] - self.motion.ref_state(frames)[0][:, 1]
        torch.testing.assert_close(y, torch.tensor([0.0, 0.0, ROUTE_TILE, ROUTE_TILE]), atol=1e-5, rtol=0)
        m = self.env.sim.mj_model
        bar = [m.geom(f"{e}/bar_a").id for e in ("route", "route1")]
        self.env.sim.forward()
        at = self.env.sim.data.geom_xpos[0, bar]
        torch.testing.assert_close(at[1] - at[0], torch.tensor([0.0, ROUTE_TILE - 0.5, 0.0]), atol=1e-5, rtol=0)
        out = self.env.step(torch.zeros(ENVS, 29))
        self.assertTrue(out["valid"].all() and torch.isfinite(out["r"]).all())
        self.assertEqual(out["contact"].shape, (ENVS, 4))

    def test_a_state_sees_only_its_own_route(self):
        f = torch.tensor([FRAMES + 3, FRAMES + 50])
        S = self.motion.ref_state(f)
        torch.testing.assert_close(self.mdp.wm_features(S, f=f), self.only2.wm_features(S, f=f - FRAMES))
        self.assertFalse(torch.allclose(self.mdp.wm_features(S, f=f), self.mdp.wm_features(S, f=f - FRAMES)))
        with self.assertRaises(ValueError):
            self.mdp.wm_features(S)


if __name__ == "__main__":
    unittest.main()
