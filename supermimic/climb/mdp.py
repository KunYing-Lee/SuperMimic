"""The climb's MDP heads on SuperMimic's state tuple, for the simulator and for
imagined world-model steps alike: hbold.climb's rewards and cuts evaluated on
forward kinematics of the state (tracked bodies, each limb's site), so the
imagined reward and termination are the simulator's own rules, not learned.

Positions are in the route's frame: the simulator puts every world's route at
the origin (env spacing 0), so the state's pelvis position is route-relative.
The anchor of the tracking terms is the torso, as in hbold's (BeyondMimic's).
"""

from __future__ import annotations

import torch
from hbold.climb import geometry, limbs, task, tracking

from ..mdp import MDP
from ..motion import CONTROL_DT
from ..robot import TRACKED_BODIES
from ..rotations import mat_to_quat
from .motion import ClimbMotion

LIMB_SITES = ("left_seat", "right_seat", "left_ball", "right_ball")  # hbold.data.LIMBS' order
LIMB_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link")
ANCHOR = TRACKED_BODIES.index("torso_link")
LIMB_IDX = [TRACKED_BODIES.index(b) for b in LIMB_BODIES]
NEAR_BARS = 4  # bars each limb sees in the world model's inputs
GOAL_EXTRA = 9 + 4  # the next landing in the pelvis frame, and which limb it is
TRACKING_MAX = sum(weight for _, weight in task.TRACKING.values())
TURN_PER_M = 10.0  # a soft cut's temperature in turn per its temperature in distance: 1 cm ~ 0.1 rad


def soft_off_hold(far, turn, bearing, threshold: float, temp: float):
    """(N, 4) a smooth stand-in for off_hold_step's off test: each of its three
    comparisons as a sigmoid of (value - cut) / temp, joined as a soft or.
    Tends to the hard test as temp -> 0."""
    cut = torch.full((far.shape[-1],), threshold, device=far.device)
    cut[2:] *= task.FOOT_SCALE  # the feet, as in off_hold_step
    on_hold = torch.sigmoid((far - cut) / temp) * bearing
    free = torch.sigmoid((far - task.FREE_CUT) / temp)
    turned = torch.sigmoid((turn - task.TURN_CUT) / (temp * TURN_PER_M))
    return 1 - (1 - on_hold) * (1 - free) * (1 - turned)


def soft_timed_out(timer_s, tolerance: float, dt: float, temp_s: float):
    """(N,) a smooth stand-in for timed_out: each limb's sigmoid of (timer - the
    cut's midpoint) / temp_s, joined as a soft or. The timers step by dt, so the
    midpoint between the last kept and the first cut value is tolerance + dt / 2."""
    return 1 - (1 - torch.sigmoid((timer_s - tolerance - dt / 2) / temp_s)).prod(-1)


def straight_through(hard, soft):
    """hard's value with soft's gradient."""
    return hard + (soft - soft.detach())


class ClimbMDP(MDP):
    analytic_termination = True
    aux_dim = 8  # each limb's seconds off its hold (off_hold), then on the hold it should have left (stayed)
    wm_extra_dim = len(LIMB_SITES) * NEAR_BARS * 10 + 4
    contact_dim = len(LIMB_SITES)
    # tracking, plus limb_pose (one per limb) and free_limb or feet_on_hold (at most one per limb)
    max_step_reward = (TRACKING_MAX + 2 * len(LIMB_SITES)) * CONTROL_DT
    critic_extra = MDP.critic_extra + 3 * len(LIMB_SITES)

    def __init__(self, kin, motion: ClimbMotion, default_q: torch.Tensor, bars: geometry.Bars,
                 board: geometry.Board):
        super().__init__(kin, motion, default_q)
        self.motion: ClimbMotion = motion
        self.bars, self.board = bars, board
        self.soft_lo, self.soft_hi = kin.q_lo, kin.q_hi  # hbold runs the joints to their full range
        self.hold_cut = task.hold_cut(0)  # the trainer moves it on by environment step

    def schedule(self, env_steps: int) -> None:
        """hbold's off-hold cut, tightening by environment step."""
        self.hold_cut = task.hold_cut(env_steps)

    # ---------------- state as hbold.climb's tensors ----------------
    def features(self, S) -> tuple:
        """One pass of forward kinematics: the tracked bodies (positions, wxyz
        quaternions, linear and angular velocities) and the limb sites (positions,
        rotations). Every head of a step reads these; the chain runs once."""
        bp, bR, bv, bw, sites, site_R = self.kin.bodies_and_sites(*S)
        return bp, mat_to_quat(bR), bv, bw, sites, site_R

    def _limbs(self, F, f):
        """(far, turn) of each limb against the reference at frame f."""
        _, bq, _, _, sites, _ = F
        ref_q = self.motion.body_quat[f][:, LIMB_IDX]
        return limbs.limb_errors(sites, bq[:, LIMB_IDX], self.motion.hold_pos[f], ref_q)

    # ---------------- task hooks ----------------
    def goal_dim(self, K: int) -> int:
        return 73 * K + GOAL_EXTRA

    def goal(self, f, K: int, S) -> torch.Tensor:
        m = self.motion
        landing = limbs.goal(S[0], mat_to_quat(S[1]), m.goal_pos[f], m.goal_quat[f])
        return torch.cat((m.goal(f, K, S), landing, limbs.limb_onehot(m.limb[f])), -1)

    def wm_features(self, S, F=(), f=None) -> torch.Tensor:
        """The route as the world model sees it: each limb's nearest bars in its
        site's frame, and the board's distance and face from the pelvis. `f`:
        each state's reference frame, which says its clip's route (needed when
        there is more than one route)."""
        sites, site_R = (F or self.features(S))[4:]
        route = None if f is None else self.motion.clip_id[f]
        if route is None and len(self.motion.names) > 1:
            raise ValueError("several routes: wm_features needs each state's reference frame")
        bars = geometry.nearest_bars(sites, mat_to_quat(site_R), self.bars, NEAR_BARS, route=route)
        board = geometry.board_features(S[0], mat_to_quat(S[1]), self.board, route=route)
        return torch.cat((bars.flatten(-3), board), -1)

    def terminate(self, S, f, aux, F=(), soft=None):
        """hbold's cuts after a step to S at reference frame f: the torso's
        height, a limb off its hold (off_hold) or on the one it should have
        left (stayed) past their tolerances. aux: (B, 8) their timers.

        soft: None, or (temp in m, temp in s) for imagination. The cut is then a
        float with the hard cut's value but the gradient of a smooth off-hold
        cut: each limb's off test and its timer's cut as sigmoids (soft_off_hold,
        soft_timed_out), through the timers carried in aux. The values, timers
        included, stay the simulator's; only the gradient says how near a limb
        is to its cut."""
        m = self.motion
        F = F or self.features(S)
        bp, sites = F[0], F[4]
        far, turn = self._limbs(F, f)
        bearing = m.bearing[f]
        off = limbs.off_hold_step(aux[:, :4], far, turn, bearing, self.hold_cut, task.FOOT_SCALE,
                                  task.FREE_CUT, task.TURN_CUT, CONTROL_DT)
        on = limbs.stayed_step(aux[:, 4:], sites, m.left_pos[f], m.hold_pos[f], bearing, task.STAY_NEAR,
                               task.STAY_GONE, CONTROL_DT)
        low = tracking.anchor_z_off(bp[:, ANCHOR], m.body_p[f][:, ANCHOR], task.ANCHOR_Z_CUT)
        timed_off = limbs.timed_out(off, task.HOLD_TOLERANCE)
        rest = low | limbs.timed_out(on, task.STAY_TOLERANCE)
        if soft is None:
            return torch.cat((off, on), -1), rest | timed_off
        temp, temp_s = soft
        is_off = straight_through((off > 0).float(), soft_off_hold(far, turn, bearing, self.hold_cut, temp))
        off = (aux[:, :4] + CONTROL_DT) * is_off  # off_hold_step's timers, differentiable
        timed_off = straight_through(timed_off.float(), soft_timed_out(off, task.HOLD_TOLERANCE, CONTROL_DT, temp_s))
        return torch.cat((off, on), -1), 1 - (1 - rest.float()) * (1 - timed_off)

    def termination(self, S, f) -> torch.Tensor:
        raise NotImplementedError("the climb's cuts carry timers: use terminate(S, f, aux)")

    # ---------------- reward ----------------
    def reward(self, S, f, a, a_prev, terms: bool = False, F=()):
        """hbold's per-step reward (times dt): mjlab's tracking terms about the
        torso, the climb's limb terms, the action rate and the joint limits."""
        m = self.motion
        F = F or self.features(S)
        bp, bq, bv, bw = F[:4]
        ref_p, ref_q = m.body_p[f], m.body_quat[f]
        rel_p, rel_q = tracking.relative_body_poses(
            bp[:, ANCHOR], bq[:, ANCHOR], ref_p[:, ANCHOR], ref_q[:, ANCHOR], ref_p, ref_q
        )
        tr = task.TRACKING
        track = {
            "anchor_pos": tracking.anchor_pos_exp(bp[:, ANCHOR], ref_p[:, ANCHOR], tr["anchor_pos"][0]),
            "anchor_ori": tracking.anchor_ori_exp(bq[:, ANCHOR], ref_q[:, ANCHOR], tr["anchor_ori"][0]),
            "body_pos": tracking.body_pos_exp(bp, rel_p, tr["body_pos"][0]),
            "body_ori": tracking.body_ori_exp(bq, rel_q, tr["body_ori"][0]),
            "body_lin_vel": tracking.body_vel_exp(bv, m.body_v[f], tr["body_lin_vel"][0]),
            "body_ang_vel": tracking.body_vel_exp(bw, m.body_w[f], tr["body_ang_vel"][0]),
        }
        far, turn = self._limbs(F, f)
        bearing = m.bearing[f]
        feet = limbs.feet_on_hold_exp(far, bearing, task.FEET_ON_HOLD_STD)
        free = limbs.free_limb_exp(far, bearing, m.left_pos[f], m.land_pos[f], task.FREE_STD_SCALE,
                                   task.FREE_STD_MIN, task.FREE_STD_MAX)
        pose = limbs.limb_pose_exp(far, turn, task.LIMB_POSE_STD, task.LIMB_TURN_STD)
        q = S[4]
        rate = ((a - a_prev) ** 2).sum(-1)
        lim = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        r_track = sum(tr[k][1] * v for k, v in track.items())
        r_climb = (task.FEET_ON_HOLD_WEIGHT * feet + task.FREE_LIMB_WEIGHT * free
                   + task.LIMB_POSE_WEIGHT * pose)
        r = (r_track + r_climb + task.ACTION_RATE_WEIGHT * rate + task.JOINT_LIMIT_WEIGHT * lim) * CONTROL_DT
        if terms:
            return r, {"track": r_track * CONTROL_DT, "climb": r_climb * CONTROL_DT, "limb_far": far.mean(-1),
                       "action_rate": rate, "joint_limit": lim}
        return r

    def critic_features(self, S, f, xi) -> torch.Tensor:
        """The flat critic's, plus each limb's distance and turn from the
        reference and whether the reference bears on it."""
        far, turn = self._limbs(self.features(S), f)
        return torch.cat((super().critic_features(S, f, xi), far, turn, self.motion.bearing[f].float()), -1)
