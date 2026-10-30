"""The climb's simulator: SuperMimic's batched loop on hbold's scene.

The scene and physics settings are taken whole from hbold's motion-tracking
task config (the hooked G1 with its stiffer gains, the clip's route, the
plane terrain, the solver budget), so this simulator and hbold's PPO env step
the same model. Every world's route sits at the origin (env spacing 0), so
states are route-relative. Each limb also gets a contact sensor on the route;
whether it presses there is the world model's auxiliary contact label.
"""

from __future__ import annotations

import re

import torch
from hbold.asset_zoo.robots.g1_hook import g1_hook_constants as g1
from hbold.data import LIMBS
from hbold.tasks.goal.goal_env_cfg import contact_sensors
from hbold.tasks.goal.mdp.commands import FOOT_N, HOOK_N
from hbold.tasks.motion_tracking.config.g1.env_cfgs import unitree_g1_motion_tracking_env_cfg
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.scene import Scene
from mjlab.sim import Simulation, SimulationCfg

from ..env import TrackingSim
from ..robot import G1_JOINT_NAMES, motor_of
from .mdp import LIMB_BODIES

PRESS_N = (HOOK_N, HOOK_N, FOOT_N, FOOT_N)  # N a hook, a foot, must press on the route to count as on it
JOINT_OFFSET = 0.01  # rad, the encoder offsets' randomized half-range (TrackingSim._randomize_context)
CONTACT_MAXMATCH = 128  # route sensors' contact matches per world; the hooks' convex parts make many


def climb_physics(clip_id: str) -> SimulationCfg:
    """hbold's physics settings for the clip, with room for the route sensors."""
    physics = unitree_g1_motion_tracking_env_cfg(clip_id).sim
    physics.contact_sensor_maxmatch = CONTACT_MAXMATCH
    return physics


class ClimbSim(TrackingSim):
    def __init__(self, sim_cfg, num_envs: int, motion, mdp, clip_id: str, **kwargs):
        self.clip_id = clip_id
        super().__init__(sim_cfg, num_envs, motion, mdp, **kwargs)
        self.route_sensors = [self.scene[f"{limb}_route"] for limb in LIMBS]
        self.press_n = torch.tensor(PRESS_N, device=self.device)

    def _build_scene(self, num_envs: int):
        scene_cfg = unitree_g1_motion_tracking_env_cfg(self.clip_id).scene
        scene_cfg.num_envs, scene_cfg.env_spacing = num_envs, 0.0
        scene_cfg.sensors = (*scene_cfg.sensors, *contact_sensors(LIMB_BODIES)[: len(LIMB_BODIES)])
        scene = Scene(scene_cfg, device=self.device)
        sim = Simulation(num_envs=num_envs, cfg=self.cfg.physics, model=scene.compile(), device=self.device)
        scene.initialize(sim.mj_model, sim.model, sim.data)
        sim.reset()
        return scene, sim

    def _action_scale(self) -> torch.Tensor:
        """hbold's per-joint scale for its stiffer gains (patterns over joint names)."""
        scale = []
        for name in G1_JOINT_NAMES:
            hits = [v for pat, v in g1.ACTION_SCALE.items() if re.fullmatch(pat, name)]
            if len(hits) != 1:
                raise ValueError(f"{name}: {len(hits)} action scales match")
            scale.append(float(hits[0]))
        return torch.tensor(scale, dtype=torch.float32, device=self.device)

    def _apply_context(self, env_ids):
        """Joint offsets and the torso's centre of mass; friction stays the
        model's own: the bars grip, the board is frictionless, and a single
        coefficient written to every geom would undo both."""
        self.sim.model.body_ipos[env_ids, self.torso_id] = self._default_com + self.xi[env_ids, 30:33]
        self.sim.recompute_constants(RecomputeLevel.set_const)

    def _randomize_context(self, env_ids):
        super()._randomize_context(env_ids)
        self.xi[env_ids, 0] = 1.0  # the friction slot is unused here; held at its eval value

    def action_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per joint, the actions whose PD target lets the motor reach either torque
        limit anywhere in its joint range at up to its rated speed:
        tau = kp (target - q) - kd qdot, target = default + offset + alpha a, the
        encoder offset within its randomized range. A PD target past this cannot
        change the torque, only the model's view of it (Codex's actions ran to 1e4).
        Rated speeds are motor ratings, not a velocity clamp in the simulation."""
        m = self.sim.mj_model
        lo, hi = [], []
        for k, name in enumerate(G1_JOINT_NAMES):
            jid = m.joint(f"robot/{name}").id
            aid = [i for i in range(m.nu) if m.actuator_trntype[i] == 0 and m.actuator_trnid[i, 0] == jid]
            if len(aid) != 1 or not m.actuator_forcelimited[aid[0]]:
                raise ValueError(f"{name}: expected one force-limited position actuator")
            kp, kd = float(m.actuator_gainprm[aid[0], 0]), -float(m.actuator_biasprm[aid[0], 2])
            margin = (float(m.actuator_forcerange[aid[0], 1]) + kd * motor_of(name)[1]) / kp + JOINT_OFFSET
            q_lo, q_hi = m.jnt_range[jid]
            lo.append((q_lo - margin - float(self.default_q[k])) / float(self.alpha[k]))
            hi.append((q_hi + margin - float(self.default_q[k])) / float(self.alpha[k]))
        return (torch.tensor(lo, dtype=torch.float32, device=self.device),
                torch.tensor(hi, dtype=torch.float32, device=self.device))

    def _labels(self) -> dict:
        """(N, 4) each limb presses on the route harder than its newtons."""
        newtons = []
        for sensor in self.route_sensors:
            d = sensor.data
            newtons.append((d.force.norm(dim=-1) * (d.found > 0)).sum(-1))
        return {"contact": torch.stack(newtons, -1) > self.press_n}
