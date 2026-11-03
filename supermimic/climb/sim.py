"""The climb's simulator: SuperMimic's batched loop on hbold's scene.

The scene and physics settings are taken whole from hbold's motion-tracking
task config (the hooked G1 with its stiffer gains, the clip's route, the
plane terrain, the solver budget), so this simulator and hbold's PPO env step
the same model. With several clips (ids joined by commas) every world holds
every clip's route, route k at y = k * ROUTE_TILE, and an episode on clip k
plays on route k: states are relative to its route's origin (env spacing 0),
so no learned input sees which tile it is on. Each limb gets a contact sensor
on each route; whether it presses there is the world model's auxiliary
contact label.
"""

from __future__ import annotations

import dataclasses
import re

import torch
from hbold.asset_zoo.robots.g1_hook import g1_hook_constants as g1
from hbold.asset_zoo.routes import kilter
from hbold.data import LIMBS, route_path
from hbold.tasks.goal.goal_env_cfg import contact_sensors
from hbold.tasks.goal.mdp.commands import FOOT_N, HOOK_N
from hbold.tasks.motion_tracking.mdp.commands import start_weights
from hbold.tasks.motion_tracking.config.g1.env_cfgs import unitree_g1_motion_tracking_env_cfg
from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.scene import Scene
from mjlab.sim import Simulation, SimulationCfg

from ..env import TrackingSim
from ..robot import G1_JOINT_NAMES, motor_of
from .mdp import LIMB_BODIES

PRESS_N = (HOOK_N, HOOK_N, FOOT_N, FOOT_N)  # N a hook, a foot, must press on the route to count as on it
JOINT_OFFSET = 0.01  # rad, the encoder offsets' randomized half-range (TrackingSim._randomize_context)
CONTACT_MAXMATCH = 128
ROUTE_TILE = 10.0  # m between routes along y: far past a board's width and a robot's reach
FAIL_ALPHA, FAIL_LAMBDA, FAIL_KERNEL, FAIL_UNIFORM = 0.001, 0.8, 3, 0.5
"""hbold's PPO start sampling (mjlab's adaptive sampler with hbold's start_weights):
the per-step decay of the failure and visit averages, the kernel spreading a
bin's failures over the bins before it, and the flat share of starts."""  # route sensors' contact matches per world; the hooks' convex parts make many


def climb_physics(clip_id: str) -> SimulationCfg:
    """hbold's physics settings (the first clip's: they are not per route),
    with room for the route sensors."""
    physics = unitree_g1_motion_tracking_env_cfg(clip_id.split(",")[0]).sim
    physics.contact_sensor_maxmatch = CONTACT_MAXMATCH
    return physics


def hook_action_scale(device: str) -> torch.Tensor:
    """hbold's per-joint action scale (SDK order) for its stiffer gains."""
    scale = []
    for name in G1_JOINT_NAMES:
        hits = [v for pat, v in g1.ACTION_SCALE.items() if re.fullmatch(pat, name)]
        if len(hits) != 1:
            raise ValueError(f"{name}: {len(hits)} action scales match")
        scale.append(float(hits[0]))
    return torch.tensor(scale, dtype=torch.float32, device=device)


def pd_action_bounds(m, default_q: torch.Tensor, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per joint (SDK order), the actions whose PD target lets the motor reach
    either torque limit anywhere in its joint range at up to its rated speed:
    tau = kp (target - q) - kd qdot, target = default + offset + alpha a, the
    encoder offset within its randomized range. A PD target past this cannot
    change the torque, only the model's view of it (Codex's actions ran to 1e4).
    Rated speeds are motor ratings, not a velocity clamp in the simulation.
    ``m``: the compiled scene's MjModel (robot joints prefixed ``robot/``)."""
    lo, hi = [], []
    for k, name in enumerate(G1_JOINT_NAMES):
        jid = m.joint(f"robot/{name}").id
        aid = [i for i in range(m.nu) if m.actuator_trntype[i] == 0 and m.actuator_trnid[i, 0] == jid]
        if len(aid) != 1 or not m.actuator_forcelimited[aid[0]]:
            raise ValueError(f"{name}: expected one force-limited position actuator")
        kp, kd = float(m.actuator_gainprm[aid[0], 0]), -float(m.actuator_biasprm[aid[0], 2])
        margin = (float(m.actuator_forcerange[aid[0], 1]) + kd * motor_of(name)[1]) / kp + JOINT_OFFSET
        q_lo, q_hi = m.jnt_range[jid]
        lo.append((q_lo - margin - float(default_q[k])) / float(alpha[k]))
        hi.append((q_hi + margin - float(default_q[k])) / float(alpha[k]))
    dev = default_q.device
    return (torch.tensor(lo, dtype=torch.float32, device=dev), torch.tensor(hi, dtype=torch.float32, device=dev))


class ClimbSim(TrackingSim):
    _EPISODE_FIELDS = (*TrackingSim._EPISODE_FIELDS, "bin_fail", "bin_visit")
    _OPTIONAL_EPISODE_FIELDS = ("bin_fail", "bin_visit")  # the failure-rate starts': older runs have none

    def __init__(self, sim_cfg, num_envs: int, motion, mdp, clip_id: str, start_sampler: str = "return",
                 clip_start_share: float = 0.0, **kwargs):
        """start_sampler: "return", SuperMimic's (bins where returns are low), or
        "failure_rate", hbold's PPO task's (bins where episodes fail most often
        for the steps spent there). clip_start_share: the share of sampled
        starts put instead at a clip's first frame (a clip drawn uniformly), as
        the climb is evaluated."""
        if start_sampler not in ("return", "failure_rate"):
            raise ValueError(f"unknown start sampler {start_sampler!r}")
        if not 0.0 <= clip_start_share <= 1.0:
            raise ValueError("clip_start_share is a share")
        self.clip_id, self.start_sampler, self.clip_start_share = clip_id, start_sampler, clip_start_share
        self.clips = clip_id.split(",")
        if len(self.clips) != len(motion.clip_len):
            raise ValueError(f"{len(self.clips)} routes for {len(motion.clip_len)} motion clips")
        self.route_origin = torch.zeros(len(self.clips), 3, device=sim_cfg.device)
        self.route_origin[:, 1] = ROUTE_TILE * torch.arange(len(self.clips), device=sim_cfg.device)
        nb, dev = motion.num_bins, sim_cfg.device
        self.bin_fail = torch.zeros(nb, device=dev)
        self.bin_visit = torch.zeros(nb, device=dev)
        kernel = torch.tensor([FAIL_LAMBDA**i for i in range(FAIL_KERNEL)], device=dev)
        self.fail_kernel = kernel / kernel.sum()
        super().__init__(sim_cfg, num_envs, motion, mdp, **kwargs)
        self.route_sensors = [[self.scene[f"{limb}_{e}"] for e in self._routes()] for limb in LIMBS]
        self.press_n = torch.tensor(PRESS_N, device=self.device)

    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        frames, bins = self._sample_frames(n)
        if self.clip_start_share > 0:
            m = self.motion
            first = torch.rand(n, device=self.device) < self.clip_start_share
            clip = torch.randint(len(m.clip_len), (n,), device=self.device)
            frames = torch.where(first, m.clip_start[clip], frames)
            bins = torch.where(first, m.frame_bin[frames], bins)
        return frames, bins

    def _sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.start_sampler == "return":
            return super().sample_frames(n)
        m = self.motion
        p = start_weights(self.bin_fail, self.bin_visit, FAIL_UNIFORM, self.fail_kernel)
        bins = torch.multinomial(p, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + 50, m.clip_end[m.bin_clip[bins]])
        return start + (torch.rand(n, device=self.device) * (end - start).float()).long(), bins

    def step(self, a: torch.Tensor) -> dict:
        out = super().step(a)
        if self.start_sampler == "failure_rate":  # this step's visits and failures, by bin, into the averages
            nb = self.motion.num_bins
            bins = self.motion.frame_bin[out["f"]]
            here = torch.bincount(bins, minlength=nb).float()
            failed = torch.bincount(bins[out["term"]], minlength=nb).float()
            self.bin_visit = FAIL_ALPHA * here + (1 - FAIL_ALPHA) * self.bin_visit
            self.bin_fail = FAIL_ALPHA * failed + (1 - FAIL_ALPHA) * self.bin_fail
        return out

    def _routes(self) -> list[str]:
        """The route entities' names, clip k's route k: hbold's "route" first."""
        return ["route", *(f"route{k}" for k in range(1, len(self.clips)))]

    def _origin(self, env_ids, frames: torch.Tensor) -> torch.Tensor:
        return self.scene.env_origins[env_ids] + self.route_origin[self.motion.clip_id[frames]]

    def _build_scene(self, num_envs: int):
        scene_cfg = unitree_g1_motion_tracking_env_cfg(self.clips[0]).scene
        scene_cfg.num_envs, scene_cfg.env_spacing = num_envs, 0.0
        sensors = []
        for k, name in enumerate(self._routes()):
            if k:
                route = kilter.get_route_cfg(kilter.load(route_path(self.clips[k])))
                route.init_state = EntityCfg.InitialStateCfg(pos=tuple(self.route_origin[k].tolist()))
                scene_cfg.entities[name] = route
            for c in contact_sensors(LIMB_BODIES)[: len(LIMB_BODIES)]:
                second = dataclasses.replace(c.secondary, entity=name)
                sensors.append(dataclasses.replace(c, name=f"{c.name}{k or ''}", secondary=second))
        scene_cfg.sensors = (*scene_cfg.sensors, *sensors)
        scene = Scene(scene_cfg, device=self.device)
        sim = Simulation(num_envs=num_envs, cfg=self.cfg.physics, model=scene.compile(), device=self.device)
        scene.initialize(sim.mj_model, sim.model, sim.data)
        sim.reset()
        return scene, sim

    def _action_scale(self) -> torch.Tensor:
        return hook_action_scale(self.device)

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
        return pd_action_bounds(self.sim.mj_model, self.default_q, self.alpha)

    def _labels(self) -> dict:
        """(N, 4) each limb presses on the routes harder than its newtons."""
        newtons = []
        for sensors in self.route_sensors:
            newtons.append(sum((s.data.force.norm(dim=-1) * (s.data.found > 0)).sum(-1) for s in sensors))
        return {"contact": torch.stack(newtons, -1) > self.press_n}
