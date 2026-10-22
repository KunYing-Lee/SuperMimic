"""Batched SuperMimic tracking on native mjlab 1.6 / MuJoCo-Warp.

Policy/motion joints use Unitree SDK order; the entity's tree ordering is mapped
explicitly on every joint read/write. Quaternions are wxyz throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import torch
import warp as wp
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

from .mdp import MAX_STEP_REWARD, MDP, flatten_state
from .motion import CONTROL_DT, MotionLib
from .robot import G1_JOINT_NAMES, action_scale, make_g1_cfg
from .rotations import mat_to_quat, matmul3, matvec3, quat_to_mat, so3_exp


@dataclass
class TrackingSimCfg:
    device: str = "cuda:0"
    decimation: int = 4
    torch_stream: torch.cuda.Stream | None = field(default=None, repr=False, compare=False)
    warp_stream: wp.Stream | None = field(default=None, repr=False, compare=False)
    physics: SimulationCfg = field(default_factory=lambda: SimulationCfg(
        njmax=500, nconmax=128,
        mujoco=MujocoCfg(timestep=CONTROL_DT / 4, integrator="implicitfast", cone="pyramidal"),
    ))


def make_sim(device: str, args=None) -> TrackingSimCfg:
    """Set up shared execution before allocating any CUDA model or motion tensors.

    CUDA graph capture cannot use the legacy null stream. The retained non-default
    stream is both Torch's and Warp's current stream, so physics, model operations,
    and tensor allocation/free all use one ordered execution queue.
    """
    cfg = TrackingSimCfg(device=device)
    if torch.device(device).type == "cuda":
        wp.init()
        torch.cuda.set_device(device)
        previous = torch.cuda.current_stream(device)
        cfg.torch_stream = torch.cuda.Stream(device=device)
        cfg.torch_stream.wait_stream(previous)
        torch.cuda.set_stream(cfg.torch_stream)
        cfg.warp_stream = wp.stream_from_torch(cfg.torch_stream)
        wp.set_stream(cfg.warp_stream)
    return cfg


def _ground(spec: mujoco.MjSpec) -> None:
    spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, 0.1], friction=[1.0, 0.005, 0.0001],
                            rgba=[0.25, 0.28, 0.32, 1.0])


def build_scene(cfg: TrackingSimCfg, num_envs: int):
    # make_sim must run before the caller's CUDA allocations; switching streams
    # here would leave those allocations with unrecorded cross-stream lifetimes.
    if torch.device(cfg.device).type == "cuda":
        if cfg.torch_stream is None or cfg.warp_stream is None:
            raise RuntimeError("Create CUDA configuration with make_sim before allocating learner tensors")
        if torch.cuda.current_stream(cfg.device).cuda_stream != cfg.torch_stream.cuda_stream:
            raise RuntimeError("Torch current stream changed after shared simulator execution setup")
        if wp.get_stream(cfg.device).cuda_stream != cfg.warp_stream.cuda_stream:
            raise RuntimeError("Warp current stream changed after shared simulator execution setup")
    if abs(cfg.physics.mujoco.timestep * cfg.decimation - CONTROL_DT) > 1e-12:
        raise ValueError("Physics decimation must equal the 50 Hz motion/control period")
    scene = Scene(SceneCfg(num_envs=num_envs, env_spacing=0.0,
                           entities={"robot": make_g1_cfg()}, spec_fn=_ground), device=cfg.device)
    sim = Simulation(num_envs=num_envs, cfg=cfg.physics, model=scene.compile(), device=cfg.device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    sim.reset()
    return scene, sim


class TrackingSim:
    """Batched G1 motion-tracking simulator.

    Uses native mjlab entities and four MuJoCo-Warp substeps per control step.
    """

    def __init__(self, sim_cfg, num_envs: int, motion: MotionLib, mdp: MDP, gamma: float = 0.99, eval_mode: bool = False,
                 max_episode_steps: int | None = None, reset_noise: bool = False):
        if not 0 < gamma < 1 or (max_episode_steps is not None and max_episode_steps < 1):
            raise ValueError("Require 0 < gamma < 1 and a positive optional episode horizon")
        self.device = sim_cfg.device
        self.cfg = sim_cfg
        self.scene, self.sim = build_scene(sim_cfg, num_envs)
        self.robot = self.scene["robot"]
        names = list(self.robot.joint_names)
        if len(names) != 29 or set(names) != set(G1_JOINT_NAMES):
            raise ValueError(f"Unexpected G1 model joints: {names}")
        self.joint_ids = torch.tensor([names.index(n) for n in G1_JOINT_NAMES], device=self.device)
        self.N = num_envs
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.eval_mode = eval_mode
        self.max_episode_steps = max_episode_steps if not eval_mode else None
        self.reset_noise = reset_noise and not eval_mode
        dev = self.device
        self.default_q = self.robot.data.default_joint_pos[0, self.joint_ids].clone()
        self.alpha = torch.tensor([action_scale(n) for n in G1_JOINT_NAMES], device=dev)
        self.torso_id = int(self.robot.indexing.body_ids[self.robot.body_names.index("torso_link")])
        self.xi = torch.zeros(self.N, 33, device=dev)
        # Domain parameters and all dependent constants must have a world axis.
        # These derived fields are mjlab 1.6.0's set_const dependency closure.
        self._dr_fields = ("geom_friction", "body_ipos", "body_subtreemass", "dof_invweight0",
                           "body_invweight0", "tendon_length0", "tendon_invweight0", "actuator_acc0")
        self.sim.expand_model_fields(self._dr_fields)
        self._default_com = self.sim.get_default_field("body_ipos")[self.torso_id].clone()
        # Empty MuJoCo state components carry no information (e.g. no mocap
        # bodies or actuator activations in this G1). Do not ask Torch to wrap
        # zero-sized vector-valued Warp arrays.
        self._integration_fields = tuple(name for name in self._INTEGRATION_FIELDS
                                         if getattr(self.sim.wp_data, name).size > 0)

        # ---- episode bookkeeping ----
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, 29, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.full((motion.num_bins,), 0.0, device=dev)
        self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=dev)
        self.bin_count = torch.zeros(motion.num_bins, dtype=torch.long, device=dev)
        self.sampling_updates = torch.zeros((), dtype=torch.long, device=dev)
        self.ep_horizon = torch.ones(self.N, dtype=torch.long, device=dev)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.reset(torch.arange(self.N, device=dev))

    # ------------------------------------------------------------------ domain / state
    def _randomize_context(self, env_ids):
        n = len(env_ids)
        if self.eval_mode:
            self.xi[env_ids] = 0.0
            self.xi[env_ids, 0] = 1.0
        else:
            self.xi[env_ids, 0] = torch.empty(n, device=self.device).uniform_(0.3, 1.2)
            self.xi[env_ids, 1:30] = torch.empty(n, 29, device=self.device).uniform_(-0.01, 0.01)
            widths = torch.tensor([0.025, 0.05, 0.05], device=self.device)
            self.xi[env_ids, 30:33] = (2 * torch.rand(n, 3, device=self.device) - 1) * widths
        self._apply_context(env_ids)

    def _apply_context(self, env_ids):
        # Both surfaces get the same sliding coefficient, so MuJoCo's contact
        # max rule yields exactly xi. Preserve source torsional/rolling friction.
        self.sim.model.geom_friction[env_ids, :, 0] = self.xi[env_ids, 0:1]
        self.sim.model.body_ipos[env_ids, self.torso_id] = self._default_com + self.xi[env_ids, 30:33]
        self.sim.recompute_constants(RecomputeLevel.set_const)

    def state(self):
        # mj_step integrates qpos/qvel after its forward pass. Reading xpos/cvel
        # here would return pre-integration kinematics. Generalized coordinates
        # are authoritative and already current without an extra forward solve.
        idx = self.robot.indexing
        root_q = self.sim.data.qpos[:, idx.free_joint_q_adr].clone()
        root_v = self.sim.data.qvel[:, idx.free_joint_v_adr].clone()
        p = root_q[:, :3] - self.scene.env_origins
        R = quat_to_mat(root_q[:, 3:7])  # mjlab/MuJoCo use wxyz.
        w = matvec3(R, root_v[:, 3:6])
        d = self.robot.data
        return (p, R, root_v[:, :3], w, d.joint_pos[:, self.joint_ids].clone(),
                d.joint_vel[:, self.joint_ids].clone())

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        self.sampling_updates += 1
        self.bin_w_avg += (soft - self.bin_w_avg) / self.sampling_updates
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + 50, m.clip_end[m.bin_clip[bins]])  # keep >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        m = self.motion
        remaining = m.clip_end[m.clip_id[frames]] - frames
        if torch.any(remaining < 1):
            raise ValueError("Reset frame must precede the final frame of its motion clip")
        p, R, v, w, q, qd = (x.clone() for x in m.ref_state(frames))
        if self.reset_noise:
            p = p + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.05, 0.05), torch.empty(n, device=dev).uniform_(-0.05, 0.05),
                 torch.empty(n, device=dev).uniform_(-0.01, 0.01)), -1)  # fmt: skip
            rv = torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.1, 0.1), torch.empty(n, device=dev).uniform_(-0.1, 0.1),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            R = matmul3(so3_exp(rv), R)
            v = v + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.5, 0.5), torch.empty(n, device=dev).uniform_(-0.5, 0.5),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            w = w + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.52, 0.52), torch.empty(n, device=dev).uniform_(-0.52, 0.52),
                 torch.empty(n, device=dev).uniform_(-0.78, 0.78)), -1)  # fmt: skip
            q = q + torch.empty_like(q).uniform_(-0.1, 0.1)
        quat = mat_to_quat(R)
        self.sim.reset(env_ids)
        self.scene.reset(env_ids)
        self._randomize_context(env_ids)
        pose = torch.cat((p + self.scene.env_origins[env_ids], quat), -1)
        self.robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q, qd, joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.set_joint_position_target(q, joint_ids=self.joint_ids, env_ids=env_ids)
        self.f[env_ids] = frames
        self.ep_horizon[env_ids] = (remaining if self.max_episode_steps is None
                                    else remaining.clamp(max=self.max_episode_steps))
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0

    def refresh(self):
        """Propagate written states so that ``state()`` is consistent after a reset."""
        self.sim.forward()
        self.scene.update(0.0)

    # ------------------------------------------------------------------ step
    def step(self, a: torch.Tensor) -> dict:
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        target = self.default_q + xi[:, 1:30] + self.alpha * a
        self.robot.set_joint_position_target(target, joint_ids=self.joint_ids)
        for _ in range(self.cfg.decimation):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.cfg.physics.mujoco.timestep)
        S1 = self.state()
        f1 = f + 1
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True)
        term = self.mdp.termination(S1, f1)
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        timeout = f1 >= clip_end
        if self.max_episode_steps is not None:
            timeout = timeout | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {
            "s": flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f, "s1": flatten_state(S1), "r": r,
            "term": term, "terminated": term, "done": done, "ep_step": ep_step, "xi": xi, "terms": terms,
        }  # fmt: skip
        # bookkeeping
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            # Finite-horizon interpretation of Appendix D.1: normalize by the
            # largest return attainable from the sampled start *before* any
            # failure. This avoids clip-length bias and invents no future reward.
            # The paper does not prescribe the estimator or EMA coefficient;
            # cumulative sample means give every observed episode equal weight.
            T = self.ep_step[ids].float()
            attainable = self.G_max * (1 - self.gamma ** self.ep_horizon[ids].float())
            G = self.G_max * self.ep_disc_ret[ids] / attainable
            b = self.start_bin[ids]
            unique, inverse = b.unique(return_inverse=True)
            count = torch.bincount(inverse, minlength=len(unique))
            total = torch.zeros(len(unique), device=self.device).scatter_add_(0, inverse, G)
            previous = self.bin_count[unique]
            self.bin_G[unique] = (self.bin_G[unique] * previous + total) / (previous + count)
            self.bin_count[unique] += count
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (self.ep_horizon[ids] * MAX_STEP_REWARD)
            self.reset(ids)
        return out

    # MuJoCo's integration state; native position actuators have no private
    # recurrent/delay state. RNG state belongs to the trainer checkpoint.
    _INTEGRATION_FIELDS = (
        "time", "qpos", "qvel", "act", "history", "tree_asleep", "qacc_warmstart", "ctrl",
        "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos", "mocap_quat", "userdata",
    )
    _EPISODE_FIELDS = (
        "xi", "f", "ep_step", "a_prev", "start_bin", "ep_disc_ret", "ep_ret",
        "bin_G", "bin_w_avg", "bin_count", "sampling_updates", "ep_horizon",
        "done_eval", "failed_eval",
    )
    _TARGET_FIELDS = ("joint_pos_target", "joint_vel_target", "joint_effort_target")

    def state_dict(self) -> dict:
        """Snapshot simulator integration, randomized model, and sampler state."""
        return {
            "version": 1,
            "num_envs": self.N,
            "joint_names": list(G1_JOINT_NAMES),
            "gamma": self.gamma,
            "max_episode_steps": self.max_episode_steps,
            "reset_noise": self.reset_noise,
            "eval_mode": self.eval_mode,
            "simulation": {name: getattr(self.sim.data, name).clone() for name in self._integration_fields},
            "episodes": {name: getattr(self, name).clone() for name in self._EPISODE_FIELDS},
            "targets": {name: getattr(self.robot.data, name).clone() for name in self._TARGET_FIELDS},
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore a matching environment; fail on incompatible checkpoint state."""
        expected = {"version": 1, "num_envs": self.N, "joint_names": list(G1_JOINT_NAMES),
                    "gamma": self.gamma, "max_episode_steps": self.max_episode_steps,
                    "reset_noise": self.reset_noise, "eval_mode": self.eval_mode}
        for name, value in expected.items():
            if state[name] != value:
                raise ValueError(f"Environment checkpoint {name} mismatch: {state[name]!r} != {value!r}")
        for name in self._EPISODE_FIELDS:
            getattr(self, name).copy_(state["episodes"][name])
        self._apply_context(torch.arange(self.N, device=self.device))
        for name in self._integration_fields:
            getattr(self.sim.data, name).copy_(state["simulation"][name])
        for name in self._TARGET_FIELDS:
            getattr(self.robot.data, name).copy_(state["targets"][name])
        self.refresh()
        # forward recomputes solver quantities; retain the saved warmstart for
        # the next step, exactly as restoring mjSTATE_INTEGRATION requires.
        self.sim.data.qacc_warmstart.copy_(state["simulation"]["qacc_warmstart"])
