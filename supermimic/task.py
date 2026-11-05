"""Task setup around the simulator, MDP and replay hooks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import torch

from .buffer import ReplayBuffer
from .kinematics import G1Kinematics
from .mdp import MDP
from .motion import MotionLib
from .robot import DATA_DIR, MJCF_PATH, default_joint_pos_list

if TYPE_CHECKING:
    from .env import TrackingSim, TrackingSimCfg


class Task(Protocol):
    name: str
    robot_path: str | Path
    sources: Sequence[Path]
    assets: Sequence[Path]
    replay_extra: dict | None
    fine_tune_args: frozenset[str]

    def add_arguments(self, parser: argparse.ArgumentParser) -> None: ...

    def setup(self, args: argparse.Namespace, sim_cfg: TrackingSimCfg) -> MDP:
        """Build the MDP; configure physics and task metadata before learner construction."""
        ...

    def make_env(self, args: argparse.Namespace, sim_cfg: TrackingSimCfg, mdp: MDP) -> TrackingSim: ...

    def sample_starts(self, replay: ReplayBuffer, batch: int, hist: int,
                      clip_end: torch.Tensor, horizon: int) -> tuple[dict, dict]: ...


class LafanTask:
    """Flat-ground LAFAN1 tracking with uniform valid replay starts."""

    name = "lafan"
    robot_path = MJCF_PATH
    sources = ()
    assets = ()
    replay_extra = None
    fine_tune_args = frozenset()

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--motion_dir", default=str(Path(DATA_DIR) / "lafan1_g1"))

    def setup(self, args: argparse.Namespace, sim_cfg: TrackingSimCfg) -> MDP:
        kin = G1Kinematics(args.device)
        motion = MotionLib.from_dir(args.motion_dir, kin, args.device)
        return MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=args.device))

    def make_env(self, args: argparse.Namespace, sim_cfg: TrackingSimCfg, mdp: MDP) -> TrackingSim:
        from .env import TrackingSim

        return TrackingSim(sim_cfg, args.num_envs, mdp.motion, mdp, gamma=args.gamma,
                           max_episode_steps=args.max_episode_steps, reset_noise=args.reset_noise)

    def sample_starts(self, replay: ReplayBuffer, batch: int, hist: int,
                      clip_end: torch.Tensor, horizon: int) -> tuple[dict, dict]:
        return replay.sample_starts(batch, hist, clip_end, horizon)
