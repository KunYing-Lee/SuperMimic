"""The climb's pieces for one clip, from hbold's data directory (HBOLD_DATA):
forward kinematics of the hooked G1 with its limb sites, the clip's motion,
the MDP over its route, and the physics settings of hbold's task."""

from __future__ import annotations

import re

import torch
from hbold.asset_zoo.robots.g1_hook import g1_hook_constants as g1
from hbold.asset_zoo.routes import kilter
from hbold.climb import geometry
from hbold.data import motion_path, route_path
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants

from ..kinematics import G1Kinematics
from ..robot import G1_JOINT_NAMES
from .mdp import LIMB_SITES, ClimbMDP
from .motion import ClimbMotion

G1_HOOK_XML = str(g1.G1_HOOK_XML)


def default_joint_pos() -> list[float]:
    """The hooked G1's initial pose (mjlab's knees-bent keyframe), SDK order."""
    pose = g1_constants.KNEES_BENT_KEYFRAME.joint_pos
    out = []
    for name in G1_JOINT_NAMES:
        hits = [v for pat, v in pose.items() if re.fullmatch(pat, name)]
        out.append(hits[-1] if hits else 0.0)
    return out


def climb_parts(clip_id: str, device: str) -> tuple[G1Kinematics, ClimbMotion, ClimbMDP]:
    """`clip_id`: one clip, or several joined by commas (clip i on route i)."""
    clips = clip_id.split(",")
    kin = G1Kinematics(device, model=g1.get_spec().compile(), sites=LIMB_SITES)
    motion = ClimbMotion([str(motion_path(c)) for c in clips], device)
    routes = [kilter.load(route_path(c)) for c in clips]
    bars = geometry.stack_bars([geometry.bars(r, device, i) for i, r in enumerate(routes)])
    board = geometry.stack_boards([geometry.board(r, device, i) for i, r in enumerate(routes)])
    default_q = torch.tensor(default_joint_pos(), device=device)
    return kin, motion, ClimbMDP(kin, motion, default_q, bars, board)
