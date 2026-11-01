"""A Kilter clip's motion.npz (hbold's trajopt) as a MotionLib.

The reference bodies, velocities and joints are the file's own, the ones
hbold's PPO task tracks, not re-derived; joints are reordered to the Unitree
SDK order this repository uses. Beside the tracking reference the file
carries, per frame, each limb's hold point, whether it bears there, the next
limb to land and where, and the points a limb in flight left and lands on.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from ..motion import CONTROL_DT, MotionLib
from ..robot import G1_JOINT_NAMES, TRACKED_BODIES
from ..rotations import matvec3, quat_to_mat


class ClimbMotion(MotionLib):
    """MotionLib's tables from motion.npz files, one clip each, concatenated in
    order (clip i's route is route i); MotionLib's own constructor reads CSV clips."""

    def __init__(self, motion_files: str | list[str], device: str):
        files = [motion_files] if isinstance(motion_files, str) else list(motion_files)
        clips = [_load(f, device) for f in files]
        self.files = files
        self.names = [os.path.basename(os.path.dirname(f)) or "clip" for f in files]
        self.metadata_files = []
        self.device = device
        self._index_clips([len(c["q"]) for c in clips], device)
        t = {k: torch.cat([c[k] for c in clips]) for k in clips[0]}
        self.root_p, self.root_R = t["root_p"], quat_to_mat(t["root_quat"])
        self.root_v, self.root_w = t["root_v"], t["root_w"]
        self.q, self.qd = t["q"], t["qd"]
        self.body_p, self.body_quat = t["body_p"], t["body_quat"]
        self.body_R = quat_to_mat(self.body_quat)
        self.body_v, self.body_w = t["body_v"], t["body_w"]
        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = matvec3(Rt, self.root_v)
        self.root_w_local = matvec3(Rt, self.root_w)
        self.root_gz = self.root_R[:, 2, 2]
        self.hold_pos = t["hold_pos"]  # (T, 4, 3) each limb's site where the reference has it
        self.bearing = t["bearing"]  # (T, 4)
        self.limb = t["limb"]  # (T,) the next limb to land
        self.goal_pos, self.goal_quat = t["goal_pos"], t["goal_quat"]
        self.left_pos, self.land_pos = t["left_pos"], t["land_pos"]


def _load(motion_file: str, device: str) -> dict[str, torch.Tensor]:
    """One clip's tables: the reference's own bodies and velocities (the root is
    body 0), joints in SDK order, the tracked bodies, and the climb's per-frame
    holds, bearing and next landing."""
    with np.load(motion_file) as z:
        if abs(1.0 / float(z["fps"]) - CONTROL_DT) > 1e-9:
            raise ValueError(f"{motion_file}: {z['fps']} fps, the control rate is {1 / CONTROL_DT}")
        joints = [str(n) for n in z["joint_names"]]
        bodies = [str(n) for n in z["body_names"]]
        if set(joints) != set(G1_JOINT_NAMES) or bodies[0] != "pelvis":
            raise ValueError(f"{motion_file}: not the G1's joints, or the pelvis is not body 0")
        order = [joints.index(n) for n in G1_JOINT_NAMES]
        tracked = [bodies.index(n) for n in TRACKED_BODIES]
        z = {k: torch.tensor(np.asarray(z[k]), device=device) for k in z.files if z[k].dtype.kind in "biuf"}
    pos, quat = z["body_pos_w"].float(), z["body_quat_w"].float()
    lin, ang = z["body_lin_vel_w"].float(), z["body_ang_vel_w"].float()
    return {
        "root_p": pos[:, 0], "root_quat": quat[:, 0], "root_v": lin[:, 0], "root_w": ang[:, 0],
        "q": z["joint_pos"][:, order].float(), "qd": z["joint_vel"][:, order].float(),
        "body_p": pos[:, tracked], "body_quat": quat[:, tracked], "body_v": lin[:, tracked], "body_w": ang[:, tracked],
        "hold_pos": z["hold_pos_w"].float(), "bearing": z["bearing"].bool(), "limb": z["limb"].long(),
        "goal_pos": z["goal_pos_w"].float(), "goal_quat": z["goal_quat_w"].float(),
        "left_pos": z["left_pos_w"].float(), "land_pos": z["land_pos_w"].float(),
    }  # fmt: skip
