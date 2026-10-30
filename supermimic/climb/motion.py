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
    """MotionLib's tables from one motion.npz; MotionLib's own constructor reads CSV clips."""

    def __init__(self, motion_file: str, device: str):
        with np.load(motion_file) as z:
            if abs(1.0 / float(z["fps"]) - CONTROL_DT) > 1e-9:
                raise ValueError(f"{motion_file}: {z['fps']} fps, the control rate is {1 / CONTROL_DT}")
            joints = [str(n) for n in z["joint_names"]]
            bodies = [str(n) for n in z["body_names"]]
            if set(joints) != set(G1_JOINT_NAMES) or bodies[0] != "pelvis":
                raise ValueError(f"{motion_file}: not the G1's joints, or the pelvis is not body 0")
            order = [joints.index(n) for n in G1_JOINT_NAMES]
            tracked = [bodies.index(n) for n in TRACKED_BODIES]
            arr = {k: torch.tensor(np.asarray(z[k]), device=device) for k in z.files if z[k].dtype.kind in "biuf"}
        self.files = [motion_file]
        self.names = [os.path.basename(os.path.dirname(motion_file)) or "clip"]
        self.metadata_files = []
        self.device = device
        frames = len(arr["joint_pos"])
        self._index_clips([frames], device)
        pos, quat = arr["body_pos_w"].float(), arr["body_quat_w"].float()
        lin, ang = arr["body_lin_vel_w"].float(), arr["body_ang_vel_w"].float()
        self.root_p, self.root_R = pos[:, 0], quat_to_mat(quat[:, 0])
        self.root_v, self.root_w = lin[:, 0], ang[:, 0]
        self.q, self.qd = arr["joint_pos"][:, order].float(), arr["joint_vel"][:, order].float()
        self.body_p, self.body_quat = pos[:, tracked], quat[:, tracked]
        self.body_R = quat_to_mat(self.body_quat)
        self.body_v, self.body_w = lin[:, tracked], ang[:, tracked]
        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = matvec3(Rt, self.root_v)
        self.root_w_local = matvec3(Rt, self.root_w)
        self.root_gz = self.root_R[:, 2, 2]
        self.hold_pos = arr["hold_pos_w"].float()  # (T, 4, 3) each limb's site where the reference has it
        self.bearing = arr["bearing"].bool()  # (T, 4)
        self.limb = arr["limb"].long()  # (T,) the next limb to land
        self.goal_pos, self.goal_quat = arr["goal_pos_w"].float(), arr["goal_quat_w"].float()
        self.left_pos, self.land_pos = arr["left_pos_w"].float(), arr["land_pos_w"].float()
