"""Differentiable robot forward kinematics from the simulator's compiled MJCF.

The same FK evaluates reference motions, simulator states, and world-model
predictions. Fixed link frames, nonzero joint anchors, and hinge reference
angles use their exact MuJoCo transforms rather than robot-specific offsets.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .robot import RobotSpec, get_robot
from .rotations import axis_angle_to_mat, matmul3, matvec3, quat_to_mat


class RobotKinematics:
    def __init__(
        self,
        device: str,
        robot: str | RobotSpec = "g1",
        tracked: list[str] | tuple[str, ...] | None = None,
    ):
        self.robot = get_robot(robot)
        self.device = device
        m = mujoco.MjModel.from_xml_path(self.robot.mjcf_path)
        joint_names = self.robot.joint_names
        jname_to_idx = {n: i for i, n in enumerate(joint_names)}
        actual = {
            m.joint(j).name
            for j in range(m.njnt)
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
        }
        if actual != set(joint_names):
            raise ValueError(
                f"MJCF joints differ from {self.robot.name} profile: {actual ^ set(joint_names)}"
            )
        root = m.body(self.robot.root_body).id
        root_joint = m.body_jntadr[root]
        if (
            m.body_jntnum[root] != 1
            or m.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE
        ):
            raise ValueError("Robot root must have exactly one free joint")
        if m.body_parentid[root] != 0:
            raise ValueError("Floating root must be attached to world")
        self.body_names, self.parent, self.joint_idx = [], [], []
        axes, offsets, rotations, anchors, references = [], [], [], [], []
        mj_to_local = {}
        for body in range(1, m.nbody):
            if body != root and m.body_parentid[body] not in mj_to_local:
                continue  # Ignore world-attached scenery outside this articulation.
            count = m.body_jntnum[body]
            jid = m.body_jntadr[body]
            if body != root and (
                count > 1
                or (count == 1 and m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE)
            ):
                raise ValueError(
                    f"Unsupported joint structure on {m.body(body).name}: expected fixed or single hinge"
                )
            mj_to_local[body] = len(self.body_names)
            self.body_names.append(m.body(body).name)
            self.parent.append(
                -1 if body == root else mj_to_local[m.body_parentid[body]]
            )
            self.joint_idx.append(
                jname_to_idx[m.joint(jid).name] if count and body != root else -1
            )
            axes.append(m.jnt_axis[jid].copy() if count else np.zeros(3))
            anchors.append(m.jnt_pos[jid].copy() if count else np.zeros(3))
            references.append(
                m.qpos0[m.jnt_qposadr[jid]] if count and body != root else 0.0
            )
            offsets.append(m.body_pos[body].copy())
            rotations.append(m.body_quat[body].copy())

        def tensor(values):
            return torch.tensor(np.array(values), dtype=torch.float32, device=device)

        self.axes = tensor(axes)
        self.off_p = tensor(offsets)
        self.off_R = quat_to_mat(tensor(rotations))
        self.anchors = tensor(anchors)
        self.q_reference = tensor(references)
        tracked = self.robot.tracked_bodies if tracked is None else tracked
        self.tracked_names = tuple(tracked)
        self.tracked_idx = [self.body_names.index(name) for name in tracked]
        self.num_bodies = len(self.body_names)
        self.num_joints = len(joint_names)
        lo, hi = np.zeros(self.num_joints), np.zeros(self.num_joints)
        for name, i in jname_to_idx.items():
            jid = m.joint(name).id
            if not m.jnt_limited[jid]:
                raise ValueError(
                    f"Unbounded robot joint {name}; analytic position-limit penalty requires limits"
                )
            lo[i], hi[i] = m.jnt_range[jid]
        self.q_lo, self.q_hi = tensor(lo), tensor(hi)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Return body world positions, rotations, origin velocities and angular velocities.

        p/R/v/w describe the floating root body in world coordinates. q/qd use
        the robot profile's named joint order. Leading batch dimensions are
        preserved; the returned body dimension follows tracked_names or
        body_names when all_bodies=True.
        """
        P, Rs, V, W = ([None] * self.num_bodies for _ in range(4))
        for i in range(self.num_bodies):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            base = P[par] + matvec3(Rs[par], self.off_p[i])
            R_off = matmul3(Rs[par], self.off_R[i])
            if j < 0:
                pi, Ri, wi = base, R_off, W[par]
                vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
            else:
                Ri = matmul3(
                    R_off,
                    axis_angle_to_mat(self.axes[i], q[..., j] - self.q_reference[i]),
                )
                pivot = base + matvec3(R_off, self.anchors[i])
                pi = pivot - matvec3(Ri, self.anchors[i])
                joint_w = matvec3(R_off, self.axes[i]) * qd[..., j : j + 1]
                wi = W[par] + joint_w
                vi = (
                    V[par]
                    + torch.cross(W[par], pi - P[par], dim=-1)
                    + torch.cross(joint_w, pi - pivot, dim=-1)
                )
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )


class G1Kinematics(RobotKinematics):
    """Compatibility constructor retaining the original G1 call signature."""

    def __init__(self, device: str, tracked: list[str] | tuple[str, ...] | None = None):
        super().__init__(device, "g1", tracked)
