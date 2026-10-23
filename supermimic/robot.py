"""Named robot profiles and native mjlab implicit-PD articulations.

The default Unitree G1 (29 DoF) follows BeyondMimic's actuator model.

The URDF is the one the LAFAN1 motions were retargeted to (g1_29dof_rev_1_0). Joint drives are
implicit PD with a 10 Hz natural frequency and damping ratio 2 on the reflected motor armature, and
the action scale is 0.25 * effort_limit / stiffness, as in BeyondMimic.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
URDF_PATH = os.path.join(DATA_DIR, "g1_description", "g1_29dof_rev_1_0.urdf")
MJCF_PATH = os.path.join(DATA_DIR, "g1_description", "g1_29dof.xml")

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0


def _kp(arm):
    return arm * NATURAL_FREQ**2


def _kd(arm):
    return 2.0 * DAMPING_RATIO * arm * NATURAL_FREQ


# Joint order of the LAFAN1 retarget CSVs (Unitree SDK order).
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip

# BeyondMimic's tracked bodies; the anchor/root is the pelvis.
TRACKED_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]  # fmt: skip

# Per-joint (effort limit, velocity limit, armature) by motor type.
_M5020 = (25.0, 37.0, ARMATURE_5020)
_M4010 = (5.0, 22.0, ARMATURE_4010)
JOINT_MOTORS = {
    "hip_pitch": (88.0, 32.0, ARMATURE_7520_14),
    "hip_roll": (139.0, 20.0, ARMATURE_7520_22),
    "hip_yaw": (88.0, 32.0, ARMATURE_7520_14),
    "knee": (139.0, 20.0, ARMATURE_7520_22),
    "ankle_pitch": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "ankle_roll": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "waist_yaw": (88.0, 32.0, ARMATURE_7520_14),
    "waist_roll": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "waist_pitch": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "shoulder_pitch": _M5020,
    "shoulder_roll": _M5020,
    "shoulder_yaw": _M5020,
    "elbow": _M5020,
    "wrist_roll": _M5020,
    "wrist_pitch": _M4010,
    "wrist_yaw": _M4010,
}


DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}


# K1's 22 physical joints, in Booster SDK/MJCF order. Motion importers must
# reorder by the source joint_names field rather than assuming array order.
K1_JOINT_NAMES = [
    "Head_Yaw", "Head_Pitch",
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch",
    "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch",
    "Right_Ankle_Pitch", "Right_Ankle_Roll",
]  # fmt: skip

K1_TRACKED_BODIES = [
    "Trunk", "Left_Hip_Roll", "Left_Shank", "left_foot_link",
    "Right_Hip_Roll", "Right_Shank", "right_foot_link", "Head_2",
    "Left_Arm_2", "Left_Arm_3", "left_hand_link",
    "Right_Arm_2", "Right_Arm_3", "right_hand_link",
]  # fmt: skip

K1_DEFAULT_JOINT_POS = {
    "Left_Shoulder_Roll": -1.4,
    "Left_Elbow_Yaw": -0.4,
    "Right_Shoulder_Roll": 1.4,
    "Right_Elbow_Yaw": 0.4,
    "Left_Hip_Pitch": -0.4,
    "Left_Knee_Pitch": 0.8,
    "Left_Ankle_Pitch": -0.4,
    "Right_Hip_Pitch": -0.4,
    "Right_Knee_Pitch": 0.8,
    "Right_Ankle_Pitch": -0.4,
}


@dataclass(frozen=True)
class MotorSpec:
    effort_limit: float
    velocity_limit: float
    armature: float
    stiffness: float
    damping: float

    @property
    def action_scale(self) -> float:
        return 0.25 * self.effort_limit / self.stiffness


@dataclass(frozen=True)
class RobotSpec:
    name: str
    joint_names: tuple[str, ...]
    tracked_bodies: tuple[str, ...]
    root_body: str
    torso_body: str
    mjcf_path: str
    urdf_path: str | None
    default_joint_positions: tuple[float, ...]
    initial_height: float
    motors: tuple[MotorSpec, ...]

    def __post_init__(self):
        n = len(self.joint_names)
        if (
            len(set(self.joint_names)) != n
            or n != len(self.motors)
            or n != len(self.default_joint_positions)
        ):
            raise ValueError(
                "Robot joints, defaults, and motors must have equal, unique ordering"
            )
        if not self.tracked_bodies or self.tracked_bodies[0] != self.root_body:
            raise ValueError("The first tracked body must be the floating root")

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def default_joint_pos(self) -> dict[str, float]:
        return dict(zip(self.joint_names, self.default_joint_positions, strict=True))

    def motor(self, joint_name: str) -> MotorSpec:
        return self.motors[self.joint_names.index(joint_name)]


def _defaults(names, patterns):
    return tuple(
        next(
            (
                value
                for pattern, value in reversed(list(patterns.items()))
                if re.fullmatch(pattern, name)
            ),
            0.0,
        )
        for name in names
    )


def _g1_motor(name):
    key = name.replace("left_", "").replace("right_", "").replace("_joint", "")
    effort, velocity, arm = JOINT_MOTORS[key]
    return MotorSpec(effort, velocity, arm, _kp(arm), _kd(arm))


# Source: K1Lab k1_constants.py, commit 30f015a61f9583cba664b5ea65b0a1c73fba57e2.
# Motor limits come from booster_train and gains from booster-lab, as documented
# in that source. The serial ankle model represents two E4310 motors; these are
# simulator actuator parameters, not a parallel-linkage hardware controller.
_K1_MOTORS = {
    "Knee_Pitch": MotorSpec(112.0, 12.57, 0.095625, 80.0, 4.0),
    "Hip_Yaw": MotorSpec(38.3, 17.59, 0.0282528, 80.0, 4.0),
    "Hip_Pitch": MotorSpec(68.0, 14.66, 0.0478125, 80.0, 4.0),
    "Hip_Roll": MotorSpec(76.0, 12.57, 0.0339552, 80.0, 4.0),
    "Ankle_Pitch": MotorSpec(38.3, 17.59, 0.0565056, 50.0, 2.0),
    "Ankle_Roll": MotorSpec(38.3, 17.59, 0.0565056, 50.0, 2.0),
    "arm": MotorSpec(14.0, 33.51, 0.001, 10.0, 1.0),
    "head": MotorSpec(6.0, 7.85, 0.001, 4.0, 0.25),
}


def _k1_motor(name):
    if name.startswith("Head_"):
        return _K1_MOTORS["head"]
    key = name.removeprefix("Left_").removeprefix("Right_")
    return _K1_MOTORS["arm" if key.startswith(("Shoulder_", "Elbow_")) else key]


G1 = RobotSpec(
    name="g1",
    joint_names=tuple(G1_JOINT_NAMES),
    tracked_bodies=tuple(TRACKED_BODIES),
    root_body="pelvis",
    torso_body="torso_link",
    mjcf_path=MJCF_PATH,
    urdf_path=URDF_PATH,
    default_joint_positions=_defaults(G1_JOINT_NAMES, DEFAULT_JOINT_POS),
    initial_height=0.76,
    motors=tuple(_g1_motor(name) for name in G1_JOINT_NAMES),
)
K1 = RobotSpec(
    name="k1",
    joint_names=tuple(K1_JOINT_NAMES),
    tracked_bodies=tuple(K1_TRACKED_BODIES),
    root_body="Trunk",
    torso_body="Trunk",
    mjcf_path=os.path.join(DATA_DIR, "k1_description", "k1.xml"),
    urdf_path=None,
    default_joint_positions=_defaults(K1_JOINT_NAMES, K1_DEFAULT_JOINT_POS),
    initial_height=0.5125,
    motors=tuple(_k1_motor(name) for name in K1_JOINT_NAMES),
)


def get_robot(robot: str | RobotSpec = "g1") -> RobotSpec:
    if isinstance(robot, RobotSpec):
        return robot
    try:
        return {"g1": G1, "k1": K1}[robot]
    except KeyError:
        raise ValueError(f"Unknown robot {robot!r}; choose g1 or k1") from None


def motor_of(
    joint_name: str, robot: str | RobotSpec = "g1"
) -> tuple[float, float, float]:
    motor = get_robot(robot).motor(joint_name)
    return motor.effort_limit, motor.velocity_limit, motor.armature


def action_scale(joint_name: str, robot: str | RobotSpec = "g1") -> float:
    return get_robot(robot).motor(joint_name).action_scale


def default_joint_pos_list(robot: str | RobotSpec = "g1") -> list[float]:
    return list(get_robot(robot).default_joint_positions)


def make_robot_cfg(robot: str | RobotSpec = "g1"):
    """Build a native MJCF articulation with implicit MuJoCo PD drives.

    Velocity limits remain motor metadata; no velocity clipping is imposed.
    All collision masks and exclusions are retained from the source model.
    """
    import mujoco
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    profile = get_robot(robot)
    actuators = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(name,),
            effort_limit=motor.effort_limit,
            stiffness=motor.stiffness,
            damping=motor.damping,
            armature=motor.armature,
        )
        for name, motor in zip(profile.joint_names, profile.motors, strict=True)
    )
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(profile.mjcf_path),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, profile.initial_height),
            joint_pos=profile.default_joint_pos,
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=actuators,
            soft_joint_pos_limit_factor=0.9,
        ),
    )


def make_g1_cfg():
    """Backward-compatible entry point for existing G1 experiments."""
    return make_robot_cfg("g1")
