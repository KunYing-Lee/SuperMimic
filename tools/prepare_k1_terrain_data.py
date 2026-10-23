"""Import the pinned X-WBC K1 LAFAN archive and construct feasible terrain pairs.

No reference pose is lifted, smoothed, clipped, or retargeted. Terrain obstacles
are solutions of sampled swept-volume clearance constraints, then checked using
native MuJoCo contacts. Flat controls are retained for every selected motion.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np

DATASET = "TonyZhang1999/xwbc-retarget-data"
REVISION = "b4e169f460bd68b27e04e75c99be6e0affbfe046"
ARCHIVE_PATH = "lafan/booster_k1.tar.gz"
ARCHIVE_SHA256 = "92fafe9e6372bf08b26061cc7ed4356f9f5c4bbffce646a86a0574a331112e42"
# NPZ omits names. This contract is independently checked against all source body
# transforms below, not inferred from the number of columns.
SOURCE_JOINT_NAMES = [
    "Head_Yaw",
    "Left_Shoulder_Pitch",
    "Right_Shoulder_Pitch",
    "Left_Hip_Pitch",
    "Right_Hip_Pitch",
    "Head_Pitch",
    "Left_Shoulder_Roll",
    "Right_Shoulder_Roll",
    "Left_Hip_Roll",
    "Right_Hip_Roll",
    "Left_Elbow_Pitch",
    "Right_Elbow_Pitch",
    "Left_Hip_Yaw",
    "Right_Hip_Yaw",
    "Left_Elbow_Yaw",
    "Right_Elbow_Yaw",
    "Left_Knee_Pitch",
    "Right_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Right_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Ankle_Roll",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_motion(source: dict, joint_names: list[str]) -> tuple[np.ndarray, float]:
    if len(joint_names) != len(SOURCE_JOINT_NAMES) or set(joint_names) != set(
        SOURCE_JOINT_NAMES
    ):
        raise ValueError("Runtime joint names must be exactly the verified 22 K1 names")
    q = np.asarray(source["joint_pos"])
    pos = np.asarray(source["body_pos_w"])
    quat = np.asarray(source["body_quat_w"])
    if q.ndim != 2 or q.shape[1] != 22 or len(q) < 2:
        raise ValueError("Expected T x 22 K1 joints, T >= 2")
    if pos.shape != (len(q), 23, 3) or quat.shape != (len(q), 23, 4):
        raise ValueError("Expected root plus 22 joint-child body transforms")
    if not all(np.isfinite(x).all() for x in (q, pos, quat)):
        raise ValueError("Non-finite source pose")
    if np.max(np.abs(np.linalg.norm(quat, axis=-1) - 1)) > 1e-4:
        raise ValueError("Source has non-unit body quaternion")
    fps_values = np.asarray(source["fps"]).reshape(-1)
    if len(fps_values) != 1 or not np.isfinite(fps_values[0]) or fps_values[0] <= 0:
        raise ValueError("Expected one positive source frame rate")
    order = [SOURCE_JOINT_NAMES.index(n) for n in joint_names]
    return np.concatenate(
        (pos[:, 0], quat[:, 0, [1, 2, 3, 0]], q[:, order]), axis=1
    ), float(fps_values[0])


def resample_qpos(csv: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Linear translation/joints and shortest-arc SLERP, with no pose correction."""
    times = (
        np.arange(int(np.floor((len(csv) - 1) / source_fps * target_fps)) + 1)
        / target_fps
        * source_fps
    )
    i = np.minimum(np.floor(times).astype(int), len(csv) - 2)
    a = (times - i)[:, None]
    out = csv[i] * (1 - a) + csv[i + 1] * a
    q0, q1 = csv[i, 3:7].copy(), csv[i + 1, 3:7].copy()
    dot = np.sum(q0 * q1, axis=1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    theta = np.arccos(np.clip(np.abs(dot), 0, 1))
    # sinc gives the exact SLERP limit for coincident rotations.
    denom = np.sinc(theta / np.pi)
    out[:, 3:7] = (
        (1 - a) * np.sinc((1 - a) * theta / np.pi) * q0
        + a * np.sinc(a * theta / np.pi) * q1
    ) / denom
    return out


def terrain_from_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    root_xy: np.ndarray,
    *,
    cell_size: float,
    clearance: float,
    max_height: float,
    min_height: float,
) -> tuple[list[dict], list[list[float]], dict]:
    """Maximize each occupied-cell height subject to conservative AABB clearance.

    Candidate cells must be traversed by the root path. Every collision geom at
    every sampled frame constrains the cell, including revisits. This excludes
    decorative off-path obstacles and never invents support contacts.
    """
    if lower.shape != upper.shape or lower.ndim != 3 or lower.shape[-1] != 3:
        raise ValueError("Expected T x G x 3 collision bounds")
    if (
        root_xy.shape != (len(lower), 2)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
    ):
        raise ValueError("Invalid collision bounds/root path")
    if (
        np.any(lower > upper)
        or cell_size <= 0
        or not 0 < min_height <= max_height
        or clearance < 0
    ):
        raise ValueError("Invalid clearance problem")
    origin = (
        np.floor(
            np.minimum(np.min(lower[..., :2], axis=(0, 1)), root_xy.min(axis=0))
            / cell_size
        )
        * cell_size
    )
    end = (
        np.ceil(
            np.maximum(np.max(upper[..., :2], axis=(0, 1)), root_xy.max(axis=0))
            / cell_size
        )
        * cell_size
    )
    shape = np.maximum(1, np.rint((end - origin) / cell_size).astype(int) + 1)
    heights = np.full(tuple(shape), max_height, dtype=float)
    low = np.floor((lower[..., :2].reshape(-1, 2) - origin) / cell_size).astype(int)
    high = np.floor((upper[..., :2].reshape(-1, 2) - origin) / cell_size).astype(int)
    limits = lower[..., 2].reshape(-1) - clearance
    # Rasterize every world AABB, including its closed upper boundary.
    extent = np.max(high - low, axis=0)
    for dx in range(int(extent[0]) + 1):
        for dy in range(int(extent[1]) + 1):
            idx = low + [dx, dy]
            mask = np.all(idx <= high, axis=1)
            np.minimum.at(heights, (idx[mask, 0], idx[mask, 1]), limits[mask])
    path = np.floor((root_xy - origin) / cell_size).astype(int)
    occupied = np.zeros(tuple(shape), dtype=bool)
    occupied[path[:, 0], path[:, 1]] = True
    valid = occupied & (heights >= min_height)
    boxes = []
    for x, y in np.argwhere(valid):
        h = float(heights[x, y])
        xy = origin + (np.array([x, y]) + 0.5) * cell_size
        boxes.append(
            {
                "center": [float(xy[0]), float(xy[1]), h / 2],
                "half_size": [cell_size / 2, cell_size / 2, h / 2],
            }
        )
    bounds = [
        np.minimum(origin, root_xy.min(axis=0)).tolist(),
        np.maximum(end, root_xy.max(axis=0)).tolist(),
    ]
    bounds = [
        (np.asarray(bounds[0]) - 2).tolist(),
        (np.asarray(bounds[1]) + 2).tolist(),
    ]
    return (
        boxes,
        bounds,
        {
            "root_path_cells": int(occupied.sum()),
            "feasible_obstacle_cells": int(valid.sum()),
            "objective": "maximum per-cell height under all sampled collision AABBs",
        },
    )


def model_indices(model, joint_names):
    import mujoco

    source_joints = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        for n in SOURCE_JOINT_NAMES
    ]
    if min(source_joints) < 0:
        raise ValueError("Robot model lacks verified K1 source joint names")
    runtime_joints = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names
    ]
    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Trunk")
    ] + model.jnt_bodyid[source_joints].tolist()
    if body_ids[0] < 0 or model.nq != 29:
        raise ValueError("Expected floating K1 Trunk with 22 joints")
    return model.jnt_qposadr[runtime_joints], body_ids


def set_pose(data, row, addresses):
    data.qpos[:3] = row[:3]
    data.qpos[3:7] = row[[6, 3, 4, 5]]
    data.qpos[addresses] = row[7:]


def verify_source_fk(model, csv, source, addresses, body_ids):
    import mujoco

    data = mujoco.MjData(model)
    max_pos, max_quat = 0.0, 0.0
    for t, row in enumerate(csv):
        set_pose(data, row, addresses)
        mujoco.mj_kinematics(model, data)
        max_pos = max(
            max_pos,
            float(
                np.max(
                    np.linalg.norm(
                        data.xpos[body_ids] - source["body_pos_w"][t], axis=-1
                    )
                )
            ),
        )
        dot = np.abs(np.sum(data.xquat[body_ids] * source["body_quat_w"][t], axis=-1))
        max_quat = max(max_quat, float(np.max(np.abs(1 - dot))))
    if max_pos > 0.001 or max_quat > 1e-4:
        raise ValueError(
            f"Source/model FK mismatch: {max_pos:.6g} m, {max_quat:.6g} quaternion error"
        )
    return {
        "frames": len(csv),
        "max_body_position_error_m": max_pos,
        "max_quaternion_1_minus_abs_dot": max_quat,
    }


def collision_bounds(model, poses, addresses):
    import mujoco

    data = mujoco.MjData(model)
    ids = np.flatnonzero(
        (model.geom_bodyid > 0)
        & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    )
    if not len(ids):
        raise ValueError("Robot has no collision geometry")
    # MuJoCo's local AABBs include mesh convex hulls and primitive geometry.
    local_center = model.geom_aabb[ids, :3]
    local_half = model.geom_aabb[ids, 3:]
    lower, upper = [], []
    for row in poses:
        set_pose(data, row, addresses)
        mujoco.mj_kinematics(model, data)
        R = data.geom_xmat[ids].reshape(-1, 3, 3)
        center = data.geom_xpos[ids] + np.einsum("gij,gj->gi", R, local_center)
        half = np.einsum("gij,gj->gi", np.abs(R), local_half)
        lower.append(center - half)
        upper.append(center + half)
    return np.asarray(lower), np.asarray(upper)


def validate_contacts(xml, poses, joint_names, boxes):
    """Independent native contact check; no clipping/filtering of robot geometry."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(xml))
    # A standalone articulation is required so baseline ground is unambiguous.
    for geom in spec.geoms:
        if geom.parent == spec.worldbody and geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError("Robot asset must not already contain a ground plane")
    spec.worldbody.add_geom(
        name="pair_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.1]
    )
    for i, box in enumerate(boxes):
        spec.worldbody.add_geom(
            name=f"pair_obstacle_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=box["center"],
            size=box["half_size"],
        )
    model = spec.compile()
    addresses, _ = model_indices(model, joint_names)
    data = mujoco.MjData(model)
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pair_ground")
    obstacle_ids = set(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"pair_obstacle_{i}")
        for i in range(len(boxes))
    )
    min_ground, min_self, min_obstacle = 0.0, 0.0, 0.0
    ground_frames = self_frames = 0
    for row in poses:
        set_pose(data, row, addresses)
        mujoco.mj_forward(model, data)
        g, s = False, False
        for contact in data.contact:
            distance = float(contact.dist)
            pair = set(map(int, contact.geom))
            if pair & obstacle_ids:
                min_obstacle = min(min_obstacle, distance)
            elif ground in pair:
                min_ground = min(min_ground, distance)
                g |= distance < -0.005
            else:
                min_self = min(min_self, distance)
                s |= distance < -0.005
        ground_frames += g
        self_frames += s
    if min_obstacle < -1e-6:
        raise ValueError(
            f"Generated obstacle intersects original reference: {min_obstacle:.6g} m"
        )
    return {
        "frames": len(poses),
        "min_obstacle_distance_m": min_obstacle,
        "min_ground_distance_m": min_ground,
        "min_self_distance_m": min_self,
        "ground_penetration_frames_over_5mm": ground_frames,
        "self_penetration_frames_over_5mm": self_frames,
    }


def segment_ranges(frames: int, fps: float, seconds: float, max_segments: int = 0):
    """Cover a source interval with fixed windows sharing only their endpoint."""
    if (
        frames < 2
        or not np.isfinite(fps)
        or fps <= 0
        or not np.isfinite(seconds)
        or seconds < 0
        or max_segments < 0
    ):
        raise ValueError("Invalid source window specification")
    if seconds == 0:
        return [(0, frames)]
    stride_float = seconds * fps
    stride = int(round(stride_float))
    if stride < 1 or abs(stride - stride_float) > 1e-8:
        raise ValueError(
            "Segment duration must contain an integral number of source intervals"
        )
    ranges = [
        (start, min(start + stride + 1, frames))
        for start in range(0, frames - 1, stride)
    ]
    return ranges[:max_segments] if max_segments else ranges


def load_retarget(directory, stem, source, source_hash, robot_xml):
    """Only consume explicitly successful, hash-bound derived motion artifacts."""
    output = directory / f"{stem}.npz"
    record_path = directory / f"{stem}.json"
    record = json.loads(record_path.read_text())
    if record["source_sha256"] != source_hash or record["output_sha256"] != sha256(
        output
    ):
        raise ValueError("Retarget source/output checksum mismatch")
    if (
        record.get("robot_xml_sha256") != sha256(robot_xml)
        or record.get("converged") is not True
    ):
        raise ValueError("Retarget must converge for the exact runtime robot model")
    start, stop = int(record["source_frame_start"]), int(record["source_frame_stop"])
    if not 0 <= start < stop <= len(source["joint_pos"]):
        raise ValueError("Retarget source frame interval is invalid")
    with np.load(output, allow_pickle=False) as f:
        derived = {k: f[k] for k in f.files}
    source_fps = float(np.asarray(source["fps"]).reshape(-1)[0])
    target_fps = float(np.asarray(derived["fps"]).reshape(-1)[0])
    original_csv, _ = convert_motion(source, SOURCE_JOINT_NAMES)
    resampled = resample_qpos(original_csv[start:stop], source_fps, target_fps)
    if len(derived["joint_pos"]) != len(resampled):
        raise ValueError("Retarget changed source duration")
    if target_fps != source_fps and record.get("target_fps") != target_fps:
        raise ValueError(
            "Retarget frame-rate conversion is missing explicit provenance"
        )
    # Only rate conversion is allowed in the fixed source root trajectory.
    if not np.allclose(
        derived["body_pos_w"][:, 0], resampled[:, :3], rtol=0, atol=5e-7
    ):
        raise ValueError("Retarget changed the fixed source root trajectory")
    root_quat = derived["body_quat_w"][:, 0, [1, 2, 3, 0]]
    dot = np.sum(root_quat * resampled[:, 3:7], axis=-1)
    root_quat = root_quat * np.where(dot < 0, -1, 1)[:, None]
    if not np.allclose(root_quat, resampled[:, 3:7], rtol=0, atol=5e-7):
        raise ValueError("Retarget changed the fixed source root orientation")
    if record.get("joint_names") != SOURCE_JOINT_NAMES:
        raise ValueError(
            "Retarget joint-order provenance differs from the source contract"
        )
    limits = np.asarray(record.get("joint_velocity_limits_rad_s", []), dtype=float)
    if limits.shape != (22,) or not np.isfinite(limits).all() or (limits <= 0).any():
        raise ValueError("Retarget must provide physical joint velocity limits")
    speed_ratio = np.abs(np.diff(derived["joint_pos"], axis=0)) * target_fps / limits
    if float(speed_ratio.max()) > 1.00001:
        raise ValueError("Retarget exceeds declared physical joint velocity limits")
    return derived, record, start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--robot-xml", type=Path, required=True)
    p.add_argument(
        "--joint-order",
        type=Path,
        required=True,
        help="JSON list of runtime K1 joint names",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--retarget-dir",
        type=Path,
        help="Validated morphology-retarget NPZ and JSON provenance directory",
    )
    p.add_argument(
        "--retarget-files",
        nargs="+",
        help="Explicit derived NPZ basenames, including __fSTART_STOP suffix",
    )
    p.add_argument(
        "--segment-seconds",
        type=float,
        default=0,
        help="Fixed source windows; zero preserves full clips",
    )
    p.add_argument(
        "--max-segments",
        type=int,
        default=0,
        help="Explicit pilot prefix per source clip; zero includes every window",
    )
    p.add_argument(
        "--clips",
        nargs="+",
        required=True,
        help="Exact source stems; use ALL for all 77",
    )
    p.add_argument("--cell-size", type=float, default=0.2)
    p.add_argument("--clearance", type=float, default=0.02)
    p.add_argument("--max-height", type=float, default=0.2)
    p.add_argument("--min-height", type=float, default=0.03)
    p.add_argument("--validation-fps", type=float, default=100.0)
    p.add_argument(
        "--contact-tolerance",
        type=float,
        default=0.001,
        help="Maximum reference ground/self penetration in metres; validation only",
    )
    a = p.parse_args()
    if not np.isfinite(a.contact_tolerance) or a.contact_tolerance < 0:
        raise ValueError("Contact tolerance must be finite and nonnegative")
    if bool(a.retarget_dir) != bool(a.retarget_files):
        raise ValueError("Use --retarget-dir together with explicit --retarget-files")
    if a.retarget_files and (
        len(a.retarget_files) != len(set(a.retarget_files))
        or any(Path(f).name != f or not f.endswith(".npz") for f in a.retarget_files)
    ):
        raise ValueError("Retarget files must be unique NPZ basenames")
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.validation_fps < 50 or a.validation_fps % 50 != 0:
        raise ValueError("Validation rate must be a multiple of 50 Hz and >= 50")
    if sha256(a.archive) != ARCHIVE_SHA256:
        raise ValueError(
            "Archive does not match the pinned user-requested K1 LAFAN source"
        )
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(a.robot_xml))
    names = json.loads(a.joint_order.read_text())
    addresses, body_ids = model_indices(model, names)
    terrains, motions, rates, records = [], {}, {}, []
    a.output.mkdir(parents=True)
    (a.output / "INCOMPLETE").write_text(
        "Validation has not completed. Do not use this directory for training.\n"
    )
    with tarfile.open(a.archive) as archive:
        members = {
            Path(m.name).stem: m
            for m in archive.getmembers()
            if m.isfile()
            and m.name.startswith("booster_k1/")
            and m.name.endswith(".npz")
        }
        requested = sorted(members) if a.clips == ["ALL"] else a.clips
        if len(requested) != len(set(requested)) or not set(requested) <= set(members):
            raise ValueError("Requested clips must be unique exact K1 LAFAN stems")
        if a.retarget_files and any(
            Path(f).stem.split("__f", 1)[0] not in requested for f in a.retarget_files
        ):
            raise ValueError(
                "Every retarget file must correspond to a requested source clip"
            )
        for stem in requested:
            raw = archive.extractfile(members[stem]).read()
            with np.load(io.BytesIO(raw), allow_pickle=False) as src:
                source = {k: src[k] for k in src.files}
            original_source = source
            derived_stems = (
                [stem]
                if not a.retarget_files
                else [
                    Path(f).stem
                    for f in a.retarget_files
                    if Path(f).stem.split("__f", 1)[0] == stem
                ]
            )
            if not derived_stems:
                raise ValueError(f"No explicit retarget files for {stem}")
            for derived_stem in derived_stems:
                source = original_source
                source_hash = hashlib.sha256(raw).hexdigest()
                retarget_record, source_offset = None, 0
                if a.retarget_dir:
                    source, retarget_record, source_offset = load_retarget(
                        a.retarget_dir, derived_stem, source, source_hash, a.robot_xml
                    )
                full_csv, fps = convert_motion(source, names)
                original_fps = float(np.asarray(original_source["fps"]).reshape(-1)[0])
                windows = segment_ranges(
                    len(full_csv), fps, a.segment_seconds, a.max_segments
                )
                for segment_index, (start, stop) in enumerate(windows):
                    csv = full_csv[start:stop]
                    source_start = source_offset + start * original_fps / fps
                    source_stop = source_offset + (stop - 1) * original_fps / fps + 1
                    source_start = (
                        round(source_start)
                        if abs(source_start - round(source_start)) < 1e-8
                        else source_start
                    )
                    source_stop = (
                        round(source_stop)
                        if abs(source_stop - round(source_stop)) < 1e-8
                        else source_stop
                    )
                    segment_source = {
                        k: v[start:stop] if v.ndim and len(v) == len(full_csv) else v
                        for k, v in source.items()
                    }
                    fk = verify_source_fk(
                        model, csv, segment_source, addresses, body_ids
                    )
                    poses = resample_qpos(csv, fps, a.validation_fps)
                    low, high = collision_bounds(model, poses, addresses)
                    boxes, bounds, feasibility = terrain_from_bounds(
                        low,
                        high,
                        poses[:, :2],
                        cell_size=a.cell_size,
                        clearance=a.clearance,
                        max_height=a.max_height,
                        min_height=a.min_height,
                    )
                    contacts = validate_contacts(a.robot_xml, poses, names, boxes)
                    if (
                        min(
                            contacts["min_ground_distance_m"],
                            contacts["min_self_distance_m"],
                        )
                        < -a.contact_tolerance
                    ):
                        raise ValueError(
                            f"Reference contact gate failed for {stem} segment {segment_index}: {contacts}"
                        )
                    control_poses = resample_qpos(csv, fps, 50.0)
                    joint_velocity = np.diff(control_poses[:, 7:], axis=0) * 50.0
                    joint_acceleration = np.diff(joint_velocity, axis=0) * 50.0
                    rate_diagnostics = {
                        "control_fps": 50,
                        "joint_speed_max_rad_s": float(np.max(np.abs(joint_velocity))),
                        "joint_speed_p99_rad_s": float(
                            np.quantile(np.abs(joint_velocity), 0.99)
                        ),
                        "joint_acceleration_max_rad_s2": float(
                            np.max(np.abs(joint_acceleration))
                        )
                        if len(joint_acceleration)
                        else None,
                        "joint_step_max_rad": float(
                            np.max(np.abs(np.diff(control_poses[:, 7:], axis=0)))
                        ),
                    }
                    record = {
                        "source": members[stem].name,
                        "source_sha256": source_hash,
                        "source_frame_start": source_start,
                        "source_frame_stop": source_stop,
                        "segment_index": segment_index,
                        "retarget": retarget_record,
                        "frames": len(csv),
                        "fps": fps,
                        "source_fk": fk,
                        "pairing": feasibility,
                        "native_contacts": contacts,
                        "rate_diagnostics": rate_diagnostics,
                    }
                    for suffix, obstacles in (
                        [("flat", []), ("terrain", boxes)] if boxes else [("flat", [])]
                    ):
                        segment_stem = (
                            f"{stem}__f{source_start:g}_{source_stop:g}"
                            if a.segment_seconds or a.retarget_dir
                            else stem
                        )
                        file = f"{segment_stem}__{suffix}.csv"
                        terrain_id = len(terrains)
                        terrains.append(
                            {
                                "id": terrain_id,
                                "name": f"{segment_stem}__{suffix}",
                                "ground_height": 0.0,
                                "boxes": obstacles,
                                "bounds_xy": bounds,
                            }
                        )
                        np.savetxt(a.output / file, csv, delimiter=",", fmt="%.9g")
                        motions[file] = {
                            "terrain_id": terrain_id,
                            "source_clip": stem,
                            "source_frame_start": source_start,
                            "source_frame_stop": source_stop,
                            "variant": suffix,
                            "csv_sha256": sha256(a.output / file),
                        }
                        rates[file] = fps
                    records.append(record)
                    print(
                        json.dumps(
                            {
                                "clip": stem,
                                "segment_index": segment_index,
                                "boxes": len(boxes),
                                "fk_max_m": fk["max_body_position_error_m"],
                                "ground_min_m": contacts["min_ground_distance_m"],
                                "self_min_m": contacts["min_self_distance_m"],
                            }
                        ),
                        flush=True,
                    )
    manifest = {
        "schema_version": 1,
        "robot": "booster_k1",
        "terrains": terrains,
        "motions": motions,
    }
    provenance = {
        "dataset": DATASET,
        "revision": REVISION,
        "archive_path": ARCHIVE_PATH,
        "archive_sha256": ARCHIVE_SHA256,
        "robot_xml_sha256": sha256(a.robot_xml),
        "source_joint_order": SOURCE_JOINT_NAMES,
        "runtime_joint_order": names,
        "source_quaternion": "wxyz",
        "csv_quaternion": "xyzw",
        "pose_transform": "joint and quaternion column permutations only after explicit morphology retarget when supplied",
        "validation_fps": a.validation_fps,
        "contact_tolerance_m": a.contact_tolerance,
        "segment_seconds": a.segment_seconds,
        "max_segments": a.max_segments,
        "retargeted": a.retarget_dir is not None,
        "retarget_files": a.retarget_files,
        "cell_size_m": a.cell_size,
        "clearance_m": a.clearance,
        "max_obstacle_height_m": a.max_height,
        "min_obstacle_height_m": a.min_height,
        "limitations": [
            "No terrain or contact labels provided by source",
            "Collision-free obstacles do not constitute stair-climbing demonstrations",
            "Sampled kinematic feasibility does not prove dynamic trackability",
            "Original ground/self intersections are reported, never repaired",
        ],
        "clips": records,
    }
    for file, value in [
        ("terrain_manifest.json", manifest),
        ("motion_rates.json", rates),
        ("import_provenance.json", provenance),
    ]:
        (a.output / file).write_text(json.dumps(value, indent=2) + "\n")
    (a.output / "INCOMPLETE").unlink()
    print(
        json.dumps(
            {
                "source_clips": len(requested),
                "source_segments": len(records),
                "motion_terrain_pairs": len(motions),
                "nonflat_pairs": sum(bool(t["boxes"]) for t in terrains),
            }
        )
    )


if __name__ == "__main__":
    main()
