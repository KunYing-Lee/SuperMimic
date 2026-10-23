"""Immutable InstinctMJ pairing, FK and exact-mesh reference rendering helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")
import mujoco
import numpy as np
import trimesh
import yaml

RELEASE_ROOT = Path("data/instinct_release/unpacked/20251116_50cm_kneeClimbStep1")
G1_XML = Path("data/instinct_release/g1/xml/g1_29dof_torsobase_popsicle.xml")
K1_XML = Path("data/k1_description/k1.xml")

# Anatomical link landmarks for the OmniRetarget interaction graph. Contact
# constraints below use collision surfaces, not these link origins.
LINKS = {
    "torso_link": "Trunk",
    "left_hip_pitch_link": "Left_Hip_Pitch",
    "right_hip_pitch_link": "Right_Hip_Pitch",
    "left_knee_link": "Left_Shank",
    "right_knee_link": "Right_Shank",
    "left_ankle_roll_link": "left_foot_link",
    "right_ankle_roll_link": "right_foot_link",
    "left_shoulder_pitch_link": "Left_Arm_1",
    "right_shoulder_pitch_link": "Right_Arm_1",
    "left_elbow_link": "left_hand_link",
    "right_elbow_link": "right_hand_link",
    "left_wrist_yaw_link": "left_hand_end_ball",
    "right_wrist_yaw_link": "right_hand_end_ball",
}
# Contact transfer is body-surface to body-surface. This permits rolling contact
# while retaining the source scene anchor, without freezing a material point.
CONTACT_BODIES = {
    "torso_link": ["Trunk"],
    "head_surface": ["Head_1", "Head_2"],
    "pelvis": ["Trunk"],
    "left_hip_roll_link": ["Left_Hip_Roll", "Left_Hip_Yaw"],
    "right_hip_roll_link": ["Right_Hip_Roll", "Right_Hip_Yaw"],
    "left_knee_link": ["Left_Shank"],
    "right_knee_link": ["Right_Shank"],
    "left_ankle_roll_link": ["left_foot_link"],
    "right_ankle_roll_link": ["right_foot_link"],
    "left_shoulder_yaw_link": ["Left_Arm_2", "Left_Arm_3"],
    "right_shoulder_yaw_link": ["Right_Arm_2", "Right_Arm_3"],
    "left_elbow_link": ["left_hand_link"],
    "right_elbow_link": ["right_hand_link"],
    "left_wrist_yaw_link": ["left_hand_link", "left_hand_end_ball"],
    "right_wrist_yaw_link": ["right_hand_link", "right_hand_end_ball"],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pairs(root=RELEASE_ROOT):
    metadata = yaml.safe_load((root / "metadata.yaml").read_text())
    terrains = {t["terrain_id"]: root / t["terrain_file"] for t in metadata["terrains"]}
    return [
        {
            "id": i,
            "motion": root / p["motion_file"],
            "terrain": terrains[p["terrain_id"]],
            "terrain_id": p["terrain_id"],
            "name": f"pair_{i:02d}_{Path(p['motion_file']).stem}",
        }
        for i, p in enumerate(metadata["motion_files"])
    ]


def load_qpos(path, model):
    with np.load(path, allow_pickle=False) as a:
        names = a["joint_names"].tolist()
        expected = [
            model.joint(j).name
            for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("Source and model joint names differ")
        q = np.tile(model.qpos0, (len(a["joint_pos"]), 1))
        q[:, :3], q[:, 3:7] = a["base_pos_w"], a["base_quat_w"]
        for j, name in enumerate(names):
            q[:, model.joint(name).qposadr[0]] = a["joint_pos"][:, j]
        if not np.isfinite(q).all() or not np.allclose(
            np.linalg.norm(q[:, 3:7], axis=1), 1, atol=1e-5
        ):
            raise ValueError("Invalid source pose")
        return q, float(a["framerate"])


def resample_source(q, source_fps, target_fps):
    """Sample the source's linear/SLERP curve before constrained retargeting."""
    from scipy.spatial.transform import Rotation, Slerp

    t = np.arange(len(q)) / source_fps
    target = np.arange(int(np.floor(t[-1] * target_fps)) + 1) / target_fps
    out = np.stack([np.interp(target, t, q[:, j]) for j in range(q.shape[1])], axis=1)
    out[:, 3:7] = Slerp(t, Rotation.from_quat(q[:, [4, 5, 6, 3]]))(target).as_quat()[
        :, [3, 0, 1, 2]
    ]
    return out


def geom_samples(model, geom_id, spacing=0.015):
    """Deterministic surface discretization; never substitute mesh centers.

    Mesh samples use the actual MuJoCo convex collision hull. Curved primitive
    surfaces use a declared spatial resolution. Results remain sampled checks.
    """
    g = geom_id
    kind, size = int(model.geom_type[g]), model.geom_size[g]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        mesh = trimesh.convex.convex_hull(model.mesh_vert[start : start + count])
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2 * size)
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=size[0])
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(
            radius=size[0], height=2 * size[1], count=[16, 16]
        )
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(
            radius=size[0], height=2 * size[1], sections=24
        )
    else:
        raise ValueError(f"Unsupported collision geom type {kind}")
    v, _ = trimesh.remesh.subdivide_to_size(
        mesh.vertices, mesh.faces, max_edge=spacing, max_iter=10
    )
    return np.unique(v, axis=0)


def collision_samples(model, spacing=0.015):
    return [
        (g, geom_samples(model, g, spacing))
        for g in range(model.ngeom)
        if model.geom_contype[g] or model.geom_conaffinity[g]
    ]


def world_samples(data, samples):
    return np.concatenate(
        [p @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] for g, p in samples]
    )


def anchor_surface_distance(model, data, geoms, probe):
    """Signed anchor-to-body-surface distance and native tangent Jacobian."""
    best = None
    for geom in geoms:
        segment = np.zeros(6)
        distance = mujoco.mj_geomDistance(model, data, geom, probe, 10.0, segment)
        if best is None or distance < best[0]:
            best = (distance, geom, segment)
    if best is None:
        raise ValueError("An anchor needs a nonempty collision group")
    raw_distance, geom, segment = best
    distance = raw_distance + model.geom_size[probe, 0]
    direction = segment[3:] - segment[:3]
    length = np.linalg.norm(direction)
    if length < 1e-12:
        raise RuntimeError("Undefined contact distance normal")
    normal = direction / length * (1 if raw_distance >= 0 else -1)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jp, jr, segment[:3], model.geom_bodyid[geom])
    return float(distance), -normal @ jp


def render(
    xml,
    qpos,
    fps,
    mesh_path,
    output,
    title,
    *,
    mesh_scale=1.0,
    stride=3,
    display_scale=None,
    failed_frame=None,
):
    """Render actual triangle surfaces; collision is audited separately."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    spec = mujoco.MjSpec.from_file(str(xml.resolve()))
    for g in spec.geoms:
        g.group = 5 if g.contype or g.conaffinity else 1
    mesh = trimesh.load(mesh_path, force="mesh")
    spec.add_mesh(
        name="release_terrain",
        uservert=(mesh.vertices * mesh_scale).flatten(),
        userface=mesh.faces.flatten(),
    )
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="release_terrain",
        contype=0,
        conaffinity=0,
        rgba=[0.48, 0.58, 0.64, 1],
        group=0,
    )
    spec.worldbody.add_light(pos=[0, -1, 5], dir=[0, 0, -1])
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 960, 640
    model = spec.compile()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=640, width=960)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = np.median(qpos[:, :3], axis=0)
    camera.lookat[2] = 0.55 * mesh_scale
    camera.distance = max(
        3.5 * mesh_scale, float(np.ptp(qpos[:, :2], axis=0).max()) + 1.5
    )
    camera.azimuth, camera.elevation = 145, -25
    options = mujoco.MjvOption()
    options.geomgroup[5] = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        output, fps=fps / stride, codec="libx264", macro_block_size=1
    ) as writer:
        for i in range(0, len(qpos), stride):
            data.qpos[:] = qpos[i]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=options)
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            draw.rectangle([0, 0, 960, 48], fill=(20, 24, 28))
            draw.text((12, 8), title, fill="white")
            reported_scale = mesh_scale if display_scale is None else display_scale
            draw.text(
                (12, 28),
                f"REFERENCE ONLY | t={i / fps:.2f}s | source scene scale={reported_scale:.6f}",
                fill="white",
            )
            if failed_frame is not None and i >= failed_frame:
                draw.rectangle([0, 48, 960, 75], fill=(180, 20, 20))
                draw.text((12, 54), "FAILED SOLVER FRAME - DO NOT USE", fill="white")
            writer.append_data(np.asarray(image))
    renderer.close()
    return output
