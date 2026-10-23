"""Render named-robot terrain references or recorded policy evaluations.

Reference-only mode reads native-rate CSV poses without changing them. Policy
mode renders the reference ghost at its original world position, so drift is
visible. Both modes map every joint by name and render the exact manifest boxes.
The renderer does not require Torch or a running training process.
"""

from __future__ import annotations

# Select the platform OpenGL backend before importing MuJoCo and support direct
# script execution outside the repository root.
# ruff: noqa: E402
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from supermimic.robot import get_robot
from supermimic.terrain_layout import atlas_layout


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_for_clip(robot, terrain, ghost: bool, width: int, height: int):
    """Load exact assets and attach a namespaced visual reference articulation."""
    spec = mujoco.MjSpec.from_file(robot.mjcf_path)
    for geom in spec.geoms:
        geom.group = 5 if geom.contype or geom.conaffinity else 1
    if ghost:
        reference = mujoco.MjSpec.from_file(robot.mjcf_path)
        for key in list(reference.keys):
            reference.delete(key)
        for actuator in list(reference.actuators):
            reference.delete(actuator)
        for geom in reference.geoms:
            geom.group = 5 if geom.contype or geom.conaffinity else 2
            geom.contype = geom.conaffinity = 0
            geom.material = ""
            geom.rgba = [0.2, 0.65, 1.0, 0.3]
        spec.attach(reference, prefix="reference_", frame=spec.worldbody.add_frame())
    spec.add_texture(
        name="terrain_grid_texture",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.62, 0.66, 0.71],
        rgb2=[0.35, 0.40, 0.47],
        width=512,
        height=512,
    )
    material = spec.add_material(
        name="terrain_grid",
        texrepeat=[2, 2],
        texuniform=True,
        reflectance=0.05,
    )
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "terrain_grid_texture"
    spec.worldbody.add_geom(
        name="terrain_ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0, 0, terrain["ground_height"]],
        size=[0, 0, 0.1],
        material="terrain_grid",
    )
    for i, box in enumerate(terrain["boxes"]):
        spec.worldbody.add_geom(
            name=f"terrain_obstacle_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=box["center"],
            size=box["half_size"],
            rgba=[0.52, 0.40, 0.26, 1.0],
        )
    spec.worldbody.add_light(
        name="render_key",
        pos=[0, 0, 6],
        dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.4, 0.4, 0.4],
    )
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    spec.visual.quality.shadowsize = 2048
    model = spec.compile()
    model.vis.headlight.ambient[:] = 0.2
    model.vis.headlight.diffuse[:] = 0.4
    return model


def pose_addresses(model, robot, prefix=""):
    """Resolve root and hinge addresses without assuming an MJCF joint order."""
    root_id = model.body(prefix + robot.root_body).id
    root_joint = int(model.body_jntadr[root_id])
    if (
        model.body_jntnum[root_id] != 1
        or model.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE
    ):
        raise ValueError("Rendering requires the named robot's floating root")
    addresses = np.asarray(
        [model.jnt_qposadr[model.joint(prefix + name).id] for name in robot.joint_names]
    )
    return int(model.jnt_qposadr[root_joint]), addresses


def write_pose(data, addresses, p, quat, q):
    if not all(np.isfinite(value).all() for value in (p, quat, q)):
        raise ValueError("Cannot render a non-finite recorded pose")
    root, joints = addresses
    if np.asarray(q).shape != joints.shape:
        raise ValueError("Recorded joints do not match the named robot profile")
    data.qpos[root : root + 3] = p
    data.qpos[root + 3 : root + 7] = quat
    data.qpos[joints] = q


def terrain_for_clip(manifest, name):
    if manifest is None:
        return {"id": 0, "name": "flat", "ground_height": 0.0, "boxes": []}
    key = name if name.endswith(".csv") else name + ".csv"
    terrain_id = manifest["motions"][key]["terrain_id"]
    matches = [item for item in manifest["terrains"] if item["id"] == terrain_id]
    if len(matches) != 1:
        raise ValueError(f"Motion {name} must map to exactly one terrain")
    return matches[0]


def geometry_for_clip(manifest, terrain):
    """Express every installed physical tile in the selected motion's frame."""
    if manifest is None:
        return terrain
    offsets, _, _ = atlas_layout(manifest["terrains"])
    origin = offsets[terrain["id"]]
    boxes = []
    for entry, offset in zip(manifest["terrains"], offsets, strict=True):
        for box in entry["boxes"]:
            boxes.append(
                {
                    "center": (np.asarray(box["center"]) + offset - origin).tolist(),
                    "half_size": box["half_size"],
                }
            )
    return {"ground_height": float(-origin[2]), "boxes": boxes}


def native_frames(length, source_fps, start, seconds, video_fps):
    """Select existing source frames on a video clock; never modify a pose."""
    if start < 0 or seconds <= 0 or source_fps <= 0:
        raise ValueError("Invalid time range or source frame rate")
    end = min((length - 1) / source_fps, start + seconds)
    if end < start:
        raise ValueError("Requested start exceeds the available motion")
    times = start + np.arange(int(np.floor((end - start) * video_fps)) + 1) / video_fps
    return np.rint(times * source_fps).astype(int)


def reference_source(directory, selected):
    files = {path.stem: path for path in sorted(directory.glob("*.csv"))}
    names = selected or list(files)
    if not names or any(name not in files for name in names):
        raise ValueError("Requested reference motion was not found")
    rates_path = directory / "motion_rates.json"
    rates = json.loads(rates_path.read_text()) if rates_path.exists() else None
    metadata_path = directory / "terrain_manifest.json"
    manifest = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    clips, sources = [], {}
    for name in names:
        path = files[name]
        raw = np.loadtxt(path, delimiter=",", ndmin=2)
        if len(raw) < 2 or not np.isfinite(raw).all():
            raise ValueError(f"Reference must contain finite poses: {name}")
        fps = float(rates[path.name]) if rates is not None else 30.0
        clips.append(
            {
                "name": name,
                "fps": fps,
                "root_p": raw[:, :3],
                "root_quat": raw[:, [6, 3, 4, 5]],
                "q": raw[:, 7:],
                "alive": np.ones(len(raw), bool),
            }
        )
        sources[str(path.resolve())] = sha256(path)
    for path in (rates_path, metadata_path):
        if path.exists():
            sources[str(path.resolve())] = sha256(path)
    return clips, manifest, sources


def trajectory_source(path, selected):
    with np.load(path, allow_pickle=False) as archive:
        record = dict(archive)
    names = list(record["names"])
    selected = selected or names
    manifest = (
        json.loads(str(record["terrain_manifest"].item()))
        if "terrain_manifest" in record
        else None
    )
    clips = []
    for name in selected:
        index = names.index(name)
        alive = record["alive"][:, index]
        if not alive[0] or (alive[1:] & ~alive[:-1]).any():
            raise ValueError("Recorded live frames must be a contiguous prefix")
        live_count = int(alive.sum())
        end = live_count
        nonfinite_terminal = False
        # Keep the first terminal pose if present and finite, never replay later
        # uncontrolled simulation or freeze the robot to hide its outcome.
        if live_count < len(alive):
            pose_values = [
                record[key][live_count, index] for key in ("root_p", "root_quat", "q")
            ]
            if all(np.isfinite(value).all() for value in pose_values):
                end += 1
            else:
                nonfinite_terminal = True
        frames = record["f"][:end, index].astype(int)
        rotations = record["ref_root_R"][frames]
        ref_quats = np.empty((end, 4))
        for row, rotation in enumerate(rotations):
            mujoco.mju_mat2Quat(ref_quats[row], rotation.astype(np.float64).reshape(9))
        clips.append(
            {
                "name": name,
                "fps": 50.0,
                "root_p": record["root_p"][:end, index],
                "root_quat": record["root_quat"][:end, index],
                "q": record["q"][:end, index],
                "alive": alive[:end],
                "ref_root_p": record["ref_root_p"][frames],
                "ref_root_quat": ref_quats,
                "ref_q": record["ref_q"][frames],
                "nonfinite_terminal": nonfinite_terminal,
            }
        )
    robot = str(record["robot"].item()) if "robot" in record else None
    return clips, manifest, {str(path.resolve()): sha256(path)}, robot


def annotation(image, lines, unvalidated=False):
    draw, font = ImageDraw.Draw(image), ImageFont.load_default(size=18)
    y = 12
    for text in lines:
        bounds = draw.textbbox((12, y), text, font=font)
        draw.rectangle(
            (bounds[0] - 5, bounds[1] - 4, bounds[2] + 5, bounds[3] + 4),
            fill=(255, 255, 255),
        )
        draw.text((12, y), text, font=font, fill=(25, 30, 35))
        y += 28
    if unvalidated:
        text = "UNVALIDATED REFERENCE - KINEMATIC PREVIEW ONLY"
        draw.rectangle(
            (0, image.height - 32, image.width, image.height), fill=(165, 28, 28)
        )
        draw.text((12, image.height - 27), text, font=font, fill="white")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--motion-dir",
        type=Path,
        help="render source poses without a policy or physics rollout",
    )
    source.add_argument(
        "--trajectory", type=Path, help="NPZ produced by evaluate.py --record"
    )
    parser.add_argument("--robot", choices=("k1", "g1"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--clips", nargs="+")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--distance", type=float, default=3.8)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-18.0)
    parser.add_argument("--title", default="")
    parser.add_argument(
        "--unvalidated",
        action="store_true",
        help="mark references that have not passed contact/limit acceptance",
    )
    args = parser.parse_args()
    if (
        args.fps < 1
        or args.width < 2
        or args.height < 2
        or args.width % 2
        or args.height % 2
    ):
        parser.error("Video dimensions must be positive even integers and fps positive")
    if args.trajectory:
        clips, manifest, sources, recorded_robot = trajectory_source(
            args.trajectory, args.clips
        )
        if args.robot and recorded_robot and args.robot != recorded_robot:
            parser.error("--robot differs from the recorded robot")
        name = args.robot or recorded_robot
    else:
        clips, manifest, sources = reference_source(args.motion_dir, args.clips)
        name = args.robot
    if name is None:
        parser.error(
            "--robot is required for reference mode or legacy trajectory files"
        )
    robot = get_robot(name)
    outputs = {
        "video": args.out,
        "contact_sheet": args.out.with_suffix(".contact.png"),
        "metadata": args.out.with_suffix(".json"),
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("Rendering requires fresh output paths")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{args.width}x{args.height}",
            "-r",
            str(args.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(args.out),
        ],
        stdin=subprocess.PIPE,
    )
    screenshots, rendered = [], []
    try:
        for clip in clips:
            terrain = terrain_for_clip(manifest, clip["name"])
            geometry = geometry_for_clip(manifest, terrain)
            model = model_for_clip(
                robot, geometry, bool(args.trajectory), args.width, args.height
            )
            data = mujoco.MjData(model)
            addresses = pose_addresses(model, robot)
            ghost_addresses = (
                pose_addresses(model, robot, "reference_") if args.trajectory else None
            )
            camera, option = mujoco.MjvCamera(), mujoco.MjvOption()
            camera.distance, camera.azimuth, camera.elevation = (
                args.distance,
                args.azimuth,
                args.elevation,
            )
            option.geomgroup[5] = 0
            frames = native_frames(
                len(clip["root_p"]), clip["fps"], args.start, args.seconds, args.fps
            )
            sample_indices = np.rint(np.linspace(0, len(frames) - 1, 4)).astype(int)
            picks = {}
            with mujoco.Renderer(model, args.height, args.width) as renderer:
                for output_index, frame in enumerate(frames):
                    write_pose(
                        data,
                        addresses,
                        clip["root_p"][frame],
                        clip["root_quat"][frame],
                        clip["q"][frame],
                    )
                    if ghost_addresses is not None:
                        write_pose(
                            data,
                            ghost_addresses,
                            clip["ref_root_p"][frame],
                            clip["ref_root_quat"][frame],
                            clip["ref_q"][frame],
                        )
                    mujoco.mj_forward(model, data)
                    lookat = (
                        clip["ref_root_p"][frame]
                        if ghost_addresses is not None
                        else clip["root_p"][frame]
                    )
                    camera.lookat[:] = lookat
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    image = Image.fromarray(renderer.render())
                    legend = (
                        "solid: policy | blue: reference at original world position"
                        if args.trajectory
                        else "REFERENCE ONLY | no policy rollout"
                    )
                    status = (
                        " | recorded episode ended" if not clip["alive"][frame] else ""
                    )
                    if (
                        clip.get("nonfinite_terminal")
                        and frame == len(clip["alive"]) - 1
                    ):
                        status += (
                            " | terminal pose is non-finite and cannot be rendered"
                        )
                    label = f"{robot.name} | {clip['name']} | {frame / clip['fps']:.2f}s{status}"
                    lines = [args.title] if args.title else []
                    image = annotation(
                        image,
                        lines
                        + [
                            label,
                            f"terrain: {terrain['name']} | {len(terrain['boxes'])} boxes",
                            legend,
                        ],
                        args.unvalidated,
                    )
                    encoder.stdin.write(np.asarray(image).tobytes())
                    if output_index in sample_indices:
                        picks[output_index] = image.resize((480, 270))
            screenshots.append([picks[index] for index in sample_indices])
            rendered.append(
                {
                    "name": clip["name"],
                    "terrain_id": terrain["id"],
                    "terrain_name": terrain["name"],
                    "installed_atlas_boxes": len(geometry["boxes"]),
                    "source_fps": clip["fps"],
                    "source_frames": frames.tolist(),
                    "output_frames": len(frames),
                    "nonfinite_terminal": clip.get("nonfinite_terminal", False),
                }
            )
    finally:
        encoder.stdin.close()
        code = encoder.wait()
    if code:
        raise RuntimeError(f"Video encoder failed with exit code {code}")
    sheet = Image.new("RGB", (4 * 480, len(screenshots) * 270), "white")
    for row, frames in enumerate(screenshots):
        for col, frame in enumerate(frames):
            sheet.paste(frame, (col * 480, row * 270))
    sheet.save(outputs["contact_sheet"])
    sources[str(Path(robot.mjcf_path).resolve())] = sha256(robot.mjcf_path)
    metadata = {
        "robot": name,
        "mode": "recorded_policy" if args.trajectory else "reference_only",
        "unvalidated": args.unvalidated,
        "source_sha256": sources,
        "clips": rendered,
        "reference_alignment": "original global position and orientation",
        "pose_processing": "none; existing source frames selected at video timestamps",
        "video_fps": args.fps,
        "outputs": {key: str(value.resolve()) for key, value in outputs.items()},
    }
    outputs["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["outputs"], indent=2))


if __name__ == "__main__":
    main()
