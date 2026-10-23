"""Strict contact-constrained OmniRetarget adaptation for released G1 pairs.

Uses the pinned project's interaction-mesh SQP directly. Its public wrapper's
constraint-dropping retries and XYZ correction are deliberately not invoked.
Scene anchors are geometric annotations (not measured forces). They constrain
the distance to the corresponding K1 collision surface, supporting rolling.
No output is accepted unless every frame is feasible AND converged.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
import open3d as o3d
import trimesh

from instinct_pairs import (
    CONTACT_BODIES,
    G1_XML,
    K1_XML,
    LINKS,
    collision_samples,
    load_qpos,
    pairs,
    resample_source,
    sha,
    world_samples,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omniretargeting.retargeting import GenericInteractionRetargeter
from omniretargeting.utils import sample_mujoco_geom_local_points

OMNI_REVISION = "6518c36a2be9dab2538cc1b95840da00fbd6d4b7"


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class MeshQuery:
    def __init__(self, mesh):
        self.mesh = mesh
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        self.scene.add_triangles(
            o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
            o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)),
        )

    def closest(self, points):
        result = self.scene.compute_closest_points(
            o3d.core.Tensor(np.asarray(points, np.float32)), nthreads=1
        )
        ids = result["primitive_ids"].numpy()
        p = trimesh.triangles.closest_point(self.mesh.triangles[ids], points)
        normals = self.mesh.face_normals[ids]
        delta = points - p
        return p, np.linalg.norm(delta, axis=1), np.sum(delta * normals, axis=1)


class ContactRetargeter(GenericInteractionRetargeter):
    def __init__(self, *args, contact_tolerance=0.002, **kwargs):
        self.active_contacts = []
        self.contact_tolerance = contact_tolerance
        self.previous = None
        self.dt = None
        super().__init__(*args, **kwargs)
        # trimesh uses its own Generator; np.random.seed does not seed this.
        self.terrain_points, _ = trimesh.sample.sample_surface(
            self.terrain_mesh, self.terrain_sample_points, seed=0
        )
        # Upstream substitutes centers for mesh colliders; include true hull
        # surface samples here for the K1 feet and head.
        self._terrain_collision_geoms = []
        for g in range(self.robot_model.ngeom):
            if not (
                self.robot_model.geom_contype[g] or self.robot_model.geom_conaffinity[g]
            ):
                continue
            if self.robot_model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = self.robot_model.geom_dataid[g]
                start, count = (
                    self.robot_model.mesh_vertadr[mid],
                    self.robot_model.mesh_vertnum[mid],
                )
                hull = trimesh.convex.convex_hull(
                    self.robot_model.mesh_vert[start : start + count]
                )
                points = np.vstack([hull.vertices, hull.triangles_center])
            else:
                points = sample_mujoco_geom_local_points(
                    self.robot_model.geom_type[g], self.robot_model.geom_size[g]
                )
            self._terrain_collision_geoms.append((g, points))
        self.probes = [
            self.robot_model.geom(f"contact_anchor_{i}").id
            for i in range(len(CONTACT_BODIES))
        ]
        self.probe_mocap_ids = [
            self.robot_model.body_mocapid[self.robot_model.geom_bodyid[g]]
            for g in self.probes
        ]
        from convex_distance import ConvexDistance

        self.contact_oracle = ConvexDistance(self.robot_model, extra_geoms=self.probes)
        self.target_groups = {}
        for source, bodies in CONTACT_BODIES.items():
            self.target_groups[source] = [
                g
                for g in range(self.robot_model.ngeom)
                if self.robot_model.body(self.robot_model.geom_bodyid[g]).name in bodies
                and (
                    self.robot_model.geom_contype[g]
                    or self.robot_model.geom_conaffinity[g]
                )
            ]
            if not self.target_groups[source]:
                raise ValueError(f"No colliders for {source}")

    def _closest_points_on_terrain(self, points):
        result = self._terrain_scene.compute_closest_points(
            o3d.core.Tensor(np.asarray(points, np.float32)), nthreads=1
        )
        ids = result["primitive_ids"].numpy()
        p = trimesh.triangles.closest_point(self.terrain_mesh.triangles[ids], points)
        return p, np.linalg.norm(points - p, axis=1), ids

    def set_contacts(self, contacts):
        self.active_contacts = contacts
        for i, record in enumerate(contacts):
            self.robot_data.mocap_pos[self.probe_mocap_ids[i]] = record["anchor"]

    def contact_distances(self):
        rows, gaps = [], []
        self.contact_oracle.update(self.robot_data)
        for i, record in enumerate(self.active_contacts):
            probe = self.probes[i]
            distances = [
                (self.contact_oracle.distance(g, probe), g)
                for g in self.target_groups[record["body"]]
            ]
            (raw, point, _, normal), geom = min(distances, key=lambda item: item[0][0])
            distance = raw + self.robot_model.geom_size[probe, 0]
            jp, jr = (
                np.zeros((3, self.robot_model.nv)),
                np.zeros((3, self.robot_model.nv)),
            )
            mujoco.mj_jac(
                self.robot_model,
                self.robot_data,
                jp,
                jr,
                point,
                self.robot_model.geom_bodyid[geom],
            )
            gradient = -normal @ jp
            gradient = gradient[self.dof_indices]
            rows.extend(
                [
                    (gradient, -self.contact_tolerance - distance),
                    (-gradient, distance - self.contact_tolerance),
                ]
            )
            gaps.append(float(distance))
        return rows, gaps

    def _optimize_configuration(self, *args, **kwargs):
        from strict_omni_sqp import solve

        return solve(self, *args, **kwargs)


def make_target():
    spec = mujoco.MjSpec.from_file(str(K1_XML.resolve()))
    for i in range(len(CONTACT_BODIES)):
        body = spec.worldbody.add_body(name=f"contact_body_{i}", mocap=True)
        body.add_geom(
            name=f"contact_anchor_{i}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[1e-5, 0, 0],
            contype=0,
            conaffinity=0,
            group=5,
        )
    return spec.compile()


def height(model):
    """Collision-surface standing height in the named model's zero joint pose."""
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    p = world_samples(data, collision_samples(model))
    return float(np.ptp(p[:, 2]))


def annotate(source_model, qpos, mesh, scale, threshold):
    query = MeshQuery(mesh)
    data = mujoco.MjData(source_model)
    samples = collision_samples(source_model)
    point_bodies = np.concatenate(
        [np.full(len(p), source_model.geom_bodyid[g]) for g, p in samples]
    )
    point_geoms = np.concatenate([np.full(len(p), g) for g, p in samples])
    # The pinned G1 model attaches its head collider to torso_link. Separate
    # that explicit asset element so head contacts cannot become torso contacts.
    if source_model.geom_bodyid[4] != source_model.body(
        "torso_link"
    ).id or not np.allclose(source_model.geom_pos[4], [0.01, 0, 0.415]):
        raise ValueError("Pinned G1 head collision element changed")
    groups = {
        name: np.flatnonzero(point_bodies == source_model.body(name).id)
        for name in CONTACT_BODIES
        if name != "head_surface"
    }
    groups["torso_link"] = groups["torso_link"][point_geoms[groups["torso_link"]] != 4]
    groups["head_surface"] = np.flatnonzero(point_geoms == 4)
    all_contacts, targets, source_gaps = [], [], []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(source_model, data)
        points = world_samples(data, samples) * scale
        anchors, distance, signed = query.closest(points)
        frame = []
        for name, indices in groups.items():
            # One nearest surface anchor per semantic body. This is a geometric
            # contact observation, not a claim of sticking or load bearing.
            if not len(indices):
                continue
            k = indices[np.argmin(distance[indices])]
            if distance[k] <= threshold:
                frame.append(
                    {
                        "body": name,
                        "anchor": anchors[k].tolist(),
                        "source_surface_point": points[k].tolist(),
                        "source_gap": float(distance[k]),
                    }
                )
        all_contacts.append(frame)
        source_gaps.append(float(signed.min()))
        targets.append(
            [data.xpos[source_model.body(name).id] * scale for name in LINKS]
        )
    return all_contacts, np.asarray(targets), source_gaps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--max-iterations", type=int, default=200)
    p.add_argument("--output-fps", type=float, default=100.0)
    p.add_argument(
        "--scene-scale",
        type=float,
        help="Default: measured K1/G1 standing collision height ratio",
    )
    p.add_argument("--contact-distance", type=float, default=0.02)
    p.add_argument("--contact-tolerance", type=float, default=0.002)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    (args.output / "INCOMPLETE").write_text("Not accepted for training\n")
    revision = subprocess.check_output(
        ["git", "-C", ".cache/omniretargeting", "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != OMNI_REVISION:
        raise ValueError("Unexpected OmniRetarget revision")
    np.random.seed(0)
    pair = pairs()[args.pair]
    source = mujoco.MjModel.from_xml_path(str(G1_XML))
    target = make_target()
    q_source, source_fps = load_qpos(pair["motion"], source)
    original_source_frames = len(q_source)
    fps = args.output_fps
    q_source = resample_source(q_source, source_fps, fps)
    full_frames = len(q_source)
    if args.max_frames is not None:
        q_source = q_source[: args.max_frames]
    scale = (
        args.scene_scale
        if args.scene_scale is not None
        else height(target) / height(source)
    )
    mesh = trimesh.load(pair["terrain"], force="mesh")
    mesh.vertices *= scale
    # Fix only face orientation, preserving vertex coordinates and topology.
    trimesh.repair.fix_winding(mesh)
    mesh.export(args.output / "terrain.ply")
    contacts, targets, source_gaps = annotate(
        source, q_source, mesh, scale, args.contact_distance
    )
    (args.output / "contacts.json").write_text(json.dumps(contacts, indent=2))
    manifest = {
        "pair": {k: str(v) if isinstance(v, Path) else v for k, v in pair.items()},
        "omni_revision": revision,
        "source_robot_sha256": sha(G1_XML),
        "target_robot_sha256": sha(K1_XML),
        "source_motion_sha256": sha(pair["motion"]),
        "source_terrain_sha256": sha(pair["terrain"]),
        "tool_sha256": sha(__file__),
        "helpers_sha256": sha(Path(__file__).with_name("instinct_pairs.py")),
        "optimizer_sha256": sha(Path(__file__).with_name("strict_omni_sqp.py")),
        "distance_oracle_sha256": sha(Path(__file__).with_name("convex_distance.py")),
        "scene_scale": scale,
        "source_frames": full_frames,
        "requested_frames": len(q_source),
        "source_fps": source_fps,
        "retarget_fps": fps,
        "original_source_frames": original_source_frames,
        "source_duration_s": (original_source_frames - 1) / source_fps,
        "retarget_duration_s": (full_frames - 1) / fps,
        "contact_detection_distance_m": args.contact_distance,
        "terrain_sample_seed": 0,
        "mujoco_version": mujoco.__version__,
        "contact_tolerance_m": args.contact_tolerance,
        "penetration_tolerance_m": 0.0,
        "max_iterations": args.max_iterations,
        "postprocessing": [],
        "fallback": False,
        "source_min_sample_signed_distance_m": min(source_gaps),
        "contact_frame_count": sum(bool(x) for x in contacts),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    engine = ContactRetargeter(
        target,
        mujoco.MjData(target),
        mesh,
        LINKS,
        height(target),
        source_target_names=list(LINKS),
        hard_penetration_constraint=True,
        penetration_tolerance=0.0,
        terrain_sample_points=100,
        solver_diagnostics=False,
        contact_tolerance=args.contact_tolerance,
    )
    engine.dt = 1 / fps
    np.save(args.output / "interaction_terrain_points.npy", engine.terrain_points)
    q = target.qpos0.copy()
    q[:3] = q_source[0, :3] * scale
    q[3:7] = q_source[0, 3:7]
    output, diagnostics = [], []

    def save_progress():
        temporary = args.output / "candidate.tmp.npz"
        np.savez_compressed(temporary, qpos=np.asarray(output), framerate=fps)
        temporary.replace(args.output / "candidate.npz")
        (args.output / "solver.json").write_text(
            json.dumps(diagnostics, default=serial, indent=2)
        )
        manifest.update(
            completed_frames=len(output), solver_passed=False, training_accepted=False
        )
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for f, (positions, contact) in enumerate(zip(targets, contacts)):
        engine.set_contacts(contact)
        engine.previous = output[-1] if output else None
        try:
            q = engine.retarget_frame(
                positions,
                q,
                max_iter=args.max_iterations,
                q_last=engine.previous,
                target_base_orientation=q_source[f, 3:7],
            )
        except (RuntimeError, ValueError) as error:
            manifest.update(
                failed_frame=f, failure_reason=f"{type(error).__name__}: {error}"
            )
            break
        diag = dict(engine.last_solve_diagnostics)
        diag["frame"] = f
        diag["contact_gaps_m"] = engine.contact_distances()[1]
        diagnostics.append(diag)
        output.append(q.copy())
        if f % 25 == 0:
            save_progress()
            print(
                json.dumps(
                    {
                        "frame": f,
                        "converged": diag["converged"],
                        "violation": diag["max_hard_violation"],
                    }
                ),
                flush=True,
            )
        if not diag["success"] or not diag["converged"]:
            manifest["failed_frame"] = f
            manifest["failure_reason"] = diag["failure_reason"] or "not_stationary"
            break
    np.savez_compressed(
        args.output / "candidate.npz", qpos=np.asarray(output), framerate=fps
    )
    (args.output / "solver.json").write_text(
        json.dumps(diagnostics, default=serial, indent=2)
    )
    manifest["completed_frames"] = len(output)
    manifest["solver_passed"] = (
        "failed_frame" not in manifest and len(output) == full_frames
    )
    manifest["training_accepted"] = (
        False  # independent geometry and dynamics gates remain mandatory
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)
    if "failed_frame" in manifest:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
