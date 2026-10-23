"""Independent dense reference audit; never promotes a failed retarget result."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp
import trimesh

from instinct_pairs import (
    K1_XML,
    CONTACT_BODIES,
    anchor_surface_distance,
    collision_samples,
    world_samples,
    sha,
)
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.robot import K1


def sample(q, fps, hz):
    if len(q) == 1:
        return q, np.array([0.0])
    source = np.arange(len(q)) / fps
    times = np.arange(int(np.floor(source[-1] * hz)) + 1) / hz
    result = np.stack(
        [np.interp(times, source, q[:, j]) for j in range(q.shape[1])], axis=1
    )
    result[:, 3:7] = Slerp(source, Rotation.from_quat(q[:, [4, 5, 6, 3]]))(
        times
    ).as_quat()[:, [3, 0, 1, 2]]
    return result, times


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate", type=Path)
    p.add_argument("--spacing", type=float, default=0.01)
    args = p.parse_args()
    root = args.candidate
    meta = json.loads((root / "manifest.json").read_text())
    a = np.load(root / "candidate.npz", allow_pickle=False)
    original, fps = a["qpos"], float(a["framerate"])
    spec = mujoco.MjSpec.from_file(str(K1_XML.resolve()))
    for i in range(len(CONTACT_BODIES)):
        body = spec.worldbody.add_body(name=f"audit_anchor_body_{i}", mocap=True)
        body.add_geom(
            name=f"audit_anchor_{i}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[1e-5, 0, 0],
            contype=0,
            conaffinity=0,
            group=5,
        )
    m = spec.compile()
    d = mujoco.MjData(m)
    probes = [m.geom(f"audit_anchor_{i}").id for i in range(len(CONTACT_BODIES))]
    mocap = [m.body_mocapid[m.geom_bodyid[g]] for g in probes]
    groups = {
        name: [
            g
            for g in range(m.ngeom)
            if m.body(m.geom_bodyid[g]).name in names
            and (m.geom_contype[g] or m.geom_conaffinity[g])
        ]
        for name, names in CONTACT_BODIES.items()
    }
    annotations = json.loads((root / "contacts.json").read_text())
    samples = collision_samples(m, spacing=args.spacing)
    point_geoms = np.concatenate([np.full(len(p), g) for g, p in samples])
    mesh = trimesh.load(root / "terrain.ply", force="mesh")
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)),
    )
    report = {
        "candidate_sha256": sha(root / "candidate.npz"),
        "terrain_sha256": sha(root / "terrain.ply"),
        "robot_sha256": sha(K1_XML),
        "audit_sha256": sha(__file__),
        "mujoco_version": mujoco.__version__,
        "surface_spacing_m": args.spacing,
        "surface_points": len(point_geoms),
        "reference_only": True,
        "solver_passed": meta["solver_passed"],
        "rates": {},
    }
    for hz in dict.fromkeys([fps, 50.0, 100.0]):
        q, times = sample(original, fps, hz)
        worst = {
            "terrain_signed_distance_m": 0.0,
            "self_distance_m": 0.0,
            "joint_limit_violation_rad": 0.0,
            "contact_gap_m": 0.0,
        }
        for f, pose in enumerate(q):
            d.qpos[:] = pose
            contacts = annotations[
                min(int(round(times[f] * fps)), len(annotations) - 1)
            ]
            for i, contact in enumerate(contacts):
                d.mocap_pos[mocap[i]] = contact["anchor"]
            mujoco.mj_forward(m, d)
            for i, contact in enumerate(contacts):
                gap, _ = anchor_surface_distance(
                    m, d, groups[contact["body"]], probes[i]
                )
                worst["contact_gap_m"] = max(worst["contact_gap_m"], abs(gap))
            points = world_samples(d, samples)
            hit = scene.compute_closest_points(
                o3d.core.Tensor(np.asarray(points, np.float32)), nthreads=1
            )
            ids = hit["primitive_ids"].numpy()
            surface = trimesh.triangles.closest_point(mesh.triangles[ids], points)
            normals = mesh.face_normals[ids]
            signed = np.einsum("ij,ij->i", points - surface, normals)
            # Report all nearest-surface signed distances, including deep ones;
            # do not share the optimizer's 0.5 m proximity exclusion.
            k = int(np.argmin(signed))
            if signed[k] < worst["terrain_signed_distance_m"]:
                worst.update(
                    terrain_signed_distance_m=float(signed[k]),
                    terrain_frame=f,
                    terrain_time_s=float(times[f]),
                    terrain_geom=m.geom(int(point_geoms[k])).name,
                    terrain_world_point=points[k].tolist(),
                )
            for contact in d.contact:
                if contact.dist < worst["self_distance_m"]:
                    worst.update(
                        self_distance_m=float(contact.dist),
                        self_frame=f,
                        self_geoms=[m.geom(int(g)).name for g in contact.geom],
                    )
            for j in range(1, m.njnt):
                if m.jnt_limited[j]:
                    value = pose[m.jnt_qposadr[j]]
                    lo, hi = m.jnt_range[j]
                    worst["joint_limit_violation_rad"] = max(
                        worst["joint_limit_violation_rad"],
                        float(lo - value),
                        float(value - hi),
                    )
        speed_ratio = 0.0
        if len(q) > 1:
            for j in range(1, m.njnt):
                velocity = np.max(np.abs(np.diff(q[:, m.jnt_qposadr[j]]))) * hz
                speed_ratio = max(
                    speed_ratio,
                    float(velocity / K1.motor(m.joint(j).name).velocity_limit),
                )
        worst["max_joint_speed_limit_ratio"] = speed_ratio
        worst["frames"] = len(q)
        worst["passed"] = bool(
            worst["terrain_signed_distance_m"] >= -0.001001
            and worst["self_distance_m"] >= -0.001001
            and worst["joint_limit_violation_rad"] <= 1e-6
            and speed_ratio <= 1.000001
            and worst["contact_gap_m"] <= meta["contact_tolerance_m"] + 1e-6
        )
        report["rates"][str(hz)] = worst
    report["geometry_passed"] = bool(
        meta["solver_passed"] and all(r["passed"] for r in report["rates"].values())
    )
    report["training_accepted"] = False
    (root / "independent_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
