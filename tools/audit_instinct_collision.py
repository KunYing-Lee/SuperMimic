"""Audit the actual MuJoCo hull scene against source contacts and K1 poses.

No source-scene transformation or corrected trajectory is written. This is an
independent gate, not a training launcher. Contact queries use broad-phase AABB
lower bounds only to eliminate shapes that provably cannot beat the distance.
"""

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from instinct_pairs import K1_XML, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("collision", type=Path)
    p.add_argument("candidate", type=Path)
    args = p.parse_args()
    cm = json.loads((args.collision / "manifest.json").read_text())
    meta = json.loads((args.candidate / "manifest.json").read_text())
    if "hulls" not in cm:
        raise ValueError("Incomplete convex decomposition")
    if (
        cm["source_sha256"] != meta["source_terrain_sha256"]
        or cm["scale"] != meta["scene_scale"]
    ):
        raise ValueError("Scene provenance mismatch")
    spec = mujoco.MjSpec.from_file(str(K1_XML.resolve()))
    for i, record in enumerate(cm["hulls"]):
        path = args.collision / record["file"]
        if sha(path) != record["sha256"]:
            raise ValueError(f"Hull changed: {path}")
        spec.add_mesh(name=f"terrain_mesh_{i}", file=str(path.resolve()))
        spec.worldbody.add_geom(
            name=f"terrain_hull_{i}",
            meshname=f"terrain_mesh_{i}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            group=3,
        )
    body = spec.worldbody.add_body(name="source_anchor", mocap=True)
    body.add_geom(
        name="source_anchor_probe",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[1e-5, 0, 0],
        contype=0,
        conaffinity=0,
        group=5,
    )
    m = spec.compile()
    d = mujoco.MjData(m)
    probe = m.geom("source_anchor_probe").id
    ids = np.array([m.geom(f"terrain_hull_{i}").id for i in range(len(cm["hulls"]))])
    mujoco.mj_forward(m, d)
    # World-space AABBs of the exact compiled convex meshes.
    lower = []
    upper = []
    for g in ids:
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = m.mesh_vert[a : a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        lower.append(v.min(0))
        upper.append(v.max(0))
    lower, upper = np.array(lower), np.array(upper)
    contacts = json.loads((args.candidate / "contacts.json").read_text())
    anchors = np.unique(
        np.array([c["anchor"] for frame in contacts for c in frame]), axis=0
    )
    distances = []
    started = time.time()
    for i, point in enumerate(anchors):
        d.mocap_pos[0] = point
        mujoco.mj_kinematics(m, d)
        bounds = np.linalg.norm(
            np.maximum(np.maximum(lower - point, point - upper), 0), axis=1
        )
        best = float("inf")
        for k in np.argsort(bounds):
            if bounds[k] > max(best, 0) + 1e-5:
                break
            distance = (
                mujoco.mj_geomDistance(m, d, int(ids[k]), probe, 1e6, None) + 1e-5
            )
            best = min(best, distance)
        distances.append(float(best))
        if i % 100 == 0:
            print(
                json.dumps({"anchors_checked": i, "anchors_total": len(anchors)}),
                flush=True,
            )
    report = {
        "collision_manifest_sha256": sha(args.collision / "manifest.json"),
        "candidate_sha256": sha(args.candidate / "candidate.npz"),
        "audit_sha256": sha(__file__),
        "hulls": len(ids),
        "anchors": len(anchors),
        "source_anchor_surface_error_max_m": float(np.max(np.abs(distances)))
        if distances
        else 0.0,
        "source_anchor_signed_distance_min_m": float(min(distances))
        if distances
        else 0.0,
        "source_anchor_signed_distance_max_m": float(max(distances))
        if distances
        else 0.0,
        "source_anchor_tolerance_m": meta["contact_tolerance_m"],
        "scene_penetration_m": 0.0,
        "self_penetration_m": 0.0,
    }
    q = np.load(args.candidate / "candidate.npz")["qpos"]
    for f, pose in enumerate(q):
        d.qpos[:] = pose
        mujoco.mj_forward(m, d)
        for c in d.contact:
            kind = "scene" if any(m.geom_bodyid[g] == 0 for g in c.geom) else "self"
            key = f"{kind}_penetration_m"
            if -c.dist > report[key]:
                report[key] = float(-c.dist)
                report[f"{kind}_worst_frame"] = f
                report[f"{kind}_worst_geoms"] = [m.geom(int(g)).name for g in c.geom]
    report.update(frames=len(q), seconds=time.time() - started, training_accepted=False)
    report["collision_passed"] = bool(
        meta["solver_passed"]
        and report["source_anchor_surface_error_max_m"] <= meta["contact_tolerance_m"]
        and max(report["scene_penetration_m"], report["self_penetration_m"]) <= 0.001001
    )
    (args.candidate / "collision_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
