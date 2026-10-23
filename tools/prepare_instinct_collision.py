"""Build a CoACD collision representation of one released scene.

No hull translation, top alignment, smoothing, or root offset is permitted.
This tool never marks a scene accepted based only on successful decomposition.
"""

import argparse
import json
from pathlib import Path
import time

import coacd
import numpy as np
import trimesh
from instinct_pairs import pairs, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", type=int, required=True)
    p.add_argument("--scale", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.005)
    p.add_argument("--preprocess-resolution", type=int, default=200)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "INCOMPLETE").write_text(
        "Needs source/robot contact and sensing validation\n"
    )
    pair = pairs()[args.pair]
    mesh = trimesh.load(pair["terrain"], force="mesh")
    mesh.vertices *= args.scale
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh, multibody=True)
    options = dict(
        threshold=args.threshold,
        preprocess_mode="auto",
        preprocess_resolution=args.preprocess_resolution,
        resolution=3000,
        mcts_nodes=20,
        mcts_iterations=150,
        mcts_max_depth=3,
        merge=True,
        decimate=False,
        max_convex_hull=-1,
        extrude=False,
        seed=0,
        real_metric=True,
    )
    manifest = {
        "source": str(pair["terrain"]),
        "source_sha256": sha(pair["terrain"]),
        "scale": args.scale,
        "options": options,
        "transformations": ["uniform scene scale", "face winding and normals"],
        "source_watertight": bool(mesh.is_watertight),
        "accepted": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    started = time.perf_counter()
    result = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces, np.int32)),
        **options,
    )
    # Preserve every part before validation, including rejected parts.
    np.savez_compressed(
        args.output / "raw_parts.npz",
        **{
            f"{kind}_{i}": value
            for i, part in enumerate(result)
            for kind, value in zip(("vertices", "faces"), part)
        },
    )
    records = []
    for i, (vertices, faces) in enumerate(result):
        # MuJoCo uses the convex hull of the vertices, independent of supplied
        # face winding. Construct that same polytope; do not move any vertex.
        try:
            hull = trimesh.convex.convex_hull(np.asarray(vertices), qhull_options="Qt")
            if hull.volume <= 1e-12:
                raise ValueError("zero-volume polytope")
        except Exception as error:
            manifest.update(
                failed_part=i, failure=str(error), seconds=time.perf_counter() - started
            )
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
            raise ValueError(
                f"Degenerate collision hull {i}; not silently skipped"
            ) from error
        output = args.output / f"hull_{i:04d}.stl"
        hull.export(output)
        records.append(
            {
                "file": output.name,
                "sha256": sha(output),
                "vertices": len(vertices),
                "faces": len(faces),
            }
        )
    manifest.update(hulls=records, seconds=time.perf_counter() - started)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {"hulls": len(records), "seconds": manifest["seconds"], "accepted": False}
        )
    )


if __name__ == "__main__":
    main()
