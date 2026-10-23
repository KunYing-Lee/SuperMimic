"""Run all released pairs with an immutable source snapshot; never starts training."""

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    snapshot = args.output / "source"
    snapshot.mkdir()
    names = [
        "retarget_instinct_k1.py",
        "strict_omni_sqp.py",
        "convex_distance.py",
        "instinct_pairs.py",
    ]
    for name in names:
        shutil.copy2(root / "tools" / name, snapshot / name)
    hashes = {
        name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
        for name in names
    }
    (args.output / "source_hashes.json").write_text(json.dumps(hashes, indent=2))
    freeze = (
        "\n".join(
            sorted(
                f"{d.metadata['Name']}=={d.version}"
                for d in importlib.metadata.distributions()
            )
        )
        + "\n"
    )
    (args.output / "environment.txt").write_text(freeze)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=str(root),
    )

    def run(i):
        target = args.output / f"pair_{i:02d}"
        cmd = [
            sys.executable,
            "-u",
            str(snapshot / "retarget_instinct_k1.py"),
            "--pair",
            str(i),
            "--output",
            str(target),
        ]
        with (args.output / f"pair_{i:02d}.log").open("w") as f:
            result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        manifest = target / "manifest.json"
        meta = json.loads(manifest.read_text()) if manifest.exists() else {}
        record = {
            "pair": i,
            "exit_code": result.returncode,
            "command": cmd,
            "solver_passed": meta.get("solver_passed", False),
            "completed_frames": meta.get("completed_frames", 0),
            "failure": meta.get("failure_reason"),
            "training_accepted": False,
        }
        if result.returncode and not record["failure"]:
            record["failure"] = "process failed; see retained log"
        (args.output / f"pair_{i:02d}_status.json").write_text(
            json.dumps(record, indent=2)
        )
        print(json.dumps(record), flush=True)
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(run, range(10)))
    (args.output / "batch_status.json").write_text(json.dumps(records, indent=2))
    if not all(r["solver_passed"] for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
