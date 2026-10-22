"""Compare exact replay placement and activation recomputation on one idle GPU.

All learning dimensions stay at their defaults. Populate the entire replay first
to measure the storage/sampling workload at capacity; this profiling warmup is
not the long-run training initialization. Each candidate is a separate run.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    lease = open(f"/tmp/supermimic-{args.gpu_uuid}.lock", "a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
    if args.gpu_uuid in occupied.splitlines():
        raise RuntimeError("Selected GPU has a compute process")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, PYTHONUNBUFFERED="1",
               TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for key, folder in {"WARP_CACHE_PATH":"warp", "TORCHINDUCTOR_CACHE_DIR":"inductor",
                        "TRITON_CACHE_DIR":"triton", "CUDA_CACHE_PATH":"cuda",
                        "MPLCONFIGDIR":"matplotlib", "TMPDIR":"tmp", "XDG_CACHE_HOME":"xdg"}.items():
        path = ROOT / ".cache" / folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    python = str(ROOT / ".venv/bin/python")
    results = []
    for storage, recompute in (("cuda:0", True), ("cuda:0", False), ("cpu", False), ("cpu", True)):
        name = f"{storage.replace(':', '')}_{'recompute' if recompute else 'retained'}"
        run = args.output / name
        run.mkdir()
        command = [python, "supermimic/train.py", "--log_dir", str(run.resolve()),
                   "--replay_device", storage, "--iterations", "30", "--warmup_steps", "1000",
                   "--log_every", "1", "--ckpt_every", "30",
                   "--activation_checkpoint" if recompute else "--no-activation_checkpoint"]
        print(f"Starting {name}", flush=True)
        readings = []
        with (run / "stdout.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                raw = subprocess.check_output(["nvidia-smi", "--id", args.gpu_uuid,
                    "--query-gpu=memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"], text=True)
                readings.append([float(value.strip()) for value in raw.strip().split(",")])
                time.sleep(.5)
        result = {"name":name, "replay_device":storage, "activation_checkpoint":recompute,
                  "returncode":process.returncode, "peak_gpu_mib":max(r[0] for r in readings),
                  "peak_power_w":max(r[2] for r in readings)}
        if process.returncode == 0:
            rows = list(csv.DictReader((run / "train_log.csv").open()))
            if len(rows) != 30 or not all(math.isfinite(float(v)) for row in rows for v in row.values()):
                raise RuntimeError(f"Invalid benchmark metrics: {name}")
            times = [1 / float(row["it_per_s"]) for row in rows[10:]]
            result.update(mean_iteration_seconds=statistics.mean(times),
                          median_iteration_seconds=statistics.median(times),
                          steady_iterations_per_second=1 / statistics.mean(times))
        results.append(result)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        (run / "gpu_samples.json").write_text(json.dumps(readings) + "\n")
        print(json.dumps(result), flush=True)
        # Candidate failure is a recorded experiment result, not a silent
        # change to its batch sizes or a substitution inside the training run.
    print("Benchmark complete; choose a measured configuration before long training.", flush=True)


if __name__ == "__main__":
    main()
