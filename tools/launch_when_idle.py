"""Reserve one idle requested NVIDIA GPU, validate at full training sizes, then start a fresh run.

No existing process is interrupted. Failed validation stops the launch. The advisory
lock coordinates launches of this script; it is not a cluster scheduler reservation.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def idle_gpus(model):
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
    occupied = set(processes.splitlines())
    candidates = []
    for row in csv.reader(io.StringIO(raw)):
        index, uuid, name, memory, utilization = [v.strip() for v in row]
        if model in name and uuid not in occupied and int(memory) <= 64 and int(utilization) == 0:
            candidates.append({"index": int(index), "uuid": uuid, "name": name})
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--gpu-model", required=True, help="Exact model substring, e.g. A800 or RTX 5090")
    parser.add_argument("--replay-device", default="cpu", choices=["cpu", "cuda:0"])
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 1 or Path(args.run_name).name != args.run_name:
        parser.error("poll-seconds must be positive and run-name must be a single directory name")
    run = args.runs_root.resolve() / args.run_name
    smoke = run.with_name(run.name + "_preflight")
    if run.exists() or smoke.exists():
        raise FileExistsError("Use a fresh run name: training and validation directories must not exist")
    previous = set()
    print("Waiting for a matching GPU with no compute process, <=64 MiB usage and zero utilization.", flush=True)
    while True:
        candidates = idle_gpus(args.gpu_model)
        available = {g["uuid"] for g in candidates}
        for gpu in candidates:
            if gpu["uuid"] not in previous:
                continue
            lease = open(f"/tmp/supermimic-{gpu['uuid']}.lock", "a+")
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close()
                continue
            if gpu["uuid"] not in {g["uuid"] for g in idle_gpus(args.gpu_model)}:
                lease.close()
                continue
            break
        else:
            if not args.wait and not candidates:
                raise RuntimeError("No matching idle GPU available; no training was launched")
            previous = available
            time.sleep(args.poll_seconds)
            continue
        break

    env = os.environ.copy()
    for cache in ("tmp", "triton", "cuda", "matplotlib"):
        (ROOT / ".cache" / cache).mkdir(parents=True, exist_ok=True)
    env.update(CUDA_VISIBLE_DEVICES=gpu["uuid"], PYTHONUNBUFFERED="1",
               WARP_CACHE_PATH=str(ROOT / ".cache/warp"),
               TORCHINDUCTOR_CACHE_DIR=str(ROOT / ".cache/inductor"),
               TRITON_CACHE_DIR=str(ROOT / ".cache/triton"),
               CUDA_CACHE_PATH=str(ROOT / ".cache/cuda"),
               XDG_CACHE_HOME=str(ROOT / ".cache"),
               MPLCONFIGDIR=str(ROOT / ".cache/matplotlib"),
               TMPDIR=str(ROOT / ".cache/tmp"))
    python = str(ROOT / ".venv/bin/python")
    smoke.mkdir(parents=True)
    print(f"Selected GPU {gpu['index']} {gpu['uuid']}; full-size validation begins.", flush=True)
    (smoke / "gpu.json").write_text(json.dumps(gpu, indent=2) + "\n")

    def execute(arguments, logfile):
        print("Running:", " ".join(arguments), flush=True)
        with logfile.open("w") as log:
            subprocess.run(arguments, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)

    execute([python, "tools/test_env.py", "--device", "cuda:0", "--num_envs", "4", "--steps", "30"], smoke / "backend.log")
    execute([python, "tools/test_dataset_reset.py", "--device", "cuda:0"], smoke / "dataset_reset.log")
    common = [python, "supermimic/train.py", "--device", "cuda:0",
              "--log_dir", str(smoke), "--replay_device", args.replay_device, "--log_every", "1", "--ckpt_every", "2"]
    # Only duration/log cadence differ. Environment count, replay capacity,
    # batches, network widths and 32-step BPTT all use the long-run defaults.
    execute(common + ["--iterations", "2"], smoke / "train.log")
    execute(common + ["--iterations", "3", "--resume", str(smoke / "last.pt")], smoke / "resume.log")
    execute([python, "supermimic/evaluate.py", "--device", "cuda:0", "--run_dir", str(smoke),
             "--ckpt", str(smoke / "ckpt_000003.pt")], smoke / "evaluate.log")
    run.mkdir()
    (run / "gpu.json").write_text(json.dumps(gpu, indent=2) + "\n")
    print(f"Validation passed. Starting fresh 320000-iteration training in {run}", flush=True)
    execute([python, "supermimic/train.py", "--device", "cuda:0", "--log_dir", str(run), "--replay_device", args.replay_device], run / "stdout.log")
    print("Training completed.", flush=True)


if __name__ == "__main__":
    main()
