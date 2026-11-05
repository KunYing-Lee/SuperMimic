"""Run provenance and complete restart state; no model-only checkpoint is a resume."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

SCHEMA_VERSION = 2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest(repo: str, args, motion, robot_path: str | Path, task_sources=(), task_assets=()) -> dict:
    """Hash core code, motion/robot assets, and the task's declared files."""
    root = Path(repo)
    source = {str(p.relative_to(root)): sha256(p) for p in sorted((root / "supermimic").rglob("*.py"))}
    source.update({str(Path(p).resolve()): sha256(Path(p)) for p in task_sources})
    robot_assets = [p for p in Path(robot_path).parent.rglob("*") if p.is_file() and p.suffix.lower() in {".xml", ".urdf", ".stl", ".obj", ".png", ".jpg"}]
    assets = {str(Path(p).resolve()): sha256(Path(p)) for p in [*motion.files, *getattr(motion, "metadata_files", []), *robot_assets, *task_assets]}
    versions = {}
    for package in ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    git_commit, git_dirty = None, None
    if (root / ".git").exists():
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip())
    return {
        "schema_version": SCHEMA_VERSION,
        "repository_root": str(root.resolve()),
        "args": vars(args),
        "source_sha256": source,
        "asset_sha256": assets,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "packages": versions,
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device),
        "gpu_memory_bytes": torch.cuda.get_device_properties(args.device).total_memory,
        "dataset": {"clips": motion.names, "frames_50hz": motion.num_frames, "control_dt": 0.02},
        "implementation_choices": {
            "termination_head": "MLP(256,128,1), ELU; invariant state + invariant reference state + future goal; BCE",
            "state_error": "squared local position/velocity/joint errors and SO(3) logarithmic orientation error",
            "goal_features": "73 floats per future frame; robot-body-local relative position, rotation6d, velocities, joints",
            "invariant_state": "height + body-frame gravity3 + body linear/angular velocity + joints; 68 dimensions",
            "reward_alignment": "relative Euler yaw via atan2(R10,R00); exact relative-forward-vertical chart singularity is not regularized",
            "geometry_precision": "3x3 matrix and matrix-vector products use scalar FP32 multiply/add independent of TF32 MLP matmuls",
            "normalization": "running mean/variance inputs and dynamics increment parameterization",
            "activation_checkpointing": "exact recomputation, preserving RNG state",
        },
    }


def validate_resume(checkpoint: dict, current: dict, allow_args=(), allow_source: bool = False):
    """A resume continues the saved run: the same code, packages, assets and
    training configuration. A deliberate fine-tune may allow other code
    (allow_source) and the arguments in allow_args to differ."""
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("This checkpoint lacks complete restart state; use a schema-v2 full last.pt checkpoint")
    previous = checkpoint["manifest"]
    for field in ("packages", "cuda") if allow_source else ("source_sha256", "packages", "cuda"):
        if previous[field] != current[field]:
            raise ValueError(f"Resume {field} differs from the saved run")
    # Paths may move together with a repository; preserve file identities and hashes.
    def asset_id(path, record):
        p = Path(path)
        repo = Path(record["repository_root"])
        if record["args"].get("motion_dir") is not None:
            motion_dir = Path(record["args"]["motion_dir"]).resolve()
            if p.is_relative_to(motion_dir):
                return "motion/" + str(p.relative_to(motion_dir))
        if p.is_relative_to(repo):
            return str(p.relative_to(repo))
        return str(p)
    old_assets = sorted((asset_id(p, previous), digest) for p, digest in previous["asset_sha256"].items())
    new_assets = sorted((asset_id(p, current), digest) for p, digest in current["asset_sha256"].items())
    if allow_source:  # external code may sit elsewhere: compare asset contents alone
        old_assets, new_assets = sorted(d for _, d in old_assets), sorted(d for _, d in new_assets)
    if old_assets != new_assets:
        raise ValueError("Resume motion/robot/task asset hashes differ from the saved run")
    # arguments that change neither the run's data nor its gradients (activation checkpointing recomputes exactly)
    operational = {"resume", "log_dir", "motion_dir", "iterations", "ckpt_every", "log_every", "fine_tune",
                   "activation_checkpoint", *allow_args}
    old_args = {k: v for k, v in previous["args"].items() if k not in operational}
    new_args = {k: v for k, v in current["args"].items() if k not in operational}
    if old_args != new_args:
        changed = sorted(k for k in old_args.keys() | new_args.keys() if old_args.get(k) != new_args.get(k))
        raise ValueError(f"Resume changes training configuration: {changed}")


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def write_json(path: str, value: dict):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(target)
