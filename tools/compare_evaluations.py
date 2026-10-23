"""Compare fixed-library evaluations without aligning or repairing trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

REFERENCE_KEYS = (
    "names",
    "clip_start",
    "clip_len",
    "ref_root_p",
    "ref_root_R",
    "ref_q",
)


def validate(data: dict, result: dict) -> dict:
    """Reject invalid live samples; report nonfinite post-episode samples explicitly."""
    names = data["names"].tolist()
    if (
        not names
        or len(set(names)) != len(names)
        or set(result["per_clip"]) != set(names)
    ):
        raise ValueError("Motion names must be unique and match result keys")
    failed = result["failed"]
    if len(set(failed)) != len(failed) or not set(failed) <= set(names):
        raise ValueError("Invalid failed motion list")
    if result["n_failed"] != len(failed):
        raise ValueError("Failure count disagrees with failed motion list")
    if not all(np.isfinite(v) for v in result["per_clip"].values()):
        raise ValueError("Nonfinite return")
    alive = data["alive"]
    if alive.dtype != np.bool_ or alive.ndim != 2 or alive.shape[1] != len(names):
        raise ValueError("Expected boolean [time, motion] alive mask")
    if not alive[0].all() or np.any(np.diff(alive.astype(int), axis=0) > 0):
        raise ValueError("Each episode must be a nonempty contiguous alive prefix")
    if data["f"].shape != alive.shape or data["f"].dtype.kind not in "iu":
        raise ValueError("Invalid reference frame indices")
    for key in ("ref_root_p", "ref_root_R", "ref_q"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Nonfinite reference: {key}")
    if data["clip_start"].shape != (len(names),) or data["clip_len"].shape != (
        len(names),
    ):
        raise ValueError("Invalid clip metadata")
    for c, name in enumerate(names):
        start, length = int(data["clip_start"][c]), int(data["clip_len"][c])
        count = int(alive[:, c].sum())
        if length < 2 or start < 0 or start + length > len(data["ref_q"]):
            raise ValueError(f"Invalid reference range: {name}")
        if count > length - 1 or (name not in failed and count != length - 1):
            raise ValueError(f"Incomplete recording or inconsistent completion: {name}")
        if not np.array_equal(data["f"][:count, c], np.arange(start, start + count)):
            raise ValueError(
                f"Evaluation must start at the first frame and advance by one: {name}"
            )
    diagnostics = {}
    for key, width in (("root_p", 3), ("root_quat", 4), ("q", data["ref_q"].shape[1])):
        values = data[key]
        if values.shape != (*alive.shape, width):
            raise ValueError(f"Invalid trajectory shape: {key}")
        bad = ~np.isfinite(values).all(axis=-1)
        if (bad & alive).any():
            raise ValueError(f"Nonfinite live trajectory: {key}")
        diagnostics[key] = {
            "post_episode_nonfinite_frames": int(bad.sum()),
            "motions": [names[c] for c in np.flatnonzero(bad.any(axis=0))],
        }
    return diagnostics


def compare(
    baseline: dict, candidate: dict, results: list[dict], base_clips: int | None = None
):
    diagnostics = [validate(d, r) for d, r in zip((baseline, candidate), results)]
    for key in REFERENCE_KEYS:
        if not np.array_equal(baseline[key], candidate[key]):
            raise ValueError(f"Reference libraries differ: {key}")
    names = baseline["names"].tolist()
    if base_clips is not None and not 0 < base_clips < len(names):
        raise ValueError("base-clips must split the library into two nonempty groups")
    rows = []
    for c, name in enumerate(names):
        row = {
            "motion": name,
            "reference_seconds": float((baseline["clip_len"][c] - 1) / 50),
        }
        errors = []
        for label, data, result in zip(
            ("baseline", "candidate"), (baseline, candidate), results
        ):
            alive = data["alive"][:, c]
            frames = data["f"][alive, c]
            xy = np.linalg.norm(
                data["root_p"][alive, c, :2] - data["ref_root_p"][frames, :2], axis=-1
            )
            joint = data["q"][alive, c] - data["ref_q"][frames]
            errors.append((xy, joint))
            metrics = {
                "return": result["per_clip"][name],
                "completed": name not in result["failed"],
                "alive_seconds": len(xy) / 50,
                "xy_mean": float(xy.mean()),
                "xy_p95": float(np.percentile(xy, 95)),
                "xy_max": float(xy.max()),
                "xy_end": float(xy[-1]),
                "xy_max_time": float(np.argmax(xy) / 50),
                "joint_rmse_rad": float(
                    np.sqrt(np.mean(joint.astype(np.float64) ** 2))
                ),
            }
            row.update({f"{key}_{label}": value for key, value in metrics.items()})
        n = min(len(e[0]) for e in errors)
        row["common_seconds"] = n / 50
        for label, (xy, joint) in zip(("baseline", "candidate"), errors):
            row[f"common_xy_mean_{label}"] = float(xy[:n].mean())
            row[f"common_joint_rmse_rad_{label}"] = float(
                np.sqrt(np.mean(joint[:n].astype(np.float64) ** 2))
            )
        row["delta_return"] = row["return_candidate"] - row["return_baseline"]
        rows.append(row)
    groups = {"all": rows}
    if base_clips is not None:
        groups.update(base=rows[:base_clips], added=rows[base_clips:])
    summary = {
        "groups": {},
        "trajectory_diagnostics": dict(zip(("baseline", "candidate"), diagnostics)),
    }
    for name, group in groups.items():
        stats = {"n": len(group)}
        for label in ("baseline", "candidate"):
            stats[label] = {"completed": sum(r[f"completed_{label}"] for r in group)}
            for key in ("return", "xy_mean", "common_xy_mean", "common_joint_rmse_rad"):
                stats[label][key] = float(np.mean([r[f"{key}_{label}"] for r in group]))
            stats[label]["frame_weighted_return"] = float(
                np.average(
                    [r[f"return_{label}"] for r in group],
                    weights=[r["reference_seconds"] for r in group],
                )
            )
        summary["groups"][name] = stats
    summary["newly_completed"] = [
        r["motion"]
        for r in rows
        if r["completed_candidate"] and not r["completed_baseline"]
    ]
    summary["newly_failed"] = [
        r["motion"]
        for r in rows
        if r["completed_baseline"] and not r["completed_candidate"]
    ]
    return rows, summary


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for side in ("baseline", "candidate"):
        parser.add_argument(f"--{side}-results", type=Path, required=True)
        parser.add_argument(f"--{side}-trajectories", type=Path, required=True)
    parser.add_argument(
        "--base-clips",
        type=int,
        help="Explicit split index for base/added motion groups",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="A new private output directory"
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    data, results, inputs = [], [], {}
    for side in ("baseline", "candidate"):
        trajectories = getattr(args, f"{side}_trajectories")
        result = getattr(args, f"{side}_results")
        with np.load(trajectories, allow_pickle=False) as archive:
            data.append(dict(archive))
        results.append(json.loads(result.read_text()))
        inputs[side] = {
            "trajectories": str(trajectories.resolve()),
            "trajectories_sha256": digest(trajectories),
            "results": str(result.resolve()),
            "results_sha256": digest(result),
        }
    rows, summary = compare(*data, results, args.base_clips)
    summary["inputs"] = inputs
    summary["protocol"] = (
        "50 Hz; first-frame full-clip evaluation; unaligned world XY; common alive prefixes"
    )
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "per_motion.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary["groups"], indent=2))


if __name__ == "__main__":
    main()
