"""Render an unmodified K1 candidate and its paired source mesh with gate labels."""

import argparse
import json
from pathlib import Path
import numpy as np
from instinct_pairs import render, K1_XML


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    meta = json.loads((args.candidate / "manifest.json").read_text())
    data = np.load(args.candidate / "candidate.npz")
    q, fps = data["qpos"], float(data["framerate"])
    if len(q) == 0:
        raise ValueError("No frames to render")
    title = f"K1 pair {meta['pair']['id']:02d}: {len(q)}/{meta['source_frames']} frames | NOT ACCEPTED FOR TRAINING"
    render(
        K1_XML,
        q,
        fps,
        args.candidate / "terrain.ply",
        args.output,
        title,
        stride=1,
        display_scale=meta["scene_scale"],
        failed_frame=meta.get("failed_frame"),
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "candidate": str(args.candidate.resolve()),
                "frames": len(q),
                "fps": fps,
                "source_scene_scale": meta["scene_scale"],
                "reference_only": True,
                "training_accepted": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
