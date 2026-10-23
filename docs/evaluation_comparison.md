# Compare fixed-library evaluations

Keep evaluation and rendering off the active training GPU. Copy an immutable `ckpt_XXXXXX.pt` snapshot, record its SHA-256, and use the same source, robot assets, motion rates, package versions and environment count for both policies. An existing recorded baseline can be reused only when these identities and the evaluation protocol match. This comparison tool does not run policies or verify those external identities for you.

Run each policy from the first frame of every clip with the native evaluator (friction 1, no randomization or action/observation noise), recording trajectories. Always pass the intended library explicitly, especially after adding clips:

```bash
CUDA_VISIBLE_DEVICES=GPU-uuid uv run python supermimic/evaluate.py \
  --run_dir logs/candidate --ckpt logs/candidate/ckpt_184000.pt \
  --motion_dir data/extended_library --record logs/candidate/trajectories.npz
```

Compare the JSON results and full NPZ recordings on CPU:

```bash
uv run python tools/compare_evaluations.py \
  --baseline-results logs/baseline/ckpt_136000_eval.json \
  --baseline-trajectories logs/baseline/trajectories.npz \
  --candidate-results logs/candidate/ckpt_184000_eval.json \
  --candidate-trajectories logs/candidate/trajectories.npz \
  --base-clips 40 --output-dir results/analysis/comparison
```

`--base-clips` is an explicit split of the identical, ordered library into base and added clips; omit it for a single group. The output directory must be new. The tool requires only NumPy and does not load Torch, a checkpoint or a simulator.

Outputs:

- `per_motion.csv`: normalized return, completion, alive duration, unaligned world-frame root XY error (mean, p95, maximum, final), joint RMSE, and common-prefix errors for each policy.
- `summary.json`: clip-mean and frame-weighted group returns, completion counts, newly completed/failed motions, input hashes, and post-episode nonfinite diagnostics.

Both libraries must have exactly matching reference arrays and clip ordering. Alive samples must form a contiguous prefix starting at the first reference frame. Truncated recordings, nonfinite live values and inconsistent completion metadata are rejected. The evaluator continues stepping finished environments until the longest clip ends; nonfinite values strictly after episode end are counted and reported, never repaired or included in errors. Original recordings remain unchanged.

A short failed rollout can have a deceptively small average error. Common-prefix metrics compare only frames where both policies are alive, and must be read alongside completion and full-clip-normalized return. Group errors are equal-weight means over clips; common joint error is the mean of per-clip joint RMSEs, in radians. Group returns use the evaluator's four-decimal per-clip values, so small rounding differences from its aggregate JSON are expected.

The native completion criterion checks height and tilt, not global XY drift. Report completion and motion quality separately. Long-horizon outcomes can be sensitive to numerical differences, including environment batching; a fixed comparison is evidence for the tested protocol, not statistical robustness across seeds or unseen terrains.

Private motion bundles, checkpoints, recorded trajectories and rendered media stay outside version control. `results/analysis/` is ignored so reports can be kept alongside their immutable inputs without making Git dirty.
