# SuperMimic on mjlab

An independent implementation of [SuperMimic](https://arxiv.org/abs/2610.28480v1), migrated from [Chong Zhang's supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) to **native mjlab 1.6.0 / MuJoCo-Warp 3.11.0**. Original history and MIT attribution are preserved.

The experiment is **one G1 policy for 40 LAFAN1 motions** (264,705 source frames at 30 Hz, approximately 2.45 hours), corresponding to the paper's controlled comparison. This is not the BONES-SEED experiment or a reproduction of every interface in the paper. Historical upstream scores and videos describe a different implementation; they are retained in Git history and [MILESTONES.md](MILESTONES.md), not presented as results of this port.

## Paper settings and disclosed choices

See [the source-backed audit](docs/paper_alignment.md) for equations, settings, discrepancies, and unspecified details. The default training configuration uses:

- 4,096 environments, 1,000 transitions/environment, 50 Hz control and four 5 ms physics steps.
- 93-dimensional proprioception with 10-frame history; 10 future reference frames; 32-dimensional bounded latent.
- World-model MLP 2048/2048/2048; policy encoder and decoder 2048/1024/512/256; critic 256/128.
- 10 world-model updates with 2,048 eight-step windows; 4,096 imagined trajectories of length 32.
- Adam learning rate 0.0002, policy betas 0.9/0.95, discount 0.99, lambda 0.95; critic 5 epochs × 4 minibatches.
- Update/data ratio 0.4; 320,000 iterations; cosine learning-rate decay over 250,000 iterations to 0.03×.
- Float32 tensors with TF32 neural-network matmuls. Three-dimensional geometry uses float32 multiply/add operations so TF32 does not distort rigid rotations. Gaussian action noise 0.1, no extra action clamp.
- Per-episode friction, joint calibration and torso CoM randomization. Native self-collision physics.
- Exact reference-state initialization, natural clip end, and eight initial collection steps to form an eight-step replay window. Additional reset perturbations and episode time limits are opt-in.

The original additional reference-orientation observation, bf16, hand-shaped soft termination, dynamics-error floors, 500 world-model-only iterations, and synthetic timeout return extrapolation are removed. The goal contains future reference poses and velocities relative to the current robot root/body frame; this exposes the position/yaw errors penalized by the reward while retaining 93-dimensional proprioception. This goal convention, the learned goal-conditioned termination MLP, and the finite-horizon return estimator are disclosed implementation choices where the paper is incomplete.

Replay storage and activation recomputation are runtime choices measured with `tools/benchmark_runtime.py`. GPU replay without recomputation was fastest on the tested RTX 5090 and fit within 32 GB; see [measurements](docs/mars5090_runtime.md). These choices preserve batch sizes, horizon, sampling distribution and gradient truncation. Full-size validation is required before long training.

## Setup

Linux with an NVIDIA GPU is required for training. The pinned Torch wheel uses CUDA 12.8; the pinned Warp wheel reports CUDA 12.9, so the driver must support both runtimes. CPU mjlab checks can run on macOS. Install `uv`, then:

```bash
./tools/setup.sh
```

This installs only the project's `.venv`, uses `uv.lock`, fetches an immutable LAFAN1 dataset revision, verifies 40 clips/264,705 frames, and generates MJCF directly from the source URDF. It needs no Isaac Lab, Isaac Sim, Newton wrapper, or USD assets. The downloader writes per-file SHA-256 hashes to `data/provenance.json`.

Data is not redistributed: LAFAN1 motions are CC BY-NC-ND 4.0; the Unitree robot description is BSD-3-Clause. The source revision and file mapping are in `tools/download_data.py`. The URDF conversion preserves mass/inertia and collision geometry. The native position actuators use the source PD gains, effort limits, and reflected armatures; source velocity ratings are metadata, not an artificial hard velocity clamp.

## Validation and launch

```bash
uv run pytest -q
uv run python tools/test_env.py --device cpu --steps 30
CUDA_VISIBLE_DEVICES=GPU-uuid uv run python tools/test_env.py --device cuda:0 --num_envs 4 --steps 30
CUDA_VISIBLE_DEVICES=GPU-uuid ./run_train.sh lafan1_mjlab
```

`run_train.sh` runs in the foreground. Run it in tmux for persistent remote training. `RUN_ROOT` can place outputs outside the repository. A fresh run refuses to overwrite an existing training directory.

For a shared NVIDIA host, the following waits for one card with no compute process, zero utilization, and at most 64 MiB used across two polls. It never stops existing workloads. A per-GPU advisory lock coordinates this launcher but cannot reserve the GPU against unrelated users.

On the A800 workspace host with system driver 550.127.05, the installed official CUDA 13 compatibility library was verified with Torch matrix multiplication and Warp CUDA graph capture. Set `LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH` for this process before launch. This uses the existing compatibility installation without changing the system driver. Keep caches, temporary files and outputs in the workspace; the launcher sets their locations under the project.

```bash
.venv/bin/python tools/launch_when_idle.py \
  --gpu-model A800 --replay-device cuda:0 \
  --run-name lafan1_mjlab_20261022 \
  --runs-root /wuji-vefps-D/wuji-il/leekunying/SuperMimic/logs --wait
```

The launcher first runs backend checks, two **full-size** training iterations, a full-state resume to iteration 3, and deterministic evaluation. Only duration/logging differ from the long run; failure stops the launch. After validation it starts a separate fresh 320,000-iteration run on the same GPU. The validation and training logs are separate. The queue waiting for a card is not an active training run.

Full restart checkpoint `last.pt` includes models, optimizers, replay, simulation integration state, domain parameters, reset statistics, observation history, random-generator state, and the update/data accumulator. Resume validates source hashes, assets, package versions, and mathematical configuration:

```bash
CUDA_VISIBLE_DEVICES=GPU-uuid ./run_train.sh lafan1_mjlab \
  --resume logs/lafan1_mjlab/last.pt
```

`ckpt_XXXXXX.pt` contains policy/world-model weights for evaluation; it is not a full restart checkpoint. The run manifest records code/dependency/data identities and implementation choices. A successful smoke or an alive process does not establish policy quality.

## Evaluation and rendering

Evaluation plays every complete clip from its first frame with friction 1, no randomization/noise, and a deterministic policy. It reports failed clip names and both clip-mean and frame-weighted normalized returns. Early failure is penalized against the entire clip length.

```bash
CUDA_VISIBLE_DEVICES=GPU-uuid uv run python supermimic/evaluate.py \
  --run_dir logs/lafan1_mjlab --ckpt logs/lafan1_mjlab/ckpt_002000.pt \
  --record logs/lafan1_mjlab/traj.npz
MUJOCO_GL=egl uv run python tools/render.py logs/lafan1_mjlab/traj.npz \
  logs/lafan1_mjlab/tracking.mp4 --global_ghost
```

Schedule evaluation on available capacity or after training. The paper's 0.82 evaluation return is a target, not a result of this port. Report motion acceptance separately from return and hardware throughput. RTX 5090 timings are not directly comparable to the paper's H100 timings.

## Dataset extensions and comparison workflow

The default remains the original 40-clip experiment. `MotionLib` also supports a per-file `motion_rates.json` map for mixed-rate libraries, resampling each clip directly to 50 Hz. `tools/prepare_motion_extension.py` imports the supported manifest-described 20-clip qpos36 bundle with checksum validation and quaternion-column reordering only. Inputs remain private.

For an authorized dataset extension, `tools/extend_checkpoint_dataset.py` explicitly migrates an append-only library into a new full checkpoint and run path. It preserves the original clips, model/optimizer/replay/simulator/RNG state and learning-rate iteration; new bins start empty and sampling-average history restarts on the expanded domain. Ordinary same-dataset resume remains strict. Inspect each tool with `--help` and retain the migration provenance.

Use [the fixed-evaluation comparison workflow](docs/evaluation_comparison.md) to compare base and added clips, global drift and common-prefix tracking errors. The reusable CPU tool replaces run-specific analysis scripts; private reports stay under ignored `results/analysis/`.

## License and attribution

[MIT](LICENSE), original copyright © 2026 ETH Zurich, Chong Zhang. Method: Q. Liao et al., *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*, arXiv:2610.28480. This repository is unofficial and independent of the paper's authors. Third-party data and libraries retain their own licenses.
