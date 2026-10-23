# K1 terrain pilot on Mars5090

This is a fresh K1 experiment on `codex/terrain-aware-k1`. It does not resume the
G1 model: the robot, observation dimensions and visual input differ. The former
60-motion G1 run was stopped after iteration 203350; its complete iteration
202000 restart was protected as `stopped_202000_full.pt` in the original run.

## Frozen pilot data

`data/k1_terrain_pilot_v1` contains four unique ten-second windows from
`jumps1_subject1`, with four flat controls and three obstacle variants: seven
pairs in total. The obstacles are 3.25–8.37 cm high. This is a small LAFAN1 pilot,
not conversion of the archive's 77 motions or evidence of stair climbing.

- Import provenance SHA-256:
  `151ff2ab78fdd8bd66cf8dc201b2428b1febb9ecc75b7d789d6f055fbaa9ed37`.
- Terrain manifest SHA-256:
  `5b48e9708a10a5095a91c9f443d42e7c15ed4efc24222fb5692e3f825ed82e43`.
- All seven pairs passed independent native contact and physical joint-speed
  checks at both 50 Hz and 100 Hz. Worst ground/self penetration was below
  20 nanometres; obstacle intersections were zero. These are sampled geometry
  checks, not proof of dynamic feasibility.
- The retarget changes joint poses to accommodate the official K1 morphology.
  The largest source-body displacement is 60.6 mm and orientation difference
  0.760 rad in the 230–240 s window. Root trajectory and timing are preserved.
- Two surveyed `obstacles3_subject3` windows are explicitly excluded because
  the declared optimizer convergence criterion was not met. Their candidates
  were contact-feasible; the exclusions do not establish physical impossibility.

The data package retains the full acceptance/rejection ledger and binds the
independent audit, source/output motion hashes and robot assets through its
provenance. See [data preparation](k1_terrain_data.md) for exact source intervals
and reconstruction commands.

## Configuration and launch

The measured 32 GB RTX 5090 configuration is 2048 native environments, 2048
dynamics windows, 2048 imagined trajectories and 256 depth examples per depth
term. Replay remains float32 on the GPU, with 1000 steps per environment;
imagination remains 32 steps. Neural layers use TF32, geometry uses FP32, and
activation recomputation is disabled.

Initial capacity measurements (20 iterations each, before the final atlas
correction) were 2.69 iterations/s at 512/512 environments/policy batch,
2.16 at 1024/1024, and 1.87 at 2048/2048. The 4096-environment configuration
exceeded the card's memory. The 2048 setting processes more simulated and
imagined samples per second than either smaller setting; these measurements
are throughput evidence, not a claim of better sample efficiency.

The terrain pilot uses 500 world-model-only iterations, as in the pinned
[InstinctMJ configuration](https://github.com/project-instinct/InstinctMJ/blob/c9d3e00c8f4c16e394ddb3c2d545bac8b3e8819b/src/instinct_mj/rl/supermimic/config.py).
It retains this repository's 10-frame history and collects 32 initial replay
steps. These are explicit experiment settings; the G1 defaults remain unchanged.

On `leeky@Mars5090`, use the independent checkout
`/home/leeky/projects/SuperMimic-terrain-k1`. The host's existing process-local
driver library is required; no system driver is replaced:

```bash
export CUDA_VISIBLE_DEVICES=GPU-bdf96c80-193b-eb2a-720e-617adb7f879d
export LD_LIBRARY_PATH=/home/leeky/projects/SuperMimic/.cache/driver/lib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
./run_train.sh k1_terrain_lafan_pilot_mars5090_20261024 \
  --robot k1 --motion_dir data/k1_terrain_pilot_v1 \
  --num_envs 2048 --buffer_steps 1000 --replay_device cuda:0 \
  --wm_batch 2048 --wm_batches 10 --pi_batch 2048 --depth_batch 256 \
  --horizon 32 --goal_frames 10 --hist 10 \
  --warmup_steps 32 --wm_only_iters 500 --no-activation_checkpoint \
  --iterations 320000 --log_every 50 --ckpt_every 2000
```

First run a finite validation with the same mathematical settings, extending
past iteration 500 to exercise policy updates. Validate the complete restart
and fixed evaluation before continuing that same frozen run with `--resume`.
Only duration and checkpoint/log intervals may differ. Do not resume the earlier
two-pair diagnostic into this seven-pair pilot.

## Earlier two-pair diagnostic

The 1000-iteration diagnostic on the 50–60 s jump's flat/obstacle pair completed
and resumed its full state to iteration 1001, including all three depth heads,
their optimizer, replay images, simulator and RNG state. Its fixed evaluation
completed the flat clip (normalized return 0.4304) but failed the obstacle clip
(0.0712). Policy and reference renders are stored under
`results/analysis/k1_terrain_validation_20261024/` and are excluded from Git.
This establishes an executable learning/evaluation path, not a trained
terrain-capable controller. Future acceptance needs fixed evaluation of all
seven pairs, global drift, held-out terrain and depth ablations.
