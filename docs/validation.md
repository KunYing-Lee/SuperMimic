# Validation record

Date: 2026-10-22. This file records implementation checks, not trained-policy quality.

## Local checks

- Algorithm checks cover circular replay chronology, terminal masks, valid history/horizon sampling, deterministic replay restoration, SE(2) goal invariance, orientation-error gradients, RNG-preserving activation recomputation, and vertical-pelvis coordinate behavior.
- Native backend: mjlab 1.6.0, MuJoCo/MuJoCo-Warp 3.11.0, Warp 1.14.0; CPU, two worlds, 30 control steps.
- Maximum FK position error 1.19e-7 m; rotation-matrix component error 1.79e-7; link velocity errors <=1.49e-8.
- Per-world friction/CoM and partial reset isolation checked.
- Checkpoint replay across clip end reproduces next transition, reward, reset state, context, and bin return exactly in the CPU check.
- All 40 real clips initialize at their exact first reference frame: maximum joint error 0; first-frame joint-limit violations 0.

## Engine limitations retained

MuJoCo-Warp reports at most one contact for cylinder-cylinder/cylinder-mesh CCD pairs. The source URDF geometry remains intact. Native MuJoCo position actuators enforce effort limits and PD/armature parameters; source velocity ratings are not turned into an artificial hard velocity clamp.

## GPU gates

A800 full-size training, restart, evaluation, memory and finite-gradient validation are performed before long training by `tools/launch_when_idle.py`. A waiting launcher is not a running experiment. Logs and manifests in the remote run directory are the authority for current GPU validation and training status.
