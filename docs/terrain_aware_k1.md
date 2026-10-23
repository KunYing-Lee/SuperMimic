# Terrain-aware SuperMimic and Booster K1

This branch adds robot-dependent dimensions and a visual world-model training path while retaining the G1 flat-ground defaults. It is an independent implementation informed by [InstinctMJ's SuperMimic method](https://github.com/project-instinct/InstinctMJ/tree/c9d3e00c8f4c16e394ddb3c2d545bac8b3e8819b/src/instinct_mj/rl/supermimic). No InstinctMJ source files or checkpoints are bundled. The upstream repository uses CC BY-NC 4.0; consult its license before copying its implementation or assets.

## Robot and environment contracts

`--robot g1|k1` selects an explicit `RobotSpec`: named joints, tracked bodies, native MJCF, motor effort limits, armature, PD gains and nominal posture. K1 has 22 joints, including the head. Joint columns are mapped by name at every simulator boundary. K1 assets are copied unchanged from a pinned K1Lab revision by `tools/prepare_k1_assets.py`; their provenance and hashes are recorded. Native collision masks/exclusions are retained, with no new collision disabling or velocity clipping.

State, observation, goal, context, replay and network dimensions derive from the selected robot. Legacy G1 constants and constructors remain available. Generalized FK supports fixed ancestors, nonzero hinge anchors and joint reference angles, and is tested against native MuJoCo poses and velocities.

Each motion maps explicitly to a terrain in `terrain_manifest.json`. An environment resets both its reference frame and terrain identity together. Terrains occupy separated static tiles in the compiled MuJoCo scene; reference and world-model coordinates remain local to that motion. Resume reconstructs the same tile assignments from the restored reference frames.

## World model and vision

- Dynamics receives a yaw-aligned 17 x 11 height scan over 1.6 x 1.0 m. A 5 cm height grid is sampled with differentiable bilinear interpolation. Queries use the same global terrain atlas as physics, including another tile's geometry if the robot drifts there; outside all tiles the geometry is the shared world ground plane.
- The simulated sensor is an explicit **terrain-only** root-mounted pinhole camera: 18 x 32 axial depth, 58-degree vertical field of view, 0.1–5 m range, position `(0.08, 0, 0.25)` and 20-degree downward pitch. It raycasts the exact ground and box primitives used for physics. It does not model robot self-occlusion or a hardware-specific K1 camera.
- Three learned depth heads predict clean normalized images from an anchor image, relative root pose, joints, camera context and local terrain scan. They receive bootstrap L1 supervision on detached dynamics predictions and real state pairs with gaps up to the imagination horizon.
- The policy encodes the current depth image alongside its reference goal and proprioceptive history. At imagination step zero the image comes from replay; later steps use predicted depth. The same range-dependent Gaussian sensor noise is applied to both paths.
- Predicted-image gradients are stopped. The dynamics path retains BPTT gradients. Ensemble assignments are randomized, balanced across the batch, and held fixed throughout each trajectory.
- The critic additionally receives the terrain scan. Existing analytic rewards and the learned termination head remain in use. This does not adopt InstinctMJ's additional contact/failure heads or its task-specific reward terms.

The range sensor, fixed camera mount, observation history and reward setup are explicit differences from InstinctMJ. This is not a claim of hardware-ready perception or an exact reproduction of its reported success rate.

## Data and acceptance

See [K1 motion and terrain pairing](k1_terrain_data.md) for pinned X-WBC provenance, the detected morphology mismatch, constrained retargeting and terrain construction. Geometric clearance creates valid obstacle-crossing pairs where source motion permits it; it cannot create elevated-support stair-climbing demonstrations from flat-ground data.

Before a long run:

1. Validate source ordering and robot morphology; bind all derived files to their source and robot hashes.
2. Validate the actual 50 Hz reference and a denser 100 Hz contact grid, including ground, self and obstacle contacts.
3. Run unit tests, `tools/validate_terrain_runtime.py`, short finite training, evaluation and representative reference/policy renders.
4. Record the code commit, dataset manifest, measured GPU configuration and new run directory. Start a fresh K1 model; a G1 checkpoint has incompatible embodiment and sensor dimensions.

The native runtime validator tests exact robot/terrain identity, depth dimensions, finite transitions and simulator restart. Passing it is not evidence that a random policy tracks the motions. Similarly, collision-free references do not establish dynamic feasibility; fixed-rollout evaluation remains necessary after training.

`tools/render_terrain.py` renders named joint mappings and each clip's exact terrain. The reference stays in its original global position; no XY/yaw alignment conceals tracking drift.

Training completion does not by itself establish that the policy uses depth: a fixed motion-to-terrain mapping can let the reference goal identify the terrain. Preserve flat/obstacle pairs of the same motion, and evaluate held-out terrains plus depth removal or matched depth-swap controls before claiming perception-dependent terrain adaptation.

See [the Mars5090 pilot record](k1_terrain_mars5090.md) for the frozen seven-pair dataset, measured single-card configuration, short-run results and launch command.
