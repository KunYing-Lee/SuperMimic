# InstinctMJ paired scenes to K1

This replaces the stopped LAFAN1 low-obstacle pilot. Data acceptance and training
acceptance are separate. A candidate directory containing `INCOMPLETE` cannot be
used for training, including when a geometric check alone passes.

## Sources and coordinates

The `supermimic-v0.1` release archive is
`20251116_50cm_kneeClimbStep1.zip`, SHA-256
`9d84a5efb5f08d3e243326f16ffcfc93c2721db50b38a0bbef99354ca2f93418`.
Its published checksum is verified by `tools/prepare_instinct_release.py`.
Only the top-level `metadata.yaml` defines selection: ten motion/scene pairs,
six terrain entries, and four named skills. Nested metadata and unlisted NPZs
are not additional accepted samples. All motions are at 90 Hz.

The source robot is InstinctMJ's **torso-root**
`g1_29dof_torsobase_popsicle.xml`, pinned to
`c9d3e00c8f4c16e394ddb3c2d545bac8b3e8819b`. Loading the same joint angles into
a pelvis-root model changes the world geometry. The importer checks names and
unit wxyz quaternions; joint array order is never assumed.

The target remains the existing pinned K1Lab serial-ankle robot. Both collision
and visual meshes are preserved. Optional whole-scene scaling applies one factor
to source positions, scene vertices, and contact anchors. The default factor is
the target/source zero-pose collision-surface height ratio. It is recorded before
optimization and is not adjusted in response to failures. It changes the physical
task height, which must be reported with every experiment. No root-height offset,
joint clipping, motion smoothing, or terrain-top alignment is applied afterward.

## Algorithm and limitations

The adapter uses the interaction-mesh Laplacian formulation from
`project-instinct/omniretargeting`, pinned to
`6518c36a2be9dab2538cc1b95840da00fbd6d4b7`. This is an independent implementation
of the OmniRetarget method, not the original authors' implementation or UMR.
It reuses `GenericInteractionRetargeter` for the interaction graph and mappings.
The public wrapper contains constraint-dropping retries and resolver switching;
those paths are not used. The optimizer is SLSQP with an SO(3) exponential chart,
analytical objective/constraint Jacobians, fixed constraint identities within
each frame, and `ftol=1e-6`. This is an explicit solver adaptation, not a claim
that the upstream public retargeting command was run unchanged.

The source release contains poses and scene meshes, but no contact-force or
contact-label annotations. Surface proximity provides explicitly **inferred**
geometric contacts. Every frame retains one nearest scene anchor per anatomical
collision-body group when its surface gap is at most 20 mm in the scaled scene.
There is no velocity filter: rolling and sliding are allowed. These observations
do not establish that a contact carries load.

Mapped K1 collision surfaces must meet those source anchors within 2 mm.
The constraint is signed point-to-convex-surface distance, differentiated through
convex closest-point/contact witnesses and body Jacobians. The closest material
point can move during rolling. These hard constraints supplement the spatial
interaction graph; increasing a soft contact-edge weight is not treated as
equivalent to enforcing contact. Self-collision and sampled scene nonpenetration
use zero penetration constraints (1 micrometre numerical feasibility tolerance), and joint speed bounds come from the existing K1 motor metadata.
Self-distance and contact constraints use Coal 3.0.3 GJK/EPA uniformly for the exact MuJoCo primitives and
convex mesh hulls. MuJoCo 3.11 can return a zero-distance/zero-witness result for
a trial shin/foot configuration with no native contact normal; that result is
not treated as a valid differentiable distance. Independent audits still use
native MuJoCo contacts. At the K1 neutral pose, all 211 enabled body-pair queries
agree between Coal and default native MuJoCo within 1.2e-10 m.
GJK and EPA query tolerances are 1e-10 m, with 1,000 iteration limits. Invalid
witnesses fail the candidate; no alternate collision oracle is substituted.
Earlier FCL candidates are retained as rejected diagnostics. Tightening MuJoCo
CCD tolerance to 1e-10 caused incorrect zero-distance witnesses in otherwise
separated pairs, so independent physics audits use the target model defaults.

Mesh collision samples include actual convex-hull vertices and triangle centers,
rather than substituting the collider center. This remains a sampled geometric
optimization, not continuous collision certification. Independent dense sampling
and native collision checks are required before accepting any complete result.
Input STL face winding can be normalized without moving vertices. Nearest-face
normal signs on nonwatertight scans are not certified solid penetration depths.
The separate native collision-scene audit is mandatory. CoACD preprocessing is
an explicit approximation, whose error must be measured against source anchors.

Every frame must satisfy both the numerical solver's convergence criterion and
the nonlinear constraints. The upstream zero-step result after a rejected
nonlinear step does not count as convergence. Rejected candidates and
diagnostics are retained, and no failed frame is silently omitted from an
accepted motion. Terrain graph samples use an explicit seed of zero, because
trimesh's generator is not controlled by `numpy.random.seed`; the actual samples
are retained with each candidate.

## Acceptance order

1. Verify release hashes, root frame, named joint mapping, pair manifest, and
   original G1 reference videos against the original terrain triangles.
2. Retarget with frozen scene scale and explicit contact/collision/velocity
   constraints; retain source anchors and solver diagnostics.
3. Independently audit all poses and interpolated training-time samples for
   self/scene penetration, contact error, joint limits and physical joint speed.
   Report body/scene approximation error separately from retarget error.
4. Inspect K1 reference videos of all skills in their paired scenes. Reference
   playback is not a simulation tracking result.
5. Validate simulator collision geometry and depth/height observations against
   the same scenes; run finite dynamics, training, restart and fixed evaluation.
6. Start a fresh long training run only after the preceding gates pass. Never
   resume the stopped pilot into the different data manifest.

The current runtime's box-only terrain bank cannot represent these scanned
scenes. A mesh-backed collision and sensing implementation must pass step 5;
loading a visually correct mesh alone does not satisfy it.

## Validation record, 2026-10-24

The previous pilot is stopped with its full iteration-2000 checkpoint preserved.
No replacement long training has been started. All new directories remain
`INCOMPLETE`; no accepted training data manifest exists.

A 100-frame (0.99 s) climb prefix converged with the FCL signed-distance version.
This is not a full motion or evidence of learned climbing. The source is sampled
from 90 Hz to the 100 Hz physics timeline before constrained retargeting; 50 Hz
policy samples are a subset. Resampling a constrained 90 Hz solution afterward
was rejected because it introduced 1.27 mm scene penetration.

The first scene-2 CoACD build used a 5 mm real-metric threshold and generated
4,008 hulls. Reconstructing the convex polytope of each returned vertex set
retains all parts without translating or inflating them. The native scene audit
of the 100-frame prefix measured **4.485 mm** maximum scene penetration and
**4.484 mm** source-anchor mismatch, exceeding the 1 mm penetration / 2 mm
contact gates. A visually aligned source-STL replay therefore cannot release
this geometry for training. No automatic top-surface correction is used.

The upstream InstinctMJ perceptive task enables CoACD top-surface auto-alignment
in `perceptive_env_cfg.py`. That post-decomposition translation is deliberately
not copied. The failed geometry and candidate remain available for diagnosis.

## Reproduction

Use an isolated Python 3.12/3.13 environment with `requirements-retarget.txt`,
and install the pinned `project-instinct/omniretargeting` checkout editable with
`--no-deps`. The code expects that checkout at `.cache/omniretargeting`; on macOS
Open3D also requires the system `libusb` library. Run tools from the repo root.

```sh
python tools/prepare_instinct_release.py
python tools/batch_instinct_retarget.py --output data/instinct_k1_new_batch --workers 6
python tools/audit_instinct_k1.py data/instinct_k1_new_batch/pair_02
python tools/render_instinct_candidate.py data/instinct_k1_new_batch/pair_02 --output results/analysis/k1_pair02.mp4
```

The batch snapshots all four retarget source modules, records installed package
versions, saves progress every 25 frames, and returns nonzero if any full pair
fails. It never starts training. Rendering labels all candidates as unaccepted.
Collision validation is separate, using `tools/audit_instinct_collision.py`.

## Final batch disposition (2026-10-24)

**Rejected for training.** All ten pairs were attempted to completion or an
explicit failure. One completed the full constrained solve and the independent
dense 50/100 Hz checks; nine failed. No failed frame or motion was removed to
make an accepted subset. These are reference poses, not policy rollouts.

| Pair | Retained frames | First failed frame (zero-based) | Outcome |
|---|---:|---:|---|
| 00 | 834/834 | — | Full solve passed; not training-accepted |
| 01 | 450/834 | 449 | Positive directional derivative for linesearch |
| 02 | 156/1107 | 155 | Positive directional derivative for linesearch |
| 03 | 257/1074 | 257 | RuntimeError: Inconsistent convex signed-distance witness |
| 04 | 310/1046 | 310 | RuntimeError: Inconsistent convex signed-distance witness |
| 05 | 452/834 | 451 | Positive directional derivative for linesearch |
| 06 | 126/1107 | 125 | Positive directional derivative for linesearch |
| 07 | 19/1074 | 19 | RuntimeError: Inconsistent convex signed-distance witness |
| 08 | 245/1046 | 244 | Iteration limit reached |
| 09 | 450/834 | 449 | Iteration limit reached |

Retained frames can include a rejected final iterate. The failing frame is
marked in the replay when an iterate was returned. An oracle exception retains
only earlier frames. Thus frame count is not an acceptance percentage.

Pair 00's dense audit measured 0.455 mm maximum sampled scene intrusion,
0.00021 mm native self penetration, and 2.001 mm maximum anchor gap (the
2 mm bound plus declared 1 micrometre numerical tolerance). It passed joint
position and velocity checks. Its mapped source/target landmark mean difference
is 5.28 cm; differing limb lengths mean that metric is not a zero-error target.

Pair 02 failed at 1.55 s with 8.52 cm contact error. Native collision on the
refined scene measured 13.40 mm penetration in that rejected pose. The refined
CoACD build used a **1 mm** threshold and preprocessing resolution **400**,
producing **9,087 hulls**. Across all **1,771** distinct source contact anchors,
the maximum scene-surface mismatch is **5.015 mm**, still above the 2 mm gate.
Neither increasing geometry resolution nor passing one flat motion releases
this dataset for training. This does not prove K1 cannot perform the skills.

The current branch provides data ingestion, constrained retargeting, independent
audits, source snapshots and diagnostic rendering. It does **not** yet provide
an accepted mesh-backed terrain training dataset/runtime. The pilot remains
stopped, with its full checkpoint retained.

Replays and machine-readable evidence are local generated artifacts under
`results/analysis/instinct_k1_20261024/`; all candidates and exact solver-source
snapshots are retained under `data/instinct_k1_coal_full_v3/`.

Validation: 9 retarget geometry/derivative tests and 17 existing terrain tests
passed; Ruff checks and formatting passed. No long-run acceptance is inferred
from those tests.

Primary sources: [release](https://github.com/project-instinct/InstinctMJ/releases/tag/supermimic-v0.1),
[pinned OmniRetarget reimplementation](https://github.com/project-instinct/omniretargeting/tree/6518c36a2be9dab2538cc1b95840da00fbd6d4b7),
[pinned upstream terrain settings](https://github.com/project-instinct/InstinctMJ/blob/c9d3e00c8f4c16e394ddb3c2d545bac8b3e8819b/src/instinct_mj/tasks/shadowing/perceptive/perceptive_env_cfg.py).
