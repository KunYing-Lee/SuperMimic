# K1 motion and terrain pairing

The source is the user's specified Hugging Face dataset,
[TonyZhang1999/xwbc-retarget-data](https://huggingface.co/datasets/TonyZhang1999/xwbc-retarget-data).
The linked `refs/convert/parquet/default/partial-train` is an automatically
converted partial view whose first rows are AMASS motions for Berkeley Humanoid
Lite. It is not a K1 LAFAN subset. The authoritative K1 LAFAN archive is
`lafan/booster_k1.tar.gz` in the same dataset.

Pinned source:

- Revision: `b4e169f460bd68b27e04e75c99be6e0affbfe046`.
- Archive SHA-256: `92fafe9e6372bf08b26061cc7ed4356f9f5c4bbffce646a86a0574a331112e42`.
- 77 motions, 30 Hz, 22 joints, 23 body transforms; root quaternion is `wxyz`.
- The NPZ files have no joint names and no terrain/contact annotations. Their
  joint order is breadth first, unlike the SDK order used by this implementation.
  The importer validates the mapping against all source body transforms; a
  dimension match alone is insufficient.

## Source morphology is not the official K1 morphology

The source body transforms put both hip pitch origins at local z = -0.062 m.
The official Booster K1 and the current K1Lab asset put them at -0.077 m. With
identical root/joint coordinates, head and arm poses agree but every leg is
shifted by 15 mm in the trunk frame. A 3,932-frame source audit measured a maximum
15.006 mm discrepancy, while orientation error was below 4e-7 in
`1 - abs(quaternion dot product)`.

The importer rejects this discrepancy. Changing a root height or silently
changing the official robot morphology would conceal a different problem.
Training on this source requires either its actual matching robot asset or an
explicit, validated morphology retarget onto the official K1. Original source
files remain immutable and their hashes remain part of derived provenance.

## Terrain construction

For each selected motion, retain a flat-ground control. A non-flat variant keeps
the same reference motion and uses axis-aligned boxes on the ground. A box may
occupy a fixed XY grid cell only when the root path traverses that cell. Its
height solves the separable maximum-height problem

```
maximize sum(cell_heights)
subject to 0 <= cell_height <= configured_max_height
           cell_height + clearance <= lower_z(geom, frame)
           for every sampled collision-geometry AABB overlapping that cell
```

The constraints include all robot collision geometry, all frames, and all
revisits to each cell. Grounded foot revisits therefore prevent an obstacle from
being placed beneath that foot. The world AABBs come from native MuJoCo geometry,
including mesh hull bounds. An independent native MuJoCo contact pass then checks
the generated boxes against every sampled pose. An obstacle intersection is a
hard error. The default validation frequency is 100 Hz and includes all 50 Hz
training-reference samples. For morphology adaptation, source targets are
resampled to 100 Hz **before** constrained IK, and its outputs are stored at
100 Hz; the 50 Hz training references are an exact subset. This is a sampled kinematic check, not continuous
collision proof or dynamic tracking acceptance.

The construction supports jump-over or step-over examples where the existing
motion provides clearance. It does **not** produce demonstrations of climbing
stairs: a flat-ground motion does not acquire elevated support contacts merely
because boxes are added. Source ground penetration and self-intersection are
reported without root shifts, joint clipping, or other repair.

## Artifacts and manifest

`tools/prepare_k1_terrain_data.py` accepts a verified archive, a standalone robot
MJCF, an explicit JSON runtime joint order, and exact source motion names. It
writes into a new directory:

- Per-pair CSVs: root xyz, quaternion xyzw, named joints in runtime order.
- `motion_rates.json`: explicit sample rate per CSV (source rate, or the recorded
  rate used by morphology retargeting).
- `terrain_manifest.json`: explicit filename-to-terrain mapping and exact boxes.
- `import_provenance.json`: source and model hashes, order conversion, FK audit,
  terrain constraints, and native collision measurements.

No input pose is altered by this importer. Any morphology changes belong to the
separate retarget tool and its retained provenance. Flat and terrain variants contain
identical pose sequences, allowing controlled comparisons.

```json
{
  "schema_version": 1,
  "robot": "booster_k1",
  "terrains": [{
    "id": 0,
    "name": "example__terrain",
    "ground_height": 0.0,
    "boxes": [{"center": [0.0, 0.0, 0.05], "half_size": [0.1, 0.1, 0.05]}],
    "bounds_xy": [[-3.0, -3.0], [3.0, 3.0]]
  }],
  "motions": {"example__terrain.csv": {"terrain_id": 0}}
}
```

The simulator, terrain scan, and depth raycaster must consume these same boxes.
Terrain fields must extend beyond the motion and obstacles; the importer adds a
2 m margin. Dataset archives, generated CSVs, and training artifacts belong under
ignored `data/` or `results/analysis/`, not in Git.

## Explicit morphology-retarget inputs and fixed windows

`--retarget-dir` accepts derived NPZs only with an explicit `--retarget-files`
list. Every NPZ has a same-stem JSON record that binds its original full NPZ
checksum, derived checksum, exact runtime MJCF checksum, source frame interval,
and successful solver status. The importer checks that the root trajectory equals the original trajectory
after the explicitly recorded linear translation / quaternion SLERP rate
conversion, checks all derived body transforms against native FK,
and rejects a ground or self penetration deeper than the declared validation
tolerance (default 1 mm) at 100 Hz. This tolerance is a validation criterion; it
does not modify motion or simulation contact geometry.

`--segment-seconds 10` splits source time into fixed ten-second windows sharing
one endpoint. `--max-segments` selects an explicit prefix for a pilot, not a
quality-ranked subset. Filenames and manifests record original half-open frame
intervals. This prevents motion revisits several minutes later from imposing a
conflicting static terrain on an earlier jump while retaining the original
poses, trajectory, and timing of each episode.

The initial geometric survey covers every fixed ten-second window in
`jumps1_subject1` and `obstacles3_subject3`. The declared selection criterion is
nonempty feasible obstacle cells under the traversed root path, measured before
any policy training. It identifies source intervals `[1500,1801)`, `[1800,2101)`,
and `[6900,7201)` in `jumps1_subject1`, and `[2700,3001)` in
`obstacles3_subject3`. Both clips' first `[0,301)` interval provide fixed flat
controls. These intervals require fresh feasibility checks **after** morphology
retargeting; the preliminary source survey is not an acceptance result. This is
a scoped pilot from two clips, not conversion or validation of all 77 clips.

The first accepted pilot is `data/k1_terrain_pilot_v1`: four unique ten-second
windows, four flat controls and three terrain variants. Its six-candidate ledger
retains both excluded windows rather than silently dropping them.

| Source interval (30 Hz, half-open) | Decision | Boxes | Maximum source-body position / orientation difference |
| --- | --- | ---: | --- |
| jumps1_subject1 `[0,301)` | Accepted flat control | 0 | 18.2 mm / 0.123 rad |
| jumps1_subject1 `[1500,1801)` | Accepted flat and terrain | 7 | 18.9 mm / 0.283 rad |
| jumps1_subject1 `[1800,2101)` | Accepted flat and terrain | 3 | 19.4 mm / 0.123 rad |
| jumps1_subject1 `[6900,7201)` | Accepted flat and terrain | 1 | 60.6 mm / 0.760 rad |
| obstacles3_subject3 `[0,301)` | Excluded: optimizer did not converge | — | No accepted derived motion |
| obstacles3_subject3 `[2700,3001)` | Excluded: optimizer did not converge | — | No accepted derived motion |

These differences measure morphology retargeting relative to the original body
targets, not FK inconsistency in the exported motion. The larger distortion in
the last jump window remains explicit. Terrain heights are 3.25–8.37 cm. All
seven exported pairs pass native collision and physical joint-speed checks at
both 50 and 100 Hz. The largest measured penetration is below 0.00003 mm; speed
limits hold within float32 rounding. This does not certify dynamic tracking.

Both excluded windows reached the declared optimizer iteration budget with
contact-feasible candidates but without convergence. Their exclusion therefore
does not establish that the motion is physically impossible. A separate offline
collision-query crash was isolated to the combination of an enlarged proximity
query margin and custom CCD precision. The unchanged runtime model and the
query model using its normal CCD settings passed the exact failing pose. The
bounded final optimizer attempts used the runtime CCD settings and emitted no
motion after failing convergence.

`import_provenance.json` binds the independent 50/100 Hz audit, selection ledger,
acceptance decisions, rejection reports, and all runtime asset hashes. The first
three historical retarget runs retain source/output/model hashes, parameters and
per-frame solver records, but did not record their tool's startup hash; the last
jump window did. They are not represented as outputs of the later tool revision.

Recreate the accepted CSV and terrain package from its recorded derived inputs:

```bash
python tools/prepare_k1_terrain_data.py \
  --archive data/k1_source/booster_k1.tar.gz \
  --robot-xml data/k1_description/k1.xml \
  --joint-order data/k1_source/runtime_joint_order.json \
  --retarget-dir data/k1_source/retarget_final_inputs \
  --retarget-files jumps1_subject1__f000000_000301.npz \
    jumps1_subject1__f001500_001801.npz \
    jumps1_subject1__f001800_002101.npz \
    jumps1_subject1__f006900_007201.npz \
  --clips jumps1_subject1 \
  --segment-seconds 10 \
  --output data/k1_terrain_pilot_rebuild
```

The output keeps an `INCOMPLETE` marker until every selected interval passes.
A directory carrying that marker must never be used for training. No failed
window is silently skipped.

The provenance also reports the maximum and 99th-percentile joint speed, maximum
joint acceleration, and largest joint step at the 50 Hz control rate. These are
uncapped measurements; the importer does not clip velocities or filter motion.
Geometric acceptance must be followed by simulator validation before training.
