# New read-only terrain root-cause diagnosis

2026-11-17. This is a new point-return/classifier investigation, not a repetition of the coordinate-alignment audit. No original/project source modification, ROS/simulator launch, model alteration or support-gate change. Full counts, source hashes and replay proof: `.work/live_next_terrain_diagnosis.json`.

## What the recorded points prove

Failed padded cells `[51,54]` and `[51,55]` correspond to global .15 m voxel columns `(-4,-1)` and `(-4,0)`, physical x interval `[-.60,-.45]` m and y intervals `[-.15,0]` / `[0,.15]` m.

I inspected every saved `terrain_sample_*.npz` from all four r2 BT sessions: seven actual point samples, containing ~11.4k points for each first update and ~91.3k for later eight-scan batches. **All seven have zero `observed_xyz` returns in both failed columns and zero observed support layers there.** They also show a rear-center strip with no nearby observed ground; the absence extends outside the .45 m exclusion disk. In episode 0's later sample, rear direction within +/-15deg has closest ground return about 1.261 m from the estimated base; sideward returns reach .45–.47 m.

I independently replayed each of the four saved first batches through the unchanged `VoxelTerrainMap` using its recorded pose/config/timestamps. All five output counts (`observed_voxels`, free, occupied, unknown, reachable support) **match the original fixture reports exactly**. Each target column has zero retained voxel and remains -1. Therefore for these exact saved scans, the target cells never reach slope, residual, neighborhood-rank or flood-connectivity tests: no layer exists to classify. Lowering 20deg/plane thresholds cannot recover nonexistent observations. Ready-time worker map memory is not retained, so this exact replay conclusion is limited to the saved first scan; all saved later point batches also have zero target returns.

## Filtering and visibility

There are two radial filters:

1. `scripts/nav2_mujoco_fixture.py` filters registered points to XY distance `>.45` and `<6` around the *latest* estimated base, before worker batching and before saving `observed_xyz`.
2. `VoxelTerrainMap.update` filters again to distance `>=.45`, radius <=6 and |Z-baseZ|<=3 around that scan's stored estimated pose.

The saved `observed_xyz` is already after the first filter. Minimum retained distances are .4501–.4539 m. **Raw returns before this filter, source IDs, per-point timestamps and exact worker voxel/layer memory were not retained.** It is impossible to truthfully report how many ground points were removed by the first radial filter from these artifacts.

The radial filter is geometrically larger than the original model's local XY bounding rectangle: chassis+wheel envelope half-extents are approximately x=.266, y=.2226, corner radius=.34685 m; .45 m excludes an additional ~.103 m radial margin beyond that corner. This proves the radial heuristic can discard externally located returns; it does not prove that these two cells had any such raw returns.

The actual original `mid360.npy` pattern has elevation range **-7.2123deg to +52.1640deg**, 800,000 directions (SHA in JSON). Both sensors are at original x=.16, y=+-.18, z=.218 and roll outward +-30deg. Using only this declared geometry and level-ground approximation, directions needed to hit these rear-center failed cells have required sensor elevations approximately **-31.6deg to -16.1deg**, entirely below the original downward envelope. This is an explicitly qualified visibility inference, not replayed raw hits; exact full attitude/pose at every return is unavailable. It agrees with actual postfilter clouds' rear-central gap. A reduced radial filter alone therefore is **not a validated solution** for the failed cells.

Raw dual scans are otherwise registered with per-return estimated timestamps/history and source extrinsic, earliest return timestamps the fused scan; unavailable history fails closed. The saved first registration rejects pre-history frames rather than extrapolating. Current evidence does not identify a map-transpose, slope-threshold or scan-TTL defect as the source of these two empty columns.

## Safe concrete next diagnostic/change

Keep current unknown/block gates, .411 m envelope, immutable original fixed start prior and original model intact. Before changing filters, add a bounded first-5-s diagnostic capture at `drain_raw_frames` **before** `process_terrain_points`: registered XYZ, source ID, per-return timestamp or scan identity, estimated pose/quaternion and extrinsic, plus pre/post filter counts by these failed columns. Capture only a few scans or compact counters to avoid large artifacts. Add per-column reasons: no retained voxel, too thick, insufficient neighbors, rank failure, slope/residual failure, unconnected support, occupied override. Then one new owned diagnostic run can distinguish no geometric return from heuristic filtering quantitatively.

If actual prefilter ground points outside the original robot solids are found, a separately versioned perception-only self filter using original robot-body geometry in each timestamped estimated base frame could replace the .45 m heuristic. It must preserve physical self-hit rejection and have no blanket `unknown->free`; it will still leave cells with no real return unknown. Do not ship a smaller radius based only on bounding-box math or synthesize ground under the footprint. If the cone genuinely cannot observe required ground, startup/route/sensor-coverage requirements must be addressed and explicitly validated; this audit has not authorized sensor/model changes or moving prior clearing.

## New actual prefilter capture follow-up

Root subsequently collected the unchanged-classifier diagnostic `runs/uc_live_terrain_diagnostic_20261117`, retaining 71 actual registered scans *before* the .45 m crop. I did not launch it. New evidence supersedes the earlier raw-unavailability limitation for this one new session only; original r2 raw remains unavailable.

- Before 5 simulation seconds: **zero raw points in either target column**; the radial filter removed zero target points.
- Throughout 71 scans: column `(-4,0)` stays raw-empty. Column `(-4,-1)` receives only 10 returns, at 7.202/7.301 s after robot movement; all are already farther than .61 m and survive the existing .45 m filter.
- I reconstructed the exact bounded worker input batches (actual submit times, 100,000-point bound, original per-scan pose and receive/sample timestamps) and replayed all seven updates. Baseline .45 m **exactly reproduces all five actual report counts in every update**.
- Replaying the same points with both fixture and voxel radial filters reduced to .411 m leaves **all seven outputs identical**: observed voxels, free/occupied/unknown counts, reachable support layers and empty target columns. Planning circle stays .411 m throughout this offline diagnostic. The newly admitted thin annulus has 167 raw returns across all 71 scans, but none changes this recorded classified map.

Thus for this actual new session there is direct evidence that reducing .45 to .411 does not repair the geometric blind cells. The run happened to start with a footprint supported by the existing fixed prior; its successful .4 m BT motion is not proof of a filter fix. A failed-startup prefilter capture would additionally test that cohort; do not infer uncollected raw-hit counts from the old failed runs.

Raw counts and sequential replay proof: `.work/live_next_raw_terrain_counts.json`, `.work/live_next_raw_filter_replay.json`. Temporary replay code is `.work/replay_raw_terrain_filters.py`; it performs CPU-only deterministic classification, not a physics rollout. No original/runtime code changed.
