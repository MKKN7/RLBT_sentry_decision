# Idle drift source audit — 2026-11-17

Read-only current-source audit. No original files, project source, configs, physics, controller, model, ROS session or existing runs were changed/launched. Only this diagnosis file was written. Parent owns live runs.

## Finding

There is no evidence that idle drift is caused by missing/stale zero `cmd_vel` transport. The original drive is a **zero-velocity proportional wrench controller**, not a position brake. The UC test spawn lies on an approximately 1.5 degree downhill surface with low contact friction. Current source and recorded movement support residual slope creep as a physical mechanism. They do not establish which individual contact, initial impulse or estimator transient contributes how much; those quantities are not in the old logs.

Calling `stop()` every idle step would be redundant in the command-free case: it zeros the same command and smoothed target that already remain zero. It does **not** zero physical velocity, lock a joint, add integral/gravity compensation or hold a world position.

## Current source evidence

Container: `Combat_Sentry2026`, originals under `/opt/mid360_mujoco`.

- `src/mid360_mujoco/simulation.py:64`: command and target begin as four zeros.
- `simulation.py:80-81`: `stop()` calls `command_velocity()` and clears target only.
- `simulation.py:83-102`: every physics step computes body-local velocity (`mj_objectVelocity(..., flg_local=1)`), proportional translational force `(target_xy - measured_local_velocity_xy) * velocity_gain * 15`, proportional yaw torque, and rotates them to world applied wrench. Translational z force is zero; there is no integral or position-hold state. Force is enabled when either geom of any current contact has group 2. This is a broad contact predicate, not a per-wheel grounded/load estimator; no recorded contact diagnostics justify changing it here.
- `simulation.py:104-120`: `drive()` runs before each original cached integration step. Collision-window update does not silently advance physics.
- `src/mid360_mujoco/ros_bridge.py:49-50,82-86`: bridge target begins zero and command receipt times are `-inf`; `apply_command()` zeros stale channels and calls `sim.command_velocity(*target)`.
- Current `scripts/nav2_mujoco_fixture.py:449`: every loop performs `bridge.poll(); bridge.apply_command()` before the original `simulation.step()`. Its `GuardedBridge` zeroes both sim and bridge targets on safety latch, and the main loop calls `simulation.stop()` while latched. Missing periodic ROS zero publishes therefore do not explain this no-command record.
- `src/mid360_mujoco/model.py:24`: every unoverridden geom uses friction `0.02 0.001 0.0001`, `condim=3`. Original wheel contact spheres and field collision hulls inherit this default. The wheel rotation joints are passive, no wheel actuator/brake is defined; the only XML actuator is the lidar mount velocity actuator (`model.py:94-95`).
- `configs/simulation.yaml`: `velocity_gain=8`, force limit 80 N, gravity 9.80665 m/s². `physics.settle_seconds=2` is a CLI motion-delay setting (`cli.py:46`), not a Simulation constructor settling step. Current fixture never consumes it.
- Robot spawn `[3.4, 9.5, 0.31, 0]` starts above local terrain (base settles approximately 5 cm in the first second). Point-LIO is launched before stepping; fixture initialization uses `direct_start=True`, `require_stationary=False`, `estimate_gyro_bias=False`, `startup_valid_frames=1`, `startup_guard_frames=0`. It admits startup motion; no stationary/settled admission barrier exists. This is relevant to map-frame settling but does not itself produce physical forces.
- Copied failed fixture reports `collision_window.rebuilds=0`, radius 4 m. Window replacement cannot explain its startup/downhill drift.

## Geometry independently checked from the exact original prepared mesh

Read `assets/navigation_reference/uc_original/field.stl` directly as binary STL; SHA256 `93ff0118af176f388c171b239acfdcac72c17cbe0b523350d5d916d3236fe10b` matches the actual run's `field_source.json`. Intersected vertical lines with original triangles using barycentric coordinates; no simulator launch.

At spawn XY `(3.4,9.5)`, triangle 260805 has height **0.26101457845 m**, upward unit normal `[0.00025581889,0.02701344041,0.99963503770]`, slope **1.54801385375 degrees**. Its height drops to 0.25966341331 at y=9.55 and 0.25696108302 at y=9.65. At y=9.672, adjacent triangle 260804 has height 0.25636745708 and slope 1.46758899525 degrees. A separate z=0 plane intersection exists below this raised support; it must not be mistaken for the chassis contact surface.

Front/back wheel-area XY samples `(3.21,9.3)`, `(3.59,9.3)`, `(3.21,9.7)`, `(3.59,9.7)` have raised surface slopes 1.47866, 1.54801, 1.54801, 1.46759 degrees, respectively. Therefore the actual local field is not a perfectly horizontal launch pad.

XML inertials total **17.53 kg**. As a simplified force-balance estimate, `m g sin(1.548°) = 4.6441 N`; `mu m g cos(1.548°)` at mu=.02 is 3.4370 N. The controller's 120 N/(m/s) proportional gain leaves a rough residual speed `(4.6441-3.4370)/120 = .01006 m/s`. This order-of-magnitude estimate is **not** a solved MuJoCo contact model: wheel rolling, contact normals/counts, hull approximation, chassis attitude, suspension and transient impulses matter. It only explains why zero target plus P control does not guarantee zero physical motion on this low-friction slope.

## Actual idle trace (diagnostic GT only)

Read `runs/uc_live_bt_demonstrations_r2_20261117/sessions/episode_000/episode_000/execution/fixture/result.json`; zero received MPPI commands, true safety latch at final report, simulation 25.15 s.

| sim time (s) | GT x (m) | GT y (m) | GT z (m) |
| --- | --- | --- | --- |
| .05 | 3.399999988 | 9.500000000 | .297496943 |
| .5 | 3.400013699 | 9.503373801 | .261353659 |
| 1 | 3.400014501 | 9.507965405 | .261231960 |
| 2 | 3.400021852 | 9.516929785 | .260995414 |
| 3 | 3.399932645 | 9.524533726 | .260807517 |
| 5 | 3.399941988 | 9.541811333 | .260345052 |
| 10 | 3.399609566 | 9.581413614 | .259321401 |
| 20 | 3.400445557 | 9.671468278 | .256914869 |
| 25 | 3.400419249 | 9.672027454 | .256907403 |

Mean +Y speed is .00896 m/s over 1–2 s, .00864 over 3–5 s, .00792 over 5–10 s, .00901 over 10–20 s. Downhill direction and magnitude match the rough physical explanation. Motion largely subsides after ~20 s; old logs do not contain contact/wrench/attitude and precise latch-time series to identify its cause. Do not claim exact steady creep through the entire episode.

GT is for this physical-drift diagnosis only, never actor input, reward, navigation localization, prior placement or success relabeling. Existing support findings remain: a fixed `(0,0)` prior and an initially shifted Point-LIO footprint can leave unsupported edge cells; moving the prior to every pose or merely translating coordinates does not create observed support.

## Narrow next steps, ranked

1. **Instrument an isolated fixture revision first.** At bounded rate capture original command/target, body-local measured twist, actual applied wrench, base attitude, contacts touching robot versus scene-only contacts, constraint force norms, suspension deflections, latch timing and Point-LIO first estimate/stationarity. Mark mechanical truth diagnostics separately from policy observation. Keep original model/config/source hashes unchanged. Compare physical zero-command drift and estimator-relative drift under one fixed, documented startup protocol; keep every admission failure.
2. **Separate settling and map admission without fabricating support.** A fixed finite settling phase and explicit stationary/estimator-quality readiness can remove falling/contact transients from initial mapping. Startup timeout must count as an admission failure; do not repeatedly restart until it passes. Waiting 2 s alone cannot guarantee stationarity given the observed continued creep. A prior center may be fixed once in a justified session coordinate, but an arbitrary coordinate shift or a moving prior is not a fix for unknown cells.
3. **If physical holding is required, treat it as an explicit private drive-controller revision.** A bounded PI zero-velocity/holding brake with reset/antiwindup and known enable/release rules can provide static load compensation while preserving the original model, friction, masses, physics and field mesh. It belongs to execution/BT/controller, never RL velocities. Such a revision requires actual isolated instrumented comparison and overshoot/release tests; source-only reasoning is not acceptance. A plain periodic zero `cmd_vel`/`stop()` initialization patch has no justified effect here. No controller change is implemented in this audit.

Do not solve by increasing friction, flattening original UC terrain, changing robot dimensions/physics, disabling unknown support, shrinking .411 m footprint, enlarging the fixed prior, teleporting state, or using GT to populate navigation.

## Provenance hashes

Original current source SHA256:

- `simulation.py`: `aab97ba8c6dc4f30f5d623fb0982f657c88e3060d080ce030e6c8a17de5de203`
- `model.py`: `37904b4c7f644ebb1eeef73e76c644da2fd2d3fc0ae84b6e367728672cd3cc7f`
- `ros_bridge.py`: `3f2500de77560a9a9a045bb9f51b4eacdaac5fc9378240c7514a969124fd6562`
- `collision_window.py`: `6822740f0b0d273c86a052967a5c90ad92254cec06411aa02ebc1fdb31b935d5`
- original robot.yaml: `602e2d57d9b577ec23d6381ca5a13fb64eaf2e37c04d66be4ed4e50d016fd128`
- original simulation.yaml: `a6c99161c2825b83f16de04ea7ec1b075aca27e4e89c01979c895e5126c96355`

Read actual episode-0 artifact SHA256:

- `fixture/result.json`: `b3fb0677e47447065a6fe7eb3cae9903ca6ce989011a455394dd84e2bcf4bfae`
- `fixture/simulation.yaml`: `2316a24dcba674bfe4266f618d647f9bef452458b8eb70cdca0a0a469af48002`
- `fixture/robot.yaml`: `5a59819498e64b2e19260c3980d2b84ba4183bc80cafcb59c6f5ba212f6e5380`
- `fixture/model.xml`: `e11d8393fdd60b6ed38a8e7843dd139240ff397896d3378e23f54ee23a6f8523`
- `fixture/private_ros_bridge.py`: `3bdce3d4e8dad8b315b1f70aed17c19de2c0cb340c16da1fa8d91023f9afff67`

Tool fallback: container has no `rg`, so source reads used `cat`, `nl`, and `grep`; no dependency installation was attempted.
