#!/usr/bin/env python3
"""Isolated Point-LIO + original MuJoCo robot on a test plane or prepared UC field.

Original model/estimator parameters remain in their source locations. This
runner changes only the transport domain/namespace and writes private configs.
Ground truth is written to evaluation logs, never to navigation odometry/TF.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_bt.estimated_twist import EstimatedPose2D, PoseDifferenceTwistEstimator
from rl_bt.uc_field_reference import DEFAULT_PREPARED_FIELD, prepared_field_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, default=150.)
    parser.add_argument("--scene", choices=("plane", "rmuc"), default="plane")
    parser.add_argument("--terrain", choices=("plane_height", "voxel"), default="plane_height")
    parser.add_argument("--field-dir", default=DEFAULT_PREPARED_FIELD)
    parser.add_argument("--terrain-execution", choices=("worker", "sync"), default="worker")
    parser.add_argument("--terrain-prior", choices=("clear", "observed"), default="clear")
    parser.add_argument("--start-footprint-prior", action="store_true")
    parser.add_argument("--terrain-sensors", choices=("primary", "dual"), default="primary")
    parser.add_argument("--support-dropout-after-s", type=float)
    parser.add_argument("--capture-terrain-input", action="store_true",
        help="Save bounded registered inputs before near-field filtering for diagnosis")
    parser.add_argument("--simulation-seed", type=int,
        help="Explicit sensor randomization seed; original robot and physics unchanged")
    args = parser.parse_args()
    if os.environ.get("ROS_DOMAIN_ID") != "79":
        raise SystemExit("This private fixture requires isolated ROS_DOMAIN_ID=79")
    args.out.mkdir(parents=True, exist_ok=False)
    original = Path("/opt/mid360_mujoco")
    sys.path.insert(0, str(original / "src"))
    from mid360_mujoco.config import load
    from mid360_mujoco.model import build, robot_config
    from mid360_mujoco.simulation import Simulation
    import rclpy
    from nav_msgs.msg import Odometry, OccupancyGrid
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Bool
    from tf2_ros import Buffer
    from tf2_msgs.msg import TFMessage
    from rl_bt.scan_registration import EstimatedPoseHistory

    config = load(); robot = robot_config()
    if args.simulation_seed is not None:
        if args.simulation_seed < 0:
            raise ValueError("Simulation seed must be nonnegative")
        config["seed"] = args.simulation_seed
    config["namespace"] = robot["namespace"] = "rl_bt_nav_probe"
    config["lidar"]["backend"] = "cpu"
    config["lidar"]["batch_steps"] = 1
    config["physics"]["collision_window_radius_m"] = 4. if args.scene == "rmuc" else 0.
    config["physics"]["cached_forward"] = True
    field = Path(args.field_dir).resolve() if args.scene == "rmuc" else None
    if args.scene == "rmuc" and args.terrain != "voxel":
        raise ValueError("The full UC field requires height-independent voxel terrain classification")
    field_source = None
    if field is not None:
        field_source = prepared_field_reference(field)
        (args.out/"field_source.json").write_text(json.dumps(field_source, indent=2)+"\n")
    xml = build(config, robot, field=field)
    (args.out / "model.xml").write_text(xml)
    (args.out / "simulation.yaml").write_text(yaml.safe_dump(config))
    (args.out / "robot.yaml").write_text(yaml.safe_dump(robot))
    simulation = Simulation(xml, config)
    configured_spawn = robot["spawn"] if field is not None else [0.,0.,.03,0.]
    voxel_terrain = None
    terrain_worker = None
    if args.terrain == "voxel":
        if args.terrain_execution == "worker":
            from rl_bt.terrain_worker import TerrainWorker
            terrain_worker = TerrainWorker(submit_period_s=1.)
        else:
            from rl_bt.voxel_terrain import VoxelTerrainMap
            voxel_terrain = VoxelTerrainMap()

    # Preserve original transport implementation with one isolated domain guard.
    bridge_source = original / "src/mid360_mujoco/ros_bridge.py"
    source = bridge_source.read_text()
    assert 'os.environ.get("ROS_DOMAIN_ID")!="65"' in source
    source = source.replace('os.environ.get("ROS_DOMAIN_ID")!="65"',
                            'os.environ.get("ROS_DOMAIN_ID")!="79"').replace(
                            "requires isolated ROS_DOMAIN_ID=65", "requires isolated ROS_DOMAIN_ID=79")
    (args.out / "private_ros_bridge.py").write_text(source)
    module = types.ModuleType("mid360_mujoco.private_nav2_bridge")
    module.__package__ = "mid360_mujoco"
    exec(compile(source, str(args.out / "private_ros_bridge.py"), "exec"), module.__dict__)

    class GuardedBridge(module.Bridge):
        def __init__(self, sim):
            self.safety_latched = False
            self.command_log = []
            super().__init__(sim)
            self.node.create_subscription(Bool, "navigation_safety_stop", self.safety, 10)

        def safety(self, msg):
            if msg.data:
                self.safety_latched = True
                self.sim.stop()
                self.target[:] = 0.

        def velocity(self, msg):
            if self.safety_latched:
                self.sim.stop(); self.target[:] = 0.; return
            self.command_log.append([self.sim.time_ns/1e9, msg.linear.x, msg.linear.y, msg.angular.z])
            super().velocity(msg)

    bridge = GuardedBridge(simulation)
    bridge.node.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
    estimator = PoseDifferenceTwistEstimator(min_interval_s=.025, max_interval_s=.3,
        max_translation_speed_m_s=4., max_rotation_speed_rad_s=4., max_data_age_s=.5)
    odom_publisher = bridge.node.create_publisher(Odometry, "odometry", 10)
    obstacle_publisher = bridge.node.create_publisher(PointCloud2, "terrain_obstacles", 5)
    support_publisher = bridge.node.create_publisher(OccupancyGrid, "terrain_support",
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                   reliability=ReliabilityPolicy.RELIABLE))
    semantic_publisher = bridge.node.create_publisher(OccupancyGrid, "terrain_support_semantic",
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                   reliability=ReliabilityPolicy.RELIABLE))
    from rl_bt.support_costmap import SupportCostmapProjector, FixedStartSupportPrior
    projector = SupportCostmapProjector(FixedStartSupportPrior((0., 0.))
        if args.start_footprint_prior else None)
    pose_history = EstimatedPoseHistory()
    tf_buffer = Buffer(node=bridge.node)
    # TransformListener's Humble defaults are absolute /tf topics. This node
    # owns relative namespaced transport, so subscribe to that transport here.
    def encoder_tf(message):
        for transform in message.transforms:
            tf_buffer.set_transform(transform, "private_encoder_and_estimated_tf")
    bridge.node.create_subscription(TFMessage, "tf", encoder_tf, 20)
    raw_frames, registration_reports = {}, []
    shared_effective = original/"generated/pointlio-direct-effective.yaml"
    effective_before = hashlib.sha256(shared_effective.read_bytes()).hexdigest() if shared_effective.exists() else None
    estimated, twists, trajectory, obstacle_counts, terrain_reports = [], [], [], [], []
    terrain_submissions, terrain_observations = [], {}
    terrain_pending = []
    last_pending_stamp = None
    discarded_pending_scans = 0
    support_reports = []
    last_pose = None
    captured_terrain_inputs = []
    if args.capture_terrain_input:
        (args.out / "terrain_inputs").mkdir()

    def estimate(message):
        nonlocal last_pose
        pose = message.pose.pose; p, q = pose.position, pose.orientation
        sampled = message.header.stamp.sec+message.header.stamp.nanosec/1e9
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        now = bridge.node.get_clock().now().nanoseconds/1e9
        result = estimator.update(EstimatedPose2D(sampled, p.x, p.y, yaw,
            message.header.frame_id, message.child_frame_id), now_s=now)
        estimated.append([sampled, p.x, p.y, p.z, yaw, result.valid, result.reason])
        if result.valid:
            added = pose_history.add(sampled, [p.x,p.y,p.z], [q.x,q.y,q.z,q.w], message.header.frame_id)
            if not added["accepted"]:
                return
            last_pose = [p.x, p.y, p.z, yaw]
            message.twist.twist.linear.x = result.vx_m_s
            message.twist.twist.linear.y = result.vy_m_s
            message.twist.twist.angular.z = result.wz_rad_s
            odom_publisher.publish(message)
            twists.append([sampled, result.vx_m_s, result.vy_m_s, result.wz_rad_s])

    def publish_obstacles(xyz, sample_s):
        out = PointCloud2(); out.header.frame_id = "odom"
        out.header.stamp = rclpy.time.Time(nanoseconds=round(sample_s*1e9)).to_msg()
        out.height = 1; out.width = len(xyz)
        out.fields = [PointField(name=k, offset=i*4, datatype=PointField.FLOAT32, count=1)
                      for i, k in enumerate(("x", "y", "z"))]
        out.is_bigendian = False; out.point_step = 12; out.row_step = 12*len(xyz)
        out.is_dense = True; out.data = xyz.tobytes()
        obstacle_publisher.publish(out)
        obstacle_counts.append([sample_s, len(xyz)])

    def use_grid(grid, status, sample_s, classification_s, observation=None, scan_count=1):
        now_s = bridge.node.get_clock().now().nanoseconds/1e9
        usable = status == "accepted" and 0 <= now_s-sample_s <= 2.5
        terrain_reports.append({"sampled_at_s":sample_s, "update_status":status,
            "consumed_at_s":now_s, "classification_wall_s":classification_s,
            "published":usable, "batch_scan_count":scan_count, **grid.diagnostics})
        if not usable:
            return
        xyz = np.ascontiguousarray(grid.obstacle_points_xyz, dtype="<f4")
        publish_obstacles(xyz, sample_s)
        if args.terrain_prior == "observed":
            if args.support_dropout_after_s is not None and now_s >= args.support_dropout_after_s:
                terrain_reports[-1]["support_published"] = False
                terrain_reports[-1]["injected_support_dropout"] = True
                return
            terrain_reports[-1]["support_published"] = True
            support = projector.project(grid)
            padding = math.ceil(2./grid.resolution_m)
            origin = grid.origin_xy_m-padding*grid.resolution_m
            strict = np.pad(support.strict_occupancy_xy, padding, constant_values=100)
            semantic = np.pad(support.occupancy_xy, padding, constant_values=-1)
            required_lower = np.asarray(last_pose[:2])-6.1
            required_upper = np.asarray(last_pose[:2])+6.1
            covered = bool(np.all(origin <= required_lower)
                and np.all(origin+np.array(strict.shape)*grid.resolution_m >= required_upper))
            if not covered:
                bridge.safety_latched = True
                raise RuntimeError("Strict support canvas does not cover rolling costmaps")
            message = OccupancyGrid(); message.header.frame_id = grid.frame_id
            message.header.stamp = rclpy.time.Time(nanoseconds=round(sample_s*1e9)).to_msg()
            message.info.map_load_time = message.header.stamp
            message.info.resolution = grid.resolution_m
            message.info.width, message.info.height = map(int, strict.shape)
            message.info.origin.position.x, message.info.origin.position.y = map(float, origin)
            message.info.origin.orientation.w = 1.
            message.data = strict.T.ravel().tolist()
            support_publisher.publish(message)
            message.data = semantic.T.ravel().tolist()
            semantic_publisher.publish(message)
            support_reports.append({"sampled_at_s":sample_s, "blocked_padding_m":padding*grid.resolution_m,
                "current_global_costmap_window_covered":covered, **support.report})
            np.savez_compressed(args.out/"support_latest.npz",
                source_occupancy_xy=grid.occupancy_xy,
                occupancy_xy=semantic, strict_occupancy_xy=strict,
                origin_xy_m=origin, source_origin_xy_m=grid.origin_xy_m, resolution_m=grid.resolution_m,
                support_height_xy_m=grid.support_height_xy_m,
                sampled_at_s=sample_s)
        if len(terrain_reports) in (1, 10, 50) and observation is not None:
            local, sampled_pose = observation
            cells = np.argwhere(grid.occupancy_xy == 0)
            support_xyz = np.column_stack((grid.origin_xy_m+(cells+.5)*grid.resolution_m,
                grid.support_height_xy_m[tuple(cells.T)]))
            np.savez_compressed(args.out/f"terrain_sample_{len(terrain_reports):04d}.npz",
                observed_xyz=local, obstacles_xyz=xyz, support_xyz=support_xyz,
                estimated_pose=sampled_pose)

    def process_terrain_points(points, sample_s):
        nonlocal last_pending_stamp, discarded_pending_scans
        if last_pose is None:
            return
        # Diagnostic only: preserve measured registered points before the
        # existing filters. Never feed these files or simulator truth to RL.
        if args.capture_terrain_input and len(captured_terrain_inputs) < 80:
            target = args.out / "terrain_inputs" / f"scan_{len(captured_terrain_inputs):04d}.npz"
            np.savez_compressed(target, registered_points_xyz=points,
                sampled_at_s=sample_s, estimated_pose_xyzyaw=np.asarray(last_pose),
                consumed_at_s=bridge.node.get_clock().now().nanoseconds/1e9)
            captured_terrain_inputs.append({"path": str(target.relative_to(args.out)),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "sampled_at_s": sample_s, "point_count": len(points),
                "GT_used": False})
        if args.terrain == "plane_height":
            # Plane fixture only; the full UC scene cannot use absolute Z cuts.
            good = np.isfinite(points).all(axis=1) & (points[:, 2] > .04) & (points[:, 2] < 2.)
            good &= np.linalg.norm(points[:, :2]-last_pose[:2], axis=1) > .45
            xyz = np.ascontiguousarray(points[good], dtype="<f4")
            publish_obstacles(xyz, sample_s)
        else:
            finite = np.isfinite(points).all(axis=1)
            distance = np.linalg.norm(points[:, :2]-last_pose[:2], axis=1)
            local = points[finite & (distance > .45) & (distance < 6.)]
            now_s = bridge.node.get_clock().now().nanoseconds/1e9
            if terrain_worker is not None:
                if not 0 <= now_s-sample_s <= .5:
                    terrain_submissions.append([sample_s, "stale_or_future_at_receive"])
                    return
                if last_pending_stamp is not None and sample_s <= last_pending_stamp:
                    terrain_submissions.append([sample_s, "out_of_order_at_receive"])
                    return
                last_pending_stamp = sample_s
                terrain_pending.append({"points_xyz":local.copy(), "sampled_at_s":sample_s,
                    "now_s":now_s, "robot_position_xyz":np.array(last_pose[:3]),
                    "frame_id":"odom"})
                while terrain_pending and (len(terrain_pending) > 30
                        or now_s-terrain_pending[0]["sampled_at_s"] > 2.5
                        or sum(len(scan["points_xyz"]) for scan in terrain_pending) > 100_000):
                    terrain_pending.pop(0); discarded_pending_scans += 1
                status = terrain_worker.submit_batch(terrain_pending)
                terrain_submissions.append([sample_s, status])
                if status == "submitted":
                    sequence = terrain_worker.ready_status()["submitted_samples"]
                    terrain_observations[sequence] = (
                        np.concatenate([scan["points_xyz"] for scan in terrain_pending]), np.array(last_pose))
                    terrain_pending.clear()
                    for old in sorted(terrain_observations)[:-4]:
                        del terrain_observations[old]
            else:
                began = time.perf_counter()
                status = voxel_terrain.update(local, sampled_at_s=sample_s, now_s=now_s,
                    robot_position_xyz=last_pose[:3], frame_id="odom")
                grid = voxel_terrain.snapshot(robot_position_xyz=last_pose[:3], now_s=now_s)
                use_grid(grid, status, sample_s, time.perf_counter()-began,
                    (local, np.array(last_pose)))

    def cloud_arrays(message, timed=False):
        fields = {field.name:field for field in message.fields}
        names = ["x","y","z"] + (["offset_time"] if timed else [])
        if any(name not in fields for name in names):
            raise ValueError("Missing point-cloud field")
        expected = [PointField.FLOAT32]*3 + ([PointField.UINT32] if timed else [])
        if any(fields[name].datatype != kind for name,kind in zip(names,expected)):
            raise ValueError("Unexpected point-cloud field type")
        endian = ">" if message.is_bigendian else "<"
        formats = [endian+"f4"]*3 + ([endian+"u4"] if timed else [])
        dtype = np.dtype({"names":names, "formats":formats,
            "offsets":[fields[name].offset for name in names], "itemsize":message.point_step})
        values = np.ndarray((message.height,message.width), dtype=dtype, buffer=message.data,
            strides=(message.row_step,message.point_step)).reshape(-1)
        xyz = np.column_stack([values[name] for name in names[:3]])
        return xyz, values["offset_time"].copy() if timed else None

    def terrain(message):
        if message.header.frame_id != "odom" or not message.data:
            return
        points, _ = cloud_arrays(message)
        process_terrain_points(points, message.header.stamp.sec+message.header.stamp.nanosec/1e9)

    def raw_scan(message, source_id):
        expected = ("front_mid360", "right_mid360")[source_id]
        if message.header.frame_id != expected or not message.data:
            return
        points, offsets = cloud_arrays(message, timed=True)
        distance = np.linalg.norm(points, axis=1)
        finite = np.isfinite(points).all(axis=1) & (distance > .15) & (distance < 6.5)
        points, offsets = points[finite], offsets[finite]
        if not len(points):
            return
        start_ns = message.header.stamp.sec*1_000_000_000+message.header.stamp.nanosec
        times = (start_ns+offsets.astype(np.int64))/1e9
        frame = raw_frames.setdefault(start_ns, {})
        frame[source_id] = (points, times)
        while len(raw_frames) > 12:
            key = min(raw_frames); del raw_frames[key]
            registration_reports.append({"frame_start_ns":key, "accepted":False, "reason":"raw_queue_overflow"})

    def extrinsic(frame, sample_s):
        transform = tf_buffer.lookup_transform("base_footprint", frame,
            rclpy.time.Time(nanoseconds=round(sample_s*1e9)))
        t, q = transform.transform.translation, transform.transform.rotation
        return np.array([t.x,t.y,t.z]), np.array([q.x,q.y,q.z,q.w])

    def drain_raw_frames():
        if pose_history.bounds_s is None or last_pose is None:
            return
        now_s = bridge.node.get_clock().now().nanoseconds/1e9
        for key in sorted(raw_frames):
            pair = raw_frames[key]
            if len(pair) != 2:
                if now_s-key/1e9 < .5: break
                registration_reports.append({"frame_start_ns":key,"accepted":False,"reason":"missing_sensor"})
                del raw_frames[key]; continue
            lo = min(float(item[1].min()) for item in pair.values())
            hi = max(float(item[1].max()) for item in pair.values())
            if hi > pose_history.bounds_s[1]+1e-4:
                if now_s-lo < .5: break
                registration_reports.append({"frame_start_ns":key,"accepted":False,"reason":"pose_history_timeout"})
                del raw_frames[key]; continue
            if not pose_history.has_coverage(lo,hi):
                registration_reports.append({"frame_start_ns":key,"accepted":False,"reason":"pose_history_not_covering_scan"})
                del raw_frames[key]; continue
            world, detail = [], []
            try:
                for source_id in (0,1):
                    frame_name = ("front_mid360","right_mid360")[source_id]
                    p0,q0 = extrinsic(frame_name,lo)
                    p1,q1 = extrinsic(frame_name,hi)
                    angle = 2*math.acos(min(1.,abs(float(np.dot(q0,q1)))))
                    if np.linalg.norm(p1-p0) > .002 or angle > .01:
                        raise ValueError("Mount motion exceeds rigid extrinsic tolerance")
                    translation, rotation = extrinsic(frame_name,(lo+hi)/2)
                    points, times = pair[source_id]
                    registered = pose_history.register_scan(points,times,translation,rotation)
                    if not registered["accepted"]:
                        raise ValueError(registered["reason"])
                    world.append(registered["points_xyz"])
                    detail.append({"source":source_id,"points":len(points),"mount_rotation_change_rad":angle,
                        "max_pose_interpolation_gap_s":registered["max_interpolation_gap_s"]})
            except Exception as error:
                if now_s-lo < .5 and "extrapolation" in str(error).lower(): break
                registration_reports.append({"frame_start_ns":key,"accepted":False,"reason":str(error)})
                del raw_frames[key]; continue
            # The earliest acquired return timestamps the fused scan, so older
            # points are never refreshed to the end of the scan or publish time.
            process_terrain_points(np.concatenate(world),lo)
            registration_reports.append({"frame_start_ns":key,"accepted":True,"oldest_point_s":lo,
                "newest_point_s":hi,"processed_at_s":now_s,"sources":detail,
                "pose_source":"PointLIO_estimated_base_history","uses_ground_truth":False})
            del raw_frames[key]

    bridge.node.create_subscription(Odometry, "odometry_raw", estimate, 10)
    if args.terrain_sensors == "dual" and args.terrain == "voxel":
        for source_id in (0,1):
            bridge.node.create_subscription(PointCloud2, f"lidar{source_id+1}/points",
                lambda message, index=source_id: raw_scan(message,index), 5)
    else:
        bridge.node.create_subscription(PointCloud2, "registered_scan", terrain, 5)
    params = yaml.safe_load((original / "ros2/config/pointlio_direct.yaml").read_text())
    tilt = math.radians(robot["mount"]["outward_tilt_deg"])
    params["initialization"].update(direct_start=True, require_stationary=False,
        estimate_gyro_bias=False, static_map_duration=0., startup_valid_frames=1,
        startup_guard_frames=0, world_frame="odom")
    params["initialization"].pop("initial_map_base_pose", None)
    params["mapping"].update(extrinsic_T=[0.,0.,0.], extrinsic_R=np.eye(3).ravel().tolist(),
        acc_norm=9.81, satu_acc=4.*9.80665, gravity=[0.,0.,-1.],
        gravity_init=[0., math.sin(tilt), -math.cos(tilt)], lidar_time_inte=1./robot["lidar"]["rate"])
    params["preprocess"].update(blind=.15, scan_line=1, timestamp_unit=0)
    params["pcd_save"]["pcd_save_en"] = False
    private_lio = args.out / "pointlio-direct.yaml"
    private_lio.write_text(yaml.safe_dump(params))
    executable_root = original / "lio_ws/install/pb2025_lio_backends/lib/pb2025_lio_backends"
    processes, logs = [], []
    common = ["--ros-args", "-r", "__ns:=/rl_bt_nav_probe", "-p", "use_sim_time:=true",
              "-r", "/tf:=tf", "-r", "/tf_static:=tf_static"]
    commands = [[str(executable_root / "pointlio_direct"), *common, "-p", f"config_file:={private_lio}",
        "-p", f"effective_config:={args.out/'pointlio-effective.yaml'}"],
        [str(executable_root / "lio_display"), *common, "-r", "odometry:=odometry_raw",
         "-p", "gravity_aligned:=true", "-p", "initial_mount:="+str([
            robot["mount"]["front_x"], robot["mount"]["half_spacing"],
            robot["mount"]["optical_height"], -tilt, 0., 0.])]]
    stopping = False
    def request_stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    started = time.monotonic()
    try:
        for name, command in zip(("pointlio", "lio_display"), commands):
            stream = (args.out / (name+".log")).open("w"); logs.append(stream)
            processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True))
        while not stopping and time.monotonic()-started < args.wall_seconds:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("A Point-LIO transport process exited; inspect fixture logs")
            bridge.poll(); bridge.apply_command()
            if args.terrain_sensors == "dual" and args.terrain == "voxel":
                drain_raw_frames()
            if terrain_worker is not None:
                result = terrain_worker.poll()
                if result is not None:
                    if result.error or result.grid is None:
                        bridge.safety_latched = True
                        raise RuntimeError("Terrain worker failed: "+str(result.error))
                    use_grid(result.grid, result.update_status, result.sampled_at_s,
                        result.classification_wall_s, terrain_observations.pop(result.sequence, None),
                        scan_count=result.scan_count)
            if bridge.safety_latched:
                simulation.stop()
            imus, scans = simulation.step()
            bridge.publish(imus, scans)
            if simulation.tick % 50 == 0:
                trajectory.append([simulation.time_ns/1e9, *simulation.data.qpos[:3].tolist()])
            elapsed_sim = simulation.time_ns/1e9
            delay = elapsed_sim-(time.monotonic()-started)
            if delay > 0:
                time.sleep(min(.002, delay))
    finally:
        simulation.stop()
        worker_status = None
        if terrain_worker is not None:
            worker_status = terrain_worker.ready_status()
            terrain_worker.close()
            worker_status["closed_status"] = terrain_worker.ready_status()
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL)
        for stream in logs: stream.close()
        report = {"artifact_type": "PointLIO_MuJoCo_execution_fixture_not_official_match",
            "scene": args.scene, "field_source": field_source,
            "configured_spawn_world_xyzyaw": configured_spawn,
            "navigation_frame": "PointLIO_session_odom_not_official_referee_frame",
            "ROS_DOMAIN_ID": 79, "namespace": "rl_bt_nav_probe", "GT_localization_input": False,
            "simulation_seconds": simulation.time_ns/1e9, "wall_seconds": time.monotonic()-started,
            "original_robot_and_physics_preserved": True, "sensor_point_rate_each": config["lidar"]["point_rate"],
            "sensor_randomization_seed": config["seed"],
            "transport_change": "domain_guard_65_to_79_and_namespace_only",
            "source_bridge_sha256": hashlib.sha256(bridge_source.read_bytes()).hexdigest(),
            "terrain_classifier": args.terrain, "terrain_reports":terrain_reports,
            "terrain_execution": args.terrain_execution, "terrain_worker":worker_status,
            "terrain_prior":args.terrain_prior, "support_reports":support_reports,
            "terrain_sensors":args.terrain_sensors, "registration_reports":registration_reports,
            "captured_terrain_inputs": captured_terrain_inputs,
            "injected_support_dropout_after_simulation_s":args.support_dropout_after_s,
            "shared_pointlio_effective_sha256_before":effective_before,
            "shared_pointlio_effective_sha256_after":hashlib.sha256(shared_effective.read_bytes()).hexdigest() if shared_effective.exists() else None,
            "fixed_start_footprint_prior":args.start_footprint_prior,
            "terrain_submissions":terrain_submissions,
            "discarded_pending_scans":discarded_pending_scans,
            "safety_stop_latched": bridge.safety_latched,
            "commands": bridge.command_log, "estimated_poses": estimated,
            "estimated_twists": twists, "GT_trajectory_diagnostic_only": trajectory,
            "obstacle_cloud_counts": obstacle_counts}
        if simulation.collision_window:
            report["collision_window"] = {"radius_m":simulation.collision_window.radius,
                "rebuilds":simulation.collision_window.rebuilds,
                "wait_seconds":simulation.collision_window.wait_seconds}
        (args.out / "result.json").write_text(json.dumps(report, indent=2)+"\n")
        bridge.close(); simulation.close()


if __name__ == "__main__":
    main()
