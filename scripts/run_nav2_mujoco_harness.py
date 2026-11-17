#!/usr/bin/env python3
"""Publish an isolated CPU MuJoCo fixture for real ROS 2 Nav2/MPPI checks.

This publishes truth odometry/TF for execution validation. It is deliberately
separate from the original dual-LiDAR simulation and does not validate Point-LIO.
Run under a ROS Python with numpy, scipy, yaml and mujoco available.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="rl_bt_nav_probe")
    parser.add_argument("--start", type=float, nargs=2, default=[1.5, 7.])
    parser.add_argument("--yaw", type=float, default=0.)
    parser.add_argument("--team", choices=[0, 1], type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=180.)
    parser.add_argument("--domain-id", type=int, default=86)
    parser.add_argument("--realtime-factor", type=float, default=1.)
    args = parser.parse_args()
    if args.domain_id == 65:
        raise SystemExit("Domain 65 belongs to the original navigation simulation; choose an isolated domain")
    if os.environ.get("ROS_DOMAIN_ID") != str(args.domain_id):
        raise SystemExit("ROS_DOMAIN_ID must equal --domain-id")
    if args.realtime_factor <= 0:
        raise SystemExit("Positive realtime factor required")
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import Twist, TransformStamped
    from nav_msgs.msg import Odometry, OccupancyGrid
    from rosgraph_msgs.msg import Clock
    from tf2_msgs.msg import TFMessage
    from std_msgs.msg import Bool, String
    from rl_bt.nav2_mujoco import MujocoNavigationFixture, occupancy_for_nav2
    sim = MujocoNavigationFixture(start_xy=args.start, yaw=args.yaw, team=args.team)
    rclpy.init()
    node = Node("rl_bt_mujoco_execution_fixture", namespace=args.namespace)
    clock_pub = node.create_publisher(Clock, "/clock", 10)
    odom_pub = node.create_publisher(Odometry, "odometry", 10)
    tf_pub = node.create_publisher(TFMessage, "/tf", 10)
    static_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    static_pub = node.create_publisher(TFMessage, "/tf_static", static_qos)
    map_pub = node.create_publisher(OccupancyGrid, "/map", static_qos)
    def command(message):
        sim.set_command(message.linear.x, message.linear.y, message.angular.z)
    node.create_subscription(Twist, "cmd_vel", command, 10)
    node.create_subscription(Bool, "navigation_safety_stop", lambda msg: sim.set_safety_stop(bool(msg.data)), 10)
    provenance_pub = node.create_publisher(String, "execution_fixture_provenance", static_qos)
    provenance_pub.publish(String(data=json.dumps({"schema": "ul_ground_raster_mujoco_v1", "odometry": "ground_truth_execution_validation_only", "pointlio": False, "cad_contacts": False})))
    def stamp(seconds):
        ns = round(seconds * 1_000_000_000)
        return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
    fixed = TransformStamped()
    fixed.header.frame_id = "map"
    fixed.child_frame_id = "odom"
    fixed.transform.rotation.w = 1.
    static_pub.publish(TFMessage(transforms=[fixed]))
    grid = OccupancyGrid()
    grid.header.frame_id = "map"
    grid.info.resolution = sim.nav.resolution_m
    grid.info.width, grid.info.height = sim.nav.shape
    grid.info.origin.position.x, grid.info.origin.position.y = sim.nav.origin_xy_m.tolist()
    grid.info.origin.orientation.w = 1.
    occupied = occupancy_for_nav2(sim.nav, args.team)
    # ROS map is row-major [y,x], while the source geometry is indexed [x,y].
    grid.data = (occupied.T.astype("int8") * 100).ravel().tolist()
    last_map = -math.inf
    stopping = False
    def stop_signal(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    started = time.monotonic()
    next_step = started
    try:
        while rclpy.ok() and not stopping and time.monotonic() - started < args.max_wall_seconds:
            rclpy.spin_once(node, timeout_sec=0.)
            if time.monotonic() < next_step:
                time.sleep(min(.002, next_step - time.monotonic()))
                continue
            sim.step(.01)
            next_step += .01 / args.realtime_factor
            state = sim.state()
            ts = stamp(state["time_s"])
            clock_pub.publish(Clock(clock=ts))
            odom = Odometry()
            odom.header.stamp = ts
            odom.header.frame_id = "odom"
            odom.child_frame_id = "base_footprint"
            odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = state["position"]
            q = state["quaternion_wxyz"]
            odom.pose.pose.orientation.w, odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z = q
            v, w = state["local_velocity"], state["local_angular_velocity"]
            odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = v
            odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z = w
            odom_pub.publish(odom)
            transform = TransformStamped()
            transform.header = odom.header
            transform.child_frame_id = "base_footprint"
            transform.transform.translation.x, transform.transform.translation.y = state["position"][:2]
            # Costmap's ground-projected frame omits physical suspension height.
            transform.transform.translation.z = 0.
            transform.transform.rotation = odom.pose.pose.orientation
            tf_pub.publish(TFMessage(transforms=[transform]))
            if state["time_s"] - last_map > 1.:
                grid.header.stamp = ts
                map_pub.publish(grid)
                last_map = state["time_s"]
    finally:
        sim.set_command(0., 0., 0.)
        report = sim.report()
        report["wall_seconds"] = time.monotonic() - started
        report["ros_domain_id"] = args.domain_id
        report["realtime_factor"] = args.realtime_factor
        report["namespace"] = args.namespace
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
