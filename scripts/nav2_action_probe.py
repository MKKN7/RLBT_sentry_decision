#!/usr/bin/env python3
"""BT gate -> real Nav2 actions -> terminal result, in an isolated fixture.

Requires the ROS Humble runtime. No imported robot ground truth is used for
planning, feedback health or task completion. Ground truth may be logged by
the fixture independently for evaluation.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import uuid
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_bt.contracts import TaskIntent
from rl_bt.navigation_contract import CandidateSnapshot, NavigationCandidate, NavigationExecutor
from rl_bt.nav2_transport import sample_freshness, terminal_status, validate_path
from rl_bt.uc_field_reference import DEFAULT_PREPARED_FIELD, prepared_field_reference, navigation_identity
from rl_bt.support_costmap import path_support_check
from rl_bt.voxel_terrain import TerrainGrid


def main():
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry, OccupancyGrid
    from sensor_msgs.msg import PointCloud2
    from nav2_msgs.action import ComputePathToPose, NavigateToPose
    from lifecycle_msgs.srv import GetState
    from std_msgs.msg import Bool, String
    from tf2_ros import Buffer, TransformListener

    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="rl_bt_nav_probe")
    parser.add_argument("--frame", default="odom")
    parser.add_argument("--goal", nargs=2, type=float, default=(1., 0.))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wall-timeout-s", type=float, default=90.)
    parser.add_argument("--cancel-after-s", type=float)
    parser.add_argument("--scene", choices=("plane", "rmuc"), default="plane")
    parser.add_argument("--field-dir", default=DEFAULT_PREPARED_FIELD)
    parser.add_argument("--terrain-prior", choices=("clear", "observed"), default="clear")
    parser.add_argument("--warmup-s", type=float, default=0.)
    args, ros_args = parser.parse_known_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    identity = (navigation_identity(prepared_field_reference(args.field_dir))
        if args.scene == "rmuc" else
        ("synthetic_nav2_fixture", "plane_fixture", "explicit_test_scene_not_UL_UC"))
    rclpy.init(args=ros_args)

    class Probe(Node):
        def __init__(self):
            super().__init__("bt_navigation_action_probe", namespace=args.namespace,
                             parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
            self.events = []
            self.odom = None
            self.terrain_stamp = None
            self.last_health_reason = "not_checked"
            self.current_support_check = None
            self.support_grid = None
            self.costmaps = {}
            self.handles = {}
            self.result = None
            self.sent = False
            self.cancel_sent = False
            self.started_s = None
            self.tf_buffer = Buffer(node=self)
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.create_subscription(Odometry, "odometry", self.on_odom, 10)
            self.create_subscription(PointCloud2, "terrain_obstacles", self.on_terrain, 5)
            self.create_subscription(OccupancyGrid, "terrain_support_semantic", self.on_support, 1)
            for name in ("global_costmap", "local_costmap"):
                self.create_subscription(OccupancyGrid, name+"/costmap",
                    lambda message, key=name: self.costmaps.update({key:message}), 1)
            self.stop_pub = self.create_publisher(Bool, "navigation_safety_stop", 10)
            self.status_pub = self.create_publisher(String, "task_execution_events", 10)
            self.client = ActionClient(self, NavigateToPose, "navigate_to_pose")
            self.planner = ActionClient(self, ComputePathToPose, "compute_path_to_pose")
            self.lifecycle_clients = {name: self.create_client(GetState, name+"/get_state")
                for name in ("planner_server", "controller_server", "bt_navigator")}
            self.lifecycle_active = {name: False for name in self.lifecycle_clients}
            self.lifecycle_futures = {}
            self.navigation_executor = NavigationExecutor(session_id=uuid.uuid4().hex,
                profile_id=identity[0], map_id=identity[1], map_hash=identity[2], frame_id=args.frame,
                acceptance_timeout_s=5., feedback_timeout_s=3., cancel_timeout_s=3.,
                execution_timeout_s=60.)
            self.snapshot = None
            self.timer = self.create_timer(.1, self.tick)

        def now_s(self):
            return self.get_clock().now().nanoseconds/1e9

        def record(self, name, **values):
            event = {"event": name, "sim_time_s": self.now_s(),
                     "wall_monotonic_s": time.monotonic(), **values}
            self.events.append(event)
            self.status_pub.publish(String(data=json.dumps(event)))
            print(json.dumps(event), flush=True)

        def on_odom(self, message):
            self.odom = message

        def on_terrain(self, message):
            if message.header.frame_id == args.frame:
                self.terrain_stamp = message.header.stamp.sec+message.header.stamp.nanosec/1e9

        def on_support(self, message):
            if message.header.frame_id != args.frame or not message.info.width or not message.info.height:
                return
            values = np.asarray(message.data, dtype=np.int8).reshape(
                (message.info.height, message.info.width)).T
            origin = message.info.origin.position
            self.support_grid = TerrainGrid(args.frame,
                message.header.stamp.sec+message.header.stamp.nanosec/1e9,
                np.array([origin.x, origin.y]), message.info.resolution, values,
                np.empty((0,0)), np.empty((0,0)), np.empty((0,3)), {})

        def health(self, *, require_current_support=False):
            self.last_health_reason = "healthy"
            if self.odom is None:
                self.last_health_reason = "odometry_missing"
                return False
            if args.terrain_prior == "observed" and (self.support_grid is None or not sample_freshness(
                    now_s=self.now_s(), sampled_at_s=self.support_grid.sampled_at_s, max_age_s=2.5)[0]):
                self.last_health_reason = "support_map_missing_or_stale"
                return False
            if args.scene == "rmuc" and (self.terrain_stamp is None or not sample_freshness(
                    now_s=self.now_s(), sampled_at_s=self.terrain_stamp, max_age_s=2.5)[0]):
                self.last_health_reason = "obstacle_scan_missing_or_stale"
                return False
            stamp = self.odom.header.stamp
            fresh, _ = sample_freshness(now_s=self.now_s(),
                sampled_at_s=stamp.sec+stamp.nanosec/1e9, max_age_s=.75)
            p = self.odom.pose.pose.position
            if not (fresh and math.isfinite(p.x) and math.isfinite(p.y)
                    and self.odom.header.frame_id == args.frame):
                self.last_health_reason = "odometry_invalid_or_stale"
                return False
            if args.terrain_prior == "observed" and require_current_support:
                self.current_support_check = path_support_check(self.support_grid, [[p.x,p.y]], radius_m=.411)
                if not self.current_support_check["accepted"]:
                    self.last_health_reason = "current_footprint_unsupported"
                    return False
            try:
                transform = self.tf_buffer.lookup_transform(args.frame, "base_footprint", rclpy.time.Time())
            except Exception:
                self.last_health_reason = "TF_unavailable"
                return False
            stamp = transform.header.stamp
            tf_fresh, _ = sample_freshness(now_s=self.now_s(),
                sampled_at_s=stamp.sec+stamp.nanosec/1e9, max_age_s=.75)
            if not tf_fresh: self.last_health_reason = "TF_stale"
            return tf_fresh

        def pose(self):
            pose = PoseStamped()
            pose.header.frame_id = args.frame
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x, pose.pose.position.y = args.goal
            pose.pose.orientation.w = 1.
            return pose

        def tick(self):
            if not self.sent:
                if self.now_s() < args.warmup_s:
                    return
                for name, client in self.lifecycle_clients.items():
                    previous = self.lifecycle_futures.get(name)
                    if client.service_is_ready() and (previous is None or previous.done()):
                        future = client.call_async(GetState.Request())
                        self.lifecycle_futures[name] = future
                        future.add_done_callback(lambda response, key=name: self.lifecycle_active.update(
                            {key: response.result().current_state.id == 3}))
                if not all(self.lifecycle_active.values()):
                    return
                if not self.health() or not self.client.server_is_ready() or not self.planner.server_is_ready():
                    return
                self.sent = True
                self.started_s = self.now_s()
                goal = ComputePathToPose.Goal()
                goal.goal = self.pose(); goal.planner_id = "GridBased"
                goal.use_start = False
                self.record("compute_path_requested", goal_xy=args.goal)
                for key, message in self.costmaps.items():
                    (args.out/(key+"_at_plan.json")).write_text(json.dumps({
                        "frame":message.header.frame_id, "width":message.info.width,
                        "height":message.info.height, "resolution_m":message.info.resolution,
                        "origin_xy_m":[message.info.origin.position.x, message.info.origin.position.y],
                        "data":list(message.data)})+"\n")
                if self.support_grid is not None:
                    np.savez_compressed(args.out/"semantic_support_at_plan.npz",
                        occupancy_xy=self.support_grid.occupancy_xy,
                        origin_xy_m=self.support_grid.origin_xy_m,
                        resolution_m=self.support_grid.resolution_m,
                        sampled_at_s=self.support_grid.sampled_at_s)
                future = self.planner.send_goal_async(goal)
                future.add_done_callback(self.planner_accepted)
                return
            if self.snapshot is None or self.result is not None:
                return
            if (args.cancel_after_s is not None and not self.cancel_sent
                    and self.now_s()-self.started_s >= args.cancel_after_s):
                self.cancel_sent = True
                self.apply(self.navigation_executor.stop(now_s=self.now_s(), reason="explicit_probe_cancellation"))
            if self.navigation_executor.active is not None:
                healthy = self.health(require_current_support=True)
                self.apply(self.navigation_executor.watchdog(now_s=self.now_s(), alive=True,
                    pose_valid=healthy, frame_ready=healthy))

        def planner_accepted(self, future):
            try:
                handle = future.result()
                if not handle.accepted:
                    self.result = "planner_goal_rejected"; return
                handle.get_result_async().add_done_callback(self.path_ready)
            except Exception as error:
                self.result = "planner_exception"; self.record("exception", detail=str(error))

        def path_ready(self, future):
            response = future.result()
            path = response.result.path
            poses = [(p.header.frame_id or path.header.frame_id,
                      p.pose.position.x, p.pose.position.y) for p in path.poses]
            valid, reason = validate_path(poses, expected_frame=args.frame,
                goal_xy=args.goal, endpoint_tolerance_m=.2)
            self.record("compute_path_result", status=response.status, valid=valid,
                        reason=reason, path_poses=len(poses), path_xy=[[p[1], p[2]] for p in poses])
            if response.status != 4 or not valid or not self.health():
                self.stop_pub.publish(Bool(data=True))
                self.result = "invalid_or_failed_plan"; return
            if args.terrain_prior == "observed":
                position = self.odom.pose.pose.position
                check = path_support_check(self.support_grid,
                    [[position.x,position.y], *[[p[1],p[2]] for p in poses]], radius_m=.411)
                self.record("path_support_check", **check,
                    support_sampled_at_s=self.support_grid.sampled_at_s)
                if not check["accepted"]:
                    self.result = "unsupported_path"
                    self.stop_pub.publish(Bool(data=True))
                    return
            now = self.now_s()
            candidate = NavigationCandidate(0, "navigate", "engineering_navigation_goal",
                (*args.goal, 0., 0.), "observed_session_ground" if args.scene == "rmuc" else "fixture_plane", "ground")
            self.snapshot = CandidateSnapshot(*identity, args.frame, 1, now, 2., (candidate,))
            self.apply(self.navigation_executor.propose(TaskIntent(0, 1, 1, now), self.snapshot,
                now_s=now, alive=True, pose_valid=self.health(), frame_ready=self.health()))

        def apply(self, decision):
            if decision.commands or decision.status in ("rejected", "terminal", "ignored"):
                self.record("BT_decision", decision=asdict(decision))
            for command in decision.commands:
                if command.kind == "send_goal":
                    goal = NavigateToPose.Goal(); goal.pose = self.pose()
                    future = self.client.send_goal_async(goal,
                        feedback_callback=lambda feedback, task=command.task_id: self.on_feedback(task, feedback))
                    future.add_done_callback(lambda result, task=command.task_id: self.goal_accepted(task, result))
                elif command.kind == "cancel_goal":
                    self.record("navigation_health_at_cancel", health_reason=self.last_health_reason,
                        support_sampled_at_s=None if self.support_grid is None else self.support_grid.sampled_at_s,
                        obstacle_sampled_at_s=self.terrain_stamp, current_support_check=self.current_support_check)
                    handle = self.handles.get(command.task_id)
                    if handle is not None:
                        handle.cancel_goal_async().add_done_callback(
                            lambda result, task=command.task_id: self.cancel_ack(task, result))
                    # Before acceptance, preserve the canceling slot; the late
                    # acceptance callback immediately requests cancellation.
                elif command.kind == "require_safety_stop":
                    self.stop_pub.publish(Bool(data=True))
                    self.record("safety_stop_requested", task_id=command.task_id, reason=command.reason)

        def goal_accepted(self, task, future):
            handle = future.result()
            now = self.now_s()
            self.record("navigation_goal_response", task_id=task, accepted=handle.accepted,
                        action_uuid=[int(byte) for byte in handle.goal_id.uuid])
            if not handle.accepted:
                self.stop_pub.publish(Bool(data=True))
                self.apply(self.navigation_executor.feedback(task_id=task, status="rejected", sampled_at_s=now, now_s=now))
                self.result = "rejected"; return
            self.handles[task] = handle
            self.apply(self.navigation_executor.feedback(task_id=task, status="accepted", sampled_at_s=now, now_s=now))
            handle.get_result_async().add_done_callback(lambda result: self.goal_result(task, result))
            if self.navigation_executor.state in {"canceling", "uncertain"}:
                handle.cancel_goal_async().add_done_callback(lambda result: self.cancel_ack(task, result))

        def cancel_ack(self, task, future):
            response = future.result()
            self.record("cancel_response", task_id=task, return_code=response.return_code,
                        goals_canceling=len(response.goals_canceling))
            if response.goals_canceling:
                now = self.now_s()
                self.apply(self.navigation_executor.feedback(task_id=task, status="cancel_accepted",
                                                  sampled_at_s=now, now_s=now))

        def on_feedback(self, task, message):
            feedback = message.feedback
            stamp = feedback.current_pose.header.stamp
            sampled = stamp.sec+stamp.nanosec/1e9
            now = self.now_s()
            if sampled <= 0 or sampled > now+1e-6:
                self.record("invalid_feedback_stamp", sampled_at_s=sampled); return
            self.apply(self.navigation_executor.feedback(task_id=task, status="progress",
                                              sampled_at_s=min(now, sampled), now_s=now))

        def goal_result(self, task, future):
            response = future.result()
            now = self.now_s()
            status = terminal_status(response.status)
            decision = self.navigation_executor.feedback(task_id=task, status=status, sampled_at_s=now, now_s=now)
            self.apply(decision)
            self.record("navigation_terminal_result", task_id=task, status=status)
            if decision.status == "terminal":
                if status != "succeeded":
                    # A failed/canceled action need not imply physical zero
                    # velocity. The isolated safety adapter latches stop.
                    self.stop_pub.publish(Bool(data=True))
                    self.record("terminal_safety_stop_requested", task_id=task, status=status)
                self.result = status

    node = Probe()
    started = time.monotonic()
    error = None
    try:
        while rclpy.ok() and node.result is None and time.monotonic()-started < args.wall_timeout_s:
            rclpy.spin_once(node, timeout_sec=.05)
        if node.result is None:
            node.result = "wall_timeout"
            node.stop_pub.publish(Bool(data=True))
            if node.navigation_executor.active is not None:
                node.apply(node.navigation_executor.stop(now_s=node.now_s(), reason="probe_wall_timeout"))
                until = time.monotonic()+3.
                while node.navigation_executor.active is not None and time.monotonic() < until:
                    rclpy.spin_once(node, timeout_sec=.05)
    except Exception as exc:
        error = repr(exc)
        node.stop_pub.publish(Bool(data=True))
        raise
    finally:
        report = {"artifact_type": "real_Nav2_action_probe_fixture_not_match_acceptance",
            "result": node.result, "error": error, "namespace": args.namespace,
            "goal_xy": args.goal, "frame": args.frame, "wall_seconds": time.monotonic()-started,
            "scene": args.scene, "profile_map_identity": identity,
            "terrain_prior":args.terrain_prior, "warmup_simulation_s":args.warmup_s,
            "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                ("rl_bt/navigation_contract.py", "rl_bt/nav2_transport.py", "scripts/nav2_action_probe.py")},
            "GT_used_as_localization": False, "events": node.events}
        (args.out / "result.json").write_text(json.dumps(report, indent=2)+"\n")
        node.destroy_node(); rclpy.shutdown()
    if node.result not in ("succeeded", "canceled"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
