#!/usr/bin/env python3
"""JSONL RPC for a single real Point-LIO/Nav2/MPPI MuJoCo episode.

Only estimated odometry, live semantic support and Nav2 action feedback enter
observations.  This process never publishes velocity or releases a safety stop.
The parent runner owns and cleans up the isolated simulation process groups.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import select
import signal
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_bt.contracts import TaskIntent
from rl_bt.navigation_contract import CandidateSnapshot, NavigationCandidate, NavigationExecutor
from rl_bt.nav2_transport import sample_freshness, terminal_status, validate_path
from rl_bt.support_costmap import path_support_check
from rl_bt.uc_field_reference import navigation_identity, prepared_field_reference
from rl_bt.voxel_terrain import TerrainGrid


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def validate_request(request, *, version, issued_at_s, now_s, mask):
    """Admission check uses the exact last returned snapshot, never a new mask."""
    if not isinstance(request, dict):
        return "request_not_object"
    action = request.get("action")
    if type(action) is not int or not 0 <= action < 16:
        return "invalid_action"
    if type(request.get("candidate_version")) is not int or request["candidate_version"] != version:
        return "candidate_version_mismatch"
    issued = request.get("issued_at_s")
    if isinstance(issued, bool) or not isinstance(issued, (int, float)) or not math.isfinite(issued):
        return "invalid_issued_at_s"
    if abs(float(issued)-issued_at_s) > 1e-6:
        return "issued_time_does_not_match_snapshot"
    if not issued <= now_s <= issued+2.:
        return "expired_or_future_intent"
    if len(mask) != 16 or not bool(mask[action]):
        return "unreachable_or_illegal"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--namespace", default="rl_bt_nav_probe")
    parser.add_argument("--goal", nargs=2, type=float, default=(0., .4))
    parser.add_argument("--scene", choices=("rmuc", "plane"), default="rmuc")
    parser.add_argument("--field-dir", default="/tmp/rl_bt_nav_probe/fields/rmuc2027_original")
    parser.add_argument("--warmup-s", type=float, default=5.)
    parser.add_argument("--task-timeout-s", type=float, default=20.)
    parser.add_argument("--decision-dt-s", type=float, default=.5)
    parser.add_argument("--startup-wall-timeout-s", type=float, default=90.)
    parser.add_argument("--step-wall-timeout-s", type=float, default=15.)
    parser.add_argument("--host-idle-timeout-s", type=float, default=5.)
    args, ros_args = parser.parse_known_args()
    if args.scene != "rmuc":
        parser.error("This live RL schema requires actual observed UC voxel support")
    for value in (args.task_timeout_s, args.decision_dt_s, args.startup_wall_timeout_s,
                  args.step_wall_timeout_s, args.host_idle_timeout_s):
        if not math.isfinite(value) or value <= 0:
            parser.error("Finite positive timeouts required")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/"observation_grids").mkdir()
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry, OccupancyGrid
    from nav2_msgs.action import ComputePathToPose, NavigateToPose
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import Bool
    from lifecycle_msgs.srv import GetState
    from tf2_ros import Buffer, TransformListener
    from rl_bt.live_nav_observation import build_live_observation

    identity = navigation_identity(prepared_field_reference(args.field_dir))
    rclpy.init(args=ros_args)

    class Session(Node):
        def __init__(self):
            super().__init__("rl_live_navigation_session", namespace=args.namespace,
                parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)])
            self.odom = None
            self.support_grid = None
            self.terrain_stamp = None
            self.tf_buffer = Buffer(node=self)
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.create_subscription(Odometry, "odometry", lambda m: setattr(self, "odom", m), 10)
            self.create_subscription(PointCloud2, "terrain_obstacles", self.on_obstacle, 5)
            self.create_subscription(OccupancyGrid, "terrain_support_semantic", self.on_support,
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                           reliability=ReliabilityPolicy.RELIABLE))
            self.create_subscription(Twist, "cmd_vel", self.on_command, 10)
            self.stop_pub = self.create_publisher(Bool, "navigation_safety_stop", 10)
            self.planner = ActionClient(self, ComputePathToPose, "compute_path_to_pose")
            self.client = ActionClient(self, NavigateToPose, "navigate_to_pose")
            self.lifecycle_clients = {name: self.create_client(GetState, name+"/get_state")
                for name in ("planner_server", "controller_server", "bt_navigator")}
            self.lifecycle_active = {name: False for name in self.lifecycle_clients}
            self.lifecycle_futures = {}
            self.bt_gate = NavigationExecutor(session_id=uuid.uuid4().hex,
                profile_id=identity[0], map_id=identity[1], map_hash=identity[2], frame_id="odom",
                acceptance_timeout_s=5., feedback_timeout_s=3., cancel_timeout_s=3.,
                execution_timeout_s=args.task_timeout_s)
            self.handles = {}
            self.events = []
            self.event_cursor = 0
            self.counts = {"planner_requests": 0, "planner_successes": 0,
                "goal_requests": 0, "accepted_goals": 0, "terminal_results": 0,
                "MPPI_cmd_vel_samples": 0, "BT_veto_count": 0, "safety_stops": 0}
            self.version = 0
            self.sequence = 0
            self.last_observation = None
            self.last_goals = None
            self.last_snapshot_s = None
            self.last_health_reason = "startup"
            self.current_support_check = None
            self.last_veto = False
            self.last_arrival = False
            self.last_recovery = False
            self.last_progress_m = 0.
            self.start_distance = None
            self.task_started_s = None
            self.active_started_s = None
            self.planning = False
            self.planning_goal = None
            self.planning_generation = 0
            self.terminal_status = None
            self.mission_success = False
            self.safety_latched = False
            self.truncated = False
            self.reset_done = False
            self.last_host_request_wall_s = time.monotonic()
            self.timer = self.create_timer(.1, self.watchdog)

        def now_s(self):
            return self.get_clock().now().nanoseconds/1e9

        def record(self, event, **values):
            item = {"event": event, "sim_time_s": self.now_s(), **json_value(values)}
            self.events.append(item)
            with (args.out/"events.jsonl").open("a") as stream:
                stream.write(json.dumps(item, allow_nan=False)+"\n")

        def on_command(self, message):
            self.counts["MPPI_cmd_vel_samples"] += 1

        def on_obstacle(self, message):
            if message.header.frame_id == "odom":
                self.terrain_stamp = message.header.stamp.sec+message.header.stamp.nanosec/1e9

        def on_support(self, message):
            if message.header.frame_id != "odom" or not message.info.width or not message.info.height:
                return
            values = np.asarray(message.data, dtype=np.int8).reshape(
                (message.info.height, message.info.width)).T
            position = message.info.origin.position
            self.support_grid = TerrainGrid("odom",
                message.header.stamp.sec+message.header.stamp.nanosec/1e9,
                np.array([position.x, position.y]), message.info.resolution, values,
                np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 3)), {})

        def age(self, stamp):
            return max(0., self.now_s()-stamp) if stamp is not None else math.inf

        def pose_xy(self):
            if self.odom is None:
                return [0., 0.]
            p = self.odom.pose.pose.position
            return [p.x, p.y]

        def health(self, require_current_support=False):
            self.last_health_reason = "healthy"
            if self.odom is None:
                self.last_health_reason = "odometry_missing"
                return False
            if self.support_grid is None or not sample_freshness(now_s=self.now_s(),
                    sampled_at_s=self.support_grid.sampled_at_s, max_age_s=2.5)[0]:
                self.last_health_reason = "support_map_missing_or_stale"
                return False
            if self.terrain_stamp is None or not sample_freshness(now_s=self.now_s(),
                    sampled_at_s=self.terrain_stamp, max_age_s=2.5)[0]:
                self.last_health_reason = "obstacle_scan_missing_or_stale"
                return False
            stamp = self.odom.header.stamp
            fresh = sample_freshness(now_s=self.now_s(),
                sampled_at_s=stamp.sec+stamp.nanosec/1e9, max_age_s=.75)[0]
            if not fresh or self.odom.header.frame_id != "odom" or not all(math.isfinite(x) for x in self.pose_xy()):
                self.last_health_reason = "odometry_invalid_or_stale"
                return False
            try:
                transform = self.tf_buffer.lookup_transform("odom", "base_footprint", rclpy.time.Time())
                stamp = transform.header.stamp
                fresh = sample_freshness(now_s=self.now_s(),
                    sampled_at_s=stamp.sec+stamp.nanosec/1e9, max_age_s=.75)[0]
            except Exception:
                self.last_health_reason = "TF_unavailable"
                return False
            if not fresh:
                self.last_health_reason = "TF_stale"
                return False
            if require_current_support:
                self.current_support_check = path_support_check(self.support_grid, [self.pose_xy()], radius_m=.411)
                if not self.current_support_check["accepted"]:
                    self.last_health_reason = "current_footprint_unsupported"
                    return False
            return True

        def latch(self, reason):
            if not self.safety_latched:
                self.safety_latched = True
                self.counts["safety_stops"] += 1
                self.record("safety_stop_latched", reason=reason)
            self.stop_pub.publish(Bool(data=True))

        def watchdog(self):
            if not self.reset_done or self.mission_success or self.terminal_status in {"failed", "canceled", "rejected"}:
                return
            if self.bt_gate.active is not None:
                healthy = self.health(require_current_support=True)
                self.apply(self.bt_gate.watchdog(now_s=self.now_s(), alive=True,
                    pose_valid=healthy, frame_ready=healthy))
            if self.task_started_s is not None and self.now_s()-self.task_started_s >= args.task_timeout_s:
                # The task explicitly exposes its finite deadline to the actor:
                # missing it is an MDP terminal failure, not a training cut.
                self.terminal_status = "failed"
                self.planning_generation += 1
                self.planning = False
                self.apply(self.bt_gate.stop(now_s=self.now_s(), reason="live_task_time_limit"))
                self.latch("live_task_time_limit")

        def wall_watchdog(self):
            # A ROS simulation-time timer cannot detect a stalled /clock.
            if (self.reset_done and (self.bt_gate.active is not None or self.planning)
                    and time.monotonic()-self.last_host_request_wall_s > args.host_idle_timeout_s):
                self.planning_generation += 1
                self.planning = False
                self.apply(self.bt_gate.stop(now_s=self.now_s(), reason="host_rpc_idle_timeout"))
                self.latch("host_rpc_idle_timeout")

        def pose_message(self, goal):
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x, pose.pose.position.y = map(float, goal)
            pose.pose.orientation.w = 1.
            return pose

        def lifecycle_poll(self):
            for name, client in self.lifecycle_clients.items():
                previous = self.lifecycle_futures.get(name)
                if client.service_is_ready() and (previous is None or previous.done()):
                    future = client.call_async(GetState.Request())
                    self.lifecycle_futures[name] = future
                    def callback(response, key=name):
                        try:
                            self.lifecycle_active[key] = response.result().current_state.id == 3
                        except Exception:
                            self.lifecycle_active[key] = False
                    future.add_done_callback(callback)

        def raw(self):
            healthy = self.health()
            odom_stamp = None if self.odom is None else self.odom.header.stamp.sec+self.odom.header.stamp.nanosec/1e9
            tf_age = math.inf
            try:
                stamp = self.tf_buffer.lookup_transform("odom", "base_footprint", rclpy.time.Time()).header.stamp
                tf_age = self.age(stamp.sec+stamp.nanosec/1e9)
            except Exception:
                pass
            active = self.bt_gate.active
            goal = None if active is None else list(active.candidate.goal_xyzyaw[:2])
            if goal is None and self.planning:
                goal = self.planning_goal
            speed = 0. if self.odom is None else math.hypot(self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y)
            distance = math.dist(self.pose_xy(), args.goal)
            return {"now_s": self.now_s(), "pose_xy_m": self.pose_xy(),
                "pose_age_s": self.age(odom_stamp), "tf_age_s": tf_age,
                "obstacle_age_s": self.age(self.terrain_stamp),
                "support_age_s": self.age(None if self.support_grid is None else self.support_grid.sampled_at_s),
                "health": healthy, "health_reason": self.last_health_reason,
                "active_goal_xy_m": goal, "executor_state": "planning" if self.planning else self.bt_gate.state,
                "active_age_s": 0. if self.active_started_s is None else self.now_s()-self.active_started_s,
                "last_progress_m": self.last_progress_m, "task_goal_xy_m": list(args.goal),
                "task_elapsed_s": 0. if self.task_started_s is None else self.now_s()-self.task_started_s,
                "task_timeout_s": args.task_timeout_s, "last_veto": self.last_veto,
                "last_arrival": self.last_arrival, "last_recovery": self.last_recovery,
                "safety_latched": self.safety_latched, "terminal_status": self.terminal_status,
                "estimated_speed_mps": speed, "task_goal_distance_m": distance,
                "task_progress_m": 0. if self.start_distance is None else self.start_distance-distance,
                "mission_success": self.mission_success,
                "GT_used_as_localization_or_actor_input": False}

        def response(self, kind, request_id, *, elapsed_s=0., action=None, reason=None):
            raw = self.raw()
            obs, goals, diag = build_live_observation(raw, self.support_grid)
            self.version += 1
            self.last_observation, self.last_goals = obs, goals
            self.last_snapshot_s = raw["now_s"]
            path = args.out/"observation_grids"/(f"{self.version:06d}.npz")
            grid_hash = None
            if self.support_grid is not None:
                np.savez_compressed(path, occupancy_xy=self.support_grid.occupancy_xy,
                    origin_xy_m=self.support_grid.origin_xy_m,
                    resolution_m=self.support_grid.resolution_m,
                    sampled_at_s=self.support_grid.sampled_at_s)
                grid_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            raw.update(candidate_goals_xy_m=goals, diagnostics=diag,
                support_grid_relative_path=None if grid_hash is None else str(path.relative_to(args.out)),
                support_grid_sha256=grid_hash)
            new_events = self.events[self.event_cursor:]
            self.event_cursor = len(self.events)
            result = {"type": kind, "request_id": request_id,
                "observation": obs, "candidate_version": self.version,
                "issued_at_s": self.last_snapshot_s, "raw": raw,
                "elapsed_s": max(0., elapsed_s), "terminated": self.terminated(),
                "truncated": self.truncated, "nav_status": self.terminal_status or self.bt_gate.state,
                "events_since": new_events, "counts": dict(self.counts),
                "proposed_action": action, "BT_veto": self.last_veto, "veto_reason": reason,
                "provenance": {"real_Nav2_in_loop": True, "PointLIO_estimated_odometry": True,
                    "real_nav2_in_loop": True,
                    "observed_support_only": True, "MPPI_controls_velocity": True,
                    "GT_used_as_actor_input": False, "profile_map_identity": identity}}
            result = json_value(result)
            with (args.out/"responses.jsonl").open("a") as stream:
                stream.write(json.dumps(result, allow_nan=False)+"\n")
            return result

        def terminated(self):
            return bool(self.mission_success or (self.safety_latched and not self.truncated)
                        or self.terminal_status in {"failed", "canceled", "rejected"})

        def veto(self, reason):
            self.last_veto = True
            self.counts["BT_veto_count"] += 1
            self.record("BT_veto", reason=reason)

        def propose(self, request):
            self.last_veto = self.last_arrival = self.last_recovery = False
            reason = validate_request(request, version=self.version,
                issued_at_s=self.last_snapshot_s, now_s=self.now_s(), mask=self.last_observation["mask"])
            if reason:
                self.veto(reason)
                return reason
            action = request["action"]
            self.sequence += 1
            if action == 13:
                self.last_recovery = True
                self.planning_generation += 1
                self.planning = False
                if self.bt_gate.active is not None:
                    self.apply(self.bt_gate.stop(now_s=self.now_s(), reason="RL_recover_requested"))
                else:
                    self.terminal_status = "canceled"
                    self.latch("RL_recover_without_active_task")
                return None
            if action == 15:
                self.record("RL_hold", active_task=self.bt_gate.active is not None)
                # Invalidate a pending plan as well: hold cannot dispatch a
                # delayed planner result after this policy decision.
                self.planning_generation += 1
                self.planning = False
                if self.bt_gate.active is not None:
                    self.apply(self.bt_gate.stop(now_s=self.now_s(), reason="RL_hold_requested"))
                return None
            if action == 14:
                if self.bt_gate.active is None and not self.planning:
                    self.veto("no_continuable_task")
                    return "no_continuable_task"
                self.record("RL_continue")
                return None
            goal = self.last_goals[action] if action < len(self.last_goals) else None
            if goal is None:
                self.veto("candidate_goal_missing")
                return "candidate_goal_missing"
            if self.planning or self.bt_gate.active is not None:
                active = self.bt_gate.active
                if active is not None and math.dist(goal, active.candidate.goal_xyzyaw[:2]) < 1e-6:
                    self.record("RL_same_goal_kept")
                    return None
                self.veto("previous_navigation_not_terminal")
                return "previous_navigation_not_terminal"
            if not self.health(require_current_support=True):
                reason = self.last_health_reason
                self.veto(reason)
                return reason
            self.planning = True
            self.planning_goal = list(map(float, goal))
            self.planning_generation += 1
            generation = self.planning_generation
            captured_version = self.version
            captured_issued = self.last_snapshot_s
            sequence = self.sequence
            self.counts["planner_requests"] += 1
            self.record("compute_path_requested", action=action, goal_xy=goal,
                candidate_version=captured_version)
            planned = ComputePathToPose.Goal()
            planned.goal = self.pose_message(goal)
            planned.planner_id = "GridBased"
            planned.use_start = False
            future = self.planner.send_goal_async(planned)
            def accepted(response):
                try:
                    handle = response.result()
                    if not handle.accepted:
                        self.planning = False
                        self.veto("planner_goal_rejected")
                        return
                    handle.get_result_async().add_done_callback(
                        lambda result: self.path_ready(result, generation, action, goal,
                            captured_version, captured_issued, sequence))
                except Exception as error:
                    self.planning = False
                    self.veto("planner_exception")
                    self.record("transport_error", detail=repr(error))
            future.add_done_callback(accepted)
            return None

        def path_ready(self, future, generation, action, goal, version, issued, sequence):
            if generation != self.planning_generation or self.safety_latched or self.truncated:
                return
            self.planning = False
            try:
                response = future.result()
                path = response.result.path
                poses = [(pose.header.frame_id or path.header.frame_id,
                    pose.pose.position.x, pose.pose.position.y) for pose in path.poses]
                valid, reason = validate_path(poses, expected_frame="odom", goal_xy=goal,
                    endpoint_tolerance_m=.2)
                self.record("compute_path_result", status=response.status, valid=valid,
                    reason=reason, path_xy=[[p[1], p[2]] for p in poses])
                if response.status != 4 or not valid:
                    self.veto("invalid_or_failed_plan")
                    return
                if not self.health(require_current_support=True):
                    self.veto(self.last_health_reason)
                    return
                check = path_support_check(self.support_grid,
                    [self.pose_xy(), *[[p[1], p[2]] for p in poses]], radius_m=.411)
                self.record("path_support_check", **check,
                    support_sampled_at_s=self.support_grid.sampled_at_s)
                if not check["accepted"]:
                    self.veto("unsupported_path")
                    return
                self.counts["planner_successes"] += 1
                candidate = NavigationCandidate(action, "navigate", "live_engineering_waypoint",
                    (*map(float, goal), 0., 0.), "observed_session_ground", "ground")
                snapshot = CandidateSnapshot(*identity, "odom", version, issued, 2., (candidate,))
                intent = TaskIntent(action, sequence, version, issued)
                self.apply(self.bt_gate.propose(intent, snapshot, now_s=self.now_s(),
                    alive=True, pose_valid=True, frame_ready=True))
            except Exception as error:
                self.veto("path_result_exception")
                self.record("transport_error", detail=repr(error))
                self.latch("path_result_exception")

        def apply(self, decision):
            if decision.commands or decision.status in {"rejected", "terminal", "ignored"}:
                self.record("BT_decision", decision=asdict(decision))
            if decision.status == "rejected":
                self.veto(decision.reason)
            for command in decision.commands:
                if command.kind == "send_goal":
                    self.counts["goal_requests"] += 1
                    self.active_started_s = self.now_s()
                    goal = NavigateToPose.Goal()
                    goal.pose = self.pose_message(command.candidate.goal_xyzyaw[:2])
                    future = self.client.send_goal_async(goal,
                        feedback_callback=lambda message, task=command.task_id: self.feedback(task, message))
                    future.add_done_callback(lambda result, task=command.task_id: self.goal_accepted(task, result))
                elif command.kind == "cancel_goal":
                    handle = self.handles.get(command.task_id)
                    if handle is not None:
                        handle.cancel_goal_async().add_done_callback(
                            lambda result, task=command.task_id: self.cancel_ack(task, result))
                    self.record("cancel_requested", task_id=command.task_id, reason=command.reason)
                elif command.kind == "require_safety_stop":
                    self.latch(command.reason)

        def goal_accepted(self, task, future):
            try:
                handle = future.result()
                self.record("navigation_goal_response", task_id=task, accepted=handle.accepted)
                now = self.now_s()
                self.apply(self.bt_gate.feedback(task_id=task,
                    status="accepted" if handle.accepted else "rejected", sampled_at_s=now, now_s=now))
                if not handle.accepted:
                    self.terminal_status = "rejected"
                    self.latch("navigation_rejected")
                    return
                self.counts["accepted_goals"] += 1
                self.handles[task] = handle
                handle.get_result_async().add_done_callback(lambda result: self.goal_result(task, result))
                if self.bt_gate.state in {"canceling", "uncertain"} or self.safety_latched:
                    handle.cancel_goal_async().add_done_callback(lambda result: self.cancel_ack(task, result))
            except Exception as error:
                self.record("transport_error", detail=repr(error))
                self.latch("goal_acceptance_exception")

        def cancel_ack(self, task, future):
            try:
                result = future.result()
                self.record("cancel_response", task_id=task, return_code=result.return_code,
                    goals_canceling=len(result.goals_canceling))
                if result.goals_canceling:
                    now = self.now_s()
                    self.apply(self.bt_gate.feedback(task_id=task, status="cancel_accepted",
                        sampled_at_s=now, now_s=now))
            except Exception as error:
                self.record("transport_error", detail=repr(error))
                self.latch("cancel_exception")

        def feedback(self, task, message):
            stamp = message.feedback.current_pose.header.stamp
            sampled = stamp.sec+stamp.nanosec/1e9
            if sampled <= 0 or sampled > self.now_s()+1e-6:
                return
            self.apply(self.bt_gate.feedback(task_id=task, status="progress",
                sampled_at_s=sampled, now_s=self.now_s()))

        def goal_result(self, task, future):
            try:
                response = future.result()
                now = self.now_s()
                status = terminal_status(response.status)
                active = self.bt_gate.active
                task_goal_selected = (active is not None and active.task_id == task and
                    math.dist(active.candidate.goal_xyzyaw[:2], args.goal) <= 1e-6)
                decision = self.bt_gate.feedback(task_id=task, status=status, sampled_at_s=now, now_s=now)
                self.apply(decision)
                self.counts["terminal_results"] += 1
                self.record("navigation_terminal_result", task_id=task, status=status)
                if decision.status != "terminal":
                    return
                self.active_started_s = None
                if status == "succeeded":
                    self.last_arrival = True
                    self.mission_success = task_goal_selected and math.dist(self.pose_xy(), args.goal) <= .2
                    self.terminal_status = "succeeded" if self.mission_success else None
                    self.record("engineering_waypoint_arrived", mission_success=self.mission_success,
                        task_goal_distance_m=math.dist(self.pose_xy(), args.goal))
                else:
                    self.terminal_status = status
                    self.latch("navigation_terminal_"+status)
            except Exception as error:
                self.record("transport_error", detail=repr(error))
                self.latch("goal_result_exception")

    node = Session()
    def interrupted(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    rpc_log = (args.out/"requests.jsonl").open("a")
    def emit(value):
        print(json.dumps(json_value(value), allow_nan=False), flush=True)
    def spin_until(predicate, timeout):
        until = time.monotonic()+timeout
        while rclpy.ok() and not predicate() and time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=.025)
        return predicate()
    closing = False
    try:
        while rclpy.ok() and not closing:
            rclpy.spin_once(node, timeout_sec=.01)
            node.wall_watchdog()
            readable, _, _ = select.select([sys.stdin], [], [], .01)
            if not readable:
                continue
            line = sys.stdin.readline()
            if not line:
                node.latch("host_stdin_EOF")
                break
            request = None
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("JSON object required")
                rpc_log.write(json.dumps(request, allow_nan=False)+"\n"); rpc_log.flush()
                node.last_host_request_wall_s = time.monotonic()
                op, request_id = request.get("op"), request.get("request_id")
                if op == "reset":
                    if node.reset_done:
                        raise ValueError("Reset requires a new physical session; mid-process teleport prohibited")
                    def ready():
                        node.lifecycle_poll()
                        return (node.now_s() >= args.warmup_s and node.health()
                            and all(node.lifecycle_active.values()) and node.client.server_is_ready()
                            and node.planner.server_is_ready())
                    if not spin_until(ready, args.startup_wall_timeout_s):
                        node.latch("startup_health_timeout")
                        emit({"type": "error", "request_id": request_id, "error": "startup_health_timeout",
                            "raw": node.raw(), "lifecycle_active": node.lifecycle_active})
                        continue
                    node.reset_done = True
                    node.task_started_s = node.now_s()
                    node.start_distance = math.dist(node.pose_xy(), args.goal)
                    node.last_host_request_wall_s = time.monotonic()
                    emit(node.response("ready", request_id))
                elif op == "step":
                    if not node.reset_done:
                        raise ValueError("reset required before step")
                    if node.terminated() or node.truncated:
                        # Physics and its watchdog continue between RPCs. A
                        # deadline/safety terminal may arrive after the last
                        # observation but before this request; report that real
                        # transition instead of turning it into transport loss.
                        node.record("terminal_before_action_admission", action=request.get("action"))
                        previous = node.last_snapshot_s
                        response = node.response("transition", request_id,
                            elapsed_s=0. if previous is None else node.now_s()-previous,
                            action=request.get("action"), reason="terminal_before_action_admission")
                        response["action_applied"] = False
                        emit(response)
                        continue
                    before = node.now_s()
                    previous_observation_s = node.last_snapshot_s
                    previous_distance = math.dist(node.pose_xy(), args.goal)
                    reason = node.propose(request)
                    reached = spin_until(lambda: node.now_s() >= before+args.decision_dt_s
                        or node.terminated() or node.truncated, args.step_wall_timeout_s)
                    if not reached:
                        node.latch("step_simulation_clock_timeout")
                    node.last_progress_m = previous_distance-math.dist(node.pose_xy(), args.goal)
                    node.record("rpc_step_duration", rpc_execution_s=node.now_s()-before,
                        observation_transition_s=node.now_s()-previous_observation_s)
                    emit(node.response("transition", request_id, elapsed_s=node.now_s()-previous_observation_s,
                        action=request.get("action"), reason=reason))
                elif op == "observe":
                    if not node.reset_done:
                        raise ValueError("reset required before observe")
                    emit(node.response("observation", request_id))
                elif op == "close":
                    node.planning_generation += 1
                    node.planning = False
                    node.apply(node.bt_gate.stop(now_s=node.now_s(), reason="session_close"))
                    node.latch("session_close")
                    spin_until(lambda: node.bt_gate.active is None, 4.)
                    emit({"type": "closed", "request_id": request_id,
                        "counts": dict(node.counts), "active_goal_terminal_confirmed": node.bt_gate.active is None,
                        "safety_latched": node.safety_latched})
                    closing = True
                else:
                    raise ValueError("Unsupported RPC operation")
            except Exception as error:
                node.latch("rpc_exception")
                emit({"type": "error", "request_id": request.get("request_id") if isinstance(request, dict) else None,
                    "error": repr(error), "raw": node.raw()})
    finally:
        node.apply(node.bt_gate.stop(now_s=node.now_s(), reason="session_process_exit"))
        node.latch("session_process_exit")
        spin_until(lambda: node.bt_gate.active is None, 3.)
        (args.out/"result.json").write_text(json.dumps(json_value({"schema": "real_nav2_rl_rpc_v1",
            "counts": node.counts, "mission_success": node.mission_success,
            "terminated": node.terminated(), "truncated": node.truncated,
            "safety_latched": node.safety_latched, "raw": node.raw(),
            "GT_used_as_actor_input": False}), indent=2, allow_nan=False)+"\n")
        rpc_log.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
