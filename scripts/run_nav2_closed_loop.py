#!/usr/bin/env python3
"""Run the project's isolated Point-LIO/Nav2/MPPI physical execution check.

Host mode stages only project files into a private directory in the existing
ROS container. Container mode starts and stops only its own process groups.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("nav2_mujoco_fixture.py", "nav2_action_probe.py", "launch_nav2_probe.py", "run_nav2_closed_loop.py")
MODULES = ("__init__.py", "contracts.py", "navigation_contract.py", "nav2_transport.py", "estimated_twist.py", "voxel_terrain.py", "uc_field_reference.py", "terrain_worker.py", "support_costmap.py", "scan_registration.py")


def check_domain_empty():
    import rclpy
    rclpy.init()
    node = rclpy.create_node("rl_bt_probe_domain_inventory")
    deadline = time.monotonic()+1.2
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=.05)
    found = [(name, ns) for name, ns in node.get_node_names_and_namespaces()
             if name != node.get_name()]
    node.destroy_node(); rclpy.shutdown()
    if found:
        raise RuntimeError(f"Domain79 already contains nodes; refusing to start: {found}")


def container_run(args):
    if os.environ.get("ROS_DOMAIN_ID") != "79":
        raise RuntimeError("Private fixture requires ROS_DOMAIN_ID=79")
    check_domain_empty()
    run = args.out.resolve(); run.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for directory, files in (("scripts", SCRIPTS), ("rl_bt", MODULES),
                             ("configs", ("nav2_probe.yaml", "nav2_probe_bt.xml"))):
        for name in files:
            relative = Path(directory)/name
            data = (ROOT/relative).read_bytes()
            destination = run/"source"/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            hashes[str(relative)] = hashlib.sha256(data).hexdigest()
    (run/"source_sha256.json").write_text(json.dumps(hashes, indent=2)+"\n")
    env = os.environ.copy()
    env.update(ROS_DOMAIN_ID="79", ROS_LOCALHOST_ONLY="1", MID360_RAY_BACKEND="cpu",
               OMP_NUM_THREADS="2", ROS_LOG_DIR=str(run / "roslogs"))
    processes, logs = [], []
    def start(name, command):
        stream = (run / (name+".log")).open("w"); logs.append(stream)
        process = subprocess.Popen(command, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        return process
    code = None
    try:
        maximum = args.wall_timeout_s+20.
        start("nav2", ["/usr/bin/python3", str(ROOT / "scripts/launch_nav2_probe.py"),
            "--output-dir", str(run / "nav2"), "--max-wall-seconds", str(maximum),
            "--terrain-prior", args.terrain_prior])
        fixture_command = ["/opt/mid360_mujoco/.venv/bin/python", str(ROOT / "scripts/nav2_mujoco_fixture.py"),
            "--out", str(run / "fixture"), "--wall-seconds", str(maximum),
            "--scene", args.scene, "--field-dir", args.field_dir,
            "--terrain", "voxel" if args.scene == "rmuc" else "plane_height",
            "--terrain-prior", args.terrain_prior, "--terrain-sensors", args.terrain_sensors]
        if args.start_footprint_prior: fixture_command += ["--start-footprint-prior"]
        if args.support_dropout_after_s is not None:
            fixture_command += ["--support-dropout-after-s", str(args.support_dropout_after_s)]
        start("fixture", fixture_command)
        command = ["/usr/bin/python3", str(ROOT / "scripts/nav2_action_probe.py"),
            "--out", str(run / "action"), "--wall-timeout-s", str(args.wall_timeout_s),
            "--goal", *map(str, args.goal), "--scene", args.scene, "--field-dir", args.field_dir,
            "--terrain-prior", args.terrain_prior, "--warmup-s", str(args.warmup_s)]
        if args.cancel_after_s is not None:
            command += ["--cancel-after-s", str(args.cancel_after_s)]
        command += ["--ros-args", "-r", "/tf:=tf", "-r", "/tf_static:=tf_static"]
        action = start("action", command)
        code = action.wait(timeout=args.wall_timeout_s+8.)
        if code == 0:
            # Keep physics and estimator running after the action's terminal
            # result. A canceled action is distinct from measured standstill.
            time.sleep(1.)
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try: process.wait(timeout=8.)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        for stream in logs: stream.close()
        (run / "processes.json").write_text(json.dumps({"owned_pids": [p.pid for p in processes],
            "returncodes": [p.returncode for p in processes], "action_returncode": code}, indent=2)+"\n")
    action = json.loads((run / "action/result.json").read_text())
    fixture = json.loads((run / "fixture/result.json").read_text())
    controller = (run / "nav2/controller_server.log").read_text()
    expected = args.expected_status or ("canceled" if args.cancel_after_s is not None else "succeeded")
    expected_code = 0 if expected in ("succeeded", "canceled") else 2
    estimate = fixture["estimated_poses"][-1] if fixture["estimated_poses"] else None
    truth = fixture["GT_trajectory_diagnostic_only"][-1] if fixture["GT_trajectory_diagnostic_only"] else None
    errors = None if estimate is None else math.hypot(estimate[1]-args.goal[0], estimate[2]-args.goal[1])
    result = {"artifact_type": "actual_PointLIO_Nav2_MPPI_MuJoCo_fixture_result",
        "expected_action_status": expected, "actual_action_status": action["result"],
        "success_for_reported_scope": code == expected_code and action["result"] == expected,
        "navigation_succeeded":action["result"] == "succeeded",
        "domain_id": 79, "goal_xy": args.goal, "scene": args.scene,
        "field_source": fixture.get("field_source"),
        "terrain_classifier": fixture["terrain_classifier"],
        "terrain_execution": fixture.get("terrain_execution", "sync"),
        "terrain_worker":fixture.get("terrain_worker"),
        "terrain_prior":args.terrain_prior, "observed_support_costmap":args.terrain_prior == "observed",
        "terrain_sensors":args.terrain_sensors,
        "shared_pointlio_effective_unchanged":fixture["shared_pointlio_effective_sha256_before"] == fixture["shared_pointlio_effective_sha256_after"],
        "fixed_start_footprint_prior":args.start_footprint_prior,
        "injected_support_dropout_after_simulation_s":args.support_dropout_after_s,
        "controller_plugin_loaded": "Created controller : FollowPath of type nav2_mppi_controller::MPPIController" in controller,
        "motion_model": "Omni", "estimated_odometry_twist_samples": len(fixture["estimated_twists"]),
        "registered_obstacle_clouds": len(fixture["obstacle_cloud_counts"]),
        "MPPI_cmd_vel_samples": len(fixture["commands"]),
        "max_command_xy_speed_m_per_s": max((math.hypot(x[1],x[2]) for x in fixture["commands"]), default=0.),
        "final_estimated_pose": estimate, "final_GT_diagnostic": truth,
        "post_terminal_observation_wall_s": 1. if code == 0 else 0.,
        "final_estimated_twist": fixture["estimated_twists"][-1] if fixture["estimated_twists"] else None,
        "estimated_goal_xy_error_m": errors,
        "GT_used_as_navigation_odometry_or_TF": False,
        "safety_stop_latched": fixture["safety_stop_latched"],
        "original_simulation_and_shared_system_modified": False,
        "RL_training_on_this_navigation_validated": False,
        "limitations": ["PointLIO session odom, not official referee coordinate registration.",
            ("Unknown support is lethal in StaticLayer; inferred support and frame accuracy still require validation."
             if args.terrain_prior == "observed" else "Rolling costmap clear-space prior is execution-test scope, not full-field unknown-safe planning."),
            "Voxel terrain classification is a local observed-point implementation, not a full HERO reproduction.",
            "TaskIntent is a test candidate, not a learned official-map strategy validation.",
            "This is a bounded execution integration test, not navigation reliability or NUC performance acceptance."]}
    (run / "verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["success_for_reported_scope"] else 2


def host_run(args):
    out = args.out.resolve()
    if out.exists(): raise FileExistsError(out)
    token = uuid.uuid4().hex[:12]
    project = "/tmp/rl_bt_nav_probe/project_"+token
    remote_run = "/tmp/rl_bt_nav_probe/run_"+token
    subprocess.run(["docker", "exec", args.container, "mkdir", "-p", project+"/scripts",
                    project+"/rl_bt", project+"/configs"], check=True)
    for directory, files in (("scripts", SCRIPTS), ("rl_bt", MODULES),
                             ("configs", ("nav2_probe.yaml", "nav2_probe_bt.xml"))):
        for name in files:
            subprocess.run(["docker", "cp", str(ROOT / directory / name),
                            args.container+":"+project+"/"+directory+"/"], check=True, capture_output=True)
    command = ["python3", project+"/scripts/run_nav2_closed_loop.py", "--inside-container",
        "--out", remote_run, "--wall-timeout-s", str(args.wall_timeout_s), "--goal", *map(str,args.goal),
        "--scene", args.scene, "--field-dir", args.field_dir,
        "--terrain-prior", args.terrain_prior, "--warmup-s", str(args.warmup_s),
        "--terrain-sensors", args.terrain_sensors]
    if args.start_footprint_prior: command += ["--start-footprint-prior"]
    if args.expected_status: command += ["--expected-status", args.expected_status]
    if args.support_dropout_after_s is not None:
        command += ["--support-dropout-after-s",str(args.support_dropout_after_s)]
    if args.cancel_after_s is not None: command += ["--cancel-after-s", str(args.cancel_after_s)]
    setup = "source /tmp/rl_bt_nav2_overlay/setup.bash; source /opt/mid360_mujoco/lio_ws/install/setup.bash; exec "+shlex.join(command)
    run = subprocess.run(["docker", "exec", "-e", "ROS_DOMAIN_ID=79", "-e", "ROS_LOCALHOST_ONLY=1",
                          args.container, "bash", "-lc", setup])
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["docker", "cp", args.container+":"+remote_run, str(out)], check=True)
    return run.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--goal", type=float, nargs=2, default=(1.,0.))
    parser.add_argument("--wall-timeout-s", type=float, default=65.)
    parser.add_argument("--cancel-after-s", type=float)
    parser.add_argument("--scene", choices=("plane", "rmuc"), default="plane")
    parser.add_argument("--field-dir", default="/tmp/rl_bt_nav_probe/fields/rmuc2027_original")
    parser.add_argument("--container", default="Combat_Sentry2026")
    parser.add_argument("--inside-container", action="store_true")
    parser.add_argument("--terrain-prior", choices=("clear", "observed"))
    parser.add_argument("--start-footprint-prior", action="store_true")
    parser.add_argument("--terrain-sensors", choices=("primary", "dual"))
    parser.add_argument("--support-dropout-after-s", type=float)
    parser.add_argument("--warmup-s", type=float, default=4.)
    parser.add_argument("--expected-status", choices=("succeeded", "canceled", "invalid_or_failed_plan", "unsupported_path"))
    args = parser.parse_args()
    if args.terrain_prior is None:
        args.terrain_prior = "observed" if args.scene == "rmuc" else "clear"
    if args.terrain_sensors is None:
        args.terrain_sensors = "dual" if args.scene == "rmuc" else "primary"
    if args.scene == "plane" and args.terrain_prior == "observed":
        parser.error("Observed support requires the voxel UC fixture")
    if not math.isfinite(args.warmup_s) or args.warmup_s < 0:
        parser.error("Finite nonnegative warmup required")
    if args.support_dropout_after_s is not None and (not math.isfinite(args.support_dropout_after_s)
            or args.support_dropout_after_s < 0 or args.terrain_prior != "observed"):
        parser.error("Support dropout requires observed mode and finite nonnegative simulation time")
    if not all(math.isfinite(x) for x in args.goal) or not 0 < args.wall_timeout_s <= 180:
        parser.error("Finite goal and 0<timeout<=180 required")
    raise SystemExit(container_run(args) if args.inside_container else host_run(args))


if __name__ == "__main__":
    main()
