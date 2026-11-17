#!/usr/bin/env python3
"""Stage and own one isolated real-navigation JSONL RL session.

stdin/stdout carry JSONL; diagnostics and all simulation/ROS logs are artifacts.
Only process groups started by this runner are stopped.  No system packages,
original robot assets, shared estimator settings or existing ROS domains change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("nav2_mujoco_fixture.py", "launch_nav2_probe.py", "nav2_rl_session.py",
           "run_nav2_rl_session.py", "run_nav2_closed_loop.py")
CONFIGS = ("nav2_probe.yaml", "nav2_probe_bt.xml")


def source_files():
    return ([Path("scripts")/name for name in SCRIPTS]
        + [Path("rl_bt")/path.name for path in sorted((ROOT/"rl_bt").glob("*.py"))]
        + [Path("configs")/name for name in CONFIGS])


def copied_arguments(args):
    command = ["--wall-timeout-s", str(args.wall_timeout_s), "--scene", args.scene,
        "--field-dir", args.field_dir, "--goal", *map(str, args.goal),
        "--warmup-s", str(args.warmup_s), "--task-timeout-s", str(args.task_timeout_s),
        "--decision-dt-s", str(args.decision_dt_s),
        "--startup-wall-timeout-s", str(args.startup_wall_timeout_s),
        "--step-wall-timeout-s", str(args.step_wall_timeout_s),
        "--host-idle-timeout-s", str(args.host_idle_timeout_s)]
    if args.start_footprint_prior:
        command.append("--start-footprint-prior")
    if args.capture_terrain_input:
        command.append("--capture-terrain-input")
    if args.simulation_seed is not None:
        command += ["--simulation-seed", str(args.simulation_seed)]
    if args.support_dropout_after_s is not None:
        command += ["--support-dropout-after-s", str(args.support_dropout_after_s)]
    return command


def container_run(args):
    if os.environ.get("ROS_DOMAIN_ID") != "79":
        raise RuntimeError("Live RL fixture requires isolated ROS_DOMAIN_ID=79")
    # Importing this helper does not launch ROS or alter the existing fixture.
    sys.path.insert(0, str(ROOT/"scripts"))
    from run_nav2_closed_loop import check_domain_empty
    check_domain_empty()
    run = args.out.resolve()
    run.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for relative in source_files():
        data = (ROOT/relative).read_bytes()
        destination = run/"source"/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        hashes[str(relative)] = hashlib.sha256(data).hexdigest()
    (run/"source_sha256.json").write_text(json.dumps(hashes, indent=2)+"\n")
    (run/"runner_pid.json").write_text(json.dumps({"pid": os.getpid(), "started_wall_s": time.time()})+"\n")
    env = os.environ.copy()
    env.update(ROS_DOMAIN_ID="79", ROS_LOCALHOST_ONLY="1", MID360_RAY_BACKEND="cpu",
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", ROS_LOG_DIR=str(run/"roslogs"),
        RCUTILS_LOGGING_USE_STDOUT="0")
    processes, streams = [], []
    started = time.monotonic()
    def launch(name, command, *, rpc=False):
        stream = (run/(name+".log")).open("w")
        streams.append(stream)
        process = subprocess.Popen(command, env=env,
            stdin=sys.stdin if rpc else subprocess.DEVNULL,
            stdout=sys.stdout if rpc else stream, stderr=stream, start_new_session=True)
        processes.append((name, process))
        return process
    code = None
    def interrupted(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        maximum = args.wall_timeout_s+10.
        launch("nav2", ["/usr/bin/python3", str(ROOT/"scripts/launch_nav2_probe.py"),
            "--output-dir", str(run/"nav2"), "--max-wall-seconds", str(maximum),
            "--terrain-prior", "observed"])
        fixture_command = ["/opt/mid360_mujoco/.venv/bin/python",
            str(ROOT/"scripts/nav2_mujoco_fixture.py"), "--out", str(run/"fixture"),
            "--wall-seconds", str(maximum), "--scene", "rmuc", "--field-dir", args.field_dir,
            "--terrain", "voxel", "--terrain-prior", "observed", "--terrain-sensors", "dual"]
        if args.start_footprint_prior:
            fixture_command.append("--start-footprint-prior")
        if args.capture_terrain_input:
            fixture_command.append("--capture-terrain-input")
        if args.simulation_seed is not None:
            fixture_command += ["--simulation-seed", str(args.simulation_seed)]
        if args.support_dropout_after_s is not None:
            fixture_command += ["--support-dropout-after-s", str(args.support_dropout_after_s)]
        launch("fixture", fixture_command)
        session_command = ["/usr/bin/python3", str(ROOT/"scripts/nav2_rl_session.py"),
            "--out", str(run/"session"), "--scene", "rmuc", "--field-dir", args.field_dir,
            "--goal", *map(str, args.goal), "--warmup-s", str(args.warmup_s),
            "--task-timeout-s", str(args.task_timeout_s), "--decision-dt-s", str(args.decision_dt_s),
            "--startup-wall-timeout-s", str(args.startup_wall_timeout_s),
            "--step-wall-timeout-s", str(args.step_wall_timeout_s),
            "--host-idle-timeout-s", str(args.host_idle_timeout_s),
            "--ros-args", "-r", "/tf:=tf", "-r", "/tf_static:=tf_static"]
        session = launch("session", session_command, rpc=True)
        try:
            code = session.wait(timeout=args.wall_timeout_s)
        except subprocess.TimeoutExpired:
            code = 124
    finally:
        # Stop the RPC first so its finalizer sends cancellation and a latched
        # safety stop while the estimator and physics still exist.
        for name, process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=8.)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for stream in streams:
            stream.close()
        report = {"schema": "real_nav2_rl_runner_v1", "session_returncode": code,
            "wall_seconds": time.monotonic()-started,
            "processes": [{"name": name, "pid": process.pid, "returncode": process.returncode}
                for name, process in processes], "domain_id": 79, "scene": "rmuc",
            "goal_xy_m": list(args.goal), "fixed_start_footprint_prior": args.start_footprint_prior,
            "PointLIO_estimated_odometry": True, "GT_used_as_actor_input": False,
            "sensor_randomization_seed": args.simulation_seed,
            "unknown_support_is_lethal": True, "original_project_modified": False,
            "fixture_result_present": (run/"fixture/result.json").exists(),
            "source_sha256": hashes}
        fixture = run/"fixture/result.json"
        if fixture.exists():
            data = json.loads(fixture.read_text())
            report.update(MPPI_cmd_vel_samples=len(data.get("commands", [])),
                estimated_pose_samples=len(data.get("estimated_poses", [])),
                final_estimated_pose=next(iter(data.get("estimated_poses", [])[-1:]), None),
                safety_stop_latched=data.get("safety_stop_latched"),
                field_source=data.get("field_source"),
                shared_pointlio_effective_unchanged=data.get("shared_pointlio_effective_sha256_before")
                    == data.get("shared_pointlio_effective_sha256_after"))
        controller = run/"nav2/controller_server.log"
        report["MPPI_plugin_loaded"] = controller.exists() and (
            "Created controller : FollowPath of type nav2_mppi_controller::MPPIController" in controller.read_text())
        (run/"verification.json").write_text(json.dumps(report, indent=2)+"\n")
    return code if code is not None else 1


def host_run(args):
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    token = uuid.uuid4().hex[:12]
    project = "/tmp/rl_bt_nav_probe/live_project_"+token
    remote_run = "/tmp/rl_bt_nav_probe/live_run_"+token
    subprocess.run(["docker", "exec", args.container, "mkdir", "-p",
        project+"/scripts", project+"/rl_bt", project+"/configs"], check=True,
        stdout=subprocess.DEVNULL)
    for relative in source_files():
        subprocess.run(["docker", "cp", str(ROOT/relative),
            args.container+":"+project+"/"+str(relative)], check=True, capture_output=True)
    command = ["/usr/bin/python3", project+"/scripts/run_nav2_rl_session.py",
        "--inside-container", "--out", remote_run, *copied_arguments(args)]
    setup = ("source /tmp/rl_bt_nav2_overlay/setup.bash; "
        "source /opt/mid360_mujoco/lio_ws/install/setup.bash; exec "+shlex.join(command))
    process = None
    interrupted_signal = None
    def interrupted(signum, frame):
        nonlocal interrupted_signal
        interrupted_signal = signum
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        process = subprocess.Popen(["docker", "exec", "-i", "-e", "ROS_DOMAIN_ID=79",
            "-e", "ROS_LOCALHOST_ONLY=1", args.container, "bash", "-lc", setup],
            stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
        return process.wait()
    finally:
        # The private pid file identifies only this runner. Sending TERM lets it
        # stop its owned groups; killing the docker client alone would not.
        if process is not None and process.poll() is None:
            stop_code = ("import json,os,signal; from pathlib import Path; "
                f"p=Path({(remote_run+'/runner_pid.json')!r}); "
                "os.kill(json.loads(p.read_text())['pid'], signal.SIGTERM) if p.exists() else None")
            subprocess.run(["docker", "exec", args.container, "/usr/bin/python3", "-c", stop_code],
                capture_output=True, timeout=15.)
            try:
                process.wait(timeout=25.)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5.)
        out.parent.mkdir(parents=True, exist_ok=True)
        copied = subprocess.run(["docker", "cp", args.container+":"+remote_run, str(out)],
            capture_output=True, timeout=40.)
        if copied.returncode:
            # Preserve a diagnostic even when startup failed before the run dir.
            out.mkdir(parents=True, exist_ok=True)
            (out/"copy_error.txt").write_bytes(copied.stderr)
        (out/"host_runner.json").write_text(json.dumps({"container": args.container,
            "private_project": project, "private_remote_run": remote_run,
            "docker_returncode": None if process is None else process.returncode,
            "interrupted_signal": interrupted_signal}, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wall-timeout-s", type=float, default=180.)
    parser.add_argument("--scene", choices=("rmuc",), default="rmuc")
    parser.add_argument("--goal", nargs=2, type=float, default=(0., .4))
    parser.add_argument("--field-dir", default="/tmp/rl_bt_nav_probe/fields/rmuc2027_original")
    parser.add_argument("--container", default="Combat_Sentry2026")
    parser.add_argument("--inside-container", action="store_true")
    parser.add_argument("--start-footprint-prior", action="store_true")
    parser.add_argument("--support-dropout-after-s", type=float)
    parser.add_argument("--capture-terrain-input", action="store_true")
    parser.add_argument("--simulation-seed", type=int)
    parser.add_argument("--warmup-s", type=float, default=5.)
    parser.add_argument("--task-timeout-s", type=float, default=20.)
    parser.add_argument("--decision-dt-s", type=float, default=.5)
    parser.add_argument("--startup-wall-timeout-s", type=float, default=90.)
    parser.add_argument("--step-wall-timeout-s", type=float, default=15.)
    parser.add_argument("--host-idle-timeout-s", type=float, default=5.)
    args = parser.parse_args()
    if not all(__import__("math").isfinite(value) for value in (*args.goal, args.wall_timeout_s,
            args.warmup_s, args.task_timeout_s, args.decision_dt_s)):
        parser.error("Finite goals and times required")
    if min(args.wall_timeout_s, args.task_timeout_s, args.decision_dt_s,
            args.startup_wall_timeout_s, args.step_wall_timeout_s, args.host_idle_timeout_s) <= 0:
        parser.error("Positive timeouts required")
    if args.warmup_s < 0 or args.wall_timeout_s > 600:
        parser.error("Nonnegative warmup and wall timeout <=600 required")
    if args.simulation_seed is not None and args.simulation_seed < 0:
        parser.error("Simulation seed must be nonnegative")
    raise SystemExit(container_run(args) if args.inside_container else host_run(args))


if __name__ == "__main__":
    main()
