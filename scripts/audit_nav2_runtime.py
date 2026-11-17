#!/usr/bin/env python3
"""Record the existing ROS runtime without installing or starting any node."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run(argv: list[str]) -> dict:
    result = subprocess.run(argv, text=True, capture_output=True, timeout=60)
    return {
        "argv": argv,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def source(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path), "present": False}
    return {
        "path": str(path),
        "present": True,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="Combat_Sentry2026")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    usage = shutil.disk_usage(project)
    ros = "source /opt/ros/humble/setup.bash; source /opt/pb2025_nav_ws/install/setup.bash; "
    commands = {
        "container_state": [
            "docker", "inspect", args.container,
            "--format", "{{json .State}}",
        ],
        "ros_packages": [
            "docker", "exec", args.container, "bash", "-lc", ros + "ros2 pkg list",
        ],
        "container_processes": [
            "docker", "exec", args.container, "ps", "-eo", "pid,args",
        ],
        "apt_candidates": [
            "docker", "exec", args.container, "apt-cache", "policy",
            "ros-humble-nav2-controller", "ros-humble-nav2-mppi-controller",
            "ros-humble-nav2-msgs", "ros-humble-nav2-bringup",
        ],
        "python_ros": [
            "docker", "exec", args.container, "bash", "-lc",
            ros + 'python3 -c "import sys,rclpy; print(sys.version); print(rclpy.__file__)"',
        ],
    }
    checks = {name: run(argv) for name, argv in commands.items()}
    packages = checks["ros_packages"]["stdout"].splitlines()
    nav2_packages = sorted(p for p in packages if p.startswith("nav2_"))
    report = {
        "schema": "nav2_runtime_audit_v1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "nodes_started": False,
        "packages_installed": False,
        "host_ros_present": Path("/opt/ros").exists(),
        "project_disk_available_bytes": usage.free,
        "container": args.container,
        "nav2_packages_visible": nav2_packages,
        "nav2_runtime_ready": "nav2_mppi_controller" in packages and "nav2_msgs" in packages,
        "commands": checks,
        "source_files": [source(p) for p in [
            Path("/home/kellen/mid360_mujoco/ros2/nav.launch.py"),
            Path("/home/kellen/mid360_mujoco/src/mid360_mujoco/ros_bridge.py"),
            Path("/home/kellen/pb2025_sim_nuc/configs/nav2_params.yaml"),
            Path("/home/kellen/pb2025_sim_nuc/src/pb2025_lio_backends/src/lio_display.cpp"),
        ]],
        "source_review": {
            "simulator_ros_domain_id": 65,
            "simulator_namespace": "red_standard_robot1",
            "simulator_velocity_topic": "/red_standard_robot1/cmd_vel",
            "simulator_clock_topic": "/clock",
            "estimated_odometry_topic": "/red_standard_robot1/odometry",
            "estimated_odometry_frame": "odom",
            "estimated_odometry_child_frame": "base_footprint",
            "registered_cloud_topic": "/red_standard_robot1/registered_scan",
            "existing_launch_scope": "LIO and display only",
            "existing_configured_controller": "pb_omni_pid_pursuit_controller::OmniPidPursuitController",
            "existing_lio_odometry_twist": "not assigned; default zero",
            "map_to_odom_registration": "not provided by this LIO launch",
            "notes": [
                "Source review describes the listed hashes; it is not an active ROS topic measurement.",
                "GT odometry is for evaluation and must not be wired as actor or production navigation feedback.",
                "Derive body velocity from estimated poses or a legal wheel-odometry source before MPPI execution.",
                "Formal map coordinates need explicit map-to-odom registration; initial LIO origin is not the CAD map origin.",
            ],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out.resolve()),
        "nav2_packages_visible": nav2_packages,
        "nav2_runtime_ready": report["nav2_runtime_ready"],
        "project_disk_available_bytes": usage.free,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
