#!/usr/bin/env python3
"""Start only private, namespaced Nav2 planner/controller/navigator processes.

Source runtime/nav2/setup.bash first. This launcher does not start a simulator,
publish localization/TF, use simulator truth, install packages or alter the
original navigation project. The test-plane prior is declared in its YAML.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    ("nav2_controller", "controller_server"),
    ("nav2_planner", "planner_server"),
    ("nav2_bt_navigator", "bt_navigator"),
    ("nav2_lifecycle_manager", "lifecycle_manager"),
)


def qualify_parameters(document, namespace):
    """Use exact fully qualified node keys, including nested costmap nodes."""
    qualified = {}

    def visit(value, parts):
        if "ros__parameters" in value:
            if len(value) != 1:
                raise ValueError(f"Mixed parameter/child-node YAML at {parts}")
            qualified["/" + "/".join([namespace, *parts])] = value
            return
        for key, child in value.items():
            if not isinstance(child, dict):
                raise ValueError(f"Expected node mapping at {parts + [key]}")
            visit(child, parts + [key])

    visit(document, [])
    return qualified


def executable(package, name):
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if not prefix:
            continue
        path = Path(prefix) / "lib" / package / name
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    raise FileNotFoundError(
        f"{package}/{name} absent from AMENT_PREFIX_PATH; source runtime/nav2/setup.bash"
    )


def configure_terrain_prior(document, namespace, mode):
    if mode == "clear":
        return document
    if mode != "observed":
        raise ValueError("Unknown terrain prior")
    for name in ("global_costmap", "local_costmap"):
        params = document[name][name]["ros__parameters"]
        params["track_unknown_space"] = True
        params["resolution"] = .15
        params.update(use_maximum=False, unknown_cost_value=255,
                      lethal_cost_threshold=100, trinary_costmap=True)
        # Circumscribed polygon preserves the full .411 m circular envelope;
        # Humble MPPI rejects consider_footprint=true in robot_radius mode.
        radius = .411 / math.cos(math.pi/16)
        params["footprint"] = json.dumps([[radius*math.cos(2*math.pi*i/16),
            radius*math.sin(2*math.pi*i/16)] for i in range(16)])
        params["plugins"] = ["support_layer", "obstacle_layer", "inflation_layer"]
        params["support_layer"] = {
            "plugin": "nav2_costmap_2d::StaticLayer", "enabled": True,
            "map_topic": "/"+namespace+"/terrain_support",
            "map_subscribe_transient_local": True, "subscribe_to_updates": False,
            "footprint_clearing_enabled": False,
        }
        # Filtered obstacles contain no measured free-space rays. Clearing
        # through them must not manufacture support or erase another obstacle.
        params["obstacle_layer"]["terrain"]["clearing"] = False
        params["obstacle_layer"]["combination_method"] = 1
        params["obstacle_layer"]["footprint_clearing_enabled"] = False
    document["controller_server"]["ros__parameters"]["FollowPath"]["CostCritic"]["consider_footprint"] = True
    return document


def prepare(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bt_source = args.bt_xml.read_text()
    root = ET.fromstring(bt_source)
    if len(root.findall(".//ComputePathToPose")) != 1:
        raise ValueError("Probe tree must contain one ComputePathToPose")
    pose_xml = args.output_dir / "navigate_to_pose.xml"
    pose_xml.write_text(bt_source)
    # Humble configures both navigator actions. Give the unused through-poses
    # action a valid minimal tree too; no recovery server is required by either.
    through_tree = ET.fromstring(bt_source)
    compute = through_tree.find(".//ComputePathToPose")
    compute.tag = "ComputePathThroughPoses"
    del compute.attrib["goal"]
    compute.set("goals", "{goals}")
    through_xml = args.output_dir / "navigate_through_poses.xml"
    ET.ElementTree(through_tree).write(through_xml, encoding="unicode", xml_declaration=True)
    params = args.config.read_text()
    for token, value in {
        "__NAMESPACE__": "/" + args.namespace,
        "__TO_POSE_BT_XML__": str(pose_xml.resolve()),
        "__THROUGH_POSES_BT_XML__": str(through_xml.resolve()),
    }.items():
        params = params.replace(token, value)
    document = configure_terrain_prior(yaml.safe_load(params), args.namespace, args.terrain_prior)
    qualified = qualify_parameters(document, args.namespace)
    params_path = args.output_dir / "nav2_params.generated.yaml"
    params_path.write_text(yaml.safe_dump(qualified, sort_keys=False))
    return params_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="rl_bt_nav_probe")
    parser.add_argument("--domain-id", type=int, default=79)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/rl_bt_nav_probe/nav2"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/nav2_probe.yaml")
    parser.add_argument("--bt-xml", type=Path, default=ROOT / "configs/nav2_probe_bt.xml")
    parser.add_argument("--cmd-vel-topic", default="cmd_vel")
    parser.add_argument("--max-wall-seconds", type=float, default=180.0)
    parser.add_argument("--params-only", action="store_true")
    parser.add_argument("--terrain-prior", choices=("clear", "observed"), default="clear")
    parser.add_argument("--print-commands", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.namespace):
        raise SystemExit("Use one nonempty ROS namespace segment")
    if args.domain_id in (61, 65) or not 0 <= args.domain_id <= 232:
        raise SystemExit("Use an isolated ROS_DOMAIN_ID (61/65 are reserved for original projects)")
    if args.max_wall_seconds <= 0:
        raise SystemExit("max-wall-seconds must be positive")
    args.output_dir = args.output_dir.resolve()
    params_path = prepare(args)
    if args.params_only:
        print(json.dumps({"params": str(params_path), "nodes_started": False}))
        return
    commands = []
    for package, name in PACKAGES:
        node_name = "lifecycle_manager_navigation" if package == "nav2_lifecycle_manager" else name
        command = [
            executable(package, name), "--ros-args", "--params-file", str(params_path),
            "-r", "__ns:=/" + args.namespace, "-r", "__node:=" + node_name,
            "-r", "/tf:=/" + args.namespace + "/tf",
            "-r", "/tf_static:=/" + args.namespace + "/tf_static",
        ]
        if package == "nav2_controller":
            command += ["-r", "cmd_vel:=" + args.cmd_vel_topic]
        commands.append(command)
    if args.print_commands:
        print(json.dumps(commands, ensure_ascii=False, indent=2))
        return
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(args.domain_id)
    env["ROS_LOCALHOST_ONLY"] = "1"
    env["ROS_LOG_DIR"] = str(args.output_dir / "ros_logs")
    processes = []
    logs = []
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.monotonic()
    reason = "wall_timeout"
    try:
        for command in commands:
            log = (args.output_dir / (command[0].rsplit("/", 1)[-1] + ".log")).open("w")
            logs.append(log)
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                              env=env, start_new_session=True))
        print(json.dumps({"params": str(params_path), "pids": [p.pid for p in processes],
                          "domain_id": args.domain_id, "namespace": args.namespace}), flush=True)
        while not stopping and time.monotonic() - started < args.max_wall_seconds:
            failed = [p for p in processes if p.poll() is not None]
            if failed:
                reason = "owned_node_exited"
                break
            time.sleep(0.1)
        if stopping:
            reason = "signal"
    finally:
        # Only process groups started above belong to this launcher.
        for p in reversed(processes):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGINT)
        deadline = time.monotonic() + 5.0
        for p in processes:
            try:
                p.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        for log in logs:
            log.close()
        (args.output_dir / "launcher_report.json").write_text(json.dumps({
            "schema": "nav2_probe_launcher_v1", "reason": reason,
            "wall_seconds": time.monotonic() - started,
            "domain_id": args.domain_id, "namespace": args.namespace,
            "commands": commands, "returncodes": [p.returncode for p in processes],
            "test_plane_free_space_prior": args.terrain_prior == "clear",
            "terrain_prior": args.terrain_prior,
            "localization_source": "Point-LIO supplied externally; no truth localization in launcher",
            "official_field_navigation_validated": False,
        }, ensure_ascii=False, indent=2) + "\n")
    if reason == "owned_node_exited":
        raise SystemExit("A Nav2 process exited; inspect per-node logs")


if __name__ == "__main__":
    main()
