#!/usr/bin/env python3
"""Check the copied mesh and run the copied robot builder on a test plane.

This does not validate full-field contacts, sensors, ROS navigation or RL training.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct
import sys
import time

sys.dont_write_bytecode = True

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    snapshot = project / "assets/navigation_reference"
    manifest = json.loads((project / "data/references/navigation_asset_manifest.json").read_text())
    for entry in manifest["files"]:
        copied = project / entry["local"]
        if hashlib.sha256(copied.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Snapshot hash mismatch: {copied}")

    mesh = snapshot / "field/field.stl"
    content = mesh.read_bytes()
    triangle_count = struct.unpack_from("<I", content, 80)[0]
    if len(content) != 84 + 50 * triangle_count:
        raise ValueError("Unexpected binary STL size")
    dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ])
    vertices = np.frombuffer(content, dtype=dtype, offset=84)["vertices"].reshape(-1, 3)
    if not np.isfinite(vertices).all():
        raise ValueError("Non-finite mesh vertices")

    sys.path.insert(0, str(snapshot / "source"))
    from mid360_mujoco.config import load
    from mid360_mujoco.model import build, robot_config

    config = load(snapshot / "configs/simulation.yaml")
    robot = robot_config(snapshot / "configs/robot.yaml")
    # field=None deliberately selects the upstream small plane test scene.
    xml = build(config, robot, field=None)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    started = time.perf_counter()
    for _ in range(1000):
        mujoco.mj_step(model, data)
    elapsed = time.perf_counter() - started
    if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
        raise ValueError("Robot test-plane simulation diverged")
    simulated_time = float(data.time)
    final_base_position = data.qpos[:3].tolist()
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    if not np.array_equal(initial_qpos, data.qpos):
        raise ValueError("Robot qpos reset differs from initial state")

    versions = {}
    for name in ["mujoco", "mujoco-warp", "warp-lang", "torch", "gymnasium", "stable-baselines3"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None

    report = {
        "status": "PASS",
        "scope": "Snapshot hashes, STL structure and finite vertices, copied robot builder, 1000 CPU physics steps on upstream small plane scene, qpos reset",
        "not_validated": ["full-field contact model", "tunnel traversal", "sensor simulation", "ROS/Nav2", "multi-robot", "GPU physics", "RL training"],
        "interpreter": sys.executable,
        "versions": versions,
        "field": {
            "triangle_count": triangle_count,
            "bytes": len(content),
            "bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
            "coordinates": "Prepared simulation world; registration to official competition coordinates is unverified",
        },
        "test_plane_robot": {
            "nq": model.nq, "nv": model.nv, "nu": model.nu,
            "nbody": model.nbody, "ngeom": model.ngeom,
            "physics_steps": 1000,
            "simulated_seconds": simulated_time,
            "wall_seconds": elapsed,
            "final_base_position_m": final_base_position,
            "qpos_reset_equal": True,
            "performance_scope": "Small test plane only; not full-field training throughput",
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
