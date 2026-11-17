#!/usr/bin/env python3
"""Read the copied reference model, without stepping or changing its parameters.

The transformed geom AABBs deliberately overestimate cylindrical/capsule extents.
This is an approximate planning envelope, not hardware clearance certification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/navigation_reference/source"
sys.path.insert(0, str(SOURCE))
from mid360_mujoco.model import build


def calculate(step_deg=2):
    if not 0 < step_deg <= 30 or 360 % step_deg:
        raise ValueError("Yaw sampling step must divide 360 and lie in (0,30]")
    paths = [ROOT / "assets/navigation_reference/configs" / name
             for name in ("robot.yaml", "simulation.yaml")]
    robot, simulation = [yaml.safe_load(p.read_text()) for p in paths]
    model = mujoco.MjModel.from_xml_string(build(simulation, robot))
    data = mujoco.MjData(model)
    # Build's plane-test spawn is not part of body geometry; remove its offset.
    data.qpos[:3] = 0
    mount = model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "lidar_yaw_joint")]
    geoms = np.flatnonzero(model.geom_bodyid != 0)
    nominal_min, nominal_max = np.full(3, np.inf), np.full(3, -np.inf)
    radius = 0.
    for yaw in np.deg2rad(np.arange(0, 360 + step_deg, step_deg)):
        data.qpos[mount] = yaw
        mujoco.mj_forward(model, data)
        rotations = data.geom_xmat[geoms].reshape(-1, 3, 3)
        centers = data.geom_xpos[geoms] + np.einsum(
            "nij,nj->ni", rotations, model.geom_aabb[geoms, :3])
        extents = np.einsum("nij,nj->ni", abs(rotations), model.geom_aabb[geoms, 3:])
        nominal_min = np.minimum(nominal_min, (centers-extents).min(axis=0))
        nominal_max = np.maximum(nominal_max, (centers+extents).max(axis=0))
        # Keep each transformed geom's bounds together; the union AABB's
        # diagonals combine extrema belonging to different mount orientations.
        furthest_xy = abs(centers[:, :2])+extents[:, :2]
        radius = max(radius, float(np.linalg.norm(furthest_xy, axis=1).max()))
    height = float(nominal_max[2]-nominal_min[2])
    return {
        "artifact_type": "approximate_envelope_of_user_selected_navigation_sim_robot_not_measured_hardware",
        "source_robot": str(paths[0].relative_to(ROOT)),
        "source_builder": "assets/navigation_reference/source/mid360_mujoco/model.py",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in paths + [SOURCE / "mid360_mujoco/model.py"]},
        "user_authorized_reference_shape": True,
        "body_geom_count": len(geoms),
        "local_bounds_nominal_zero_suspension_m": [nominal_min.tolist(), nominal_max.tolist()],
        "nominal_span_m": (nominal_max-nominal_min).tolist(),
        "conservative_planar_radius_m": radius,
        "nominal_height_above_contact_plane_m": height,
        "static_full_suspension_extension_added_height_m": robot["chassis"]["suspension_travel_down"],
        "planning_radius_m": float(np.ceil((radius+.01)*1000)/1000),
        "planning_height_m": float(np.ceil((height+robot["chassis"]["suspension_travel_down"])*1000)/1000),
        "extra_planar_margin_m": .02,
        "extra_vertical_margin_m": .01,
        "suspension_and_roll_pitch_swept_envelope_verified": False,
        "mesh_ground_navigation_only": True,
        "all_mount_yaws_sampled_degrees": step_deg,
        "warning": "Geometric AABBs and full mount-yaw sweep; static extension allowance included. Dynamic attitude, actual armor and cables unverified. Source parameters unmodified.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "configs/reference_robot_envelope.json")
    args = parser.parse_args()
    report = calculate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
