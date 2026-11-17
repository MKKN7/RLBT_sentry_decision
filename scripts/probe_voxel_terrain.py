#!/usr/bin/env python3
"""Reproduce observed-voxel terrain checks; optional STL sample is diagnostic only."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_bt.layered_geometry import read_binary_stl
from rl_bt.voxel_terrain import VoxelTerrainConfig, VoxelTerrainMap


def plane(angle=0., offset=0.):
    axis = np.arange(-1.475, 1.5, .05)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack((x.ravel(), y.ravel(),
        (x * np.tan(np.deg2rad(angle)) + offset).ravel()))


def evaluate(points, pose, config):
    mapper = VoxelTerrainMap(config)
    started = time.perf_counter()
    accepted = mapper.update(points, sampled_at_s=1., now_s=1.,
        robot_position_xyz=pose, frame_id="odom")
    update_s = time.perf_counter() - started
    started = time.perf_counter()
    result = mapper.snapshot(robot_position_xyz=pose, now_s=1.)
    snapshot_s = time.perf_counter() - started
    return result, {"update_result": accepted, "input_points": len(points),
        "update_wall_s": update_s, "snapshot_wall_s": snapshot_s, **result.diagnostics}


def mesh_sample(mesh, pose, radius, rng, sample_count=80_000):
    """Area sampling has NO occlusion model; never feed this to online nav."""
    triangles = read_binary_stl(mesh)
    lo, hi = triangles.min(axis=1), triangles.max(axis=1)
    candidate = (np.all(hi[:, :2] >= pose[:2] - radius, axis=1)
                 & np.all(lo[:, :2] <= pose[:2] + radius, axis=1))
    triangles = triangles[candidate]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                  triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    keep = area > 1e-12
    triangles, area = triangles[keep], area[keep]
    selected = triangles[rng.choice(len(triangles), sample_count, p=area / area.sum())]
    u = np.sqrt(rng.random(sample_count)); v = rng.random(sample_count)
    return (selected[:, 0] * (1 - u[:, None])
        + selected[:, 1] * (u * (1 - v))[:, None]
        + selected[:, 2] * (u * v)[:, None])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--mesh-pose", type=float, nargs=3, default=[3.4, 9.5, .27])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = VoxelTerrainConfig(resolution_m=.1, radius_m=2.,
        self_exclusion_radius_m=0., seed_height_tolerance_m=.025)
    scenes = [("flat_floor", plane()), ("20_degree_inclusive", plane(20.)),
        ("25_degree_blocked", plane(25.)),
        ("230mm_roof_blocked", np.vstack((plane(), plane(offset=.23)))),
        ("400mm_roof_separate", np.vstack((plane(), plane(offset=.4))))]
    results, reports = [], {}
    for name, points in scenes:
        result, report = evaluate(points, np.zeros(3), config)
        results.append((name, result)); reports[name] = report
    acceptance = {
        "flat_support_observed": reports["flat_floor"]["free_cells"] > 0,
        "20_degree_supported": reports["20_degree_inclusive"]["free_cells"] > 0,
        "25_degree_not_free": reports["25_degree_blocked"]["free_cells"] == 0,
        "low_roof_blocks": reports["230mm_roof_blocked"]["free_cells"] == 0,
        "high_roof_does_not_flatten_into_ground": reports["400mm_roof_separate"]["free_cells"] > 0,
        "unknown_remains_unknown": all(x["unknown_cells"] > 0 for x in reports.values())}
    mesh_report = None
    if args.mesh:
        pose = np.asarray(args.mesh_pose)
        mesh_config = VoxelTerrainConfig(radius_m=3., self_exclusion_radius_m=.45)
        samples = mesh_sample(args.mesh, pose, mesh_config.radius_m, np.random.default_rng(73103))
        result, report = evaluate(samples, pose, mesh_config)
        reports["rmuc_STL_synthetic_diagnostic"] = report
        results.append(("STL diagnostic: NO sensor occlusion", result))
        mesh_report = {"path": str(args.mesh.resolve()),
            "sha256": hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
            "pose_xyz_m": pose.tolist(), "area_sample_seed": 73103,
            "has_sensor_occlusion_model": False,
            "online_navigation_input": False, "actual_PointLIO_run": False}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for axis, (name, result) in zip(axes.ravel(), results):
        display = np.where(result.occupancy_xy == -1, .5,
                          np.where(result.occupancy_xy == 100, 1., 0.))
        origin = result.origin_xy_m
        shape = np.asarray(result.occupancy_xy.shape)
        axis.imshow(display.T, origin="lower", interpolation="nearest", cmap="gray_r",
            vmin=0., vmax=1., extent=[origin[0], origin[0] + shape[0] * result.resolution_m,
                                    origin[1], origin[1] + shape[1] * result.resolution_m])
        axis.set(title=name, xlabel="x (m)", ylabel="y (m)")
    for axis in axes.ravel()[len(results):]:
        axis.axis("off")
    fig.suptitle("Observed support: white free, black occupied, gray unknown; no footprint inflation")
    fig.savefig(args.out / "terrain_checks.png", dpi=130)
    plt.close(fig)
    summary = {"artifact_type": "voxel_terrain_unit_and_geometry_diagnostic",
        "schema": "observed_voxel_terrain_v1", "source_hashes": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [ROOT / "rl_bt/voxel_terrain.py", Path(__file__).resolve()]},
        "synthetic_config": asdict(config), "acceptance": acceptance,
        "scenes": reports, "mesh_diagnostic": mesh_report,
        "all_synthetic_acceptance_pass": all(acceptance.values()),
        "not_verified": ["NUC runtime", "hardware envelope", "sensor-space ray free volume",
            "full HERO voxel plugin equivalence", "real-time Nav2 UC execution"]}
    (args.out / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": summary["all_synthetic_acceptance_pass"],
        "acceptance": acceptance, "scenes": {k: {field: v[field] for field in
        ("free_cells", "occupied_cells", "unknown_cells", "snapshot_wall_s")} for k, v in reports.items()}}, indent=2))
    if not summary["all_synthetic_acceptance_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
