#!/usr/bin/env python3
"""Diagnose startup support from copied live artifacts; never starts ROS.

Unknown support remains unknown. Ground truth is read only as an explicitly
separate displacement diagnostic, never as a navigation or actor input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_bt.support_costmap import path_support_check
from rl_bt.voxel_terrain import TerrainGrid


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def ready_response(path):
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("type") == "ready" and "raw" in row:
                return row
    raise ValueError(f"No actual ready response: {path}")


def grid_from_arrays(origin, resolution, occupancy, sampled_at_s):
    return TerrainGrid("odom", float(sampled_at_s), np.asarray(origin, float),
        float(resolution), occupancy, np.empty((0, 0)), np.empty((0, 0)),
        np.empty((0, 3)), {})


def intersecting_nonfree_cells(grid, center, prior_center, radius):
    """Closed square/disk intersection, using physical cell bounds."""
    cells = np.argwhere(grid.occupancy_xy != 0)
    lower = grid.origin_xy_m + cells * grid.resolution_m
    upper = lower + grid.resolution_m
    delta = np.maximum(np.maximum(lower - center, center - upper), 0.)
    distances = np.linalg.norm(delta, axis=1)
    selected = np.flatnonzero(distances <= radius + 1e-12)
    rows = []
    for index in selected:
        cell = cells[index]
        prior_distance = None
        if prior_center is not None:
            prior_distance = float(np.linalg.norm(np.maximum(np.maximum(
                lower[index] - prior_center, prior_center - upper[index]), 0.)))
        value = int(grid.occupancy_xy[tuple(cell)])
        rows.append({"cell_xy": cell.tolist(), "value": value,
            "label": "unknown" if value == -1 else "occupied",
            "lower_xy_m": lower[index].tolist(), "upper_xy_m": upper[index].tolist(),
            "distance_to_estimated_pose_m": float(distances[index]),
            "distance_to_fixed_prior_center_m": prior_distance})
    return rows


def gt_displacement_diagnostic(fixture, ready_s):
    """Use only world-coordinate displacement, never map/odom registration."""
    values = np.asarray(fixture.get("GT_trajectory_diagnostic_only", []), float)
    spawn = np.asarray(fixture.get("configured_spawn_world_xyzyaw", []), float)
    if values.ndim != 2 or values.shape[1] < 3 or spawn.shape != (4,) or not len(values):
        return {"available": False, "used_as_policy_or_support_input": False}
    if not np.isfinite(values[:, :3]).all() or not np.isfinite(spawn).all():
        raise ValueError("Nonfinite ground-truth diagnostic trajectory")
    nearest = values[int(np.argmin(np.abs(values[:, 0] - ready_s)))]
    last = values[-1]
    return {"available": True, "source": "GT_trajectory_diagnostic_only",
        "used_as_policy_or_support_input": False,
        "world_frame_registered_to_session_odom": False,
        "configured_spawn_world_xy_m": spawn[:2].tolist(),
        "nearest_ready_sample_s": float(nearest[0]),
        "nearest_ready_world_xy_m": nearest[1:3].tolist(),
        "nearest_ready_displacement_from_spawn_m": float(np.linalg.norm(nearest[1:3] - spawn[:2])),
        "final_sample_s": float(last[0]), "final_world_xy_m": last[1:3].tolist(),
        "final_displacement_from_spawn_m": float(np.linalg.norm(last[1:3] - spawn[:2]))}


def diagnose_episode(execution, *, radius):
    session = execution / "session"
    responses = session / "responses.jsonl"
    response = ready_response(responses)
    raw = response["raw"]
    pose = np.asarray(raw["pose_xy_m"], float)
    if pose.shape != (2,) or not np.isfinite(pose).all():
        raise ValueError(f"Invalid ready estimated pose: {execution}")
    relative = raw.get("support_grid_relative_path")
    if not relative:
        raise ValueError(f"Ready response lacks an actual support grid: {execution}")
    grid_path = (session / relative).resolve()
    if not grid_path.is_relative_to(session.resolve()):
        raise ValueError("Support snapshot path escapes its copied session")
    grid_hash = sha256(grid_path)
    if grid_hash != raw.get("support_grid_sha256"):
        raise ValueError(f"Copied support snapshot hash mismatch: {grid_path}")
    with np.load(grid_path, allow_pickle=False) as arrays:
        grid = grid_from_arrays(arrays["origin_xy_m"], arrays["resolution_m"],
            arrays["occupancy_xy"].copy(), arrays["sampled_at_s"])
    fixture_path = execution / "fixture/result.json"
    fixture = read_json(fixture_path)
    reports = [item for item in fixture.get("support_reports", [])
        if abs(float(item["sampled_at_s"]) - grid.sampled_at_s) <= 1e-6]
    prior = reports[-1].get("initial_support_prior") if reports else None
    prior_center = None if prior is None else np.asarray(prior["center_xy_m"], float)
    current = path_support_check(grid, [pose], radius_m=radius)
    at_prior = (None if prior_center is None else
        path_support_check(grid, [prior_center], radius_m=radius))
    translated = grid_from_arrays(grid.origin_xy_m - pose, grid.resolution_m,
        grid.occupancy_xy, grid.sampled_at_s)
    translated_check = path_support_check(translated, [[0., 0.]], radius_m=radius)
    recorded = raw.get("diagnostics", {}).get("current_footprint_check", {})
    if recorded.get("accepted") != current["accepted"] or recorded.get("reason") != current["reason"]:
        raise ValueError(f"Recomputed startup footprint disagrees with recorded check: {execution}")
    cells = intersecting_nonfree_cells(grid, pose, prior_center, radius)
    commands = fixture.get("commands", [])
    nonzero = sum(float(row[1]) ** 2 + float(row[2]) ** 2 > 1e-12 for row in commands)
    source = {}
    for name in ("scripts/nav2_mujoco_fixture.py", "scripts/nav2_rl_session.py",
                 "rl_bt/support_costmap.py", "rl_bt/live_nav_observation.py"):
        path = execution / "source" / name
        if path.is_file():
            source[name] = sha256(path)
    return {"execution_dir": str(execution), "ready_sim_s": float(raw["now_s"]),
        "estimated_pose_xy_m": pose.tolist(), "estimated_speed_mps": raw.get("estimated_speed_mps"),
        "raw_health_without_current_footprint": raw.get("health"),
        "legal_action_count": int(np.count_nonzero(response["observation"]["mask"])),
        "legal_navigation_candidate_count": int(np.count_nonzero(response["observation"]["mask"][:13])),
        "grid": {"origin_xy_m": grid.origin_xy_m.tolist(), "resolution_m": grid.resolution_m,
            "shape_xy": list(grid.occupancy_xy.shape), "sampled_at_s": grid.sampled_at_s},
        "fixed_start_prior": prior, "fixed_prior_matches_known_settled_base_center": False,
        "check_at_estimated_pose": current, "check_at_fixed_prior_center": at_prior,
        "intersecting_nonfree_cells": cells,
        "unknown_cell_count": sum(item["value"] == -1 for item in cells),
        "occupied_cell_count": sum(item["value"] == 100 for item in cells),
        "same_map_and_pose_after_consistent_translation": translated_check,
        "translation_preserves_acceptance": current["accepted"] == translated_check["accepted"],
        "MPPI_command_samples": len(commands), "nonzero_MPPI_command_samples": nonzero,
        "GT_displacement_diagnostic_only": gt_displacement_diagnostic(fixture, float(raw["now_s"])),
        "original_robot_and_physics_preserved": fixture.get("original_robot_and_physics_preserved"),
        "source_sha256": {"responses_jsonl": sha256(responses), "support_grid_npz": grid_hash,
            "fixture_result_json": sha256(fixture_path), "copied_implementation": source}}


def markdown(report):
    lines = ["# Live 启动支撑诊断", "", f"来源：`{report['source_run']}`。",
        "仅读取实际采集的 estimated pose、语义栅格和复制源码；未启动 ROS 或物理仿真，未修改支撑标签。",
        "", "| Episode | 首帧估计 XY (m) | 相交 unknown/occupied | 当前足迹 | 固定先验中心 | MPPI 指令 |",
        "|---|---|---:|---|---|---:|"]
    for index, row in enumerate(report["episodes"]):
        x, y = row["estimated_pose_xy_m"]
        prior = row["check_at_fixed_prior_center"]
        lines.append(f"| {index} | ({x:.6f}, {y:.6f}) | {row['unknown_cell_count']}/{row['occupied_cell_count']} | "
            f"{row['check_at_estimated_pose']['reason']} | {prior['reason'] if prior else '无先验记录'} | "
            f"{row['MPPI_command_samples']} |")
    lines.extend(["", "固定先验中心与估计器稳定后的 base 中心没有经过注册证明。"
        "未知栅格与实际估计足迹的相交距离、单元边界和源码 hash 均保存在同名 JSON 中。",
        "一致平移地图和位姿不会增加已观测支撑；不能仅重设坐标原点、移动先验或清除当前足迹来消除这些拒绝。",
        "", "GT 仅用于独立的物理位移诊断，不参与 actor、reward 或可通行标签。"
        "零 MPPI 指令情况下的位移应与导航执行区分。", ""])
    for index, row in enumerate(report["episodes"]):
        gt = row["GT_displacement_diagnostic_only"]
        if gt["available"]:
            lines.append(f"- Episode {index}：相对配置出生点的物理诊断位移，ready 附近 "
                f"{gt['nearest_ready_displacement_from_spawn_m']:.6f} m，末尾 "
                f"{gt['final_displacement_from_spawn_m']:.6f} m；实际指令 {row['MPPI_command_samples']} 条。")
    lines.extend(["", "保留全部启动失败，并将起点可执行率与任务完成率分别报告。"
        "本诊断没有证明可安全修改初始化、机器人参数或支撑门禁。", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    summary, config = read_json(run / "summary.json"), read_json(run / "config.json")
    executions = sorted(path.parent.parent for path in run.glob("sessions/**/fixture/result.json"))
    if not executions:
        raise ValueError("Run has no copied actual fixture results")
    rows = [diagnose_episode(path, radius=.411) for path in executions]
    report = {"artifact_type": "read_only_live_startup_support_diagnosis",
        "source_run": str(run), "scenario_id": summary.get("scenario_id"),
        "recorded_run_status": summary.get("status"), "radius_m": .411,
        "completed_episode_count": summary.get("completed_episodes"),
        "copied_physical_session_count": len(rows),
        "actual_observed_support_used": True, "GT_used_for_support_or_policy": False,
        "current_pose_unknown_filling": False, "ROS_or_simulator_launched": False,
        "ready_footprint_supported_count": sum(row["check_at_estimated_pose"]["accepted"] for row in rows),
        "consistent_translation_preserved_all_checks": all(row["translation_preserves_acceptance"] for row in rows),
        "episodes": rows, "source_sha256": {"summary_json": sha256(run / "summary.json"),
            "config_json": sha256(run / "config.json"), "diagnostic_script": sha256(__file__)},
        "declared_implementation_sha256": config.get("implementation_sha256", {}),
        "limits": ["Fixed prior center is an explicit engineering assumption, not observed support.",
            "Consistent coordinate translation cannot create support.",
            "GT world displacement is offline diagnostic only; no map/odom registration is inferred.",
            "All copied sessions are reported, including startup failures."]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "startup_support_diagnosis.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "startup_support_diagnosis.md").write_text(markdown(report))
    print(json.dumps({"output": str(path.resolve()), "sessions": len(rows),
        "ready_supported": report["ready_footprint_supported_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
