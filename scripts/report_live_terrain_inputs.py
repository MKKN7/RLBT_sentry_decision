#!/usr/bin/env python3
"""Replay bounded, measured prefilter inputs; never launches ROS or physics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_bt.voxel_terrain import VoxelTerrainConfig, VoxelTerrainMap


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(execution, output_dir):
    execution = execution.resolve()
    fixture_dir = execution / "fixture"
    fixture_path = fixture_dir / "result.json"
    fixture = json.loads(fixture_path.read_text())
    targets = ((-4, -1), (-4, 0))
    captures, counts = [], []
    for item in fixture["captured_terrain_inputs"]:
        path = fixture_dir / item["path"]
        if digest(path) != item["sha256"] or item["GT_used"] is not False:
            raise ValueError("Prefilter capture hash/provenance mismatch")
        with np.load(path) as z:
            pts = z["registered_points_xyz"].copy()
            pose = z["estimated_pose_xyzyaw"].copy()
            sample, now = float(z["sampled_at_s"]), float(z["consumed_at_s"])
        captures.append((pts, pose, sample, now))
        distance = np.linalg.norm(pts[:, :2] - pose[:2], axis=1)
        columns = np.floor(pts[:, :2] / .15).astype(int)
        counts.append({"path": str(path), "sha256": digest(path), "sampled_at_s": sample,
            "target_columns": [{"global_column_xy": list(cell),
                "prefilter_returns": int(np.all(columns == cell, axis=1).sum()),
                "retained_045_returns": int((np.all(columns == cell, axis=1) & (distance > .45)).sum()),
                "retained_0411_returns": int((np.all(columns == cell, axis=1) & (distance > .411)).sum())}
                for cell in targets]})
    submitted = {round(float(x[0]), 9) for x in fixture["terrain_submissions"] if x[1] == "submitted"}
    published = {round(float(x["sampled_at_s"]), 9): x for x in fixture["terrain_reports"]}
    fields = ("observed_voxels", "free_cells", "occupied_cells", "unknown_cells", "reachable_support_layers")
    outputs = []
    raw_maps = []
    for radius in (.45, .411):
        mapper = VoxelTerrainMap(VoxelTerrainConfig(self_exclusion_radius_m=radius))
        pending, rows, maps = [], [], []
        for pts, pose, sample, now in captures:
            distance = np.linalg.norm(pts[:, :2] - pose[:2], axis=1)
            keep = np.isfinite(pts).all(axis=1) & (distance > radius) & (distance < 6.)
            pending.append(dict(points_xyz=pts[keep], sampled_at_s=sample, now_s=now,
                robot_position_xyz=pose[:3], frame_id="odom"))
            while pending and (len(pending) > 30 or now-pending[0]["sampled_at_s"] > 2.5 or
                               sum(len(scan["points_xyz"]) for scan in pending) > 100_000):
                pending.pop(0)
            if round(sample, 9) not in submitted:
                continue
            for scan in pending:
                status = mapper.update(**scan)
                if status != "accepted":
                    raise ValueError(f"Recorded input failed replay: {status}")
            grid = mapper.snapshot(robot_position_xyz=pose[:3], now_s=now)
            actual = published.get(round(sample, 9))
            match = actual is not None and all(grid.diagnostics[k] == actual[k] for k in fields)
            rows.append({"sampled_at_s": sample, "batch_scan_count": len(pending),
                "counts": {k: grid.diagnostics[k] for k in fields},
                "counts_match_recorded_baseline": match if radius == .45 else None})
            maps.append(grid.occupancy_xy.copy())
            pending = []
        outputs.append({"offline_self_filter_radius_m": radius, "replays": rows})
        raw_maps.append(maps)
    identical = len(raw_maps[0]) == len(raw_maps[1]) and all(
        np.array_equal(a, b) for a, b in zip(*raw_maps))
    baseline_ok = bool(outputs[0]["replays"]) and all(
        row["counts_match_recorded_baseline"] for row in outputs[0]["replays"])
    if not baseline_ok:
        raise ValueError("Replay does not reproduce actual baseline; no filter comparison conclusion")
    result = {"artifact_type": "measured_prefilter_input_diagnosis_not_navigation_training",
        "execution": str(execution), "fixture_sha256": digest(fixture_path),
        "script_sha256": digest(Path(__file__)), "capture_count": len(captures),
        "capture_limit": 80, "capture_limit_reached": len(captures) == 80,
        "uses_GT_or_mesh_labels": False, "planning_radius_unchanged_m": .411,
        "online_self_filter_unchanged_m": .45, "baseline_reproduced": baseline_ok,
        "all_compared_occupancy_arrays_identical": identical,
        "startup_before_5s_target_return_counts": [sum(cell["prefilter_returns"] for row in counts
            if row["sampled_at_s"] < 5. for cell in row["target_columns"]
            if cell["global_column_xy"] == list(target)) for target in targets],
        "counts": counts, "offline_replays": outputs,
        "limitations": ["Only bounded captured inputs replayed; no new physical execution.",
            "Capture is registered XYZ before radial filters, not raw sensor packets.",
            "Offline reduced exclusion is diagnostic only and was not deployed.",
            "No observations, support, prior or safety labels were synthesized."]}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "terrain_input_diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    (output_dir / "terrain_input_diagnosis.md").write_text(
        "# 过滤前注册点云诊断\n\n"
        f"采集 {len(captures)} 帧；按原始样本时间、接收时间、批次及TTL重放 "
        f"{len(outputs[0]['replays'])} 次分类，基线计数全部一致。\n\n"
        f"启动前5秒两块缺失列的返回点数为 {result['startup_before_5s_target_return_counts']}。"
        f"离线将 .45m 自车排除半径改为 .411m 后，所有重放占用数组完全一致：{identical}。"
        "因此原过滤不作修改；没有返回点的区域仍为unknown。\n\n"
        "本报告是采集文件的离线诊断，未移动机器人或生成训练样本；未使用GT或mesh生成支撑。"
        "圆足迹.411m、固定先验和unknown阻挡保持。详细文件hash、逐帧计数及限制见同名JSON。\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.execution, args.output_dir)
    print(json.dumps({k: result[k] for k in ("capture_count", "baseline_reproduced",
        "all_compared_occupancy_arrays_identical", "startup_before_5s_target_return_counts")}))
