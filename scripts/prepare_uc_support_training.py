#!/usr/bin/env python3
"""Prepare an honest, small UC observed-support curriculum dataset.

This replays observed semantic grids, not mesh truth.  It does not produce
official match regions, opponent labels, or a replacement for Nav2 execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_bt.support_costmap import path_support_check
from rl_bt.voxel_terrain import TerrainGrid

MAP_SHA = "93ff0118af176f388c171b239acfdcac72c17cbe0b523350d5d916d3236fe10b"
RADIUS_M = .411
CANVAS = 121
MAX_GOALS = 16


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family_name(run_name):
    """Keep retries and final snapshots of a route together."""
    family = re.sub(r"_\d{8}$", "", run_name)
    family = family.replace("_final", "").replace("_namespaced", "")
    return family


def split_families(families):
    # This algorithm is fixed before training. No scores enter the ordering.
    ordered = sorted(set(families), key=lambda value:
        hashlib.sha256(("uc_support_training_v1:" + value).encode()).hexdigest())
    count = len(ordered)
    validation_n = max(1, round(count * .2)) if count >= 3 else 0
    test_n = max(1, round(count * .2)) if count >= 3 else 0
    return {family: ("test" if i < test_n else "validation"
        if i < test_n + validation_n else "train") for i, family in enumerate(ordered)}


def pad_unknown(occupancy, origin, resolution):
    occupancy = np.asarray(occupancy, dtype=np.int8)
    if occupancy.ndim != 2 or max(occupancy.shape) > CANVAS:
        raise ValueError("Expected semantic grid no larger than 121 by 121")
    if not np.isin(occupancy, (-1, 0, 100)).all():
        raise ValueError("Semantic labels must be unknown=-1, free=0, occupied=100")
    low = (np.array([CANVAS, CANVAS]) - occupancy.shape) // 2
    high = np.array([CANVAS, CANVAS]) - occupancy.shape - low
    result = np.pad(occupancy, tuple(zip(low, high)), constant_values=-1)
    return result, np.asarray(origin, float) - low * resolution, low


def matched_estimated_pose(rows, stamp, maximum_delta=.25):
    valid = [r for r in rows if len(r) >= 3 and np.isfinite(r[:3]).all()]
    if not valid:
        return None
    row = min(valid, key=lambda r: abs(float(r[0]) - stamp))
    delta = abs(float(row[0]) - stamp)
    if delta > maximum_delta:
        return None
    return np.asarray(row[1:3], float), float(row[0]), delta


def make_grid(occupancy, origin, resolution, stamp):
    return TerrainGrid("odom", stamp, np.asarray(origin), resolution,
        np.asarray(occupancy), np.empty((0, 0)), np.empty((0, 0)),
        np.empty((0, 3)), {})


def extract_goals(grid, start):
    """Use EDT proposals plus exact continuous disk checks for short corridors.

    EDT radius + a cell diagonal is a sufficient, deliberately conservative
    fast filter. A second proposal pass has no EDT margin and is accepted only
    by the same continuous .411 m disk check used by the live BT. This retains
    short genuinely supported corridors without filling any unknown cells.
    """
    if not path_support_check(grid, [start], radius_m=RADIUS_M)["accepted"]:
        return np.empty((0, 2)), [], 0
    free = grid.occupancy_xy == 0
    edt = distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1] * grid.resolution_m
    robust = edt > RADIUS_M + np.sqrt(2.) * grid.resolution_m
    cells = np.argwhere(free)
    centers = grid.origin_xy_m + (cells + .5) * grid.resolution_m
    distances = np.linalg.norm(centers - start, axis=1)
    near = (.12 <= distances) & (distances <= 2.)
    centers, cells = centers[near], cells[near]
    robust_order = np.argsort(~robust[tuple(cells.T)], kind="stable")
    proposals = [(centers[i], "edt_margin_and_exact" if robust[tuple(cells[i])]
        else "exact_cell_center") for i in robust_order]
    # .05 m targets are task suggestions, not newly freed map cells.
    proposals += [(start + np.array([x, y]), "exact_short_offset")
        for x in np.arange(-.5, .501, .05)
        for y in np.arange(-.5, .501, .05)
        if .12 <= np.hypot(x, y) <= .6]
    accepted, methods, seen = [], [], set()
    for point, method in proposals:
        key = tuple(np.round(point, 5))
        if key in seen:
            continue
        seen.add(key)
        if path_support_check(grid, [start, point], radius_m=RADIUS_M)["accepted"]:
            accepted.append(point)
            methods.append(method)
    if not accepted:
        return np.empty((0, 2)), [], int(robust.sum())
    points = np.asarray(accepted)
    # Farthest-point subsampling retains long alternatives and direction spread.
    selected = [int(np.argmax(np.linalg.norm(points - start, axis=1)))]
    while len(selected) < min(MAX_GOALS, len(points)):
        separation = np.min(np.linalg.norm(points[:, None] - points[selected], axis=2), axis=1)
        separation[selected] = -1.
        selected.append(int(np.argmax(separation)))
    return points[selected], [methods[i] for i in selected], int(robust.sum())


def support_prior_at(fixture, stamp):
    reports = fixture.get("support_reports", [])
    if not reports:
        return None
    report = min(reports, key=lambda item: abs(item["sampled_at_s"] - stamp))
    return report.get("initial_support_prior")


def summary(values):
    array = np.asarray(values, float)
    return {"n": int(len(array)), "min": float(array.min()) if len(array) else None,
        "median": float(np.median(array)) if len(array) else None,
        "max": float(array.max()) if len(array) else None,
        "values": array.tolist()}


def observation_signature(occupancy, origin, resolution, start):
    """Exact replay identity; file names and source timestamps are excluded."""
    h = hashlib.sha256()
    for array in (np.asarray(occupancy, dtype=np.int8),
            np.asarray(origin, dtype="<f8"), np.asarray([resolution], dtype="<f8"),
            np.asarray(start, dtype="<f8")):
        h.update(str(array.shape).encode("ascii"))
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def execution_calibration(runs_dir, family_splits):
    records = []
    for run in sorted(runs_dir.glob("uc_*")):
        fixture_path, action_path = run / "fixture/result.json", run / "action/result.json"
        fixture, action = read_json(fixture_path), read_json(action_path)
        if fixture.get("field_source", {}).get("prepared_mesh_sha256") != MAP_SHA or not action:
            continue
        events = action.get("events", [])
        response = next((e for e in events if e["event"] == "navigation_goal_response"
            and e.get("accepted")), None)
        if response is None:
            continue
        submitted = next((e for e in events if e["event"] == "BT_decision"
            and any(c["kind"] == "send_goal" for c in e["decision"]["commands"])), None)
        terminal = next((e for e in events if e["event"] == "navigation_terminal_result"), None)
        cancel = next((e for e in events if e["event"] == "navigation_health_at_cancel"), None)
        stamp = float(response["sim_time_s"])
        initial = matched_estimated_pose(fixture.get("estimated_poses", []), stamp)
        split = family_splits.get(family_name(run.name), "train")
        record = {"source_run": run.name, "family": family_name(run.name), "split": split,
            "result": action.get("result"), "source_sha256": {
                "fixture/result.json": digest(fixture_path), "action/result.json": digest(action_path)},
            "terrain_prior": fixture.get("terrain_prior", "clear_or_legacy_execution_only"),
            "acceptance_delay_s": None if submitted is None else max(0., stamp - submitted["sim_time_s"]),
            "task_time_s": None if terminal is None else terminal["sim_time_s"] - stamp,
            "goal_length_m": None, "effective_goal_speed_mps": None,
            "estimated_displacement_speed_mps": None,
            "support_dropout_cancel_delay_s": None, "stale_threshold_excess_s": None}
        if initial is not None and terminal is not None:
            record["goal_length_m"] = float(np.linalg.norm(np.asarray(action["goal_xy"]) - initial[0]))
            final = matched_estimated_pose(fixture.get("estimated_poses", []), terminal["sim_time_s"])
            duration = record["task_time_s"]
            if duration > 0 and action.get("result") == "succeeded":
                record["effective_goal_speed_mps"] = record["goal_length_m"] / duration
                if final is not None:
                    record["estimated_displacement_speed_mps"] = float(np.linalg.norm(final[0] - initial[0]) / duration)
        if cancel is not None and cancel.get("health_reason") == "support_map_missing_or_stale":
            injection = fixture.get("injected_support_dropout_after_simulation_s")
            if injection is not None:
                record["support_dropout_cancel_delay_s"] = cancel["sim_time_s"] - injection
            record["stale_threshold_excess_s"] = cancel["sim_time_s"] - cancel["support_sampled_at_s"] - 2.5
        records.append(record)
    keys = ["acceptance_delay_s", "task_time_s", "goal_length_m", "effective_goal_speed_mps",
        "estimated_displacement_speed_mps", "support_dropout_cancel_delay_s", "stale_threshold_excess_s"]
    return {"method": "Simulation-clock action events; estimated Point-LIO poses matched within .25 s; no GT. Speed summaries use successful tasks only. Clear-prior runs contribute timing only, never support labels.",
        "units": {"time": "simulation seconds", "length": "meters", "speed": "meters per simulation second"},
        "zero_acceptance_delay_interpretation": "Send and response may share a quantized simulator clock stamp. Zero does not establish zero transport/compute delay, and must not suppress deployment delay randomization.",
        "speed_interpretation": "Effective goal length divided by task time includes acceleration, stopping and goal tolerance; estimated displacement speed is not GT physical speed.",
        "training_summary": {k: summary([r[k] for r in records if r["split"] == "train" and r[k] is not None]) for k in keys},
        "all_observations_summary_for_audit_only": {k: summary([r[k] for r in records if r[k] is not None]) for k in keys},
        "all_records_for_audit_only": records,
        "heldout_records_used_to_fit_training_calibration": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--out", type=Path, default=ROOT / "data/derived/uc_support_training_v1")
    args = parser.parse_args()
    run_dirs = sorted(args.runs_dir.glob("uc_support*"))
    eligible = [run for run in run_dirs if read_json(run / "fixture/result.json").get(
        "field_source", {}).get("prepared_mesh_sha256") == MAP_SHA]
    family_splits = split_families([family_name(run.name) for run in eligible])
    samples, skipped, grids, origins, resolutions, poses, stamps, kinds = [], [], [], [], [], [], [], []
    goals, masks, starts, start_kinds, foots, split_ids = [], [], [], [], [], []
    for run in eligible:
        fixture = read_json(run / "fixture/result.json")
        for source_kind, relative in enumerate(("fixture/support_latest.npz", "action/semantic_support_at_plan.npz")):
            path = run / relative
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as source:
                stamp, resolution = float(source["sampled_at_s"]), float(source["resolution_m"])
                occupancy, origin, padding = pad_unknown(source["occupancy_xy"], source["origin_xy_m"], resolution)
            pose = matched_estimated_pose(fixture.get("estimated_poses", []), stamp)
            if pose is None:
                skipped.append({"source_run": run.name, "source_file": relative,
                    "reason": "No estimated pose within .25 simulation seconds"})
                continue
            grid = make_grid(occupancy, origin, resolution, stamp)
            foot_check = path_support_check(grid, [pose[0]], radius_m=RADIUS_M)
            prior = support_prior_at(fixture, stamp)
            selected_start, start_kind = pose[0], 0
            pool, pool_methods, robust_n = extract_goals(grid, selected_start)
            if len(pool) == 0 and prior is not None:
                fixed = np.asarray(prior["center_xy_m"], float)
                fixed_pool, fixed_methods, robust_n = extract_goals(grid, fixed)
                if len(fixed_pool):
                    selected_start, start_kind, pool, pool_methods = fixed, 1, fixed_pool, fixed_methods
            goal_array = np.zeros((MAX_GOALS, 2), np.float64)
            mask = np.zeros(MAX_GOALS, bool)
            goal_array[:len(pool)], mask[:len(pool)] = pool, True
            family, split = family_name(run.name), family_splits[family_name(run.name)]
            input_files = ["fixture/result.json", "action/result.json", "fixture/field_source.json",
                "verification.json", "source_sha256.json"]
            lengths = np.linalg.norm(pool - selected_start, axis=1)
            samples.append({"sample_id": len(samples), "source_run": run.name, "family": family,
                "split": split, "source_file": relative, "source_file_sha256": digest(path),
                "source_map_sha256": MAP_SHA,
                "input_artifact_sha256": {name: digest(run / name) for name in input_files if (run / name).exists()},
                "sampled_at_s": stamp, "pose_sampled_at_s": pose[1], "pose_time_delta_s": pose[2],
                "pose_source": "fixture.estimated_poses_PointLIO_only", "current_footprint_check": foot_check,
                "initial_prior": prior, "initial_prior_filled": bool(prior and prior.get("unknown_cells_assumed_supported", 0)),
                "start_kind": "matched_estimated_pose" if start_kind == 0 else "explicit_fixed_engineering_initial_xy",
                "fixed_start_reset_is_synthetic": start_kind == 1,
                "map_was_modified_beyond_unknown_padding": False,
                "added_unknown_padding_cells_xy": padding.tolist(), "robust_edt_endpoint_cells": robust_n,
                "candidate_count": len(pool), "candidate_methods": pool_methods,
                "candidate_length_range_m": [float(lengths.min()), float(lengths.max())] if len(lengths) else None,
                "observation_reset_sha256": observation_signature(occupancy, origin, resolution, selected_start),
                "no_candidate_condition": "wait_or_BT_recovery" if not len(pool) else None})
            grids.append(occupancy); origins.append(origin); resolutions.append(resolution)
            poses.append(pose[0]); stamps.append(stamp); kinds.append(source_kind)
            goals.append(goal_array); masks.append(mask); starts.append(selected_start)
            start_kinds.append(start_kind); foots.append(foot_check["accepted"])
            split_ids.append({"train": 0, "validation": 1, "test": 2}[split])
    if not samples:
        raise SystemExit("No matched observed support snapshots")
    identities = {}
    for sample in samples:
        identities.setdefault(sample["observation_reset_sha256"], []).append(sample)
    repeated = [{"sha256": key, "sample_ids": [sample["sample_id"] for sample in group],
        "splits": sorted({sample["split"] for sample in group})}
        for key, group in identities.items() if len(group) > 1]
    cross_split = [group for group in repeated if len(group["splits"]) > 1]
    if cross_split:
        raise ValueError("Exact occupancy/origin/resolution/reset-start replay duplicates cross dataset splits")
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "data.npz", occupancy_xy=np.asarray(grids, np.int8),
        origin_xy_m=np.asarray(origins), resolution_m=np.asarray(resolutions),
        pose_xy_m=np.asarray(poses), sampled_at_s=np.asarray(stamps), source_kind=np.asarray(kinds, np.int8),
        supported_goal_xy_m=np.asarray(goals), goal_mask=np.asarray(masks),
        goal_count=np.sum(masks, axis=1).astype(np.int16), start_xy_m=np.asarray(starts),
        start_source_kind=np.asarray(start_kinds, np.int8),
        current_footprint_supported=np.asarray(foots), split_id=np.asarray(split_ids, np.int8))
    splits = {"method": "Fixed SHA256 family ordering, 60/20/20 rounded families; no outcome or reward used. Same route retries/final runs remain one family.",
        "family_splits": family_splits,
        "sample_indices": {name: [s["sample_id"] for s in samples if s["split"] == name]
            for name in ("train", "validation", "test")},
        "episode_seeds": {"train": {"start": 1000, "exclusive_end": 85000},
            "validation": list(range(85001, 85033)), "test": list(range(95001, 95065))},
        "test_seeds_preregistered_before_training": True}
    splits["exact_observation_reset_duplicate_check"] = {
        "method": "SHA256 of exact padded occupancy, origin, resolution and reset start bytes; timestamps and file names excluded.",
        "groups": repeated, "cross_split_groups": cross_split,
        "local_geometry_overlap_remains": True}
    manifest = {"dataset": "uc_support_training_v1", "artifact_type": "observed_support_navigation_task_proxy_not_full_UC_match",
        "source_map_sha256": MAP_SHA, "samples": len(samples), "candidate_slots": MAX_GOALS,
        "canvas_xy": [CANVAS, CANVAS], "footprint_radius_m": RADIUS_M,
        "allowed_tasks": ["goto", "waypoint", "wait", "continue", "BT_recover"],
        "official_RFID_or_match_region_labels": False, "opponent_data_included": False,
        "GT_or_STL_as_actor_input": False, "unknown_filled_by_dataset_builder": False,
        "existing_fixed_initial_support_prior_retained_and_reported": True,
        "synthetic_resets_to_original_fixed_start_labeled": True,
        "pointlio_nav2_mppi_rollout_training": False,
        "limits": ["Only local observations around one engineering spawn; no geometric or match generalization claim.",
            "Maps are frozen replay snapshots; executor timing proxy is calibrated from separate actual events.",
            "Matched estimated current poses may be unsupported. Such conditions are retained; explicit fixed-start resets are labeled.",
            "No base healing, money, ammo, posture, target shooting, or opponent policy is modeled by this dataset.",
            "Held-out families and seeds reduce snapshot leakage but share the same local geometry."],
        "source_kind": {"0": "fixture/support_latest", "1": "action/semantic_support_at_plan"},
        "start_source_kind": {"0": "matched_estimated_pose", "1": "explicit_fixed_initial_prior_center"},
        "split_id": {"0": "train", "1": "validation", "2": "test"},
        "no_candidate_samples": sum(s["candidate_count"] == 0 for s in samples),
        "eligible_source_runs": len(eligible), "skipped": skipped}
    for name, document in (("metadata.json", {"samples": samples}), ("splits.json", splits),
            ("manifest.json", manifest), ("execution_calibration.json", execution_calibration(args.runs_dir, family_splits))):
        (args.out / name).write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "samples": len(samples), "split_indices": splits["sample_indices"],
        "candidate_counts": [s["candidate_count"] for s in samples]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
