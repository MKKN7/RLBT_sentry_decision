#!/usr/bin/env python3
"""Derive trace sanity statistics, UC mesh face features and a data-use manifest."""
from pathlib import Path
import csv
import hashlib
import json
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT / "data/derived"
    out.mkdir(exist_ok=True)
    trace = ROOT / "data/demos/alpharm_probe/episode_000.npz"
    with np.load(trace, allow_pickle=False) as a:
        speed = np.linalg.norm(a["privileged_state/vel"], axis=-1)
        fired = a["privileged_state/last_fired"][1:]
        stats = {"source": str(trace.relative_to(ROOT)), "source_sha256": digest(trace),
            "dataset_kind": "one_fixed_synthetic_3v3_episode_not_real_calibration",
            "environment_steps": len(a["actions"]), "agents": 6,
            "speed_mps_percentiles": dict(zip(["p50", "p90", "p99", "max"],
                                              np.quantile(speed, [.5, .9, .99, 1]).tolist())),
            "firing_agent_tick_fraction": float(fired.mean()),
            "observation_shape": list(a["obs"].shape),
            "used_as_direct_task_PPO_samples": False,
            "reason": "Original six-dimensional low-level actions lack our candidate/task labels."}
    (out / "alpharm_trace_statistics.json").write_text(json.dumps(stats, indent=2) + "\n")

    mesh = ROOT / "assets/navigation_reference/field/field.stl"
    content = mesh.read_bytes()
    n = struct.unpack_from("<I", content, 80)[0]
    if len(content) != 84 + 50 * n:
        raise ValueError("Not the expected binary STL")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    triangles = np.frombuffer(content, dtype=dtype, offset=84)["vertices"].astype(np.float64)
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norm = np.linalg.norm(normal, axis=1)
    valid = norm > 1e-12
    slope = np.full(n, 90.)
    slope[valid] = np.degrees(np.arccos(np.clip(np.abs(normal[valid, 2]) / norm[valid], 0, 1)))
    centroid = triangles.mean(axis=1)
    threshold = json.loads((ROOT / "configs/training_design.json").read_text())["terrain_design"]["max_traversable_slope_deg"]
    low = valid & (slope <= threshold + 1e-6)
    # Keep separate height samples per XY bin, never collapse to one height map.
    bins = np.floor(centroid[low, :2] / .20).astype(np.int32)
    unique, inverse = np.unique(bins, axis=0, return_inverse=True)
    zmin, zmax = np.full(len(unique), np.inf), np.full(len(unique), -np.inf)
    np.minimum.at(zmin, inverse, centroid[low, 2]); np.maximum.at(zmax, inverse, centroid[low, 2])
    derived = out / "uc_low_slope_face_candidates.npz"
    np.savez_compressed(derived, triangle_index=np.flatnonzero(low),
        centroid_m=centroid[low].astype(np.float32), slope_deg=slope[low].astype(np.float32),
        area_m2=(norm[low] * .5).astype(np.float32), xy_bin=bins,
        triangle_z_min_m=triangles[low, :, 2].min(axis=1).astype(np.float32),
        triangle_z_max_m=triangles[low, :, 2].max(axis=1).astype(np.float32))
    report = {"artifact_type": "mesh_face_features_not_a_traversability_map",
        "source": str(mesh.relative_to(ROOT)), "source_sha256": digest(mesh),
        "triangles": n, "degenerate_triangles": int((~valid).sum()),
        "bounds_m": [triangles.min(axis=(0, 1)).tolist(), triangles.max(axis=(0, 1)).tolist()],
        "selected_slope_threshold_deg": threshold, "low_slope_candidate_faces": int(low.sum()),
        "low_slope_total_face_area_m2_not_walkable_area": float(norm[low].sum() * .5),
        "xy_bin_size_m": .20, "xy_bins_with_height_span_over_015m": int(((zmax-zmin) > .15).sum()),
        "face_cache": str(derived.relative_to(ROOT)), "face_cache_sha256": digest(derived),
        "limitations": ["Centroid samples are not a rasterized support surface or complete coverage.",
            "Low slopes include roofs/ceilings; winding and support/overhead checks not certified.",
            "Multiple heights retained per XY bin; no single-layer projection used for planning.",
            "No official semantic zones, robot swept-envelope clearance, tunnel links or slope connectivity yet.",
            "Source CAD registration and team robot dimensions remain unverified."]}
    (out / "uc_geometry_report.json").write_text(json.dumps(report, indent=2) + "\n")

    catalog_path = ROOT / "data/demos/combat_video_catalog.csv"
    with catalog_path.open(newline="") as f:
        videos = list(csv.DictReader(f))
    queue = out / "combat_annotation_queue.csv"
    columns = ["video_id", "url", "source_season", "profile_hint", "start_s", "end_s",
               "actor_role", "task_label", "route_label", "confidence", "status"]
    with queue.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for v in videos:
            hint = "RMUL" if v["category"].startswith("RMUL") else (
                   "RMUC" if v["category"].startswith("RMUC") else "unknown")
            writer.writerow({"video_id": v["id"], "url": v["url"], "source_season": v["season"],
                             "profile_hint": hint, "status": "unannotated_no_training_labels"})
    usage = {"prepared_on": "2026-11-15", "real_opponent_telemetry_samples": 0,
        "sources": [
            {"path": "data/rule_facts.json", "use": "UC combat/stance/tunnel timer modules and boundary tests", "limits": "Partial rule implementation; not injected into simplified alphaRM game."},
            {"path": "data/references/alpharm_jax_review_manifest.json", "use": "Frozen ONNX environment, background policies, map constants and provenance", "limits": "Simplified game; original sources unmodified."},
            {"path": "data/demos/alpharm_probe/episode_000.npz", "use": "Offline sanity statistics", "limits": "No direct relabeling of velocities to BT candidate actions."},
            {"path": "assets/navigation_reference/field/field.stl", "use": "Layer-preserving low-slope face cache", "limits": "Not yet used as certified training/navigation map."},
            {"path": "data/demos/combat_video_catalog.csv", "use": "Versioned annotation queue", "limits": "No video downloaded, no invented actions/labels."},
            {"path": "data/references/combat_bt_review_manifest.json", "use": "Candidate/validation/monitoring boundary reference", "limits": "Python proxy gate only; actual Combat XML/C++/ROS not integrated."},
            {"path": "data/references/hero_nav_review_manifest.json", "use": "Actual navigation design and future calibration interface", "limits": "Point-LIO/Nav2/MPPI not run by phase-I proxy."},
            {"path": "data/references/bibliography.json", "use": "Actor-critic, temporal discount and held-out opponent design reference", "limits": "No claim of reproducing paper performance."}],
        "outputs": [str(p.relative_to(ROOT)) for p in sorted(out.glob('*')) if p.is_file()]}
    navigation_stats = out / "navigation_recording_statistics.json"
    if navigation_stats.is_file():
        navigation = json.loads(navigation_stats.read_text())
        usage["sources"].append({"path": str(navigation_stats.relative_to(ROOT)),
            "use": f"{sum(r['rows'] for r in navigation['records'])} simulated odometry rows summarized with source hashes",
            "limits": navigation["reason"]})
    usage["sources"].append({"path": "rl_bt/economy.py",
        "use": "Conservative UC quote/budget/transaction reference from user economy constraint",
        "limits": "Independent unit tests only; no economic training or live team-wallet synchronization."})
    (ROOT / "data/reports/phase1_data_reuse.json").write_text(json.dumps(usage, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"trace_steps": stats["environment_steps"], "uc_face_candidates": int(low.sum()),
                      "multi_height_xy_bins": report["xy_bins_with_height_span_over_015m"],
                      "video_annotation_queue": len(videos)}, indent=2))


if __name__ == "__main__":
    main()
