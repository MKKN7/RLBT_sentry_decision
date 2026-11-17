#!/usr/bin/env python3
"""Exercise the UL course graph and save small reproducible proxy artifacts."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_bt.ul_navigation import ULNavigation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data/derived/ul_navigation_proxy_v1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    nav = ULNavigation()
    build_s = time.perf_counter() - started
    candidates = {str(team): nav.goal_candidates(team) for team in (0, 1)}
    checks, paths = {}, {}
    for team in (0, 1):
        own = candidates[str(team)][0]["xy"]
        for candidate in candidates[str(team)]:
            path = nav.plan(own, candidate["xy"], team)
            key = f"team{team}_candidate{candidate['id']}"
            checks[key] = {"path_found": path is not None,
                "static_legal": candidate["static_legal"],
                "path_length_m": None if path is None else float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())}
            if path is not None:
                paths[key] = path
        enemy = [10.5, 1.] if team == 0 else [1.5, 7.]
        checks[f"team{team}_enemy_supply"] = {"path_found": nav.plan(own, enemy, team) is not None}
    query_started = time.perf_counter()
    for _ in range(1000):
        nav.plan([1.5, 7.], [6., 4.], 0)
    thousand_queries_s = time.perf_counter() - query_started
    np.savez_compressed(args.out / "graph_masks.npz", origin_xy_m=nav.origin_xy_m,
        resolution_m=nav.resolution_m, supported=nav.supported_cells,
        obstacle=nav.obstacle_cells, red_free=nav.free_by_team[0], blue_free=nav.free_by_team[1])
    np.savez_compressed(args.out / "candidate_paths.npz", **paths)
    report = nav.summary() | {"build_s": build_s,
        "cached_1000_queries_s": thousand_queries_s, "candidates": candidates,
        "checks": checks, "candidate_schema_version": "ul_ground_course_v1",
        "tunnel_clearance_nominal_m": .2495,
        "tunnel_reference_height_test": "blocked: planning height .255m exceeds nominal .2495m gap",
        "ground_support_policy": "upward near-ground faces from -1e-6 to .015m only; floor bottom -.1m excluded",
        "grid_inflation_policy": "planning circle plus both raw/destination square half-diagonals",
        "enemy_supply_policy": "reject entire pose cell when planning circle may intersect nominal enemy supply",
        "training_use": "task-level course motion proxy only; not a formal match or live hardware navigation mask"}
    report["written_npz_bytes"] = sum(p.stat().st_size for p in args.out.glob("*.npz"))
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "build_s": build_s,
        "cached_1000_queries_s": thousand_queries_s,
        "written_npz_bytes": report["written_npz_bytes"],
        "free_cells_red_blue": report["free_cells_red_blue"],
        "legal_candidate_ids": {team: [c["id"] for c in cc if c["static_legal"]]
                                for team, cc in candidates.items()},
        "hardware_reachability_certified": False}))


if __name__ == "__main__":
    main()
