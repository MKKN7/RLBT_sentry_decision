#!/usr/bin/env python3
"""Summarize actual UC execution records without interpreting them as RL scores."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def summarize(run):
    verification = json.loads((run / "verification.json").read_text())
    fixture = json.loads((run / "fixture/result.json").read_text())
    action = json.loads((run / "action/result.json").read_text())
    hashes = json.loads((run / "source_sha256.json").read_text())
    for name, expected in hashes.items():
        if hashlib.sha256((run / "source" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Execution source snapshot mismatch: {run}/{name}")
    estimate = np.asarray([row[:5] for row in fixture["estimated_poses"]], float)
    truth = np.asarray(fixture["GT_trajectory_diagnostic_only"], float)
    spawn = np.asarray(fixture["configured_spawn_world_xyzyaw"], float)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].plot(estimate[:, 1], estimate[:, 2], label="Point-LIO")
    # This fixed configured spawn subtraction is a diagnostic only. It is not
    # injected into TF, a filter, observations or a learned policy.
    axes[0].plot(truth[:, 1]-spawn[0], truth[:, 2]-spawn[1], "--", label="GT diagnostic")
    goal = np.asarray(verification["goal_xy"])
    axes[0].scatter(*goal, marker="*", s=90, label="goal")
    axes[0].set(xlabel="session x (m)", ylabel="session y (m)", title="Actual trajectory")
    axes[0].axis("equal"); axes[0].legend(fontsize=8)
    axes[1].plot(estimate[:, 0], np.linalg.norm(estimate[:, 1:3]-goal, axis=1))
    axes[1].axhline(.15, color="gray", linestyle="--", label="configured XY tolerance")
    axes[1].set(xlabel="simulation time (s)", ylabel="estimated distance (m)", title="Goal progress")
    axes[1].legend(fontsize=7)
    terrain = fixture["terrain_reports"]
    for key, label in (("free_cells", "observed support"), ("occupied_cells", "obstacle"),
                       ("unknown_cells", "unknown")):
        axes[2].plot([r["sampled_at_s"] for r in terrain], [r[key] for r in terrain], label=label)
    axes[2].set(xlabel="scan time (s)", ylabel="un-inflated terrain cells", title="Classifier output")
    axes[2].legend(fontsize=8)
    fig.suptitle(run.name + " | " + str(verification["actual_action_status"]), fontsize=10)
    fig.savefig(run / "trajectory.png", dpi=140); plt.close(fig)
    terminals = [event for event in action["events"] if event["event"] == "navigation_terminal_result"]
    terminal_s = terminals[-1]["sim_time_s"] if terminals else None
    return {"run": str(run), "verification": str(run / "verification.json"),
        "source_snapshot_hashes_verified": len(hashes), "scene": verification["scene"],
        "field_source": verification["field_source"], "goal_xy_m": verification["goal_xy"],
        "actual_action_status": verification["actual_action_status"],
        "success_for_reported_scope": verification["success_for_reported_scope"],
        "final_estimated_goal_error_m": verification["estimated_goal_xy_error_m"],
        "MPPI_command_samples": verification["MPPI_cmd_vel_samples"],
        "simulation_s": fixture["simulation_seconds"], "wall_s": fixture["wall_seconds"],
        "simulation_to_wall_ratio": fixture["simulation_seconds"] / fixture["wall_seconds"],
        "terminal_sim_time_s": terminal_s,
        "post_terminal_simulation_s": None if terminal_s is None else fixture["simulation_seconds"]-terminal_s,
        "post_terminal_wait_clock": "runner uses wall time, distinct from simulation time",
        "safety_stop_latched": verification["safety_stop_latched"],
        "GT_used_for_navigation": False, "RL_training_validated": False,
        "unknown_support_enforced_in_costmap": verification.get("observed_support_costmap", False),
        "fixed_start_footprint_prior":verification.get("fixed_start_footprint_prior",False),
        "terrain_sensors":verification.get("terrain_sensors","primary"),
        "navigation_succeeded":verification.get("navigation_succeeded",verification["actual_action_status"]=="succeeded"),
        "plot": str(run / "trajectory.png")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {"artifact_type": "actual_UC_navigation_execution_not_RL_training",
        "runs": [summarize(path) for path in args.run],
        "formal_match_or_full_field_reliability_validated": False,
        "original_robot_or_shared_system_modified": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
