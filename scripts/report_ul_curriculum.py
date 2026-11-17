#!/usr/bin/env python3
"""Summarize completed, versioned UL courses without inventing match metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    evaluation = json.loads((run / "evaluation.json").read_text())
    log = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()]
    episodes = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    hashes = config["implementation_sha256"]
    source_match = {}
    for relative, digest in hashes.items():
        frozen = run / "source" / relative
        if hashlib.sha256(frozen.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source does not match manifest: {relative}")
        source_match[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    test_seeds = set(config["evaluation_seeds"])
    train_seeds = {row["training_seed"] for row in episodes}
    warmstart = run / "bt_warmstart_report.json"
    if warmstart.exists():
        train_seeds |= set(json.loads(warmstart.read_text())["teacher_episode_seeds"])
    if train_seeds & test_seeds:
        raise ValueError("Training/evaluation seed overlap")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].plot([x["decisions"] for x in log], [x["recent_task_success_rate"] for x in log])
    axes[0].set(xlabel="PPO decisions", ylabel="Recent 20 task success fraction", ylim=(-.02, 1.02))
    labels = ["After BC", "After PPO", "BT"]
    counts = [evaluation["summary"][name]["successes"] for name in ("initialRL", "finalRL", "BT")]
    total = len(evaluation["paired_results"])
    bars = axes[1].bar(labels, counts, color=["#799bb5", "#356b91", "#67996c"])
    axes[1].bar_label(bars, labels=[f"{n}/{total}" for n in counts])
    axes[1].set(ylabel="Held-out tasks completed", ylim=(0, total*1.15))
    fig.suptitle(f"{summary['scenario_id']} — task curriculum, not match wins")
    fig.savefig(run / "training_progress.png", dpi=140)
    plt.close(fig)
    result = {"artifact_type": "UL_ground_course_result_not_match_acceptance",
        "run": str(run.relative_to(ROOT)), "summary": summary,
        "actor_parameters": config["actor_parameters"],
        "all_frozen_sources_match_recorded_hash": True,
        "current_source_matches_run": source_match,
        "training_evaluation_seed_overlap": [],
        "red_blue_and_three_objective_coverage": evaluation["covers_two_sides_and_three_objectives"],
        "training_geometry_failures": sum(row["geometry_failure"] for row in episodes),
        "training_BT_recoveries": sum(row["recovery_count"] for row in episodes),
        "UL_sentry_allowance_always_750_in_no_combat_course": all(row["ammo"] == 750 for row in episodes),
        "not_included": ["complete_3v3_combat", "opponent_style_curriculum", "UC_match_training",
            "PointLIO_Nav2_MPPI_runtime", "real_robot_tunnel_certification"],
        "reward_scope": "return_heal_capture_curriculum_not_formal_match_objective"}
    output = run / "verification.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
