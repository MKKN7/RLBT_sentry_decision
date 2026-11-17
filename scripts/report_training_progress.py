#!/usr/bin/env python3
"""Read existing logs/checkpoints; show training trajectory without claiming convergence."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    with (run / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("No completed PPO update yet")
    configuration = json.loads((run / "config.json").read_text())
    reports = []
    for p in sorted(run.glob("validation_*/report.json")) + sorted(run.glob("test_*/report.json")):
        j = json.loads(p.read_text())
        reports.append({"report": str(p.relative_to(ROOT)), "decisions": j["training_decisions"],
            "split": j["evaluation_split"], "seeds": j["evaluation_seeds"],
            "groups": j["groups"], "paired": j["paired"], "by_style": j["by_style"],
            "by_side": j["by_side"]})
    reports.sort(key=lambda r: (r["split"], r["decisions"]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = [int(r["decisions"]) for r in rows]
    axes[0, 0].plot(x, [float(r["recent_return"]) for r in rows], label="Recent sampled training return")
    axes[0, 0].set(xlabel="Cumulative task decisions", ylabel="Return",
                   title="Training episodes fluctuate; not evaluation win rate")
    axes[0, 1].plot(x, [float(r["entropy"]) for r in rows], label="Entropy")
    axes[0, 1].plot(x, [float(r["explained_variance"]) for r in rows], label="Value explained variance")
    axes[0, 1].set(xlabel="Cumulative task decisions", title="Exploration and value fitting")
    axes[0, 1].legend()
    for key, label in (("choice_home_fraction", "Home"), ("choice_center_fraction", "Center variants"),
                       ("choice_continue_fraction", "Continue")):
        axes[1, 0].plot(x, [float(r[key]) for r in rows], label=label)
    axes[1, 0].set(xlabel="Cumulative task decisions", ylabel="Sampled action fraction", ylim=(0, 1),
                   title="Includes forced CONTINUE while dead")
    axes[1, 0].legend()
    validation = [r for r in reports if r["split"] == "validation"]
    for policy, label in (("bt_baseline", "BT validation"), ("learned", "RL validation")):
        values = [r["groups"][f"{policy}/all_styles"]["win_rate"] for r in validation]
        axes[1, 1].plot([r["decisions"] for r in validation], values, marker="o", label=label)
    axes[1, 1].set(xlabel="Frozen-checkpoint decisions", ylabel="Validation win fraction", ylim=(0, 1),
                   title="Paired seeds/sides/styles; small validation suite")
    axes[1, 1].legend()
    fig.suptitle("Continued candidate-selection PPO - simplified alphaRM proxy")
    fig.savefig(run / "training_progress.png", dpi=150)
    plt.close(fig)
    latest = rows[-1]
    report = {"scenario_id": configuration["scenario_id"], "official_UL_UC_ready": False,
        "training_status": json.loads((run / "status.json").read_text()),
        "resume_from": configuration["resume_from"], "resume_mode": configuration["resume_mode"],
        "target_cumulative_decisions": configuration["target_cumulative_decisions"],
        "latest_metrics": latest, "evaluations": reports,
        "convergence_status": "not_established_by_step_count_or_small_validation_suite",
        "figure": str((run / "training_progress.png").relative_to(ROOT))}
    (run / "progress_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"decisions": int(latest["decisions"]), "completed_training_episodes": int(latest["episode_count"]),
                      "evaluated_checkpoints": len(reports), "figure": report["figure"]}, indent=2))


if __name__ == "__main__":
    main()
