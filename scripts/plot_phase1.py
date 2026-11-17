#!/usr/bin/env python3
"""Standalone diagnostic figures for the prototype run and UC geometry cache."""
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
from matplotlib.patches import Polygon, Circle
import numpy as np
from rl_bt.geometry import reference_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    with (args.run / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    report = json.loads((args.run / "evaluation/report.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    steps = [int(r["decisions"]) for r in rows]
    axes[0, 0].plot(steps, [float(r["recent_return"]) for r in rows], marker=".")
    axes[0, 0].set(xlabel="Task decisions", ylabel="Recent training episode return",
                   title="Training reward (not evaluation win rate)")
    axes[0, 1].plot(steps, [float(r["value_loss"]) for r in rows], label="Value loss")
    axes[0, 1].set(xlabel="Task decisions", ylabel="MSE", title="PPO value fitting")
    names, means, low, high = [], [], [], []
    for name, g in report["groups"].items():
        if not g["episodes"]:
            continue
        names.append(name.replace("bt_baseline", "BT").replace("learned", "RL").replace("/", "\n"))
        means.append(g["win_rate"])
        low.append(g["win_rate"]-g["win_rate_wilson_95"][0])
        high.append(g["win_rate_wilson_95"][1]-g["win_rate"])
    axes[1, 0].bar(np.arange(len(names)), means, yerr=[low, high], capsize=4)
    axes[1, 0].set_xticks(np.arange(len(names)), names, fontsize=8)
    axes[1, 0].set(ylim=(0, 1.05), ylabel="Win fraction (Wilson 95% interval)",
                   title="Small paired prototype evaluation")
    cfg = reference_config()
    ax = axes[1, 1]
    for cover in cfg.covers:
        ax.add_patch(Polygon(cover, color="#586575"))
    ax.add_patch(Circle((0, 0), cfg.center_radius, color="#f4d35e", alpha=.25))
    for policy in ("bt_baseline", "learned"):
        with np.load(args.run / "evaluation" / f"{policy}_pursuit_side0.npz", allow_pickle=False) as a:
            xy = a["own_position"]
        ax.plot(xy[:, 0], xy[:, 1], label=policy.replace("bt_baseline", "BT").replace("learned", "RL"))
    ax.set(xlim=(-6, 6), ylim=(-4, 4), xlabel="World x (m)", ylabel="World y (m)",
           title="Learner paths vs held-out pursuit (red side)", aspect="equal")
    ax.legend(fontsize=8)
    fig.suptitle("alphaRM task prototype - not official RMUL/RMUC performance")
    fig.savefig(args.run / "diagnostics.png", dpi=150)
    plt.close(fig)
    with np.load(ROOT / "data/derived/uc_low_slope_face_candidates.npz", allow_pickle=False) as a:
        centroids = a["centroid_m"]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    plot = ax.scatter(centroids[:, 0], centroids[:, 1], c=centroids[:, 2], s=.5, cmap="viridis")
    fig.colorbar(plot, ax=ax, label="Face-centroid height (m)")
    ax.set(xlabel="Source world x (m)", ylabel="Source world y (m)", aspect="equal",
           title="UC low-slope face candidates: height retained, NOT a traversability map")
    fig.savefig(ROOT / "data/derived/uc_geometry_preview.png", dpi=150)
    plt.close(fig)
    print(args.run / "diagnostics.png")
    print(ROOT / "data/derived/uc_geometry_preview.png")


if __name__ == "__main__":
    main()
