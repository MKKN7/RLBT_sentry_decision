#!/usr/bin/env python3
"""Compare two frozen actors on identical recorded observations, without ROS."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_bt.uc_support_policy_v2 import UCSupportV2ActorCritic


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(checkpoint_a, checkpoint_b, runs):
    torch.set_num_threads(2)
    models, proofs = [], []
    for path in (checkpoint_a, checkpoint_b):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("scenario_id") != "uc_original_live_pointlio_nav2_mppi_task_v1":
            raise ValueError("Checkpoint is not from the live navigation scenario")
        model = UCSupportV2ActorCritic().eval()
        model.load_state_dict(checkpoint["model"])
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models.append(model)
        proofs.append({"path": str(path.resolve()), "sha256": digest(path)})
    rows = []
    for run in runs:
        path = run / "transitions.jsonl"
        traces = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        choice_count, differences = 0, []
        probability_l1, argmax_margins = [], []
        for row in traces:
            obs = row["observation"]
            tensors = {key: torch.as_tensor(np.asarray(obs[key]),
                dtype=torch.bool if key == "mask" else torch.float32).unsqueeze(0)
                for key in ("state", "candidates", "mask")}
            if not bool(tensors["mask"].any()):
                raise ValueError("Recorded observation has no legal action")
            choice_count += int(tensors["mask"].sum() > 1)
            with torch.no_grad():
                probs = [model.actor(**tensors).softmax(-1)[0] for model in models]
            actions = [int(p.argmax()) for p in probs]
            probability_l1.append(float((probs[0]-probs[1]).abs().sum()))
            if actions[0] != actions[1]:
                differences.append({"decision": row["decision"], "actions": actions})
        rows.append({"run": str(run.resolve()), "transition_sha256": digest(path),
            "recorded_observations": len(traces), "multiple_legal_action_observations": choice_count,
            "greedy_action_differences": len(differences), "different_decisions": differences,
            "mean_probability_L1_change": float(np.mean(probability_l1)) if traces else None})
    return {"artifact_type": "offline_identical_recorded_observation_comparison_not_physical_evaluation",
        "checkpoint_proof": proofs, "source_sha256": digest(Path(__file__)), "runs": rows,
        "total_observations": sum(row["recorded_observations"] for row in rows),
        "total_greedy_action_differences": sum(row["greedy_action_differences"] for row in rows),
        "limits": ["No physical rollout, optimizer, reward relabeling or hidden-state policy input.",
            "Identical actions on recorded observations do not prove identical behavior on unseen states.",
            "This comparison does not turn independent physics into paired seeded trials."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.checkpoint_a, args.checkpoint_b, args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("total_observations", "total_greedy_action_differences")}))
