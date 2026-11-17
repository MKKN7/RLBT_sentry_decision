#!/usr/bin/env python3
"""Frozen common-case evaluation of simplified full-match temporal policies.

This explicitly does not run Nav2, official UL/UC rules or the Combat XML BT.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from rl_bt.alpharm_task_env import (AlphaRuntime, AlphaTaskEnv, TRAIN_STYLES,
                                   HELD_OUT_STYLES, bt_baseline)
from rl_bt.temporal_baselines import BTMemoryPolicy

SCENARIO_ID = "alpharm_temporal_global_phase_v1"


def model_weights_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def make_cases(seeds, styles, sides):
    if not seeds or not styles or not sides:
        raise ValueError("At least one evaluation seed, style and side is required")
    if any(len(set(items)) != len(items) for items in (seeds, styles, sides)):
        raise ValueError("Case dimensions must not contain duplicates")
    if any(seed < 0 for seed in seeds) or any(side not in (0, 1) for side in sides) or \
            any(style not in TRAIN_STYLES + HELD_OUT_STYLES for style in styles):
        raise ValueError("Unknown evaluation seed/style/side")
    return [{"seed": int(seed), "style": style, "side": int(side)}
            for seed, style, side in itertools.product(seeds, styles, sides)]


class FrozenPolicy:
    """Per-episode actor memory; shared model is inference-only."""
    def __init__(self, kind, model=None, *, reset_each_step=False):
        if kind not in ("bt_reactive", "bt_memory", "learned"):
            raise ValueError("Unknown policy kind")
        if kind == "learned" and model is None:
            raise ValueError("Learned inference requires a frozen model")
        self.kind, self.model = kind, model
        self.reset_each_step = bool(reset_each_step)
        self.bt = BTMemoryPolicy() if kind == "bt_memory" else None
        self.reset()

    def reset(self):
        self.steps = 0
        self.hidden = None
        self.hidden_reset_count = 0
        if self.bt is not None:
            self.bt.reset()
        if self.model is not None:
            self.hidden = self.model.initial_state(1, device="cpu")
            self.hidden_reset_count = 1

    @torch.no_grad()
    def act(self, observation):
        if self.kind == "bt_reactive":
            action = bt_baseline(observation)
        elif self.kind == "bt_memory":
            action = self.bt.act(observation)
        else:
            if self.reset_each_step and self.steps:
                self.hidden = self.model.initial_state(1, device="cpu")
                self.hidden_reset_count += 1
            tensors = {key: torch.as_tensor(np.asarray(observation[key])[None])
                       for key in ("state", "candidates", "mask")}
            distribution, _, self.hidden = self.model.distribution_value(
                **tensors, hidden=self.hidden,
                episode_start=torch.as_tensor([self.steps == 0 or self.reset_each_step]))
            action = int(distribution.logits.argmax(-1).item())
        self.steps += 1
        if not np.asarray(observation["mask"])[action]:
            raise RuntimeError("Frozen policy proposed an illegal candidate")
        return int(action)


def run_trial(runtime, spec, case, *, max_decisions=400, save_trace=False,
              env_factory=AlphaTaskEnv):
    """Policy inputs are only cached public tensors; truth is diagnostic-only."""
    env = env_factory(runtime, case["seed"], randomization=True,
                      opponent_style=case["style"], side=case["side"],
                      reward_profile="upstream")
    policy = FrozenPolicy(spec["kind"], spec.get("model"),
                          reset_each_step=spec.get("reset_each_step", False))
    observation = env.cached
    choices, executed, reasons = np.zeros(16, int), np.zeros(16, int), Counter()
    sampled_s = alive_s = center_s = home_s = own_hp_integral = 0.
    own_score_integral = enemy_score_integral = injected_missing = 0.
    trace = {key: [] for key in ("state", "candidates", "mask", "action",
                                 "executed", "reward", "duration_s", "terminated")}
    final_info = {}
    completed = False
    for _ in range(max_decisions):
        action = policy.act(observation)
        if save_trace:
            for key in ("state", "candidates", "mask"):
                trace[key].append(observation[key].copy())
        observation, reward, done, info = env.step(action)
        final_info = info
        choices[action] += 1
        executed[info["executed_candidate"]] += 1
        if info["override_reason"]:
            reasons[str(info["override_reason"])] += 1
        duration = float(info["duration_s"])
        if not np.isfinite(duration) or duration <= 0:
            raise RuntimeError("Actual transition duration must be positive")
        own, state = env.obs[env.learner], env.state_dict()
        own_alive = float(own[11] > .5)
        alive_s += duration * own_alive
        center_s += duration * own_alive * float(own[21] > .5)
        home_s += duration * own_alive * float(own[20] > .5)
        own_hp_integral += duration * float(own[7])
        own_score_integral += duration * float(state["score"][case["side"]])
        enemy_score_integral += duration * float(state["score"][1-case["side"]])
        injected_missing += duration * float(np.asarray(env.dropout).mean())
        sampled_s += duration
        if save_trace:
            for key, value in (("action", action), ("executed", info["executed_candidate"]),
                               ("reward", reward), ("duration_s", duration),
                               ("terminated", done)):
                trace[key].append(value)
        if done:
            completed = True
            break
    state = env.state_dict()
    team = case["side"]
    dealt = float(state["cum_damage"][env.learner].sum())
    team_dealt = float(state["cum_damage"][team*3:team*3+3].sum())
    episode = dict(final_info.get("episode", {}))
    # An unexpected decision cutoff is never silently called a match draw.
    if not completed:
        episode = {"return": float(env.episode_return), "decisions": int(choices.sum()),
                   "switches": int(env.switches), "overrides": int(env.overrides),
                   "recoveries": int(env.recoveries), "win": None, "draw": None,
                   "winner": None, "outcome": None, "score": state["score"].tolist()}
    result = {**episode, "policy": spec["label"], **case,
              "opponent_style": case["style"], "team": team,
              "held_out_style": case["style"] in HELD_OUT_STYLES,
              "completed": completed, "truncated": not completed,
              "actual_duration_s": sampled_s, "real_nav2_in_loop": False,
              "official_UL_UC_training": False,
              "memory_reset_each_step": bool(spec.get("reset_each_step", False)),
              "hidden_reset_count": policy.hidden_reset_count,
              "diagnostics": {"candidate_counts": choices.tolist(),
                "executed_candidate_counts": executed.tolist(), "override_reasons": dict(reasons),
                "sampled_alive_fraction": alive_s / sampled_s,
                "sampled_center_fraction": center_s / sampled_s,
                "sampled_home_fraction": home_s / sampled_s,
                "sampled_mean_own_hp_fraction": own_hp_integral / sampled_s,
                "sampled_mean_own_score": own_score_integral / sampled_s,
                "sampled_mean_enemy_score": enemy_score_integral / sampled_s,
                "injected_detection_missing_fraction": injected_missing / sampled_s,
                "damage_dealt": dealt, "damage_share_of_team": dealt / max(team_dealt, 1e-9),
                "overrides_per_decision": sum(reasons.values()) / int(choices.sum()),
                "ammunition": None,
                "ammunition_note": "No ammo state in this immutable simplified simulator; not inferred.",
                "sampling_note": "Post-decision diagnostic samples; no truth is supplied to any policy."}}
    if policy.bt is not None:
        result["bt_memory_final_state"] = policy.bt.state_dict()
    packed = {key: np.stack(values) for key, values in trace.items()} if save_trace else None
    return result, packed


def seed_cluster_interval(values, *, repeats=10000):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return [None, None]
    rng = np.random.default_rng(20261117)
    means = values[rng.integers(len(values), size=(repeats, len(values)))].mean(1)
    return np.quantile(means, [.025, .975]).tolist()


def summarize_policy(episodes):
    complete = [row for row in episodes if row["completed"]]
    result = {"episodes": len(episodes), "completed": len(complete),
              "truncated": len(episodes)-len(complete), "wins": sum(bool(e["win"]) for e in complete),
              "draws": sum(bool(e["draw"]) for e in complete)}
    result["win_rate"] = result["wins"] / len(complete) if complete else None
    result["match_points_rate"] = ((result["wins"]+.5*result["draws"]) / len(complete)
                                    if complete else None)
    for key in ("return", "decisions", "actual_duration_s", "switches", "overrides", "recoveries"):
        result["mean_" + key] = float(np.mean([row[key] for row in complete])) if complete else None
    for key in ("sampled_alive_fraction", "sampled_center_fraction", "sampled_home_fraction",
                "sampled_mean_own_hp_fraction", "damage_dealt", "damage_share_of_team",
                "injected_detection_missing_fraction", "overrides_per_decision"):
        result["mean_" + key] = (float(np.mean([row["diagnostics"][key] for row in complete]))
                                if complete else None)
    seeds = sorted({row["seed"] for row in complete})
    seed_means = [float(np.mean([row["win"] for row in complete if row["seed"] == seed]))
                  for seed in seeds]
    result.update(independent_exogenous_seeds=len(seeds),
                  win_rate_seed_bootstrap_95=seed_cluster_interval(seed_means))
    return result


def paired_contrast(episodes, left, right):
    pairs = {}
    for row in episodes:
        if row["policy"] not in (left, right):
            continue
        key = (row["seed"], row["style"], row["side"])
        pair = pairs.setdefault(key, {})
        if row["policy"] in pair:
            raise ValueError("Duplicate policy episode within an evaluation case")
        pair[row["policy"]] = row
    if any(set(pair) != {left, right} for pair in pairs.values()):
        raise ValueError("Each contrast requires matching complete case identities")
    differences = []
    for (seed, style, side), pair in sorted(pairs.items()):
        a, b = pair[left], pair[right]
        if not a["completed"] or not b["completed"]:
            continue
        differences.append({"seed": seed, "style": style, "side": side,
            "win": int(a["win"])-int(b["win"]),
            "match_points": int(a["win"])+.5*int(a["draw"])-int(b["win"])-.5*int(b["draw"]),
            "return": a["return"]-b["return"],
            "sampled_center_fraction": a["diagnostics"]["sampled_center_fraction"]
                                      -b["diagnostics"]["sampled_center_fraction"],
            "damage_dealt": a["diagnostics"]["damage_dealt"]-b["diagnostics"]["damage_dealt"],
            "switches": a["switches"]-b["switches"], "overrides": a["overrides"]-b["overrides"]})
    seeds = sorted({row["seed"] for row in differences})
    metrics = {}
    for key in ("win", "match_points", "return", "sampled_center_fraction", "damage_dealt", "switches", "overrides"):
        means = [float(np.mean([row[key] for row in differences if row["seed"] == seed]))
                 for seed in seeds]
        metrics[key] = {"mean": float(np.mean(means)) if means else None,
                        "seed_bootstrap_95": seed_cluster_interval(means),
                        "per_seed": dict(zip(map(str, seeds), means))}
    return {"left": left, "right": right, "difference": "left_minus_right",
            "matched_cases": len(pairs), "completed_pairs": len(differences),
            "independent_exogenous_seeds": len(seeds), "metrics": metrics,
            "interpretation": "Common perturbation seeds/styles/sides, not head-to-head games; policies change trajectories."}


def summarize(episodes, labels):
    groups = {}
    for split, held in (("all", None), ("seen", False), ("held_out", True)):
        selected = [row for row in episodes if held is None or row["held_out_style"] == held]
        groups[split] = {label: summarize_policy([row for row in selected if row["policy"] == label])
                         for label in labels}
    return {"groups": groups,
            "paired_contrasts": {f"{left}_minus_{right}": paired_contrast(episodes, left, right)
                                 for left, right in itertools.combinations(labels, 2)},
            "by_style": {style: {label: summarize_policy([row for row in episodes
                        if row["policy"] == label and row["style"] == style]) for label in labels}
                         for style in sorted({row["style"] for row in episodes})}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--gru-reset-ablation", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[51001, 51002])
    parser.add_argument("--styles", nargs="+", choices=TRAIN_STYLES+HELD_OUT_STYLES,
                        default=["capture", "pursuit"])
    parser.add_argument("--sides", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-decisions", type=int, default=400)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--trace", choices=("none", "first-case", "all"), default="first-case")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.max_decisions < 1:
        parser.error("Decision bound must be positive")
    cases = make_cases(args.seeds, args.styles, args.sides)
    specs = [{"label": "bt_reactive", "kind": "bt_reactive"},
             {"label": "bt_memory", "kind": "bt_memory"}]
    torch.set_num_threads(2)
    checkpoint_manifest = []
    if args.checkpoint:
        from scripts.train_temporal_phase import load_temporal_checkpoint
        for item in args.checkpoint:
            label, separator, raw_path = item.partition("=")
            if not separator or not label or not label.replace("_", "").replace("-", "").isalnum():
                parser.error("Each checkpoint must use a simple LABEL=PATH")
            path = Path(raw_path).resolve()
            checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            model, payload = load_temporal_checkpoint(path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint_hash:
                raise RuntimeError("Checkpoint changed while loading; evaluate an immutable file")
            model.eval()
            specs.append({"label": label, "kind": "learned", "model": model})
            checkpoint_manifest.append({"label": label, "path": str(path),
                "sha256": checkpoint_hash,
                "weights_before_sha256": model_weights_sha256(model),
                "memory": payload["metadata"]["memory"], "training_decisions": payload["decisions"],
                "training_seed": payload["metadata"].get("seed")})
            if args.gru_reset_ablation and payload["metadata"]["memory"] == "gru":
                specs.append({"label": label+"_reset", "kind": "learned", "model": model,
                              "reset_each_step": True})
    labels = [spec["label"] for spec in specs]
    if len(set(labels)) != len(labels):
        parser.error("Policy labels must be unique")
    args.out.mkdir(parents=True, exist_ok=False)
    runtime = AlphaRuntime()
    manifest = {"scenario_id": SCENARIO_ID, "evaluation_split": args.split,
        "checkpoints": checkpoint_manifest, "cases": cases, "policies": labels,
        "opponent_style_training_split": list(TRAIN_STYLES), "held_out_styles": list(HELD_OUT_STYLES),
        "evaluation_reward_profile": "upstream", "upstream_commit": runtime.source_commit,
        "randomization": True, "same_shared_executor": True,
        "learned_action_selection": "frozen_deterministic_argmax", "SGD_steps": 0,
        "real_nav2_in_loop": False, "official_UL_UC_training": False,
        "expected_episodes": len(cases)*len(specs), "max_decisions_per_episode": args.max_decisions,
        "workers": args.workers, "bt_memory_config": BTMemoryPolicy().state_dict()["config"],
        "implementation_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), ROOT/"rl_bt/temporal_baselines.py",
                         ROOT/"rl_bt/alpharm_task_env.py", ROOT/"rl_bt/geometry.py",
                         ROOT/"rl_bt/temporal_policy.py", ROOT/"scripts/train_temporal_phase.py")
            if path.is_file()},
        "limits": ["Simplified 180-second 3v3 proxy; not official UL/UC or real Nav2.",
            "HUD/enemy observations follow the frozen upstream prototype, not official legal telemetry.",
            "Diagnostic truth is recorded after action and never passed to any actor.",
            "No ammunition/economy state; unavailable fields are null.",
            "Two independent test seeds are insufficient for a strong generalization claim."]}
    (args.out/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    for relative in manifest["implementation_sha256"]:
        target = args.out/"source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
    trials = [(case_index, spec, case) for case_index, case in enumerate(cases) for spec in specs]
    def execute(trial):
        case_index, spec, case = trial
        keep = args.trace == "all" or (args.trace == "first-case" and case_index == 0)
        return run_trial(runtime, spec, case, max_decisions=args.max_decisions, save_trace=keep)
    started, results = time.perf_counter(), []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool, \
                (args.out/"episodes.jsonl").open("w") as output:
            for trial, (result, trace) in zip(trials, pool.map(execute, trials)):
                results.append(result)
                output.write(json.dumps(result, ensure_ascii=False)+"\n")
                output.flush()
                if trace is not None:
                    np.savez_compressed(args.out/f"{result['policy']}_seed{result['seed']}_{result['style']}_side{result['side']}.npz", **trace)
                status = {"status": "running", "completed_episodes": len(results),
                          "expected_episodes": len(trials), "wall_s": time.perf_counter()-started}
                (args.out/"status.json").write_text(json.dumps(status, indent=2)+"\n")
                print(json.dumps({"completed": len(results), "expected": len(trials),
                    "policy": result["policy"], "seed": result["seed"], "style": result["style"],
                    "side": result["side"], "win": result["win"], "truncated": result["truncated"]}), flush=True)
        for entry in checkpoint_manifest:
            model = next(spec["model"] for spec in specs if spec["label"] == entry["label"])
            entry["weights_after_sha256"] = model_weights_sha256(model)
            entry["weights_unchanged"] = entry["weights_after_sha256"] == entry["weights_before_sha256"]
            if not entry["weights_unchanged"]:
                raise RuntimeError("Frozen evaluation unexpectedly changed model weights")
        summary = {**manifest, **summarize(results, labels),
            "wall_s": time.perf_counter()-started, "actual_episodes": len(results),
            "unexpected_truncations": sum(row["truncated"] for row in results),
            "status": "complete" if all(row["completed"] for row in results) else "incomplete",
            "episodes": results,
            "statistics_note": "Seed-block bootstrap is descriptive, especially with only two seeds; no checkpoint selected."}
        (args.out/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
        (args.out/"status.json").write_text(json.dumps({"status": summary["status"],
            "completed_episodes": len(results), "expected_episodes": len(trials),
            "unexpected_truncations": summary["unexpected_truncations"]}, indent=2)+"\n")
        if summary["unexpected_truncations"]:
            raise RuntimeError("At least one evaluation episode was cut short; not counted as draw")
    except BaseException as error:
        (args.out/"status.json").write_text(json.dumps({"status": "error", "error": repr(error),
            "completed_episodes": len(results), "expected_episodes": len(trials)}, indent=2)+"\n")
        raise


if __name__ == "__main__":
    main()
