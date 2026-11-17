#!/usr/bin/env python3
"""Multi-seed paired full-match prototype evaluation; not official UL/UC."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from rl_bt.alpharm_task_env import (
    AlphaRuntime, AlphaTaskEnv, SCENARIO, TRAIN_STYLES, HELD_OUT_STYLES, bt_baseline,
)
from rl_bt.policy import TaskActorCritic, choose


def wilson(wins, count):
    """Descriptive interval only: repeated sides/styles within a seed correlate."""
    if not count:
        return [None, None]
    z = 1.95996398454
    p = wins / count
    center = (p + z*z/(2*count)) / (1 + z*z/count)
    half = z * np.sqrt(p*(1-p)/count + z*z/(4*count*count)) / (1 + z*z/count)
    return [float(center-half), float(center+half)]


def cluster_interval(seed_means, bootstrap_samples=10000):
    """Resample complete perturbation-seed blocks, preserving paired matches."""
    values = np.asarray(seed_means, dtype=np.float64)
    if len(values) < 2:
        return [None, None]
    rng = np.random.default_rng(20261116)
    means = values[rng.integers(len(values), size=(bootstrap_samples, len(values)))].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def policy_summary(episodes):
    if not episodes:
        return {"episodes": 0, "wins": 0, "win_rate": None,
                "win_rate_wilson_95": [None, None], "draws": 0, "mean_return": None,
                "independent_seeds": 0, "win_rate_seed_bootstrap_95": [None, None]}
    wins = sum(e["win"] for e in episodes)
    seeds = sorted({e["seed"] for e in episodes})
    seed_wins = [np.mean([e["win"] for e in episodes if e["seed"] == seed]) for seed in seeds]
    result = {"episodes": len(episodes), "wins": wins,
            "win_rate": wins / len(episodes),
            "win_rate_wilson_95": wilson(wins, len(episodes)),
            "draws": sum(e["draw"] for e in episodes),
            "mean_return": float(np.mean([e["return"] for e in episodes])),
            "mean_switches": float(np.mean([e["switches"] for e in episodes])),
            "mean_overrides": float(np.mean([e["overrides"] for e in episodes])),
            "mean_recoveries": float(np.mean([e["recoveries"] for e in episodes])),
            "independent_seeds": len(seeds),
            "win_rate_seed_bootstrap_95": cluster_interval(seed_wins)}
    for key in ("sampled_alive_fraction", "sampled_center_fraction", "sampled_home_fraction",
                "damage_dealt", "damage_share_of_team", "overrides_per_decision"):
        values = [e["diagnostics"][key] for e in episodes if key in e.get("diagnostics", {})]
        if values:
            result["mean_" + key] = float(np.mean(values))
    return result


def run_trial(runtime, model, seed, style, side, policy, max_decisions, save_trace,
              reward_profile="upstream"):
    """Read-only model/session sharing; per-trial mutable state belongs to env."""
    env = AlphaTaskEnv(runtime, seed, randomization=True, opponent_style=style, side=side,
                       reward_profile=reward_profile)
    obs = env.cached
    trace = {k: [] for k in ("state", "candidates", "mask", "actions", "executed",
                             "reward", "duration", "done", "own_position")} if save_trace else None
    choices = np.zeros(16, dtype=int)
    executed = np.zeros(16, dtype=int)
    reasons = Counter()
    alive_s = center_s = home_s = sampled_s = 0.
    for _ in range(max_decisions):
        action = bt_baseline(obs) if policy == "bt_baseline" else choose(model, obs)
        if trace is not None:
            for key in ("state", "candidates", "mask"):
                trace[key].append(obs[key].copy())
            trace["own_position"].append(env.state_dict()["pos"][env.learner].copy())
        obs, reward, done, info = env.step(action)
        choices[action] += 1
        executed[info["executed_candidate"]] += 1
        if info["override_reason"]:
            reasons[info["override_reason"]] += 1
        duration = info["duration_s"]
        # Diagnostic sampling of raw own state after each task decision;
        # none of these privileged values are fed into the actor.
        own = env.obs[env.learner]
        alive_s += duration * float(own[11] > .5)
        center_s += duration * float(own[11] > .5 and own[21] > .5)
        home_s += duration * float(own[11] > .5 and own[20] > .5)
        sampled_s += duration
        if trace is not None:
            for key, value in (("actions", action), ("executed", info["executed_candidate"]),
                ("reward", reward), ("duration", duration), ("done", done)):
                trace[key].append(value)
        if done:
            state = env.state_dict()
            dealt = float(state["cum_damage"][env.learner].sum())
            teammates = slice(side * 3, side * 3 + 3)
            team_damage = float(state["cum_damage"][teammates].sum())
            result = {"policy": policy, "seed": seed, "held_out_style": style in HELD_OUT_STYLES,
                      **info["episode"], "diagnostics": {
                "candidate_counts": choices.tolist(), "executed_candidate_counts": executed.tolist(),
                "override_reasons": dict(reasons), "sampled_alive_fraction": alive_s / sampled_s,
                "sampled_center_fraction": center_s / sampled_s, "sampled_home_fraction": home_s / sampled_s,
                "damage_dealt": dealt, "damage_share_of_team": dealt / max(team_damage, 1e-9),
                "overrides_per_decision": sum(reasons.values()) / int(choices.sum()),
                "sampling_note": "Own alive/zone fractions sampled at post-decision states; no privileged actor input."}}
            packed = None if trace is None else {key: np.stack(values) for key, values in trace.items()}
            if packed is not None:
                packed.update(terminal_state=obs["state"], seed=np.asarray(seed))
            return result, packed
    raise RuntimeError("Unfinished full-episode evaluation; not counted as draw")


def paired_summary(episodes):
    """Compare policies in matching seed/style/side blocks, never head-to-head."""
    pairs = {}
    for episode in episodes:
        key = (episode["seed"], episode["opponent_style"], episode["team"])
        pair = pairs.setdefault(key, {})
        if episode["policy"] in pair:
            raise ValueError(f"Duplicate paired episode: {key}, {episode['policy']}")
        pair[episode["policy"]] = episode
    if any(set(pair) != {"bt_baseline", "learned"} for pair in pairs.values()):
        raise ValueError("Paired comparison requires one BT and one learned episode per block")
    differences = []
    for (seed, style, side), pair in sorted(pairs.items()):
        bt, learned = pair["bt_baseline"], pair["learned"]
        differences.append({"seed": seed, "style": style, "side": side,
            "win_difference": int(learned["win"]) - int(bt["win"]),
            "match_points_difference": (int(learned["win"]) + .5 * int(learned["draw"]))
                - (int(bt["win"]) + .5 * int(bt["draw"])),
            "return_difference": learned["return"] - bt["return"]})
    seeds = sorted({d["seed"] for d in differences})
    metrics = {}
    for metric in ("win_difference", "match_points_difference", "return_difference"):
        means = [float(np.mean([d[metric] for d in differences if d["seed"] == seed]))
                 for seed in seeds]
        metrics[metric] = {"mean": float(np.mean(means)) if means else None,
                           "seed_bootstrap_95": cluster_interval(means),
                           "per_seed": [{"seed": seed, "mean": value}
                                        for seed, value in zip(seeds, means)]}
    return {"pairs": len(pairs), "independent_seeds": len(seeds),
            "learned_wins_bt_does_not": sum(d["win_difference"] == 1 for d in differences),
            "bt_wins_learned_does_not": sum(d["win_difference"] == -1 for d in differences),
            "same_win_indicator": sum(d["win_difference"] == 0 for d in differences),
            "learned_minus_bt": metrics,
            "interpretation": "Matched trials against the same opponent, not direct RL-vs-BT games."}


def summarize(results):
    groups, paired, by_style, by_side = {}, {}, {}, {}
    splits = (("all_styles", None), ("seen_styles", False), ("held_out_style", True))
    for split, held in splits:
        group = [e for e in results if held is None or e["held_out_style"] == held]
        paired[split] = paired_summary(group)
        for policy in ("bt_baseline", "learned"):
            groups[f"{policy}/{split}"] = policy_summary([e for e in group if e["policy"] == policy])
    for style in sorted({e["opponent_style"] for e in results}):
        group = [e for e in results if e["opponent_style"] == style]
        by_style[style] = {"paired": paired_summary(group), **{
            policy: policy_summary([e for e in group if e["policy"] == policy])
            for policy in ("bt_baseline", "learned")}}
    for side in (0, 1):
        group = [e for e in results if e["team"] == side]
        by_side[str(side)] = {"paired": paired_summary(group), **{
            policy: policy_summary([e for e in group if e["policy"] == policy])
            for policy in ("bt_baseline", "learned")}}
    return {"groups": groups, "paired": paired, "by_style": by_style, "by_side": by_side}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10001,
                        help="First evaluation seed when --seeds is omitted")
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", help="Explicit independent evaluation seeds")
    parser.add_argument("--split", choices=("validation", "test"), default="validation",
                        help="Repeated checkpoint selection uses validation; reserve untouched test seeds")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--styles", nargs="+", choices=TRAIN_STYLES + HELD_OUT_STYLES,
                        default=list(TRAIN_STYLES + HELD_OUT_STYLES))
    parser.add_argument("--trace", choices=("none", "first-seed", "all"), default="first-seed")
    parser.add_argument("--workers", type=int, default=1, help="Parallel independent complete episodes")
    args = parser.parse_args()
    if args.seed_count < 1:
        parser.error("--seed-count must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    seeds = args.seeds if args.seeds is not None else list(range(args.seed, args.seed + args.seed_count))
    if len(set(seeds)) != len(seeds) or len(set(args.styles)) != len(args.styles):
        parser.error("Seeds and styles must not contain duplicates")
    torch.set_num_threads(2)
    model = TaskActorCritic()
    # Read once: a concurrently updated latest.pt must not yield a hash for a
    # different policy than the one held fixed throughout this evaluation.
    checkpoint_bytes = args.checkpoint.read_bytes()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    out = args.out or args.checkpoint.parent / "evaluation"
    out.mkdir(parents=True, exist_ok=False)
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    manifest = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
                "training_decisions": checkpoint.get("decisions"), "evaluation_split": args.split,
                "training_reward_profile": checkpoint["metadata"].get("reward_profile", "upstream"),
                "evaluation_reward_profile": "upstream",
                "reward_comparison": "Common original reward for all models; trained profiles never alter evaluation physics",
                "evaluation_seeds": seeds, "styles": args.styles, "sides": [0, 1],
                "randomization": True, "learned_action_selection": "deterministic_argmax",
                "trace": args.trace, "paired_on": ["seed", "style", "side"],
                "workers": args.workers,
                "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "implementation_sha256": {
                    p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (ROOT / "rl_bt").glob("*.py")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    runtime = AlphaRuntime()
    max_decisions = int(np.ceil(runtime.fixture["config"]["episode_time"] / .5)) + 10
    results = []
    started = time.perf_counter()
    trials = [(seed, style, side, policy) for seed in seeds for style in args.styles
              for side in (0, 1) for policy in ("bt_baseline", "learned")]
    def execute(trial):
        seed, style, side, policy = trial
        keep = args.trace == "all" or (args.trace == "first-seed" and seed == seeds[0])
        return run_trial(runtime, model, seed, style, side, policy, max_decisions, keep)
    with ThreadPoolExecutor(max_workers=args.workers) as pool, (out / "episodes.jsonl").open("w") as episode_file:
        for trial, (result, packed) in zip(trials, pool.map(execute, trials)):
            seed, style, side, policy = trial
            results.append(result)
            episode_file.write(json.dumps(result) + "\n")
            episode_file.flush()
            if packed is not None:
                suffix = "" if seed == seeds[0] else f"_seed{seed}"
                np.savez_compressed(out / f"{policy}_{style}_side{side}{suffix}.npz", **packed)
            (out / "status.json").write_text(json.dumps({"status": "running",
                "completed": len(results), "expected": len(trials)}, indent=2) + "\n")
            print(json.dumps({"completed": len(results), "expected": len(trials), **result}), flush=True)
    summary = {"scenario_id": SCENARIO, "official_competition_profile": None,
        **manifest, "evaluation_seed": seeds[0] if len(seeds) == 1 else None,
        "wall_s": time.perf_counter()-started, **summarize(results), "episodes": results,
        "assessment": "prototype_checkpoint_measurement_only; training convergence not established",
        "statistics": {"primary_comparison": "paired learned-minus-BT mean by independent seed",
            "confidence_interval": "percentile bootstrap of whole seed blocks, 10000 resamples",
            "wilson_interval": "descriptive unclustered interval; use seed bootstrap for comparisons",
            "bootstrap_limit": "Constant observed outcomes can collapse a bootstrap interval; this does not establish equivalence or bound unseen failures.",
            "seed_count_warning": "Fewer than 8 independent seeds gives a very unstable interval."
                if len(seeds) < 8 else None,
            "checkpoint_selection": "Repeated checkpoint comparisons belong to validation; test labels do not enforce seed isolation."},
        "limits": ["Simplified alphaRM task proxy, not formal RMUL/RMUC win rates.",
            "Fixed proxy navigation and autoaim; no ROS/MPPI or sentry stance validation.",
            "Opponent styles share the same environment and may not represent real teams.",
            "A completed training budget does not establish policy convergence."]}
    (out / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (out / "status.json").write_text(json.dumps({"status": "completed", "episodes": len(results)}, indent=2) + "\n")
    print(json.dumps({"groups": summary["groups"], "paired": summary["paired"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
