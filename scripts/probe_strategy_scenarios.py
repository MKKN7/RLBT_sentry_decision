#!/usr/bin/env python3
"""Development-only control gate for isolated role-balanced proxy scenes.

No learned model is loaded, no optimizer exists and this is never a final test.
The original ONNX terminal, physical rules and skill execution remain intact.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rl_bt.alpharm_task_env import AlphaRuntime
from rl_bt.strategy_scenarios import RoleBalancedTaskEnv, SCENARIO_ID, SUPPORT_MODES
from scripts.audit_temporal_policy_leverage import run_control_trial
from scripts.evaluate_temporal_phase import make_cases, run_trial, summarize_policy

POLICIES = ("continue_only", "fixed_center", "bt_reactive", "bt_memory")
SCRIPTED_STYLES = ("capture", "resupply")
ALLOWED_STYLES = SCRIPTED_STYLES + ("pretrained",)
MODES = SUPPORT_MODES
SCENE_CONFIG = "configs/strategy_scenarios_v2.json"
SOURCES = ("scripts/probe_strategy_scenarios.py", "rl_bt/strategy_scenarios.py", SCENE_CONFIG,
    "scripts/audit_temporal_policy_leverage.py", "scripts/evaluate_temporal_phase.py",
    "rl_bt/temporal_baselines.py", "rl_bt/alpharm_task_env.py", "rl_bt/geometry.py",
    "rl_bt/contracts.py", "rl_bt/rewards.py", "rl_bt/reward_components.py",
    "scripts/probe_alpharm_onnx.py", "data/references/alpharm_jax_review_manifest.json")
GATE_THRESHOLDS = {"minimum_different_case_fraction": .25,
    "minimum_BT_return_advantage_strictly_greater": .5,
    "minimum_median_duration_s": 60., "long_episode_s": 90., "minimum_long_fraction": .25}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def points(row):
    return float(row["win"]) + .5*float(row["draw"])


def evaluate_gate(rows, mode, *, expected_cases=None):
    """Read raw scripted-case rows; pretrained strong-pool never drives gate."""
    selected = [row for row in rows if row["support_mode"] == mode and
                row["style"] in SCRIPTED_STYLES]
    by_case = {}
    duplicate = False
    for row in selected:
        key = (row["seed"], row["style"], row["side"])
        entries = by_case.setdefault(key, {})
        duplicate |= row["policy"] in entries
        entries[row["policy"]] = row
    expected = (set(expected_cases) if expected_cases is not None else set(by_case))
    identities_match = bool(expected) and set(by_case) == expected and not duplicate and \
        all(set(entries) == set(POLICIES) for entries in by_case.values())
    complete = identities_match and all(row.get("completed", False) and not row.get("truncated", True)
                                         for row in selected)
    # Incomplete cases remain visible and force rejection; they are never draw.
    eligible = [row for row in selected if row.get("completed") and not row.get("truncated")]
    wins = sum(bool(row["win"]) for row in eligible)
    losses = sum(not row["win"] and not row["draw"] for row in eligible)
    mixed = wins > 0 and losses > 0
    paired = []
    for key, entries in sorted(by_case.items()):
        if not {"continue_only", "bt_reactive", "bt_memory"} <= set(entries) or \
                any(not entries[label].get("completed") or entries[label].get("truncated")
                    for label in ("continue_only", "bt_reactive", "bt_memory")):
            continue
        hold = entries["continue_only"]
        comparison = {}
        for label in ("bt_reactive", "bt_memory"):
            bt = entries[label]
            comparison[label] = {"win_changed": bool(bt["win"]) != bool(hold["win"]),
                "points_changed": points(bt) != points(hold),
                "points_delta": points(bt)-points(hold),
                "return_delta": float(bt["return"])-float(hold["return"]),
                "center_delta": bt["diagnostics"]["sampled_center_fraction"]
                                -hold["diagnostics"]["sampled_center_fraction"],
                "damage_delta": bt["diagnostics"]["damage_dealt"]-hold["diagnostics"]["damage_dealt"]}
        paired.append({"seed": key[0], "style": key[1], "side": key[2], **comparison})
    different = sum(any(row[label]["win_changed"] or row[label]["points_changed"]
                        for label in ("bt_reactive", "bt_memory")) for row in paired)
    different_fraction = different / len(paired) if paired else 0.
    comparisons = {}
    for label in ("bt_reactive", "bt_memory"):
        comparisons[label] = {key: (float(np.mean([row[label][key] for row in paired])) if paired else None)
                              for key in ("points_delta", "return_delta", "center_delta", "damage_delta")}
    bt_better = any(row["points_delta"] is not None and (row["points_delta"] > 0 or
                     row["return_delta"] > GATE_THRESHOLDS["minimum_BT_return_advantage_strictly_greater"])
                    for row in comparisons.values())
    duration = np.asarray([row["actual_duration_s"] for row in eligible], float)
    median = float(np.median(duration)) if len(duration) else None
    long_fraction = float(np.mean(duration >= GATE_THRESHOLDS["long_episode_s"])) if len(duration) else 0.
    criteria = {"complete_case_identity_and_true_episodes": bool(complete),
                "at_least_one_true_win_and_loss": bool(mixed),
                "HOLD_vs_BT_outcomes_differ_at_least_25pct_cases": bool(
                    different_fraction >= GATE_THRESHOLDS["minimum_different_case_fraction"]),
                "BT_better_mean_points_or_return_above_half": bool(bt_better),
                "median_duration_at_least_60s": bool(median is not None and median >= 60.),
                "at_least_quarter_episodes_90s": bool(long_fraction >= .25)}
    return {"support_mode": mode, "passed": all(criteria.values()), "criteria": criteria,
        "thresholds": GATE_THRESHOLDS, "scripted_styles_only": list(SCRIPTED_STYLES),
        "expected_scripted_cases": len(expected), "observed_unique_scripted_cases": len(by_case),
        "scripted_episodes": len(selected), "true_complete_episodes": len(eligible),
        "wins": wins, "losses": losses, "draws": sum(bool(row["draw"]) for row in eligible),
        "paired_cases": len(paired), "HOLD_vs_either_BT_different_cases": different,
        "HOLD_vs_either_BT_different_case_fraction": different_fraction,
        "BT_minus_HOLD": comparisons, "matched_raw_differences": paired,
        "duration_quantiles_s": (dict(zip(("min", "q25", "median", "q75", "max"),
                                          np.quantile(duration, [0,.25,.5,.75,1]).tolist()))
                                   if len(duration) else None),
        "fraction_at_least_90s": long_fraction,
        "assessment": "development engineering gate only; no significance, final-test or training acceptance claim"}


def run_one(runtime, mode, policy, case, *, max_decisions=400, env_factory=RoleBalancedTaskEnv):
    factory = partial(env_factory, support_mode=mode)
    if policy in ("continue_only", "fixed_center"):
        row = run_control_trial(runtime, policy, case, max_decisions=max_decisions, env_factory=factory)
    else:
        row, _ = run_trial(runtime, {"label": policy, "kind": policy}, case,
                           max_decisions=max_decisions, save_trace=False, env_factory=factory)
    if not row["completed"] or row["truncated"]:
        raise RuntimeError(f"Unexpected development cutoff for {mode}/{policy}/{case}; not counted as draw")
    if row.get("scenario_id") != SCENARIO_ID or row.get("strategy_scene", {}).get("support_mode") != mode:
        raise RuntimeError("Actual scene terminal metadata differs from requested role-balanced mode")
    row.update(support_mode=mode, data_role="development_calibration", exploratory_development=True,
               exploratory_posthoc=False, independent_new_test=False, checkpoint_selection=False,
               learned_weights_loaded=False, SGD_steps=0)
    return row


def gate_markdown(gates):
    lines = ["# Role-balanced 开发场景门槛", "",
        "只统计 scripted capture/resupply；pretrained 强对手池单列。此为开发校准，未训练模型，也不是最终测试。", "",
        "| 模式 | 通过 | 完整局 | 胜／负／平 | HOLD与BT不同场景占比 | 局长中位数s | ≥90s占比 |", "|---|---|---:|---:|---:|---:|---:|"]
    for mode, gate in gates.items():
        duration = gate["duration_quantiles_s"]
        median = "NA" if duration is None else f"{duration['median']:.1f}"
        lines.append(f"| {mode} | {gate['passed']} | {gate['true_complete_episodes']}/{gate['scripted_episodes']} | "
            f"{gate['wins']}/{gate['losses']}/{gate['draws']} | {gate['HOLD_vs_either_BT_different_case_fraction']:.3f} | {median} | {gate['fraction_at_least_90s']:.3f} |")
    for mode, gate in gates.items():
        lines.extend(["", f"{mode} 各项：", ""]+[f"- {key}: {value}" for key, value in gate["criteria"].items()]+[""])
    lines += ["", "失败项保留，门槛不通过不自动训练。真实终局、原物理／规则／奖励常数与安全条件保持。", ""]
    return "\n".join(lines)


def policy_statistics(rows):
    """Extend measured-policy summary with actual action and outcome counts."""
    result = summarize_policy(rows)
    result["true_outcome_code_counts"] = {str(code):sum(row.get("outcome") == code for row in rows)
        for code in sorted({row["outcome"] for row in rows if row.get("outcome") is not None})}
    for key in ("candidate_counts", "executed_candidate_counts"):
        arrays = [row["diagnostics"][key] for row in rows if key in row.get("diagnostics", {})]
        result["total_"+key] = np.asarray(arrays, dtype=np.int64).sum(0).tolist() if arrays else None
    duration = [row["actual_duration_s"] for row in rows if row.get("completed") and not row.get("truncated")]
    result["duration_quantiles_s"] = (dict(zip(("min","q25","median","q75","max"),
        np.quantile(duration,[0,.25,.5,.75,1]).tolist())) if duration else None)
    return result


def run_probe(out, *, seeds=(62011,), styles=ALLOWED_STYLES, modes=MODES,
              workers=2, max_decisions=400, runtime_factory=AlphaRuntime):
    if workers not in (1,2) or max_decisions < 1:
        raise ValueError("Workers1/2 and positive decision bound required")
    if any(style not in ALLOWED_STYLES for style in styles) or not set(SCRIPTED_STYLES) <= set(styles):
        raise ValueError("Development gate requires capture/resupply; only pretrained may be added")
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError("Choose unique declared support modes")
    if any(seed in (51001,51002,73001,73002) for seed in seeds):
        raise ValueError("Frozen old/new final-test seeds cannot enter development calibration")
    cases = make_cases(list(seeds), list(styles), [0,1])
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_hashes = {relative: sha(ROOT/relative) for relative in SOURCES}
    runtime = runtime_factory()
    expected = len(modes)*len(cases)*len(POLICIES)
    manifest = {"scenario_id": SCENARIO_ID, "artifact_type": "development_control_calibration_not_final_test",
        "data_role": "development_calibration", "exploratory_development": True,
        "independent_new_test": False, "checkpoint_selection": False,
        "policies": list(POLICIES), "modes": list(modes), "cases": cases,
        "expected_episodes": expected, "gate_thresholds": GATE_THRESHOLDS,
        "gate_styles": list(SCRIPTED_STYLES), "strong_pool_styles": ["pretrained"],
        "learned_weights_loaded": False, "optimizers_created": 0, "SGD_steps": 0,
        "real_nav2_in_loop": False, "official_UL_UC_training": False,
        "same_shared_executor": True, "randomization": True, "reward_profile": "upstream",
        "workers": workers, "max_decisions": max_decisions, "upstream_commit": runtime.source_commit,
        "implementation_sha256_before": source_hashes,
        "scene_config_sha256": source_hashes[SCENE_CONFIG],
        "limits": ["Synthetic role curriculum; no official UL/UC or actual Nav2 episode.",
            "Development results may guide later scene selection; never report as independent final test.",
            "Continue preserves the existing combat skills and other robot execution.",
            "No terminal suppression, fixture mutation or physical/rule/reward constant edits."]}
    write_json(out/"manifest.json", manifest)
    for relative in SOURCES:
        target = out/"source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
    trials = list(itertools.product(modes, cases, POLICIES))
    results, started = [], time.perf_counter()
    write_json(out/"status.json", {"status":"running", "completed_episodes":0, "expected_episodes":expected})
    try:
        def execute(trial):
            mode, case, policy = trial
            return run_one(runtime, mode, policy, case, max_decisions=max_decisions)
        with ThreadPoolExecutor(max_workers=workers) as pool, (out/"episodes.jsonl").open("w") as output:
            for row in pool.map(execute, trials):
                results.append(row)
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
                output.flush()
                status = {"status":"running", "completed_episodes":len(results), "expected_episodes":expected,
                    "last_policy":row["policy"], "last_mode":row["support_mode"],
                    "last_case":{key:row[key] for key in ("seed","style","side")},
                    "last_win":row["win"], "actual_duration_s":row["actual_duration_s"],
                    "wall_s":time.perf_counter()-started}
                write_json(out/"status.json", status)
                print(json.dumps(status), flush=True)
        hashes_after = {relative: sha(ROOT/relative) for relative in SOURCES}
        if hashes_after != source_hashes or any(sha(out/"source"/relative) != expected_hash
                                               for relative,expected_hash in source_hashes.items()):
            raise RuntimeError("Scene/probe source changed during development calibration")
        expected_scripted = {(case["seed"],case["style"],case["side"]) for case in cases
                             if case["style"] in SCRIPTED_STYLES}
        gates = {mode: evaluate_gate(results, mode, expected_cases=expected_scripted) for mode in modes}
        groups = {mode:{split:{policy:policy_statistics([row for row in results if
            row["support_mode"] == mode and row["policy"] == policy and (split == "all" or
             (row["style"] in SCRIPTED_STYLES if split == "scripted" else row["style"] == "pretrained"))])
            for policy in POLICIES} for split in ("all","scripted","strong_pretrained")} for mode in modes}
        summary = {**manifest, "status":"complete", "actual_episodes":len(results), "all_completed":True,
            "unexpected_truncations":0, "wall_s":time.perf_counter()-started, "groups":groups,
            "gates":gates, "implementation_sha256_after":hashes_after, "sources_unchanged":True,
            "automatic_training_launched":False}
        write_json(out/"summary.json", summary)
        write_json(out/"development_gate.json", {"gates":gates, "automatic_training_launched":False})
        (out/"development_gate.md").write_text(gate_markdown(gates))
        write_json(out/"status.json", {"status":"complete", "completed_episodes":len(results),
            "expected_episodes":expected, "gate_passed_by_mode":{mode:gate["passed"] for mode,gate in gates.items()},
            "sources_unchanged":True, "automatic_training_launched":False})
        print(json.dumps({"status":"complete", "episodes":len(results), "gates":gates}), flush=True)
        return summary
    except BaseException as error:
        write_json(out/"status.json", {"status":"error", "completed_episodes":len(results),
            "expected_episodes":expected,"error":repr(error), "no_draw_invented":True})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[62011])
    parser.add_argument("--styles", nargs="+", choices=ALLOWED_STYLES, default=list(ALLOWED_STYLES))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--workers", type=int, choices=(1,2), default=2)
    parser.add_argument("--max-decisions", type=int, default=400)
    args = parser.parse_args(argv)
    return run_probe(args.out, seeds=args.seeds, styles=args.styles, modes=args.modes,
                     workers=args.workers, max_decisions=args.max_decisions)


if __name__ == "__main__":
    main()
