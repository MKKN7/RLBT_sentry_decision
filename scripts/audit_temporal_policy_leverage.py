#!/usr/bin/env python3
"""Posthoc fixed-controller leverage audit on already-seen alphaRM cases.

No trained model, optimizer, official UL/UC rules, or live Nav2 is involved.
The purpose is to check how discriminating the earlier all-win cases are.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rl_bt.alpharm_task_env import AlphaRuntime, AlphaTaskEnv, HELD_OUT_STYLES
from scripts.evaluate_temporal_phase import make_cases, summarize_policy

POLICIES = ("continue_only", "fixed_center")
SOURCES = ("scripts/audit_temporal_policy_leverage.py", "scripts/evaluate_temporal_phase.py",
           "rl_bt/alpharm_task_env.py", "rl_bt/geometry.py", "rl_bt/contracts.py",
           "rl_bt/rewards.py", "rl_bt/reward_components.py", "scripts/probe_alpharm_onnx.py")


def fixed_control(observation, policy):
    """Read only the cached public legal mask; no truth or model state."""
    if policy not in POLICIES:
        raise ValueError("Unknown fixed controller")
    mask = np.asarray(observation["mask"])
    if mask.shape != (16,) or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("A boolean16 mask with a legal fallback is required")
    if policy == "fixed_center" and mask[1]:
        return 1
    if mask[0]:
        return 0
    return int(np.flatnonzero(mask)[0])


def run_control_trial(runtime, policy, case, *, max_decisions=400, env_factory=AlphaTaskEnv):
    if max_decisions < 1:
        raise ValueError("Positive decision bound required")
    env = env_factory(runtime, case["seed"], randomization=True,
        opponent_style=case["style"], side=case["side"], reward_profile="upstream")
    observation = env.cached
    choices, executed, reasons = np.zeros(16, int), np.zeros(16, int), Counter()
    elapsed = alive_s = center_s = home_s = hp_integral = injected_missing = 0.
    final_info = None
    for _ in range(max_decisions):
        action = fixed_control(observation, policy)
        observation, _, done, info = env.step(action)
        choices[action] += 1
        executed[int(info["executed_candidate"])] += 1
        if info["override_reason"]:
            reasons[str(info["override_reason"])] += 1
        duration = float(info["duration_s"])
        if not np.isfinite(duration) or duration <= 0:
            raise RuntimeError("Actual transition duration must be finite and positive")
        # Diagnostic-only current simulator observations after execution. They
        # are never passed back to fixed_control, which reads only cached mask.
        own = env.obs[env.learner]
        alive = float(own[11] > .5)
        alive_s += duration * alive
        center_s += duration * alive * float(own[21] > .5)
        home_s += duration * alive * float(own[20] > .5)
        hp_integral += duration * float(own[7])
        injected_missing += duration * float(np.asarray(env.dropout).mean())
        elapsed += duration
        if done:
            final_info = info
            break
    if final_info is None:
        raise RuntimeError(f"Unexpected cutoff for {policy} {case}; no invented draw recorded")
    if "episode" not in final_info:
        raise RuntimeError("True terminal transition is missing its episode result")
    state = env.state_dict()
    own_damage = float(state["cum_damage"][env.learner].sum())
    team = case["side"]
    team_damage = float(state["cum_damage"][team*3:team*3+3].sum())
    return {**final_info["episode"], "policy": policy, **case,
        "completed": True, "truncated": False, "actual_duration_s": elapsed,
        "held_out_style": case["style"] in HELD_OUT_STYLES,
        "exploratory_posthoc": True, "independent_new_test": False,
        "checkpoint_selection": False, "SGD_steps": 0,
        "real_nav2_in_loop": False, "official_UL_UC_training": False,
        "diagnostics": {"candidate_counts": choices.tolist(),
            "executed_candidate_counts": executed.tolist(), "override_reasons": dict(reasons),
            "sampled_alive_fraction": alive_s/elapsed,
            "sampled_center_fraction": center_s/elapsed,
            "sampled_home_fraction": home_s/elapsed,
            "sampled_mean_own_hp_fraction": hp_integral/elapsed,
            "damage_dealt": own_damage, "team_damage_dealt": team_damage,
            "damage_share_of_team": own_damage/max(team_damage, 1e-9),
            "injected_detection_missing_fraction": injected_missing/elapsed,
            "overrides_per_decision": sum(reasons.values())/int(choices.sum()),
            "ammunition": None,
            "sampling_note": "Post-decision diagnostic samples; truth is not a controller input."}}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def audit(out, *, workers=2, max_decisions=400):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = make_cases([51001, 51002], ["capture", "pursuit"], [0, 1])
    runtime = AlphaRuntime()
    manifest_path = ROOT/"data/references/alpharm_jax_review_manifest.json"
    manifest = {"scenario_id": "alpharm_temporal_policy_leverage_audit_v1",
        "exploratory_posthoc": True, "independent_new_test": False,
        "case_note": "These exact cases were already seen in the frozen temporal evaluation.",
        "purpose": "Fixed-controller outcome sensitivity; no tuning or checkpoint selection.",
        "policies": list(POLICIES), "cases": cases, "expected_episodes": 16,
        "controller_input": "cached legal mask only", "learned_weights_loaded": False,
        "optimizers_created": 0, "SGD_steps": 0, "checkpoint_selection": False,
        "real_nav2_in_loop": False, "official_UL_UC_training": False,
        "same_shared_executor": True, "randomization": True,
        "reward_profile": "upstream", "workers": workers,
        "max_decisions_per_episode": max_decisions, "upstream_commit": runtime.source_commit,
        "upstream_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "implementation_sha256": {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
        "control_definitions": {
            "continue_only": "Action0 when legal, otherwise first legal. Existing executor intent continues/holds.",
            "fixed_center": "Action1 when legal, otherwise action0 when legal, otherwise first legal."},
        "limits": ["Continue does not disable the existing combat executor or other robots.",
            "Simplified alphaRM3v3, not official UL/UC or live Point-LIO/Nav2/MPPI.",
            "Posthoc reused cases cannot establish independent generalization.",
            "No trained checkpoint is selected or altered by this control audit."]}
    write_json(out/"manifest.json", manifest)
    for relative in SOURCES:
        target = out/"source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
    trials = [(policy, case) for case in cases for policy in POLICIES]
    started, results = time.perf_counter(), []
    write_json(out/"status.json", {"status":"running", "completed_episodes":0, "expected_episodes":16})
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool, (out/"episodes.jsonl").open("w") as handle:
            for result in pool.map(lambda item:run_control_trial(runtime, *item, max_decisions=max_decisions), trials):
                results.append(result)
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+"\n")
                handle.flush()
                status = {"status":"running", "completed_episodes":len(results),
                    "expected_episodes":16, "last_policy":result["policy"],
                    "last_case":{key:result[key] for key in ("seed","style","side")},
                    "last_win":result["win"], "wall_s":time.perf_counter()-started}
                write_json(out/"status.json", status)
                print(json.dumps(status), flush=True)
        summary = {"status":"complete", "episodes":len(results),
            "all_completed":all(row["completed"] for row in results),
            "SGD_steps":0, "learned_weights_loaded":False, "exploratory_posthoc":True,
            "independent_new_test":False, "checkpoint_selection":False,
            "real_nav2_in_loop":False, "official_UL_UC_training":False,
            "wall_s":time.perf_counter()-started,
            "policies":{policy:summarize_policy([row for row in results if row["policy"]==policy])
                        for policy in POLICIES}}
        write_json(out/"summary.json", summary)
        write_json(out/"status.json", summary)
        print(json.dumps(summary), flush=True)
        return summary
    except BaseException as error:
        write_json(out/"status.json", {"status":"error", "completed_episodes":len(results),
            "error_type":type(error).__name__, "error":str(error), "no_draw_invented":True})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1,2), default=2)
    parser.add_argument("--max-decisions", type=int, default=400)
    args = parser.parse_args(argv)
    if args.max_decisions < 1:
        parser.error("Positive decision bound required")
    return audit(args.out, workers=args.workers, max_decisions=args.max_decisions)


if __name__ == "__main__":
    main()
