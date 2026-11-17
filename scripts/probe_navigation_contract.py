#!/usr/bin/env python3
"""Deterministic transport replay; does not start or validate a ROS navigator."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_bt.contracts import TaskIntent
from rl_bt.navigation_contract import CandidateSnapshot, NavigationCandidate, NavigationExecutor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence: {args.out}")
    health = dict(alive=True, pose_valid=True, frame_ready=True)
    executor = NavigationExecutor(session_id="synthetic-replay", profile_id="synthetic_contract",
        map_id="synthetic_map", map_hash="synthetic_not_official_geometry", frame_id="map",
        acceptance_timeout_s=.4, feedback_timeout_s=.5, cancel_timeout_s=.3,
        execution_timeout_s=5.)
    snapshot = CandidateSnapshot("synthetic_contract", "synthetic_map",
        "synthetic_not_official_geometry", "map", 1, 0., 10., (
            NavigationCandidate(1, "navigate", "synthetic_control", (1., 2., 0., 0.),
                                "synthetic_ground", "ground"),
            NavigationCandidate(2, "navigate", "synthetic_supply", (3., 4., 0., 0.),
                                "synthetic_ground", "underpass")))
    events = []

    def record(label, now, result):
        events.append({"event": label, "delivered_at_s": now, "decision": asdict(result),
                       "executor_state": executor.state,
                       "active_task_id": None if executor.active is None else executor.active.task_id,
                       "pending_candidate_id": None if executor.pending is None else
                           executor.pending.candidate.candidate_id})
        return result

    first = record("submit_first", 0., executor.propose(TaskIntent(1, 1, 1, 0.),
                   snapshot, now_s=0., **health))
    first_id = first.commands[0].task_id
    record("accepted_first", .1, executor.feedback(task_id=first_id, status="accepted",
           sampled_at_s=.1, now_s=.1))
    record("request_replacement", .2, executor.propose(TaskIntent(2, 2, 1, .2),
           snapshot, now_s=.2, **health))
    record("cancel_request_accepted_not_terminal", .21, executor.feedback(task_id=first_id,
           status="cancel_accepted", sampled_at_s=.21, now_s=.21))
    blocked = record("replacement_still_blocked", .22,
                     executor.dispatch_pending(snapshot, now_s=.22, **health))
    assert blocked.status == "waiting" and not blocked.commands
    record("old_action_terminal", .25, executor.feedback(task_id=first_id, status="canceled",
           sampled_at_s=.25, now_s=.25))
    second = record("replacement_submitted", .26,
                    executor.dispatch_pending(snapshot, now_s=.26, **health))
    second_id = second.commands[0].task_id
    assert second_id != first_id
    late = record("late_old_success_ignored", .27, executor.feedback(task_id=first_id,
           status="succeeded", sampled_at_s=.27, now_s=.27))
    assert late.status == "ignored" and executor.active.task_id == second_id
    timeout = record("acceptance_timeout", .67, executor.watchdog(now_s=.67, **health))
    assert [command.kind for command in timeout.commands] == ["cancel_goal", "require_safety_stop"]
    record("late_acceptance_does_not_resume", .68, executor.feedback(task_id=second_id,
           status="accepted", sampled_at_s=.68, now_s=.68))
    assert executor.state == "uncertain"
    record("terminal_after_timeout", .7, executor.feedback(task_id=second_id, status="canceled",
           sampled_at_s=.7, now_s=.7))
    assert executor.active is None and executor.state == "idle"
    sources = ["rl_bt/navigation_contract.py", "rl_bt/contracts.py",
               "scripts/probe_navigation_contract.py"]
    report = {
        "artifact_type": "offline_navigation_contract_replay",
        "official_map_or_navigation_validation": False,
        "timeout_values": "synthetic_engineering_inputs_not_measured_hardware_values",
        "snapshot": asdict(snapshot),
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                          for path in sources},
        "checks": {"cancel_acceptance_does_not_release_old_action": True,
                   "terminal_result_required_before_replacement": True,
                   "stale_result_does_not_affect_new_action": True,
                   "ack_timeout_retains_uncertain_action_and_requests_stop": True,
                   "late_acceptance_does_not_resume_canceled_action": True},
        "events": events,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(args.out.resolve()), "checks": report["checks"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
