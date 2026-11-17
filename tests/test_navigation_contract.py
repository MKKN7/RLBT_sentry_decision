from dataclasses import replace

import pytest

from rl_bt.contracts import TaskIntent
from rl_bt.navigation_contract import CandidateSnapshot, NavigationCandidate, NavigationExecutor


HEALTH = dict(alive=True, pose_valid=True, frame_ready=True)


def setup():
    executor = NavigationExecutor(session_id="offline-test-session", profile_id="rmul_2027",
        map_id="ul-map", map_hash="reference-hash", frame_id="map",
        acceptance_timeout_s=.4, feedback_timeout_s=.5, cancel_timeout_s=.3,
        execution_timeout_s=5.)
    candidates = (
        NavigationCandidate(0, "continue", "continue_current_task"),
        NavigationCandidate(1, "navigate", "control", (1., 2., 0., 0.), "ground", "ground"),
        NavigationCandidate(2, "navigate", "supply", (3., 4., 0., 0.), "ground", "underpass"),
        NavigationCandidate(3, "hold", "BT_hold"),
    )
    snapshot = CandidateSnapshot("rmul_2027", "ul-map", "reference-hash", "map",
                                 7, 0., 10., candidates)
    return executor, snapshot


def propose(executor, snapshot, *, candidate=1, sequence=1, now=0., ttl=2.):
    intent = TaskIntent(candidate, sequence, snapshot.version, now, ttl)
    return executor.propose(intent, snapshot, now_s=now, **HEALTH)


def feedback(executor, *, status="accepted", now=.1, task_id=None, sampled=None):
    task_id = task_id or executor.active.task_id
    return executor.feedback(task_id=task_id, status=status, now_s=now,
                             sampled_at_s=now if sampled is None else sampled)


def test_goal_metadata_and_pose_only_and_same_goal_is_not_resubmitted():
    executor, snapshot = setup()
    decision = propose(executor, snapshot)
    command, = decision.commands
    assert command.kind == "send_goal" and command.frame_id == "map"
    assert command.candidate_version == 7
    assert command.candidate.goal_xyzyaw == (1., 2., 0., 0.)
    assert command.candidate.support_layer_id == "ground"
    assert not hasattr(command, "cmd_vel")
    task_id = command.task_id
    feedback(executor)
    duplicate = propose(executor, snapshot, sequence=2, now=.2)
    assert duplicate.status == "continued" and duplicate.commands == ()
    assert executor.active.task_id == task_id


def test_replacement_waits_for_terminal_cancel_not_just_cancel_request_ack():
    executor, snapshot = setup()
    old_id = propose(executor, snapshot).commands[0].task_id
    feedback(executor)
    decision = propose(executor, snapshot, candidate=2, sequence=2, now=.2)
    assert [command.kind for command in decision.commands] == ["cancel_goal"]
    feedback(executor, status="cancel_accepted", now=.21)
    assert executor.dispatch_pending(snapshot, now_s=.22, **HEALTH).status == "waiting"
    # Late acceptance/progress must not resurrect a canceling task.
    feedback(executor, status="accepted", now=.23)
    feedback(executor, status="progress", now=.24)
    assert executor.state == "canceling"
    feedback(executor, status="canceled", now=.25)
    new = executor.dispatch_pending(snapshot, now_s=.26, **HEALTH)
    assert new.status == "submitted" and new.commands[0].task_id != old_id
    assert new.commands[0].candidate.candidate_id == 2
    stale = feedback(executor, status="succeeded", task_id=old_id, now=.27)
    assert stale.reason == "stale_or_unknown_task"
    assert executor.active.task_id == new.commands[0].task_id


def test_replacement_is_revalidated_after_cancel_and_expired_intent_is_dropped():
    executor, snapshot = setup()
    propose(executor, snapshot)
    propose(executor, snapshot, candidate=2, sequence=2, now=.1, ttl=.1)
    feedback(executor, status="canceled", now=.21)
    result = executor.dispatch_pending(snapshot, now_s=.21, **HEALTH)
    assert result.reason == "pending_intent_expired" and not result.commands
    assert executor.active is None
    propose(executor, snapshot, sequence=3, now=.3)
    propose(executor, snapshot, candidate=2, sequence=4, now=.4)
    feedback(executor, status="canceled", now=.5)
    updated = replace(snapshot, version=8, generated_at_s=.5)
    result = executor.dispatch_pending(updated, now_s=.5, **HEALTH)
    assert result.reason == "pending_candidate_version_mismatch"


def test_cancel_timeout_blocks_replacement_until_authoritative_terminal_result():
    executor, snapshot = setup()
    propose(executor, snapshot)
    propose(executor, snapshot, candidate=2, sequence=2, now=.1)
    result = executor.watchdog(now_s=.41, **HEALTH)
    assert result.reason == "cancel_terminal_result_timeout"
    assert [command.kind for command in result.commands] == ["require_safety_stop"]
    assert executor.state == "uncertain" and executor.active is not None
    assert executor.dispatch_pending(snapshot, now_s=.42, **HEALTH).status == "waiting"
    assert not executor.watchdog(now_s=.43, **HEALTH).commands
    feedback(executor, status="canceled", now=.44)
    assert executor.dispatch_pending(snapshot, now_s=.45, **HEALTH).status == "submitted"


def test_acceptance_timeout_never_assumes_unaccepted_goal_is_absent():
    executor, snapshot = setup()
    propose(executor, snapshot)
    result = executor.watchdog(now_s=.4, **HEALTH)
    assert result.reason == "navigation_acceptance_timeout"
    assert [command.kind for command in result.commands] == ["cancel_goal", "require_safety_stop"]
    feedback(executor, status="accepted", now=.41)
    assert executor.state == "uncertain"


def test_intent_ttl_is_admission_deadline_not_accepted_task_duration():
    executor, snapshot = setup()
    propose(executor, snapshot, ttl=.2)
    feedback(executor)
    for now in (.3, .6, .9):
        feedback(executor, status="progress", now=now)
        assert executor.watchdog(now_s=now, **HEALTH).status == "running"
    result = executor.watchdog(now_s=1.4, **HEALTH)
    assert result.reason == "navigation_feedback_timeout"


def test_invalid_pose_or_death_cancels_and_clears_pending_goal():
    executor, snapshot = setup()
    propose(executor, snapshot)
    propose(executor, snapshot, candidate=2, sequence=2, now=.1)
    result = executor.watchdog(now_s=.11, alive=False, pose_valid=True, frame_ready=True)
    assert result.reason == "robot_dead" and executor.pending is None
    feedback(executor, status="canceled", now=.12)
    assert executor.dispatch_pending(snapshot, now_s=.13, **HEALTH).status == "idle"


def test_cross_profile_frame_staleness_masks_and_rejected_sequence_replay():
    executor, snapshot = setup()
    wrong = replace(snapshot, profile_id="rmuc_2027")
    assert propose(executor, wrong).reason == "profile_map_or_frame_mismatch"
    assert propose(executor, snapshot).reason == "stale_sequence"
    old = replace(snapshot, valid_for_s=.1)
    assert propose(executor, old, sequence=2, now=.2).reason == "stale_or_future_candidate_snapshot"
    blocked = replace(snapshot, candidates=tuple(
        replace(candidate, reachable=False) if candidate.candidate_id == 1 else candidate
        for candidate in snapshot.candidates))
    assert propose(executor, blocked, sequence=3, now=.2).reason == "unreachable_or_illegal"
    assert executor.active is None


def test_feedback_reordering_clock_reversal_and_unknown_values_fail_closed():
    executor, snapshot = setup()
    propose(executor, snapshot)
    feedback(executor, now=.2)
    assert feedback(executor, status="progress", now=.3, sampled=.1).status == "ignored"
    assert feedback(executor, status="progress", now=.4, sampled=.5).status == "ignored"
    with pytest.raises(ValueError):
        executor.watchdog(now_s=.2, **HEALTH)
    with pytest.raises(ValueError):
        propose(executor, snapshot, sequence=3, now=float("nan"))
    with pytest.raises(ValueError):
        NavigationCandidate(4, "navigate", "test", (float("nan"), 0., 0., 0.), "ground", "ground")
    with pytest.raises(ValueError):
        NavigationCandidate(4, "navigate", "test", (0., 0., 0., 0.), None, "ground")


def test_hold_cancels_navigation_and_continue_never_spawns_goal():
    executor, snapshot = setup()
    assert propose(executor, snapshot, candidate=0).reason == "no_continuable_task"
    propose(executor, snapshot, sequence=2, now=.1)
    assert propose(executor, snapshot, candidate=0, sequence=3, now=.2).status == "continued"
    result = propose(executor, snapshot, candidate=3, sequence=4, now=.3)
    assert [command.kind for command in result.commands] == ["cancel_goal"]
    assert executor.pending is None


def test_delayed_feedback_cannot_refresh_watchdog_using_its_receive_time():
    executor, snapshot = setup()
    propose(executor, snapshot)
    feedback(executor)
    feedback(executor, status="progress", now=.8, sampled=.2)
    result = executor.watchdog(now_s=.8, **HEALTH)
    assert result.reason == "navigation_feedback_timeout"


def test_execution_deadline_cancels_even_when_progress_keeps_arriving():
    executor, snapshot = setup()
    propose(executor, snapshot)
    feedback(executor)
    for tick in range(1, 10):
        feedback(executor, status="progress", now=tick / 2)
    feedback(executor, status="progress", now=5.)
    assert executor.watchdog(now_s=5., **HEALTH).reason == "navigation_execution_timeout"


def test_localization_failure_on_proposal_immediately_requests_stop_of_active_goal():
    executor, snapshot = setup()
    propose(executor, snapshot)
    intent = TaskIntent(2, 2, snapshot.version, .1)
    result = executor.propose(intent, snapshot, now_s=.1, alive=True,
                              pose_valid=False, frame_ready=True)
    assert result.reason == "localization_invalid"
    assert [command.kind for command in result.commands] == ["cancel_goal", "require_safety_stop"]
    assert executor.pending is None and executor.active is not None
