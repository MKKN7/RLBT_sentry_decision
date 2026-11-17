"""Offline BT/Navigation action contract, independent of ROS and motor control.

This state machine emits requests for a future Nav2 adapter. It does not send
``cmd_vel``, declare a map traversable, or run Point-LIO/Nav2/MPPI. All timeout
values are supplied by the caller and are engineering settings, not rule facts.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from rl_bt.contracts import IntentGate, TaskIntent


def _clock(value, name="time"):
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Finite nonnegative {name} required")


def _positive(value, name):
    _clock(value, name)
    if value == 0:
        raise ValueError(f"Positive {name} required")


@dataclass(frozen=True)
class NavigationCandidate:
    candidate_id: int
    kind: str  # navigate, continue, hold
    task_type: str
    goal_xyzyaw: tuple[float, float, float, float] | None = None
    support_layer_id: str | None = None
    route_type: str | None = None
    legal: bool = True
    reachable: bool = True

    def __post_init__(self):
        if type(self.candidate_id) is not int or self.candidate_id < 0:
            raise ValueError("Nonnegative integer candidate ID required")
        if self.kind not in {"navigate", "continue", "hold"} or not self.task_type:
            raise ValueError("Explicit candidate kind and task type required")
        if type(self.legal) is not bool or type(self.reachable) is not bool:
            raise ValueError("Explicit Boolean legal/reachable state required")
        if self.kind == "navigate":
            if self.goal_xyzyaw is None or len(self.goal_xyzyaw) != 4:
                raise ValueError("Navigation goal requires x/y/z/yaw")
            if not all(not isinstance(x, bool) and math.isfinite(x)
                       for x in self.goal_xyzyaw):
                raise ValueError("Finite goal pose required")
            if not self.support_layer_id or not self.route_type:
                raise ValueError("Explicit support layer and route type required")
            object.__setattr__(self, "goal_xyzyaw", tuple(self.goal_xyzyaw))
        elif self.goal_xyzyaw is not None:
            raise ValueError("Continue/hold must not contain a navigation goal")


@dataclass(frozen=True)
class CandidateSnapshot:
    profile_id: str
    map_id: str
    map_hash: str
    frame_id: str
    version: int
    generated_at_s: float
    valid_for_s: float
    candidates: tuple[NavigationCandidate, ...]

    def __post_init__(self):
        if not all((self.profile_id, self.map_id, self.map_hash, self.frame_id)):
            raise ValueError("Profile, map identity/hash and frame are required")
        if type(self.version) is not int or self.version < 0:
            raise ValueError("Nonnegative integer candidate version required")
        _clock(self.generated_at_s)
        _positive(self.valid_for_s, "candidate validity interval")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("Candidate IDs must be unique within a snapshot")


@dataclass(frozen=True)
class NavigationCommand:
    kind: str  # send_goal, cancel_goal, require_safety_stop
    task_id: str
    reason: str
    candidate: NavigationCandidate | None = None
    frame_id: str | None = None
    candidate_version: int | None = None


@dataclass(frozen=True)
class NavigationDecision:
    status: str
    reason: str
    commands: tuple[NavigationCommand, ...] = ()


@dataclass(frozen=True)
class _Pending:
    intent: TaskIntent
    candidate: NavigationCandidate


@dataclass
class _Task:
    task_id: str
    candidate: NavigationCandidate
    candidate_version: int
    dispatched_at_s: float
    last_feedback_at_s: float
    last_sampled_at_s: float
    cancel_started_at_s: float | None = None
    stop_requested: bool = False


class NavigationExecutor:
    """Serialize BT-authorized goals and isolate stale asynchronous messages.

    An intent TTL limits admission only. Accepted navigation may run longer.
    A replacement waits for the previous action's *terminal* result. Nav2's
    acceptance of a cancel request alone does not prove that motion has stopped.
    ``session_id`` must be unique across adapter restarts so old transport events
    cannot match a newly created task. Methods are intended for one event loop.
    """
    TERMINAL = {"rejected", "succeeded", "failed", "canceled"}

    def __init__(self, *, session_id, profile_id, map_id, map_hash, frame_id,
                 acceptance_timeout_s, feedback_timeout_s, cancel_timeout_s,
                 execution_timeout_s):
        if not all((session_id, profile_id, map_id, map_hash, frame_id)):
            raise ValueError("Session, profile, map identity/hash and frame required")
        for name, value in (("acceptance timeout", acceptance_timeout_s),
                            ("feedback timeout", feedback_timeout_s),
                            ("cancel timeout", cancel_timeout_s),
                            ("execution timeout", execution_timeout_s)):
            _positive(value, name)
        self.session_id = session_id
        self.identity = (profile_id, map_id, map_hash, frame_id)
        self.acceptance_timeout_s = acceptance_timeout_s
        self.feedback_timeout_s = feedback_timeout_s
        self.cancel_timeout_s = cancel_timeout_s
        self.execution_timeout_s = execution_timeout_s
        self.gate = IntentGate()
        self.active: _Task | None = None
        self.pending: _Pending | None = None
        self.state = "idle"
        self._counter = 0
        self._now = 0.

    def _advance(self, now_s):
        _clock(now_s)
        if now_s < self._now:
            raise ValueError("Event delivery time must be monotonic")
        self._now = now_s

    @staticmethod
    def _validate_intent(intent):
        if any(type(x) is not int or x < 0 for x in
               (intent.candidate_id, intent.sequence, intent.candidate_version)):
            raise ValueError("Integer nonnegative intent IDs/version required")
        _clock(intent.issued_at_s)
        _positive(intent.ttl_s, "intent TTL")

    def _snapshot_reason(self, snapshot, now_s):
        if (snapshot.profile_id, snapshot.map_id, snapshot.map_hash,
                snapshot.frame_id) != self.identity:
            return "profile_map_or_frame_mismatch"
        if not snapshot.generated_at_s <= now_s <= snapshot.generated_at_s + snapshot.valid_for_s:
            return "stale_or_future_candidate_snapshot"
        return None

    @staticmethod
    def _health_reason(*, alive, pose_valid, frame_ready):
        if any(type(value) is not bool for value in (alive, pose_valid, frame_ready)):
            raise ValueError("Explicit Boolean execution health required")
        if not alive:
            return "robot_dead"
        if not pose_valid:
            return "localization_invalid"
        if not frame_ready:
            return "frame_unavailable"
        return None

    def propose(self, intent: TaskIntent, snapshot: CandidateSnapshot, *, now_s,
                alive, pose_valid, frame_ready):
        self._advance(now_s)
        self._validate_intent(intent)
        health_reason = self._health_reason(alive=alive, pose_valid=pose_valid,
                                             frame_ready=frame_ready)
        # Consume every newer intent, including a rejected one, preventing replay.
        by_id = {candidate.candidate_id: candidate for candidate in snapshot.candidates}
        size = max(by_id, default=-1) + 1
        mask = [False] * size
        for index, candidate in by_id.items():
            mask[index] = candidate.legal and candidate.reachable
        reason = self.gate.validate(intent, version=snapshot.version, now=now_s,
                                    mask=mask, alive=alive)
        reason = reason or self._snapshot_reason(snapshot, now_s)
        if health_reason:
            safety = self.watchdog(now_s=now_s, alive=alive, pose_valid=pose_valid,
                                   frame_ready=frame_ready)
            return NavigationDecision("rejected", health_reason, safety.commands)
        if reason:
            return NavigationDecision("rejected", reason)
        candidate = by_id[intent.candidate_id]
        if candidate.kind == "hold":
            return self.stop(now_s=now_s, reason="BT_hold_selected")
        if candidate.kind == "continue":
            if self.active is None or self.state in {"canceling", "uncertain"}:
                return NavigationDecision("rejected", "no_continuable_task")
            self.pending = None
            return NavigationDecision("continued", "existing_task_kept")
        self.pending = _Pending(intent, candidate)
        if self.active is not None:
            if candidate == self.active.candidate and self.state not in {"canceling", "uncertain"}:
                self.pending = None
                return NavigationDecision("continued", "same_goal_no_resubmission")
            return self._cancel(now_s, "replace_goal_wait_for_terminal_result")
        return self.dispatch_pending(snapshot, now_s=now_s, alive=alive,
                                     pose_valid=pose_valid, frame_ready=frame_ready)

    def dispatch_pending(self, snapshot: CandidateSnapshot, *, now_s, alive,
                         pose_valid, frame_ready):
        """Recheck current candidates after cancel termination, before any send."""
        self._advance(now_s)
        health_reason = self._health_reason(alive=alive, pose_valid=pose_valid,
                                             frame_ready=frame_ready)
        if health_reason:
            safety = self.watchdog(now_s=now_s, alive=alive, pose_valid=pose_valid,
                                   frame_ready=frame_ready)
            return NavigationDecision("rejected", health_reason, safety.commands)
        if self.active is not None:
            return NavigationDecision("waiting", "previous_action_not_terminal")
        if self.pending is None:
            return NavigationDecision("idle", "no_pending_intent")
        pending = self.pending
        reason = self._snapshot_reason(snapshot, now_s)
        reason = reason or self._health_reason(alive=alive, pose_valid=pose_valid,
                                              frame_ready=frame_ready)
        if reason is None:
            if not pending.intent.issued_at_s <= now_s <= pending.intent.issued_at_s + pending.intent.ttl_s:
                reason = "pending_intent_expired"
            elif pending.intent.candidate_version != snapshot.version:
                reason = "pending_candidate_version_mismatch"
            else:
                current = next((candidate for candidate in snapshot.candidates
                                if candidate.candidate_id == pending.intent.candidate_id), None)
                if current != pending.candidate or not current.legal or not current.reachable:
                    reason = "pending_candidate_changed_or_illegal"
        self.pending = None
        if reason:
            return NavigationDecision("rejected", reason)
        self._counter += 1
        task_id = f"{self.session_id}:{self._counter}:{pending.intent.sequence}"
        self.active = _Task(task_id, pending.candidate, snapshot.version, now_s, now_s, now_s)
        self.state = "awaiting_acceptance"
        command = NavigationCommand("send_goal", task_id, "validated_BT_goal",
                                    pending.candidate, snapshot.frame_id, snapshot.version)
        return NavigationDecision("submitted", "awaiting_navigation_acceptance", (command,))

    def _cancel(self, now_s, reason):
        if self.active is None:
            self.state = "idle"
            return NavigationDecision("idle", reason)
        if self.state in {"canceling", "uncertain"}:
            return NavigationDecision("waiting", "previous_cancel_not_terminal")
        self.active.cancel_started_at_s = now_s
        self.state = "canceling"
        command = NavigationCommand("cancel_goal", self.active.task_id, reason)
        return NavigationDecision("canceling", reason, (command,))

    def stop(self, *, now_s, reason):
        self._advance(now_s)
        self.pending = None
        return self._cancel(now_s, reason)

    def feedback(self, *, task_id, status, sampled_at_s, now_s):
        self._advance(now_s)
        _clock(sampled_at_s)
        if status not in self.TERMINAL | {"accepted", "progress", "cancel_accepted"}:
            raise ValueError("Unknown navigation feedback status")
        if self.active is None or task_id != self.active.task_id:
            return NavigationDecision("ignored", "stale_or_unknown_task")
        task = self.active
        if not task.last_sampled_at_s <= sampled_at_s <= now_s:
            return NavigationDecision("ignored", "out_of_order_or_future_feedback")
        if status == "cancel_accepted" and self.state not in {"canceling", "uncertain"}:
            return NavigationDecision("ignored", "unsolicited_cancel_acceptance")
        if status == "accepted" and self.state == "active":
            return NavigationDecision("ignored", "duplicate_acceptance")
        task.last_sampled_at_s = sampled_at_s
        task.last_feedback_at_s = now_s
        if status in self.TERMINAL:
            self.active = None
            self.state = "idle"
            return NavigationDecision("terminal", status)
        if status in {"accepted", "progress"} and self.state == "awaiting_acceptance":
            self.state = "active"
        # Neither progress nor cancel acceptance releases a task being canceled.
        return NavigationDecision("updated", status)

    def watchdog(self, *, now_s, alive, pose_valid, frame_ready):
        """Request cancellation/stop; never declare an unconfirmed action gone."""
        self._advance(now_s)
        health_reason = self._health_reason(alive=alive, pose_valid=pose_valid,
                                             frame_ready=frame_ready)
        if health_reason:
            self.pending = None
        if self.active is None:
            return NavigationDecision("idle", health_reason or "no_active_navigation")
        task = self.active
        if self.state in {"canceling", "uncertain"}:
            if now_s >= task.cancel_started_at_s + self.cancel_timeout_s:
                return self._require_stop("cancel_terminal_result_timeout")
            if health_reason:
                return self._require_stop(health_reason)
            return NavigationDecision("waiting", "awaiting_cancel_terminal_result")
        reason = health_reason
        if reason is None and self.state == "awaiting_acceptance":
            if now_s >= task.dispatched_at_s + self.acceptance_timeout_s:
                reason = "navigation_acceptance_timeout"
        if reason is None and now_s >= task.last_sampled_at_s + self.feedback_timeout_s:
            reason = "navigation_feedback_timeout"
        if reason is None and now_s >= task.dispatched_at_s + self.execution_timeout_s:
            reason = "navigation_execution_timeout"
        if reason:
            self.pending = None
            decision = self._cancel(now_s, reason)
            stop = self._require_stop(reason)
            return NavigationDecision("uncertain", reason, decision.commands + stop.commands)
        return NavigationDecision("running", "navigation_monitored")

    def _require_stop(self, reason):
        self.state = "uncertain"
        if self.active.stop_requested:
            return NavigationDecision("uncertain", reason)
        self.active.stop_requested = True
        command = NavigationCommand("require_safety_stop", self.active.task_id, reason)
        return NavigationDecision("uncertain", reason, (command,))
