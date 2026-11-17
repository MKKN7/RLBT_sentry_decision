"""Versioned intention validation shared by the prototype BT-side executor."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskIntent:
    candidate_id: int
    sequence: int
    candidate_version: int
    issued_at_s: float
    ttl_s: float = 2.0


class IntentGate:
    def __init__(self):
        self.last_sequence = -1

    def validate(self, intent, *, version, now, mask, alive=True):
        if intent.sequence <= self.last_sequence:
            return "stale_sequence"
        # Even a rejected newer request cannot later be resurrected.
        self.last_sequence = intent.sequence
        if intent.candidate_version != version:
            return "candidate_version_mismatch"
        if now < intent.issued_at_s or now - intent.issued_at_s > intent.ttl_s:
            return "expired_or_future_intent"
        if not alive:
            return "robot_dead"
        if not 0 <= intent.candidate_id < len(mask) or not mask[intent.candidate_id]:
            return "unreachable_or_illegal"
        return None
