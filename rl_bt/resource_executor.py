"""Offline UC remote-heal transaction/effect adapter; no live referee I/O.

An event loop must advance to purchase effect deadlines and supply authoritative
life counters. Late or incomplete real telemetry requires referee reconciliation,
not reconstructing past HP from these helpers. UL is deliberately unsupported.
"""
from dataclasses import dataclass
import math

from rl_bt.healing import HealingEligibility, uc_remote_healing_effect


@dataclass(frozen=True)
class HealSettlement:
    status: str
    hp_delta: float = 0.
    raw_heal_hp: float = 0.


class UCRemoteHealExecutor:
    """Bind request IDs, initial lifetime, budget ACK and exactly-once effect.

    ``life_seq`` increments whenever the robot dies, even if it revives before
    the next observation. Team wallet synchronization stays outside this offline
    reference. The caller applies a returned positive HP delta only once.
    """
    def __init__(self, ledger):
        self.ledger = ledger
        self.lifetimes = {}

    @staticmethod
    def _validate_clock(now_s, life_seq=None):
        if isinstance(now_s, bool) or not math.isfinite(now_s) or now_s < 0:
            raise ValueError("Finite nonnegative event time required")
        if life_seq is not None and (type(life_seq) is not int or life_seq < 0):
            raise ValueError("Explicit nonnegative death/lifetime counter required")

    def reserve(self, request, *, now_s, life_seq, eligibility: HealingEligibility):
        self._validate_clock(now_s, life_seq)
        if request.kind != "remote_heal":
            return "unsupported_purchase"
        if not eligibility.can_heal:
            return "rule_ineligible"
        with self.ledger.lock:
            reason = self.ledger.reserve(request, now_s=now_s, alive=eligibility.alive,
                                         out_of_combat=eligibility.out_of_combat)
            if reason is None:
                self.lifetimes[request.request_id] = life_seq
            return reason

    def confirm(self, request_id, *, now_s):
        self._validate_clock(now_s)
        with self.ledger.lock:
            if request_id not in self.lifetimes:
                raise ValueError("Remote heal was not reserved by this executor")
            request = self.ledger.requests[request_id]["request"]
            if now_s < request.issued_at_s:
                raise ValueError("Confirmation precedes request")
            return self.ledger.accept(request_id, now_s=now_s)

    def settle(self, request_id, *, now_s, life_seq, hp, max_hp, own_base_hp_lost,
               eligibility: HealingEligibility):
        self._validate_clock(now_s, life_seq)
        with self.ledger.lock:
            if request_id not in self.lifetimes:
                raise ValueError("Unknown remote heal request")
            if life_seq < self.lifetimes[request_id]:
                raise ValueError("Lifetime counter went backwards")
            transaction = self.ledger.requests[request_id]
            if transaction["status"] in self.ledger.TERMINAL:
                return HealSettlement("already_finalized")
            if transaction["status"] != "accepted_pending_effect":
                return HealSettlement("awaiting_authoritative_acceptance")
            effect = uc_remote_healing_effect(now_s=now_s,
                confirmed_at_s=transaction["accepted_at_s"], hp=hp, max_hp=max_hp,
                own_base_hp_lost=own_base_hp_lost, eligibility=eligibility,
                died_during_delay=life_seq != self.lifetimes[request_id])
            if effect.status == "invalid_death_no_refund":
                self.ledger.effect(request_id, now_s=now_s, succeeded=False)
                return HealSettlement(effect.status)
            if effect.status != "due_effect_not_transaction_ack":
                return HealSettlement(effect.status)
            if not self.ledger.effect(request_id, now_s=now_s, succeeded=True):
                raise RuntimeError("Effect timing/state diverged from accepted transaction")
            return HealSettlement("effect_applied_once", effect.hp_after - hp, effect.raw_heal_hp)
