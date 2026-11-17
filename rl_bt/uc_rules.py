"""Individually testable UC V1.0.0 timers; not a complete referee emulator."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rule_facts():
    return {f["id"]: f for f in json.loads((ROOT / "data/rule_facts.json").read_text())["facts"]}


@dataclass
class CombatClock:
    quiet_period_s: float = 6.0
    last_activity_s: float | None = None
    previously_alive: bool = True
    last_update_s: float | None = None

    @classmethod
    def from_facts(cls):
        return cls(quiet_period_s=rule_facts()["out_of_combat_quiet_period"]["value"])

    def update(self, now_s, *, fired=False, damage_taken=False, alive=True):
        if self.last_update_s is not None and now_s < self.last_update_s:
            raise ValueError("Combat timestamps must be monotonic")
        # Six quiet seconds must be spent alive; death time cannot satisfy it.
        if fired or damage_taken or not alive or not self.previously_alive:
            self.last_activity_s = now_s
        self.previously_alive = alive
        self.last_update_s = now_s
        return alive and (self.last_activity_s is None or
                          now_s - self.last_activity_s >= self.quiet_period_s - 1e-9)


@dataclass
class StanceClock:
    """Base 0/1/2 = movement/attack/defense; enhanced 3/4/5 correspondingly.

    This module accumulates budgets and exposes constraints. Exhaustion does
    not invent an automatic stance transition; the caller must handle the
    authoritative referee state. Cooldown treatment of automatic transitions
    must be verified before a full simulator is certified.
    """
    cooldown_s: float = 5.0
    degradation_s: float = 180.0
    enhanced_budget_s: float = 15.0
    current: int = 0
    last_switch_s: float = float("-inf")
    cumulative_s: list = field(default_factory=lambda: [0., 0., 0.])
    enhanced_used_s: list = field(default_factory=lambda: [0., 0., 0.])

    @classmethod
    def from_facts(cls):
        facts = rule_facts()
        return cls(cooldown_s=facts["stance_switch_cooldown"]["value"],
                   degradation_s=facts["stance_degradation_cumulative_time"]["value"],
                   enhanced_budget_s=facts["enhanced_stance_budget_each"]["value"])

    def advance(self, dt_s):
        if dt_s < 0:
            raise ValueError("Time cannot run backwards")
        base = self.current % 3
        if self.current >= 3 and dt_s > self.remaining_enhanced_s(base) + 1e-9:
            raise ValueError("Advance must split at the enhanced budget boundary")
        self.cumulative_s[base] += dt_s
        if self.current >= 3:
            self.enhanced_used_s[base] += dt_s

    def remaining_enhanced_s(self, base):
        return max(0., self.enhanced_budget_s - self.enhanced_used_s[base])

    def mask(self, now_s):
        result = [False] * 6
        result[self.current] = True  # continuing a stance is not a switch
        if now_s - self.last_switch_s >= self.cooldown_s - 1e-9:
            for i in range(6):
                result[i] = i < 3 or self.remaining_enhanced_s(i % 3) > 1e-9
        # Continuing an exhausted enhancement is not a new legal intention.
        if self.current >= 3 and self.remaining_enhanced_s(self.current % 3) <= 1e-9:
            result[self.current] = False
        return result

    def switch(self, desired, now_s):
        if not 0 <= desired < 6 or not self.mask(now_s)[desired]:
            return False
        if desired != self.current:
            self.current = desired
            self.last_switch_s = now_s
        return True

    def degraded(self, base):
        return self.cumulative_s[base] > self.degradation_s


class TunnelSequence:
    """Card-order evidence only; physical crossing and buff ACK stay separate."""
    def __init__(self, tunnel_id, window_s=3.):
        self.tunnel_id = tunnel_id
        self.window_s = window_s
        self.cards = []

    @classmethod
    def from_facts(cls, tunnel_id):
        return cls(tunnel_id, rule_facts()["terrain_trigger_window_road_tunnel"]["value"])

    def observe(self, tunnel_id, card, now_s):
        if self.cards and now_s < self.cards[-1][1]:
            raise ValueError("Card timestamps must be monotonic")
        if tunnel_id != self.tunnel_id:
            self.cards = []
            return None
        if self.cards and now_s - self.cards[0][1] > self.window_s + 1e-9:
            self.cards = []
        if self.cards and self.cards[-1][0] == card:
            return None  # duplicate card detections do not restart the window
        self.cards.append((card, now_s))
        order = [x[0] for x in self.cards]
        if order in (["end_a", "inside", "end_b"], ["end_b", "inside", "end_a"]):
            self.cards = []
            return "underpass_sequence_completed_not_buff_ack"
        if order in (["end_a", "end_b"], ["end_b", "end_a"]):
            self.cards = []
            return "overpass_sequence_completed_not_buff_ack"
        valid = (["end_a"], ["end_b"], ["end_a", "inside"], ["end_b", "inside"])
        if order not in valid:
            self.cards = [(card, now_s)] if card in ("end_a", "end_b") else []
        return None
