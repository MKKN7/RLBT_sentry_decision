"""UC V1.0.0 healing calculations, separate from location and wallet inference.

These helpers consume confirmed field-interaction/control states. A navigation
goal, being near home, a base buff, or safe cover alone grants no healing.
The caller owns referee settlement timing, activation persistence, remote-heal
transaction idempotency, and splitting intervals at eligibility changes.
This is not a full referee simulator or an RMUL rules implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
import math

from rl_bt.uc_rules import rule_facts


@lru_cache(maxsize=1)
def _facts():
    return rule_facts()


def _number(name, value, *, minimum=0., maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} outside the UC calculation range")
    return value


def logistics_coefficient(*, own_base_hp_lost):
    """§5.6.1/table5-19: lambda = 1 - lost base HP / 20000."""
    lost = _number("own_base_hp_lost", own_base_hp_lost,
                   maximum=_facts()["base_initial_hp"]["value"])
    return 1. - lost / 20000.


@dataclass(frozen=True)
class HealingEligibility:
    """Authoritative states, not inferred from a candidate's name or distance.

    ``own_outpost_fully_controlled`` refers to the same outpost position as the
    activated outpost buff; temporary control is insufficient. Team policy may
    separately disable an otherwise legal healing source.
    """
    alive: bool = True
    abnormal_offline: bool = False
    ejected: bool = False
    own_supply_active: bool = False
    own_outpost_active: bool = False
    own_outpost_fully_controlled: bool = False
    out_of_combat: bool = False
    allow_supply: bool = True
    allow_outpost: bool = True

    def __post_init__(self):
        if any(type(value) is not bool for value in self.__dict__.values()):
            raise ValueError("Healing eligibility flags must be explicit booleans")

    @property
    def can_heal(self):
        return self.alive and not self.abnormal_offline and not self.ejected


@dataclass(frozen=True)
class HealingRate:
    source: str
    fraction_max_hp_per_s: float
    hp_per_s: float
    logistics: float


def uc_passive_healing_rate(*, elapsed_s, max_hp, own_base_hp_lost,
                            eligibility: HealingEligibility):
    """§5.3.1: supply/outpost rates; simultaneous same-type buffs use max.

    At the 240-second phase boundary the late phase is considered entered.
    Actual referee update cadence and integer settlements are caller-owned.
    """
    _number("elapsed_s", elapsed_s, maximum=_facts()["match_duration"]["value"])
    _number("max_hp", max_hp, minimum=1.)
    logistics = logistics_coefficient(own_base_hp_lost=own_base_hp_lost)
    supply_fraction = _facts()["supply_heal_fraction_per_second"]["value"]
    if elapsed_s >= 240. and eligibility.out_of_combat:
        supply_fraction = _facts()["supply_late_out_of_combat_heal_fraction"]["value"]
    choices = [("none", 0.)]
    if eligibility.can_heal:
        if eligibility.own_supply_active and eligibility.allow_supply:
            choices.append(("own_supply", supply_fraction))
        if (eligibility.own_outpost_active and eligibility.own_outpost_fully_controlled
                and eligibility.allow_outpost):
            choices.append(("own_controlled_outpost", supply_fraction
                            * _facts()["outpost_heal_rate_relative_to_supply"]["value"]))
    source, fraction = max(choices, key=lambda choice: choice[1])
    fraction *= logistics
    return HealingRate(source, fraction, max_hp * fraction, logistics)


def settle_healing(*, hp, max_hp, raw_heal_hp):
    """Apply one authoritative settlement, round half up and cap at max HP.

    This function must not be called repeatedly with arbitrary small timesteps
    as a substitute for a verified referee settlement schedule.
    """
    _number("max_hp", max_hp, minimum=1.)
    _number("hp", hp, maximum=max_hp)
    _number("raw_heal_hp", raw_heal_hp)
    rounded = int(Decimal(str(raw_heal_hp)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return min(max_hp, hp + rounded)


@dataclass(frozen=True)
class RemoteHealEffect:
    status: str
    raw_heal_hp: float
    hp_after: float


def uc_remote_healing_effect(*, now_s, confirmed_at_s, hp, max_hp, own_base_hp_lost,
                             eligibility: HealingEligibility, died_during_delay=False):
    """Calculate an accepted purchase's one-shot effect at/after six seconds.

    Request legality (including out-of-combat at purchase), ACK, cost, and exactly
    once application belong to ``SentryBudgetLedger``/the referee adapter. The
    robot need not remain out of combat after confirmation. The caller must
    retain any intervening death even if the robot has since respawned.
    Current max HP and logistics at effect time are used.
    """
    _number("now_s", now_s)
    _number("confirmed_at_s", confirmed_at_s, maximum=now_s)
    _number("max_hp", max_hp, minimum=1.)
    _number("hp", hp, maximum=max_hp)
    logistics = logistics_coefficient(own_base_hp_lost=own_base_hp_lost)
    if type(died_during_delay) is not bool:
        raise ValueError("died_during_delay must be an explicit boolean")
    if died_during_delay or not eligibility.alive:
        return RemoteHealEffect("invalid_death_no_refund", 0., hp)
    if now_s - confirmed_at_s < _facts()["remote_heal_delay"]["value"]:
        return RemoteHealEffect("pending", 0., hp)
    if not eligibility.can_heal:
        # Timing, refund and any later retry for abnormal-offline/ejected cases
        # are deliberately not invented: authoritative reconciliation required.
        return RemoteHealEffect("ineligible_requires_referee_reconciliation", 0., hp)
    raw = max_hp * _facts()["remote_heal_fraction"]["value"] * logistics
    return RemoteHealEffect("due_effect_not_transaction_ack", raw,
                            settle_healing(hp=hp, max_hp=max_hp, raw_heal_hp=raw))
