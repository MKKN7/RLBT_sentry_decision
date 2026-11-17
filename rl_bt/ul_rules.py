"""Selected official RMUL 2027 3v3 rule calculations, not a full match engine.

Authoritative activation/control events are inputs. No UC economy, outpost,
stance or base rules are inherited. Actor observation permissions and physical
RFID detection remain separate contracts.
"""
from dataclasses import dataclass, field
from functools import lru_cache
import json
import math

from rl_bt.geometry import ROOT
from rl_bt.healing import HealingEligibility, settle_healing


@lru_cache(maxsize=1)
def ul_facts():
    return {f["id"]: f for f in json.loads((ROOT / "data/rmul_rule_facts.json").read_text())["facts"]}


def _finite(name, value, minimum=0.):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"Finite {name} >= {minimum} required")
    return float(value)


def ul_supply_healing_rate(*, max_hp, own_control, enemy_control, eligibility: HealingEligibility):
    """§3.3.2.1/3.3.4.2: 25%, or 33%/50% for the trailing side.

    No out-of-combat or UC logistics coefficient is introduced. Outpost flags
    cannot grant healing in UL. Caller owns settlement cadence and RFID state.
    """
    _finite("max_hp", max_hp, 1.)
    _finite("own_control", own_control); _finite("enemy_control", enemy_control)
    fraction = ul_facts()["supply_healing_rate"]["value"]
    for threshold in ul_facts()["comeback_healing_and_revive"]["value"]:
        if enemy_control - own_control >= threshold["control_gap_at_least"]:
            fraction = threshold["supply_heal_fraction_max_hp_per_s"]
    if not (eligibility.can_heal and eligibility.own_supply_active and eligibility.allow_supply):
        return 0.
    return max_hp * fraction


def ul_revive_progress_required(*, death_count, own_control, enemy_control):
    """Return unrounded progress threshold; no unverified tick rounding imposed."""
    if type(death_count) is not int or death_count < 1:
        raise ValueError("death_count starts at the first death, value 1")
    _finite("own_control", own_control); _finite("enemy_control", enemy_control)
    reduction = 0.
    facts = ul_facts()
    for threshold in facts["comeback_healing_and_revive"]["value"]:
        if enemy_control - own_control >= threshold["control_gap_at_least"]:
            reduction = threshold["revive_progress_reduction_fraction"]
    nominal = (facts["first_death_revive_progress_required"]["value"]
        + (death_count - 1) * facts["revive_progress_increment_per_death"]["value"])
    return nominal * (1. - reduction)


@dataclass
class ULSentryAmmo:
    """Allowance ledger, updated by authoritative fired-projectile deltas."""
    remaining: int = 750

    def __post_init__(self):
        cap = ul_facts()["sentry_ammo_allowance_initial"]["value"]
        if type(self.remaining) is not int or not 0 <= self.remaining <= cap:
            raise ValueError("UL sentry allowance must be within 0..750")

    def record_shots(self, count):
        if type(count) is not int or count < 0:
            raise ValueError("Nonnegative projectile delta required")
        if count > self.remaining:
            raise ValueError("Shots exceed UL allowance; referee fault handling required")
        self.remaining -= count
        return self.remaining

    @staticmethod
    def purchase_mask():
        return {"remote_heal": False, "remote_ammo": False, "local_ammo": False}


@dataclass(frozen=True)
class ULMatchOutcome:
    done: bool
    winner: int | None  # 0 red, 1 blue, None unresolved or draw
    reason: str


def ul_match_outcome(*, remaining_s, control, team_attack_damage, team_remaining_hp):
    """§3.3.7 comparison only; does not invent simultaneous-zero settlement."""
    _finite("remaining_s", remaining_s)
    if remaining_s > ul_facts()["match_duration"]["value"]:
        raise ValueError("UL remaining time exceeds 300 seconds")
    arrays = (control, team_attack_damage, team_remaining_hp)
    for name, values in zip(("control", "damage", "hp"), arrays):
        if len(values) != 2:
            raise ValueError(f"Two team values required for {name}")
        for value in values:
            _finite(name, value)
    zeros = [float(value) <= 0 for value in control]
    if all(zeros):
        return ULMatchOutcome(True, None, "simultaneous_zero_requires_referee_reconciliation")
    if any(zeros):
        return ULMatchOutcome(True, 1 if zeros[0] else 0, "opponent_control_zero")
    if remaining_s > 0:
        return ULMatchOutcome(False, None, "running")
    for name, values in zip(("higher_control_at_timeout", "higher_team_attack_damage",
                             "higher_team_remaining_hp"), arrays):
        if values[0] != values[1]:
            return ULMatchOutcome(True, int(values[1] > values[0]), name)
    return ULMatchOutcome(True, None, "draw")


def ul_apply_healing_settlement(*, hp, max_hp, hp_per_s, settlement_duration_s):
    """One event settlement; do not assume this defines referee update cadence."""
    _finite("hp_per_s", hp_per_s); _finite("settlement_duration_s", settlement_duration_s)
    return settle_healing(hp=hp, max_hp=max_hp, raw_heal_hp=hp_per_s * settlement_duration_s)


@dataclass
class ULControlZone:
    """§3.3.4.1 event-interval capture progress with persistent control.

    Activation inputs already account for RFID expiry and robot weak/dead state.
    Returns each side's controlled seconds for downstream score integration.
    Referee tick cadence is not inferred from the rule's per-second rates.
    """
    progress: list = field(default_factory=lambda: [0., 0.])
    owner: int | None = None

    def advance(self, *, duration_s, red_active, blue_active):
        duration = _finite("duration_s", duration_s)
        if type(red_active) is not bool or type(blue_active) is not bool:
            raise ValueError("Explicit team activation booleans required")
        if self.owner not in (None, 0, 1) or len(self.progress) != 2:
            raise ValueError("Invalid control-zone state")
        limit = ul_facts()["control_progress_bounds"]["value"][1]
        for value in self.progress:
            if not 0 <= _finite("capture progress", value) <= limit:
                raise ValueError("Control progress outside bounds")
        if self.progress[0] > 0 and self.progress[1] > 0:
            raise ValueError("Both progress values positive requires referee reconciliation")
        if self.owner is not None and self.progress[self.owner] <= 0:
            raise ValueError("A zero-progress owner is inconsistent")
        if self.owner is None and max(self.progress) >= limit:
            raise ValueError("Full capture requires an owner")
        controlled = [0., 0.]
        if red_active and blue_active:
            if self.owner is not None:
                controlled[self.owner] = duration
            return controlled
        if not red_active and not blue_active:
            if self.owner is None and max(self.progress) > 0:
                # The text explicitly covers non-neutral idle zones only.
                raise ValueError("Neutral partial-capture idle decay requires referee clarification")
            if self.owner is not None:
                owner = self.owner
                controlled[owner] = min(duration, self.progress[owner])
                self.progress = [max(0., x-duration) for x in self.progress]
                if self.progress[owner] <= 0:
                    self.owner = None
            return controlled
        active = 0 if red_active else 1
        enemy = 1 - active
        if self.owner == active:
            controlled[active] = duration
        remaining = duration
        if self.progress[enemy] > 0:
            removed = min(remaining, self.progress[enemy])
            if self.owner == enemy:
                controlled[enemy] = removed
            self.progress[enemy] = max(0., self.progress[enemy]-removed)
            remaining -= removed
            if self.progress[enemy] <= 0 and self.owner == enemy:
                self.owner = None
        if remaining > 0:
            needed = limit - self.progress[active]
            self.progress[active] = min(limit, self.progress[active]+remaining)
            # Fifty 0.1s updates must agree with one 5s interval; this is a
            # numerical tolerance, not an inferred referee tick cadence.
            reached_limit = remaining >= needed or math.isclose(
                remaining, needed, rel_tol=0., abs_tol=1e-10)
            if reached_limit:
                self.progress[active] = float(limit)
            if self.owner is None and reached_limit:
                self.owner = active
                controlled[active] += max(0., remaining-needed)
        return controlled


def ul_control_score_delta(*, controlled_seconds_by_team, deaths_by_team):
    """Sum interval zone time first; this does not settle winner or tick order."""
    if len(controlled_seconds_by_team) != 2 or len(deaths_by_team) != 2:
        raise ValueError("Two team inputs required")
    facts = ul_facts()
    for value in controlled_seconds_by_team:
        _finite("controlled zone seconds", value)
    if any(type(value) is not int or value < 0 for value in deaths_by_team):
        raise ValueError("Nonnegative integer authoritative death counts required")
    drain = facts["enemy_control_score_drain_per_controlled_zone"]["value"]
    death_cost = facts["own_control_score_death_loss"]["value"]
    return [-controlled_seconds_by_team[1]*drain-deaths_by_team[0]*death_cost,
            -controlled_seconds_by_team[0]*drain-deaths_by_team[1]*death_cost]
