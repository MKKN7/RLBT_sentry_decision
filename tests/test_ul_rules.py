from dataclasses import replace

import pytest

from rl_bt.healing import HealingEligibility
from rl_bt.ul_rules import (ULSentryAmmo, ul_apply_healing_settlement, ul_match_outcome,
                            ul_revive_progress_required, ul_supply_healing_rate,
                            ULControlZone, ul_control_score_delta)


def test_capture_threshold_is_invariant_to_float_interval_subdivision():
    whole = ULControlZone()
    whole.advance(duration_s=5., red_active=True, blue_active=False)
    split = ULControlZone()
    for _ in range(49):
        split.advance(duration_s=.1, red_active=True, blue_active=False)
    assert split.owner is None  # do not capture a genuine 0.1s early
    split.advance(duration_s=.1, red_active=True, blue_active=False)
    assert split.owner == whole.owner == 0
    assert split.progress == whole.progress == [5., 0.]


def test_ul_healing_only_at_own_supply_with_current_comeback_thresholds():
    active = HealingEligibility(own_supply_active=True, out_of_combat=False)
    def rate(gap, eligibility=active):
        return ul_supply_healing_rate(max_hp=400, own_control=100, enemy_control=100+gap,
                                       eligibility=eligibility)
    assert rate(0) == rate(199.999) == 100
    assert rate(200) == rate(399.999) == 132
    assert rate(400) == 200
    assert rate(-100) == 100
    assert rate(400, replace(active, alive=False)) == 0
    assert rate(0, replace(active, own_supply_active=False, own_outpost_active=True,
                           own_outpost_fully_controlled=True)) == 0
    assert ul_apply_healing_settlement(hp=390, max_hp=400, hp_per_s=100,
                                       settlement_duration_s=1) == 400


def test_ul_ammo_never_refills_at_supply_and_purchase_candidates_are_disabled():
    ammo = ULSentryAmmo()
    assert ammo.record_shots(250) == 500
    assert ammo.record_shots(500) == 0
    assert not any(ammo.purchase_mask().values())
    with pytest.raises(ValueError):
        ammo.record_shots(1)
    with pytest.raises(ValueError):
        ULSentryAmmo(751)


def test_ul_revive_threshold_is_not_uc_formula_or_fixed_wait():
    assert ul_revive_progress_required(death_count=1, own_control=600, enemy_control=600) == 5
    assert ul_revive_progress_required(death_count=2, own_control=400, enemy_control=600) == pytest.approx(6.7)
    assert ul_revive_progress_required(death_count=3, own_control=200, enemy_control=600) == 7.5


def test_ul_terminal_priority_and_unresolved_simultaneous_zero():
    args = dict(remaining_s=300, control=[600, 600], team_attack_damage=[10, 20],
                 team_remaining_hp=[100, 300])
    assert not ul_match_outcome(**args).done
    result = ul_match_outcome(**dict(args, control=[1, 0]))
    assert result.winner == 0 and result.reason == "opponent_control_zero"
    assert ul_match_outcome(**dict(args, remaining_s=0, control=[500, 400])).winner == 0
    assert ul_match_outcome(**dict(args, remaining_s=0)).reason == "higher_team_attack_damage"
    assert ul_match_outcome(**dict(args, remaining_s=0, team_attack_damage=[20, 20])).reason == "higher_team_remaining_hp"
    assert ul_match_outcome(**dict(args, remaining_s=0, team_attack_damage=[20, 20],
                                  team_remaining_hp=[100, 100])).reason == "draw"
    result = ul_match_outcome(**dict(args, control=[0, 0]))
    assert result.winner is None and "reconciliation" in result.reason


def test_ul_capture_first_clears_enemy_progress_then_claims_control():
    zone = ULControlZone()
    assert zone.advance(duration_s=6, red_active=True, blue_active=False) == [1, 0]
    assert zone.progress == [5, 0] and zone.owner == 0
    assert zone.advance(duration_s=2, red_active=True, blue_active=True) == [2, 0]
    assert zone.progress == [5, 0]
    assert zone.advance(duration_s=7, red_active=False, blue_active=True) == [5, 0]
    assert zone.progress == [0, 2] and zone.owner is None
    assert zone.advance(duration_s=4, red_active=False, blue_active=True) == [0, 1]
    assert zone.owner == 1 and zone.progress == [0, 5]
    assert zone.advance(duration_s=2, red_active=False, blue_active=False) == [0, 2]
    assert zone.owner == 1 and zone.progress == [0, 3]
    assert zone.advance(duration_s=4, red_active=False, blue_active=False) == [0, 3]
    assert zone.owner is None


def test_ul_control_drain_and_death_penalty_are_different_from_center_circle_score():
    assert ul_control_score_delta(controlled_seconds_by_team=[6, 2], deaths_by_team=[1, 0]) == [-12, -6]
    partial = ULControlZone(progress=[2., 0.])
    with pytest.raises(ValueError, match="clarification"):
        partial.advance(duration_s=1, red_active=False, blue_active=False)
