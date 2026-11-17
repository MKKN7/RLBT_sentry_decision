from dataclasses import replace

import pytest

from rl_bt.healing import (HealingEligibility, logistics_coefficient, settle_healing,
                          uc_passive_healing_rate, uc_remote_healing_effect)


def rate(eligibility, elapsed=0., lost=0.):
    return uc_passive_healing_rate(elapsed_s=elapsed, max_hp=400,
                                  own_base_hp_lost=lost, eligibility=eligibility)


def test_only_confirmed_healing_sources_grant_passive_healing():
    neutral = HealingEligibility()
    # Navigating home, cover and a base-defense buff are not healing activations.
    assert rate(neutral).hp_per_s == 0.
    assert rate(replace(neutral, own_supply_active=True)).hp_per_s == 40.
    assert rate(replace(neutral, own_outpost_active=True)).hp_per_s == 0.
    outpost = replace(neutral, own_outpost_active=True, own_outpost_fully_controlled=True)
    assert rate(outpost).hp_per_s == 20.
    both = replace(outpost, own_supply_active=True)
    assert rate(both).hp_per_s == 40.  # maximum, not 40 + 20
    assert rate(both).source == "own_supply"
    assert rate(replace(both, allow_supply=False)).hp_per_s == 20.
    assert rate(replace(both, allow_supply=False, allow_outpost=False)).hp_per_s == 0.


def test_late_phase_requires_out_of_combat_and_ends_immediately_on_combat():
    active = HealingEligibility(own_supply_active=True, out_of_combat=True)
    assert rate(active, elapsed=239.999).hp_per_s == 40.
    assert rate(active, elapsed=240.).hp_per_s == 100.
    assert rate(replace(active, out_of_combat=False), elapsed=241.).hp_per_s == 40.
    outpost = replace(active, own_supply_active=False, own_outpost_active=True,
                      own_outpost_fully_controlled=True)
    assert rate(outpost, elapsed=241.).hp_per_s == 50.


@pytest.mark.parametrize("changed", [dict(alive=False), dict(abnormal_offline=True), dict(ejected=True)])
def test_ineligible_robot_has_no_passive_healing(changed):
    assert rate(HealingEligibility(own_supply_active=True, **changed)).hp_per_s == 0.


def test_logistics_and_authoritative_integer_settlement():
    assert logistics_coefficient(own_base_hp_lost=0) == 1.
    assert logistics_coefficient(own_base_hp_lost=10000) == .5
    assert rate(HealingEligibility(own_supply_active=True), lost=5000).hp_per_s == pytest.approx(30.)
    assert settle_healing(hp=100, max_hp=400, raw_heal_hp=2.5) == 103
    assert settle_healing(hp=399, max_hp=400, raw_heal_hp=30.) == 400


def test_remote_heal_uses_effect_time_max_hp_and_allows_post_purchase_combat():
    args = dict(confirmed_at_s=10., hp=100, max_hp=500, own_base_hp_lost=5000,
                eligibility=HealingEligibility(out_of_combat=False))
    assert uc_remote_healing_effect(now_s=15.999, **args).status == "pending"
    effect = uc_remote_healing_effect(now_s=16., **args)
    assert effect.raw_heal_hp == 225.
    assert effect.hp_after == 325.
    assert effect.status == "due_effect_not_transaction_ack"
    args["hp"] = 490
    assert uc_remote_healing_effect(now_s=16., **args).hp_after == 500.


def test_death_during_delay_invalidates_even_after_respawn_and_no_refund_is_invented():
    args = dict(now_s=16., confirmed_at_s=10., hp=40, max_hp=400,
                own_base_hp_lost=0, eligibility=HealingEligibility())
    effect = uc_remote_healing_effect(died_during_delay=True, **args)
    assert effect.status == "invalid_death_no_refund"
    assert effect.hp_after == 40 and effect.raw_heal_hp == 0.
    args["eligibility"] = HealingEligibility(abnormal_offline=True)
    assert uc_remote_healing_effect(**args).status == "ineligible_requires_referee_reconciliation"


@pytest.mark.parametrize("lost", [-1, 10001, float("nan"), float("inf"), True])
def test_invalid_logistics_input_is_not_silently_clipped(lost):
    with pytest.raises(ValueError):
        logistics_coefficient(own_base_hp_lost=lost)


def test_unknown_activation_or_invalid_hp_and_times_are_not_accepted():
    with pytest.raises(ValueError):
        HealingEligibility(own_supply_active=None)
    with pytest.raises(ValueError):
        rate(HealingEligibility(), elapsed=421.)
    with pytest.raises(ValueError):
        settle_healing(hp=401, max_hp=400, raw_heal_hp=10)
    with pytest.raises(ValueError):
        uc_remote_healing_effect(now_s=10., confirmed_at_s=11., hp=100, max_hp=400,
                                 own_base_hp_lost=0, eligibility=HealingEligibility())
