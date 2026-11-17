from dataclasses import replace

import pytest

from rl_bt.economy import SentryBudgetLedger, uc_purchase_quote
from rl_bt.healing import HealingEligibility
from rl_bt.resource_executor import UCRemoteHealExecutor


def setup_purchase(*, life_seq=1):
    ledger = SentryBudgetLedger(team_balance=500, sentry_match_cap=200, team_reserve_floor=100)
    executor = UCRemoteHealExecutor(ledger)
    eligibility = HealingEligibility(out_of_combat=True)
    request = uc_purchase_quote(request_id="heal_1", kind="remote_heal", quantity=1,
        remaining_s=420, wallet_version=0, now_s=0)
    assert executor.reserve(request, now_s=0, life_seq=life_seq, eligibility=eligibility) is None
    assert ledger.mark_sent("heal_1")
    return ledger, executor, request, eligibility


def settle(executor, eligibility, *, now=6., life_seq=1, hp=100, max_hp=400):
    return executor.settle("heal_1", now_s=now, life_seq=life_seq, hp=hp, max_hp=max_hp,
                           own_base_hp_lost=0, eligibility=eligibility)


def test_confirm_once_effect_at_six_seconds_and_duplicate_effect_grants_nothing():
    ledger, executor, request, eligible = setup_purchase()
    assert ledger.balance == 500
    assert executor.confirm(request.request_id, now_s=0)
    assert not executor.confirm(request.request_id, now_s=0)
    assert ledger.balance == 450 and ledger.spent == 50
    assert settle(executor, eligible, now=5.999).status == "pending"
    # Re-entering combat after a legal request does not cancel the confirmed heal.
    effect = settle(executor, replace(eligible, out_of_combat=False))
    assert effect.hp_delta == 240 and effect.status == "effect_applied_once"
    assert settle(executor, eligible, hp=340).hp_delta == 0
    assert ledger.balance == 450 and ledger.spent == 50


def test_timeout_does_not_refund_or_create_effect_until_authoritative_acceptance():
    ledger, executor, request, eligible = setup_purchase()
    ledger.timeout(request.request_id)
    assert settle(executor, eligible).status == "awaiting_authoritative_acceptance"
    assert ledger.balance == 500
    assert executor.confirm(request.request_id, now_s=6)
    assert settle(executor, eligible, now=11.999).hp_delta == 0
    assert settle(executor, eligible, now=12).hp_delta == 240
    assert ledger.balance == 450


def test_death_then_revive_before_effect_is_not_mistaken_for_same_lifetime():
    ledger, executor, request, eligible = setup_purchase()
    executor.confirm(request.request_id, now_s=0)
    effect = settle(executor, eligible, life_seq=2, hp=40)
    assert effect.status == "invalid_death_no_refund" and effect.hp_delta == 0
    assert ledger.balance == 450 and ledger.spent == 50
    assert ledger.requests[request.request_id]["status"] == "failed_no_refund"
    assert settle(executor, eligible, life_seq=2, hp=40).status == "already_finalized"


def test_team_reserve_and_eligibility_are_hard_constraints_not_reward_preferences():
    ledger, executor, request, eligible = setup_purchase()
    other = replace(request, request_id="another")
    assert executor.reserve(other, now_s=0, life_seq=1, eligibility=eligible) == "purchase_in_flight"
    ledger.deny(request.request_id)
    assert executor.reserve(other, now_s=0, life_seq=1,
                             eligibility=replace(eligible, out_of_combat=False)) == "rule_ineligible"
    ledger.balance = 140
    assert executor.reserve(other, now_s=0, life_seq=1, eligibility=eligible) == "team_reserve_protected"


def test_effect_caps_actual_gain_and_rejects_counter_or_confirmation_reversal():
    ledger, executor, request, eligible = setup_purchase()
    with pytest.raises(ValueError):
        executor.confirm(request.request_id, now_s=-1)
    executor.confirm(request.request_id, now_s=0)
    with pytest.raises(ValueError):
        settle(executor, eligible, life_seq=0)
    assert settle(executor, eligible, hp=390).hp_delta == 10
