import pytest
from rl_bt.economy import PurchaseRequest, SentryBudgetLedger, remote_heal_price, uc_purchase_quote


def test_uc_heal_price_rounds_whole_product_not_elapsed_minutes():
    assert remote_heal_price(420) == 50
    assert remote_heal_price(419.9) == 51
    assert remote_heal_price(417) == 51
    assert remote_heal_price(416.99) == 52
    assert remote_heal_price(360) == 70
    assert remote_heal_price(60) == 170
    assert remote_heal_price(0) == 190


def test_team_reserve_budget_duplicate_and_unknown_ack():
    # Synthetic unit-test budgets, not recommended team policy values.
    ledger = SentryBudgetLedger(team_balance=300, sentry_match_cap=150, team_reserve_floor=200)
    q = PurchaseRequest("q1", "remote_heal", 50, 0, 10)
    assert ledger.reserve(q, now_s=10, alive=True, out_of_combat=False) == "rule_ineligible"
    assert ledger.reserve(q, now_s=10, alive=True, out_of_combat=True) is None
    assert ledger.reserve(q, now_s=10, alive=True, out_of_combat=True) == "duplicate_request"
    assert ledger.mark_sent("q1")
    ledger.timeout("q1")
    second = PurchaseRequest("q2", "remote_ammo", 150, 0, 11)
    assert ledger.reserve(second, now_s=11, alive=True, out_of_combat=True) == "purchase_in_flight"
    assert ledger.accept("q1", now_s=11)
    assert not ledger.accept("q1", now_s=11)
    assert ledger.balance == 250 and ledger.spent == 50
    assert not ledger.effect("q1", now_s=16.999, succeeded=True)
    assert ledger.effect("q1", now_s=17, succeeded=True)
    reserve_breaking = PurchaseRequest("q3", "local_ammo", 100, 1, 18)
    assert ledger.reserve(reserve_breaking, now_s=18, alive=True, out_of_combat=True) == "team_reserve_protected"
    cap_breaking = PurchaseRequest("q4", "remote_ammo", 150, 1, 18)
    assert ledger.reserve(cap_breaking, now_s=18, alive=True, out_of_combat=True) == "sentry_budget_exceeded"


def test_death_before_remote_heal_effect_does_not_refund():
    ledger = SentryBudgetLedger(team_balance=300, sentry_match_cap=100, team_reserve_floor=100)
    q = PurchaseRequest("heal", "remote_heal", 50, 0, 0)
    assert ledger.reserve(q, now_s=0, alive=True, out_of_combat=True) is None
    assert ledger.mark_sent("heal")
    assert ledger.accept("heal", now_s=0)
    assert ledger.effect("heal", now_s=2, succeeded=False)
    assert ledger.balance == 250 and ledger.spent == 50
    assert not ledger.deny("heal")


def test_ammo_bundle_prices_and_team_limit_exclude_free_allowance():
    args = dict(request_id="ammo", remaining_s=300, wallet_version=0, now_s=120)
    q = uc_purchase_quote(kind="remote_ammo", quantity=100, team_purchased_17mm=900, **args)
    assert q.coins == 150 and q.quantity == 100
    q = uc_purchase_quote(kind="local_ammo", quantity=20, team_purchased_17mm=0, **args)
    assert q.coins == 20
    with pytest.raises(ValueError):
        uc_purchase_quote(kind="remote_ammo", quantity=10, team_purchased_17mm=0, **args)
    with pytest.raises(ValueError):
        uc_purchase_quote(kind="remote_ammo", quantity=100, **args)
    with pytest.raises(ValueError):
        uc_purchase_quote(kind="remote_ammo", quantity=100, team_purchased_17mm=910, **args)
