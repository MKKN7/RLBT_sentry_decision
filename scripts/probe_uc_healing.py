#!/usr/bin/env python3
"""Execute UC healing/budget integration cases without inventing map positions."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_bt.economy import SentryBudgetLedger, uc_purchase_quote
from rl_bt.healing import HealingEligibility, uc_passive_healing_rate
from rl_bt.resource_executor import UCRemoteHealExecutor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/reports/uc_healing_execution_20261116.json")
    args = parser.parse_args()
    active = HealingEligibility(own_supply_active=True, out_of_combat=True)
    source_states = {
        "safe_cover_or_base_buff_only": HealingEligibility(),
        "own_supply": active,
        "outpost_temporarily_controlled": HealingEligibility(own_outpost_active=True),
        "outpost_fully_controlled": HealingEligibility(own_outpost_active=True,
            own_outpost_fully_controlled=True, out_of_combat=True),
        "dead_at_supply": HealingEligibility(alive=False, own_supply_active=True),
    }
    rates = {name: {str(t): asdict(uc_passive_healing_rate(elapsed_s=t, max_hp=400,
        own_base_hp_lost=0, eligibility=e)) for t in (239.999, 240.)}
        for name, e in source_states.items()}
    assert rates["safe_cover_or_base_buff_only"]["240.0"]["hp_per_s"] == 0
    assert rates["outpost_temporarily_controlled"]["240.0"]["hp_per_s"] == 0
    assert rates["own_supply"]["239.999"]["hp_per_s"] == 40
    assert rates["own_supply"]["240.0"]["hp_per_s"] == 100
    trials = []
    for died in (False, True):
        ledger = SentryBudgetLedger(team_balance=500, sentry_match_cap=200, team_reserve_floor=100)
        executor = UCRemoteHealExecutor(ledger)
        quote = uc_purchase_quote(request_id="demonstration", kind="remote_heal", quantity=1,
            remaining_s=420, wallet_version=ledger.version, now_s=0)
        assert executor.reserve(quote, now_s=0, life_seq=1, eligibility=active) is None
        assert ledger.mark_sent(quote.request_id)
        assert executor.confirm(quote.request_id, now_s=0)
        pending = executor.settle(quote.request_id, now_s=5.999, life_seq=1, hp=100,
            max_hp=400, own_base_hp_lost=0, eligibility=active)
        effect = executor.settle(quote.request_id, now_s=6, life_seq=2 if died else 1,
            hp=100, max_hp=400, own_base_hp_lost=0, eligibility=active)
        duplicate = executor.settle(quote.request_id, now_s=6, life_seq=2 if died else 1,
            hp=100+effect.hp_delta, max_hp=400, own_base_hp_lost=0, eligibility=active)
        assert effect.hp_delta == (0 if died else 240)
        assert duplicate.hp_delta == 0
        assert ledger.balance == 450 and ledger.spent == 50
        trials.append({"death_during_delay": died, "before_effect": asdict(pending),
            "due_effect": asdict(effect), "repeated_settlement": asdict(duplicate),
            "team_balance_after": ledger.balance, "sentry_spent": ledger.spent,
            "transaction_status": ledger.requests[quote.request_id]["status"]})
    report = {"status": "PASS", "artifact_type": "offline_UC_rule_and_economy_integration_smoke",
        "rules_basis": "user_supplied_UC_V1.0.0_20261015", "max_hp_test_case": 400,
        "passive_rates": rates, "remote_transactions": trials,
        "budgets_are_illustrative_test_inputs_not_team_policy": True,
        "map_coordinates_invented": False, "live_referee_connected": False,
        "phase1_alphaRM_training_changed": False,
        "limits": ["Explicit event inputs, not observations inferred from candidate names.",
            "Requires an event loop that advances to effect deadlines and tracks deaths.",
            "No complete UC battle simulator, autonomous purchase policy or official UL implementation."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
