#!/usr/bin/env python3
"""Check stored UL/UC source provenance and report implemented partial modules."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from rl_bt.healing import HealingEligibility
from rl_bt.ul_rules import ULSentryAmmo, ul_supply_healing_rate, ULControlZone, ul_control_score_delta


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    official = ROOT / "data/references/official_rmul_2027"
    rules = json.loads((official / "manifest.json").read_text())
    maps = json.loads((official / "map_manifest.json").read_text())
    verified = []
    for entry in [rules] + maps["files"]:
        path = ROOT / entry["local_path"]
        data = path.read_bytes()
        assert digest(path) == entry["sha256"]
        assert len(data) == entry["attachment"]["len"]
        assert hashlib.md5(data).hexdigest() == entry["attachment"]["md5"]
        verified.append({"file": entry["local_path"], "sha256": digest(path), "bytes": len(data)})
    evidence = json.loads((ROOT / "data/reports/uc_reference_sources_20261116.json").read_text())
    source_manifest = json.loads((ROOT / "assets/navigation_reference/field/source_manifest.summary.json").read_text())
    current = evidence["files"][1]["summary"]
    assert source_manifest["source_sha256"] == current["source_sha256"]
    np.testing.assert_allclose(source_manifest["translation"], current["translation"], atol=0)
    gaps = []
    with np.load(ROOT / "data/derived/ul_surface_layers_005m/surfaces.npz") as a:
        for name, xy in (("red_reference_tunnel", [3.5, 7.5]), ("blue_reference_tunnel", [8.5, .5])):
            cell = np.floor((np.asarray(xy)-a["origin_xy_m"])/float(a["resolution_m"])).astype(int)
            column = int(cell[0])*int(a["grid_shape_xy"][1])+int(cell[1])
            lo, hi = a["column_offsets"][column:column+2]
            z = a["surface_z_m"][lo:hi]
            floor = z[np.isclose(z, .0005, atol=1e-5)]
            underside = z[np.isclose(z, .25, atol=1e-5)]
            assert len(floor) == len(underside) == 1
            gap = float(underside[0]-floor[0])
            assert abs(gap-.2495) < 1e-5
            gaps.append({"id": name, "sample_xy_m": xy, "all_surface_z_m": z.tolist(),
                         "floor_to_roof_underside_m": gap, "robot_clearance_passed": False})
    active = HealingEligibility(own_supply_active=True)
    ul_rates = [ul_supply_healing_rate(max_hp=400, own_control=100, enemy_control=100+gap,
                                      eligibility=active) for gap in [0, 200, 400]]
    assert ul_rates == [100., 132., 200.]
    zone = ULControlZone()
    red_time = zone.advance(duration_s=6, red_active=True, blue_active=False)
    recapture_time = zone.advance(duration_s=11, red_active=False, blue_active=True)
    assert red_time == [1., 0.] and recapture_time == [5., 1.]
    report = {"artifact_type": "source_and_partial_rules_readiness_not_full_match_acceptance",
        "checked_on": "2026-11-16", "official_UL_files_verified": verified,
        "UL_rule_fact_count": len(json.loads((ROOT / "data/rmul_rule_facts.json").read_text())["facts"]),
        "UL_supply_hp_per_s_at_400max_for_control_gaps_0_200_400": ul_rates,
        "UL_sentry_purchase_mask": ULSentryAmmo.purchase_mask(),
        "UL_control_event_replay": {"first_red_controlled_seconds": red_time,
            "blue_recapture_interval_controlled_seconds": recapture_time,
            "two_interval_control_score_delta": ul_control_score_delta(
                controlled_seconds_by_team=[6., 1.], deaths_by_team=[0, 0])},
        "UL_tunnel_vertical_sample_check": gaps,
        "UC_source_summary_matches_current_container": True,
        "UC_remote_heal_integration": "data/reports/uc_healing_execution_20261116.json",
        "mesh_layer_reports": ["data/derived/uc_surface_layers_010m/report.json",
                               "data/derived/ul_surface_layers_005m/report.json",
                               "data/derived/uc_original_surface_layers_010m/report.json"],
        "fully_implemented_UL_UC_match_simulator": False,
        "real_Nav2_MPPI_connected": True,
        "real_navigation_scope": "actual_PointLIO_Nav2_MPPI_original_UC_local_BT_BC_PPO_and_frozen_evaluations_not_full_match",
        "UC_original_STL_navigation_execution": "data/reports/uc_nav2_execution_20261116.json",
        "UC_original_field_preparation": "data/reports/uc_original_field_20261116.json",
        "UC_observed_support_execution": "data/reports/uc_support_execution_20261116.json",
        "UC_unknown_support_costmap_connected": True,
        "UC_dynamic_support_and_dropout_stop_validated": True,
        "UC_long_route_support_stability_validated": False,
        "formal_UL_UC_navigation_connected": False,
        "navigation_integration_document": "docs/17_nav2_mppi_integration.md",
        "new_RL_training_started": True,
        "new_training_scope": "UC_actual_PointLIO_Nav2_MPPI_local_PPO_pilot_and_separate_historical_proxy_courses_not_full_match",
        "new_training_run": "runs/uc_live_actor_validity_train_20261117",
        "UC_live_actor_validity_report": "data/reports/live_nav2_validity_20261117/report.md",
        "UC_live_actor_validity_frozen_evaluation": {"BC_successes": 3, "PPO_successes": 2,
            "episodes_per_policy": 4, "sensor_seeds": [41000, 41001, 41002, 41003],
            "identical_physics_pairing": False, "PPO_gain_demonstrated": False,
            "recorded_observation_comparison_count": 135, "recorded_greedy_action_differences": 0},
        "UC_live_training_continuation_report": "data/reports/live_nav2_continue_20261117/report.md",
        "UC_live_frozen_evaluation": {"BC_successes": 4, "PPO_successes": 0,
            "episodes_per_policy": 4, "identical_physics_pairing": False,
            "startup_support_cohorts_differ": True, "PPO_gain_demonstrated": False},
        "UC_live_training_report": "data/reports/live_nav2_training_20261117.json",
        "UC_live_training_actual_Nav2_rollouts": True,
        "UC_v2_training_report": "data/reports/uc_support_v2_training_20261117.json",
        "UC_v2_optimizer_comparison": "data/reports/uc_support_v2_20261117/report.md",
        "UC_training_report": "data/reports/uc_support_training_20261116.json",
        "UC_training_actual_Nav2_rollouts": False,
        "UC_PPO_gain_demonstrated": False,
        "UL_payload_and_public_cache": "docs/15_ul_observation_protocol.md",
        "navigation_contract": "docs/navigation_execution_contract.md",
        "reference_robot_envelope": "configs/reference_robot_envelope.json",
        "UL_ground_navigation_proxy": "data/derived/ul_navigation_proxy_v1/report.json",
        "remaining": ["UL live serial coverage, byte order, clocks and coordinate registration; UC actor observation projection.",
            "UC official frame/zone registration; UL nominal surfaces need RFID/body footprint registration.",
            "Full robot support, side clearance, tunnel contacts and physical execution validation.",
            "Referee settlement cadence, neutral partial UL capture idle handling and simultaneous-zero outcomes.",
            "Whole battle damage, role, revive, economy and policy training integration."],
        "status": "PASS_for_reported_scope_only"}
    output = ROOT / "data/reports/competition_readiness_20261116.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "official_UL_files": len(verified),
                      "out": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
