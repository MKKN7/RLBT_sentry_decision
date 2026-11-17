import copy
import json
from types import SimpleNamespace

import pytest

from scripts import probe_strategy_scenarios as probe


def row(policy, seed=62011, style="capture", side=0, mode="anchors", *, win=True,
        draw=False, reward=2., duration=100.):
    return {"policy":policy, "seed":seed, "style":style, "side":side,
        "support_mode":mode, "scenario_id":probe.SCENARIO_ID,
        "strategy_scene":{"support_mode":mode}, "completed":True, "truncated":False,
        "win":win, "draw":draw, "return":reward, "actual_duration_s":duration,
        "decisions":200, "switches":1, "overrides":0, "recoveries":0,
        "diagnostics":{"sampled_center_fraction":.5, "damage_dealt":100.,
            "sampled_alive_fraction":.8, "sampled_home_fraction":.1,
            "sampled_mean_own_hp_fraction":.7, "damage_share_of_team":.3,
            "injected_detection_missing_fraction":.1, "overrides_per_decision":0.}}


def passing_rows(seeds=(62011,)):
    return [row(policy, seed, style, side, win=policy != "continue_only",
                reward=0. if policy == "continue_only" else 2.)
            for seed in seeds for style in probe.SCRIPTED_STYLES for side in (0,1)
            for policy in probe.POLICIES]


def test_predeclared_gate_passes_informative_complete_long_cases():
    gate = probe.evaluate_gate(passing_rows(), "anchors")
    assert gate["passed"] and all(gate["criteria"].values())
    assert gate["paired_cases"] == 4 and gate["scripted_episodes"] == 16
    assert gate["HOLD_vs_either_BT_different_case_fraction"] == 1.
    assert gate["BT_minus_HOLD"]["bt_reactive"]["points_delta"] == 1.


def test_strong_pool_is_reported_but_never_changes_scripted_gate():
    rows = passing_rows()
    base = probe.evaluate_gate(rows, "anchors")
    rows.extend(row(policy, style="pretrained", win=True, duration=1.) for policy in probe.POLICIES)
    assert probe.evaluate_gate(rows, "anchors") == base


def test_gate_fails_ceiling_and_draw_is_not_loss():
    rows = passing_rows()
    for item in rows:
        item["win"] = True
    gate = probe.evaluate_gate(rows, "anchors")
    assert not gate["passed"] and not gate["criteria"]["at_least_one_true_win_and_loss"]
    assert not gate["criteria"]["HOLD_vs_BT_outcomes_differ_at_least_25pct_cases"]
    for item in rows:
        if item["policy"] == "continue_only":
            item.update(win=False, draw=True)
    gate = probe.evaluate_gate(rows, "anchors")
    assert gate["losses"] == 0 and not gate["criteria"]["at_least_one_true_win_and_loss"]


def test_gate_checks_difference_threshold_and_direction_separately():
    rows = passing_rows(seeds=(62011,62012))
    for item in rows:
        item["win"] = True
        if item["policy"] == "continue_only" and item["seed"] == 62011 and \
                item["style"] == "capture" and item["side"] == 0:
            item["win"] = False
    gate = probe.evaluate_gate(rows,"anchors")
    assert gate["HOLD_vs_either_BT_different_case_fraction"] == .125
    assert not gate["criteria"]["HOLD_vs_BT_outcomes_differ_at_least_25pct_cases"]
    assert gate["criteria"]["BT_better_mean_points_or_return_above_half"]
    rows = passing_rows()
    for item in rows:
        item["win"] = item["policy"] == "continue_only"
        item["return"] = 1.
    gate = probe.evaluate_gate(rows,"anchors")
    assert gate["criteria"]["HOLD_vs_BT_outcomes_differ_at_least_25pct_cases"]
    assert not gate["criteria"]["BT_better_mean_points_or_return_above_half"]


def test_gate_checks_long_fraction_and_median_independently():
    rows = passing_rows()
    for item in rows:
        item["actual_duration_s"] = 80.
    gate = probe.evaluate_gate(rows,"anchors")
    assert gate["criteria"]["median_duration_at_least_60s"]
    assert not gate["criteria"]["at_least_quarter_episodes_90s"]
    for index,item in enumerate(rows):
        item["actual_duration_s"] = 30. if index < 10 else 100.
    gate = probe.evaluate_gate(rows,"anchors")
    assert not gate["criteria"]["median_duration_at_least_60s"]
    assert gate["criteria"]["at_least_quarter_episodes_90s"]


@pytest.mark.parametrize("mutation",("missing","duplicate","truncated","wrong_expected"))
def test_gate_rejects_incomplete_or_duplicate_case_identity(mutation):
    rows=passing_rows()
    expected={(62011,style,side) for style in probe.SCRIPTED_STYLES for side in (0,1)}
    if mutation == "missing": rows.pop()
    elif mutation == "duplicate": rows.append(copy.deepcopy(rows[0]))
    elif mutation == "truncated": rows[0].update(completed=False,truncated=True,win=None,draw=None)
    else: expected.add((62012,"capture",0))
    gate=probe.evaluate_gate(rows,"anchors",expected_cases=expected)
    assert not gate["passed"] and not gate["criteria"]["complete_case_identity_and_true_episodes"]


def test_run_one_reuses_fixed_and_bt_engines_and_marks_development(monkeypatch):
    seen=[]
    def fixed(runtime,policy,case,**kwargs):
        seen.append((policy,kwargs["env_factory"].keywords["support_mode"]))
        return row(policy,**case)
    def baseline(runtime,spec,case,**kwargs):
        seen.append((spec["label"],kwargs["env_factory"].keywords["support_mode"]))
        return row(spec["label"],**case),None
    monkeypatch.setattr(probe,"run_control_trial",fixed)
    monkeypatch.setattr(probe,"run_trial",baseline)
    case={"seed":62011,"style":"capture","side":0}
    for policy in probe.POLICIES:
        measured=probe.run_one(None,"anchors",policy,case)
        assert measured["data_role"] == "development_calibration"
        assert not measured["independent_new_test"] and not measured["exploratory_posthoc"]
        assert measured["SGD_steps"] == 0 and not measured["learned_weights_loaded"]
    assert seen == [(policy,"anchors") for policy in probe.POLICIES]


def test_run_one_rejects_fake_draw_and_wrong_scene_metadata(monkeypatch):
    case={"seed":62011,"style":"capture","side":0}
    bad=row("bt_reactive",**case)
    bad.update(completed=False,truncated=True,win=None,draw=None)
    monkeypatch.setattr(probe,"run_trial",lambda *a,**k:(bad,None))
    with pytest.raises(RuntimeError,match="cutoff"):
        probe.run_one(None,"anchors","bt_reactive",case)
    bad.update(completed=True,truncated=False,scenario_id="old_scenario")
    with pytest.raises(RuntimeError,match="metadata"):
        probe.run_one(None,"anchors","bt_reactive",case)


def test_probe_writes_source_and_gate_artifacts_without_any_model(monkeypatch,tmp_path):
    def fake_one(runtime,mode,policy,case,**kwargs):
        return row(policy,mode=mode,**case,win=policy != "continue_only",
                   reward=0. if policy == "continue_only" else 2.)
    monkeypatch.setattr(probe,"run_one",fake_one)
    result=probe.run_probe(tmp_path/"probe",seeds=[62011],styles=["capture","resupply","pretrained"],
        modes=["anchors","balanced"],workers=1,runtime_factory=lambda:SimpleNamespace(source_commit="test"))
    assert result["actual_episodes"] == 48 and result["sources_unchanged"]
    assert all(gate["passed"] for gate in result["gates"].values())
    assert not result["automatic_training_launched"] and result["SGD_steps"] == 0
    assert (tmp_path/"probe/development_gate.md").is_file()
    assert len((tmp_path/"probe/episodes.jsonl").read_text().splitlines()) == 48
    status=json.loads((tmp_path/"probe/status.json").read_text())
    assert status["status"] == "complete"
    with pytest.raises(FileExistsError):
        probe.run_probe(tmp_path/"probe")


@pytest.mark.parametrize("seed",(51001,51002,73001,73002))
def test_old_and_new_test_seeds_cannot_enter_development(tmp_path,seed):
    with pytest.raises(ValueError,match="final-test"):
        probe.run_probe(tmp_path/"bad",seeds=[seed])
    assert not (tmp_path/"bad").exists()
