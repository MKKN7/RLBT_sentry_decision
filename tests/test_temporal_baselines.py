import copy

import numpy as np
import pytest
import torch

from rl_bt.temporal_baselines import BTMemoryConfig, BTMemoryPolicy
from scripts.evaluate_temporal_phase import (FrozenPolicy, make_cases, model_weights_sha256,
                                            paired_contrast, run_trial, summarize_policy)


def observation(hp=1., elapsed=0., active=0):
    state = np.zeros(148, dtype=np.float32)
    state[7], state[11], state[125], state[140] = hp, 1., 1.-elapsed/180., active/15.
    candidates = np.zeros((16, 16), dtype=np.float32)
    candidates[:, 2:5] = .5
    candidates[[1, 3, 4], 5] = 1
    candidates[2, 6] = 1
    candidates[5:11, 7] = 1
    candidates[11:, 8] = 1
    return {"state": state, "candidates": candidates, "mask": np.ones(16, dtype=bool)}


def test_retreat_hysteresis_and_episode_reset():
    bt = BTMemoryPolicy()
    assert bt.act(observation(.2)) == 2
    assert bt.act(observation(.6, elapsed=10)) == 2
    assert bt.act(observation(.85, elapsed=20)) in (1, 3, 4)
    bt.reset()
    assert not bt.retreating and bt.committed_candidate is None and bt.decisions == 0
    assert bt.act(observation(.6)) in (1, 3, 4)


def test_task_commitment_carries_until_observed_clock_expires():
    bt = BTMemoryPolicy()
    obs = observation()
    obs["candidates"][1, 3] = 0.
    assert bt.act(obs) == 1
    changed = observation(elapsed=1, active=1)
    changed["candidates"][1, 3] = 10.
    changed["candidates"][3, 3] = 0.
    assert bt.act(changed) == 0
    assert bt.last_reason == "committed_task"
    changed["state"][125] = 1.-6./180.
    assert bt.act(changed) == 3


def test_memory_never_keeps_illegal_or_dead_task():
    bt = BTMemoryPolicy()
    assert bt.act(observation()) == 1
    obs = observation(elapsed=1, active=1)
    obs["mask"][1] = False
    assert bt.act(obs) in (3, 4)
    obs["state"][11] = 0
    obs["mask"][:] = False
    obs["mask"][0] = True
    assert bt.act(obs) == 0
    assert bt.committed_candidate is None
    assert bt.state_dict()["privileged_inputs"] is False


def test_bt_accepts_only_public_schema_and_obeys_mask():
    bt = BTMemoryPolicy()
    obs = observation(.1)
    obs["mask"][:] = False
    obs["mask"][0] = True
    assert bt.act(obs) == 0
    with pytest.raises(ValueError):
        bt.act({**obs, "mask": np.zeros(16, bool)})
    with pytest.raises(ValueError):
        BTMemoryConfig(retreat_enter_hp=.9, retreat_exit_hp=.8)


def test_common_case_manifest_is_explicit_and_unique():
    cases = make_cases([51001, 51002], ["capture", "pursuit"], [0, 1])
    assert len(cases) == 8
    assert cases[0] == {"seed": 51001, "style": "capture", "side": 0}
    with pytest.raises(ValueError):
        make_cases([1, 1], ["capture"], [0])
    with pytest.raises(ValueError):
        make_cases([1], ["unknown"], [0])


def test_frozen_gru_carry_and_reset_ablation_do_not_train():
    from rl_bt.temporal_policy import TemporalActorCritic
    torch.manual_seed(27)
    model = TemporalActorCritic(memory="gru")
    model.eval()
    before = model_weights_sha256(model)
    carried = FrozenPolicy("learned", model)
    reset = FrozenPolicy("learned", model, reset_each_step=True)
    obs = observation()
    for _ in range(3):
        assert 0 <= carried.act(obs) < 16
        assert 0 <= reset.act(obs) < 16
    assert carried.hidden_reset_count == 1
    assert reset.hidden_reset_count == 3
    assert model_weights_sha256(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())
    carried.reset()
    assert carried.steps == 0
    assert torch.count_nonzero(carried.hidden[0]) == 0


class FakeEnv:
    def __init__(self, runtime, seed, **kwargs):
        assert kwargs == {"randomization": True, "opponent_style": "capture",
                          "side": 0, "reward_profile": "upstream"}
        self.cached = observation()
        self.obs = np.zeros((6, 140), np.float32)
        self.obs[:, 7] = .7
        self.obs[:, 11] = 1
        self.obs[2, 21] = 1
        self.learner = 2
        self.dropout = np.asarray([True, False, False])
        self.episode_return = self.switches = self.overrides = self.recoveries = self.steps = 0
        self.terminate_after = runtime

    def state_dict(self):
        return {"cum_damage": np.ones((6, 6), np.float32), "score": np.asarray([3., 2.])}

    def step(self, action):
        self.steps += 1
        self.episode_return += 1.
        done = self.steps >= self.terminate_after
        info = {"duration_s": .5, "executed_candidate": action, "override_reason": None}
        if done:
            info["episode"] = {"return": self.episode_return, "decisions": self.steps,
                "switches": 0, "overrides": 0, "recoveries": 0, "win": True, "draw": False,
                "winner": 1, "outcome": 0, "score": [3., 2.]}
        return observation(), 1., done, info


def test_trial_truth_is_diagnostic_and_cutoff_is_not_draw():
    case = {"seed": 51001, "style": "capture", "side": 0}
    spec = {"label": "bt_reactive", "kind": "bt_reactive"}
    row, trace = run_trial(2, spec, case, max_decisions=3, save_trace=True, env_factory=FakeEnv)
    assert row["completed"] and row["win"] and row["actual_duration_s"] == 1.
    assert row["diagnostics"]["ammunition"] is None
    assert row["diagnostics"]["sampled_center_fraction"] == 1.
    assert not row["real_nav2_in_loop"] and len(trace["action"]) == 2
    cutoff, _ = run_trial(5, spec, case, max_decisions=2, env_factory=FakeEnv)
    assert cutoff["truncated"] and cutoff["win"] is None and cutoff["draw"] is None
    summary = summarize_policy([row, cutoff])
    assert summary["completed"] == 1 and summary["truncated"] == 1 and summary["wins"] == 1


def test_paired_contrast_requires_matching_cases_and_counts_completed_only():
    case = {"seed": 51001, "style": "capture", "side": 0}
    left, _ = run_trial(2, {"label": "left", "kind": "bt_reactive"}, case, env_factory=FakeEnv)
    right = copy.deepcopy(left)
    right.update(policy="right", win=False, draw=False)
    result = paired_contrast([left, right], "left", "right")
    assert result["matched_cases"] == result["completed_pairs"] == 1
    assert result["metrics"]["win"]["mean"] == 1.
    assert result["metrics"]["win"]["seed_bootstrap_95"] == [None, None]
    with pytest.raises(ValueError):
        paired_contrast([left], "left", "right")
    with pytest.raises(ValueError):
        paired_contrast([left, right, right], "left", "right")
