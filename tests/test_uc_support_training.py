"""PPO time-limit and exact-resume contracts, independent of UC map availability."""
import copy
import random

import numpy as np
import pytest
import torch

from rl_bt.uc_support_policy import UCSupportActorCritic as ULCurriculumActorCritic
from scripts.train_uc_support import (
    RESERVED_SEEDS, TEST_SEEDS, VALIDATION_SEEDS, collect_rollout, gae,
    load_checkpoint, next_training_seed, save_checkpoint, summarize_evaluation,
    truncate_logs_to_checkpoint)
from scripts.train_ul_curriculum import ppo_update


def test_uc_gae_termination_truncation_and_rollout_cut():
    rewards = [1., 1., 1.]
    values = [2., 2., 2.]
    next_values = [3., 3., 3.]
    # A true terminal cannot bootstrap; the time-limit final observation can.
    advantage, returns = gae(rewards, values, next_values,
                            [True, False, False], [False, True, False], gamma=.9, lam=.8)
    np.testing.assert_allclose(advantage, [-1., 1.7, 1.7], atol=1e-6)
    np.testing.assert_allclose(returns, [1., 3.7, 3.7], atol=1e-6)
    # A rollout cut remains a continuing transition and bootstraps its final obs.
    advantage, _ = gae([1., 1.], [2., 2.], [3., 3.], [False, False],
                       [False, False], gamma=.9, lam=.8)
    np.testing.assert_allclose(advantage, [1.7 + .9 * .8 * 1.7, 1.7], atol=1e-6)


def test_uc_gae_rejects_misaligned_or_nonfinite_inputs():
    with pytest.raises(ValueError, match="identical"):
        gae([1., 2.], [1.], [1., 2.], [False, False], [False, False])
    with pytest.raises(ValueError, match="finite"):
        gae([np.nan], [0.], [0.], [False], [False])


def test_uc_training_episode_seeds_exclude_both_evaluation_ranges():
    class ControlledGenerator:
        def __init__(self):
            self.samples = iter([85001, 95064, 1234567])
        def integers(self, low, high):
            assert (low, high) == (1000, 2 ** 31)
            return next(self.samples)
    assert next_training_seed(ControlledGenerator()) == 1234567
    rng = np.random.default_rng(271)
    seeds = [next_training_seed(rng) for _ in range(1000)]
    assert all(1000 <= seed < 2 ** 31 for seed in seeds)
    assert not set(seeds) & RESERVED_SEEDS
    assert len(VALIDATION_SEEDS) == 32 and len(TEST_SEEDS) == 64
    assert not set(VALIDATION_SEEDS) & set(TEST_SEEDS)


class ToySupportEnv:
    """Stochastic, episode-resetting stand-in; no training result claim."""
    def __init__(self):
        self.reset(seed=2001)
    def reset(self, *, seed):
        self.rng = np.random.default_rng(seed)
        self.steps, self.total = 0, 0.
        self.obs = self._observation()
        return copy.deepcopy(self.obs)
    def _observation(self):
        mask = np.zeros(16, dtype=bool)
        mask[:4] = True
        return {"state": self.rng.normal(size=24).astype(np.float32),
                "candidates": self.rng.normal(size=(16, 12)).astype(np.float32),
                "mask": mask}
    def step(self, action):
        self.steps += 1
        reward = float(self.rng.normal() + .1 * action)
        self.total += reward
        done = self.steps == 3
        self.obs = self._observation()
        return copy.deepcopy(self.obs), reward, done, {
            "terminated": False, "truncated": done,
            "reward_components": {"task": reward}, "BT_veto": action == 3,
            "episode": {"return": self.total, "success": self.total > 0,
                        "steps": self.steps, "time_s": self.steps * .5,
                        "recoveries": 0, "source_run_id": "synthetic_test_only"}}
    def state_dict(self):
        return {"rng": copy.deepcopy(self.rng.bit_generator.state),
                "steps": self.steps, "total": self.total, "obs": copy.deepcopy(self.obs)}
    def load_state_dict(self, state):
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.steps, self.total = state["steps"], state["total"]
        self.obs = copy.deepcopy(state["obs"])


def checkpoint_metadata():
    return {"scenario_id": "synthetic_resume_test", "observation_schema": {"state": [24]},
            "implementation_sha256": {"test": "fixed"}, "dataset_sha256": {"test": "fixed"},
            "seed": 271, "rollout": 8, "epochs": 1, "eval_split": "none"}


def test_uc_checkpoint_restores_optimizer_rng_environment_and_current_observation(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(271)
    random.seed(271)
    np.random.seed(271)
    rng = np.random.default_rng(271)
    environment = ToySupportEnv()
    observation = copy.deepcopy(environment.obs)
    model = ULCurriculumActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    first = collect_rollout(model, environment, observation, rng, count=8, episode_seed=2001)
    batch, observation, episode_seed, episodes, _, _ = first
    ppo_update(model, optimizer, batch, rng, epochs=1, minibatch_size=4)
    checkpoint = tmp_path / "latest.pt"
    save_checkpoint(checkpoint, model, optimizer, checkpoint_metadata(), rng, environment,
                    observation, decisions=8, updates=1, episodes=episodes,
                    episode_seed=episode_seed, wall_s=1.)
    reference = collect_rollout(model, environment, observation, rng,
                                count=8, episode_seed=episode_seed)
    ppo_update(model, optimizer, reference[0], rng, epochs=1, minibatch_size=4)
    reference_weights = copy.deepcopy(model.state_dict())
    reference_numpy, reference_python = np.random.random(), random.random()
    restored_model = ULCurriculumActorCritic()
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=3e-4)
    restored_env = ToySupportEnv()
    restored_rng = np.random.default_rng(999)
    payload = load_checkpoint(checkpoint, restored_model, restored_optimizer,
                              checkpoint_metadata(), restored_rng, restored_env)
    restored = collect_rollout(restored_model, restored_env, payload["current_observation"],
                               restored_rng, count=8, episode_seed=payload["episode_seed"])
    for key in reference[0]:
        assert torch.equal(reference[0][key], restored[0][key]), key
    assert reference[2:] == restored[2:]
    ppo_update(restored_model, restored_optimizer, restored[0], restored_rng,
               epochs=1, minibatch_size=4)
    for key, value in reference_weights.items():
        assert torch.equal(value, restored_model.state_dict()[key]), key
    assert np.random.random() == reference_numpy
    assert random.random() == reference_python
    assert payload["decisions"] == 8 and payload["updates"] == 1


def test_uc_resume_rejects_source_or_dataset_change(tmp_path):
    model = ULCurriculumActorCritic()
    optimizer = torch.optim.Adam(model.parameters())
    environment = ToySupportEnv()
    rng = np.random.default_rng(271)
    checkpoint = tmp_path / "latest.pt"
    save_checkpoint(checkpoint, model, optimizer, checkpoint_metadata(), rng, environment,
        environment.obs, decisions=0, updates=0, episodes=[], episode_seed=2001, wall_s=0.)
    for key in ("scenario_id", "implementation_sha256", "dataset_sha256", "observation_schema"):
        metadata = checkpoint_metadata()
        metadata[key] = "different"
        with pytest.raises(ValueError, match=key):
            load_checkpoint(checkpoint, model, optimizer, metadata, rng, environment)


def test_uc_evaluation_compares_identical_source_runs():
    row = {"seed": 85001, "source_run_id": "run_a", "success": True,
           "return": 2., "steps": 3, "time_s": 1.5, "BT_veto_count": 0, "recoveries": 0}
    groups = {label: [copy.deepcopy(row)] for label in
              ("untrainedRL", "postBC_RL", "finalPPO_RL", "BT")}
    result = summarize_evaluation(groups, [85001], "validation", "uc_test")
    assert result["summary"]["BT"]["successes"] == 1
    assert "not_match_win_rate" in result["metric"]
    groups["BT"][0]["source_run_id"] = "different"
    with pytest.raises(ValueError, match="identical source"):
        summarize_evaluation(groups, [85001], "validation", "uc_test")


def test_uc_resume_removes_unsaved_metrics_tail_only(tmp_path):
    (tmp_path / "train.jsonl").write_text(
        '{"decisions": 8}\n{"decisions": 16}\n{"decisions":')
    (tmp_path / "episodes.jsonl").write_text('{"id": 0}\n{"id": 1}\n')
    truncate_logs_to_checkpoint(tmp_path, decisions=8, episode_count=1)
    assert (tmp_path / "train.jsonl").read_text() == '{"decisions": 8}\n'
    assert (tmp_path / "episodes.jsonl").read_text() == '{"id": 0}\n'
