"""Optimizer isolation, minority coverage and exact continuation contracts."""
import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl_bt.uc_support_policy_v2 import UCSupportV2ActorCritic
from scripts.train_uc_support_v2 import (
    RESERVED_SEEDS, anchor_losses, balanced_indices, collect_rollout,
    effective_anchor_coefficients, fit_bc_actor, gae, load_checkpoint,
    make_model, next_training_seed, optimizer_rng_for_update, ppo_v2_update, records_sha256,
    save_checkpoint, source_snapshot, summarize_evaluation, tensor_bc_records,
    truncate_logs_to_checkpoint, weights_sha256,
)


@pytest.fixture(autouse=True)
def small_cpu_runtime():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def bc_records(count=64):
    rng = np.random.default_rng(17)
    return {"state": rng.normal(size=(count, 24)).astype(np.float32),
            "candidates": rng.normal(size=(count, 16, 12)).astype(np.float32),
            "mask": np.ones((count, 16), dtype=bool),
            "actions": np.resize(np.array([0, 13, 14, 15], np.int64), count)}


def test_gae_distinguishes_terminal_failure_from_external_truncation():
    advantages, returns = gae([1., 1., 1.], [2., 2., 2.], [3., 3., 3.],
        [True, False, False], [False, True, False], gamma=.9, lam=.8)
    np.testing.assert_allclose(advantages, [-1., 1.7, 1.7], atol=1e-6)
    np.testing.assert_allclose(returns, [1., 3.7, 3.7], atol=1e-6)


def test_minority_sampling_balances_semantic_groups_not_waypoint_indices():
    labels = np.array([0]*900 + [13] + [14]*90 + [15]*9)
    sampled = labels[balanced_indices(labels, 12000, np.random.default_rng(2))]
    for action in (0, 13, 14, 15):
        assert .23 < np.mean(sampled == action) < .27
    assert balanced_indices(np.array([14]), 4, np.random.default_rng(2)).tolist() == [0]*4


def test_bc_actor_fit_leaves_all_critic_weights_unchanged():
    torch.manual_seed(4)
    model = UCSupportV2ActorCritic()
    before = [p.detach().clone() for p in model.critic_parameters()]
    actor_before = weights_sha256(model.actor.state_dict())
    report = fit_bc_actor(model, bc_records(), seed=5, epochs=2)
    assert report["balanced_sampling"]
    assert all(report["training_label_accuracy_by_group"][g]["count"] == 16
               for g in ("navigate", "recover", "continue", "hold"))
    assert all(torch.equal(p, q) for p, q in zip(before, model.critic_parameters()))
    assert actor_before != weights_sha256(model.actor.state_dict())


def test_auxiliary_anchor_cannot_train_critic():
    torch.manual_seed(3)
    model, teacher = make_model()
    with torch.no_grad():
        next(model.actor.parameters()).add_(.01)
    kl, ce = anchor_losses(model, teacher, tensor_bc_records(bc_records()))
    assert kl >= 0 and torch.isfinite(ce)
    (kl + ce).backward()
    assert any(p.grad is not None for p in model.actor.parameters())
    assert all(p.grad is None for p in model.critic_parameters())
    assert all(p.grad is None for p in teacher.parameters())


class ToySupportEnv:
    """Stateful stochastic fixture; it is not a simulation result."""
    def __init__(self):
        self.reset(seed=2001)

    def reset(self, *, seed):
        self.rng = np.random.default_rng(seed)
        self.steps, self.total = 0, 0.
        self.obs = self._observation()
        return copy.deepcopy(self.obs)

    def _observation(self):
        return {"state": self.rng.normal(size=24).astype(np.float32),
                "candidates": self.rng.normal(size=(16, 12)).astype(np.float32),
                "mask": np.ones(16, dtype=bool)}

    def step(self, action):
        self.steps += 1
        reward = float(self.rng.normal() + .1*action)
        self.total += reward
        done = self.steps == 3
        self.obs = self._observation()
        return copy.deepcopy(self.obs), reward, done, {
            "terminated": done, "truncated": False, "reward_components": {"task": reward},
            "BT_veto": False, "episode": {"return": self.total, "success": self.total > 0,
                "steps": self.steps, "time_s": self.steps*.5, "recoveries": 0,
                "source_run_id": "synthetic_test_fixture"}}

    def state_dict(self):
        return {"rng": copy.deepcopy(self.rng.bit_generator.state), "steps": self.steps,
                "total": self.total, "obs": copy.deepcopy(self.obs)}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.steps, self.total = state["steps"], state["total"]
        self.obs = copy.deepcopy(state["obs"])


def optimizer_pair(model):
    return (torch.optim.Adam(model.actor.parameters(), lr=1e-4, eps=1e-5),
            torch.optim.Adam(model.critic_parameters(), lr=3e-4, eps=1e-5))


def checkpoint_metadata(records):
    return {"scenario_id": "synthetic_v2_resume_test", "algorithm": "anchored",
        "observation_schema": {"state": [24]}, "implementation_sha256": {"test": "fixed"},
        "dataset_sha256": {"test": "fixed"}, "bc_labels_sha256": records_sha256(records),
        "seed": 271, "rollout": 8, "epochs": 1,
        "anchor_kl_coefficient": .05, "anchor_ce_coefficient": .05}


def test_exact_resume_reproduces_next_rollout_and_both_optimizer_updates(tmp_path):
    torch.manual_seed(271)
    random.seed(271)
    np.random.seed(271)
    rng, env = np.random.default_rng(271), ToySupportEnv()
    model, teacher = make_model()
    actor_optimizer, critic_optimizer = optimizer_pair(model)
    records = bc_records()
    anchors = tensor_bc_records(records)
    first = collect_rollout(model, env, env.obs, rng, count=8, episode_seed=2001)
    ppo_v2_update(model, actor_optimizer, critic_optimizer, first[0], anchors,
                  teacher, rng, algorithm="anchored", epochs=1, minibatch_size=4)
    # A critic encoder missing from its optimizer is a silent training defect.
    assert all(p in critic_optimizer.state for p in model.critic_state_encoder.parameters())
    metadata, checkpoint = checkpoint_metadata(records), tmp_path/"latest.pt"
    save_checkpoint(checkpoint, model, teacher, actor_optimizer, critic_optimizer,
        metadata, rng, env, first[1], records, decisions=8, updates=1,
        episodes=first[3], episode_seed=first[2], wall_s=1.)
    reference = collect_rollout(model, env, first[1], rng, count=8, episode_seed=first[2])
    reference_update = ppo_v2_update(model, actor_optimizer, critic_optimizer,
        reference[0], anchors, teacher, rng, algorithm="anchored", epochs=1, minibatch_size=4)
    reference_state = copy.deepcopy(model.state_dict())
    reference_random = (np.random.random(), random.random(), torch.rand(1).item())
    restored_model, restored_teacher = make_model()
    restored_actor, restored_critic = optimizer_pair(restored_model)
    restored_rng, restored_env = np.random.default_rng(9), ToySupportEnv()
    payload = load_checkpoint(checkpoint, restored_model, restored_teacher,
        restored_actor, restored_critic, metadata, restored_rng, restored_env)
    replay = collect_rollout(restored_model, restored_env, payload["current_observation"],
        restored_rng, count=8, episode_seed=payload["episode_seed"])
    replay_update = ppo_v2_update(restored_model, restored_actor, restored_critic,
        replay[0], tensor_bc_records(payload["bc_records"]), restored_teacher,
        restored_rng, algorithm="anchored", epochs=1, minibatch_size=4)
    for key in reference[0]:
        assert torch.equal(reference[0][key], replay[0][key]), key
    assert reference[2:] == replay[2:]
    assert reference_update == replay_update
    assert all(torch.equal(value, restored_model.state_dict()[key])
               for key, value in reference_state.items())
    assert reference_random == (np.random.random(), random.random(), torch.rand(1).item())
    for optimizer in (restored_actor, restored_critic):
        assert optimizer.state
        assert all(torch.isfinite(value).all() for state in optimizer.state.values()
                   for value in state.values() if isinstance(value, torch.Tensor))


@pytest.mark.parametrize("key,value", [("algorithm", "vanilla"),
    ("scenario_id", "v1"), ("anchor_kl_coefficient", .1), ("epochs", 2)])
def test_resume_rejects_experiment_mismatch(tmp_path, key, value):
    model, teacher = make_model()
    actor, critic = optimizer_pair(model)
    records, rng, env = bc_records(), np.random.default_rng(4), ToySupportEnv()
    metadata = checkpoint_metadata(records)
    checkpoint = tmp_path/"checkpoint.pt"
    save_checkpoint(checkpoint, model, teacher, actor, critic, metadata, rng, env,
                    env.obs, records, decisions=0, updates=0, episodes=[], episode_seed=2001, wall_s=0.)
    with pytest.raises(ValueError, match=key):
        load_checkpoint(checkpoint, model, teacher, actor, critic,
                        {**metadata, key: value}, rng, env)


def test_vanilla_has_no_auxiliary_coefficients_and_source_records_schedule():
    args = SimpleNamespace(algorithm="vanilla", anchor_kl=.05, anchor_ce=.05)
    assert effective_anchor_coefficients(args) == (0., 0.)
    assert "configs/uc_support_training_schedule_v2.json" in source_snapshot()


def test_reserved_old_and_new_evaluation_seed_ranges_are_excluded():
    class Controlled:
        def __init__(self):
            self.values = iter([85001, 95064, 105001, 115064, 123456])
        def integers(self, low, high):
            return next(self.values)
    assert next_training_seed(Controlled()) == 123456
    assert not {next_training_seed(np.random.default_rng(i)) for i in range(1000)} & RESERVED_SEEDS


def test_optimizer_rng_is_reconstructible_and_does_not_consume_episode_stream():
    episode_rng = np.random.default_rng(271)
    before = copy.deepcopy(episode_rng.bit_generator.state)
    first = optimizer_rng_for_update(271, 8).integers(0, 100000, 100)
    second = optimizer_rng_for_update(271, 8).integers(0, 100000, 100)
    np.testing.assert_array_equal(first, second)
    assert before == episode_rng.bit_generator.state
    assert not np.array_equal(first, optimizer_rng_for_update(271, 9).integers(0, 100000, 100))


def test_resume_removes_only_unsaved_log_tail(tmp_path):
    (tmp_path/"train.jsonl").write_text('{"decisions": 8}\n{"decisions": 16}\n{')
    (tmp_path/"episodes.jsonl").write_text('{"id": 0}\n{"id": 1}\n')
    truncate_logs_to_checkpoint(tmp_path, decisions=8, episode_count=1)
    assert (tmp_path/"train.jsonl").read_text() == '{"decisions": 8}\n'
    assert (tmp_path/"episodes.jsonl").read_text() == '{"id": 0}\n'


def test_paired_evaluation_rejects_different_exogenous_descriptors():
    row = {"seed": 1, "source_run_id": "fixture", "parameters": {"schedule": 1},
           "success": True, "return": 1., "steps": 2, "score_100": 95.}
    other = {**row, "parameters": {"schedule": 2}}
    with pytest.raises(ValueError, match="exogenous schedules"):
        summarize_evaluation({"BT": [row], "PPO": [other]}, [1], "validation")
