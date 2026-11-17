"""Temporal PPO checks use small deterministic mocks; no match claims."""
import copy

import numpy as np
import pytest
import torch

from rl_bt.temporal_policy import TemporalActorCritic
from scripts.train_temporal_phase import (
    collect_episode, episode_gae, evaluate_sequence, observation_tensors,
    parse_args, ppo_update, load_temporal_checkpoint, SCENARIO_ID,
    CHECKPOINT_SCHEMA, OBSERVATION_SCHEMA, CONTRACT_SOURCES, MANIFEST,
    ROOT, file_sha256,
)
from rl_bt.alpharm_task_env import SCENARIO, TRAIN_STYLES


def observation(step=0, *, forced=False):
    state = np.zeros(148, np.float32)
    state[0] = step / 5
    state[7] = .7
    candidates = np.zeros((16, 16), np.float32)
    candidates[:, 0] = np.linspace(-1, 1, 16)
    candidates[:, 1] = step / 9
    mask = np.zeros(16, bool)
    mask[:1 if forced else 4] = True
    return {"state": state, "candidates": candidates, "mask": mask}


class FakeEpisode:
    def __init__(self, terminal_after=4, *, forced=False, style="capture"):
        self.done = False
        self.decisions = 0
        self.style = style
        self.team = 0
        self.cached = observation(forced=forced)
        self.terminal_after = terminal_after
        self.forced = forced

    def step(self, action):
        assert self.cached["mask"][action]
        self.decisions += 1
        self.done = self.decisions >= self.terminal_after
        self.cached = observation(self.decisions, forced=self.forced)
        info = {"duration_s": .5, "executed_candidate": action, "override_reason": None}
        if self.done:
            info["episode"] = {"win": False, "return": 1., "decisions": self.decisions}
        return self.cached, .1 * self.decisions, self.done, info


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("memory", ["mlp", "gru"])
def test_collection_and_sequence_recomputation_have_exact_onpolicy_logp(memory):
    torch.manual_seed(8)
    model = TemporalActorCritic(memory=memory)
    sequence = collect_episode(model, FakeEpisode(5), 8, 17)
    assert len(sequence["actions"]) == 5
    assert sequence["terminated"].tolist() == [False] * 4 + [True]
    assert not sequence["truncated"].any()
    assert sequence["bootstrap_value"] == 0.
    logp, values, _ = evaluate_sequence(model, sequence)
    np.testing.assert_allclose(logp.detach(), sequence["old_logp"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(values.detach(), sequence["values"], rtol=1e-6, atol=1e-6)
    assert sequence["opponent_style"] == "capture"


def test_budget_truncation_is_explicit_and_bootstraps_recurrent_next_observation():
    torch.manual_seed(9)
    model = TemporalActorCritic(memory="gru")
    sequence = collect_episode(model, FakeEpisode(20), 3, 19)
    assert sequence["truncated"].tolist() == [False, False, True]
    assert not sequence["terminated"].any()
    assert sequence["episode"] is None
    assert sequence["truncation_reason"] == "episode_decision_limit"
    hidden = model.initial_state(1)
    with torch.no_grad():
        for i in range(4):
            _, value, hidden = model.distribution_value(**observation_tensors([observation(i)]), hidden=hidden)
    assert sequence["bootstrap_value"] == pytest.approx(float(value))


def test_duration_gae_respects_true_terminal_and_final_external_truncation():
    sequence = {"rewards": [0., 1.], "values": [2., 3.], "duration_s": [.5, 2.],
        "terminated": [False, True], "truncated": [False, False], "bootstrap_value": 100.}
    adv, ret = episode_gae(sequence, gamma_per_second=.9, lambda_per_second=.8)
    expected_last = 1 - 3
    expected_first = .9**.5 * 3 - 2 + .9**.5 * .8**.5 * expected_last
    np.testing.assert_allclose(adv, [expected_first, expected_last], rtol=1e-6)
    np.testing.assert_allclose(ret, adv + [2., 3.], rtol=1e-6)
    sequence["terminated"][-1] = False
    sequence["truncated"][-1] = True
    truncated, _ = episode_gae(sequence, gamma_per_second=.9, lambda_per_second=.8)
    assert truncated[-1] == pytest.approx(1 + .9**2 * 100 - 3)


@pytest.mark.parametrize("bad", ["early_terminal", "no_end", "negative_duration", "nonfinite"])
def test_gae_rejects_invalid_episode_sequences(bad):
    sequence = {"rewards": [0., 1.], "values": [0., 0.], "duration_s": [.5, .5],
        "terminated": [False, True], "truncated": [False, False], "bootstrap_value": 0.}
    if bad == "early_terminal":
        sequence["terminated"][0] = True
    elif bad == "no_end":
        sequence["terminated"][-1] = False
    elif bad == "negative_duration":
        sequence["duration_s"][0] = -.1
    else:
        sequence["rewards"][0] = float("nan")
    with pytest.raises(ValueError):
        episode_gae(sequence)


def test_collection_rejects_heldout_style_before_any_step():
    with pytest.raises(ValueError, match="Held-out"):
        collect_episode(TemporalActorCritic(memory="gru"), FakeEpisode(style="pursuit"), 5, 3)


@pytest.mark.parametrize("memory", ["mlp", "gru"])
def test_finite_ppo_updates_and_independent_critic_actor_parameters(memory):
    torch.manual_seed(31)
    model = TemporalActorCritic(memory=memory)
    sequences = [collect_episode(model, FakeEpisode(5), 6, 1),
                 collect_episode(model, FakeEpisode(4), 6, 2)]
    actor = [p.detach().clone() for p in model.actor.parameters()]
    critic = [p.detach().clone() for p in model.critic.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    metrics = ppo_update(model, optimizer, sequences, epochs=2)
    assert metrics["decisions"] == metrics["actor_decisions"] == 9
    assert metrics["first_epoch_approx_kl"] < 1e-6
    assert metrics["individual_timesteps_shuffled"] is False
    assert any(not torch.equal(before, after) for before, after in zip(actor, model.actor.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(critic, model.critic.parameters()))
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())


def test_all_forced_batch_preserves_actor_even_with_existing_adam_momentum():
    torch.manual_seed(27)
    model = TemporalActorCritic(memory="gru")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    normal = collect_episode(model, FakeEpisode(4), 8, 1)
    ppo_update(model, optimizer, [normal], epochs=2)
    before = [p.detach().clone() for p in model.actor.parameters()]
    critic = [p.detach().clone() for p in model.critic.parameters()]
    forced = collect_episode(model, FakeEpisode(4, forced=True), 8, 2)
    metrics = ppo_update(model, optimizer, [forced], epochs=2)
    assert metrics["actor_decisions"] == 0
    assert metrics["forced_single_legal_decisions"] == 4
    assert all(torch.equal(x, y) for x, y in zip(before, model.actor.parameters()))
    assert any(not torch.equal(x, y) for x, y in zip(critic, model.critic.parameters()))


def test_sequence_final_loss_backpropagates_through_prior_observations():
    torch.manual_seed(73)
    model = TemporalActorCritic(memory="gru")
    sequence = collect_episode(model, FakeEpisode(3), 8, 1)
    inputs = []
    handle = model.actor.observation_encoder.state_encoder[0].register_forward_pre_hook(
        lambda module, args: inputs.append(args[0]))
    # Compare gradients after changing only earlier observations, while all current
    # final observations and sampled final action are fixed.
    logp, _, _ = evaluate_sequence(model, sequence)
    logp[-1].backward()
    first_grad = model.actor.memory_context.core.weight_hh.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    changed = copy.deepcopy(sequence)
    changed["state"][:-1, 0] += 2.
    logp, _, _ = evaluate_sequence(model, changed)
    logp[-1].backward()
    handle.remove()
    assert not torch.allclose(first_grad, model.actor.memory_context.core.weight_hh.grad)


def test_observation_validation_and_cli():
    observation_tensors([observation()])
    broken = observation()
    broken["mask"] = broken["mask"].astype(np.float32)
    with pytest.raises(ValueError, match="boolean"):
        observation_tensors([broken])
    args = parse_args(["--run", "/tmp/only-parse", "--memory", "gru", "--decisions", "4096",
                       "--episodes-per-update", "4", "--epochs", "2", "--seed", "271"])
    assert args.decisions == 4096 and args.gamma_per_second == .997 and args.lambda_per_second == .97


def test_checkpoint_contract_rejects_changed_source_or_false_live_labels(tmp_path):
    model = TemporalActorCritic(memory="gru")
    metadata = {"environment_scenario_id": SCENARIO, "observation_schema": OBSERVATION_SCHEMA,
        "real_nav2_in_loop": False, "official_UL_UC_training": False,
        "train_styles": list(TRAIN_STYLES), "memory": "gru", "hidden_size": 64,
        "implementation_sha256": {r: file_sha256(ROOT / r) for r in CONTRACT_SOURCES},
        "upstream_manifest_sha256": file_sha256(ROOT / MANIFEST)}
    payload = {"checkpoint_schema": CHECKPOINT_SCHEMA, "scenario_id": SCENARIO_ID,
               "metadata": metadata, "model": model.state_dict()}
    path = tmp_path / "frozen.pt"
    torch.save(payload, path)
    loaded, recovered = load_temporal_checkpoint(path)
    assert loaded.memory == "gru" and recovered["scenario_id"] == SCENARIO_ID
    metadata["real_nav2_in_loop"] = True
    torch.save(payload, path)
    with pytest.raises(ValueError, match="real/official"):
        load_temporal_checkpoint(path)
    metadata["real_nav2_in_loop"] = False
    metadata["implementation_sha256"][CONTRACT_SOURCES[0]] = "bad"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="source mismatch"):
        load_temporal_checkpoint(path)
