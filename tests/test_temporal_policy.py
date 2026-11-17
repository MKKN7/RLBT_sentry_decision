"""Observation legality, recurrent history and actor/value isolation contracts."""
import copy

import pytest
import torch

from rl_bt.temporal_policy import TemporalActorCritic


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observation(batch=2, count=16):
    generator = torch.Generator().manual_seed(817)
    return {"state": torch.randn(batch, 148, generator=generator),
            "candidates": torch.randn(batch, count, 16, generator=generator),
            "mask": torch.ones(batch, count, dtype=torch.bool)}


@pytest.mark.parametrize("memory", ["mlp", "gru"])
def test_common_interface_masks_illegal_actions_and_has_finite_outputs(memory):
    model = TemporalActorCritic(memory=memory)
    data = observation()
    data["mask"][0, 1:4] = False
    data["mask"][1, 1:] = False
    initial = model.initial_state(2)
    distribution, value, hidden = model.distribution_value(**data, hidden=initial)
    assert value.shape == (2,)
    assert all(item.shape == (2, 64) for item in hidden)
    assert torch.isfinite(distribution.logits).all()
    assert torch.isfinite(value).all()
    assert torch.equal(distribution.probs[~data["mask"]], torch.zeros_like(distribution.probs[~data["mask"]]))
    assert distribution.probs[1, 0] == 1.
    if memory == "mlp":
        assert all(torch.count_nonzero(item) == 0 for item in hidden)
    else:
        assert all(torch.count_nonzero(item) > 0 for item in hidden)


@pytest.mark.parametrize("memory", ["mlp", "gru"])
def test_candidate_permutation_preserves_policy_and_value(memory):
    model = TemporalActorCritic(memory=memory)
    data = observation()
    data["mask"][:, 4:8] = False
    permutation = torch.randperm(16, generator=torch.Generator().manual_seed(83))
    distribution, value, hidden = model.distribution_value(**data)
    permuted, other_value, other_hidden = model.distribution_value(
        data["state"], data["candidates"][:, permutation], data["mask"][:, permutation])
    torch.testing.assert_close(permuted.probs, distribution.probs[:, permutation])
    torch.testing.assert_close(other_value, value)
    for actual, expected in zip(other_hidden, hidden):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("memory", ["mlp", "gru"])
def test_illegal_candidate_values_do_not_change_legal_outputs(memory):
    model = TemporalActorCritic(memory=memory)
    data = observation()
    data["mask"][:, 4:8] = False
    distribution, value, hidden = model.distribution_value(**data)
    data["candidates"][:, 4:8] = 1000.
    changed, changed_value, changed_hidden = model.distribution_value(**data)
    torch.testing.assert_close(changed.probs, distribution.probs)
    torch.testing.assert_close(changed_value, value)
    for actual, expected in zip(changed_hidden, hidden):
        torch.testing.assert_close(actual, expected)


def test_gru_uses_history_and_episode_reset_reproduces_fresh_output():
    torch.manual_seed(19)
    model = TemporalActorCritic(memory="gru")
    data = observation()
    fresh, fresh_value, _ = model.distribution_value(**data)
    prior = observation()
    prior["state"] = -prior["state"]
    _, _, previous_hidden = model.distribution_value(**prior)
    remembered, remembered_value, _ = model.distribution_value(**data, hidden=previous_hidden)
    assert not torch.allclose(remembered.probs, fresh.probs)
    assert not torch.allclose(remembered_value, fresh_value)
    reset, reset_value, reset_hidden = model.distribution_value(
        **data, hidden=previous_hidden, episode_start=torch.tensor([True, False]))
    torch.testing.assert_close(reset.probs[0], fresh.probs[0])
    torch.testing.assert_close(reset_value[0], fresh_value[0])
    torch.testing.assert_close(reset.probs[1], remembered.probs[1])
    torch.testing.assert_close(reset_value[1], remembered_value[1])
    assert all(torch.isfinite(item).all() for item in reset_hidden)


def test_gru_backpropagates_through_history_but_not_across_reset():
    model = TemporalActorCritic(memory="gru")
    first = observation()
    first["state"].requires_grad_()
    _, _, hidden = model.distribution_value(**first)
    second = observation()
    distribution, _, _ = model.distribution_value(**second, hidden=hidden)
    distribution.log_prob(torch.tensor([1, 2])).sum().backward()
    assert first["state"].grad is not None
    assert first["state"].grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    first["state"].grad = None
    _, _, hidden = model.distribution_value(**first)
    distribution, _, _ = model.distribution_value(
        **second, hidden=hidden, episode_start=torch.ones(2, dtype=torch.bool))
    distribution.log_prob(torch.tensor([1, 2])).sum().backward()
    assert first["state"].grad is not None
    assert first["state"].grad.abs().sum() == 0


def test_mlp_ignores_history_and_gru_batch_processing_matches_individual_rows():
    data = observation()
    model = TemporalActorCritic(memory="mlp", hidden_size=24)
    fresh, value, _ = model.distribution_value(**data)
    supplied = tuple(torch.randn(2, 24) for _ in range(2))
    remembered, remembered_value, _ = model.distribution_value(**data, hidden=supplied)
    torch.testing.assert_close(remembered.probs, fresh.probs)
    torch.testing.assert_close(remembered_value, value)
    recurrent = TemporalActorCritic(memory="gru", hidden_size=24)
    batched, value, hidden = recurrent.distribution_value(**data, hidden=supplied)
    for row in range(2):
        individual, row_value, row_hidden = recurrent.distribution_value(
            **{key: item[row:row+1] for key, item in data.items()},
            hidden=tuple(item[row:row+1] for item in supplied))
        torch.testing.assert_close(individual.probs[0], batched.probs[row])
        torch.testing.assert_close(row_value[0], value[row])
        for actual, expected in zip(row_hidden, hidden):
            torch.testing.assert_close(actual[0], expected[row])


@pytest.mark.parametrize("memory", ["mlp", "gru"])
def test_actor_critic_gradients_and_optimizers_are_disjoint(memory):
    torch.manual_seed(381)
    model = TemporalActorCritic(memory=memory)
    data = observation()
    actor_ids = {id(item) for item in model.actor_parameters()}
    critic_ids = {id(item) for item in model.critic_parameters()}
    assert not actor_ids & critic_ids
    assert actor_ids | critic_ids == {id(item) for item in model.parameters()}
    actor_before = copy.deepcopy(model.actor.state_dict())
    distribution, value, _ = model.distribution_value(**data)
    expected_logits = distribution.logits.detach().clone()
    critic_optimizer = torch.optim.Adam(model.critic_parameters(), lr=.01)
    value.square().sum().backward()
    assert all(item.grad is None for item in model.actor_parameters())
    critic_optimizer.step()
    actual_logits = model.distribution_value(**data)[0].logits
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    for key, actual in model.actor.state_dict().items():
        torch.testing.assert_close(actual, actor_before[key], rtol=0, atol=0)
    model.zero_grad(set_to_none=True)
    model.distribution_value(**data)[0].log_prob(torch.tensor([1, 2])).sum().backward()
    assert all(item.grad is None for item in model.critic_parameters())


@pytest.mark.parametrize("mutation,exception", [
    (lambda data: data.update(state=data["state"][:, :-1]), ValueError),
    (lambda data: data.update(candidates=data["candidates"][:, :, :-1]), ValueError),
    (lambda data: data.update(mask=data["mask"].float()), TypeError),
    (lambda data: data["mask"][0].fill_(False), ValueError),
    (lambda data: data["state"][0, 0].fill_(float("nan")), ValueError),
    (lambda data: data.update(candidates=data["candidates"].double()), TypeError),
])
def test_invalid_observations_fail_explicitly(mutation, exception):
    data = observation()
    mutation(data)
    with pytest.raises(exception):
        TemporalActorCritic().distribution_value(**data)


def test_invalid_configuration_hidden_and_boundary_shapes_fail_explicitly():
    with pytest.raises(ValueError):
        TemporalActorCritic(memory="transformer")
    with pytest.raises(ValueError):
        TemporalActorCritic(hidden_size=0)
    model = TemporalActorCritic(memory="gru")
    with pytest.raises(ValueError):
        model.initial_state(0)
    with pytest.raises(ValueError):
        model.distribution_value(**observation(), hidden=torch.zeros(2, 64))
    with pytest.raises(ValueError):
        model.distribution_value(**observation(), hidden=(torch.zeros(1, 64), torch.zeros(2, 64)))
    with pytest.raises(ValueError):
        model.distribution_value(**observation(), episode_start=torch.ones(1, dtype=torch.bool))
    with pytest.raises(TypeError):
        model.distribution_value(**observation(), episode_start=torch.ones(2))
