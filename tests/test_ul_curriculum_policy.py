import copy

import numpy as np
import pytest
import torch

from rl_bt.ul_curriculum_policy import (
    ULCurriculumActorCritic, choose, tensor_observations)
from scripts.train_ul_curriculum import gae, ppo_update, summarize_evaluation


def observations(count=4):
    rng = np.random.default_rng(17)
    result = []
    for _ in range(count):
        mask = rng.random(16) > .4
        mask[0] = True
        mask[5] = False
        result.append({"state": rng.normal(size=24).astype(np.float32),
                       "candidates": rng.normal(size=(16, 12)).astype(np.float32), "mask": mask})
    return result


def test_masked_candidates_have_zero_probability_and_permutation_equivariance():
    torch.manual_seed(27)
    model = ULCurriculumActorCritic()
    batch = tensor_observations(observations())
    distribution, values = model.distribution_value(**batch)
    assert torch.isfinite(distribution.logits).all()
    assert torch.isfinite(values).all()
    assert (distribution.probs[~batch["mask"]] == 0).all()
    samples = distribution.sample((128,))
    assert batch["mask"].gather(1, samples.T).all()
    order = torch.randperm(16)
    first = model.actor(**batch)
    second = model.actor(batch["state"], batch["candidates"][:, order], batch["mask"][:, order])
    torch.testing.assert_close(first[:, order], second)
    assert observations()[0]["mask"][choose(model, observations()[0])]
    assert sum(parameter.numel() for parameter in model.actor.parameters()) == 12449
    assert sum(parameter.numel() for parameter in model.critic.parameters()) == 4225


def test_empty_mask_and_mismatched_schema_are_rejected():
    model = ULCurriculumActorCritic()
    batch = tensor_observations(observations(1))
    batch["mask"][:] = False
    with pytest.raises(ValueError, match="legal fallback"):
        model.distribution_value(**batch)
    wrong = observations(1)
    wrong[0]["state"] = np.zeros(148, np.float32)
    with pytest.raises(ValueError, match="state shape"):
        tensor_observations(wrong)


def test_ppo_update_is_finite_and_changes_actor_and_critic():
    torch.manual_seed(31)
    model = ULCurriculumActorCritic()
    batch = tensor_observations(observations(32))
    with torch.no_grad():
        distribution, values = model.distribution_value(**batch)
        batch["actions"] = distribution.sample()
        batch["old_logp"] = distribution.log_prob(batch["actions"])
    batch["advantages"] = torch.linspace(-1., 1., 32)
    batch["returns"] = values + torch.linspace(.1, 1., 32)
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    metrics = ppo_update(model, optimizer, batch, np.random.default_rng(31), epochs=2, minibatch_size=16)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["value_loss"] > 0
    assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items()
               if name.startswith("actor."))
    assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items()
               if name.startswith("critic."))


def test_gae_terminal_reset_and_rollout_cut_are_distinct():
    advantage, returns = gae([1., 100.], [0., 0.], [999., 999.], [True, True])
    np.testing.assert_array_equal(advantage, [1., 100.])
    np.testing.assert_array_equal(returns, advantage)
    advantage, _ = gae([1.], [2.], [5.], [False], gamma=.9)
    np.testing.assert_allclose(advantage, [1. + .9 * 5. - 2.])


def test_evaluation_uses_paired_task_labels_and_never_calls_them_match_wins():
    rows = [{"seed": seed, "side": seed % 2,
             "objective": ("capture", "resupply", "recover_then_capture")[(seed // 2) % 3],
             "success": True, "return": 1., "steps": 2} for seed in range(6)]
    result = summarize_evaluation(rows, rows, rows, range(6))
    assert result["covers_two_sides_and_three_objectives"]
    assert result["summary"]["BT"]["task_success_rate"] == 1.
    assert "not_match_win_rate" in result["metric"]
    mismatch = copy.deepcopy(rows)
    mismatch[0]["side"] = 1
    with pytest.raises(ValueError, match="same side and objective"):
        summarize_evaluation(rows, mismatch, rows, range(6))
