"""Learning-mechanics tests only; these are not real navigation results."""
import copy

import numpy as np
import pytest
import torch

from rl_bt.live_actor_learning import (live_actor_eligibility, live_ppo_update,
    normalize_live_actor_advantages)
from rl_bt.uc_support_policy_v2 import UCSupportV2ActorCritic
from scripts.train_live_nav2 import duration_gae


def make_batch(model, *, kinds=("valid", "valid", "not_applied", "forced")):
    generator = torch.Generator().manual_seed(97)
    count = len(kinds)
    state = torch.randn(count, 24, generator=generator)
    candidates = torch.randn(count, 16, 12, generator=generator)
    mask = torch.zeros(count, 16, dtype=torch.bool)
    actions = torch.zeros(count, dtype=torch.long)
    applied = torch.ones(count, dtype=torch.bool)
    for index, kind in enumerate(kinds):
        if kind == "forced":
            mask[index, 15] = True
            actions[index] = 15
        else:
            mask[index, [0, 15]] = True
            actions[index] = index % 2 * 15
            applied[index] = kind != "not_applied"
    with torch.no_grad():
        distribution, values = model.distribution_value(state, candidates, mask)
    return {"state": state, "candidates": candidates, "mask": mask, "actions": actions,
        "action_applied": applied, "old_logp": distribution.log_prob(actions),
        "advantages": torch.arange(count, dtype=torch.float32) * 3 - 1,
        "returns": values + torch.arange(count, dtype=torch.float32) + 1,
        "elapsed_s": torch.linspace(.1, 1., count)}


def setup_models(seed=123):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    model = UCSupportV2ActorCritic()
    actor = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
    critic = torch.optim.Adam(model.critic_parameters(), lr=1e-3)
    return model, actor, critic


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def update(model, actor, critic, batch, **kwargs):
    return live_ppo_update(model, actor, critic, batch, kwargs.pop("bc_records", None),
        kwargs.pop("teacher", None), np.random.default_rng(13), algorithm=kwargs.pop("algorithm", "vanilla"),
        epochs=kwargs.pop("epochs", 1), minibatch_size=32, **kwargs)


def test_actor_eligibility_requires_applied_and_multiple_legal_choices():
    mask = torch.tensor([[True, True], [True, True], [True, False]])
    applied = torch.tensor([True, False, True])
    assert live_actor_eligibility(mask, applied).tolist() == [True, False, False]
    with pytest.raises(ValueError, match="explicit"):
        live_actor_eligibility(mask, None)
    with pytest.raises(ValueError, match="explicit"):
        live_actor_eligibility(mask, applied.float())
    with pytest.raises(ValueError, match="fallback"):
        live_actor_eligibility(torch.zeros(1, 2, dtype=torch.bool), torch.tensor([True]))


def test_advantage_normalization_ignores_forced_and_nonapplied_extremes():
    advantages = torch.tensor([1., 3., 10000., -50000.])
    valid = torch.tensor([True, True, False, False])
    original = advantages.clone()
    normalized = normalize_live_actor_advantages(advantages, valid)
    torch.testing.assert_close(normalized, torch.tensor([-1., 1., 0., 0.]))
    assert torch.equal(advantages, original)
    single = normalize_live_actor_advantages(advantages, torch.tensor([True, False, False, False]))
    assert single.tolist() == [1., 0., 0., 0.]


def test_nonapplied_and_forced_samples_do_not_change_actor_loss_or_entropy():
    model_a, actor_a, critic_a = setup_models()
    model_b, actor_b, critic_b = setup_models()
    batch_a = make_batch(model_a)
    batch_b = copy.deepcopy(batch_a)
    batch_b["state"][2:] *= -100
    batch_b["candidates"][2:] *= 50
    batch_b["advantages"][2:] = torch.tensor([10000., -10000.])
    batch_b["old_logp"][2:] = torch.tensor([-10., 0.])
    batch_b["returns"][2:] = 1000
    stats_a = update(model_a, actor_a, critic_a, batch_a)
    stats_b = update(model_b, actor_b, critic_b, batch_b)
    assert stats_a["on_policy_actor_sample_count"] == 2
    assert stats_a["actor_excluded_not_applied_count"] == 1
    assert stats_a["actor_excluded_applied_single_choice_count"] == 1
    for key in ("policy_loss", "entropy", "on_policy_actor_gradient_norm", "post_update_approx_kl_old_policy"):
        assert stats_a[key] == pytest.approx(stats_b[key], abs=1e-9)
    assert_tree_equal(model_a.actor.state_dict(), model_b.actor.state_dict())


def test_all_forced_critic_updates_skip_actor_step_even_with_existing_adam_momentum():
    model, actor, critic = setup_models()
    update(model, actor, critic, make_batch(model, kinds=("valid", "valid")))
    assert actor.state_dict()["state"]
    before_actor = copy.deepcopy(model.actor.state_dict())
    before_optimizer = copy.deepcopy(actor.state_dict())
    before_critic = [parameter.detach().clone() for parameter in model.critic_parameters()]
    forced = make_batch(model, kinds=("forced",) * 4)
    original = copy.deepcopy(forced)
    stats = update(model, actor, critic, forced, epochs=2)
    assert stats["actor_optimizer_steps"] == 0
    assert stats["critic_optimizer_steps"] == 2
    assert stats["on_policy_actor_sample_count"] == 0 and stats["critic_sample_count"] == 4
    assert stats["policy_loss"] == stats["entropy"] == stats["on_policy_actor_gradient_norm"] == 0
    assert_tree_equal(before_actor, model.actor.state_dict())
    assert_tree_equal(before_optimizer, actor.state_dict())
    assert any(not torch.equal(before, after) for before, after in zip(before_critic, model.critic_parameters()))
    assert_tree_equal(original, forced)


def test_nonapplied_multiple_choice_terminal_keeps_critic_but_no_actor_update():
    model, actor, critic = setup_models()
    before = copy.deepcopy(model.actor.state_dict())
    batch = make_batch(model, kinds=("not_applied", "not_applied"))
    stats = update(model, actor, critic, batch)
    assert stats["actor_excluded_not_applied_count"] == stats["critic_sample_count"] == 2
    assert stats["actor_optimizer_steps"] == stats["on_policy_actor_sample_count"] == 0
    assert stats["critic_gradient_norm"] > 0 and stats["value_loss"] > 0
    assert_tree_equal(before, model.actor.state_dict())


def test_single_valid_choice_retains_policy_gradient_instead_of_centering_to_zero():
    model, actor, critic = setup_models()
    batch = make_batch(model, kinds=("valid",))
    batch["advantages"][:] = 2.
    stats = update(model, actor, critic, batch, entropy_coefficient=0)
    assert stats["on_policy_actor_sample_count"] == 1
    assert stats["on_policy_actor_gradient_norm"] > 0
    assert stats["policy_loss"] == pytest.approx(-2.)


def test_anchor_only_update_is_explicit_train_only_and_logged_separately():
    model, actor, critic = setup_models()
    teacher = copy.deepcopy(model)
    teacher.requires_grad_(False)
    anchor = make_batch(model, kinds=("valid", "valid", "forced"))
    records = {key: anchor[key] for key in ("state", "candidates", "mask", "actions")}
    forced = make_batch(model, kinds=("forced", "forced"))
    with pytest.raises(ValueError, match="never evaluation"):
        update(model, actor, critic, forced, algorithm="anchored", bc_records=records,
               teacher=teacher, anchor_data_role="evaluation")
    stats = update(model, actor, critic, forced, algorithm="anchored", bc_records=records,
                   teacher=teacher, anchor_data_role="train_demonstration")
    assert stats["on_policy_actor_sample_count"] == stats["on_policy_actor_gradient_norm"] == 0
    assert stats["anchor_actor_gradient_norm"] > 0
    assert stats["actor_optimizer_steps"] == stats["anchor_only_actor_optimizer_steps"] == 1
    assert stats["on_policy_actor_optimizer_steps"] == 0
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert stats["anchor_data_role"] == "train_demonstration"


def test_uninformative_anchors_skip_actor_and_unfrozen_teacher_is_rejected():
    model, actor, critic = setup_models()
    forced = make_batch(model, kinds=("forced", "forced"))
    records = {key: forced[key] for key in ("state", "candidates", "mask", "actions")}
    teacher = copy.deepcopy(model)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="frozen"):
        update(model, actor, critic, forced, algorithm="anchored", bc_records=records,
               teacher=teacher, anchor_data_role="train_demonstration")
    assert_tree_equal(before, model.state_dict())
    teacher.requires_grad_(False)
    stats = update(model, actor, critic, forced, algorithm="anchored", bc_records=records,
                   teacher=teacher, anchor_data_role="train_demonstration")
    assert stats["anchor_informative_records"] == 0
    assert stats["actor_optimizer_steps"] == stats["anchor_actor_gradient_norm"] == 0
    assert stats["critic_optimizer_steps"] == 1
    assert_tree_equal({key: value for key, value in before.items() if key.startswith("actor.")},
                      {key: value for key, value in model.state_dict().items() if key.startswith("actor.")})


def test_duration_gae_retains_nonapplied_terminal_reward_and_all_critic_targets():
    # First actual choice receives delayed credit through forced/non-applied
    # terminal feedback. Actor eligibility must not filter this GAE sequence.
    advantages, returns = duration_gae([0., 0., -2.], [0., 0., 0.], [0., 0., 0.],
        [False, False, True], [False, False, False], [.25, .5, .0], gamma=.81, lam=.64)
    expected = torch.tensor([-2 * .72 * .5184, -2 * .5184, -2.])
    torch.testing.assert_close(torch.from_numpy(advantages), expected)
    model, actor, critic = setup_models()
    batch = make_batch(model, kinds=("valid", "forced", "not_applied"))
    batch["advantages"] = torch.from_numpy(advantages)
    batch["returns"] = torch.from_numpy(returns)
    before = copy.deepcopy(batch)
    stats = update(model, actor, critic, batch, entropy_coefficient=0)
    assert stats["critic_sample_count"] == 3 and stats["on_policy_actor_sample_count"] == 1
    assert stats["on_policy_actor_gradient_norm"] > 0
    assert_tree_equal(before, batch)


def test_missing_application_flag_fails_closed_before_updating_parameters():
    model, actor, critic = setup_models()
    batch = make_batch(model)
    del batch["action_applied"]
    initial = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="action_applied"):
        update(model, actor, critic, batch)
    assert_tree_equal(initial, model.state_dict())
