"""PPO bookkeeping for live actions that may lose admission between RPCs.

Every actual transition remains in the duration-aware GAE and critic batch.
Only applied actions with a genuine choice contribute an on-policy actor loss.
The caller owns live provenance checks and may pass only verified *training*
demonstrations as anchors; this module never reads evaluation records.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F


ACTION_GROUPS = {"navigate": tuple(range(13)), "recover": (13,),
                 "continue": (14,), "hold": (15,)}


def live_actor_eligibility(mask, action_applied):
    """Require explicit Boolean admission flags, preserving all critic rows."""
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or \
            mask.ndim != 2 or mask.shape[1] < 1:
        raise ValueError("Live actor mask must be a Boolean [N, A] tensor")
    if not isinstance(action_applied, torch.Tensor) or action_applied.dtype != torch.bool or \
            action_applied.shape != mask.shape[:1] or action_applied.device != mask.device:
        raise ValueError("Live action_applied must be an explicit matching Boolean tensor")
    if not bool(mask.any(-1).all()):
        raise ValueError("Every live observation needs a legal fallback")
    return action_applied & (mask.sum(-1) > 1)


def normalize_live_actor_advantages(advantages, eligible):
    """Normalize choice rows only; leave a lone choice's advantage unchanged.

    A single choice has no population variance. Centering it would erase its
    policy-gradient signal, so its original GAE advantage is retained instead.
    Invalid actor rows are zeroed without editing GAE returns or critic targets.
    """
    if not isinstance(advantages, torch.Tensor) or advantages.ndim != 1 or \
            not bool(torch.isfinite(advantages).all()):
        raise ValueError("Live actor advantages must be a finite vector")
    if not isinstance(eligible, torch.Tensor) or eligible.dtype != torch.bool or \
            eligible.shape != advantages.shape or eligible.device != advantages.device:
        raise ValueError("Actor eligibility must match the advantage vector")
    output = torch.zeros_like(advantages, dtype=torch.float32)
    selected = advantages[eligible].float()
    if len(selected) > 1:
        selected = (selected - selected.mean()) / (selected.std(unbiased=False) + 1e-8)
    output[eligible] = selected
    return output


def _gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    total = sum(float(gradient.detach().double().square().sum())
                for gradient in gradients if gradient is not None)
    return math.sqrt(total)


def _anchor_indices(records, count, rng):
    """Balance represented train-label groups among informative BC rows."""
    actions = records["actions"].detach().cpu().numpy()
    choices = (records["mask"].sum(-1) > 1).detach().cpu().numpy()
    groups = [np.flatnonzero(choices & np.isin(actions, ids))
              for ids in ACTION_GROUPS.values()]
    groups = [indices for indices in groups if len(indices)]
    if not groups:
        return None
    indices = []
    for _ in range(count):
        group = groups[int(rng.integers(len(groups)))]
        indices.append(int(group[int(rng.integers(len(group)))]))
    return torch.as_tensor(indices, dtype=torch.long, device=records["actions"].device)


def _anchor_loss(model, teacher, records, indices, kl_coefficient, ce_coefficient):
    state, candidates, mask = (records[key][indices] for key in ("state", "candidates", "mask"))
    student_logits = model.actor(state, candidates, mask)
    with torch.no_grad():
        teacher_probability = teacher.actor(state, candidates, mask).softmax(-1)
    student_logp = student_logits.log_softmax(-1)
    kl = (teacher_probability * (teacher_probability.clamp_min(1e-12).log() -
                                student_logp)).sum(-1).mean()
    ce = F.cross_entropy(student_logits, records["actions"][indices])
    return kl_coefficient * kl + ce_coefficient * ce, kl, ce


def live_ppo_update(model, actor_optimizer, critic_optimizer, batch, bc_records,
                    teacher, rng, *, algorithm, epochs=4, minibatch_size=128,
                    clip=.2, entropy_coefficient=.01, anchor_kl_coefficient=.05,
                    anchor_ce_coefficient=.05, anchor_data_role=None):
    """Update all critic rows and only eligible on-policy actor choices.

    ``batch['action_applied']`` is mandatory. BT vetoes are still applied
    *proposals* and can receive actor credit; only a transition explicitly
    reported before action admission is excluded. Forced waits keep their
    rewards and time, allowing GAE to credit the earlier real choices.

    Train-only anchors are a separate actor objective and gradient metric.
    Adam is not stepped in a minibatch lacking both choices and informative
    anchors, so existing optimizer momentum cannot move an idle actor.
    """
    if algorithm not in ("vanilla", "anchored"):
        raise ValueError("algorithm must be vanilla or anchored")
    count = len(batch["actions"])
    if count < 1 or min(epochs, minibatch_size) < 1:
        raise ValueError("Live PPO requires samples and positive update sizes")
    if not all(math.isfinite(value) for value in
               (clip, entropy_coefficient, anchor_kl_coefficient, anchor_ce_coefficient)) or \
            clip <= 0 or min(entropy_coefficient, anchor_kl_coefficient, anchor_ce_coefficient) < 0:
        raise ValueError("Live PPO coefficients must be finite and nonnegative")
    if "action_applied" not in batch:
        raise ValueError("Live PPO requires explicit action_applied provenance")
    eligible = live_actor_eligibility(batch["mask"], batch["action_applied"])
    if eligible.shape != batch["actions"].shape:
        raise ValueError("Live action count differs from eligibility")
    if batch["actions"].dtype != torch.long or \
            not bool(((batch["actions"] >= 0) & (batch["actions"] < 16)).all()) or \
            not bool(batch["mask"].gather(1, batch["actions"][:, None]).all()):
        raise ValueError("Live PPO actions must be legal integer intents")
    advantages = normalize_live_actor_advantages(batch["advantages"], eligible)
    actor_parameters = list(model.actor.parameters())
    critic_parameters = list(model.critic_parameters())
    if {id(parameter) for parameter in actor_parameters} & {id(parameter) for parameter in critic_parameters}:
        raise ValueError("Live actor and critic parameters must be independent")
    use_anchors = algorithm == "anchored" and (anchor_kl_coefficient > 0 or anchor_ce_coefficient > 0)
    if use_anchors and anchor_data_role != "train_demonstration":
        raise ValueError("Live anchors require train_demonstration data, never evaluation data")
    if use_anchors and (bc_records is None or teacher is None):
        raise ValueError("Anchors require verified training records and a frozen teacher")
    if use_anchors and any(parameter.requires_grad for parameter in teacher.parameters()):
        raise ValueError("The BC teacher must be frozen")
    anchor_records = 0 if not use_anchors else int((bc_records["mask"].sum(-1) > 1).sum())
    metrics = {key: [] for key in ("policy_loss", "value_loss", "entropy", "approx_kl_old_policy",
        "kl_to_frozen_BC", "auxiliary_CE", "actor_gradient_norm", "critic_gradient_norm",
        "on_policy_actor_gradient_norm", "anchor_actor_gradient_norm")}
    actor_steps = critic_steps = on_policy_steps = anchor_steps = anchor_only_steps = 0
    for _ in range(epochs):
        order = rng.permutation(count)
        for begin in range(0, count, minibatch_size):
            idx = torch.as_tensor(order[begin:begin+minibatch_size], dtype=torch.long,
                                  device=batch["actions"].device)
            distribution, value = model.distribution_value(
                batch["state"][idx], batch["candidates"][idx], batch["mask"][idx])
            local = eligible[idx]
            actor_loss = None
            if bool(local.any()):
                log_ratio = distribution.log_prob(batch["actions"][idx])[local] - batch["old_logp"][idx][local]
                ratio = log_ratio.exp()
                selected_advantages = advantages[idx][local]
                policy_loss = torch.maximum(-selected_advantages * ratio,
                    -selected_advantages * ratio.clamp(1.-clip, 1.+clip)).mean()
                entropy = distribution.entropy()[local].mean()
                actor_loss = policy_loss - entropy_coefficient * entropy
                metrics["policy_loss"].append(float(policy_loss.detach()))
                metrics["entropy"].append(float(entropy.detach()))
                metrics["approx_kl_old_policy"].append(float(((ratio-1.)-log_ratio).mean().detach()))
                metrics["on_policy_actor_gradient_norm"].append(_gradient_norm(actor_loss, actor_parameters))
                on_policy_steps += 1
            if use_anchors and anchor_records:
                indices = _anchor_indices(bc_records, len(idx), rng)
                anchor_loss, kl, ce = _anchor_loss(model, teacher, bc_records, indices,
                                                   anchor_kl_coefficient, anchor_ce_coefficient)
                metrics["anchor_actor_gradient_norm"].append(_gradient_norm(anchor_loss, actor_parameters))
                metrics["kl_to_frozen_BC"].append(float(kl.detach()))
                metrics["auxiliary_CE"].append(float(ce.detach()))
                actor_loss = anchor_loss if actor_loss is None else actor_loss + anchor_loss
                anchor_steps += 1
                anchor_only_steps += int(not bool(local.any()))
            actor_optimizer.zero_grad(set_to_none=True)
            if actor_loss is not None:
                if not bool(torch.isfinite(actor_loss)):
                    raise FloatingPointError("Non-finite live actor loss")
                actor_loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(actor_parameters, .5)
                if not bool(torch.isfinite(gradient)):
                    raise FloatingPointError("Non-finite live actor gradient")
                metrics["actor_gradient_norm"].append(float(gradient))
                actor_optimizer.step()
                actor_steps += 1
            critic_loss = .5 * F.mse_loss(value, batch["returns"][idx])
            if not bool(torch.isfinite(critic_loss)):
                raise FloatingPointError("Non-finite live critic loss")
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_gradient = torch.nn.utils.clip_grad_norm_(critic_parameters, .5)
            if not bool(torch.isfinite(critic_gradient)):
                raise FloatingPointError("Non-finite live critic gradient")
            critic_optimizer.step()
            critic_steps += 1
            metrics["value_loss"].append(float(critic_loss.detach()))
            metrics["critic_gradient_norm"].append(float(critic_gradient))
            if any(not bool(torch.isfinite(parameter).all()) for parameter in (*actor_parameters, *critic_parameters)):
                raise FloatingPointError("Non-finite live model parameter")
    with torch.no_grad():
        distribution, _ = model.distribution_value(batch["state"], batch["candidates"], batch["mask"])
        log_ratio = distribution.log_prob(batch["actions"])[eligible] - batch["old_logp"][eligible]
        ratio = log_ratio.exp()
    action_counts = np.bincount(batch["actions"].detach().cpu().numpy(), minlength=16)
    stats = {key: float(np.mean(values)) if values else 0. for key, values in metrics.items()}
    stats.update(actor_learning_revision=1, critic_sample_count=count,
        on_policy_actor_sample_count=int(eligible.sum()),
        actor_excluded_applied_single_choice_count=int((batch["action_applied"] &
            (batch["mask"].sum(-1) == 1)).sum()),
        actor_excluded_not_applied_count=int((~batch["action_applied"]).sum()),
        excluded_actor_sample_count=int((~eligible).sum()),
        anchor_informative_records=anchor_records,
        anchor_data_role=anchor_data_role if use_anchors else None,
        actor_optimizer_steps=actor_steps, critic_optimizer_steps=critic_steps,
        on_policy_actor_optimizer_steps=on_policy_steps,
        anchor_only_actor_optimizer_steps=anchor_only_steps,
        on_policy_actor_minibatches=on_policy_steps, anchor_actor_minibatches=anchor_steps,
        actor_steps_skipped=critic_steps-actor_steps,
        actor_advantage_normalization="eligible_only_singleton_unscaled",
        post_update_approx_kl_old_policy=float(((ratio-1.)-log_ratio).mean()) if len(ratio) else 0.,
        post_update_clip_fraction=float(((ratio-1.).abs() > clip).float().mean()) if len(ratio) else 0.,
        rollout_action_counts={str(i): int(value) for i, value in enumerate(action_counts)},
        rollout_action_groups={name: int(sum(action_counts[i] for i in ids))
                               for name, ids in ACTION_GROUPS.items()})
    return stats
