#!/usr/bin/env python3
"""Controlled vanilla PPO vs BC-anchored PPO for UC local support tasks.

This is a local observed-support course trainer, not a Nav2-in-the-loop or
full-match trainer. Both algorithms start from identical BT imitation data;
the anchored variant adds train-only imitation and forward-KL regularization.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from rl_bt.uc_support_env_v2 import (
    UC_SUPPORT_V2_SCENARIO_ID, UCSupportEnvV2, uc_support_v2_bt_baseline)
from rl_bt.uc_support_policy_v2 import UCSupportV2ActorCritic

STATE_DIM, CANDIDATE_COUNT, CANDIDATE_DIM = 24, 16, 12
VALIDATION_SEEDS = tuple(range(105001, 105033))
TEST_SEEDS = tuple(range(115001, 115065))
RESERVED_SEEDS = frozenset((*range(85001, 85033), *range(95001, 95065),
                            *VALIDATION_SEEDS, *TEST_SEEDS))
CHECKPOINT_SCHEMA = 1
SCHEDULE_PATH = ROOT / "configs/uc_support_training_schedule_v2.json"
ANCHOR_ACTION_GROUPS = {"navigate": tuple(range(13)), "recover": (13,),
                        "continue": (14,), "hold": (15,)}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def observation_tensors(observations, device="cpu"):
    if not observations:
        raise ValueError("At least one observation is required")
    state = np.stack([row["state"] for row in observations])
    candidates = np.stack([row["candidates"] for row in observations])
    mask = np.stack([row["mask"] for row in observations])
    if state.shape != (len(observations), STATE_DIM):
        raise ValueError("UC v2 state schema mismatch")
    if candidates.shape != (len(observations), CANDIDATE_COUNT, CANDIDATE_DIM):
        raise ValueError("UC v2 candidate schema mismatch")
    if mask.shape != (len(observations), CANDIDATE_COUNT) or mask.dtype != np.bool_:
        raise ValueError("UC v2 candidate mask schema mismatch")
    if not np.isfinite(state).all() or not np.isfinite(candidates).all():
        raise ValueError("UC v2 observations must be finite")
    if not mask.any(axis=1).all():
        raise ValueError("Each observation needs a legal BT fallback")
    return {"state": torch.as_tensor(state, dtype=torch.float32, device=device),
            "candidates": torch.as_tensor(candidates, dtype=torch.float32, device=device),
            "mask": torch.as_tensor(mask, dtype=torch.bool, device=device)}


def gae(rewards, values, next_values, terminated, truncated, *, gamma=.99, lam=.95):
    """Bootstrap nonterminal transitions, but reset GAE at all episode ends."""
    arrays = [np.asarray(x, dtype=np.float32) for x in (rewards, values, next_values)]
    terminated, truncated = np.asarray(terminated, bool), np.asarray(truncated, bool)
    if arrays[0].ndim != 1 or any(x.shape != arrays[0].shape for x in arrays[1:]) or \
            terminated.shape != arrays[0].shape or truncated.shape != arrays[0].shape:
        raise ValueError("GAE arrays must have identical one-dimensional shapes")
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("GAE inputs must be finite")
    advantages = np.zeros_like(arrays[0])
    carry = 0.
    for i in reversed(range(len(advantages))):
        bootstrap = float(not terminated[i])
        continuing = float(not (terminated[i] or truncated[i]))
        delta = arrays[0][i] + gamma * bootstrap * arrays[2][i] - arrays[1][i]
        carry = delta + gamma * lam * continuing * carry
        advantages[i] = carry
    return advantages, advantages + arrays[1]


def action_group(action):
    action = int(action)
    if 0 <= action < 13:
        return "navigate"
    if action == 13:
        return "recover"
    if action == 14:
        return "continue"
    if action == 15:
        return "hold"
    raise ValueError(f"Action {action} is outside the v2 intent space")


def balanced_indices(actions, count, rng):
    """Sample equally from represented intent classes, with replacement."""
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().numpy()
    actions = np.asarray(actions, dtype=np.int64)
    if actions.ndim != 1 or not len(actions) or count < 1:
        raise ValueError("Balanced sampling needs nonempty labels and positive count")
    groups = {name: np.flatnonzero(np.isin(actions, ids))
              for name, ids in ANCHOR_ACTION_GROUPS.items()}
    groups = {name: indices for name, indices in groups.items() if len(indices)}
    names = tuple(groups)
    draws = []
    for _ in range(count):
        name = names[int(rng.integers(len(names)))]
        candidates = groups[name]
        draws.append(int(candidates[int(rng.integers(len(candidates)))]))
    return np.asarray(draws, dtype=np.int64)


def validate_bc_records(records):
    required = {"state", "candidates", "mask", "actions"}
    if set(records) != required:
        raise ValueError("BC records must contain state/candidates/mask/actions")
    count = len(records["actions"])
    if records["state"].shape != (count, STATE_DIM) or \
            records["candidates"].shape != (count, CANDIDATE_COUNT, CANDIDATE_DIM) or \
            records["mask"].shape != (count, CANDIDATE_COUNT):
        raise ValueError("BC record shape mismatch")
    if records["mask"].dtype != np.bool_ or not np.isfinite(records["state"]).all() or \
            not np.isfinite(records["candidates"]).all():
        raise ValueError("BC records have invalid masks or non-finite observations")
    actions = np.asarray(records["actions"], dtype=np.int64)
    if ((actions < 0) | (actions >= CANDIDATE_COUNT)).any() or \
            not records["mask"][np.arange(count), actions].all():
        raise ValueError("BC teacher labels must pass the BT candidate mask")


def make_bc_records(environment, baseline, *, decisions, seed, rng):
    if decisions < 1:
        raise ValueError("At least one BT label is required")
    episode_seed = int(seed)
    observation = environment.reset(seed=episode_seed)
    records = {key: [] for key in ("state", "candidates", "mask", "actions")}
    episode_seeds = [episode_seed]
    for _ in range(decisions):
        action = int(baseline(observation))
        if not 0 <= action < CANDIDATE_COUNT or not observation["mask"][action]:
            raise RuntimeError("BT teacher returned a candidate rejected by its own mask")
        for key in ("state", "candidates", "mask"):
            records[key].append(np.asarray(observation[key]).copy())
        records["actions"].append(action)
        observation, _, done, info = environment.step(action)
        if done:
            episode_seed = next_training_seed(rng)
            episode_seeds.append(episode_seed)
            observation = environment.reset(seed=episode_seed)
    records = {key: np.stack(value) for key, value in records.items()}
    validate_bc_records(records)
    groups = {name: int(sum(action_group(a) == name for a in records["actions"]))
              for name in ANCHOR_ACTION_GROUPS}
    return records, episode_seeds, groups


def fit_bc_actor(model, records, *, seed, epochs=10):
    validate_bc_records(records)
    rng = np.random.default_rng(seed)
    batch = {key: torch.as_tensor(value) for key, value in records.items()}
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
    losses = []
    for _ in range(epochs):
        for begin in range(0, len(batch["actions"]), 64):
            indices = balanced_indices(batch["actions"].numpy(),
                                       min(64, len(batch["actions"]) - begin), rng)
            # Balance on the whole label set, independent of contiguous order.
            indices = torch.as_tensor(indices, dtype=torch.long)
            logits = model.actor(batch["state"][indices], batch["candidates"][indices],
                                 batch["mask"][indices])
            loss = F.cross_entropy(logits, batch["actions"][indices])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite UC v2 BC loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.)
            if not bool(torch.isfinite(grad)):
                raise FloatingPointError("Non-finite UC v2 BC gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
    with torch.no_grad():
        correct = model.actor(batch["state"], batch["candidates"], batch["mask"]).argmax(-1)
        accuracy = float((correct == batch["actions"]).float().mean())
    by_group = {}
    for name, ids in ANCHOR_ACTION_GROUPS.items():
        selector = torch.as_tensor(np.isin(records["actions"], ids))
        by_group[name] = {"count": int(selector.sum()),
            "accuracy": float((correct[selector] == batch["actions"][selector]).float().mean())
                if bool(selector.any()) else None}
    return {"epochs": epochs, "balanced_sampling": True,
            "training_label_accuracy": accuracy, "training_label_accuracy_by_group": by_group,
            "mean_loss": float(np.mean(losses))}


def _actor_logits(model, tensors):
    return model.actor(tensors["state"], tensors["candidates"], tensors["mask"])


def anchor_losses(model, teacher, bc_batch):
    """Forward KL(BC teacher || current actor) and train-label cross entropy."""
    student_logits = _actor_logits(model, bc_batch)
    with torch.no_grad():
        teacher_logits = _actor_logits(teacher, bc_batch)
        teacher_prob = torch.softmax(teacher_logits, dim=-1)
    student_log_prob = torch.log_softmax(student_logits, dim=-1)
    forward_kl = (teacher_prob * (torch.log(teacher_prob.clamp_min(1e-12)) -
                                  student_log_prob)).sum(-1).mean()
    supervised = F.cross_entropy(student_logits, bc_batch["actions"])
    return forward_kl, supervised


def ppo_v2_update(model, actor_optimizer, critic_optimizer, batch, bc_records,
                  teacher, rng, *, algorithm, epochs=4, minibatch_size=128,
                  clip=.2, entropy_coefficient=.01, anchor_kl_coefficient=.05,
                  anchor_ce_coefficient=.05):
    """PPO update with disjoint actor/critic gradients and train-only anchors."""
    if algorithm not in ("vanilla", "anchored"):
        raise ValueError("algorithm must be vanilla or anchored")
    count = len(batch["actions"])
    if count < 1 or min(epochs, minibatch_size) < 1:
        raise ValueError("PPO update requires samples and positive update sizes")
    actor_parameters = list(model.actor.parameters())
    critic_parameters = list(model.critic_parameters())
    if {id(p) for p in actor_parameters} & {id(p) for p in critic_parameters}:
        raise ValueError("UC v2 actor and critic must have independent parameters")
    advantages = batch["advantages"].float()
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    summaries, action_counts = [], np.zeros(CANDIDATE_COUNT, np.int64)
    for action in batch["actions"].tolist():
        action_counts[int(action)] += 1
    for _ in range(epochs):
        order = rng.permutation(count)
        for begin in range(0, count, minibatch_size):
            idx = torch.as_tensor(order[begin:begin + minibatch_size], dtype=torch.long)
            distribution, predicted_value = model.distribution_value(
                batch["state"][idx], batch["candidates"][idx], batch["mask"][idx])
            logp = distribution.log_prob(batch["actions"][idx])
            log_ratio = logp - batch["old_logp"][idx]
            ratio = log_ratio.exp()
            policy_loss = torch.maximum(-advantages[idx] * ratio,
                -advantages[idx] * ratio.clamp(1. - clip, 1. + clip)).mean()
            entropy = distribution.entropy().mean()
            actor_loss = policy_loss - entropy_coefficient * entropy
            kl_value = ce_value = actor_loss.new_zeros(())
            if algorithm == "anchored":
                anchor_idx = balanced_indices(bc_records["actions"], len(idx), rng)
                anchor_idx = torch.as_tensor(anchor_idx, dtype=torch.long)
                anchor_batch = {key: value[anchor_idx] for key, value in bc_records.items()
                                if key in ("state", "candidates", "mask", "actions")}
                kl_value, ce_value = anchor_losses(model, teacher, anchor_batch)
                actor_loss = actor_loss + anchor_kl_coefficient * kl_value + \
                    anchor_ce_coefficient * ce_value
            critic_loss = .5 * F.mse_loss(predicted_value, batch["returns"][idx])
            if not bool(torch.isfinite(actor_loss)) or not bool(torch.isfinite(critic_loss)):
                raise FloatingPointError("Non-finite UC v2 PPO loss")
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad = torch.nn.utils.clip_grad_norm_(actor_parameters, .5)
            if not bool(torch.isfinite(actor_grad)):
                raise FloatingPointError("Non-finite UC v2 actor gradient")
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_grad = torch.nn.utils.clip_grad_norm_(critic_parameters, .5)
            if not bool(torch.isfinite(critic_grad)):
                raise FloatingPointError("Non-finite UC v2 critic gradient")
            actor_optimizer.step()
            critic_optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in (*actor_parameters, *critic_parameters)):
                raise FloatingPointError("Non-finite UC v2 model parameter")
            summaries.append((policy_loss.item(), critic_loss.item(), entropy.item(),
                ((ratio - 1.) - log_ratio).mean().item(), kl_value.item(), ce_value.item(),
                actor_grad.item(), critic_grad.item()))
    means = np.mean(summaries, axis=0)
    with torch.no_grad():
        final_distribution, _ = model.distribution_value(
            batch["state"], batch["candidates"], batch["mask"])
        final_log_ratio = final_distribution.log_prob(batch["actions"]) - batch["old_logp"]
        final_ratio = final_log_ratio.exp()
    return {"policy_loss": float(means[0]), "value_loss": float(means[1]),
        "entropy": float(means[2]), "approx_kl_old_policy": float(means[3]),
        "kl_to_frozen_BC": float(means[4]), "auxiliary_CE": float(means[5]),
        "actor_gradient_norm": float(means[6]), "critic_gradient_norm": float(means[7]),
        "post_update_approx_kl_old_policy": float(((final_ratio - 1.) - final_log_ratio).mean()),
        "post_update_clip_fraction": float(((final_ratio - 1.).abs() > clip).float().mean()),
        "rollout_action_counts": {str(i): int(count) for i, count in enumerate(action_counts)},
        "rollout_action_groups": {name: int(sum(action_counts[i] for i in ids))
                                  for name, ids in ANCHOR_ACTION_GROUPS.items()}}


def next_training_seed(rng):
    while True:
        seed = int(rng.integers(1000, 2 ** 31))
        if seed not in RESERVED_SEEDS:
            return seed


def collect_rollout(model, environment, observation, rng, *, count, episode_seed):
    keys = ("state", "candidates", "mask", "actions", "old_logp", "values",
            "next_values", "rewards", "terminated", "truncated")
    records = {key: [] for key in keys}
    episodes, components, step_rows = [], [], []
    for _ in range(count):
        tensors = observation_tensors([observation])
        with torch.no_grad():
            distribution, value = model.distribution_value(**tensors)
            action = distribution.sample()
            logp = distribution.log_prob(action)
        selected = int(action.item())
        if not observation["mask"][selected]:
            raise RuntimeError("UC v2 actor sampled a BT-masked candidate")
        following, reward, done, info = environment.step(selected)
        terminated, truncated = bool(info["terminated"]), bool(info["truncated"])
        if bool(done) != (terminated or truncated):
            raise ValueError("UC v2 done must match terminated or truncated")
        with torch.no_grad():
            _, following_value = model.distribution_value(**observation_tensors([following]))
        for key in ("state", "candidates", "mask"):
            records[key].append(np.asarray(observation[key]).copy())
        for key, value_ in (("actions", selected), ("old_logp", float(logp.item())),
                ("values", float(value.item())), ("next_values", float(following_value.item())),
                ("rewards", float(reward)), ("terminated", terminated), ("truncated", truncated)):
            records[key].append(value_)
        components.append(info.get("reward_components", {}))
        is_useful_recovery = selected == 13 and float(observation["state"][8]) >= 1./3.
        step_rows.append({"action": selected, "action_group": action_group(selected),
            "useful_recovery": is_useful_recovery,
            "needless_recovery": selected == 13 and not is_useful_recovery,
            "BT_veto": bool(info.get("BT_veto", False))})
        if done:
            episodes.append({**copy.deepcopy(info["episode"]), "training_seed": episode_seed})
            episode_seed = next_training_seed(rng)
            observation = environment.reset(seed=episode_seed)
        else:
            observation = following
    arrays = {key: np.stack(value) for key, value in records.items()}
    advantages, returns = gae(arrays["rewards"], arrays["values"], arrays["next_values"],
                              arrays["terminated"], arrays["truncated"])
    batch = {key: torch.as_tensor(value) for key, value in arrays.items()}
    for key in ("old_logp", "values", "next_values", "rewards"):
        batch[key] = batch[key].float()
    batch["actions"] = batch["actions"].long()
    batch["advantages"] = torch.as_tensor(advantages, dtype=torch.float32)
    batch["returns"] = torch.as_tensor(returns, dtype=torch.float32)
    return batch, observation, episode_seed, episodes, components, step_rows


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_hashes(dataset_dir):
    hashes = {}
    for path in sorted(Path(dataset_dir).rglob("*")):
        if path.is_file() and path.suffix in (".json", ".npz"):
            hashes[path.relative_to(dataset_dir).as_posix()] = sha256_file(path)
    if not any(name.endswith(".npz") for name in hashes) or not hashes:
        raise ValueError("UC v2 training needs prepared NPZ data and manifests")
    return hashes


def source_snapshot(run=None):
    paths = sorted((ROOT / "rl_bt").glob("*.py")) + [Path(__file__).resolve(), SCHEDULE_PATH]
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        hashes[relative.as_posix()] = sha256_file(path)
        if run is not None:
            destination = run / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    return hashes


def capture_rng(rng):
    return {"python": random.getstate(), "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(), "trainer_numpy": copy.deepcopy(rng.bit_generator.state)}


def restore_rng(saved, rng):
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy_global"])
    torch.set_rng_state(saved["torch_cpu"])
    rng.bit_generator.state = copy.deepcopy(saved["trainer_numpy"])


def checkpoint_payload(model, teacher, actor_optimizer, critic_optimizer, metadata, rng,
                       environment, observation, records, *, decisions, updates, episodes,
                       episode_seed, wall_s):
    return {"checkpoint_schema": CHECKPOINT_SCHEMA, "model": model.state_dict(),
        "teacher_actor": teacher.actor.state_dict(), "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(), "metadata": metadata,
        "rng_states": capture_rng(rng), "environment_state": environment.state_dict(),
        "current_observation": copy.deepcopy(observation), "bc_records": records,
        "bc_sha256": metadata["bc_labels_sha256"], "decisions": decisions, "updates": updates,
        "episodes": copy.deepcopy(episodes), "episode_seed": episode_seed, "wall_s": wall_s}


def save_checkpoint(path, *args, **kwargs):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint_payload(*args, **kwargs), temporary)
    temporary.replace(path)


def load_checkpoint(path, model, teacher, actor_optimizer, critic_optimizer, metadata,
                    rng, environment):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("UC v2 checkpoint schema mismatch")
    saved = payload["metadata"]
    for key in ("scenario_id", "algorithm", "observation_schema", "implementation_sha256",
                "dataset_sha256", "bc_labels_sha256", "seed", "rollout", "epochs",
                "anchor_kl_coefficient", "anchor_ce_coefficient"):
        if saved.get(key) != metadata.get(key):
            raise ValueError(f"UC v2 resume mismatch: {key}")
    if payload.get("bc_sha256") != metadata.get("bc_labels_sha256"):
        raise ValueError("UC v2 resume mismatch: BC labels")
    model.load_state_dict(payload["model"])
    teacher.actor.load_state_dict(payload["teacher_actor"])
    actor_optimizer.load_state_dict(payload["actor_optimizer"])
    critic_optimizer.load_state_dict(payload["critic_optimizer"])
    environment.load_state_dict(payload["environment_state"])
    restore_rng(payload["rng_states"], rng)
    records = payload["bc_records"]
    validate_bc_records(records)
    if records_sha256(records) != metadata["bc_labels_sha256"]:
        raise ValueError("UC v2 checkpoint BC label fingerprint mismatch")
    return payload


def truncate_logs_to_checkpoint(run, *, decisions, episode_count):
    """Discard an interrupted append tail before replaying a saved state."""
    for filename, predicate in (
            ("train.jsonl", lambda index, row: row["decisions"] <= decisions),
            ("episodes.jsonl", lambda index, row: index < episode_count)):
        path = Path(run) / filename
        if not path.is_file():
            continue
        kept, lines = [], path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines)-1:
                    raise
                break
            if predicate(index, row):
                kept.append(line)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("\n".join(kept) + ("\n" if kept else ""))
        temporary.replace(path)


def records_sha256(records):
    digest = hashlib.sha256()
    for key in sorted(records):
        array = np.ascontiguousarray(records[key])
        digest.update(key.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def weights_sha256(state_dict):
    """Content fingerprint independent of torch's checkpoint container format."""
    return records_sha256({key: value.detach().cpu().numpy()
                           for key, value in state_dict.items()})


def effective_anchor_coefficients(args):
    return (args.anchor_kl, args.anchor_ce) if args.algorithm == "anchored" else (0., 0.)


def optimizer_rng_for_update(seed, completed_updates):
    """Keep anchor minibatch draws independent of future episode seed draws."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(completed_updates), 913]))


def tensor_bc_records(records):
    validate_bc_records(records)
    return {key: torch.as_tensor(value, dtype=(torch.bool if key == "mask" else
                  torch.long if key == "actions" else torch.float32))
            for key, value in records.items()}


def choose_actor(actor, observation):
    with torch.no_grad():
        logits = _actor_logits(actor, observation_tensors([observation]))
        return int(logits.argmax(-1).item())


def evaluate_policy(policy, environment_type, dataset_dir, *, split, seeds, max_steps=512):
    rows = []
    for seed in seeds:
        env = environment_type(seed=seed, split=split, randomization=True, dataset_dir=dataset_dir)
        observation = env.reset(seed=seed)
        total = 0.
        for _ in range(max_steps):
            action = int(policy(observation, seed))
            if action < 0 or action >= CANDIDATE_COUNT or not observation["mask"][action]:
                raise RuntimeError("UC v2 evaluation policy proposed a masked intent")
            observation, reward, done, info = env.step(action)
            total += float(reward)
            if done:
                episode = copy.deepcopy(info["episode"])
                if not np.isclose(total, episode["return"], rtol=1e-5, atol=1e-5):
                    raise ValueError("UC v2 episode return does not match step rewards")
                rows.append({"seed": seed, **episode})
                break
        else:
            raise RuntimeError(f"UC v2 evaluation seed {seed} exceeded {max_steps} steps")
    return rows


def summarize_evaluation(groups, seeds, split):
    expected = list(seeds)
    for name, rows in groups.items():
        if [row["seed"] for row in rows] != expected:
            raise ValueError(f"UC v2 paired evaluation seed mismatch in {name}")
    paired = []
    for index, seed in enumerate(expected):
        source_ids = {rows[index].get("source_run_id") for rows in groups.values()}
        if len(source_ids) > 1:
            raise ValueError("UC v2 policies must share the same sampled source episode")
        descriptors = {json.dumps(rows[index].get("parameters", {}), sort_keys=True,
                                  allow_nan=False) for rows in groups.values()}
        if len(descriptors) > 1:
            raise ValueError("UC v2 policies must share identical tasks and exogenous schedules")
        paired.append({"seed": seed, "source_run_id": next(iter(source_ids)),
            **{name: {"success": bool(rows[index]["success"]),
                      "score_100": rows[index].get("score_100"),
                      "return": rows[index].get("return")} for name, rows in groups.items()}})
    summary = {}
    for name, rows in groups.items():
        summary[name] = {"episodes": len(rows), "successes": int(sum(r["success"] for r in rows)),
            "task_success_rate": float(np.mean([r["success"] for r in rows])),
            "mean_return": float(np.mean([r["return"] for r in rows])),
            "mean_steps": float(np.mean([r["steps"] for r in rows])),
            "mean_score_100": float(np.mean([r["score_100"] for r in rows]))
                if all("score_100" in r for r in rows) else None,
            "mean_BT_veto_count": float(np.mean([r.get("BT_veto_count", 0) for r in rows])),
            "mean_recoveries": float(np.mean([r.get("recoveries", 0) for r in rows]))}
    return {"scenario_id": UC_SUPPORT_V2_SCENARIO_ID, "dataset_split": split,
        "seeds_predeclared_before_training": expected, "paired_source_episode": True,
        "metric": "UC local observed-support task success, not match win rate",
        "checkpoint_selection": "fixed scheduled/final checkpoint",
        "summary": summary, "paired_results": paired, "episodes": groups,
        "limits": ["Offline local support/execution course; not Nav2/MPPI-in-the-loop.",
                   "No opponent, aiming, firing, match lifecycle, or shared team economy model."]}


def make_model():
    model = UCSupportV2ActorCritic()
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return model, teacher


def write_config(run, args, env, bc_groups, bc_report, hashes, data_hashes, bc_hash):
    anchor_kl, anchor_ce = effective_anchor_coefficients(args)
    config = {"artifact_type": "UC_v2_local_observed_support_optimizer_ablation",
        "scenario_id": UC_SUPPORT_V2_SCENARIO_ID, "algorithm": args.algorithm,
        "seed": args.seed, "total_ppo_decisions": args.decisions,
        "bt_imitation_decisions": args.bc_decisions, "bc_action_groups": bc_groups,
        "bc_report": bc_report, "bc_labels_sha256": bc_hash,
        "dataset_dir": str(args.dataset_dir.resolve()), "dataset_manifest": env.dataset_manifest,
        "dataset_sha256": data_hashes,
        "observation_schema": {"state": [24], "candidates": [16, 12], "mask": [16]},
        "decision_dt_s": .5, "architecture": "independent actor and critic encoders; no shared weights",
        "optimizer": {"actor": "Adam", "critic": "Adam",
            "actor_learning_rate": 1e-4 if args.algorithm == "anchored" else 3e-4,
            "critic_learning_rate": 3e-4}, "gamma": .99, "gae_lambda": .95,
        "ppo_clip": .2, "entropy_coefficient": .01, "anchor_kl_coefficient": anchor_kl,
        "anchor_ce_coefficient": anchor_ce,
        "anchor_data": "balanced BT labels from train split only",
        "rollout": args.rollout, "epochs": args.epochs, "threads": 2, "device": "cpu",
        "optimizer_random_stream": "SeedSequence(training_seed, completed_updates, 913); separate from episode-seed RNG",
        "validation_seeds": list(VALIDATION_SEEDS), "test_seeds": list(TEST_SEEDS),
        "excluded_legacy_evaluation_seeds": list(sorted(RESERVED_SEEDS -
            set(VALIDATION_SEEDS) - set(TEST_SEEDS))),
        "resume_supported": True, "checkpoint_schema": CHECKPOINT_SCHEMA,
        "implementation_sha256": hashes,
        "limits": ["Local observation/execution course only; no live Nav2/MPPI.",
                   "Success/score cannot be interpreted as match win rate."]}
    write_json(run / "config.json", config)
    return config


def train(args):
    if min(args.decisions, args.rollout, args.epochs) < 1 or args.bc_decisions < 1:
        raise ValueError("decisions, rollout, epochs and BC label count must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    run = args.run.resolve()
    dataset_dir = args.dataset_dir.resolve()
    dataset_sha = dataset_hashes(dataset_dir)
    if args.resume:
        if args.resume.name == "untrained.pt":
            raise ValueError("untrained.pt is an evaluation baseline, not a resume checkpoint")
        if args.resume.resolve().parent != run:
            raise ValueError("Resume checkpoint must be inside its original --run directory")
        if not (run / "config.json").is_file():
            raise ValueError("Resume run is missing config.json")
    else:
        run.mkdir(parents=True, exist_ok=False)
    first_seed = next_training_seed(rng)
    env = UCSupportEnvV2(seed=first_seed, split="train", randomization=True,
                         dataset_dir=dataset_dir)
    model, teacher = make_model()
    actor_lr = 1e-4 if args.algorithm == "anchored" else 3e-4
    anchor_kl, anchor_ce = effective_anchor_coefficients(args)
    actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=actor_lr, eps=1e-5)
    critic_optimizer = torch.optim.Adam(model.critic_parameters(), lr=3e-4, eps=1e-5)
    source_hash = source_snapshot(None)
    bc_file = run / "bt_labels.npz"
    elapsed_before, updates, decisions, episodes = 0., 0, 0, []

    if args.resume:
        old_config = json.loads((run / "config.json").read_text())
        current = {**old_config, "scenario_id": UC_SUPPORT_V2_SCENARIO_ID,
            "algorithm": args.algorithm,
            "observation_schema": {"state": [24], "candidates": [16, 12], "mask": [16]},
            "implementation_sha256": source_hash, "dataset_sha256": dataset_sha,
            "seed": args.seed, "rollout": args.rollout, "epochs": args.epochs,
            "anchor_kl_coefficient": anchor_kl,
            "anchor_ce_coefficient": anchor_ce}
        if args.bc_decisions != old_config["bt_imitation_decisions"]:
            raise ValueError("Resume BC label count must match original training run")
        payload = load_checkpoint(args.resume, model, teacher, actor_optimizer, critic_optimizer,
                                  current, rng, env)
        decisions, updates = payload["decisions"], payload["updates"]
        episodes, elapsed_before = payload["episodes"], payload["wall_s"]
        observation, episode_seed = payload["current_observation"], payload["episode_seed"]
        bc_records = payload["bc_records"]
        if args.decisions < decisions:
            raise ValueError("--decisions is a total target and cannot be below saved progress")
    else:
        untrained_state = copy.deepcopy(model.state_dict())
        observation = env.reset(seed=first_seed)
        bc_records, bc_seeds, bc_groups = make_bc_records(
            env, uc_support_v2_bt_baseline, decisions=args.bc_decisions,
            seed=first_seed, rng=rng)
        with bc_file.open("wb") as handle:
            np.savez_compressed(handle, **bc_records)
        bc_hash = records_sha256(bc_records)
        bc_report = fit_bc_actor(model, bc_records, seed=args.seed + 77, epochs=10)
        teacher.load_state_dict(model.state_dict())
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        hashes = source_snapshot(run)
        config = write_config(run, args, env, bc_groups, bc_report, hashes,
                              dataset_sha, bc_hash)
        config["untrained_actor_weights_sha256"] = weights_sha256({
            key.removeprefix("actor."): value for key, value in untrained_state.items()
            if key.startswith("actor.")})
        config["post_BC_actor_weights_sha256"] = weights_sha256(model.actor.state_dict())
        write_json(run / "config.json", config)
        for path in sorted(dataset_dir.rglob("*.json")):
            destination = run / "source" / "dataset" / path.relative_to(dataset_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
        write_json(run / "bc_labels.json", {"sha256": bc_hash, "episode_seeds": bc_seeds,
            "action_group_counts": bc_groups, "labels_from_split": "train",
            "validation_or_test_labels_used": False, **bc_report})
        episode_seed = next_training_seed(rng)
        observation = env.reset(seed=episode_seed)
        trained_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(untrained_state)
        save_checkpoint(run / "untrained.pt", model, teacher, actor_optimizer, critic_optimizer,
            config, rng, env, observation, bc_records, decisions=0, updates=0,
            episodes=[], episode_seed=episode_seed, wall_s=0.)
        model.load_state_dict(trained_state)
        save_checkpoint(run / "initial.pt", model, teacher, actor_optimizer, critic_optimizer,
            config, rng, env, observation, bc_records, decisions=0, updates=0,
            episodes=[], episode_seed=episode_seed, wall_s=0.)

    if args.resume:
        config = json.loads((run / "config.json").read_text())
        truncate_logs_to_checkpoint(run, decisions=decisions, episode_count=len(episodes))
        # Resume only from trainer-owned checkpoints; the frozen teacher and BC labels
        # are restored from the checkpoint and validated against their content hash.
    bc_tensor = tensor_bc_records(bc_records)
    started = time.perf_counter()
    train_path = run / "train.jsonl"
    episode_path = run / "episodes.jsonl"
    with train_path.open("a" if args.resume else "w") as metrics, \
            episode_path.open("a" if args.resume else "w") as episode_file:
        while decisions < args.decisions:
            count = min(args.rollout, args.decisions - decisions)
            batch, observation, episode_seed, completed, components, step_rows = collect_rollout(
                model, env, observation, rng, count=count, episode_seed=episode_seed)
            decisions += count
            episodes.extend(completed)
            for episode in completed:
                episode_file.write(json.dumps(episode, allow_nan=False) + "\n")
            episode_file.flush()
            update_rng = optimizer_rng_for_update(args.seed, updates)
            update = ppo_v2_update(model, actor_optimizer, critic_optimizer, batch, bc_tensor,
                teacher, update_rng, algorithm=args.algorithm, epochs=args.epochs,
                anchor_kl_coefficient=anchor_kl, anchor_ce_coefficient=anchor_ce)
            updates += 1
            recent = episodes[-32:]
            row = {**update, "update": updates, "decisions": decisions,
                "completed_episodes": len(episodes),
                "recent_task_success_rate": float(np.mean([e["success"] for e in recent]))
                    if recent else None,
                "recent_score_100": float(np.mean([e["score_100"] for e in recent]))
                    if recent and all("score_100" in e for e in recent) else None,
                "recent_return": float(np.mean([e["return"] for e in recent])) if recent else None,
                "useful_recovery_actions": sum(r["useful_recovery"] for r in step_rows),
                "needless_recovery_actions": sum(r["needless_recovery"] for r in step_rows),
                "BT_vetoes": sum(r["BT_veto"] for r in step_rows),
                "wall_s": elapsed_before + time.perf_counter() - started}
            for key in sorted({key for sample in components for key in sample}):
                row["reward_" + key + "_mean"] = float(np.mean([sample.get(key, 0.) for sample in components]))
            metrics.write(json.dumps(row, allow_nan=False) + "\n")
            metrics.flush()
            write_json(run / "status.json", {"status": "training", "algorithm": args.algorithm, **row})
            print(json.dumps(row, allow_nan=False), flush=True)
            if updates % 8 == 0 or decisions % 16384 == 0 or decisions == args.decisions:
                elapsed = elapsed_before + time.perf_counter() - started
                save_checkpoint(run / "latest.pt", model, teacher, actor_optimizer, critic_optimizer,
                    config, rng, env, observation, bc_records, decisions=decisions, updates=updates,
                    episodes=episodes, episode_seed=episode_seed, wall_s=elapsed)
                if decisions % 16384 == 0:
                    save_checkpoint(run / f"round_{decisions:08d}.pt", model, teacher,
                        actor_optimizer, critic_optimizer, config, rng, env, observation,
                        bc_records, decisions=decisions, updates=updates, episodes=episodes,
                        episode_seed=episode_seed, wall_s=elapsed)
                should_validate = args.eval_split == "validation" and \
                    (decisions % args.eval_every == 0 or decisions == args.decisions)
                if should_validate:
                    result = run_evaluation(run, model, teacher, bc_records, args,
                                            split="validation", seeds=VALIDATION_SEEDS)
                    write_json(run / f"validation_{decisions:08d}.json", result)
    wall = elapsed_before + time.perf_counter() - started
    save_checkpoint(run / "final.pt", model, teacher, actor_optimizer, critic_optimizer,
        config, rng, env, observation, bc_records, decisions=decisions, updates=updates,
        episodes=episodes, episode_seed=episode_seed, wall_s=wall)
    evaluation = None
    if args.eval_split == "test":
        evaluation = run_evaluation(run, model, teacher, bc_records, args,
                                    split="test", seeds=TEST_SEEDS)
        write_json(run / "evaluation_test.json", evaluation)
    summary = {"status": "completed", "run": str(run),
        "scenario_id": UC_SUPPORT_V2_SCENARIO_ID, "algorithm": args.algorithm,
        "ppo_decisions": decisions, "bc_decisions": args.bc_decisions,
        "completed_training_episodes": len(episodes), "updates": updates,
        "wall_s": wall, "evaluation_split": args.eval_split,
        "evaluation": None if evaluation is None else evaluation["summary"],
        "metric": "UC local observed-support task score/success; not match win rate"}
    write_json(run / "summary.json", summary)
    write_json(run / "status.json", summary)
    print(json.dumps(summary, allow_nan=False), flush=True)
    return summary


def run_evaluation(run, model, teacher, bc_records, args, *, split, seeds):
    torch.set_num_threads(2)
    groups = {}
    random_streams = {seed: np.random.default_rng(seed ^ 0x5A17) for seed in seeds}
    def random_policy(_observation, seed):
        # Each episode owns a deterministic independent stream shared by algorithms.
        return int(random_streams[seed].choice(np.flatnonzero(_observation["mask"])))
    groups["random"] = evaluate_policy(random_policy, UCSupportEnvV2, args.dataset_dir,
                                       split=split, seeds=seeds)
    bc_model = copy.deepcopy(model).eval()
    bc_model.actor.load_state_dict(teacher.actor.state_dict())
    groups["BC"] = evaluate_policy(lambda obs, _seed: choose_actor(bc_model, obs),
                                    UCSupportEnvV2, args.dataset_dir, split=split, seeds=seeds)
    eval_model = copy.deepcopy(model).eval()
    groups["PPO"] = evaluate_policy(lambda obs, _seed: choose_actor(eval_model, obs),
                                     UCSupportEnvV2, args.dataset_dir, split=split, seeds=seeds)
    groups["BT"] = evaluate_policy(lambda obs, _seed: int(uc_support_v2_bt_baseline(obs)),
                                   UCSupportEnvV2, args.dataset_dir, split=split, seeds=seeds)
    return summarize_evaluation(groups, seeds, split)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("vanilla", "anchored"), required=True)
    parser.add_argument("--decisions", type=int, default=32768,
                        help="Total PPO decision target; includes restored progress")
    parser.add_argument("--bc-decisions", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dataset-dir", type=Path,
                        default=ROOT / "data/derived/uc_support_training_v1")
    parser.add_argument("--rollout", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=16384)
    parser.add_argument("--eval-split", choices=("validation", "test", "none"),
                        default="validation")
    parser.add_argument("--anchor-kl", type=float, default=.05)
    parser.add_argument("--anchor-ce", type=float, default=.05)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32 or args.eval_every < 1 or \
            not np.isfinite([args.anchor_kl, args.anchor_ce]).all() or \
            min(args.anchor_kl, args.anchor_ce) < 0:
        parser.error("Invalid seed, evaluation interval, or anchor coefficient")
    train(args)


if __name__ == "__main__":
    main()
