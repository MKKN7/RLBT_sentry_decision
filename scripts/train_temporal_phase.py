#!/usr/bin/env python3
"""Sequence PPO for the frozen, simplified alphaRM 3v3 task prototype.

This is an isolated temporal/global-decision experiment. It does not replace
or claim to be the real Point-LIO/Nav2/MPPI or official UL/UC training path.
Every optimization batch contains complete chronological episodes (or an
explicit safety-limit truncation), recomputed from zero recurrent state.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from rl_bt.alpharm_task_env import AlphaRuntime, AlphaTaskEnv, SCENARIO, TRAIN_STYLES
from rl_bt.temporal_policy import TemporalActorCritic

SCENARIO_ID = "alpharm_temporal_global_phase_v1"
CHECKPOINT_SCHEMA = 1
OBSERVATION_SCHEMA = {"state": [148], "candidates": [16, 16], "mask": [16]}
CONTRACT_SOURCES = ("rl_bt/temporal_policy.py", "rl_bt/alpharm_task_env.py",
    "rl_bt/geometry.py", "rl_bt/rewards.py", "rl_bt/contracts.py",
    "scripts/probe_alpharm_onnx.py")
SOURCE_FILES = CONTRACT_SOURCES + ("scripts/train_temporal_phase.py",)
MANIFEST = "data/references/alpharm_jax_review_manifest.json"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False) + "\n")


def observation_tensors(observations, device="cpu"):
    if not observations:
        raise ValueError("At least one observation is required")
    data = {key: np.stack([np.asarray(row[key]) for row in observations])
            for key in ("state", "candidates", "mask")}
    count = len(observations)
    if data["state"].shape != (count, 148) or data["candidates"].shape != (count, 16, 16):
        raise ValueError("Temporal observation shape differs from the frozen prototype")
    if data["mask"].shape != (count, 16) or data["mask"].dtype != np.bool_:
        raise ValueError("Temporal masks must be boolean [B,16]")
    if not data["mask"].any(axis=-1).all():
        raise ValueError("Every observation requires a legal fallback")
    if not np.isfinite(data["state"]).all() or not np.isfinite(data["candidates"]).all():
        raise ValueError("Temporal actor inputs must be finite")
    return {key: torch.as_tensor(value, device=device,
            dtype=torch.bool if key == "mask" else torch.float32) for key, value in data.items()}


def validate_checkpoint(payload):
    metadata = payload.get("metadata", {})
    if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA or payload.get("scenario_id") != SCENARIO_ID:
        raise ValueError("Checkpoint is not a temporal prototype experiment")
    if metadata.get("environment_scenario_id") != SCENARIO or metadata.get("observation_schema") != OBSERVATION_SCHEMA:
        raise ValueError("Checkpoint environment/observation contract differs")
    if metadata.get("real_nav2_in_loop") is not False or metadata.get("official_UL_UC_training") is not False:
        raise ValueError("Temporal prototype must not be labelled real/official navigation")
    if metadata.get("train_styles") != list(TRAIN_STYLES):
        raise ValueError("Training styles differ from the frozen train split")
    for relative in CONTRACT_SOURCES:
        if metadata.get("implementation_sha256", {}).get(relative) != file_sha256(ROOT / relative):
            raise ValueError(f"Checkpoint contract source mismatch: {relative}")
    if metadata.get("upstream_manifest_sha256") != file_sha256(ROOT / MANIFEST):
        raise ValueError("Checkpoint upstream manifest changed")
    if metadata.get("memory") not in ("mlp", "gru") or metadata.get("hidden_size") != 64:
        raise ValueError("Unknown temporal model architecture")


def load_temporal_checkpoint(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    validate_checkpoint(payload)
    model = TemporalActorCritic(memory=payload["metadata"]["memory"],
                               hidden_size=payload["metadata"]["hidden_size"])
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload


def episode_gae(sequence, *, gamma_per_second=.997, lambda_per_second=.97):
    """SMDP GAE: retain final truncation bootstrap; terminate at true game end."""
    if not 0 < gamma_per_second <= 1 or not 0 < lambda_per_second <= 1:
        raise ValueError("Discount and GAE lambda must lie in (0,1]")
    rewards = np.asarray(sequence["rewards"], np.float64)
    values = np.asarray(sequence["values"], np.float64)
    durations = np.asarray(sequence["duration_s"], np.float64)
    terminated = np.asarray(sequence["terminated"], bool)
    truncated = np.asarray(sequence["truncated"], bool)
    shape = rewards.shape
    if rewards.ndim != 1 or len(rewards) < 1 or any(v.shape != shape for v in
            (values, durations, terminated, truncated)):
        raise ValueError("A sequence needs matching nonempty one-dimensional arrays")
    bootstrap = float(sequence["bootstrap_value"])
    if not all(np.isfinite(v).all() for v in (rewards, values, durations)) or \
            not math.isfinite(bootstrap) or (durations <= 0).any():
        raise ValueError("Sequence rewards/values/durations must be finite and durations positive")
    if (terminated[:-1] | truncated[:-1]).any() or not bool(terminated[-1] or truncated[-1]):
        raise ValueError("A sequence must end once, at a terminal or explicit truncation")
    if bool(terminated[-1] and truncated[-1]):
        raise ValueError("True termination and external truncation are distinct")
    next_values = np.r_[values[1:], bootstrap]
    advantages = np.zeros(shape, np.float64)
    carry = 0.
    for index in reversed(range(len(rewards))):
        discount = gamma_per_second ** durations[index]
        trace = lambda_per_second ** durations[index]
        delta = rewards[index] + discount * (not terminated[index]) * next_values[index] - values[index]
        carry = delta + discount * trace * (not (terminated[index] or truncated[index])) * carry
        advantages[index] = carry
    return advantages.astype(np.float32), (advantages + values).astype(np.float32)


def _device(model):
    return next(model.parameters()).device


def collect_episode(model, environment, max_decisions, seed):
    """Collect one policy version; no optimizer or privileged actor input."""
    if max_decisions < 1 or environment.done or environment.decisions != 0:
        raise ValueError("Collection requires a fresh environment and positive decision budget")
    if environment.style not in TRAIN_STYLES:
        raise ValueError("Held-out opponent styles must never enter PPO collection")
    observation = {key: np.asarray(value).copy() for key, value in environment.cached.items()}
    hidden = model.initial_state(1, device=_device(model))
    records = {key: [] for key in ("state", "candidates", "mask", "actions", "old_logp",
        "values", "rewards", "terminated", "truncated", "duration_s", "executed_candidate", "override")}
    last_info = {}
    for step in range(max_decisions):
        tensors = observation_tensors([observation], _device(model))
        with torch.no_grad():
            distribution, value, hidden = model.distribution_value(**tensors, hidden=hidden,
                episode_start=torch.tensor([step == 0], device=_device(model)))
            action_tensor = distribution.sample()
            logp = distribution.log_prob(action_tensor)
        action = int(action_tensor.item())
        if not observation["mask"][action]:
            raise RuntimeError("Temporal actor sampled an illegal candidate")
        following, reward, done, info = environment.step(action)
        observation_tensors([following], _device(model))
        truncated = bool(not done and step + 1 == max_decisions)
        for key in ("state", "candidates", "mask"):
            records[key].append(observation[key].copy())
        for key, val in (("actions", action), ("old_logp", float(logp.item())),
                ("values", float(value.item())), ("rewards", float(reward)),
                ("terminated", bool(done)), ("truncated", truncated),
                ("duration_s", float(info["duration_s"])),
                ("executed_candidate", int(info["executed_candidate"])),
                ("override", info.get("override_reason") is not None)):
            records[key].append(val)
        observation, last_info = following, info
        if done or truncated:
            break
    with torch.no_grad():
        if records["terminated"][-1]:
            bootstrap = 0.
        else:
            _, final_value, _ = model.distribution_value(
                **observation_tensors([observation], _device(model)), hidden=hidden,
                episode_start=torch.tensor([False], device=_device(model)))
            bootstrap = float(final_value.item())
    sequence = {key: np.stack(value) for key, value in records.items()}
    sequence["bootstrap_value"] = bootstrap
    sequence["seed"] = int(seed)
    sequence["episode"] = copy.deepcopy(last_info.get("episode"))
    sequence["opponent_style"] = environment.style
    sequence["team"] = int(environment.team)
    sequence["truncation_reason"] = None if records["terminated"][-1] else "episode_decision_limit"
    sequence["simulated_s"] = float(np.asarray(sequence["duration_s"]).sum())
    return sequence


def evaluate_sequence(model, sequence):
    """Recompute current recurrent states from episode start, keeping BPTT."""
    arrays = observation_tensors([{key: sequence[key][i] for key in
        ("state", "candidates", "mask")} for i in range(len(sequence["actions"]))], _device(model))
    hidden = model.initial_state(1, device=_device(model))
    logp, values, entropies = [], [], []
    for i in range(len(sequence["actions"])):
        distribution, value, hidden = model.distribution_value(
            **{key: val[i:i + 1] for key, val in arrays.items()}, hidden=hidden,
            episode_start=torch.tensor([i == 0], device=_device(model)))
        action = torch.as_tensor([sequence["actions"][i]], device=_device(model), dtype=torch.long)
        logp.append(distribution.log_prob(action)[0])
        values.append(value[0])
        entropies.append(distribution.entropy()[0])
    return torch.stack(logp), torch.stack(values), torch.stack(entropies)


def ppo_update(model, optimizer, sequences, *, epochs=3, gamma_per_second=.997,
               lambda_per_second=.97, clip=.2, entropy_coef=.01, value_coef=.5,
               max_grad_norm=.5):
    """One minibatch of whole episodes, with full current-policy recomputation.

    Actor gradients exclude forced single-legal observations. Their real
    reward, elapsed time and critic target still affect earlier decisions.
    """
    if not sequences or epochs < 1:
        raise ValueError("PPO needs one or more episodes and positive epochs")
    if not 0 < clip < 1 or entropy_coef < 0 or value_coef <= 0 or max_grad_norm <= 0:
        raise ValueError("Invalid PPO optimization parameters")
    prepared = []
    for sequence in sequences:
        advantage, returns = episode_gae(sequence, gamma_per_second=gamma_per_second,
                                         lambda_per_second=lambda_per_second)
        legal = np.asarray(sequence["mask"], bool)
        actions = np.asarray(sequence["actions"], np.int64)
        if legal.shape != (len(actions), 16) or (actions < 0).any() or (actions >= 16).any() or \
                not legal[np.arange(len(actions)), actions].all():
            raise ValueError("PPO actions must be legal in their recorded masks")
        prepared.append((sequence, advantage, returns, legal.sum(axis=-1) > 1))
    valid_advantages = np.concatenate([adv[valid] for _, adv, _, valid in prepared])
    adv_mean = float(valid_advantages.mean()) if len(valid_advantages) else 0.
    adv_std = float(valid_advantages.std()) if len(valid_advantages) else 1.
    count = sum(len(item[0]["actions"]) for item in prepared)
    valid_count = sum(int(item[3].sum()) for item in prepared)
    metrics = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        row = {"policy_loss": 0., "value_loss": 0., "entropy": 0., "approx_kl": 0., "clip_fraction": 0.}
        for sequence, advantages, returns, actor_valid in prepared:
            logp, predicted, entropy = evaluate_sequence(model, sequence)
            device = logp.device
            old_logp = torch.as_tensor(sequence["old_logp"], device=device, dtype=torch.float32)
            advantage = torch.as_tensor((advantages - adv_mean) / max(adv_std, 1e-8), device=device)
            targets = torch.as_tensor(returns, device=device)
            valid = torch.as_tensor(actor_valid, device=device)
            value_loss = F.mse_loss(predicted, targets, reduction="sum") / count
            if bool(valid.any()):
                log_ratio = logp[valid] - old_logp[valid]
                ratio = log_ratio.exp()
                surrogate = torch.maximum(-advantage[valid] * ratio,
                    -advantage[valid] * ratio.clamp(1 - clip, 1 + clip))
                policy_loss = surrogate.sum() / valid_count
                entropy_loss = entropy[valid].sum() / valid_count
                kl = ((ratio - 1) - log_ratio).sum() / valid_count
                clipped = ((ratio - 1).abs() > clip).float().sum() / valid_count
            else:
                policy_loss = logp.sum() * 0.
                entropy_loss = entropy.sum() * 0.
                kl = clipped = logp.sum().detach() * 0.
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite temporal PPO loss")
            loss.backward()
            for name, value in (("policy_loss", policy_loss), ("value_loss", value_loss),
                    ("entropy", entropy_loss), ("approx_kl", kl), ("clip_fraction", clipped)):
                row[name] += float(value.detach())
        if not valid_count:
            # Adam's momentum must not move actor weights in an all-forced batch.
            for parameter in model.actor.parameters():
                parameter.grad = None
        actor_norm = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), max_grad_norm)
        critic_norm = torch.nn.utils.clip_grad_norm_(model.critic.parameters(), max_grad_norm)
        if not bool(torch.isfinite(actor_norm) and torch.isfinite(critic_norm)):
            raise FloatingPointError("Non-finite temporal PPO gradient")
        optimizer.step()
        row["actor_gradient_norm"] = float(actor_norm)
        row["critic_gradient_norm"] = float(critic_norm)
        metrics.append(row)
    result = {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}
    result.update({"epochs": epochs, "sequence_count": len(sequences), "decisions": count,
        "actor_decisions": valid_count, "forced_single_legal_decisions": count - valid_count,
        "recurrent_recomputation": "chronological_from_episode_start_full_BPTT",
        "individual_timesteps_shuffled": False, "first_epoch_approx_kl": metrics[0]["approx_kl"]})
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--memory", choices=("mlp", "gru"), required=True)
    parser.add_argument("--decisions", type=int, default=14400)
    parser.add_argument("--episodes-per-update", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--episode-max-decisions", type=int, default=400)
    parser.add_argument("--gamma-per-second", type=float, default=.997)
    parser.add_argument("--lambda-per-second", type=float, default=.97)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=.01)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--no-randomization", action="store_true")
    args = parser.parse_args(argv)
    if min(args.decisions, args.episodes_per_update, args.epochs, args.episode_max_decisions,
           args.torch_threads) < 1 or args.learning_rate <= 0 or args.entropy_coef < 0:
        parser.error("Budgets/threads/learning rate must be positive; entropy must be nonnegative")
    if not 0 < args.gamma_per_second <= 1 or not 0 < args.lambda_per_second <= 1:
        parser.error("Discount and GAE lambda must lie in (0,1]")
    return args


def train(args):
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    run = args.run.resolve()
    if run.exists():
        raise FileExistsError(f"Refusing to overwrite training artifacts: {run}")
    run.mkdir(parents=True)
    (run / "rollouts").mkdir()
    runtime = AlphaRuntime()
    model = TemporalActorCritic(memory=args.memory, hidden_size=64).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    metadata = {"scenario_id": SCENARIO_ID, "environment_scenario_id": SCENARIO,
        "memory": args.memory, "hidden_size": 64, "seed": args.seed,
        "training_episode_seed_scheme": "seed+100003+zero_based_sequence_index",
        "observation_schema": OBSERVATION_SCHEMA, "train_styles": list(TRAIN_STYLES),
        "heldout_styles_used_for_training": False, "data_role": "training",
        "real_nav2_in_loop": False, "official_UL_UC_training": False,
        "reward_profile": "upstream", "reward_unchanged_from_frozen_environment": True,
        "reward_note": "upstream team reward summed at each fixed physics tick; same .002 task-switch cost for every policy",
        "gamma_per_second": args.gamma_per_second, "lambda_per_second": args.lambda_per_second,
        "recurrent_training": "full_sequence_BPTT_from_zero_state_each_episode_each_epoch",
        "matched_budget_unit": "minimum_actual_candidate_decisions_then_finish_episode",
        "target_decisions": args.decisions,
        "episodes_per_update": args.episodes_per_update, "epochs": args.epochs,
        "episode_max_decisions": args.episode_max_decisions, "learning_rate": args.learning_rate,
        "entropy_coef": args.entropy_coef, "randomization": not args.no_randomization,
        "torch_threads": args.torch_threads, "torch_version": torch.__version__,
        "device": "cpu", "upstream_commit": runtime.source_commit,
        "upstream_manifest_sha256": file_sha256(ROOT / MANIFEST),
        "implementation_sha256": {relative: file_sha256(ROOT / relative) for relative in SOURCE_FILES},
        "excluded_outputs": ["velocity", "aim", "fire", "thermal_control"],
        "BC_updates": 0, "final_checkpoint_selection": "fixed_final_budget_no_evaluation_selection",
        "exact_resume_supported": False,
        "limits": ["Simplified 180s homogeneous 3v3 prototype, not official UL/UC.",
            "No live Nav2/MPPI, tunnels, official sentry stance or shared economy.",
            "Noise/delay ranges are engineering assumptions, not calibrated match data.",
            "A recurrent network is not guaranteed to outperform a stateful behavior tree."]}
    for relative in SOURCE_FILES:
        target = run / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    write_json(run / "config.json", metadata)
    started = time.perf_counter()
    decisions = updates = sequence_count = completed = truncated_count = 0
    episode_seeds = []
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    def checkpoint(name, *, episode_boundary=True):
        payload = {"checkpoint_schema": CHECKPOINT_SCHEMA, "scenario_id": SCENARIO_ID,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "metadata": metadata,
            "decisions": decisions, "updates": updates, "sequence_count": sequence_count,
            "numpy_rng_state": copy.deepcopy(rng.bit_generator.state),
            "python_rng_state": random.getstate(), "torch_rng_state": torch.get_rng_state(),
            "episode_boundary": episode_boundary, "training_episode_seeds": episode_seeds.copy()}
        temporary = run / (name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(run / name)
    checkpoint("initial.pt")
    try:
        write_json(run / "status.json", {"status": "running", "decisions": 0, "target_decisions": args.decisions})
        while decisions < args.decisions:
            sequences = []
            for _ in range(args.episodes_per_update):
                if decisions >= args.decisions:
                    break
                episode_seed = args.seed + 100_003 + sequence_count
                environment = AlphaTaskEnv(runtime, episode_seed,
                    randomization=not args.no_randomization, reward_profile="upstream")
                sequence = collect_episode(model, environment, args.episode_max_decisions, episode_seed)
                count = len(sequence["actions"])
                sequence_count += 1
                decisions += count
                episode_seeds.append(episode_seed)
                true_terminal = bool(sequence["terminated"][-1])
                completed += int(true_terminal)
                truncated_count += int(not true_terminal)
                arrays = {key: value for key, value in sequence.items() if isinstance(value, np.ndarray)}
                arrays["bootstrap_value"] = np.asarray(sequence["bootstrap_value"])
                np.savez_compressed(run / "rollouts" / f"sequence_{sequence_count:05d}.npz", **arrays)
                episode_row = {"sequence": sequence_count, "seed": episode_seed,
                    "decisions": count, "cumulative_decisions": decisions,
                    "simulated_s": sequence["simulated_s"], "terminated": true_terminal,
                    "truncated": not true_terminal, "truncation_reason": sequence["truncation_reason"],
                    "opponent_style": sequence["opponent_style"], "team": sequence["team"],
                    "return": float(sequence["rewards"].sum()), "episode": sequence["episode"]}
                append_json(run / "episodes.jsonl", episode_row)
                sequences.append(sequence)
                write_json(run / "status.json", {"status": "collecting", **episode_row,
                    "updates": updates, "target_decisions": args.decisions,
                    "wall_s": time.perf_counter() - started})
            metrics = ppo_update(model, optimizer, sequences, epochs=args.epochs,
                gamma_per_second=args.gamma_per_second, lambda_per_second=args.lambda_per_second,
                entropy_coef=args.entropy_coef)
            updates += 1
            row = {"update": updates, "decisions": decisions, "completed_episodes": completed,
                "truncated_sequences": truncated_count, "wall_s": time.perf_counter() - started,
                **metrics}
            # Batch decisions are already reported by ppo_update; keep the cumulative counter explicit.
            row["batch_decisions"] = metrics["decisions"]
            row["decisions"] = decisions
            append_json(run / "updates.jsonl", row)
            checkpoint("latest.pt")
            write_json(run / "status.json", {"status": "running", "target_decisions": args.decisions, **row})
            print(json.dumps(jsonable(row), allow_nan=False), flush=True)
        checkpoint("final.pt")
        change = math.sqrt(sum(float((value.detach() - initial[key]).double().square().sum())
                               for key, value in model.state_dict().items()))
        summary = {"status": "complete", "scenario_id": SCENARIO_ID, "memory": args.memory,
            "decisions": decisions, "updates": updates, "sequence_count": sequence_count,
            "completed_training_episodes": completed, "truncated_sequences": truncated_count,
            "target_decisions": args.decisions, "actual_budget_meets_target": decisions >= args.decisions,
            "budget_overshoot_decisions": decisions - args.decisions,
            "model_weight_change_l2": change, "all_parameters_finite": all(bool(torch.isfinite(p).all()) for p in model.parameters()),
            "real_nav2_in_loop": False, "official_UL_UC_training": False,
            "wall_s": time.perf_counter() - started, "evaluation_status": "independent_frozen_common_seed_evaluation_required"}
        write_json(run / "summary.json", summary)
        write_json(run / "status.json", summary)
        print(json.dumps(summary), flush=True)
        return summary
    except BaseException as exc:
        checkpoint("interrupted.pt", episode_boundary=False)
        write_json(run / "status.json", {"status": "error", "decisions": decisions,
            "updates": updates, "error_type": type(exc).__name__, "error": str(exc),
            "real_nav2_in_loop": False, "official_UL_UC_training": False})
        raise


if __name__ == "__main__":
    train(parse_args())
