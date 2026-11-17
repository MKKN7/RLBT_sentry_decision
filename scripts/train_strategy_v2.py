#!/usr/bin/env python3
"""Isolated sequence PPO for the role-balanced alphaRM strategy prototype.

The v1 model/environment/learning code remains immutable. This entrypoint
version-tags the new role-controller environment and fixes only the v2
single-sample/constant-advantage normalization edge case. No official UL/UC or
real Nav2/MPPI claim is made by these isolated strategy runs.
"""
from __future__ import annotations

import argparse
import copy
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

from rl_bt.alpharm_task_env import AlphaRuntime, TRAIN_STYLES
from rl_bt.strategy_scenarios import StrategyTaskEnv, DEFAULT_CONFIG, SUPPORT_MODES
from rl_bt.temporal_policy import TemporalActorCritic
from scripts.train_temporal_phase import (collect_episode, episode_gae,
    evaluate_sequence, observation_tensors, file_sha256, jsonable, write_json, append_json)

SCENARIO_ID = "alpharm_role_balanced_temporal_v2"
ENVIRONMENT_SCENARIO_ID = "alpharm_role_balanced_v2"
CHECKPOINT_SCHEMA = 2
OBSERVATION_SCHEMA = {"state": [148], "candidates": [16, 16], "mask": [16]}
OBSERVATION_SEMANTICS_ID = "unchanged_alpharm_task_state148_candidates16x16_mask16"
SCENE_CONFIG = "configs/strategy_scenarios_v2.json"
MANIFEST = "data/references/alpharm_jax_review_manifest.json"
UPSTREAM_PREFIX = "data/references/alpharm_jax_review/upstream/"
UPSTREAM_FILES = tuple(UPSTREAM_PREFIX + relative for relative in (
    "demo/web/actor.onnx", "demo/web/env_step.onnx", "demo/web/fixture.json",
    "jaxmarl/environments/rm/config.py"))
CONTRACT_SOURCES = ("rl_bt/strategy_scenarios.py", SCENE_CONFIG,
    "rl_bt/temporal_policy.py", "rl_bt/alpharm_task_env.py", "rl_bt/geometry.py",
    "rl_bt/rewards.py", "rl_bt/reward_components.py", "rl_bt/contracts.py",
    "scripts/probe_alpharm_onnx.py", "scripts/train_temporal_phase.py", MANIFEST)
SOURCE_FILES = CONTRACT_SOURCES + UPSTREAM_FILES + ("scripts/train_strategy_v2.py",)
VALIDATION_SEEDS = (62011, 62012)
TEST_SEEDS = (73001, 73002)
SEED_BASE, SEED_STRIDE = 800001, 10000


def validate_train_styles(styles):
    styles = tuple(styles)
    if not styles or len(styles) != len(set(styles)) or any(style not in TRAIN_STYLES for style in styles):
        raise ValueError("Training styles must be a nonempty unique subset of TRAIN_STYLES")
    return styles


def training_episode_seed(seed, sequence_index):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 or \
            isinstance(sequence_index, bool) or not isinstance(sequence_index, int) or \
            not 0 <= sequence_index < SEED_STRIDE:
        raise ValueError("Training seeds must be uint32; sequence index must remain in [0,10000)")
    value = SEED_BASE + seed * SEED_STRIDE + sequence_index
    if value in VALIDATION_SEEDS + TEST_SEEDS:
        raise ValueError("A training episode seed collides with reserved validation/test seeds")
    return value


def verify_upstream_files():
    manifest = json.loads((ROOT / MANIFEST).read_text())
    expected = {entry["local_path"]: entry["sha256"] for entry in manifest["files"]}
    actual = {}
    for relative in UPSTREAM_FILES:
        digest = file_sha256(ROOT / relative)
        if expected.get(relative) != digest:
            raise ValueError(f"Frozen upstream dependency changed: {relative}")
        actual[relative] = digest
    return manifest, actual


def validate_checkpoint(payload):
    metadata = payload.get("metadata", {})
    if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA or payload.get("scenario_id") != SCENARIO_ID:
        raise ValueError("Checkpoint is not a role-balanced temporal v2 experiment")
    if metadata.get("environment_scenario_id") != ENVIRONMENT_SCENARIO_ID or \
            metadata.get("observation_schema") != OBSERVATION_SCHEMA or \
            metadata.get("observation_semantics_id") != OBSERVATION_SEMANTICS_ID:
        raise ValueError("Checkpoint environment/observation contract differs from strategy v2")
    if metadata.get("real_nav2_in_loop") is not False or metadata.get("official_UL_UC_training") is not False:
        raise ValueError("Strategy prototype must not be labelled real/official navigation")
    if metadata.get("support_mode") not in SUPPORT_MODES:
        raise ValueError("Unknown strategy support mode")
    validate_train_styles(metadata.get("train_styles", ()))
    if metadata.get("scene_config_path") != SCENE_CONFIG or \
            metadata.get("scene_config_sha256") != file_sha256(ROOT / SCENE_CONFIG):
        raise ValueError("Checkpoint strategy scene config changed")
    for relative in CONTRACT_SOURCES:
        if metadata.get("implementation_sha256", {}).get(relative) != file_sha256(ROOT / relative):
            raise ValueError(f"Checkpoint contract source mismatch: {relative}")
    manifest, actual = verify_upstream_files()
    if metadata.get("upstream_files_sha256") != actual or metadata.get("upstream_commit") != manifest["commit"]:
        raise ValueError("Checkpoint upstream execution assets changed")
    if metadata.get("memory") not in ("mlp", "gru") or metadata.get("hidden_size") != 64:
        raise ValueError("Unknown strategy model architecture")
    if metadata.get("BC_updates") != 0 or metadata.get("historical_model_initialization") is not False:
        raise ValueError("Strategy v2 requires fresh model initialization without BC")
    if metadata.get("actor_advantage_normalization") != "mean_std_if_count>=2_and_std>=1e-6_else_raw":
        raise ValueError("Checkpoint advantage normalization differs from strategy v2")
    if metadata.get("seed_role") != "train" or metadata.get("validation_seeds") != list(VALIDATION_SEEDS) or \
            metadata.get("test_seeds") != list(TEST_SEEDS):
        raise ValueError("Checkpoint seed roles differ from the v2 split")


def load_strategy_checkpoint(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    validate_checkpoint(payload)
    model = TemporalActorCritic(memory=payload["metadata"]["memory"], hidden_size=64).cpu()
    model.load_state_dict(payload["model"], strict=True)
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise ValueError("Checkpoint model contains non-finite parameters")
    model.eval()
    return model, payload


def normalize_actor_advantages(advantages, *, epsilon=1e-6):
    """Keep a singleton or a constant-sign batch's real return signal."""
    advantages = np.asarray(advantages, np.float32)
    if advantages.ndim != 1 or not np.isfinite(advantages).all() or epsilon <= 0:
        raise ValueError("Actor advantages must be finite one-dimensional values")
    count = len(advantages)
    mean = float(advantages.astype(np.float64).mean()) if count else 0.
    std = float(advantages.astype(np.float64).std()) if count else 0.
    if not count:
        mode = "no_actor_decisions"
        result = advantages.copy()
    elif count < 2:
        mode = "raw_single_actor_decision"
        result = advantages.copy()
    elif std < epsilon:
        mode = "raw_low_variance"
        result = advantages.copy()
    else:
        mode = "mean_std"
        result = ((advantages.astype(np.float64) - mean) / std).astype(np.float32)
    return result, {"advantage_normalization_mode": mode,
        "actor_advantage_mean_raw": mean, "actor_advantage_std_raw": std,
        "advantage_normalization_epsilon": float(epsilon)}


def ppo_update(model, optimizer, sequences, *, epochs=4, gamma_per_second=.997,
               lambda_per_second=.995, clip=.2, entropy_coef=.001, value_coef=.5,
               max_grad_norm=.5):
    """Full chronological episode BPTT; v2 advantage normalization only."""
    if not sequences or epochs < 1:
        raise ValueError("PPO needs one or more episodes and positive epochs")
    if not 0 < clip < 1 or entropy_coef < 0 or value_coef <= 0 or max_grad_norm <= 0:
        raise ValueError("Invalid PPO optimization parameters")
    prepared = []
    for sequence in sequences:
        advantages, returns = episode_gae(sequence, gamma_per_second=gamma_per_second,
                                          lambda_per_second=lambda_per_second)
        legal = np.asarray(sequence["mask"], bool)
        actions = np.asarray(sequence["actions"], np.int64)
        if legal.shape != (len(actions), 16) or (actions < 0).any() or (actions >= 16).any() or \
                not legal[np.arange(len(actions)), actions].all():
            raise ValueError("PPO actions must be legal in their recorded masks")
        prepared.append((sequence, advantages, returns, legal.sum(axis=-1) > 1))
    raw_advantages = np.concatenate([advantage[valid] for _, advantage, _, valid in prepared])
    normalized, normalization = normalize_actor_advantages(raw_advantages)
    count = sum(len(row[0]["actions"]) for row in prepared)
    valid_count = len(raw_advantages)
    metrics = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        row = {"policy_loss": 0., "value_loss": 0., "entropy": 0., "approx_kl": 0., "clip_fraction": 0.}
        valid_offset = 0
        for sequence, _, returns, actor_valid in prepared:
            logp, predicted, entropy = evaluate_sequence(model, sequence)
            device = logp.device
            old_logp = torch.as_tensor(sequence["old_logp"], device=device, dtype=torch.float32)
            targets = torch.as_tensor(returns, device=device)
            valid = torch.as_tensor(actor_valid, device=device)
            value_loss = F.mse_loss(predicted, targets, reduction="sum") / count
            local_valid_count = int(actor_valid.sum())
            if local_valid_count:
                advantage = torch.as_tensor(normalized[valid_offset:valid_offset + local_valid_count], device=device)
                valid_offset += local_valid_count
                log_ratio = logp[valid] - old_logp[valid]
                ratio = log_ratio.exp()
                surrogate = torch.maximum(-advantage * ratio,
                    -advantage * ratio.clamp(1 - clip, 1 + clip))
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
                raise FloatingPointError("Non-finite strategy v2 PPO loss")
            loss.backward()
            for name, value in (("policy_loss", policy_loss), ("value_loss", value_loss),
                    ("entropy", entropy_loss), ("approx_kl", kl), ("clip_fraction", clipped)):
                row[name] += float(value.detach())
        if not valid_count:
            for parameter in model.actor.parameters():
                parameter.grad = None
        actor_norm = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), max_grad_norm)
        critic_norm = torch.nn.utils.clip_grad_norm_(model.critic.parameters(), max_grad_norm)
        if not bool(torch.isfinite(actor_norm) and torch.isfinite(critic_norm)):
            raise FloatingPointError("Non-finite strategy v2 PPO gradient")
        optimizer.step()
        row.update(actor_gradient_norm=float(actor_norm), critic_gradient_norm=float(critic_norm))
        metrics.append(row)
    result = {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}
    result.update({"epochs": epochs, "sequence_count": len(sequences), "decisions": count,
        "actor_decisions": valid_count, "forced_single_legal_decisions": count - valid_count,
        "recurrent_recomputation": "chronological_from_episode_start_full_BPTT",
        "individual_timesteps_shuffled": False, "first_epoch_approx_kl": metrics[0]["approx_kl"],
        **normalization})
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--memory", choices=("mlp", "gru"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--decisions", type=int, required=True)
    parser.add_argument("--support-mode", choices=SUPPORT_MODES, default="anchors")
    parser.add_argument("--train-styles", nargs="+", choices=TRAIN_STYLES, default=list(TRAIN_STYLES))
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--gamma-per-second", type=float, default=.997)
    parser.add_argument("--lambda-per-second", type=float, default=.995)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--entropy-coef", type=float, default=.001)
    parser.add_argument("--torch-threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--episode-max-decisions", type=int, default=400)
    args = parser.parse_args(argv)
    try:
        training_episode_seed(args.seed, 0)
        validate_train_styles(args.train_styles)
    except ValueError as error:
        parser.error(str(error))
    if min(args.decisions, args.episodes_per_update, args.epochs, args.episode_max_decisions) < 1 or \
            not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or \
            not math.isfinite(args.entropy_coef) or args.entropy_coef < 0:
        parser.error("Budgets/learning rate must be positive; entropy must be nonnegative and finite")
    if not 0 < args.gamma_per_second <= 1 or not 0 < args.lambda_per_second <= 1:
        parser.error("Discount and GAE lambda must lie in (0,1]")
    return args


def build_metadata(args, runtime):
    styles = validate_train_styles(args.train_styles)
    manifest, upstream_hashes = verify_upstream_files()
    scene_config = json.loads((ROOT / SCENE_CONFIG).read_text())
    if scene_config.get("scenario_id") != ENVIRONMENT_SCENARIO_ID or \
            scene_config.get("observation_semantics") != OBSERVATION_SEMANTICS_ID or \
            scene_config.get("real_nav2_in_loop") is not False or \
            scene_config.get("official_UL_UC_training") is not False:
        raise ValueError("Scene configuration has an incompatible contract")
    if Path(DEFAULT_CONFIG).resolve() != (ROOT / SCENE_CONFIG).resolve():
        raise ValueError("Environment default scene config changed")
    if runtime.source_commit != manifest["commit"]:
        raise ValueError("Runtime did not load the registered upstream commit")
    return {"scenario_id": SCENARIO_ID, "environment_scenario_id": ENVIRONMENT_SCENARIO_ID,
        "memory": args.memory, "hidden_size": 64, "seed": args.seed, "seed_role": "train",
        "training_episode_seed_scheme": "800001+seed*10000+zero_based_sequence_index",
        "training_episode_seed_namespace_size": SEED_STRIDE,
        "validation_seeds": list(VALIDATION_SEEDS), "test_seeds": list(TEST_SEEDS),
        "observation_schema": OBSERVATION_SCHEMA, "observation_semantics_id": OBSERVATION_SEMANTICS_ID,
        "train_styles": list(styles), "heldout_styles_used_for_training": False,
        "style_sampler": "trainer_numpy_rng_choice_from_exact_train_styles",
        "support_mode": args.support_mode, "scene_config_path": SCENE_CONFIG,
        "scene_config_sha256": file_sha256(ROOT / SCENE_CONFIG), "data_role": "training",
        "real_nav2_in_loop": False, "official_UL_UC_training": False,
        "reward_profile": "upstream", "physical_rule_reward_constants_changed": False,
        "reward_note": "unchanged upstream team reward with the existing .002 task-switch cost",
        "gamma_per_second": args.gamma_per_second, "lambda_per_second": args.lambda_per_second,
        "recurrent_training": "full_sequence_BPTT_from_zero_state_each_episode_each_epoch",
        "actor_advantage_normalization": "mean_std_if_count>=2_and_std>=1e-6_else_raw",
        "matched_budget_unit": "minimum_actual_candidate_decisions_then_finish_episode",
        "target_decisions": args.decisions, "episodes_per_update": args.episodes_per_update,
        "epochs": args.epochs, "episode_max_decisions": args.episode_max_decisions,
        "learning_rate": args.learning_rate, "entropy_coef": args.entropy_coef,
        "randomization": True, "torch_threads": args.torch_threads,
        "torch_version": torch.__version__, "device": "cpu",
        "upstream_commit": runtime.source_commit, "upstream_files_sha256": upstream_hashes,
        "upstream_manifest_sha256": file_sha256(ROOT / MANIFEST),
        "implementation_sha256": {relative: file_sha256(ROOT / relative) for relative in SOURCE_FILES},
        "excluded_outputs": ["velocity", "aim", "fire", "thermal_control"],
        "BC_updates": 0, "historical_model_initialization": False,
        "final_checkpoint_selection": "fixed_final_budget_no_evaluation_selection",
        "exact_resume_supported": False,
        "limits": ["Isolated role-balanced simplified 3v3 strategy scenario, not official UL/UC.",
            "No live Nav2/MPPI, tunnels, official stance or shared team economy.",
            "Noise/delay ranges are engineering assumptions, not calibrated match data.",
            "Changing support assignment does not establish an RL advantage."]}


def train(args):
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    training_episode_seed(args.seed, 0)
    validate_train_styles(args.train_styles)
    run = args.run.resolve()
    if run.exists():
        raise FileExistsError(f"Refusing to overwrite training artifacts or imply exact resume: {run}")
    runtime = AlphaRuntime()
    metadata = build_metadata(args, runtime)
    model = TemporalActorCritic(memory=args.memory, hidden_size=64).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    run.mkdir(parents=True, exist_ok=False)
    (run / "rollouts").mkdir()
    for relative in SOURCE_FILES:
        target = run / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
        if file_sha256(target) != metadata["implementation_sha256"][relative]:
            raise ValueError(f"Source changed while snapshotting: {relative}")
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
        write_json(run / "status.json", {"status": "running", "scenario_id": SCENARIO_ID,
            "decisions": 0, "target_decisions": args.decisions})
        while decisions < args.decisions:
            sequences = []
            for _ in range(args.episodes_per_update):
                if decisions >= args.decisions:
                    break
                episode_seed = training_episode_seed(args.seed, sequence_count)
                style = str(rng.choice(metadata["train_styles"]))
                environment = StrategyTaskEnv(runtime, episode_seed, support_mode=args.support_mode,
                    randomization=True, opponent_style=style, reward_profile="upstream",
                    config_path=ROOT / SCENE_CONFIG)
                if environment.style != style or environment.config_sha256 != metadata["scene_config_sha256"]:
                    raise ValueError("Factory did not retain the exact declared style/scene config")
                sequence = collect_episode(model, environment, args.episode_max_decisions, episode_seed)
                count = len(sequence["actions"])
                sequence_count += 1
                decisions += count
                episode_seeds.append(episode_seed)
                terminal = bool(sequence["terminated"][-1])
                completed += int(terminal)
                truncated_count += int(not terminal)
                arrays = {key: value for key, value in sequence.items() if isinstance(value, np.ndarray)}
                arrays["bootstrap_value"] = np.asarray(sequence["bootstrap_value"])
                np.savez_compressed(run / "rollouts" / f"sequence_{sequence_count:05d}.npz", **arrays)
                episode_row = {"sequence": sequence_count, "seed": episode_seed, "seed_role": "train",
                    "decisions": count, "cumulative_decisions": decisions,
                    "simulated_s": sequence["simulated_s"], "terminated": terminal,
                    "truncated": not terminal, "truncation_reason": sequence["truncation_reason"],
                    "opponent_style": style, "team": sequence["team"], "support_mode": args.support_mode,
                    "return": float(sequence["rewards"].sum()), "episode": sequence["episode"]}
                append_json(run / "episodes.jsonl", episode_row)
                sequences.append(sequence)
                write_json(run / "status.json", {"status": "collecting", **episode_row,
                    "updates": updates, "target_decisions": args.decisions,
                    "wall_s": time.perf_counter() - started})
                if not terminal:
                    raise RuntimeError("Episode reached the explicit decision limit; refusing a complete-episode training claim")
            metrics = ppo_update(model, optimizer, sequences, epochs=args.epochs,
                gamma_per_second=args.gamma_per_second, lambda_per_second=args.lambda_per_second,
                entropy_coef=args.entropy_coef)
            updates += 1
            row = {"update": updates, "completed_episodes": completed,
                "truncated_sequences": truncated_count, "wall_s": time.perf_counter() - started, **metrics}
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
            "support_mode": args.support_mode, "decisions": decisions, "updates": updates,
            "sequence_count": sequence_count, "completed_training_episodes": completed,
            "truncated_sequences": truncated_count, "target_decisions": args.decisions,
            "actual_budget_meets_target": decisions >= args.decisions,
            "budget_overshoot_decisions": decisions - args.decisions,
            "model_weight_change_l2": change,
            "all_parameters_finite": all(bool(torch.isfinite(p).all()) for p in model.parameters()),
            "seed_role": "train", "real_nav2_in_loop": False, "official_UL_UC_training": False,
            "wall_s": time.perf_counter() - started,
            "evaluation_status": "independent_frozen_validation_and_test_required"}
        write_json(run / "summary.json", summary)
        write_json(run / "status.json", summary)
        print(json.dumps(summary), flush=True)
        return summary
    except BaseException as error:
        checkpoint("interrupted.pt", episode_boundary=False)
        write_json(run / "status.json", {"status": "error", "decisions": decisions,
            "updates": updates, "error_type": type(error).__name__, "error": str(error),
            "real_nav2_in_loop": False, "official_UL_UC_training": False})
        raise


if __name__ == "__main__":
    train(parse_args())
