#!/usr/bin/env python3
"""Small on-policy PPO updates from live Point-LIO/Nav2/MPPI transitions.

The ROS transport, geometry gate and rewards belong to LiveNav2Env. This
trainer never substitutes an execution proxy or imports a historical policy.
Each episode starts a new owned simulator session; physics cannot be resumed.
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

from rl_bt.uc_support_policy_v2 import UCSupportV2ActorCritic
from rl_bt.live_nav_observation import LIVE_NAV_SCENARIO_ID, live_nav_bt_baseline
from rl_bt.live_actor_learning import live_ppo_update
from scripts.train_uc_support_v2 import (observation_tensors,
    weights_sha256, balanced_indices, validate_bc_records, ANCHOR_ACTION_GROUPS)

SCENARIO_ID = LIVE_NAV_SCENARIO_ID
CHECKPOINT_SCHEMA = 1
OBSERVATION_SCHEMA = {"state": [24], "candidates": [16, 12], "mask": [16]}
CONTRACT_SOURCES = ("rl_bt/live_nav_observation.py", "scripts/nav2_rl_session.py")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def goal_schedule(args):
    """Use a frozen engineering-goal cycle; no simulator state is consulted."""
    if args.goal_schedule is None:
        goals = [args.goal]
    else:
        goals = json.loads(Path(args.goal_schedule).read_text())
        if isinstance(goals, dict):
            goals = goals.get("goals")
    array = np.asarray(goals, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 1 or not np.isfinite(array).all():
        raise ValueError("Goal schedule must be a nonempty JSON list of finite XY pairs")
    return array.tolist()


def validate_checkpoint_contract(checkpoint):
    if checkpoint.get("checkpoint_schema") != CHECKPOINT_SCHEMA or \
            checkpoint.get("scenario_id") != SCENARIO_ID:
        raise ValueError("Checkpoint is not from the current live navigation scenario")
    metadata = checkpoint.get("metadata", {})
    if metadata.get("observation_schema") != OBSERVATION_SCHEMA:
        raise ValueError("Checkpoint observation schema differs from live navigation")
    for relative in CONTRACT_SOURCES:
        if metadata.get("implementation_sha256", {}).get(relative) != file_sha256(ROOT / relative):
            raise ValueError(f"Checkpoint live contract source mismatch: {relative}")


def load_live_demonstrations(teacher_run):
    """Accept only actual, same-contract BT training traces and copied proof."""
    teacher_run = Path(teacher_run).resolve()
    summary = json.loads((teacher_run / "summary.json").read_text())
    config = json.loads((teacher_run / "config.json").read_text())
    if summary.get("scenario_id") != SCENARIO_ID or summary.get("policy") != "bt" or \
            config.get("policy") != "bt" or not summary.get("real_nav2_in_loop") or \
            summary.get("status") != "complete":
        raise ValueError("BC requires a complete verified live BT run, never mock/proxy/RL traces")
    if config.get("data_role") != "train_demonstration" or config.get("evaluation_only", False):
        raise ValueError("Evaluation demonstrations must not enter training")
    if config.get("observation_schema") != OBSERVATION_SCHEMA:
        raise ValueError("BT demonstration schema differs from live navigation")
    for relative in CONTRACT_SOURCES:
        expected = file_sha256(ROOT / relative)
        copied = teacher_run / "source" / relative
        if config.get("implementation_sha256", {}).get(relative) != expected or \
                not copied.is_file() or file_sha256(copied) != expected:
            raise ValueError(f"BT demonstration live contract source mismatch: {relative}")
    runtime = verify_live_artifacts(teacher_run)
    if len(runtime) != int(summary.get("completed_episodes", -1)) or \
            not runtime or not all(row["runtime_verified"] for row in runtime):
        raise ValueError("BT demonstration copied runtime evidence does not verify all episodes")
    recorded_runtime = json.loads((teacher_run / "runtime_evidence.json").read_text())
    if [(row.get("fixture_sha256"), row.get("runtime_verified")) for row in runtime] != \
            [(row.get("fixture_sha256"), row.get("runtime_verified")) for row in recorded_runtime]:
        raise ValueError("BT demonstration runtime proof changed after collection")
    records = {key: [] for key in ("state", "candidates", "mask", "actions")}
    rows = [json.loads(line) for line in (teacher_run / "transitions.jsonl").read_text().splitlines()]
    if not rows or len(rows) != int(summary.get("decisions", -1)):
        raise ValueError("BT demonstration decision count mismatch")
    for row in rows:
        if row.get("policy") != "bt" or not _transition_evidence(row.get("info", {}))["verified"]:
            raise ValueError("BT labels must originate in verified live RPC transitions")
        original = row["observation"]
        if np.asarray(original["mask"]).dtype != np.bool_:
            raise ValueError("BT demonstration masks must preserve recorded boolean values")
        observation = {"state": np.asarray(original["state"], np.float32),
            "candidates": np.asarray(original["candidates"], np.float32),
            "mask": np.asarray(original["mask"], bool)}
        observation_tensors([observation])
        original_action = row["proposed_action"]
        if not isinstance(original_action, int) or isinstance(original_action, bool):
            raise ValueError("BT demonstration action must be an integer intent")
        action = int(original_action)
        if not 0 <= action < 16 or not observation["mask"][action]:
            raise ValueError("BT teacher label is rejected by its recorded mask")
        if action != bt_control(observation):
            raise ValueError("Recorded BT action differs from the frozen observation-only teacher")
        for key in observation:
            records[key].append(observation[key])
        records["actions"].append(action)
    records = {key: np.stack(value) for key, value in records.items()}
    validate_bc_records(records)
    proof = {"source_run": str(teacher_run), "scenario_id": SCENARIO_ID,
        "policy": "bt", "data_role": "train_demonstration", "records": len(rows),
        "source_sha256": {name: file_sha256(teacher_run / name) for name in
            ("summary.json", "config.json", "runtime_evidence.json", "transitions.jsonl")},
        "contract_sha256": {name: file_sha256(ROOT / name) for name in CONTRACT_SOURCES},
        "runtime_evidence": runtime, "labels_are_on_policy_PPO": False}
    return records, proof


def fit_live_bc_actor(model, records, *, updates, seed):
    """Balance represented intent classes; optimize actor only, never critic."""
    validate_bc_records(records)
    if updates < 1:
        raise ValueError("BC updates must be positive")
    rng = np.random.default_rng(seed)
    batch = {key: torch.as_tensor(value) for key, value in records.items()}
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
    losses = []
    for _ in range(updates):
        idx = torch.as_tensor(balanced_indices(records["actions"], min(64, len(records["actions"])), rng))
        logits = model.actor(batch["state"][idx], batch["candidates"][idx], batch["mask"][idx])
        loss = F.cross_entropy(logits, batch["actions"][idx])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite live BC actor loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.)
        if not bool(torch.isfinite(grad)):
            raise FloatingPointError("Non-finite live BC actor gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        predicted = model.actor(batch["state"], batch["candidates"], batch["mask"]).argmax(-1)
    groups = {}
    for name, ids in ANCHOR_ACTION_GROUPS.items():
        selected = torch.as_tensor(np.isin(records["actions"], ids))
        groups[name] = {"count": int(selected.sum()), "accuracy":
            float((predicted[selected] == batch["actions"][selected]).float().mean())
            if bool(selected.any()) else None}
    return {"updates": updates, "actor_only": True, "learning_rate": 1e-3,
        "balanced_represented_classes": True, "mean_loss": float(np.mean(losses)),
        "training_label_accuracy": float((predicted == batch["actions"]).float().mean()),
        "original_label_counts_and_accuracy": groups}


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
        handle.flush()


def duration_gae(rewards, values, next_values, terminated, truncated, elapsed_s,
                 *, gamma=.99, lam=.95, reference_dt_s=.5):
    """SMDP discount by real simulation duration, not host RPC wall latency.

    Terminal states have zero value. External truncations retain the final
    observation value while stopping the advantage chain across episodes.
    """
    arrays = [np.asarray(v, dtype=np.float64) for v in
              (rewards, values, next_values, elapsed_s)]
    term, trunc = np.asarray(terminated, bool), np.asarray(truncated, bool)
    shape = arrays[0].shape
    if len(shape) != 1 or any(v.shape != shape for v in (*arrays, term, trunc)):
        raise ValueError("SMDP GAE needs matching one-dimensional arrays")
    if not all(np.isfinite(v).all() for v in arrays) or np.any(arrays[3] < 0):
        raise ValueError("SMDP GAE requires finite rewards/values and nonnegative durations")
    if not 0 < gamma <= 1 or not 0 < lam <= 1 or reference_dt_s <= 0:
        raise ValueError("Invalid SMDP discount parameters")
    advantages = np.zeros(shape, dtype=np.float64)
    carry = 0.
    for i in reversed(range(len(advantages))):
        exponent = arrays[3][i] / reference_dt_s
        discount, trace = gamma ** exponent, lam ** exponent
        delta = arrays[0][i] + discount * float(not term[i]) * arrays[2][i] - arrays[1][i]
        carry = delta + discount * trace * float(not (term[i] or trunc[i])) * carry
        advantages[i] = carry
    return advantages.astype(np.float32), (advantages + arrays[1]).astype(np.float32)


def bt_control(observation):
    """Use the canonical observation-only commissioning policy."""
    return live_nav_bt_baseline(observation)


def make_batch(records):
    arrays = {key: np.stack([r[key] for r in records]) for key in records[0]}
    advantages, returns = duration_gae(arrays["rewards"], arrays["values"],
        arrays["next_values"], arrays["terminated"], arrays["truncated"], arrays["elapsed_s"])
    batch = {key: torch.as_tensor(value) for key, value in arrays.items()}
    for key in ("old_logp", "values", "next_values", "rewards"):
        batch[key] = batch[key].float()
    batch["actions"] = batch["actions"].long()
    batch["advantages"], batch["returns"] = torch.from_numpy(advantages), torch.from_numpy(returns)
    return batch


def source_snapshot(run):
    paths = sorted((ROOT / "rl_bt").glob("*.py")) + [Path(__file__).resolve(),
        ROOT / "scripts/train_uc_support_v2.py", ROOT / "scripts/nav2_rl_session.py",
        ROOT / "scripts/run_nav2_rl_session.py", ROOT / "scripts/nav2_mujoco_fixture.py",
        ROOT / "scripts/launch_nav2_probe.py", ROOT / "configs/nav2_probe.yaml",
        ROOT / "configs/nav2_probe_bt.xml", ROOT / "scripts/evaluate_live_nav2.py"]
    hashes = {}
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        target = run / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    return hashes


def save_checkpoint(path, model, actor_optimizer, critic_optimizer, metadata,
                    *, decisions, updates, episodes, rng):
    temporary = Path(path).with_suffix(".pt.tmp")
    torch.save({"checkpoint_schema": CHECKPOINT_SCHEMA, "scenario_id": SCENARIO_ID,
        "model": model.state_dict(), "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(), "metadata": metadata,
        "decisions": decisions, "updates": updates, "episodes": copy.deepcopy(episodes),
        "rng_states": {"python": random.getstate(), "numpy": np.random.get_state(),
                       "torch": torch.get_rng_state(), "trainer": rng.bit_generator.state},
        "exact_resume_supported": False,
        "resume_limitation": "Live ROS/MuJoCo physics and sensor queues are not serializable."}, temporary)
    temporary.replace(path)


def _transition_evidence(info):
    """Require explicit environment provenance; mock transitions never count."""
    provenance = info.get("provenance", {})
    raw = info.get("raw", {})
    remote = bool(info.get("real_nav2_in_loop", False) or
                  provenance.get("real_nav2_in_loop", False))
    rpc_id = info.get("request_id", info.get("rpc_request_id", raw.get("request_id")))
    return {"remote_provenance": remote, "request_id": rpc_id,
            "verified": remote and rpc_id is not None}


def verify_live_artifacts(run):
    """Read only this run's copied runtime proof after owned session cleanup."""
    rows = []
    for path in sorted((Path(run) / "sessions").glob("**/fixture/result.json")):
        execution = path.parent.parent
        controller = execution / "nav2/controller_server.log"
        try:
            fixture = json.loads(path.read_text())
            controller_text = controller.read_text() if controller.is_file() else ""
            commands = fixture.get("commands", [])
            nonzero = sum(math.hypot(float(row[1]), float(row[2])) > 1e-6 for row in commands)
            loaded = ("Created controller : FollowPath of type "
                      "nav2_mppi_controller::MPPIController") in controller_text
            field_hash = fixture.get("field_source", {}).get("prepared_mesh_sha256")
            strict = fixture.get("terrain_prior") == "observed"
            no_gt = fixture.get("GT_localization_input") is False
            valid = loaded and no_gt and strict and field_hash == (
                "93ff0118af176f388c171b239acfdcac72c17cbe0b523350d5d916d3236fe10b")
            rows.append({"execution_dir": str(execution), "fixture_sha256":
                hashlib.sha256(path.read_bytes()).hexdigest(), "runtime_verified": valid,
                "controller_plugin_loaded": loaded, "GT_localization_input": not no_gt,
                "observed_support": strict, "prepared_mesh_sha256": field_hash,
                "MPPI_command_samples": len(commands), "nonzero_MPPI_command_samples": nonzero,
                "estimated_pose_samples": len(fixture.get("estimated_poses", [])),
                "safety_stop_latched": fixture.get("safety_stop_latched")})
        except (ValueError, TypeError, IndexError, OSError) as exc:
            rows.append({"execution_dir": str(execution), "runtime_verified": False,
                         "error": f"{type(exc).__name__}: {exc}"})
    return rows


def train(args, *, env_factory=None, evaluation_checkpoint=None):
    if min(args.episodes, args.max_decisions, args.rollout, args.epochs) < 1:
        raise ValueError("Episode, decision, rollout and update counts must be positive")
    if args.episode_wall_timeout <= 0 or args.task_timeout_s <= 0:
        raise ValueError("Episode timeouts must be positive")
    if not np.isfinite([args.anchor_ce, args.anchor_kl]).all() or min(args.anchor_ce, args.anchor_kl) < 0:
        raise ValueError("BC anchor coefficients must be finite and nonnegative")
    evaluation_only = evaluation_checkpoint is not None
    if args.warmstart_live is not None and (args.policy != "rl" or evaluation_only):
        raise ValueError("BT warmstart is only valid for actual RL training")
    goals = goal_schedule(args)
    if env_factory is None:
        from rl_bt.live_nav_env import LiveNav2Env
        env_factory = LiveNav2Env
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = UCSupportV2ActorCritic()
    checkpoint_proof = None
    if evaluation_only:
        checkpoint = torch.load(evaluation_checkpoint, map_location="cpu", weights_only=False)
        validate_checkpoint_contract(checkpoint)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        checkpoint_proof = {"path": str(Path(evaluation_checkpoint).resolve()),
            "sha256": file_sha256(evaluation_checkpoint), "decisions": checkpoint["decisions"],
            "updates": checkpoint["updates"], "model_weights_sha256": weights_sha256(model.state_dict())}
    actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-4, eps=1e-5)
    critic_optimizer = torch.optim.Adam(model.critic_parameters(), lr=3e-4, eps=1e-5)
    initial_weights = copy.deepcopy(model.state_dict())
    anchored = args.warmstart_live is not None and (args.anchor_ce > 0 or args.anchor_kl > 0)
    metadata = {"scenario_id": SCENARIO_ID, "policy": args.policy, "seed": args.seed,
        "observation_schema": OBSERVATION_SCHEMA,
        "data_role": "evaluation" if evaluation_only else
            ("train_demonstration" if args.policy == "bt" else "on_policy_training"),
        "evaluation_only": evaluation_only,
        "fresh_initialization": not evaluation_only, "historical_checkpoint_loaded": checkpoint_proof,
        "algorithm": ("frozen greedy evaluation" if evaluation_only else
            ("on-policy PPO with live BT BC anchors" if anchored else
             "on-policy vanilla PPO")) if args.policy == "rl" else "BT control only",
        "episodes_requested": args.episodes, "max_decisions": args.max_decisions,
        "goal_xy_m": list(args.goal), "task_timeout_s": args.task_timeout_s,
        "goal_schedule_xy_m": goals, "goal_schedule_source":
            ({"path": str(Path(args.goal_schedule).resolve()), "sha256": file_sha256(args.goal_schedule)}
             if args.goal_schedule is not None else None),
        "episode_wall_timeout_s": args.episode_wall_timeout,
        "simulation_seed_base": args.simulation_seed_base,
        "simulation_seed_schedule": ([args.simulation_seed_base+i for i in range(args.episodes)]
            if args.simulation_seed_base is not None else None),
        "physics_identically_seeded_or_paired": False,
        "decision_dt_s": .5, "gamma_per_decision_interval": .99,
        "lambda_per_decision_interval": .95, "discount_uses_actual_sim_elapsed_s": True,
        "actor_learning_rate": 1e-4, "critic_learning_rate": 3e-4,
        "rollout": args.rollout, "epochs": args.epochs,
        "anchor_ce_coefficient": args.anchor_ce if anchored else 0.,
        "anchor_kl_coefficient": args.anchor_kl if anchored else 0.,
        "device": "cpu", "threads": 2, "exact_resume_supported": False,
        "actor_learning_revision": 1,
        "actor_eligibility": "applied_proposal_and_multiple_legal_actions",
        "critic_retains_all_actual_transitions": True,
        "implementation_sha256": source_snapshot(run),
        "limits": ["Short local navigation tasks, not match training or match win rate.",
            "No aiming, firing, opponent, shared economy, or low-level velocity policy.",
            "BT retains all observed-support and freshness gates.",
            "No exact resume of live simulator state; model/optimizer snapshots only."]}
    write_json(run / "config.json", metadata)
    decisions = updates = verified = 0
    actor_counts = {key: 0 for key in ("on_policy_actor_sample_count",
        "actor_excluded_not_applied_count", "actor_excluded_applied_single_choice_count",
        "actor_optimizer_steps", "critic_optimizer_steps", "anchor_only_actor_optimizer_steps")}
    episodes, sessions, buffer = [], [], []
    started = time.monotonic()
    save_checkpoint(run / "untrained.pt", model, actor_optimizer, critic_optimizer,
                    metadata, decisions=0, updates=0, episodes=[], rng=rng)
    write_json(run / "goal_schedule.json", goals)
    failure = None
    bc_records = teacher = None

    def update_pending():
        nonlocal updates, buffer
        if args.policy != "rl" or evaluation_only or not buffer:
            buffer = []
            return
        batch = make_batch(buffer)
        stats = live_ppo_update(model, actor_optimizer, critic_optimizer, batch,
            bc_records, teacher, rng, algorithm="anchored" if anchored else "vanilla", epochs=args.epochs,
            minibatch_size=min(16, len(buffer)), anchor_ce_coefficient=args.anchor_ce,
            anchor_kl_coefficient=args.anchor_kl,
            anchor_data_role="train_demonstration" if anchored else None)
        for key in actor_counts:
            actor_counts[key] += int(stats[key])
        updates += 1
        row = {"update": updates, "decisions": decisions,
               "rollout_decisions": len(buffer), "wall_s": time.monotonic() - started, **stats}
        append_json(run / "updates.jsonl", row)
        print(json.dumps(row, allow_nan=False), flush=True)
        buffer = []
        save_checkpoint(run / "latest.pt", model, actor_optimizer, critic_optimizer,
                        metadata, decisions=decisions, updates=updates, episodes=episodes, rng=rng)

    try:
        if args.warmstart_live is not None:
            records, proof = load_live_demonstrations(args.warmstart_live)
            np.savez_compressed(run / "live_BT_demonstrations.npz", **records)
            proof["copied_records_sha256"] = file_sha256(run / "live_BT_demonstrations.npz")
            write_json(run / "live_BT_demonstrations_proof.json", proof)
            report = fit_live_bc_actor(model, records, updates=args.bc_updates, seed=args.seed+1)
            write_json(run / "bc_report.json", report)
            metadata["live_BT_warmstart"] = {"proof": proof, "BC": report}
            bc_records = {key: torch.as_tensor(value) for key, value in records.items()}
            teacher = copy.deepcopy(model).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            torch.save({"scenario_id": SCENARIO_ID, "model": teacher.state_dict(),
                        "weights_sha256": weights_sha256(teacher.state_dict())}, run / "BC_teacher.pt")
            write_json(run / "config.json", metadata)
        save_checkpoint(run / "initial.pt", model, actor_optimizer, critic_optimizer,
            metadata, decisions=0, updates=0, episodes=[], rng=rng)
        learning_initial_weights = copy.deepcopy(model.state_dict())
        for episode_index in range(args.episodes):
            if decisions >= args.max_decisions:
                break
            env = env_factory(run_dir=run / "sessions" / f"episode_{episode_index:03d}",
                goal=tuple(goals[episode_index % len(goals)]), wall_timeout_s=args.episode_wall_timeout,
                task_timeout_s=args.task_timeout_s, decision_dt_s=.5,
                simulation_seed=(None if args.simulation_seed_base is None else
                    args.simulation_seed_base+episode_index))
            total, episode_steps, done = 0., 0, False
            episode_started = time.monotonic()
            try:
                observation = env.reset()
                observation_tensors([observation])
                while decisions < args.max_decisions:
                    tensors = observation_tensors([observation])
                    with torch.no_grad():
                        distribution, value = model.distribution_value(**tensors)
                        if args.policy == "bt":
                            action = bt_control(observation)
                        elif evaluation_only:
                            action = int(distribution.logits.argmax(-1).item())
                        else:
                            action = int(distribution.sample().item())
                        old_logp = float(distribution.log_prob(torch.tensor([action])).item())
                    if not observation["mask"][action]:
                        raise RuntimeError("Policy proposed a BT-masked live navigation intent")
                    following, reward, done, info = env.step(action)
                    observation_tensors([following])
                    if not np.isfinite(reward):
                        raise FloatingPointError("Live reward must be finite")
                    terminated, truncated = bool(info["terminated"]), bool(info["truncated"])
                    if done != (terminated or truncated):
                        raise ValueError("Live done differs from terminated/truncated flags")
                    if "elapsed_s" not in info:
                        raise ValueError("Live transition lacks actual simulation elapsed_s")
                    duration = float(info["elapsed_s"])
                    if not np.isfinite(duration) or duration < 0:
                        raise ValueError("Invalid actual simulation transition duration")
                    decisions += 1
                    episode_steps += 1
                    total += float(reward)
                    budget_truncated = not done and decisions == args.max_decisions
                    truncated = truncated or budget_truncated
                    with torch.no_grad():
                        _, following_value = model.distribution_value(**observation_tensors([following]))
                    record = {key: np.asarray(observation[key]).copy()
                              for key in ("state", "candidates", "mask")}
                    applied = info.get("action_applied", True)
                    if type(applied) is not bool:
                        raise ValueError("Live action_applied must preserve explicit Boolean provenance")
                    record.update(actions=action, old_logp=old_logp, values=float(value.item()),
                        next_values=float(following_value.item()), rewards=float(reward),
                        terminated=terminated, truncated=truncated, elapsed_s=duration,
                        action_applied=applied)
                    if not evaluation_only:
                        buffer.append(record)
                    evidence = _transition_evidence(info)
                    verified += int(evidence["verified"])
                    append_json(run / "transitions.jsonl", {"decision": decisions,
                        "episode": episode_index, "episode_step": episode_steps,
                        "policy": args.policy, "proposed_action": action,
                        "evaluation_only": evaluation_only,
                        "goal_xy_m": goals[episode_index % len(goals)],
                        "observation": observation, "following_observation": following,
                        "log_probability": old_logp, "value": float(value.item()),
                        "reward": reward, "elapsed_s": duration, "terminated": terminated,
                        "truncated": truncated, "decision_budget_truncated": budget_truncated,
                        "remote_evidence": evidence, "info": info})
                    write_json(run / "status.json", {"status": "running", "episode": episode_index,
                        "decisions": decisions, "updates": updates, "episode_return": total,
                        "last_nav_status": info.get("nav_status"), "wall_s": time.monotonic()-started})
                    observation = following
                    if len(buffer) >= args.rollout:
                        update_pending()
                    if done or budget_truncated:
                        episode = {"episode": episode_index, "steps": episode_steps,
                            "goal_xy_m": goals[episode_index % len(goals)],
                            "return": total, "success": False, "terminated": terminated,
                            "truncated": truncated, "decision_budget_truncated": budget_truncated,
                            "wall_s": time.monotonic()-episode_started,
                            **copy.deepcopy(info.get("episode", {}))}
                        episodes.append(episode)
                        append_json(run / "episodes.jsonl", episode)
                        break
            finally:
                close_value = env.close()
                close_report = close_value if isinstance(close_value, dict) else getattr(env, "close_report", {})
                sessions.append({"episode": episode_index, "directory": str(run / "sessions" /
                    f"episode_{episode_index:03d}"), "close_report": jsonable(close_report)})
                write_json(run / "sessions.json", sessions)
        update_pending()
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        write_json(run / "error.json", failure)
        # Already-collected live transitions remain useful and auditable. Do not
        # update a partial rollout after an unknown transport failure.
    finally:
        artifact_evidence = verify_live_artifacts(run)
        write_json(run / "runtime_evidence.json", artifact_evidence)
        actor_delta_sq = sum(float((model.state_dict()[key]-value).square().sum())
            for key, value in initial_weights.items() if key.startswith("actor."))
        critic_delta_sq = sum(float((model.state_dict()[key]-value).square().sum())
            for key, value in initial_weights.items() if not key.startswith("actor."))
        summary = {"scenario_id": SCENARIO_ID, "policy": args.policy,
            "evaluation_only": evaluation_only,
            "status": "error" if failure else "complete", "error": failure,
            "decisions": decisions, "ppo_updates": updates,
            "completed_episodes": len(episodes), "successes": sum(bool(e.get("success")) for e in episodes),
            "episodes": episodes, "verified_live_rpc_transitions": verified,
            "real_nav2_in_loop": bool(decisions and verified == decisions and
                artifact_evidence and all(row["runtime_verified"] for row in artifact_evidence)),
            "runtime_verified_session_count": sum(row["runtime_verified"] for row in artifact_evidence),
            "MPPI_command_samples": sum(row.get("MPPI_command_samples", 0) for row in artifact_evidence),
            "nonzero_MPPI_command_samples": sum(row.get("nonzero_MPPI_command_samples", 0)
                                                  for row in artifact_evidence),
            "actor_weight_change_l2": actor_delta_sq ** .5,
            "critic_weight_change_l2": critic_delta_sq ** .5,
            "all_parameters_finite": all(bool(torch.isfinite(p).all()) for p in model.parameters()),
            "initial_model_weights_sha256": weights_sha256(initial_weights),
            "final_model_weights_sha256": weights_sha256(model.state_dict()),
            "wall_s": time.monotonic()-started,
            "navigation_command_evidence": "runtime_evidence.json; RPC provenance alone does not prove motion or task success.",
            "metric": "Live local navigation task completion; not match win rate",
            "exact_resume_supported": False}
        summary["actor_learning_revision"] = 1
        summary["actor_learning_counts"] = actor_counts
        if "learning_initial_weights" in locals():
            summary["PPO_actor_weight_change_l2"] = sum(float(
                (model.state_dict()[key]-value).square().sum()) for key, value in
                learning_initial_weights.items() if key.startswith("actor.")) ** .5
        if evaluation_only:
            summary["frozen_checkpoint"] = checkpoint_proof
            summary["greedy_actions_only"] = True
            summary["weights_unchanged"] = summary["initial_model_weights_sha256"] == summary["final_model_weights_sha256"]
        save_checkpoint(run / "final.pt", model, actor_optimizer, critic_optimizer,
            {**metadata, "verified_live_rpc_transitions": verified},
            decisions=decisions, updates=updates, episodes=episodes, rng=rng)
        write_json(run / "summary.json", summary)
        write_json(run / "status.json", summary)
    if failure:
        raise RuntimeError(f"Live training failed; artifacts preserved in {run}: {failure['message']}")
    return summary


def evaluate(args, *, env_factory=None):
    """Frozen greedy policy; never runs BC, PPO, or any optimizer step."""
    return train(args, env_factory=env_factory, evaluation_checkpoint=args.checkpoint)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--policy", choices=("rl", "bt"), default="rl")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-decisions", type=int, default=64)
    parser.add_argument("--rollout", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--simulation-seed-base", type=int)
    parser.add_argument("--episode-wall-timeout", type=float, default=180.)
    parser.add_argument("--task-timeout-s", type=float, default=20.)
    parser.add_argument("--goal", type=float, nargs=2, default=(0., .4))
    parser.add_argument("--goal-schedule", type=Path)
    parser.add_argument("--warmstart-live", type=Path)
    parser.add_argument("--bc-updates", type=int, default=200)
    parser.add_argument("--anchor-ce", type=float, default=.05)
    parser.add_argument("--anchor-kl", type=float, default=.05)
    args = parser.parse_args(argv)
    if args.simulation_seed_base is not None and args.simulation_seed_base < 0:
        parser.error("Simulation seed base must be nonnegative")
    return args


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), ensure_ascii=False, allow_nan=False), flush=True)
