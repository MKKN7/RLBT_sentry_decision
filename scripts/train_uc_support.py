#!/usr/bin/env python3
"""CPU PPO for UC observed-support execution courses, not full-match training.

The actor and PPO update are reused from the UL curriculum; the critic also
sees candidate availability. UC
checkpoints, data splits and time-limit-aware GAE are independent. --decisions
is the total PPO target, including decisions already present in --resume.
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

from rl_bt.ul_curriculum_policy import (
    CANDIDATE_COUNT, CANDIDATE_DIM, STATE_DIM, ULCurriculumActorCritic,
    choose, tensor_observations)
from scripts.train_ul_curriculum import ppo_update
from rl_bt.uc_support_policy import UCSupportActorCritic

VALIDATION_SEEDS = tuple(range(85001, 85033))
TEST_SEEDS = tuple(range(95001, 95065))
RESERVED_SEEDS = frozenset(VALIDATION_SEEDS + TEST_SEEDS)
CHECKPOINT_SCHEMA = 1
OBSERVATION_SCHEMA = {"state": [STATE_DIM],
                      "candidates": [CANDIDATE_COUNT, CANDIDATE_DIM],
                      "mask": [CANDIDATE_COUNT]}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n")
    temporary.replace(path)


def gae(rewards, values, next_values, terminated, truncated, *, gamma=.99, lam=.95):
    """Bootstrap a time limit, but never carry GAE across any episode boundary."""
    rewards, values, next_values = (np.asarray(x, dtype=np.float32)
                                    for x in (rewards, values, next_values))
    terminated, truncated = (np.asarray(x, dtype=bool) for x in (terminated, truncated))
    if rewards.ndim != 1 or not all(x.shape == rewards.shape for x in
            (values, next_values, terminated, truncated)):
        raise ValueError("GAE requires identical one-dimensional arrays")
    if not all(np.isfinite(x).all() for x in (rewards, values, next_values)):
        raise ValueError("GAE inputs must be finite")
    advantage = np.zeros_like(rewards)
    carry = 0.
    for step in reversed(range(len(rewards))):
        bootstrap = float(not terminated[step])
        continuing = float(not (terminated[step] or truncated[step]))
        delta = rewards[step] + gamma * bootstrap * next_values[step] - values[step]
        carry = delta + gamma * lam * continuing * carry
        advantage[step] = carry
    return advantage, advantage + values


def next_training_seed(rng):
    """Episode seeds are disjoint from both held-out split seed ranges."""
    while True:
        seed = int(rng.integers(1000, 2 ** 31))
        if seed not in RESERVED_SEEDS:
            return seed


def source_hashes(run=None):
    paths = sorted((ROOT / "rl_bt").glob("*.py")) + [Path(__file__).resolve(),
            ROOT / "scripts/train_ul_curriculum.py"]
    result = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        data = path.read_bytes()
        result[relative.as_posix()] = hashlib.sha256(data).hexdigest()
        if run is not None:
            destination = run / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    return result


def dataset_hashes(dataset_dir):
    """Fingerprint every dataset artifact, including NPZ/calibration/split manifest."""
    result = {}
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".npz"):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            result[path.relative_to(dataset_dir).as_posix()] = digest.hexdigest()
    if not result or not any(path.endswith(".npz") for path in result):
        raise ValueError("A prepared UC support dataset with NPZ and metadata is required")
    return result


def rng_state(rng):
    return {"python": random.getstate(), "numpy_global": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "minibatch_numpy": copy.deepcopy(rng.bit_generator.state)}


def restore_rng(saved, rng):
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy_global"])
    torch.set_rng_state(saved["torch_cpu"])
    rng.bit_generator.state = copy.deepcopy(saved["minibatch_numpy"])


def save_checkpoint(path, model, optimizer, metadata, rng, environment,
                    observation, *, decisions, updates, episodes, episode_seed,
                    wall_s):
    payload = {"checkpoint_schema": CHECKPOINT_SCHEMA, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "metadata": metadata,
        "rng_states": rng_state(rng), "environment_state": environment.state_dict(),
        "current_observation": copy.deepcopy(observation), "decisions": decisions,
        "updates": updates, "episodes": copy.deepcopy(episodes),
        "episode_seed": episode_seed, "wall_s": wall_s, "resume_supported": True}
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, metadata, rng, environment):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = payload["metadata"]
    if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("UC checkpoint schema mismatch")
    for key in ("scenario_id", "observation_schema", "implementation_sha256",
                "dataset_sha256", "seed", "rollout", "epochs", "eval_split"):
        if saved.get(key) != metadata.get(key):
            raise ValueError(f"UC resume mismatch: {key}")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    environment.load_state_dict(payload["environment_state"])
    tensor_observations([payload["current_observation"]])
    restore_rng(payload["rng_states"], rng)
    return payload


def truncate_logs_to_checkpoint(run, *, decisions, episode_count):
    """Remove only unsaved append-log tails when resuming an earlier checkpoint."""
    for filename, predicate in (
            ("train.jsonl", lambda index, row: row["decisions"] <= decisions),
            ("episodes.jsonl", lambda index, row: index < episode_count)):
        path = run / filename
        if not path.is_file():
            continue
        kept = []
        for index, line in enumerate(path.read_text().splitlines()):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # An interrupted last write is not a completed training record.
                if index != len(path.read_text().splitlines()) - 1:
                    raise
                break
            if predicate(index, row):
                kept.append(line)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("\n".join(kept) + ("\n" if kept else ""))
        temporary.replace(path)


def collect_rollout(model, environment, observation, rng, *, count, episode_seed):
    records = {key: [] for key in ("state", "candidates", "mask", "actions",
        "old_logp", "values", "next_values", "rewards", "terminated", "truncated")}
    episodes, components, vetoes = [], [], []
    for _ in range(count):
        tensors = tensor_observations([observation])
        with torch.no_grad():
            distribution, value = model.distribution_value(**tensors)
            action = distribution.sample()
            old_logp = distribution.log_prob(action)
        selected = int(action.item())
        if not observation["mask"][selected]:
            raise RuntimeError("Policy sampled a masked UC candidate")
        after, reward, done, info = environment.step(selected)
        terminated, truncated = bool(info["terminated"]), bool(info["truncated"])
        if bool(done) != (terminated or truncated):
            raise ValueError("Environment done must agree with terminated/truncated")
        with torch.no_grad():
            _, next_value = model.distribution_value(**tensor_observations([after]))
        for key in ("state", "candidates", "mask"):
            records[key].append(observation[key].copy())
        for key, item in (("actions", selected), ("old_logp", float(old_logp.item())),
                ("values", float(value.item())), ("next_values", float(next_value.item())),
                ("rewards", float(reward)), ("terminated", terminated),
                ("truncated", truncated)):
            records[key].append(item)
        components.append(info.get("reward_components", {}))
        vetoes.append(bool(info.get("BT_veto", False)))
        if done:
            episodes.append({**info["episode"], "training_seed": episode_seed})
            episode_seed = next_training_seed(rng)
            observation = environment.reset(seed=episode_seed)
        else:
            observation = after
    arrays = {key: np.stack(value) for key, value in records.items()}
    advantages, returns = gae(arrays["rewards"], arrays["values"], arrays["next_values"],
                             arrays["terminated"], arrays["truncated"])
    batch = {key: torch.as_tensor(value) for key, value in arrays.items()}
    batch["old_logp"] = batch["old_logp"].float()
    batch["advantages"] = torch.as_tensor(advantages)
    batch["returns"] = torch.as_tensor(returns)
    return batch, observation, episode_seed, episodes, components, vetoes


def imitation_warmstart(model, environment_type, baseline, dataset_dir, *, decisions,
                        seed, run):
    rng = np.random.default_rng(seed + 1_000_003)
    episode_seed = next_training_seed(rng)
    environment = environment_type(seed=episode_seed, split="train", randomization=True,
                                   dataset_dir=dataset_dir)
    observation = environment.reset(seed=episode_seed)
    records = {key: [] for key in ("state", "candidates", "mask", "actions")}
    seeds = [episode_seed]
    for _ in range(decisions):
        action = int(baseline(observation))
        if not observation["mask"][action]:
            raise RuntimeError("BT teacher chose a masked UC candidate")
        for key in ("state", "candidates", "mask"):
            records[key].append(observation[key].copy())
        records["actions"].append(action)
        observation, _, done, _ = environment.step(action)
        if done:
            episode_seed = next_training_seed(rng)
            seeds.append(episode_seed)
            observation = environment.reset(seed=episode_seed)
    arrays = {key: np.stack(value) for key, value in records.items()}
    np.savez_compressed(run / "bt_warmstart.npz", **arrays)
    batch = {key: torch.as_tensor(value) for key, value in arrays.items()}
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
    losses = []
    for _ in range(10):
        order = rng.permutation(decisions)
        for begin in range(0, decisions, 64):
            index = torch.as_tensor(order[begin:begin + 64], dtype=torch.long)
            logits = model.actor(batch["state"][index], batch["candidates"][index], batch["mask"][index])
            loss = F.cross_entropy(logits, batch["actions"][index])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite UC imitation loss")
            optimizer.zero_grad()
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.)
            if not bool(torch.isfinite(gradient)):
                raise FloatingPointError("Non-finite UC imitation gradient")
            optimizer.step()
            losses.append(float(loss.item()))
    with torch.no_grad():
        prediction = model.actor(batch["state"], batch["candidates"], batch["mask"]).argmax(-1)
        accuracy = float((prediction == batch["actions"]).float().mean())
    report = {"decisions": decisions, "epochs": 10, "dataset_split": "train",
        "teacher_episode_seeds": seeds, "evaluation_samples_used": False,
        "teacher": "observation_only_uc_support_bt_baseline", "training_label_accuracy": accuracy,
        "mean_loss": float(np.mean(losses)),
        "interpretation": "Training-label imitation accuracy, not held-out task success."}
    write_json(run / "bt_warmstart_report.json", report)
    return report


def evaluate_policy(policy, environment_type, dataset_dir, *, split, seeds, max_steps=4096):
    rows = []
    for seed in seeds:
        environment = environment_type(seed=seed, split=split, randomization=True,
                                       dataset_dir=dataset_dir)
        observation = environment.reset(seed=seed)
        total = 0.
        for _ in range(max_steps):
            action = int(policy(observation))
            if not 0 <= action < CANDIDATE_COUNT or not observation["mask"][action]:
                raise RuntimeError("Evaluation policy chose a masked UC candidate")
            observation, reward, done, info = environment.step(action)
            total += float(reward)
            if done:
                episode = copy.deepcopy(info["episode"])
                if not np.isclose(total, episode["return"], rtol=1e-5, atol=1e-5):
                    raise ValueError("Episode return disagrees with step reward accounting")
                rows.append({"seed": seed, **episode})
                break
        else:
            raise RuntimeError(f"UC evaluation seed {seed} exceeded {max_steps} decisions")
    return rows


def summarize_evaluation(groups, seeds, split, scenario_id):
    for rows in groups.values():
        if [row["seed"] for row in rows] != list(seeds):
            raise ValueError("Paired UC evaluation seed order mismatch")
    paired = []
    for index, seed in enumerate(seeds):
        sources = {rows[index].get("source_run_id") for rows in groups.values()}
        if len(sources) != 1:
            raise ValueError("Paired UC evaluation must begin on identical source runs")
        paired.append({"seed": seed, "source_run_id": next(iter(sources)),
                       **{name: bool(rows[index]["success"]) for name, rows in groups.items()},
                       "score_100": {name: rows[index].get("score_100") for name, rows in groups.items()}})
    summary = {}
    for name, rows in groups.items():
        summary[name] = {"successes": sum(bool(row["success"]) for row in rows),
            "episodes": len(rows),
            "task_success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_return": float(np.mean([row["return"] for row in rows])),
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
            "mean_time_s": float(np.mean([row["time_s"] for row in rows])),
            "mean_BT_veto_count": float(np.mean([row.get("BT_veto_count", 0) for row in rows])),
            "mean_recoveries": float(np.mean([row.get("recoveries", 0) for row in rows]))}
        if all("score_100" in row for row in rows):
            summary[name]["mean_score_100"] = float(np.mean([row["score_100"] for row in rows]))
            summary[name]["score_100_p10_p50_p90"] = np.percentile(
                [row["score_100"] for row in rows], [10, 50, 90]).tolist()
            summary[name]["mean_score_components"] = {key: float(np.mean([
                row["score_components"][key] for row in rows]))
                for key in rows[0]["score_components"]}
            summary[name]["by_scenario"] = {}
            for scenario in sorted({row["parameters"]["scenario"] for row in rows}):
                subset = [row for row in rows if row["parameters"]["scenario"] == scenario]
                summary[name]["by_scenario"][scenario] = {"n": len(subset),
                    "successes": sum(row["success"] for row in subset),
                    "mean_score_100": float(np.mean([row["score_100"] for row in subset]))}
        if all("navigation_success" in row for row in rows):
            navigating = [row for row in rows if not row.get("parameters", {}).get("no_safe_goal", False)]
            summary[name]["navigation_only"] = {
                "episodes": len(navigating),
                "successes": sum(bool(row["navigation_success"]) for row in navigating)}
            waiting = [row for row in rows if row.get("parameters", {}).get("no_safe_goal", False)]
            summary[name]["wait_only"] = {"episodes": len(waiting),
                "successes": sum(bool(row["wait_success"]) for row in waiting)}
    differences = {}
    if "finalPPO_RL" in groups and all("score_100" in row for row in groups["finalPPO_RL"]):
        for label in ("postBC_RL", "BT"):
            delta = np.array([a["score_100"]-b["score_100"]
                for a,b in zip(groups["finalPPO_RL"], groups[label])])
            differences["finalPPO_minus_"+label] = {"mean_score_delta": float(delta.mean()),
                "positive": int((delta>1e-6).sum()), "negative": int((delta < -1e-6).sum()),
                "tied": int((np.abs(delta)<=1e-6).sum())}
    return {"scenario_id": scenario_id, "dataset_split": split,
        "seeds_predeclared_before_training": list(seeds),
        "metric": "observed_support_course_task_success_rate_not_match_win_rate",
        "identical_seed_and_execution_proxy": True,
        "checkpoint_selection": "fixed_final_no_best_checkpoint_selection_on_evaluation",
        "summary": summary, "paired_score_differences": differences,
        "paired_results": paired, "episodes": groups,
        "limits": ["Offline observed-support course with calibrated execution proxy; not live Nav2 training.",
            "No complete UC opponents, referee match lifecycle or team economy model.",
            "Source-run split is not evidence of new-map or robot generalization.",
            "Validation results may guide development; test results cannot select a checkpoint."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=int, default=32768,
                        help="Total PPO target, including decisions restored by --resume")
    parser.add_argument("--bc-decisions", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=271,
                        help="Global RNG seed; actual training episode seeds are >= 1000")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/derived/uc_support_training_v1")
    parser.add_argument("--rollout", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--eval-split", choices=("validation", "test", "none"), default="validation")
    args = parser.parse_args()
    if min(args.decisions, args.rollout, args.epochs) < 1 or args.bc_decisions < 0 or not 0 <= args.seed < 2 ** 32:
        parser.error("Counts must be positive, BC nonnegative, and global seed uint32")
    from rl_bt.uc_support_env import UCSupportEnv, UC_SUPPORT_SCENARIO_ID, uc_support_bt_baseline
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    run, dataset_dir = args.run.resolve(), args.dataset_dir.resolve()
    if args.resume:
        if args.resume.name == "untrained.pt":
            parser.error("Pre-BC untrained.pt is an evaluation baseline, not a resumable training checkpoint")
        if args.resume.resolve().parent != run or not (run / "config.json").is_file():
            parser.error("Resume must use a checkpoint in the existing --run directory")
    else:
        run.mkdir(parents=True, exist_ok=False)
    episode_seed = next_training_seed(rng)
    environment = UCSupportEnv(seed=episode_seed, split="train", randomization=True, dataset_dir=dataset_dir)
    observation = environment.reset(seed=episode_seed)
    tensor_observations([observation])
    model = UCSupportActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    if args.resume:
        metadata = json.loads((run / "config.json").read_text())
        current = {**metadata, "scenario_id": UC_SUPPORT_SCENARIO_ID,
            "observation_schema": OBSERVATION_SCHEMA,
            "implementation_sha256": source_hashes(), "dataset_sha256": dataset_hashes(dataset_dir),
            "seed": args.seed, "rollout": args.rollout, "epochs": args.epochs, "eval_split": args.eval_split}
        payload = load_checkpoint(args.resume, model, optimizer, current, rng, environment)
        if args.bc_decisions != metadata["bt_imitation_decisions"]:
            parser.error("Resume --bc-decisions must match the original run; BC is not repeated")
        if args.decisions < payload["decisions"]:
            parser.error("--decisions is a total target and must not be smaller than saved decisions")
        observation = payload["current_observation"]
        decisions, updates = payload["decisions"], payload["updates"]
        episodes, episode_seed = payload["episodes"], payload["episode_seed"]
        elapsed_before = payload["wall_s"]
        truncate_logs_to_checkpoint(run, decisions=decisions, episode_count=len(episodes))
        resume_log = {"checkpoint": str(args.resume.resolve()), "restored_decisions": decisions,
                      "total_target_decisions": args.decisions, "restore_exact_rng_env_optimizer": True}
        with (run / "resume.jsonl").open("a") as handle:
            handle.write(json.dumps(resume_log) + "\n")
    else:
        metadata = {"artifact_type": "UC_observed_support_execution_course_training",
            "scenario_id": UC_SUPPORT_SCENARIO_ID, "seed": args.seed,
            "ppo_decisions_target_initial": args.decisions, "bt_imitation_decisions": args.bc_decisions,
            "observation_schema": OBSERVATION_SCHEMA, "decision_dt_s": .5,
            "dataset_dir": str(dataset_dir), "dataset_manifest": environment.dataset_manifest,
            "evaluation_score": {"max": 100, "task_progress": 60,
                "time_efficiency": 20, "recovery_discipline": 10, "safety": 10},
            "reward": {"milestone": 1.5, "success": 3., "per_decision": -.02,
                "veto": -.15, "switch": -.06, "useful_recovery": -.03,
                "needless_recovery": -.15, "potential": "gamma*Phi(next)-Phi(now); Phi=-.5*fixed_mission_remaining_m; terminal=0"},
            "dataset_sha256": dataset_hashes(dataset_dir),
            "architecture": {"reuse": "UL candidate actor, independent UC candidate-aware critic, no UL weights",
                "actor_parameters": sum(p.numel() for p in model.actor.parameters()),
                "critic_parameters": sum(p.numel() for p in model.critic.parameters())},
            "optimizer": "Adam", "learning_rate": 3e-4, "gamma_per_decision": .99,
            "gae_lambda_per_decision": .95, "ppo_clip": .2, "entropy_coefficient": .01,
            "rollout": args.rollout, "epochs": args.epochs,
            "ppo_implementation": "scripts/train_ul_curriculum.py:ppo_update unchanged",
            "eval_split": args.eval_split, "validation_seeds": list(VALIDATION_SEEDS),
            "test_seeds": list(TEST_SEEDS), "train_episode_seed_min": 1000,
            "evaluation_seed_ranges_excluded_from_training": True,
            "device": "cpu", "threads": 2, "torch_version": str(torch.__version__),
            "resume_supported": True, "checkpoint_schema": CHECKPOINT_SCHEMA,
            "implementation_sha256": source_hashes(run),
            "limits": ["UC independent curriculum; cannot load UL/alphaRM checkpoints.",
                "Offline observed-support execution proxy, not live Nav2/MPPI or a complete UC match.",
                "Outputs choose masked task candidates; no aiming/fire/thermal/velocity policy."]}
        write_json(run / "config.json", metadata)
        for path in dataset_dir.rglob("*.json"):
            destination = run / "source" / "dataset" / path.relative_to(dataset_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
        decisions, updates, episodes, elapsed_before = 0, 0, [], 0.
        save_checkpoint(run / "untrained.pt", model, optimizer, metadata, rng, environment,
            observation, decisions=0, updates=0, episodes=[], episode_seed=episode_seed, wall_s=0.)
        write_json(run / "status.json", {"status": "imitation", "decisions": 0})
        if args.bc_decisions:
            imitation_warmstart(model, UCSupportEnv, uc_support_bt_baseline, dataset_dir,
                decisions=args.bc_decisions, seed=args.seed, run=run)
        save_checkpoint(run / "initial.pt", model, optimizer, metadata, rng, environment,
            observation, decisions=0, updates=0, episodes=[], episode_seed=episode_seed, wall_s=0.)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(run / "tb"), purge_step=decisions if args.resume else None)
    except ImportError:
        pass
    started = time.perf_counter()
    try:
        with (run / "train.jsonl").open("a" if args.resume else "w") as metrics_file, \
                (run / "episodes.jsonl").open("a" if args.resume else "w") as episode_file:
            while decisions < args.decisions:
                count = min(args.rollout, args.decisions - decisions)
                batch, observation, episode_seed, completed, components, vetoes = collect_rollout(
                    model, environment, observation, rng, count=count, episode_seed=episode_seed)
                decisions += count
                episodes.extend(completed)
                for episode in completed:
                    episode_file.write(json.dumps(episode, allow_nan=False) + "\n")
                episode_file.flush()
                row = ppo_update(model, optimizer, batch, rng, epochs=args.epochs)
                updates += 1
                recent = episodes[-32:]
                row.update({"update": updates, "decisions": decisions,
                    "completed_episodes": len(episodes),
                    "recent_task_success_rate": float(np.mean([e["success"] for e in recent])) if recent else None,
                    "recent_return": float(np.mean([e["return"] for e in recent])) if recent else None,
                    "recent_recoveries": float(np.mean([e.get("recoveries", 0) for e in recent])) if recent else None,
                    "recent_score_100": float(np.mean([e["score_100"] for e in recent]))
                        if recent and all("score_100" in e for e in recent) else None,
                    "mean_step_reward": float(batch["rewards"].mean()),
                    "mean_legal_candidates": float(batch["mask"].sum(-1).float().mean()),
                    "BT_veto_fraction": float(np.mean(vetoes)),
                    "wall_s": elapsed_before + time.perf_counter() - started})
                for key in sorted({key for sample in components for key in sample}):
                    row["reward_" + key + "_mean"] = float(np.mean([
                        sample.get(key, 0.) for sample in components]))
                metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
                metrics_file.flush()
                write_json(run / "status.json", {"status": "training", **row})
                if writer is not None:
                    for key, value in row.items():
                        if isinstance(value, (float, int)):
                            writer.add_scalar("train/" + key, value, decisions)
                    writer.flush()
                print(json.dumps(row, allow_nan=False), flush=True)
                if updates % 8 == 0:
                    save_checkpoint(run / "latest.pt", model, optimizer, metadata, rng, environment,
                        observation, decisions=decisions, updates=updates, episodes=episodes,
                        episode_seed=episode_seed, wall_s=row["wall_s"])
        wall_s = elapsed_before + time.perf_counter() - started
        save_checkpoint(run / "final.pt", model, optimizer, metadata, rng, environment,
            observation, decisions=decisions, updates=updates, episodes=episodes,
            episode_seed=episode_seed, wall_s=wall_s)
        evaluation = None
        if args.eval_split != "none":
            write_json(run / "status.json", {"status": "evaluating", "decisions": decisions})
            seeds = VALIDATION_SEEDS if args.eval_split == "validation" else TEST_SEEDS
            groups = {}
            for label, checkpoint in (("untrainedRL", "untrained.pt"),
                    ("postBC_RL", "initial.pt"), ("finalPPO_RL", "final.pt")):
                write_json(run / "status.json", {"status": "evaluating", "decisions": decisions,
                           "policy": label, "split": args.eval_split})
                evaluated_model = copy.deepcopy(model).eval()
                evaluated_model.load_state_dict(torch.load(run / checkpoint,
                    map_location="cpu", weights_only=False)["model"])
                groups[label] = evaluate_policy(lambda obs: choose(evaluated_model, obs),
                    UCSupportEnv, dataset_dir, split=args.eval_split, seeds=seeds)
            groups["BT"] = evaluate_policy(uc_support_bt_baseline, UCSupportEnv, dataset_dir,
                                           split=args.eval_split, seeds=seeds)
            evaluation = summarize_evaluation(groups, seeds, args.eval_split, UC_SUPPORT_SCENARIO_ID)
            write_json(run / "evaluation.json", evaluation)
        summary = {"status": "completed", "run": str(run), "scenario_id": UC_SUPPORT_SCENARIO_ID,
            "ppo_decisions": decisions, "bt_imitation_decisions": metadata["bt_imitation_decisions"],
            "completed_training_episodes": len(episodes), "updates": updates,
            "training_wall_s": wall_s, "total_wall_s": elapsed_before + time.perf_counter() - started,
            "evaluation": None if evaluation is None else evaluation["summary"],
            "metric": "observed_support_course_task_success_rate_not_match_win_rate",
            "resume_supported": True}
        write_json(run / "summary.json", summary)
        write_json(run / "status.json", summary)
        print(json.dumps(summary, allow_nan=False), flush=True)
    except BaseException as error:
        write_json(run / "status.json", {"status": "failed", "decisions": decisions,
                   "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
