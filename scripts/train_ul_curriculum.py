#!/usr/bin/env python3
"""CPU PPO for UL ground-task courses; success rates are not match win rates."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from rl_bt.ul_curriculum_policy import (
    CANDIDATE_COUNT, CANDIDATE_DIM, SCENARIO_ID, STATE_DIM,
    ULCurriculumActorCritic, choose, tensor_observations)

DEFAULT_EVAL_SEEDS = tuple(range(51001, 51013))


def gae(rewards, values, next_values, done, *, gamma=.99, lam=.95):
    """Decision-step GAE; terminal resets never bootstrap into the next course."""
    rewards, values, next_values = (np.asarray(x, dtype=np.float32)
                                    for x in (rewards, values, next_values))
    done = np.asarray(done, dtype=bool)
    if not (rewards.shape == values.shape == next_values.shape == done.shape):
        raise ValueError("GAE arrays must have identical shapes")
    advantage = np.zeros_like(rewards)
    carry = 0.
    for step in reversed(range(len(rewards))):
        continuing = float(not done[step])
        delta = rewards[step] + gamma * next_values[step] * continuing - values[step]
        carry = delta + gamma * lam * continuing * carry
        advantage[step] = carry
    return advantage, advantage + values


def ppo_update(model, optimizer, batch, rng, *, epochs=4, minibatch_size=128,
               clip=.2, entropy_coefficient=.01):
    """One finite-checked PPO update over recorded on-policy observations."""
    count = len(batch["actions"])
    if not count or min(epochs, minibatch_size) < 1:
        raise ValueError("PPO requires nonempty samples and positive update counts")
    advantage = batch["advantages"].float()
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    losses = []
    for _ in range(epochs):
        order = rng.permutation(count)
        for begin in range(0, count, minibatch_size):
            index = torch.as_tensor(order[begin:begin + minibatch_size], dtype=torch.long)
            distribution, predicted = model.distribution_value(
                batch["state"][index], batch["candidates"][index], batch["mask"][index])
            logp = distribution.log_prob(batch["actions"][index])
            log_ratio = logp - batch["old_logp"][index]
            ratio = log_ratio.exp()
            policy_loss = torch.maximum(-advantage[index] * ratio,
                -advantage[index] * ratio.clamp(1. - clip, 1. + clip)).mean()
            value_loss = F.mse_loss(predicted, batch["returns"][index])
            entropy = distribution.entropy().mean()
            loss = policy_loss + .5 * value_loss - entropy_coefficient * entropy
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite UL PPO loss")
            optimizer.zero_grad()
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            if not bool(torch.isfinite(gradient)):
                raise FloatingPointError("Non-finite UL PPO gradient")
            optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                raise FloatingPointError("Non-finite UL PPO parameter")
            losses.append((policy_loss.item(), value_loss.item(), entropy.item(),
                           ((ratio - 1.) - log_ratio).mean().item(), gradient.item()))
    return dict(zip(("policy_loss", "value_loss", "entropy", "approx_kl", "gradient_norm"),
                    map(float, np.mean(losses, axis=0))))


def parse_eval_seeds(value):
    try:
        result = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use comma-separated integer seeds") from error
    if not result or min(result) < 0 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Evaluation seeds must be unique nonnegative integers")
    return result


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _rng_states(rng, environment):
    return {"python": random.getstate(), "numpy_global": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "minibatch_numpy": copy.deepcopy(rng.bit_generator.state),
            "environment_numpy_generators": {
                key: copy.deepcopy(value.bit_generator.state)
                for key, value in vars(environment).items() if isinstance(value, np.random.Generator)}}


def _source_snapshot(run):
    paths = sorted((ROOT / "rl_bt").glob("*.py")) + [Path(__file__).resolve()]
    for relative in ("configs/reference_robot_envelope.json", "data/rmul_rule_facts.json",
                     "data/rmu_2027_protocol_facts.json", "assets/rmul_2027/derived/manifest.json"):
        if (ROOT / relative).is_file():
            paths.append(ROOT / relative)
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        data = path.read_bytes()
        hashes[relative.as_posix()] = hashlib.sha256(data).hexdigest()
        destination = run / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    return hashes


def _next_training_seed(rng, excluded):
    while True:
        seed = int(rng.integers(0, 2 ** 31 - 1))
        if seed not in excluded:
            return seed


def imitation_warmstart(model, environment_type, nav, *, decisions, seed, excluded, rng, run):
    from rl_bt.ul_task_env import ul_bt_baseline
    while seed in excluded:
        seed += 1
    environment = environment_type(seed=seed, randomization=True, nav=nav)
    observation = environment.reset(seed=seed)
    records = {key: [] for key in ("state", "candidates", "mask", "actions")}
    seeds = [seed]
    for _ in range(decisions):
        action = ul_bt_baseline(observation)
        if not observation["mask"][action]:
            raise RuntimeError("BT teacher chose an illegal candidate")
        for key in ("state", "candidates", "mask"):
            records[key].append(observation[key].copy())
        records["actions"].append(action)
        observation, _, done, _ = environment.step(action)
        if done:
            seed = _next_training_seed(rng, excluded)
            seeds.append(seed)
            observation = environment.reset(seed=seed)
    arrays = {key: np.stack(value) for key, value in records.items()}
    np.savez_compressed(run / "bt_warmstart.npz", **arrays)
    batch = {key: torch.as_tensor(value) for key, value in arrays.items()}
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
    for _ in range(20):
        order = rng.permutation(decisions)
        for begin in range(0, decisions, 64):
            index = order[begin:begin + 64]
            logits = model.actor(batch["state"][index], batch["candidates"][index], batch["mask"][index])
            loss = F.cross_entropy(logits, batch["actions"][index])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite UL imitation loss")
            optimizer.zero_grad()
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.)
            if not bool(torch.isfinite(gradient)):
                raise FloatingPointError("Non-finite UL imitation gradient")
            optimizer.step()
    with torch.no_grad():
        prediction = model.actor(batch["state"], batch["candidates"], batch["mask"]).argmax(-1)
        accuracy = float((prediction == batch["actions"]).float().mean())
    report = {"decisions": decisions, "source": "fresh_BT_interactions_in_training_courses",
              "teacher_episode_seeds": seeds, "evaluation_samples_used": False,
              "epochs": 20, "training_label_accuracy": accuracy,
              "interpretation": "Imitation accuracy on training labels, not held-out success."}
    _write_json(run / "bt_warmstart_report.json", report)
    return report


def evaluate_policy(policy, environment_type, nav, seeds, *, max_steps=4096):
    rows = []
    for seed in seeds:
        environment = environment_type(seed=seed, randomization=True, nav=nav)
        observation = environment.reset(seed=seed)
        total = 0.
        for step in range(1, max_steps + 1):
            action = int(policy(observation))
            if not 0 <= action < CANDIDATE_COUNT or not observation["mask"][action]:
                raise RuntimeError("Evaluation policy proposed an illegal candidate")
            observation, reward, done, info = environment.step(action)
            total += float(reward)
            if done:
                episode = info["episode"]
                rows.append({"seed": seed, "success": bool(episode["success"]),
                    "return": float(episode["return"]), "steps": int(episode["steps"]),
                    "objective": str(episode["objective"]), "side": int(episode["side"]),
                    "diagnostics": {key: episode[key] for key in ("time_s", "hp", "ammo",
                        "BT_veto_count", "recovery_count", "geometry_failure", "parameters") if key in episode}})
                if not np.isclose(total, episode["return"], rtol=1e-5, atol=1e-5):
                    raise ValueError("Episode reward accounting disagrees with step rewards")
                break
        else:
            raise RuntimeError(f"Evaluation seed {seed} exceeded {max_steps} steps without done")
    return rows


def summarize_evaluation(initial, final, baseline, seeds):
    groups = {"initialRL": initial, "finalRL": final, "BT": baseline}
    for rows in groups.values():
        if [row["seed"] for row in rows] != list(seeds):
            raise ValueError("Evaluation seed order mismatch")
    paired = []
    for before, after, bt in zip(initial, final, baseline):
        if len({(row["side"], row["objective"]) for row in (before, after, bt)}) != 1:
            raise ValueError("Paired seeds must generate the same side and objective")
        paired.append({"seed": before["seed"], "side": before["side"],
            "objective": before["objective"], "initialRL": before["success"],
            "finalRL": after["success"], "BT": bt["success"]})
    summary = {}
    for name, rows in groups.items():
        per_objective = {}
        for objective in sorted({row["objective"] for row in rows}):
            subset = [row for row in rows if row["objective"] == objective]
            per_objective[objective] = {"successes": sum(row["success"] for row in subset),
                                      "episodes": len(subset)}
        summary[name] = {"successes": sum(row["success"] for row in rows),
            "episodes": len(rows), "task_success_rate": float(np.mean([row["success"] for row in rows])),
            "mean_return": float(np.mean([row["return"] for row in rows])),
            "mean_steps": float(np.mean([row["steps"] for row in rows])),
            "by_objective": per_objective}
    coverage = sorted({(row["side"], row["objective"]) for row in final})
    return {"scenario_id": SCENARIO_ID, "seeds_predeclared_before_training": list(seeds),
        "metric": "ground_course_task_success_rate_not_match_win_rate",
        "initialRL_stage": "after_optional_BT_imitation_before_PPO",
        "same_seed_paired_tasks": True,
        "coverage": [{"side": side, "objective": objective} for side, objective in coverage],
        "covers_two_sides_and_three_objectives":
            set(coverage) == {(side, objective) for side in (0, 1)
                              for objective in ("capture", "resupply", "recover_then_capture")},
        "summary": summary, "paired_results": paired, "episodes": groups,
        "limits": ["These courses have no complete match opponent, team economy or match win label.",
                   "Task success does not establish Nav2, sensor or real-robot reliability.",
                   "This evaluation set becomes validation data if used to change training settings."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--eval-seeds", type=parse_eval_seeds, default=DEFAULT_EVAL_SEEDS)
    parser.add_argument("--bc-decisions", type=int, default=0,
                        help="Fresh BT teacher decisions before PPO; zero disables imitation")
    parser.add_argument("--rollout", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    if min(args.decisions, args.rollout, args.epochs) < 1 or args.seed < 0 or args.bc_decisions < 0:
        parser.error("Counts must be positive and seed must be nonnegative")
    if args.seed in args.eval_seeds:
        parser.error("Training seed must not be one of the predeclared evaluation seeds")
    # This import is deliberately independent of the old alphaRM environment.
    from rl_bt.ul_task_env import ULTaskEnv, ul_bt_baseline
    from rl_bt import ul_task_env
    if (ul_task_env.STATE_DIM, ul_task_env.CANDIDATE_DIM, ul_task_env.SCENARIO_ID) != (
            STATE_DIM, CANDIDATE_DIM, SCENARIO_ID):
        raise ValueError("UL environment and policy schema do not agree")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    run = args.run or ROOT / "runs" / f"ul_curriculum_{stamp}"
    run = run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    environment = ULTaskEnv(seed=args.seed, randomization=True)
    observation = environment.reset(seed=args.seed)
    tensor_observations([observation])
    model = ULCurriculumActorCritic()
    metadata = {"artifact_type": "UL_ground_task_curriculum_training", "scenario_id": SCENARIO_ID,
        "seed": args.seed, "ppo_decisions": args.decisions, "bt_imitation_decisions": args.bc_decisions,
        "evaluation_seeds": list(args.eval_seeds), "rollout": args.rollout, "epochs": args.epochs,
        "observation_schema": {"state": [STATE_DIM], "candidates": [CANDIDATE_COUNT, CANDIDATE_DIM],
                               "mask": [CANDIDATE_COUNT]},
        "state_fields": ul_task_env.STATE_FIELDS,
        "candidate_fields": ul_task_env.CANDIDATE_FIELDS,
        "navigation_source": environment.nav.summary(),
        "architecture": {"state_encoder": [24, 64, 64], "candidate_encoder": [12, 32],
                         "scorer": [96, 64, 1], "critic_training_only": [64, 64, 1], "activation": "Tanh"},
        "actor_parameters": sum(parameter.numel() for parameter in model.actor.parameters()),
        "critic_parameters": sum(parameter.numel() for parameter in model.critic.parameters()),
        "learning_rate": 3e-4, "gamma_per_decision": .99, "gae_lambda_per_decision": .95,
        "ppo_clip": .2, "entropy_coefficient": .01, "device": "cpu", "torch_version": torch.__version__,
        "randomization": True, "resume_supported": False, "old_checkpoint_import_supported": False,
        "implementation_sha256": _source_snapshot(run),
        "limits": ["Ground navigation/capture/resupply curriculum, not a complete official match.",
                   "Navigation is an offline path/execution proxy, not Point-LIO/Nav2/MPPI.",
                   "No match win-rate claim, no learned velocity/aim/fire/thermal outputs.",
                   "RNG states are saved, but a resume CLI and exact environment restoration are not implemented."]}
    _write_json(run / "config.json", metadata)
    _write_json(run / "status.json", {"status": "initializing", "decisions": 0})
    if args.bc_decisions:
        imitation_warmstart(model, ULTaskEnv, environment.nav, decisions=args.bc_decisions,
            seed=args.seed + 1_000_003, excluded=set(args.eval_seeds), rng=rng, run=run)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    initial = copy.deepcopy(model).eval()

    def save(path, decisions, updates):
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "metadata": metadata, "decisions": decisions, "updates": updates,
            "rng_states": _rng_states(rng, environment), "current_observation": observation,
            "resume_supported": False}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    save(run / "initial.pt", 0, 0)
    started = time.perf_counter()
    decisions, updates = 0, 0
    episodes = []
    episode_seed = args.seed
    with (run / "train.jsonl").open("w") as metrics_file, (run / "episodes.jsonl").open("w") as episode_file:
        while decisions < args.decisions:
            component_samples, veto_samples = [], []
            records = {key: [] for key in ("state", "candidates", "mask", "actions", "old_logp",
                       "values", "next_values", "rewards", "done")}
            for _ in range(min(args.rollout, args.decisions - decisions)):
                tensors = tensor_observations([observation])
                with torch.no_grad():
                    distribution, value = model.distribution_value(**tensors)
                    action = distribution.sample()
                    old_logp = distribution.log_prob(action)
                chosen = int(action.item())
                if not observation["mask"][chosen]:
                    raise RuntimeError("Policy sampled an illegal candidate")
                after, reward, done, info = environment.step(chosen)
                component_samples.append(info.get("reward_components", {}))
                veto_samples.append(bool(info.get("BT_veto", False)))
                with torch.no_grad():
                    _, next_value = model.distribution_value(**tensor_observations([after]))
                for key in ("state", "candidates", "mask"):
                    records[key].append(observation[key].copy())
                for key, item in (("actions", chosen), ("old_logp", float(old_logp.item())),
                    ("values", float(value.item())), ("next_values", float(next_value.item())),
                    ("rewards", float(reward)), ("done", bool(done))):
                    records[key].append(item)
                if done:
                    episode = {**info["episode"], "training_seed": episode_seed}
                    episodes.append(episode)
                    episode_file.write(json.dumps(episode, allow_nan=False) + "\n")
                    episode_file.flush()
                    episode_seed = _next_training_seed(rng, set(args.eval_seeds))
                    observation = environment.reset(seed=episode_seed)
                else:
                    observation = after
                decisions += 1
            arrays = {key: np.stack(value) for key, value in records.items()}
            advantages, returns = gae(arrays["rewards"], arrays["values"], arrays["next_values"], arrays["done"])
            batch = {key: torch.as_tensor(value) for key, value in arrays.items()}
            batch["old_logp"] = batch["old_logp"].float()
            batch["advantages"] = torch.as_tensor(advantages)
            batch["returns"] = torch.as_tensor(returns)
            row = ppo_update(model, optimizer, batch, rng, epochs=args.epochs)
            updates += 1
            recent = episodes[-20:]
            row.update({"update": updates, "decisions": decisions, "completed_episodes": len(episodes),
                "recent_task_success_rate": float(np.mean([episode["success"] for episode in recent])) if recent else None,
                "recent_return": float(np.mean([episode["return"] for episode in recent])) if recent else None,
                "mean_step_reward": float(arrays["rewards"].mean()),
                "mean_legal_candidates": float(arrays["mask"].sum(-1).mean()),
                "BT_veto_fraction": float(np.mean(veto_samples)),
                "wall_s": time.perf_counter() - started})
            for key in sorted({key for sample in component_samples for key in sample}):
                row["reward_" + key + "_mean"] = float(np.mean([
                    sample.get(key, 0.) for sample in component_samples]))
            metrics_file.write(json.dumps(row, allow_nan=False) + "\n")
            metrics_file.flush()
            _write_json(run / "status.json", {"status": "training", **row})
            print(json.dumps(row, allow_nan=False), flush=True)
    save(run / "final.pt", decisions, updates)
    _write_json(run / "status.json", {"status": "evaluating", "decisions": decisions})
    model.eval()
    before = evaluate_policy(lambda obs: choose(initial, obs), ULTaskEnv, environment.nav, args.eval_seeds)
    after = evaluate_policy(lambda obs: choose(model, obs), ULTaskEnv, environment.nav, args.eval_seeds)
    baseline = evaluate_policy(ul_bt_baseline, ULTaskEnv, environment.nav, args.eval_seeds)
    evaluation = summarize_evaluation(before, after, baseline, args.eval_seeds)
    _write_json(run / "evaluation.json", evaluation)
    summary = {"status": "completed", "run": str(run), "scenario_id": SCENARIO_ID,
        "ppo_decisions": decisions, "bt_imitation_decisions": args.bc_decisions,
        "completed_training_episodes": len(episodes), "updates": updates,
        "wall_s": time.perf_counter() - started, "evaluation": evaluation["summary"],
        "metric": evaluation["metric"], "resume_supported": False}
    _write_json(run / "summary.json", summary)
    _write_json(run / "status.json", summary)
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
