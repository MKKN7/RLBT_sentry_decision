#!/usr/bin/env python3
"""Bounded task-selection PPO in alphaRM's simplified game, not official UL/UC."""
from __future__ import annotations

import argparse
import csv
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from rl_bt.alpharm_task_env import AlphaRuntime, AlphaTaskEnv, SCENARIO, TRAIN_STYLES, bt_baseline
from rl_bt.policy import TaskActorCritic, tensor_observations
from rl_bt.rewards import REWARD_PROFILES


def snapshot_environments(envs):
    """Runtime sessions/navigation are immutable shared services, not state."""
    return [{k: copy.deepcopy(v) for k, v in env.__dict__.items()
             if k not in ("runtime", "nav")} for env in envs]


def restore_environments(envs, snapshots):
    if len(envs) != len(snapshots):
        raise ValueError("Exact resume requires the saved environment count")
    for env, snapshot in zip(envs, snapshots):
        env.__dict__.update(copy.deepcopy(snapshot))


def step_environment(item):
    env, action = item
    return env.step(action)


def gae(rewards, values, next_values, done, duration, gamma_per_second=.99, lambda_per_second=.90):
    """True episode terminals remove bootstrap; a rollout cut preserves it."""
    gamma = gamma_per_second ** duration
    lam = lambda_per_second ** duration
    result = np.zeros_like(rewards)
    carry = np.zeros(rewards.shape[1], np.float32)
    for t in reversed(range(len(rewards))):
        continuing = 1. - done[t].astype(np.float32)
        delta = rewards[t] + gamma[t] * next_values[t] * continuing - values[t]
        carry = delta + gamma[t] * lam[t] * continuing * carry
        result[t] = carry
    return result, result + values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--bc-decisions", type=int, default=0)
    parser.add_argument("--no-randomization", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", type=Path, help="Additional --steps from an existing local checkpoint")
    parser.add_argument("--init-from", type=Path,
                        help="New experiment: reuse model weights, reset optimizer, episodes and RNG")
    parser.add_argument("--reward-profile", choices=REWARD_PROFILES, default="upstream")
    parser.add_argument("--workers", type=int, default=1, help="Independent environment sampling threads")
    parser.add_argument("--checkpoint-every", type=int, default=8, help="Keep immutable checkpoint every N updates")
    parser.add_argument("--rollout-save-every", type=int, default=1)
    args = parser.parse_args()
    if min(args.steps, args.num_envs, args.rollout, args.epochs) < 1:
        parser.error("Training counts must be positive")
    if args.bc_decisions < 0:
        parser.error("--bc-decisions cannot be negative")
    if min(args.workers, args.checkpoint_every, args.rollout_save_every) < 1:
        parser.error("Workers and save intervals must be positive")
    if args.resume and args.bc_decisions:
        parser.error("Resume continues the existing policy; do not repeat BC warmstart")
    if args.init_from and (args.resume or args.bc_decisions):
        parser.error("--init-from is a new weight-initialized experiment, separate from resume/BC")
    if args.steps % (args.num_envs * args.rollout):
        parser.error("--steps must be a multiple of num-envs * rollout")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    run = args.out or ROOT / "runs" / f"phase1_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "rollouts").mkdir()
    (run / "checkpoints").mkdir()
    runtime = AlphaRuntime()
    envs = [AlphaTaskEnv(runtime, args.seed + 100 * i,
                         randomization=not args.no_randomization,
                         reward_profile=args.reward_profile) for i in range(args.num_envs)]
    obs = [{k: v.copy() for k, v in e.cached.items()} for e in envs]
    model = TaskActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    initial_steps, initial_updates = 0, 0
    restored_exactly = False
    init_metadata = None
    if args.init_from:
        checkpoint = torch.load(args.init_from, weights_only=False, map_location="cpu")
        init_metadata = checkpoint["metadata"]
        if init_metadata.get("scenario_id") != SCENARIO:
            raise ValueError("Initialization checkpoint belongs to another scenario")
        expected = init_metadata.get("implementation_sha256", {}).get("rl_bt/policy.py")
        if expected and hashlib.sha256((ROOT / "rl_bt/policy.py").read_bytes()).hexdigest() != expected:
            raise ValueError("Initialization policy architecture/source changed")
        model.load_state_dict(checkpoint["model"])
    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False, map_location="cpu")
        old_metadata = checkpoint["metadata"]
        if old_metadata.get("scenario_id") != SCENARIO:
            raise ValueError("Resume checkpoint belongs to another scenario")
        for rel in ("rl_bt/alpharm_task_env.py", "rl_bt/geometry.py", "rl_bt/policy.py",
                    "rl_bt/rewards.py", "rl_bt/reward_components.py"):
            expected = old_metadata.get("implementation_sha256", {}).get(rel)
            if expected and hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Resume implementation changed: {rel}; create a documented new experiment")
        if old_metadata.get("randomization") != (not args.no_randomization):
            raise ValueError("Resume must preserve randomization setting")
        if old_metadata.get("reward_profile", "upstream") != args.reward_profile:
            raise ValueError("Resume must preserve reward profile; use --init-from for a new experiment")
        model.load_state_dict(checkpoint["model"])
        if "optimizer" not in checkpoint:
            raise ValueError("Resume requires an optimizer checkpoint, not actor-only export")
        optimizer.load_state_dict(checkpoint["optimizer"])
        initial_steps = int(checkpoint.get("decisions", 0))
        historical_batch = old_metadata.get("num_envs", 1) * old_metadata.get("rollout", 1)
        initial_updates = int(checkpoint.get("total_updates", initial_steps // historical_batch))
        if "environment_snapshots" in checkpoint:
            restore_environments(envs, checkpoint["environment_snapshots"])
            obs = [{k: v.copy() for k, v in e.cached.items()} for e in envs]
            rng.bit_generator.state = checkpoint["numpy_rng_state"]
            torch.set_rng_state(checkpoint["torch_rng_state"])
            restored_exactly = True
    metadata = {"artifact_type": "phase1_simplified_task_selection_training",
        "scenario_id": SCENARIO, "official_competition_profile": None,
        "upstream_commit": runtime.source_commit, "seed": args.seed,
        "steps": args.steps, "num_envs": args.num_envs, "rollout": args.rollout,
        "steps_are_additional": True, "initial_decisions": initial_steps,
        "target_cumulative_decisions": initial_steps + args.steps,
        "resume_from": str(args.resume.resolve()) if args.resume else None,
        "resume_checkpoint_sha256": hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None,
        "resume_mode": "exact_runtime_and_rng" if restored_exactly else (
            "policy_and_optimizer_with_new_episode_stream" if args.resume else (
                "weights_only_new_experiment" if args.init_from else "new_run")),
        "initialization_checkpoint": str(args.init_from.resolve()) if args.init_from else None,
        "initialization_checkpoint_sha256": hashlib.sha256(args.init_from.read_bytes()).hexdigest()
            if args.init_from else None,
        "initialization_parent_decisions": checkpoint.get("decisions") if args.init_from else None,
        "optimizer_reset": bool(args.init_from),
        "reward_profile": args.reward_profile,
        "workers": min(args.workers, args.num_envs), "checkpoint_every_updates": args.checkpoint_every,
        "rollout_save_every_updates": args.rollout_save_every,
        "epochs": args.epochs, "learning_rate": 3e-4, "gamma_per_second": .99,
        "lambda_per_second": .90, "ppo_clip": .2, "entropy_coefficient": .01,
        "train_styles": TRAIN_STYLES, "randomization": not args.no_randomization,
        "device": "cpu", "torch_version": torch.__version__,
        "actor_parameters": sum(p.numel() for p in model.actor.parameters()),
        "critic_parameters": sum(p.numel() for p in model.critic.parameters()),
        "trained_outputs": ["candidate_selection"],
        "stance_head": "architecture present; inactive/untrained in simplified game",
        "bt_warmstart_decisions": args.bc_decisions,
        "excluded_outputs": ["velocity", "aim", "fire", "thermal_control"],
        "reward": "reward_profile tick contributions summed minus 0.002 per changed goal; see rl_bt/rewards.py",
        "observation_contract": "state[148], candidates[K,16], mask[K]",
        "limits": ["Simplified flat map and homogeneous units, no official sentry attributes.",
                   "No UC rules, tunnels, economy or Nav2 execution in this training env.",
                   "Upstream observation includes simplified HUD data; official observation adapter still required.",
                   "Web export collision/push terminal thresholds are disabled; penalties/contacts remain.",
                   "Noise/latency ranges are explicit initial assumptions, not real-log calibration.",
                   "Background teammates and pretrained opponents use the upstream recurrent ONNX actor."]}
    metadata["implementation_sha256"] = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for directory in (ROOT / "rl_bt", ROOT / "scripts")
        for p in directory.glob("*.py")
    }
    for rel in metadata["implementation_sha256"]:
        target = run / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / rel).read_bytes())
    (run / "config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "metadata": metadata, "decisions": initial_steps}, run / "initial.pt")
    if args.bc_decisions:
        teacher = AlphaTaskEnv(runtime, args.seed + 7001,
                              randomization=not args.no_randomization, reward_profile=args.reward_profile)
        teacher_obs = teacher.cached
        examples = {k: [] for k in ("state", "candidates", "mask", "actions", "next_state",
                                     "rewards", "done", "executed_candidate")}
        for _ in range(args.bc_decisions):
            action = bt_baseline(teacher_obs)
            for key in ("state", "candidates", "mask"):
                examples[key].append(teacher_obs[key].copy())
            after, reward, done, info = teacher.step(action)
            for key, item in (("actions", action), ("next_state", after["state"]),
                ("rewards", reward), ("done", done), ("executed_candidate", info["executed_candidate"])):
                examples[key].append(item)
            teacher_obs = teacher.reset() if done else after
        examples = {k: np.stack(v) for k, v in examples.items()}
        np.savez_compressed(run / "bt_warmstart.npz", **examples)
        tensors = {k: torch.as_tensor(v) for k, v in examples.items()}
        bc_optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
        for _ in range(30):
            for idx in np.array_split(rng.permutation(args.bc_decisions), max(1, args.bc_decisions // 64)):
                logits, _, _ = model.actor.encode(tensors["state"][idx], tensors["candidates"][idx],
                                                  tensors["mask"][idx])
                loss = F.cross_entropy(logits, tensors["actions"][idx])
                bc_optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.)
                bc_optimizer.step()
        with torch.no_grad():
            logits, _, _ = model.actor.encode(tensors["state"], tensors["candidates"], tensors["mask"])
            accuracy = float((logits.argmax(-1) == tensors["actions"]).float().mean())
        report = {"samples": args.bc_decisions, "source": "new_scripted_BT_interactions_training_styles_only",
                  "teacher_seed": args.seed + 7001, "teacher_optimal": False,
                  "evaluation_episodes_used": False, "epochs": 30,
                  "training_label_accuracy": accuracy,
                  "note": "Training-set imitation accuracy, not held-out tactical performance."}
        (run / "bt_warmstart_report.json").write_text(json.dumps(report, indent=2) + "\n")
        torch.save({"model": model.state_dict(), "metadata": metadata, "decisions": 0}, run / "warmstart.pt")
        print(json.dumps({"warmstart": report}), flush=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(run / "tensorboard"))
    except ImportError:
        writer = None
    started = time.perf_counter()
    episodes = []
    metric_rows = []
    pool = ThreadPoolExecutor(max_workers=min(args.workers, args.num_envs)) if args.workers > 1 else None
    def save_checkpoint(path, decisions, updates):
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "metadata": metadata, "decisions": decisions, "total_updates": updates,
            "environment_snapshots": snapshot_environments(envs),
            "numpy_rng_state": copy.deepcopy(rng.bit_generator.state),
            "torch_rng_state": torch.get_rng_state(),
            "run_training_episode_count": len(episodes)}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
    save_checkpoint(run / "checkpoints" / f"step_{initial_steps:09d}.pt", initial_steps, initial_updates)
    with (run / "episodes.jsonl").open("w") as episode_file, (run / "status.json").open("w") as status_file:
        json.dump({"status": "running", "target_cumulative_decisions": initial_steps + args.steps}, status_file)
    with (run / "episodes.jsonl").open("a") as episode_file:
        for update in range(args.steps // (args.num_envs * args.rollout)):
            reward_components = []
            records = {k: [] for k in ("state", "candidates", "mask", "actions", "old_logp",
                "next_state", "next_candidates", "next_mask", "values", "next_values",
                "rewards", "done", "duration", "executed_candidate", "override")}
            for _ in range(args.rollout):
                tensors = tensor_observations(obs)
                with torch.no_grad():
                    distribution, value = model.distribution_value(**tensors)
                    actions = distribution.sample()
                    old_logp = distribution.log_prob(actions)
                next_obs, rewards, dones, durations, executed, overridden = [], [], [], [], [], []
                chosen_actions = [int(a) for a in actions]
                for i, action in enumerate(chosen_actions):
                    if not obs[i]["mask"][action]:
                        raise RuntimeError("Policy sampled an illegal candidate")
                items = list(zip(envs, chosen_actions))
                results = list(pool.map(step_environment, items)) if pool else [step_environment(x) for x in items]
                for after, reward, done, info in results:
                    reward_components.append(info["reward_components"])
                    next_obs.append(after); rewards.append(reward); dones.append(done)
                    durations.append(info["duration_s"]); executed.append(info["executed_candidate"])
                    overridden.append(info["override_reason"] is not None)
                    if done:
                        episodes.append(info["episode"])
                        episode_file.write(json.dumps(info["episode"]) + "\n")
                        episode_file.flush()
                with torch.no_grad():
                    _, bootstrap = model.distribution_value(**tensor_observations(next_obs))
                for key in ("state", "candidates", "mask"):
                    records[key].append(tensors[key].numpy())
                    records["next_" + key].append(np.stack([o[key] for o in next_obs]))
                for key, value_np in (("actions", actions.numpy()), ("old_logp", old_logp.numpy()),
                    ("values", value.numpy()), ("next_values", bootstrap.numpy()),
                    ("rewards", rewards), ("done", dones), ("duration", durations),
                    ("executed_candidate", executed), ("override", overridden)):
                    records[key].append(np.asarray(value_np))
                obs = [env.reset() if done else after
                       for env, done, after in zip(envs, dones, next_obs)]
            data = {k: np.stack(v) for k, v in records.items()}
            advantages, returns = gae(data["rewards"], data["values"], data["next_values"],
                                       data["done"], data["duration"])
            if (update + 1) % args.rollout_save_every == 0 or update == 0:
                np.savez_compressed(run / "rollouts" / f"update_{update:05d}.npz", **data,
                                    advantages=advantages, returns=returns)
            batch = {k: torch.as_tensor(v.reshape((-1,) + v.shape[2:])) for k, v in data.items()}
            adv = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32)
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            targets = torch.as_tensor(returns.reshape(-1), dtype=torch.float32)
            losses = []
            count = len(adv)
            for _ in range(args.epochs):
                order = rng.permutation(count)
                for begin in range(0, count, 128):
                    idx = order[begin:begin + 128]
                    distribution, predicted = model.distribution_value(
                        batch["state"][idx], batch["candidates"][idx], batch["mask"][idx])
                    logp = distribution.log_prob(batch["actions"][idx])
                    log_ratio = logp - batch["old_logp"][idx]
                    ratio = log_ratio.exp()
                    policy_loss = torch.maximum(-adv[idx] * ratio,
                        -adv[idx] * ratio.clamp(.8, 1.2)).mean()
                    value_loss = F.mse_loss(predicted, targets[idx])
                    entropy = distribution.entropy().mean()
                    loss = policy_loss + .5 * value_loss - .01 * entropy
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite PPO loss")
                    optimizer.zero_grad(); loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                    if not torch.isfinite(grad):
                        raise FloatingPointError("Non-finite gradient")
                    optimizer.step()
                    losses.append((policy_loss.item(), value_loss.item(), entropy.item(),
                                   ((ratio - 1) - log_ratio).mean().item()))
            metrics = np.mean(losses, axis=0)
            total_steps = initial_steps + (update + 1) * count
            recent = episodes[-20:]
            choices = np.bincount(data["actions"].reshape(-1), minlength=16)
            alive_samples = data["state"][..., 11] > .5
            legal_counts = data["mask"].sum(axis=-1)
            target_variance = float(np.var(returns))
            explained_variance = 1. - float(np.var(returns - data["values"])) / target_variance if target_variance > 1e-10 else 0.
            row = {"update": update + 1, "decisions": total_steps,
                "policy_loss": float(metrics[0]), "value_loss": float(metrics[1]),
                "entropy": float(metrics[2]), "approx_kl": float(metrics[3]),
                "mean_step_reward": float(data["rewards"].mean()),
                "episode_count": len(episodes),
                "recent_return": float(np.mean([e["return"] for e in recent])) if recent else None,
                "recent_win_rate": float(np.mean([e["win"] for e in recent])) if recent else None,
                "override_fraction": float(data["override"].mean()),
                "alive_sample_fraction": float(alive_samples.mean()),
                "mean_legal_candidates": float(legal_counts.mean()),
                "explained_variance": explained_variance,
                "choice_continue_fraction": float(choices[0] / count),
                "choice_home_fraction": float(choices[2] / count),
                "choice_center_fraction": float(choices[[1, 3, 4]].sum() / count),
                "distinct_candidates_sampled": int((choices > 0).sum()),
                "decisions_per_s": (update + 1) * count / (time.perf_counter() - started),
                "wall_s": time.perf_counter() - started}
            for name in reward_components[0]:
                row["reward_" + name + "_mean"] = float(np.mean([c[name] for c in reward_components]))
            metric_rows.append(row)
            if writer:
                for key, number in row.items():
                    if number is not None:
                        writer.add_scalar(key, number, total_steps)
                writer.flush()
            with (run / "metrics.csv").open("w", newline="") as f:
                csv_writer = csv.DictWriter(f, fieldnames=row.keys())
                csv_writer.writeheader(); csv_writer.writerows(metric_rows)
            save_checkpoint(run / "latest.pt", total_steps, initial_updates + update + 1)
            if (update + 1) % args.checkpoint_every == 0 or total_steps == initial_steps + args.steps:
                frozen = run / "checkpoints" / f"step_{total_steps:09d}.pt"
                if frozen.exists():
                    raise FileExistsError(frozen)
                save_checkpoint(frozen, total_steps, initial_updates + update + 1)
            (run / "status.json").write_text(json.dumps({"status": "running", **row,
                "target_cumulative_decisions": initial_steps + args.steps}, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    if writer:
        writer.close()
    if pool:
        pool.shutdown()
    summary = {"status": "completed", "run": str(run), "decisions": initial_steps + args.steps,
        "additional_decisions": args.steps, "initial_decisions": initial_steps,
        "completed_training_episodes": len(episodes), "wall_s": time.perf_counter() - started,
        "evaluation_status": "separate paired full-episode evaluation required",
        "official_UL_UC_training": False}
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run / "status.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
