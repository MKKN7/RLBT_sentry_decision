#!/usr/bin/env python3
"""Export a bounded synthetic alphaRM demo episode; no JAX or training needed.

The deterministic decoder follows upstream demo/web/engine.js. This is an
interface probe of the simplified upstream game, not an RMUL/RMUC dataset.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(value: dict) -> np.ndarray:
    return np.frombuffer(
        base64.b64decode(value["data"]), dtype=np.dtype(value["dtype"])
    ).copy().reshape(value["shape"])


def action_masks(obs: np.ndarray, state: dict, cfg: dict) -> tuple:
    """Use observed visibility/relative range, plus the demo's fire gates."""
    enemies = np.asarray([[3, 4, 5]] * 3 + [[0, 1, 2]] * 3)
    enemy_obs = obs[:, 68:119].reshape(6, 3, 17)
    distance = np.linalg.norm(enemy_obs[:, :, 1:3], axis=-1) * cfg["lock_range"]
    target_mask = np.ones((6, 4), dtype=bool)
    target_mask[:, 1:] = (
        (enemy_obs[:, :, 0] > 0.5)
        & (enemy_obs[:, :, 14] > 0.5)
        & (distance < cfg["lock_range"])
        & state["alive"][:, None]
    )
    # Match env.get_fire_mask: this gate refers to the previous lock.
    target = state["target"]
    target_agent = enemies[np.arange(6), np.maximum(target - 1, 0)]
    fire_mask = np.ones((6, 2), dtype=bool)
    fire_mask[:, 1] = (
        (target > 0)
        & state["alive"]
        & (state["invuln"] <= 0)
        & ~state["overheated"]
        & (state["heat"] < cfg["heat_max"])
        & (state["invuln"][target_agent] <= 0)
    )
    return target_mask, fire_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=1800)
    parser.add_argument("--out", type=Path, default=ROOT / "data/demos/alpharm_probe")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 1800:
        parser.error("--max-steps must be within 1..1800 for this fixed fixture")
    manifest = json.loads(
        (ROOT / "data/references/alpharm_jax_review_manifest.json").read_text()
    )
    # Avoid silently pairing a changed fixture/model with this decoder.
    web_dir = ROOT / "data/references/alpharm_jax_review/upstream/demo/web"
    for entry in manifest["files"]:
        path = ROOT / entry["local_path"]
        if path.parent == web_dir and path.name in (
            "actor.onnx", "env_step.onnx", "fixture.json"
        ):
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Snapshot hash mismatch: {path}")
    fixture = json.loads((web_dir / "fixture.json").read_text())
    state = [decode(x) for x in fixture["state"]]
    state_index = {x["name"]: x["index"] for x in fixture["state"]}
    obs, hidden = decode(fixture["obs"]), decode(fixture["hidden"])
    initial_hidden = hidden.copy()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    actor = ort.InferenceSession(
        str(web_dir / "actor.onnx"), sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    env = ort.InferenceSession(
        str(web_dir / "env_step.onnx"), sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    if len(env.get_inputs()) != len(state) + 1 or len(env.get_outputs()) != len(state) + 2:
        raise ValueError("Environment signature disagrees with fixed fixture")
    records = {"obs": [obs.copy()], "actions": [], "target_mask": [],
               "fire_mask": [], "reward": [], "environment_done": []}
    for name, i in state_index.items():
        records[f"privileged_state/{name}"] = [state[i].copy()]
    started = time.perf_counter()
    for step in range(args.max_steps):
        current = {name: state[i] for name, i in state_index.items()}
        tm, fm = action_masks(obs, current, fixture["config"])
        raw, hidden = actor.run(None, {"obs": obs, "hidden": hidden})
        continuous = np.tanh(3 * np.tanh(raw[:, :4] / 3))
        target = np.argmax(np.where(tm, raw[:, 8:12], -np.inf), axis=-1)
        fire = np.argmax(np.where(fm, raw[:, 12:14], -np.inf), axis=-1)
        actions = np.concatenate(
            (continuous, target[:, None], fire[:, None]), axis=-1
        ).astype(np.float32)
        assert tm[np.arange(6), target].all() and fm[np.arange(6), fire].all()
        feeds = {v.name: x for v, x in zip(env.get_inputs()[:-1], state)}
        feeds[env.get_inputs()[-1].name] = actions
        outputs = env.run(None, feeds)
        state, obs, done = outputs[:-2], outputs[-2], bool(outputs[-1])
        reward = state[state_index["reward_dense"]] + state[state_index["reward_terminal"]]
        for name, value in (
            ("obs", obs), ("actions", actions), ("target_mask", tm),
            ("fire_mask", fm), ("reward", reward), ("environment_done", done)
        ):
            records[name].append(np.asarray(value).copy())
        for name, i in state_index.items():
            records[f"privileged_state/{name}"].append(state[i].copy())
        if (step + 1) % 300 == 0 or done:
            print(f"steps={step + 1} done={done}", flush=True)
        if done:
            break
    arrays = {k: np.stack(v) for k, v in records.items()}
    arrays["initial_hidden"] = initial_hidden
    count = len(arrays["actions"])
    arrays["collector_truncated"] = np.zeros(count, dtype=bool)
    arrays["collector_truncated"][-1] = not done
    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"Non-finite values in {name}")
    assert arrays["obs"].shape == (count + 1, 6, 140)
    assert arrays["actions"].shape == (count, 6, 6)
    assert not arrays["environment_done"][:-1].any()
    args.out.mkdir(parents=True, exist_ok=True)
    sample_path = args.out / "episode_000.npz"
    np.savez_compressed(sample_path, **arrays)
    report = {
        "artifact_type": "synthetic_upstream_demo_interface_probe_not_official_match_data",
        "source_commit": manifest["commit"], "scenario_id": "alpharm_simplified_3v3_web",
        "official_competition_profile": None, "seed": fixture["seed"],
        "episodes": 1, "environment_steps": count, "agent_transitions": count * 6,
        "environment_done": done, "collector_truncated": not done,
        "final_score": state[state_index["score"]].tolist(),
        "final_winner_code": int(state[state_index["winner"]]),
        "final_outcome_code": int(state[state_index["outcome"]]),
        "final_time_remaining_s": float(state[state_index["time"]]),
        "rollout_wall_s": time.perf_counter() - started,
        "provider": "CPUExecutionProvider", "onnxruntime_version": ort.__version__,
        "policy_mode": "deterministic_pretrained_actor_6_agents_shared_weights",
        "noise": "disabled; follows deterministic exported environment, no JS noise injection",
        "rules": "upstream web export; collision/push terminal thresholds set to 1e9",
        "sample_path": str(sample_path.relative_to(ROOT)) if sample_path.is_relative_to(ROOT) else str(sample_path),
        "sample_sha256": sha256(sample_path), "sample_bytes": sample_path.stat().st_size,
        "array_shapes": {k: list(v.shape) for k, v in arrays.items()},
        "alignment": "obs[t], actions[t], reward[t], obs[t+1]; privileged_state has T+1 entries",
        "checks": ["fixed source hashes", "legal target/fire actions", "finite arrays",
                   "T+1 observation alignment", "stop at first done"],
        "limits": [
            "One fixed deterministic episode; repeated runs are not independent matches.",
            "No JAX/ONNX numerical parity comparison performed on this host.",
            "No BT tasks/candidate labels, behavior log-probabilities or value targets.",
            "environment_done is not a per-agent PPO terminated/truncated label.",
            "Useful as interface/replay probe; not direct PPO training or real opponent calibration.",
            "No official RMUL/RMUC rules, 3D tunnel, Nav2 or MPPI validation."
        ]
    }
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k not in ("array_shapes", "limits")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
