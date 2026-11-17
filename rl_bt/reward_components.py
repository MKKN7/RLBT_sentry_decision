"""Reward diagnostics and narrowly scoped ablations for the alphaRM prototype.

The ONNX model remains the source of truth for physical state and total reward.
Non-contact shaping is reconstructed from its exported before/after state; the
remaining dense reward is explicitly a residual, not a claimed contact model.
Nothing here describes official RMUL/RMUC scoring or shared team economy.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from typing import Mapping

import numpy as np

from rl_bt.geometry import ROOT, reference_config


REWARD_PROFILES = ("upstream", "no_home", "no_fire", "no_home_no_fire")
REMOVED_COMPONENTS = {
    "upstream": (),
    "no_home": ("low_hp_home",),
    "no_fire": ("fire_heat",),
    "no_home_no_fire": ("low_hp_home", "fire_heat"),
}


@lru_cache(maxsize=1)
def reward_config():
    """Read coefficients from the saved upstream source without importing JAX."""
    rel = "data/references/alpharm_jax_review/upstream/jaxmarl/environments/rm/config.py"
    manifest = json.loads((ROOT / "data/references/alpharm_jax_review_manifest.json").read_text())
    expected = next(item["sha256"] for item in manifest["files"] if item["local_path"] == rel)
    if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != expected:
        raise ValueError("Upstream reward coefficients changed; revalidate against ONNX export")
    return reference_config()


def _inside_polygon(points, polygon):
    """Match upstream point_in_rect, including its edge convention."""
    points, polygon = np.asarray(points), np.asarray(polygon)
    x, y = points[..., 0], points[..., 1]
    inside = np.zeros_like(x, dtype=bool)
    for (x1, y1), (x2, y2) in zip(polygon, np.roll(polygon, -1, axis=0)):
        crosses = (y1 > y) != (y2 > y)
        denominator = 1e-9 if abs(y2 - y1) < 1e-9 else y2 - y1
        crossing_x = (x2 - x1) * (y - y1) / denominator + x1
        inside ^= crosses & (x < crossing_x)
    return inside


def per_agent_reward_components(before: Mapping, after: Mapping, config=None):
    """Return signed per-agent contributions whose sum equals ONNX reward.

    ``contact_and_precision_residual`` retains contact effects and NumPy/ONNX
    rounding differences. Terminal reward is read directly rather than inferred
    from a winner, preserving stagnation and other special outcomes.
    """
    cfg = reward_config() if config is None else config
    team = np.repeat(np.arange(2), cfg.n_per_team)
    alive = np.asarray(after["alive"], dtype=bool)
    alive_f = alive.astype(float)
    died = np.asarray(after["just_died"], dtype=bool)
    pos, vel = np.asarray(after["pos"]), np.asarray(after["vel"])
    hp = np.asarray(after["hp"])
    center = np.linalg.norm(pos, axis=-1)
    in_center = center <= cfg.center_radius
    low = hp < cfg.low_hp_threshold * cfg.hp_max
    own_spawns = np.asarray([cfg.red_spawn, cfg.blue_spawn])
    own_centers = own_spawns.mean(axis=1)[team]
    in_spawn = np.asarray([_inside_polygon(pos[i], own_spawns[team[i]])
                           for i in range(cfg.n_agents)])

    center_near = (cfg.k_center_reward
                   * np.exp(-center ** 2 / (2 * cfg.center_sigma ** 2))
                   * (center > cfg.center_radius) * alive_f)
    center_progress = (cfg.k_center_progress
                       * (np.asarray(before["center_dist_prev"]) - center) * alive_f)
    occupancy = cfg.k_occupy_reward * in_center * alive_f * (np.asarray(after["invuln"]) <= 0)

    def zero_sum_living(raw):
        difference = np.sum(raw[team == 0]) - np.sum(raw[team == 1])
        counts = np.asarray([max(np.sum(alive[team == t]), 1) for t in range(2)])
        return np.where(team == 0, difference, -difference) / counts[team] * alive_f

    score_delta = (np.asarray(after["score"]) - np.asarray(before["score"])
                   - np.asarray(after["last_kill_score"]))
    score_diff = score_delta[0] - score_delta[1]
    elapsed = float(after["episode_elapsed"])
    speed = np.linalg.norm(vel, axis=-1)
    camp = ((in_spawn | (center > cfg.center_radius)) & ~low
            & (elapsed > cfg.camp_grace_time) & (speed < cfg.camp_speed)
            & (np.asarray(after["target"]) == 0) & alive)
    ramp = np.clip((elapsed - cfg.camp_grace_time)
                   / max(cfg.episode_time - cfg.camp_grace_time, 1e-6), 0, 1)
    outward_speed = np.where(team == 0, np.maximum(vel[:, 0], -vel[:, 1]),
                             np.maximum(-vel[:, 0], vel[:, 1])) / cfg.v_max
    # Kill credit is assigned before this substep's respawns. A newly revived
    # agent must not receive a kill bonus from the same substep.
    kill_eligible = np.asarray(before["alive"], dtype=bool) & ~died
    kill_count = np.asarray(after["last_kill_score"]) / max(cfg.kill_score, 1e-9)
    parts = {
        "hit": cfg.k_hit_reward * np.asarray(after["last_hit_damage"]) / cfg.hp_max * alive_f,
        "center_near": zero_sum_living(center_near),
        "center_progress": zero_sum_living(center_progress),
        "occupancy": zero_sum_living(occupancy),
        "score": cfg.k_score_reward * np.where(team == 0, score_diff, -score_diff) * alive_f,
        "kill": cfg.k_kill_team_reward * kill_count[team] * kill_eligible * alive_f,
        "survival": cfg.k_survival_reward * hp / cfg.hp_max * alive_f,
        "low_hp_home": (cfg.k_low_hp_home_reward
                         * np.exp(-np.linalg.norm(pos - own_centers, axis=-1) ** 2
                                  / (2 * cfg.home_sigma ** 2)) * low * alive_f),
        "outward": (cfg.k_outward_speed_reward * np.clip(outward_speed, -1, 1)
                    * (hp >= cfg.hp_max - 1e-6) * in_spawn * alive_f),
        "death": -cfg.k_death_penalty * died.astype(float),
        "camp": -(cfg.k_camp_penalty + cfg.k_camp_ramp * ramp) * camp,
        "fire_heat": -cfg.k_fire_heat_penalty * np.asarray(after["last_fired"]) * alive_f,
    }
    reconstructed = sum(parts.values())
    parts["contact_and_precision_residual"] = np.asarray(after["reward_dense"]) - reconstructed
    parts["terminal"] = np.asarray(after["reward_terminal"], dtype=float).copy()
    if not all(np.isfinite(values).all() for values in parts.values()):
        raise FloatingPointError("Non-finite reward diagnostic component")
    return {name: np.asarray(values, dtype=float) for name, values in parts.items()}


def team_reward_components(before: Mapping, after: Mapping, team: int, config=None):
    """Apply the task wrapper's arithmetic team mean to each contribution."""
    cfg = reward_config() if config is None else config
    if team not in (0, 1):
        raise ValueError("team must be 0 or 1")
    selected = slice(team * cfg.n_per_team, (team + 1) * cfg.n_per_team)
    return {name: float(np.mean(values[selected]))
            for name, values in per_agent_reward_components(before, after, cfg).items()}


def apply_reward_profile(upstream_reward: float, components: Mapping[str, float],
                         profile: str = "upstream"):
    """Remove selected shaping without changing combat execution or dynamics.

    The baseline uses the original scalar directly, preserving its exact
    floating-point value. Other profiles subtract only explicitly named terms;
    independent comparisons should continue reporting official simulator outcomes
    and the unchanged upstream return as well as the training return.
    """
    if profile not in REMOVED_COMPONENTS:
        raise ValueError(f"Unknown reward profile: {profile}")
    correction = -sum(float(components[name]) for name in REMOVED_COMPONENTS[profile])
    return float(upstream_reward) + correction
