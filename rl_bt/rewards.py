"""Task-wrapper reward profiles with immutable alphaRM simulator dynamics."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from rl_bt.reward_components import (
    REWARD_PROFILES as REMOVAL_PROFILES,
    apply_reward_profile,
    reward_config,
    team_reward_components,
)

REWARD_PROFILES = REMOVAL_PROFILES + ("own_damage",)


def reward_tick(before, after, *, team, learner, config, profile="upstream"):
    """Return scalar tick diagnostics and the selected task-training reward.

    No privileged diagnostics are passed to the actor. ``config`` is the saved
    fixture's partial configuration; static reward constants come from the saved
    upstream configuration source. Only fixture keys in that config are replaced.
    ``own_damage`` adds two thirds of the learner's existing raw hit reward,
    increasing its coefficient from one-third to one. It is not HP-clipped
    effective damage, and death-frame hit reward remains zero.
    """
    cfg = reward_config()
    cfg = replace(cfg, **{key: value for key, value in config.items()
                          if key in cfg.__dataclass_fields__})
    if learner // cfg.n_per_team != team:
        raise ValueError("learner must belong to selected team")
    selected = slice(team * cfg.n_per_team, (team + 1) * cfg.n_per_team)
    dense = float(np.mean(np.asarray(after["reward_dense"])[selected]))
    terminal = float(np.mean(np.asarray(after["reward_terminal"])[selected]))
    # Preserve the old wrapper's float32 add followed by mean exactly.
    upstream = float(np.mean((np.asarray(after["reward_dense"])
                              + np.asarray(after["reward_terminal"]))[selected]))
    parts = team_reward_components(before, after, team, cfg)
    if profile not in REWARD_PROFILES:
        raise ValueError(f"Unknown reward profile: {profile}")
    own_hit_raw = (cfg.k_hit_reward * float(after["last_hit_damage"][learner]) / cfg.hp_max
                   * float(after["alive"][learner]))
    modified = upstream + (1. - 1. / cfg.n_per_team) * own_hit_raw if profile == "own_damage" else (
        apply_reward_profile(upstream, parts, profile))
    return {**parts, "upstream_dense": dense, "upstream_terminal": terminal,
            "upstream_total": upstream, "profile_adjustment": modified - upstream,
            "own_hit_raw": own_hit_raw,
            "selected_total": modified}
