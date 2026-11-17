"""Isolated role-balanced scene adapter for the frozen alphaRM task prototype.

Physical dynamics, rule constants, rewards, observations and the existing
navigation/combat skills remain those of AlphaTaskEnv. Only the two supporting
allies' controller assignment changes. This is not official UL/UC or Nav2.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from rl_bt.alpharm_task_env import AlphaTaskEnv, ROOT
from scripts.probe_alpharm_onnx import action_masks

SCENARIO_ID = "alpharm_role_balanced_v2"
DEFAULT_CONFIG = ROOT / "configs/strategy_scenarios_v2.json"
SUPPORT_MODES = ("anchors", "balanced")
RAW_CONTINUOUS_BOUND = float(np.tanh(3.))


def _inverse_continuous(command):
    """Invert frozen tanh(3*tanh(raw/3)) on its finite representable range.

    Exact commands +/-1 cannot be represented by that decoder. Saturate at
    tanh(3)-1e-7 and report the explicit adapter error; do not hide a physical
    or navigation parameter change. Other command components round-trip.
    """
    command = np.asarray(command, dtype=np.float64)
    if command.shape != (4,) or not np.isfinite(command).all():
        raise ValueError("Fixed skill must supply four finite continuous commands")
    if np.max(np.abs(command)) > 1. + 1e-6:
        raise ValueError("Normalized fixed skill commands must lie in [-1,1]")
    bound = RAW_CONTINUOUS_BOUND - 1e-7
    represented = np.clip(command, -bound, bound)
    inner = np.arctanh(represented) / 3.
    raw = (3. * np.arctanh(inner)).astype(np.float32)
    return raw, float(np.max(np.abs(command - represented)))


class _RoleActorView:
    """Per-environment facade; never replace a shared runtime actor globally."""

    def __init__(self, base_actor, owner):
        self.base_actor = base_actor
        self.owner = owner

    def run(self, output_names, feeds):
        env = self.owner
        if env.style == "pretrained":
            raw, following_hidden = self.base_actor.run(output_names, feeds)
            raw = np.asarray(raw).copy()
            env._base_actor_calls += 1
        else:
            # AlphaTaskEnv overwrites every opposing and learner action. Only
            # the two support actions below survive, so no base inference is
            # necessary in a completely scripted scene.
            raw = np.zeros((6, 14), dtype=np.float32)
            following_hidden = np.asarray(feeds["hidden"]).copy()
            env._base_actor_calls_skipped += 1
        if raw.shape != (6, 14) or not np.isfinite(raw).all():
            raise ValueError("Frozen actor action interface changed")
        tm, fm = action_masks(env.obs, env.state_dict(), env.runtime.fixture["config"])
        for robot in (env.team * 3, env.team * 3 + 1):
            candidate = env._own_support_candidate(robot)
            desired = env.fixed_combat_action(robot, tm, fm)
            command = env.navigation_command(robot, candidate)
            if command is None:
                desired[:2] = 0.
                env._support_navigation_unreachable += 1
            else:
                desired[:2] = command
            raw[robot, :4], error = _inverse_continuous(desired[:4])
            env._maximum_support_encoding_error = max(env._maximum_support_encoding_error, error)
            env._support_encoding_saturations += int(error > 1e-6)
            target, fire = int(desired[4]), int(desired[5])
            if not tm[robot, target] or not fm[robot, fire]:
                raise RuntimeError("Fixed support skill proposed an illegal target/fire action")
            raw[robot, 8:12] = -1e6
            raw[robot, 8 + target] = 1e6
            raw[robot, 12:14] = -1e6
            raw[robot, 12 + fire] = 1e6
            env._support_candidate_counts[robot % 3][candidate] += 1
        return [raw, following_hidden]


class _RuntimeView:
    def __init__(self, original, owner):
        self.original = original
        self.actor = _RoleActorView(original.actor, owner)

    def __getattr__(self, name):
        return getattr(self.original, name)


class RoleBalancedTaskEnv(AlphaTaskEnv):
    """Same task/observation contract, with fixed role support on the own side.

    Scripted enemy support is mirrored under ``anchors``. ``pretrained`` keeps
    the actual frozen pretrained enemy team, explicitly a strong opponent pool
    rather than pretending both teams then use identical control logic.
    """

    scenario_id = SCENARIO_ID

    def __init__(self, runtime, seed=0, *, support_mode="anchors", randomization=True,
                 opponent_style=None, side=None, reward_profile="upstream", config_path=None):
        self.config_path = Path(config_path or DEFAULT_CONFIG).resolve()
        config_bytes = self.config_path.read_bytes()
        self.config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        self.scene_config = json.loads(config_bytes)
        if self.scene_config.get("scenario_id") != SCENARIO_ID:
            raise ValueError("Scene config scenario differs from the role-balanced adapter")
        if self.scene_config.get("observation_semantics") != \
                "unchanged_alpharm_task_state148_candidates16x16_mask16":
            raise ValueError("This adapter requires the unchanged prototype observation schema")
        if self.scene_config.get("support_modes") != list(SUPPORT_MODES) or support_mode not in SUPPORT_MODES:
            raise ValueError("Unknown or changed fixed support mode set")
        if self.scene_config.get("real_nav2_in_loop") is not False or \
                self.scene_config.get("official_UL_UC_training") is not False:
            raise ValueError("Role-balanced prototype cannot be labelled real or official")
        anchors = self.scene_config.get("anchor_candidates")
        if anchors != [5, 7] or any(isinstance(item, bool) for item in anchors):
            raise ValueError("The versioned role assignments require support anchors [5,7]")
        self.anchor_candidates = tuple(anchors)
        threshold = self.scene_config.get("support_retreat_hp_fraction", .40)
        if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1:
            raise ValueError("Support HP retreat threshold must lie in [0,1]")
        self.retreat_hp_threshold = float(threshold)
        self.support_mode = support_mode
        self.base_runtime = runtime
        super().__init__(_RuntimeView(runtime, self), seed, randomization=randomization,
            opponent_style=opponent_style, side=side, reward_profile=reward_profile)

    def reset(self):
        self._base_actor_calls = self._base_actor_calls_skipped = 0
        self._support_navigation_unreachable = self._support_encoding_saturations = 0
        self._maximum_support_encoding_error = 0.
        self._support_candidate_counts = (Counter(), Counter())
        # Inherited reset samples observe exactly once. Do not sample a second
        # noisy observation just to attach scene metadata.
        return super().reset()

    def _own_support_candidate(self, robot):
        if robot // 3 != self.team or robot % 3 >= 2:
            raise ValueError("Only the two own support robots use this role controller")
        if self.obs[robot, 7] < self.retreat_hp_threshold:
            return 2
        if self.support_mode == "anchors":
            return self.anchor_candidates[robot % 3]
        return (1, 3, 4)[robot % 3]

    def scripted_candidate(self, robot):
        # AlphaTaskEnv calls this for the scripted opposing team. Both sides'
        # supporting roles share anchor geometry and observed-HP retreat.
        if self.support_mode == "anchors" and robot % 3 < 2:
            return 2 if self.obs[robot, 7] < self.retreat_hp_threshold else self.anchor_candidates[robot % 3]
        return super().scripted_candidate(robot)

    def scene_metadata(self):
        return {"scenario_id":SCENARIO_ID, "support_mode":self.support_mode,
            "config_path":str(self.config_path), "config_sha256":self.config_sha256,
            "anchor_candidates":list(self.anchor_candidates),
            "support_retreat_hp_threshold":self.retreat_hp_threshold,
            "own_support_uses_pretrained_actor":False,
            "enemy_uses_pretrained_actor":self.style == "pretrained",
            "original_observation_semantics_retained":True,
            "official_observation_legality_claimed":False,
            "physical_rule_reward_constants_changed":False,
            "true_done_preserved":True, "real_nav2_in_loop":False,
            "official_UL_UC_training":False,
            "base_actor_calls":self._base_actor_calls,
            "base_actor_calls_skipped":self._base_actor_calls_skipped,
            "support_navigation_unreachable":self._support_navigation_unreachable,
            "support_encoding_saturations":self._support_encoding_saturations,
            "maximum_support_encoding_error":self._maximum_support_encoding_error,
            "continuous_decoder_limit":RAW_CONTINUOUS_BOUND,
            "support_candidate_counts":[dict(counter) for counter in self._support_candidate_counts]}

    def step(self, action):
        observation, reward, done, info = super().step(action)
        metadata = self.scene_metadata()
        info["strategy_scene"] = metadata
        if "episode" in info:
            info["episode"] = {**info["episode"], "scenario_id":SCENARIO_ID,
                               "strategy_scene":metadata}
        return observation, reward, done, info


StrategyTaskEnv = RoleBalancedTaskEnv
