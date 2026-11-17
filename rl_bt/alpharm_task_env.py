"""Task-level single-learner wrapper around the published alphaRM ONNX demo.

This is deliberately named a simplified prototype, not an official RMUL/UC env.
Only candidate selection is learned. A fixed executor handles movement/aim/fire.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from rl_bt.contracts import IntentGate, TaskIntent
from rl_bt.geometry import PlanarProxy, ROOT
from rl_bt.rewards import REWARD_PROFILES, reward_tick
from scripts.probe_alpharm_onnx import action_masks, decode

SCENARIO = "alpharm_task_prototype_v1"
STATE_DIM, CANDIDATE_DIM, N_CANDIDATES = 148, 16, 16
CANDIDATE_NAMES = ["continue", "center", "home", "center_north", "center_south",
                   "cover_west_lower", "cover_west_inner", "cover_west_upper",
                   "cover_east_upper", "cover_east_inner", "cover_east_lower",
                   "flank_southwest", "flank_northeast", "flank_north",
                   "flank_south", "flank_east"]
TRAIN_STYLES = ("pretrained", "capture", "defense", "resupply")
HELD_OUT_STYLES = ("pursuit",)


class AlphaRuntime:
    """Share read-only ORT sessions across sequential env instances."""
    def __init__(self):
        directory = ROOT / "data/references/alpharm_jax_review/upstream/demo/web"
        manifest = json.loads((ROOT / "data/references/alpharm_jax_review_manifest.json").read_text())
        for entry in manifest["files"]:
            path = ROOT / entry["local_path"]
            if path.parent == directory and path.name in ("actor.onnx", "env_step.onnx", "fixture.json"):
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError(f"Changed upstream snapshot: {path}")
        self.source_commit = manifest["commit"]
        self.fixture = json.loads((directory / "fixture.json").read_text())
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.actor = ort.InferenceSession(str(directory / "actor.onnx"), sess_options=opts,
                                          providers=["CPUExecutionProvider"])
        self.env = ort.InferenceSession(str(directory / "env_step.onnx"), sess_options=opts,
                                        providers=["CPUExecutionProvider"])
        self.inputs = [x.name for x in self.env.get_inputs()]
        self.index = {x["name"]: x["index"] for x in self.fixture["state"]}
        self.navigation = PlanarProxy()


class AlphaTaskEnv:
    def __init__(self, runtime, seed=0, *, randomization=True, opponent_style=None, side=None,
                 reward_profile="upstream"):
        if reward_profile not in REWARD_PROFILES:
            raise ValueError(f"Unknown reward profile: {reward_profile}")
        self.runtime = runtime
        self.reward_profile = reward_profile
        self.rng = np.random.default_rng(seed)
        self.randomization = randomization
        self.style_override, self.side_override = opponent_style, side
        self.nav = runtime.navigation
        self.reset()

    def state_dict(self):
        return {k: self.state[i] for k, i in self.runtime.index.items()}

    @property
    def now(self):
        return float(self.state[self.runtime.index["episode_elapsed"]])

    def reset(self):
        fixture = self.runtime.fixture
        self.state = [decode(x) for x in fixture["state"]]
        self.obs = decode(fixture["obs"])
        self.hidden = decode(fixture["hidden"])
        self.team = int(self.rng.integers(2)) if self.side_override is None else self.side_override
        self.learner = 2 + 3 * self.team
        self.style = self.style_override or str(self.rng.choice(TRAIN_STYLES))
        if self.style not in TRAIN_STYLES + HELD_OUT_STYLES:
            raise ValueError(f"Unknown opponent style: {self.style}")
        # Explicit phase-I assumptions, not calibrated real sensor parameters.
        self.position_std_m = float(self.rng.uniform(0, .08)) if self.randomization else 0.
        self.miss_probability = float(self.rng.uniform(0, .12)) if self.randomization else 0.
        self.obs_delay = int(self.rng.integers(2)) if self.randomization else 0
        self.command_delay_ticks = int(self.rng.integers(3)) if self.randomization else 0
        self.speed_scale = float(self.rng.uniform(.75, 1.)) if self.randomization else 1.
        self.dropout = np.zeros(3, bool)
        self.history = deque([(self.obs.copy(), self.now)], maxlen=3)
        self.active = 0
        self.active_since = 0.
        self.gate = IntentGate()
        self.version = 0
        self.last_override = False
        self.switches = self.overrides = self.recoveries = 0
        self.progress_position = self.state_dict()["pos"][self.learner].copy()
        self.no_progress_s = 0.
        self.episode_return = 0.
        self.episode_reward_components = {}
        self.decisions = 0
        self.done = False
        return self.observe()

    def observed_position(self, obs, i):
        sign = 1 if i < 3 else -1
        return obs[i, :2] * np.asarray([6., 4.]) * sign

    def visible_enemy_positions(self, local, position, team):
        block = local[68:119].reshape(3, 17)
        valid = (block[:, 0] > .5) & (block[:, 14] > .5)
        positions = position + block[valid, 1:3] * 12 * (1 if team == 0 else -1)
        return positions

    def observe(self):
        history_index = max(0, len(self.history) - 1 - self.obs_delay)
        snapshot, sampled_at = self.history[history_index]
        local = snapshot[self.learner].copy()
        # Correlated missing detections; each observation is sampled once and
        # the resulting cached tensor is used for both features and PPO logp.
        self.dropout = np.where(self.dropout, self.rng.random(3) < .65,
                                self.rng.random(3) < self.miss_probability)
        enemy = local[68:119].reshape(3, 17)
        enemy[self.dropout] = 0
        local[:2] += self.rng.normal(0, self.position_std_m, 2) / [6., 4.]
        for e in enemy:
            if e[0] > .5:
                e[1:3] += self.rng.normal(0, self.position_std_m, 2) / 12
        # Candidate features are computed solely from the actor's sampled obs.
        all_local = np.zeros((6, 140), np.float32)
        all_local[self.learner] = local
        pos = self.observed_position(all_local, self.learner)
        goals = self.nav.goals[self.team]
        if self.active:
            active_goal = goals[self.active - 1]
        else:
            active_goal = pos
        points = np.vstack((active_goal, goals))
        costs = np.r_[0. if not self.active else self.nav.costs(pos, self.team)[self.active - 1],
                      self.nav.costs(pos, self.team)]
        enemies = self.visible_enemy_positions(local, pos, self.team)
        threat = np.zeros(N_CANDIDATES)
        for p in enemies:
            threat += np.exp(-np.linalg.norm(points - p, axis=1) / 3.)
        feature = np.zeros((N_CANDIDATES, CANDIDATE_DIM), np.float32)
        sign = 1 if self.team == 0 else -1
        feature[:, :2] = (points - pos) * sign / 12
        feature[:, 2] = np.linalg.norm(points - pos, axis=1) / 12
        feature[:, 3] = np.minimum(costs, 100) / 20
        # Nominal ETA; never expose the sampled execution-error truth.
        feature[:, 4] = np.minimum(costs / 3, 100) / 15
        feature[[1, 3, 4], 5] = 1  # central-control-area variants
        feature[2, 6] = 1
        feature[5:11, 7] = 1
        feature[11:, 8] = 1
        feature[0, 9] = 1
        feature[self.active, 10] = 1
        feature[:, 11] = threat / 3
        feature[:, 12] = len(enemies) / 3
        feature[:, 13:15] = local[7:9]
        feature[:, 15] = local[125]  # source HUD time channel, see upstream get_obs
        mask = np.isfinite(costs)
        if local[11] < .5:
            mask[:] = False
        mask[0] = True  # executor can always continue/hold safely
        context = np.asarray([self.active / 15, (self.now - self.active_since) / 20,
                              np.linalg.norm(active_goal - pos) / 20,
                              (self.now - sampled_at) / 2, float(self.last_override),
                              float(self.active > 0), 0., 1.], np.float32)
        self.cached = {"state": np.r_[local, context].astype(np.float32),
                       "candidates": feature, "mask": mask}
        return {k: v.copy() for k, v in self.cached.items()}

    def fixed_combat_action(self, i, tm, fm):
        """Nearest visible target; fire whenever the fixed upstream gate allows."""
        local = self.obs[i]
        enemy = local[68:119].reshape(3, 17)
        dist = np.linalg.norm(enemy[:, 1:3], axis=1)
        legal = tm[i, 1:]
        target = int(np.argmin(np.where(legal, dist, np.inf))) + 1 if legal.any() else 0
        return np.asarray([0., 0., .35 if not target else 0., 0., target,
                           float(fm[i, 1] and target > 0)], np.float32)

    def navigation_command(self, i, candidate_id, speed_scale=1.):
        if not candidate_id:
            return np.zeros(2, np.float32)
        team = 0 if i < 3 else 1
        # Execution receives current own odometry, not hidden enemy truth.
        pos = self.observed_position(self.obs, i)
        point = self.nav.waypoint(pos, team, candidate_id - 1)
        if point is None:
            return None
        goal = self.nav.goals[team, candidate_id - 1]
        if np.linalg.norm(pos - goal) < .22:
            return np.zeros(2, np.float32)
        delta = point - pos
        if np.linalg.norm(delta) < .05:
            delta = goal - pos
        velocity = delta * 4.
        norm = np.linalg.norm(velocity)
        velocity *= min(1., 3 * speed_scale / max(norm, 1e-6))
        return (velocity / 3 * (1 if team == 0 else -1)).astype(np.float32)

    def scripted_candidate(self, i):
        local = self.obs[i]
        team = 0 if i < 3 else 1
        if self.style == "defense":
            return 2 if local[7] < .55 else 7
        if self.style == "resupply" and local[7] < .55:
            return 2
        if self.style == "pursuit":
            pos = self.observed_position(self.obs, i)
            enemies = self.visible_enemy_positions(local, pos, team)
            if len(enemies):
                nearest = enemies[np.argmin(np.linalg.norm(enemies - pos, axis=1))]
                return int(np.argmin(np.linalg.norm(self.nav.goals[team] - nearest, axis=1))) + 1
        return (1, 3, 4)[i % 3]

    def step(self, action):
        if self.done:
            raise RuntimeError("reset() required after terminal state")
        proposed = int(action)
        request = TaskIntent(proposed, self.decisions, self.version, self.now)
        reason = self.gate.validate(request, version=self.version, now=self.now,
                                    mask=self.cached["mask"],
                                    alive=bool(self.state_dict()["alive"][self.learner]))
        if reason is None and proposed:
            if not np.isfinite(self.nav.costs(self.state_dict()["pos"][self.learner],
                                              self.team)[proposed - 1]):
                reason = "navigation_unreachable"
        selected = self.active if reason or not proposed else proposed
        switched = selected != self.active
        self.last_override = reason is not None
        self.switches += int(switched)
        old_active = self.active
        reward = 0.
        reward_components = {}
        tick_count = 0
        before = self.now
        for tick in range(5):  # 10 Hz physics/proxy, 2 Hz task decisions
            if tick >= self.command_delay_ticks and self.active != selected:
                self.active = selected
                self.active_since = self.now
            st = self.state_dict()
            tm, fm = action_masks(self.obs, st, self.runtime.fixture["config"])
            raw, self.hidden = self.runtime.actor.run(None, {"obs": self.obs, "hidden": self.hidden})
            actions = np.column_stack((np.tanh(3 * np.tanh(raw[:, :4] / 3)),
                                       np.argmax(np.where(tm, raw[:, 8:12], -np.inf), axis=1),
                                       np.argmax(np.where(fm, raw[:, 12:14], -np.inf), axis=1))).astype(np.float32)
            # Other team: one style per episode, switching behavior by local state.
            if self.style != "pretrained":
                for i in range(6):
                    if (i < 3) != (self.team == 0):
                        actions[i] = self.fixed_combat_action(i, tm, fm)
                        command = self.navigation_command(i, self.scripted_candidate(i))
                        actions[i, :2] = 0 if command is None else command
            actions[self.learner] = self.fixed_combat_action(self.learner, tm, fm)
            command = self.navigation_command(self.learner, self.active, self.speed_scale)
            if command is None:
                # BT monitor stops, reports failure and falls back to HOLD.
                self.active = 0
                command = np.zeros(2, np.float32)
                reason = "navigation_failed"
                self.last_override = True
            actions[self.learner, :2] = command
            feeds = dict(zip(self.runtime.inputs[:-1], self.state))
            feeds[self.runtime.inputs[-1]] = actions
            before_state = st
            outputs = self.runtime.env.run(None, feeds)
            self.state, self.obs, self.done = outputs[:-2], outputs[-2], bool(outputs[-1])
            st = self.state_dict()
            # ONNX outputs allocate new arrays; before_state retains pre-tick values.
            parts = reward_tick(before_state, st, team=self.team, learner=self.learner,
                                config=self.runtime.fixture["config"], profile=self.reward_profile)
            reward += parts["selected_total"]
            for key, value in parts.items():
                reward_components[key] = reward_components.get(key, 0.) + value
            tick_count += 1
            if self.done:
                break
        # Progress monitor: abort a blocked task to HOLD; no invented recovery velocities.
        current = self.state_dict()["pos"][self.learner]
        if self.active and np.linalg.norm(current - self.progress_position) < .025:
            goal = self.nav.goals[self.team, self.active - 1]
            self.no_progress_s = self.no_progress_s + tick_count * .1 if np.linalg.norm(current - goal) > .35 else 0.
        else:
            self.no_progress_s = 0.
        if self.no_progress_s >= 2.:
            self.active = 0
            self.no_progress_s = 0.
            self.recoveries += 1
            reason = "blocked_abort_to_hold"
            self.last_override = True
        self.progress_position = current.copy()
        self.overrides += int(reason is not None)
        # Prototype shaping; this is not an official scoring rule.
        switch_cost = .002 * switched
        reward -= switch_cost
        reward_components["switch_cost"] = switch_cost
        reward_components["total"] = reward
        for key, value in reward_components.items():
            self.episode_reward_components[key] = self.episode_reward_components.get(key, 0.) + value
        self.episode_return += reward
        self.decisions += 1
        self.version += 1
        self.history.append((self.obs.copy(), self.now))
        result = self.observe()
        info = {"proposed_candidate": proposed, "executed_candidate": self.active,
                "previous_candidate": old_active, "override_reason": reason,
                "candidate_version": request.candidate_version, "duration_s": tick_count * .1,
                "started_at_s": before, "ended_at_s": self.now, "switch_cost": switch_cost,
                "team": self.team, "opponent_style": self.style}
        info["reward_components"] = reward_components
        if self.done:
            st = self.state_dict()
            info["episode"] = {"scenario_id": SCENARIO, "seed_note": "see run seed and reset order",
                "team": self.team, "opponent_style": self.style,
                "return": self.episode_return, "decisions": self.decisions,
                "reward_profile": self.reward_profile,
                "reward_components": dict(self.episode_reward_components),
                "winner": int(st["winner"]), "outcome": int(st["outcome"]),
                "score": st["score"].tolist(), "win": int(st["winner"]) == self.team + 1,
                "draw": int(st["winner"]) == 0, "switches": self.switches,
                "overrides": self.overrides, "recoveries": self.recoveries,
                "randomization": {"position_std_m": self.position_std_m,
                    "miss_probability": self.miss_probability, "obs_delay_decisions": self.obs_delay,
                    "command_delay_ticks": self.command_delay_ticks, "speed_scale": self.speed_scale}}
        return result, reward, self.done, info


def bt_baseline(observation):
    """Explicit handwritten task baseline, not the real Combat XML runtime."""
    if observation["state"][7] < .40 and observation["mask"][2]:
        return 2
    return 1 if observation["mask"][1] else 0
