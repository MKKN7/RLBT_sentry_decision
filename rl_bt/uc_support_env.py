"""UC local observed-support task course, with a calibrated execution proxy.

This is deliberately NOT a MuJoCo/Nav2 training environment or a match model.
Only previously observed support labels are used; all routes use exact checked
hub edges. Delays, stalls and missing updates are explicit engineering stress
distributions, not learned opponent statistics. Actors only propose intents.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from rl_bt.support_costmap import path_support_check
from rl_bt.voxel_terrain import TerrainGrid

ROOT = Path(__file__).resolve().parents[1]
UC_SUPPORT_SCENARIO_ID = "uc2027_observed_support_multistage_v1"
DT = .5
GAMMA = .99
RADIUS_M = .411
MAX_STEPS = 120
STATE_FEATURES = (
    "objective_dx", "objective_dy", "objective_route_remaining", "time_left",
    "stage_fraction", "stages_remaining", "active", "active_age",
    "stall_age", "support_valid", "support_age", "last_veto",
    "last_arrival", "last_recovery", "active_target_dx", "active_target_dy",
    "active_route_remaining", "acceptance_pending", "no_safe_goals",
    "pose_valid", "progress_last_step", "legal_count", "recovery_cooldown",
    "time_since_goal_switch")
CANDIDATE_FEATURES = (
    "dx", "dy", "route_length", "distance_to_objective", "objective",
    "waypoint", "recover", "continue", "hold", "same_as_active",
    "support_age", "available")
SCORE_DEFINITION = {"task_progress": 60., "time_efficiency": 20.,
                    "recovery_discipline": 10., "safety": 10.}


class UCSupportEnv:
    """Fixed .5 s task-intent MDP, 24 state and 16x12 candidate features.

    0..12 propose a checked waypoint, 13 cancels/recovers, 14 continues,
    15 waits. BT autonomously stops for missing support; recovery never frees
    cells or bypasses its gate. Multi-stage objectives remain fixed throughout
    the episode, so the shaping potential cannot be changed by an action.
    """
    def __init__(self, seed=271, split="train", randomization=True, dataset_dir=None):
        self.dataset_dir = Path(dataset_dir or ROOT / "data/derived/uc_support_training_v1")
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown dataset split")
        self.split, self.randomization = split, bool(randomization)
        self.dataset_manifest = json.loads((self.dataset_dir / "manifest.json").read_text())
        self.dataset_sha = hashlib.sha256((self.dataset_dir / "data.npz").read_bytes()).hexdigest()
        with np.load(self.dataset_dir / "data.npz", allow_pickle=False) as archive:
            self.data = {k: archive[k].copy() for k in archive.files}
        self.metadata = json.loads((self.dataset_dir / "metadata.json").read_text())["samples"]
        self.splits = json.loads((self.dataset_dir / "splits.json").read_text())
        self.indices = self.splits["sample_indices"][split]
        if not self.indices:
            raise ValueError("Empty source split")
        self.calibration = json.loads((self.dataset_dir / "execution_calibration.json").read_text())
        speeds = self.calibration["training_summary"]["estimated_displacement_speed_mps"]
        self.base_speed = float(speeds["median"] or .18)
        self.grids, self.hubs, self.pools = {}, {}, {}
        for i in self.indices:
            grid = TerrainGrid("odom", float(self.data["sampled_at_s"][i]),
                self.data["origin_xy_m"][i], float(self.data["resolution_m"][i]),
                self.data["occupancy_xy"][i], np.empty((0, 0)), np.empty((0, 0)),
                np.empty((0, 3)), {})
            self.grids[i] = grid
            self.hubs[i] = self.data["start_xy_m"][i].copy()
            pool = self.data["supported_goal_xy_m"][i][self.data["goal_mask"][i]].copy()
            # Revalidate derived proposals instead of trusting a cached mask.
            for point in pool:
                if not path_support_check(grid, [self.hubs[i], point], radius_m=RADIUS_M)["accepted"]:
                    raise ValueError(f"Unsupported dataset edge at sample {i}")
            self.pools[i] = pool
        self.rng = np.random.default_rng(seed)
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.episode_seed = int(seed) if seed is not None else None
        self.sample_index = int(self.rng.choice(self.indices))
        self.hub = self.hubs[self.sample_index].copy()
        self.pool = self.pools[self.sample_index]
        self.no_goals = len(self.pool) == 0
        self.pos = self.hub.copy()  # explicit reset, not a claimed physical transition
        self.anchor = self.hub.copy()
        self.stage = 0
        count = int(self.rng.integers(2, 6)) if self.randomization else 3
        if self.no_goals:
            self.objectives = np.repeat(self.hub[None], count, axis=0)
        else:
            order = self.rng.choice(len(self.pool), count, replace=len(self.pool) < count)
            self.objectives = self.pool[order].copy()
        # Fixed candidate identities for each episode; objective is inserted
        # into slot zero. Other slots are distinct distractor waypoints.
        self.candidate_order = self.rng.permutation(len(self.pool))
        self.steps = 0
        self.time_s = 0.
        self.active = None
        self.route = []
        self.pending = 0.
        self.active_age = self.stall_age = self.cooldown = self.switch_age = 0.
        self.last_veto = self.last_arrival = self.last_recovery = 0.
        self.last_progress = 0.
        self.recoveries = self.veto_count = self.switches = self.needless_recoveries = 0
        self.health_stops = self.unsafe_executions = 0
        self.support_gate_rejections = self.checked_motion_segments = 0
        self.return_total = 0.
        self.done = False
        self.fault_cleared = False
        self.scenario = str(self.rng.choice(["nominal", "delayed_acceptance", "support_gap", "executor_stall"])) if self.randomization else "nominal"
        self.speed = self.base_speed * (float(self.rng.uniform(.65, 1.25)) if self.randomization else 1.)
        self.acceptance_delay = float(self.rng.uniform(.1, .8)) if self.randomization else .2
        if self.scenario == "delayed_acceptance":
            self.acceptance_delay = float(self.rng.uniform(.8, 1.8))
        # Stress intervals are sampled once by simulated time, independently
        # of the policy's sequence or number of recoveries.
        self.fault_begin = float(self.rng.uniform(2., 5.))
        self.fault_end = self.fault_begin + float(self.rng.uniform(3., 6.))
        self.pose_noise = self.rng.normal(0., .008 if self.randomization else 0., (MAX_STEPS + 2, 2))
        self.support_received = self.rng.random(MAX_STEPS + 2) >= (.08 if self.randomization else 0.)
        self.support_age = 0.
        self.scheduled_length = self._total_mission_length()
        self.ideal_time = self.scheduled_length / self.base_speed + len(self.objectives) * .2
        self.wait_required_s = 5.
        self.obs = self._observation()
        return copy.deepcopy(self.obs)

    def _total_mission_length(self):
        current = self.hub
        length = 0.
        for target in self.objectives:
            length += float(np.linalg.norm(current - self.hub) + np.linalg.norm(target - self.hub))
            current = target
        return length

    def _route_to(self, target):
        # Reverse the current certified hub edge before changing branches.
        # This avoids an unchecked diagonal between free endpoints.
        if np.linalg.norm(self.pos - self.anchor) < 1e-9:
            points = [self.hub.copy(), target.copy()]
        elif np.linalg.norm(self.anchor - self.hub) < 1e-9:
            points = [self.hub.copy(), target.copy()]
        else:
            # While travelling hub->endpoint, pos lies on its certified edge;
            # returning directly to hub stays on that same edge.
            points = [self.hub.copy(), target.copy()]
        if self.active is not None and np.linalg.norm(target - self.active) < 1e-8:
            return [p.copy() for p in self.route]
        return [p for p in points if np.linalg.norm(p - self.pos) > 1e-9]

    def _length(self, route):
        p = self.pos
        total = 0.
        for q in route:
            total += float(np.linalg.norm(q - p))
            p = q
        return total

    def _objective(self):
        return self.objectives[min(self.stage, len(self.objectives) - 1)]

    def _remaining(self):
        if self.no_goals or self.stage == len(self.objectives):
            return 0.
        total = self._length(self._route_to(self._objective()))
        for k in range(self.stage + 1, len(self.objectives)):
            total += float(np.linalg.norm(self.objectives[k-1] - self.hub)
                           + np.linalg.norm(self.objectives[k] - self.hub))
        return total

    def _healthy(self):
        return self.support_age <= 1.5 and not self.no_goals

    def _candidate_points(self):
        if self.no_goals:
            return []
        points = [self._objective().copy()]
        points.extend(self.pool[i].copy() for i in self.candidate_order
                      if np.linalg.norm(self.pool[i] - self._objective()) > 1e-8)
        return points[:13]

    def _observation(self):
        est = self.pos + self.pose_noise[min(self.steps, MAX_STEPS)]
        goal = self._objective()
        active_delta = np.zeros(2) if self.active is None else self.active - est
        state = np.array([*(goal-est), self._remaining()/5.,
            1.-self.steps/MAX_STEPS, self.stage/len(self.objectives),
            (len(self.objectives)-self.stage)/5., float(self.active is not None),
            min(self.active_age/10., 1.), min(self.stall_age/3., 1.),
            float(self._healthy()), min(self.support_age/3., 1.),
            self.last_veto, self.last_arrival, self.last_recovery,
            *active_delta, self._length(self.route)/5., float(self.pending > 0.),
            float(self.no_goals), 1., self.last_progress/.15, 0.,
            self.cooldown, min(self.switch_age/5., 1.)], dtype=np.float32)
        cand = np.zeros((16, 12), np.float32)
        mask = np.zeros(16, bool)
        self.points = self._candidate_points()
        for k, point in enumerate(self.points):
            same = self.active is not None and np.linalg.norm(point-self.active) < 1e-8
            cand[k] = [*(point-est), self._length(self._route_to(point)),
                np.linalg.norm(point-goal), float(np.linalg.norm(point-goal) < 1e-8),
                1., 0., 0., 0., float(same), min(self.support_age/3., 1.), 1.]
            mask[k] = self._healthy() and self.cooldown <= 0.
        cand[13, 6] = 1.
        cand[14, 7] = 1.
        cand[15, 8] = 1.
        mask[13] = self.active is not None and self.cooldown <= 0.
        mask[14] = self.active is not None and self._healthy()
        mask[15] = True
        cand[13:, 11] = mask[13:]
        state[21] = mask.sum()/16.
        return {"state": np.clip(state, -10., 10.), "candidates": np.clip(cand, -10., 10.), "mask": mask}

    def _advance(self):
        if self.active is None:
            return
        self.active_age += DT
        if self.pending > 0.:
            self.pending = max(0., self.pending-DT)
            return
        if self.scenario == "executor_stall" and self.fault_begin <= self.time_s < self.fault_end and not self.fault_cleared:
            self.stall_age += DT
            return
        self.stall_age = 0.
        budget = self.speed * DT
        old = self.pos.copy()
        while budget > 1e-10 and self.route:
            delta = self.route[0]-self.pos
            distance = float(np.linalg.norm(delta))
            endpoint = (self.route[0].copy() if distance <= budget else
                        self.pos + delta/distance*budget)
            check = path_support_check(self.grids[self.sample_index],
                [self.pos, endpoint], radius_m=RADIUS_M)
            self.checked_motion_segments += 1
            if not check["accepted"]:
                self.support_gate_rejections += 1
                self.active, self.route = None, []
                self.pending = 0.
                return
            if distance <= budget:
                self.pos = self.route.pop(0).copy()
                self.anchor = self.pos.copy()
                budget -= distance
            else:
                self.pos += delta/distance * budget
                budget = 0.
        self.last_progress = float(np.linalg.norm(self.pos-old))
        if not self.route:
            if np.linalg.norm(self.pos-self._objective()) < 1e-7:
                self.stage += 1
                self.last_arrival = 1.
            self.active = None
            self.active_age = self.stall_age = 0.

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset required after episode end")
        proposed = int(action)
        old_potential = -.5 * self._remaining()
        old_stage = self.stage
        self.last_veto = self.last_arrival = self.last_recovery = self.last_progress = 0.
        parts = {"task": 0., "time": -.02, "potential": 0., "veto": 0.,
                 "switch": 0., "recovery": 0.}
        executed = proposed
        reason = None
        if proposed < 0 or proposed >= 16 or not self.obs["mask"][proposed]:
            self.last_veto = 1.
            self.veto_count += 1
            parts["veto"] = -.15
            executed, reason = 15, "unavailable_intent"
        if executed < 13:
            target = self.points[executed]
            if self.active is None or np.linalg.norm(target-self.active) > 1e-8:
                if self.active is not None:
                    parts["switch"] = -.06
                    self.switches += 1
                self.route = self._route_to(target)
                self.active = target.copy()
                self.pending = self.acceptance_delay
                self.active_age = self.stall_age = self.switch_age = 0.
        elif executed == 13:
            useful = self.stall_age >= 1.
            self.recoveries += 1
            self.needless_recoveries += int(not useful)
            self.last_recovery = 1.
            parts["recovery"] = -.03 if useful else -.15
            if useful:
                self.fault_cleared = True
            self.active, self.route = None, []
            self.pending = self.stall_age = 0.
            self.cooldown = .5
        # WAIT holds the robot and cancels active intent; CONTINUE preserves it.
        elif executed == 15 and self.active is not None:
            self.active, self.route = None, []
            self.pending = 0.
            self.switches += 1
            parts["switch"] = -.06
        self.steps += 1
        self.time_s = self.steps*DT
        self.switch_age += DT
        self.cooldown = max(0., self.cooldown-DT)
        gap = self.scenario == "support_gap" and self.fault_begin <= self.time_s < self.fault_end
        if gap or not self.support_received[self.steps]:
            self.support_age += DT
        else:
            self.support_age = 0.
        if self.active is not None and not self._healthy():
            self.active, self.route = None, []
            self.pending = 0.
            self.health_stops += 1
            reason = "support_stale_BT_stop"
        self._advance()
        parts["task"] = 1.5*(self.stage-old_stage)
        success = self.stage == len(self.objectives) if not self.no_goals else self.time_s >= self.wait_required_s
        terminated = bool(success)
        truncated = self.steps >= MAX_STEPS and not terminated
        self.done = terminated or truncated
        if success:
            parts["task"] += 3.
        # True terminal potential is zero; truncation retains its final value.
        next_potential = 0. if terminated else -.5*self._remaining()
        parts["potential"] = GAMMA*next_potential-old_potential
        reward = float(sum(parts.values()))
        self.return_total += reward
        self.obs = self._observation()
        info = {"reward_components": parts, "BT_veto": bool(self.last_veto),
            "proposed_action": proposed, "executed_action": executed,
            "BT_reason": reason, "terminated": terminated, "truncated": truncated}
        if self.done:
            progress = float(success) if self.no_goals else self.stage/len(self.objectives)
            # Fixed 100 point score is an evaluation metric, not the RL reward.
            time_reference = self.wait_required_s if self.no_goals else max(DT, self.ideal_time)
            scores = {"task_progress": 60.*progress,
                "time_efficiency": 20.*progress*min(1., time_reference/max(DT, self.time_s)),
                "recovery_discipline": 10.*progress/(1.+self.needless_recoveries+.2*self.switches),
                "safety": 10./(1.+self.veto_count+self.unsafe_executions)}
            info["episode"] = {"success": bool(success),
                "navigation_success": bool(success and not self.no_goals),
                "wait_success": bool(success and self.no_goals),
                "return": self.return_total, "steps": self.steps, "time_s": self.time_s,
                "recoveries": self.recoveries, "BT_veto_count": self.veto_count,
                "switches": self.switches, "needless_recoveries": self.needless_recoveries,
                "health_stops": self.health_stops, "unsafe_executions": self.unsafe_executions,
                "support_gate_rejections": self.support_gate_rejections,
                "checked_motion_segments": self.checked_motion_segments,
                "stage_completed": self.stage, "stages": len(self.objectives),
                "score_100": float(sum(scores.values())), "score_components": scores,
                "source_run_id": self.metadata[self.sample_index]["source_run"],
                "parameters": {"seed": self.episode_seed, "source_sample": self.sample_index,
                    "split": self.split, "scenario": self.scenario, "speed_mps": self.speed,
                    "acceptance_delay_s": self.acceptance_delay, "fault_begin_s": self.fault_begin,
                    "fault_end_s": self.fault_end, "objective_xy_m": self.objectives.tolist(),
                    "scheduled_length_m": self.scheduled_length, "no_safe_goal": self.no_goals,
                    "fixed_start_reset": self.metadata[self.sample_index]["fixed_start_reset_is_synthetic"],
                    "executor": "calibrated_proxy_NOT_Nav2_rollout"}}
        return copy.deepcopy(self.obs), reward, self.done, info

    def state_dict(self):
        excluded = {"data", "metadata", "splits", "indices", "calibration", "grids", "hubs", "pools", "dataset_manifest", "dataset_dir", "rng"}
        return {"scenario_id": UC_SUPPORT_SCENARIO_ID, "dataset_sha256": self.dataset_sha,
                "rng": copy.deepcopy(self.rng.bit_generator.state),
                "state": copy.deepcopy({k: v for k, v in self.__dict__.items() if k not in excluded})}

    def load_state_dict(self, snapshot):
        if snapshot["scenario_id"] != UC_SUPPORT_SCENARIO_ID or snapshot["dataset_sha256"] != self.dataset_sha:
            raise ValueError("Environment snapshot identity mismatch")
        if snapshot["state"]["split"] != self.split:
            raise ValueError("Cannot resume a different source split")
        self.__dict__.update(copy.deepcopy(snapshot["state"]))
        self.rng.bit_generator.state = copy.deepcopy(snapshot["rng"])


def uc_support_bt_baseline(observation):
    """Deterministic teacher with exactly the policy-visible information."""
    state, candidates, mask = (observation[k] for k in ("state", "candidates", "mask"))
    if mask[13] and state[8] >= 1./3.:
        return 13
    if mask[14]:
        return 14
    goals = np.flatnonzero(mask[:13])
    if len(goals):
        return int(min(goals, key=lambda i: (candidates[i, 3], candidates[i, 2])))
    return 15
