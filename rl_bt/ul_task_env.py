"""UL ground-task curriculum with official map/rule submodules.

This is a motion/event proxy, not a complete 3v3 match or Nav2 simulator.
The actor sees delayed self/referee samples and estimated geometry only.
Capture is an uninterrupted BT skill; no undefined neutral-idle decay is guessed.
"""
from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
import struct

import numpy as np

from rl_bt.contracts import TaskIntent
from rl_bt.healing import HealingEligibility
from rl_bt.navigation_contract import (CandidateSnapshot, NavigationCandidate,
                                      NavigationDecision, NavigationExecutor)
from rl_bt.ul_observation import ULPublicObservationCache
from rl_bt.ul_rules import ULControlZone, ULSentryAmmo, ul_supply_healing_rate

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ID = "ul2027_ground_task_curriculum_v4"
STATE_DIM, CANDIDATE_DIM = 24, 12
STATE_FIELDS = ["hp_fraction", "ammo_fraction", "course_time_remaining", "pose_x",
    "pose_y", "pose_sample_age", "pose_valid", "own_supply_rfid", "center_rfid",
    "center_control_own", "center_control_valid", "navigation_active", "task_age",
    "navigation_blocked", "last_goal_distance", "last_BT_veto", "hp_valid",
    "objective_requires_heal", "objective_requires_capture", "team_side",
    "active_resupply", "active_capture", "ammo_valid", "rfid_valid"]
CANDIDATE_FIELDS = ["relative_x", "relative_y", "euclidean_distance", "path_length",
    "resupply", "capture", "position", "tunnel", "highland", "continue", "hold",
    "current_goal"]


class ULTaskEnv:
    period_s = .5
    horizon_s = 60.

    def __init__(self, seed=1, randomization=True, nav=None):
        if nav is None:
            from rl_bt.ul_navigation import ULNavigation
            nav = ULNavigation()
        self.nav = nav
        self.randomization = randomization
        self.rng = np.random.default_rng(seed)
        self.episode_index = -1
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.episode_index += 1
        self.side = int(self.rng.integers(2))
        self.objective = str(self.rng.choice(["capture", "resupply", "recover_then_capture"]))
        self.require_heal = self.objective != "capture"
        self.require_capture = self.objective != "resupply"
        self.goals = self.nav.goal_candidates(self.side)
        if len(self.goals) != 14 or [g["id"] for g in self.goals] != list(range(14)):
            raise ValueError("UL course requires fixed 14 navigation candidates")
        self.supply_id = next(g["id"] for g in self.goals if g["task_type"] == "resupply")
        self.center_id = next(g["id"] for g in self.goals if g["task_type"] == "capture")
        self.supply = np.array(self.goals[self.supply_id]["xy"], float)
        # Start on a known connected ground anchor, never snap onto a highland.
        starts = [g for g in self.goals if g["task_type"] == "position" and g["static_legal"]
                  and self.nav.plan(g["xy"], self.supply, self.side) is not None
                  and self.nav.plan(g["xy"], self.goals[self.center_id]["xy"], self.side) is not None]
        if not starts:
            raise ValueError("No course start connects supply and center under reference envelope")
        self.position = np.array(starts[int(self.rng.integers(len(starts)))]["xy"], float)
        self.hp = float(self.rng.integers(80, 260)) if self.require_heal else 400.
        self.ammo = ULSentryAmmo()
        self.center = ULControlZone()
        self.capture_skill_active = False
        self.capture_interrupted = False
        self.healed = False
        self.time_s, self.steps, self.return_ = 0., 0, 0.
        self.terminal = False
        self.version, self.sequence = 0, 0
        self.last_veto = False
        self.veto_count = 0
        self.recovery_count = 0
        self.path = None
        self.path_index = 0
        self.goal_id = None
        self.accept_after = None
        self.blocked_s = 0.
        self.last_speed = 0.
        self.heal_integral = 0.
        self.heal_settlement_at = 1.
        self.cache = ULPublicObservationCache()
        self.queue = []
        self.pose_queue = []
        self.pose_sample = None
        self.queue_counter = 0
        self.executor = NavigationExecutor(session_id=f"course-{self.episode_index}",
            profile_id="rmul_2027", map_id="rmul2027_3v3_V1.0.0",
            map_hash=self._map_hash(), frame_id="ul_cad_southwest",
            acceptance_timeout_s=1., feedback_timeout_s=1., cancel_timeout_s=.5,
            execution_timeout_s=30.)
        self.parameters = {
            "speed_m_per_s": float(self.rng.uniform(.8, 1.4)) if self.randomization else 1.2,
            "pose_std_m": float(self.rng.uniform(.005, .025)) if self.randomization else 0.,
            "pose_bias_xy_m": self.rng.normal(0., .01, 2).tolist() if self.randomization else [0., 0.],
            "packet_delay_s": float(self.rng.uniform(.03, .20)) if self.randomization else 0.,
            "pose_delay_s": float(self.rng.uniform(.02, .15)) if self.randomization else 0.,
            "packet_drop_probability": float(self.rng.uniform(0., .08)) if self.randomization else 0.,
            "execution_start_delay_s": float(self.rng.uniform(0., .3)) if self.randomization else 0.,
            "block_start_s": float(self.rng.uniform(3., 15.)) if self.randomization else 1e9,
            "block_duration_s": float(self.rng.uniform(1., 3.)) if self.randomization and self.rng.random() < .35 else 0.,
        }
        self._sample_public(force=True)
        self._sample_pose(force=True)
        self._deliver_public()
        self.cached = self._observe()
        return self.cached

    @staticmethod
    def _map_hash():
        return json.loads((ROOT / "assets/rmul_2027/derived/manifest.json").read_text())["geometry_sha256"]

    def _inside_supply(self):
        x, y = self.position
        r = self.nav.radius_m
        if self.side == 0:
            return r <= x <= 2.5-r and 6.+r <= y <= 8.-r
        return 9.5+r <= x <= 12.-r and r <= y <= 2.-r

    def _inside_center(self):
        x, y = self.position
        # Course activation rectangle: proxy for RFID, not measured card coverage.
        return 4.7 <= x <= 7.3 and 2.7 <= y <= 5.3

    def _sample_public(self, force=False):
        owner = 1 if self.center.owner == self.side else 0
        activation = 1 if self._inside_center() else 0
        first_word = int(self._inside_supply())
        second_word = (activation << 24) | (owner << 26)
        rfid = int(self._inside_supply()) | (int(self._inside_center()) << 13)
        payloads = [(0x0201, struct.pack("<BB5HfB", 7+100*self.side, 1,
                      round(self.hp), 400, 30, 260, 100, 0., 7)),
                    (0x0208, struct.pack("<4H", self.ammo.remaining, 0, 0, 0)),
                    (0x0209, struct.pack("<IH", rfid, 0)),
                    (0x0101, struct.pack("<II", first_word, second_word))]
        for command, payload in payloads:
            if not force and self.rng.random() < self.parameters["packet_drop_probability"]:
                continue
            self.queue_counter += 1
            delay = 0. if force else self.parameters["packet_delay_s"]
            heapq.heappush(self.queue, (self.time_s+delay, self.queue_counter,
                                      self.time_s, command, payload))

    def _deliver_public(self):
        while self.queue and self.queue[0][0] <= self.time_s+1e-9:
            receive, _, sample, command, payload = heapq.heappop(self.queue)
            self.cache.ingest(command, payload, byte_order="little", received_at_s=receive,
                              sampled_at_s=sample)
        while self.pose_queue and self.pose_queue[0][0] <= self.time_s+1e-9:
            _, _, sampled, xy = heapq.heappop(self.pose_queue)
            if self.pose_sample is None or sampled > self.pose_sample[0]:
                self.pose_sample = sampled, xy

    def _sample_pose(self, force=False):
        bias = np.asarray(self.parameters["pose_bias_xy_m"])
        xy = self.position+bias+self.rng.normal(0., self.parameters["pose_std_m"], 2)
        self.queue_counter += 1
        delay = 0. if force else self.parameters["pose_delay_s"]
        heapq.heappush(self.pose_queue, (self.time_s+delay, self.queue_counter, self.time_s, xy))

    def _observe(self):
        self.version += 1
        sample_time, self.estimated_xy = self.pose_sample
        pose_age = self.time_s-sample_time
        self.pose_valid = bool(pose_age <= .8 and self.nav.is_free(self.estimated_xy, self.side))
        snap = self.cache.snapshot(now_s=self.time_s,
            max_sample_age_s={0x0201: .8, 0x0208: .8, 0x0209: 1., 0x0101: 2.})
        hp = snap[0x0201]
        ammo = snap[0x0208]
        rfid = snap[0x0209].fields
        events = snap[0x0101].fields
        active = self.executor.active
        state = np.zeros(STATE_DIM, np.float32)
        state[:7] = [hp.fields.get("own_hp", 0)/400.,
            ammo.fields.get("own_allowance_17mm", 0)/750.,
            max(0., 1.-self.time_s/self.horizon_s),
            self.estimated_xy[0]/12., self.estimated_xy[1]/8., min(1., pose_age), self.pose_valid]
        state[7:11] = [rfid.get("own_supply_rfid", False), rfid.get("center_rfid", False),
                       events.get("center_control", 0) == 1, snap[0x0101].valid]
        goal_distance = 0. if self.goal_id is None else np.linalg.norm(
            self.estimated_xy-np.asarray(self.goals[self.goal_id]["xy"]))
        state[11:] = [active is not None or self.capture_skill_active,
            0. if active is None else min(1., (self.time_s-active.dispatched_at_s)/30.),
            self.blocked_s > 0., min(1., goal_distance/15.), self.last_veto, hp.valid,
            self.require_heal, self.require_capture, self.side,
            self.goal_id == self.supply_id, self.goal_id == self.center_id,
            ammo.valid, snap[0x0209].valid]
        candidates = np.zeros((16, CANDIDATE_DIM), np.float32)
        mask = np.zeros(16, bool)
        self.paths = {}
        kinds = {"resupply": 4, "capture": 5, "position": 6, "tunnel": 7, "highland": 8}
        for item in self.goals:
            i = item["id"]
            xy = np.asarray(item["xy"], float)
            delta = xy-self.estimated_xy
            candidates[i, :3] = [delta[0]/12., delta[1]/8., np.linalg.norm(delta)/15.]
            candidates[i, kinds[item["task_type"]]] = 1.
            candidates[i, 11] = self.goal_id == i
            path = self.nav.plan(self.estimated_xy, xy, self.side) if self.pose_valid and item["static_legal"] else None
            if path is not None:
                self.paths[i] = path
                candidates[i, 3] = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()/20.
                mask[i] = True
            else:
                candidates[i, 3] = 1.
        candidates[14, 9] = 1.; candidates[15, 10] = 1.
        mask[14] = ((active is not None and self.executor.state == "active")
                    or self.capture_skill_active) and self.pose_valid
        mask[15] = True
        if self.capture_skill_active:
            # BT's bounded capture skill admits continue/hold only. Both keep
            # the chassis still; neither is permission to abandon activation.
            mask[:14] = False
        if not self.pose_valid:
            mask[:15] = False
        return {"state": state, "candidates": candidates, "mask": mask}

    def _snapshot(self):
        candidates = [NavigationCandidate(i, "navigate", g["task_type"],
            (*g["xy"], 0., 0.), "ul_ground", "ground_proxy",
            legal=bool(g["static_legal"]), reachable=bool(self.cached["mask"][i]))
            for i, g in enumerate(self.goals)]
        candidates += [NavigationCandidate(14, "continue", "continue", reachable=bool(self.cached["mask"][14])),
                       NavigationCandidate(15, "hold", "hold")]
        return CandidateSnapshot("rmul_2027", "rmul2027_3v3_V1.0.0", self._map_hash(),
            "ul_cad_southwest", self.version, self.time_s, 2., tuple(candidates))

    def _apply_commands(self, decision, snapshot):
        for command in decision.commands:
            if command.kind == "cancel_goal":
                # The proxy's cancel completes immediately. The live contract
                # still requires a terminal action result before replacement.
                self.executor.feedback(task_id=command.task_id, status="canceled",
                                       sampled_at_s=self.time_s, now_s=self.time_s)
                self.path, self.goal_id, self.accept_after = None, None, None
                follow = self.executor.dispatch_pending(snapshot, now_s=self.time_s,
                    alive=self.hp > 0, pose_valid=self.pose_valid, frame_ready=True)
                self._apply_commands(follow, snapshot)
            elif command.kind == "send_goal":
                i = command.candidate.candidate_id
                path = self.nav.plan(self.position, self.goals[i]["xy"], self.side)
                if path is None:
                    self.executor.feedback(task_id=command.task_id, status="rejected",
                                           sampled_at_s=self.time_s, now_s=self.time_s)
                    self.last_veto = True; self.veto_count += 1
                    continue
                self.path, self.path_index, self.goal_id = np.asarray(path), 0, i
                self.accept_after = self.time_s+self.parameters["execution_start_delay_s"]
            elif command.kind == "require_safety_stop":
                self.path = None

    def _motion(self, dt):
        self.last_speed = 0.
        active = self.executor.active
        if active is None or self.path is None:
            return
        if self.accept_after is not None and self.time_s >= self.accept_after-1e-9:
            self.executor.feedback(task_id=active.task_id, status="accepted",
                                   sampled_at_s=self.time_s, now_s=self.time_s)
            self.accept_after = None
        if self.executor.state != "active":
            return
        p = self.parameters
        blocked = p["block_start_s"] <= self.time_s < p["block_start_s"]+p["block_duration_s"]
        self.blocked_s = self.blocked_s+dt if blocked else 0.
        if not blocked:
            travel = p["speed_m_per_s"]*dt
            while travel > 1e-9 and self.path_index < len(self.path):
                delta = self.path[self.path_index]-self.position
                distance = np.linalg.norm(delta)
                step = min(distance, travel)
                if distance > 1e-9:
                    self.position += delta*step/distance
                travel -= step
                self.last_speed += step/dt
                if distance <= step+1e-9:
                    self.path_index += 1
            if self.path_index == len(self.path):
                self.executor.feedback(task_id=active.task_id, status="succeeded",
                                       sampled_at_s=self.time_s, now_s=self.time_s)
                self.path = None
                return
        self.executor.feedback(task_id=active.task_id, status="progress",
                               sampled_at_s=self.time_s, now_s=self.time_s)

    def step(self, action):
        if type(action) not in (int, np.int64, np.int32) or not 0 <= action < 16:
            raise ValueError("Integer candidate ID 0..15 required")
        if self.terminal:
            raise RuntimeError("Reset a terminal curriculum episode before stepping")
        target = self.supply if self.require_heal and not self.healed else np.asarray(self.goals[self.center_id]["xy"])
        before_distance = float(np.linalg.norm(self.position-target))
        before_hp = self.hp
        self.last_veto = False
        snapshot = self._snapshot()
        self.sequence += 1
        selected = int(action)
        # Atomic capture dwell prevents undefined partial-neutral idle behavior.
        capture_locked = self.capture_skill_active
        if capture_locked:
            if selected not in (14, 15):
                self.last_veto = True; self.veto_count += 1
            selected = 14
            self.executor.gate.last_sequence = self.sequence
            decision = NavigationDecision("continued", "BT_atomic_capture_skill_running")
        else:
            decision = self.executor.propose(TaskIntent(selected, self.sequence, self.version, self.time_s),
                snapshot, now_s=self.time_s, alive=self.hp > 0, pose_valid=self.pose_valid, frame_ready=True)
        if decision.status == "rejected":
            self.last_veto = True; self.veto_count += 1
        self._apply_commands(decision, snapshot)
        captured_now, healed_now = False, False
        for _ in range(5):
            self.time_s = round(self.time_s+.1, 8)
            # Once capture activation starts, BT keeps the robot still for the
            # isolated five-second skill. No speed is an RL output.
            if self.capture_skill_active:
                self.last_speed = 0.
            else:
                self._motion(.1)
            # Only an explicitly selected capture task runs this atomic skill.
            # Incidental RFID activation while another route crosses the zone
            # is excluded in this course, rather than given a forced BT dwell.
            if (self.require_capture and self.goal_id == self.center_id
                    and self._inside_center() and (not self.require_heal or self.healed)):
                if not self.capture_skill_active and self.center.owner != self.side:
                    stop = self.executor.stop(now_s=self.time_s, reason="BT_atomic_capture_started")
                    self._apply_commands(stop, snapshot)
                    self.goal_id = self.center_id
                    self.capture_skill_active = True
                old = self.center.owner
                self.center.advance(duration_s=.1, red_active=self.side == 0, blue_active=self.side == 1)
                captured_now |= old != self.side and self.center.owner == self.side
                if self.center.owner == self.side:
                    self.capture_skill_active = False
            rate = ul_supply_healing_rate(max_hp=400., own_control=600., enemy_control=600.,
                eligibility=HealingEligibility(own_supply_active=bool(self._inside_supply())))
            self.heal_integral += rate*.1
            if self.time_s+1e-9 >= self.heal_settlement_at:
                self.hp = min(400., self.hp+round(self.heal_integral))
                self.heal_integral = 0.; self.heal_settlement_at += 1.
            if not self.healed and self.hp >= 360. and self._inside_supply():
                self.healed = True; healed_now = True
            self._deliver_public()
        if self.blocked_s >= 1.5:
            recovery = self.executor.stop(now_s=self.time_s, reason="BT_stall_recovery")
            self._apply_commands(recovery, snapshot)
            self.recovery_count += 1; self.blocked_s = 0.
        watch = self.executor.watchdog(now_s=self.time_s, alive=True,
                                      pose_valid=self.pose_valid, frame_ready=True)
        self._apply_commands(watch, snapshot)
        self._sample_public()
        self._sample_pose()
        self._deliver_public()
        success = ((not self.require_heal or self.healed)
                   and (not self.require_capture or self.center.owner == self.side))
        failure = not self.nav.is_free(self.position, self.side)
        done = bool(success or failure or self.time_s >= self.horizon_s)
        self.terminal = done
        after_distance = float(np.linalg.norm(self.position-target))
        # This is a task curriculum objective, not a match-win surrogate.
        components = {"time": -.005, "distance_progress": .10*(before_distance-after_distance),
            "effective_healing": .5*(self.hp-before_hp)/400. if self.require_heal else 0.,
            "heal_milestone": 1. if healed_now and self.require_heal else 0.,
            "capture_milestone": 2. if captured_now and self.require_capture else 0.,
            "success": 3. if success else 0., "BT_veto": -.02*int(self.last_veto),
            "failure": -2. if failure else 0.}
        reward = float(sum(components.values()))
        self.steps += 1; self.return_ += reward
        self.cached = self._observe()
        if self.capture_skill_active and (not self.pose_valid or not self._inside_center()):
            # Exceptional interruption is unresolved by the rule submodule;
            # terminate this course, never silently retain an idle partial.
            self.capture_interrupted = True
            done = True
            self.terminal = True
        info = {"proposed_candidate": int(action), "applied_candidate": selected,
            "BT_reason": decision.reason, "BT_veto": self.last_veto,
            "reward_components": components, "scenario_id": SCENARIO_ID}
        if done:
            info["episode"] = {"success": bool(success and not failure and not self.capture_interrupted), "return": self.return_,
                "steps": self.steps, "objective": self.objective, "side": self.side,
                "time_s": self.time_s, "hp": self.hp, "ammo": self.ammo.remaining,
                "BT_veto_count": self.veto_count, "recovery_count": self.recovery_count,
                "geometry_failure": bool(failure), "parameters": self.parameters}
            info["episode"]["capture_interrupted_requires_referee_reconciliation"] = self.capture_interrupted
        return self.cached, reward, done, info


def ul_bt_baseline(obs):
    """Deterministic rule baseline using exactly the same actor observation."""
    state, candidates, mask = obs["state"], obs["candidates"], obs["mask"]
    if not state[6] or not state[16]:
        return 15
    kind = 4 if state[17] and state[0] < .9 else 5
    choices = np.flatnonzero((candidates[:, kind] > .5) & mask)
    if len(choices):
        return int(choices[0])
    return 14 if mask[14] else 15
