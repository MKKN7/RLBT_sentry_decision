"""Versioned correction of the UC local course's finite task deadline.

V1 remains unchanged and loadable from its source snapshot. A 60 s task with
observed time left is a finite-horizon task: exhausting its deadline is a true
failure terminal, rather than an external truncation of a continuing process.
This version also permutes navigation slots and gives the continue candidate
its observable active-route features. It adds no new free cells or action gate.
"""
from __future__ import annotations

import copy

import numpy as np

from rl_bt.uc_support_env import (DT, GAMMA, MAX_STEPS, UCSupportEnv,
                                  uc_support_bt_baseline)

UC_SUPPORT_V2_SCENARIO_ID = "uc2027_observed_support_finite_task_v2"
UC_SUPPORT_SCENARIO_ID = UC_SUPPORT_V2_SCENARIO_ID
STATE_FEATURES_V2 = "unchanged 24-feature public task-state schema from v1"


class UCSupportEnvV2(UCSupportEnv):
    """Same actual observations, geometry, score and exogenous stress as V1.

    Timeout correction and its explicit failure reward change training returns,
    not the task-completion criterion or 100-point evaluation score. A wait is
    still legal in a healthy state; policy failures are not hidden by a BT mask.
    """
    def reset(self, seed=None):
        # Reset the slot permutation before the superclass builds its first obs.
        # A separate seeded RNG does not change maps/tasks/fault/noise schedules
        # relative to V1, making legacy-actor transfer comparisons meaningful.
        permutation_rng = np.random.default_rng(None if seed is None else int(seed) + 2_000_003)
        self.slot_permutation = permutation_rng.permutation(13)
        obs = super().reset(seed)
        self.idle_healthy_steps = self.recovery_opportunity_steps = 0
        self.action_counts = np.zeros(16, dtype=np.int64)
        return obs

    def _candidate_points(self):
        points = super()._candidate_points()
        if not points:
            return points
        return [points[int(i)] for i in self.slot_permutation if i < len(points)]

    def _observation(self):
        obs = super()._observation()
        if self.active is not None:
            est = self.pos + self.pose_noise[min(self.steps, MAX_STEPS)]
            candidate = obs["candidates"][14]
            candidate[:4] = [*(self.active-est), self._length(self.route),
                              np.linalg.norm(self.active-self._objective())]
            candidate[4] = float(np.linalg.norm(self.active-self._objective()) < 1e-8)
            candidate[9] = 1.
            candidate[10] = min(self.support_age/3., 1.)
        return obs

    def step(self, action):
        previous_stage = self.stage
        old_healthy_idle = self._healthy() and self.active is None
        if 0 <= int(action) < 16:
            self.action_counts[int(action)] += 1
        self.idle_healthy_steps += int(old_healthy_idle and int(action) == 15)
        self.recovery_opportunity_steps += int(self.active is not None and self.stall_age >= 1.)
        obs, reward, done, info = super().step(action)
        if info["truncated"]:
            # V1 emitted gamma*Phi(final)-Phi(old). A true absorbing failure
            # terminal requires Phi(final)=0, with the same fixed objective.
            correction = GAMMA * .5 * self._remaining()
            failure = -3. * (1. - self.stage/len(self.objectives))
            info["reward_components"]["potential"] += correction
            info["reward_components"]["deadline_failure"] = failure
            reward += correction + failure
            self.return_total += correction + failure
            info["terminated"], info["truncated"] = True, False
            info["terminal_reason"] = "finite_task_deadline_failure"
        elif done:
            info["terminal_reason"] = "safe_wait_completed" if self.no_goals else "all_stages_completed"
        else:
            info["terminal_reason"] = None
        if done:
            episode = info["episode"]
            episode.update({"return": self.return_total,
                "terminal_reason": info["terminal_reason"],
                "idle_healthy_steps": self.idle_healthy_steps,
                "recovery_opportunity_steps": self.recovery_opportunity_steps,
                "action_counts": self.action_counts.tolist()})
            episode["parameters"]["scenario_id"] = UC_SUPPORT_V2_SCENARIO_ID
            episode["parameters"]["navigation_slot_permutation"] = self.slot_permutation.tolist()
        info["stage_progress"] = self.stage-previous_stage
        return obs, float(reward), done, info

    def state_dict(self):
        state = super().state_dict()
        state["scenario_id"] = UC_SUPPORT_V2_SCENARIO_ID
        return state

    def load_state_dict(self, snapshot):
        if snapshot["scenario_id"] != UC_SUPPORT_V2_SCENARIO_ID:
            raise ValueError("Environment v2 snapshot identity mismatch")
        adjusted = copy.deepcopy(snapshot)
        # The parent implementation checks its own V1 schema, then restores the
        # complete subclass state, including slot permutation and diagnostics.
        from rl_bt.uc_support_env import UC_SUPPORT_SCENARIO_ID as V1_ID
        adjusted["scenario_id"] = V1_ID
        super().load_state_dict(adjusted)


uc_support_v2_bt_baseline = uc_support_bt_baseline
