"""Observable, stateful BT reference for the simplified alphaRM task wrapper.

This is an auditable Python decision policy, not the team's Combat XML tree.
It receives exactly the tensors used by the learned policy.  Internal memory
contains only retreat hysteresis, a task commitment and elapsed HUD time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BTMemoryConfig:
    retreat_enter_hp: float = .30
    retreat_exit_hp: float = .80
    minimum_commitment_s: float = 4.
    episode_time_s: float = 180.
    arrival_distance_m: float = .25
    path_cost_weight: float = .30
    threat_weight: float = .30
    center_priority: float = 1.40
    cover_priority: float = .15

    def __post_init__(self):
        if not 0 <= self.retreat_enter_hp < self.retreat_exit_hp <= 1:
            raise ValueError("Retreat thresholds must satisfy 0 <= enter < exit <= 1")
        if min(self.minimum_commitment_s, self.episode_time_s, self.arrival_distance_m) <= 0:
            raise ValueError("Commitment, episode duration and arrival radius must be positive")


class BTMemoryPolicy:
    """Hysteretic retreat plus committed, path-aware center-control decisions.

    Clock uses the actor-visible normalized remaining time at state[125].
    Active task/age come from the wrapper's public context at state[140:143].
    Candidate roles, path cost and visible-target threat use existing features.
    No future observation, simulator position or privileged opponent state is read.
    """
    def __init__(self, config=None):
        self.config = config or BTMemoryConfig()
        self.reset()

    def reset(self):
        self.retreating = False
        self.committed_candidate = None
        self.commit_until_s = 0.
        self.clock_s = 0.
        self.decisions = 0
        self.last_reason = "reset"

    def state_dict(self):
        return {"config": asdict(self.config), "retreating": self.retreating,
                "committed_candidate": self.committed_candidate,
                "commit_until_s": self.commit_until_s, "clock_s": self.clock_s,
                "decisions": self.decisions, "last_reason": self.last_reason,
                "privileged_inputs": False}

    @staticmethod
    def _validate(observation):
        state = np.asarray(observation["state"])
        candidates = np.asarray(observation["candidates"])
        mask = np.asarray(observation["mask"])
        if state.shape != (148,) or candidates.shape != (16, 16) or mask.shape != (16,):
            raise ValueError("Expected the alphaRM public task observation schema")
        if mask.dtype != np.bool_ or not mask.any() or not np.isfinite(state).all() or \
                not np.isfinite(candidates).all():
            raise ValueError("Observation and nonempty boolean mask must be finite")
        return state, candidates, mask

    def _commit(self, candidate, reason):
        if candidate != self.committed_candidate:
            self.committed_candidate = int(candidate)
            self.commit_until_s = self.clock_s + self.config.minimum_commitment_s
        self.last_reason = reason
        return int(candidate)

    def act(self, observation):
        state, candidates, mask = self._validate(observation)
        self.clock_s = max(self.clock_s, (1. - float(np.clip(state[125], 0., 1.)))
                           * self.config.episode_time_s)
        self.decisions += 1
        active = int(np.clip(np.rint(state[140] * 15), 0, 15))
        if state[11] < .5:
            # The executor's continue/hold remains legal while dead.  Do not
            # carry an earlier attack commitment through a respawn.
            self.committed_candidate = None
            self.commit_until_s = self.clock_s
            self.last_reason = "dead_hold"
            return int(np.flatnonzero(mask)[0])
        hp = float(state[7])
        if hp < self.config.retreat_enter_hp:
            self.retreating = True
        elif hp >= self.config.retreat_exit_hp:
            self.retreating = False
        if self.retreating:
            if mask[2]:
                return self._commit(2, "retreat_hysteresis")
            # Never keep a now-illegal destination solely due to memory.
            legal = np.flatnonzero(mask)
            safest = min(legal, key=lambda i: (candidates[i, 11], candidates[i, 3]))
            return self._commit(safest, "retreat_home_unavailable")
        committed = self.committed_candidate
        overridden = state[144] > .5
        if committed not in (None, 0) and mask[committed] and not overridden:
            en_route = candidates[committed, 2] * 12 > self.config.arrival_distance_m
            if self.clock_s < self.commit_until_s and en_route:
                # Reissue during command delay; thereafter continue the same
                # active task.  Both choices have the same shared executor.
                self.last_reason = "committed_task"
                return 0 if active == committed and mask[0] else int(committed)
        legal = np.flatnonzero(mask)
        destinations = legal[legal != 0]
        if not len(destinations):
            self.last_reason = "only_hold_legal"
            return int(legal[0])
        # Same objective for both sides: retain center control, while ETA/path
        # and observed threat disambiguate the three center-area candidates.
        score = (self.config.center_priority * candidates[:, 5]
                 + self.config.cover_priority * candidates[:, 7]
                 - self.config.path_cost_weight * candidates[:, 3]
                 - self.config.threat_weight * candidates[:, 11])
        # Home is an emergency resource task, not an attack objective.
        score -= .5 * candidates[:, 6]
        selected = int(destinations[np.argmax(score[destinations])])
        if active == selected and candidates[selected, 2] * 12 <= self.config.arrival_distance_m \
                and mask[0]:
            self.last_reason = "hold_reached_objective"
            return 0
        return self._commit(selected, "center_path_threat_score")
