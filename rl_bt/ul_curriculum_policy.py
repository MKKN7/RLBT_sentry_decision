"""Small candidate policy for the independent UL ground-task curriculum.

The actor selects an already masked task candidate.  It does not command
velocity, firing or aiming.  The state critic is needed only during training.
This schema intentionally cannot load the alphaRM task policy unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

STATE_DIM = 24
CANDIDATE_DIM = 12
CANDIDATE_COUNT = 16
SCENARIO_ID = "ul2027_ground_task_curriculum_v4"


class ULCurriculumActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.candidate_encoder = nn.Sequential(nn.Linear(CANDIDATE_DIM, 32), nn.Tanh())
        self.scorer = nn.Sequential(nn.Linear(96, 64), nn.Tanh(), nn.Linear(64, 1))

    def encode(self, state, candidates, mask):
        if state.ndim != 2 or state.shape[-1] != STATE_DIM:
            raise ValueError(f"Expected batched state with {STATE_DIM} features")
        if candidates.ndim != 3 or candidates.shape[-1] != CANDIDATE_DIM:
            raise ValueError(f"Expected batched candidates with {CANDIDATE_DIM} features")
        if candidates.shape[0] != state.shape[0] or mask.shape != candidates.shape[:2]:
            raise ValueError("State, candidate and mask batch dimensions must agree")
        if mask.dtype != torch.bool:
            raise TypeError("Candidate mask must be boolean")
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("Each observation needs at least one legal fallback candidate")
        encoded = self.state_encoder(state)
        candidate = self.candidate_encoder(candidates)
        joined = torch.cat((encoded[:, None, :].expand(-1, candidates.shape[1], -1),
                            candidate), dim=-1)
        logits = self.scorer(joined).squeeze(-1)
        # The finite minimum gives masked candidates exactly zero softmax mass.
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits, encoded

    def forward(self, state, candidates, mask):
        return self.encode(state, candidates, mask)[0]


class ULCurriculumActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = ULCurriculumActor()
        self.critic = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def distribution_value(self, state, candidates, mask):
        logits, encoded = self.actor.encode(state, candidates, mask)
        return Categorical(logits=logits), self.critic(encoded).squeeze(-1)


def tensor_observations(observations, device="cpu"):
    if not observations:
        raise ValueError("At least one observation is required")
    arrays = {key: np.stack([observation[key] for observation in observations])
              for key in ("state", "candidates", "mask")}
    if arrays["state"].shape[1:] != (STATE_DIM,):
        raise ValueError("UL curriculum state shape mismatch")
    if arrays["candidates"].shape[1:] != (CANDIDATE_COUNT, CANDIDATE_DIM):
        raise ValueError("UL curriculum candidate shape mismatch")
    if arrays["mask"].shape[1:] != (CANDIDATE_COUNT,) or arrays["mask"].dtype != np.bool_:
        raise ValueError("UL curriculum boolean candidate mask shape mismatch")
    if not np.isfinite(arrays["state"]).all() or not np.isfinite(arrays["candidates"]).all():
        raise ValueError("Policy observations must be finite")
    return {"state": torch.as_tensor(arrays["state"], dtype=torch.float32, device=device),
            "candidates": torch.as_tensor(arrays["candidates"], dtype=torch.float32, device=device),
            "mask": torch.as_tensor(arrays["mask"], dtype=torch.bool, device=device)}


@torch.no_grad()
def choose(model, observation):
    tensors = tensor_observations([observation], next(model.parameters()).device)
    logits = model.actor(**tensors)
    return int(logits.argmax(dim=-1).item())
