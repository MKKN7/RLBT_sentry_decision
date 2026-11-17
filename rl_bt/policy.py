"""Shared small MLPs for candidate scoring; critic is training-only."""
import torch
from torch import nn
from torch.distributions import Categorical

from rl_bt.alpharm_task_env import STATE_DIM, CANDIDATE_DIM


class CandidateActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(STATE_DIM, 64), nn.Tanh(),
                                           nn.Linear(64, 64), nn.Tanh())
        self.candidate_encoder = nn.Sequential(nn.Linear(CANDIDATE_DIM, 32), nn.Tanh())
        self.scorer = nn.Sequential(nn.Linear(96, 64), nn.Tanh(), nn.Linear(64, 1))
        self.stance_head = nn.Sequential(nn.Linear(96, 32), nn.Tanh(), nn.Linear(32, 6))

    def encode(self, state, candidates, mask):
        s = self.state_encoder(state)
        c = self.candidate_encoder(candidates)
        joined = torch.cat((s[:, None, :].expand(-1, candidates.shape[1], -1), c), dim=-1)
        logits = self.scorer(joined).squeeze(-1).masked_fill(~mask, -1e9)
        return logits, s, joined

    def forward(self, state, candidates, mask, stance_mask):
        logits, _, joined = self.encode(state, candidates, mask)
        selected = logits.argmax(dim=-1)
        chosen = joined.gather(1, selected[:, None, None].expand(-1, 1, 96)).squeeze(1)
        stance_logits = self.stance_head(chosen).masked_fill(~stance_mask, -1e9)
        return logits, stance_logits


class TaskActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = CandidateActor()
        self.critic = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def distribution_value(self, state, candidates, mask):
        logits, encoded, _ = self.actor.encode(state, candidates, mask)
        return Categorical(logits=logits), self.critic(encoded).squeeze(-1)


def tensor_observations(observations, device="cpu"):
    import numpy as np
    return {k: torch.as_tensor(np.stack([o[k] for o in observations]), device=device)
            for k in ("state", "candidates", "mask")}


@torch.no_grad()
def choose(model, observation):
    tensors = tensor_observations([observation], next(model.parameters()).device)
    distribution, _ = model.distribution_value(**tensors)
    return int(distribution.logits.argmax(dim=-1).item())
