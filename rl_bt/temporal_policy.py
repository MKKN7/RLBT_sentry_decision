"""Small observation-only task policies with optional recurrent memory.

This is a separate policy for the existing simplified alphaRM task schema.
It does not control navigation velocity, aiming, or firing, and is not the
live Nav2 policy.  Actor and critic have entirely separate parameters so value
regression cannot change deployed action logits.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from rl_bt.alpharm_task_env import CANDIDATE_DIM, STATE_DIM


class _ObservationContext(nn.Module):
    """Encode legal candidates as a permutation-invariant set."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.candidate_encoder = nn.Sequential(
            nn.Linear(CANDIDATE_DIM, 32), nn.Tanh())

    def forward(self, state, candidates, mask):
        encoded_state = self.state_encoder(state)
        encoded_candidates = self.candidate_encoder(candidates)
        available = mask.unsqueeze(-1)
        mean = encoded_candidates.masked_fill(~available, 0.).sum(dim=1)
        mean = mean / available.sum(dim=1).clamp_min(1)
        maximum = encoded_candidates.masked_fill(
            ~available, torch.finfo(encoded_candidates.dtype).min).max(dim=1).values
        context = torch.cat((encoded_state, mean, maximum), dim=-1)
        return context, encoded_candidates


class _MemoryContext(nn.Module):
    def __init__(self, memory: str, hidden_size: int):
        super().__init__()
        self.memory = memory
        self.hidden_size = hidden_size
        if memory == "gru":
            self.core = nn.GRUCell(hidden_size + 64, hidden_size)
        else:
            self.core = nn.Sequential(nn.Linear(hidden_size + 64, hidden_size), nn.Tanh())

    def forward(self, context, hidden):
        if self.memory == "gru":
            next_hidden = self.core(context, hidden)
            return next_hidden, next_hidden
        return self.core(context), torch.zeros_like(hidden)


class TemporalCandidateActor(nn.Module):
    def __init__(self, memory: str, hidden_size: int):
        super().__init__()
        self.observation_encoder = _ObservationContext(hidden_size)
        self.memory_context = _MemoryContext(memory, hidden_size)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size + 32, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))

    def forward(self, state, candidates, mask, hidden):
        context, encoded_candidates = self.observation_encoder(state, candidates, mask)
        encoded, next_hidden = self.memory_context(context, hidden)
        joined = torch.cat((encoded.unsqueeze(1).expand(-1, candidates.shape[1], -1),
                            encoded_candidates), dim=-1)
        logits = self.scorer(joined).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits, next_hidden


class TemporalValueCritic(nn.Module):
    def __init__(self, memory: str, hidden_size: int):
        super().__init__()
        self.observation_encoder = _ObservationContext(hidden_size)
        self.memory_context = _MemoryContext(memory, hidden_size)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))

    def forward(self, state, candidates, mask, hidden):
        context, _ = self.observation_encoder(state, candidates, mask)
        encoded, next_hidden = self.memory_context(context, hidden)
        return self.value_head(encoded).squeeze(-1), next_hidden


class TemporalActorCritic(nn.Module):
    """Common MLP/GRU interface for fair observation-matched experiments.

    ``hidden`` is a tuple of actor and critic tensors, each ``[batch, H]``.
    Keep it across decisions in an episode.  ``episode_start`` resets selected
    rows before processing the current observation, including during training.
    Gradients remain connected through the returned GRU states: a trainer must
    preserve chronological sequences or explicitly recompute a burn-in prefix.
    """

    def __init__(self, memory: str = "mlp", hidden_size: int = 64):
        super().__init__()
        if memory not in ("mlp", "gru"):
            raise ValueError("memory must be 'mlp' or 'gru'")
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        self.memory = memory
        self.hidden_size = hidden_size
        self.actor = TemporalCandidateActor(memory, hidden_size)
        self.critic = TemporalValueCritic(memory, hidden_size)

    def initial_state(self, batch_size: int, device=None):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        return tuple(torch.zeros(batch_size, self.hidden_size, dtype=parameter.dtype,
                                 device=device) for _ in range(2))

    def actor_parameters(self):
        return self.actor.parameters()

    def critic_parameters(self):
        return self.critic.parameters()

    def _validate_observation(self, state, candidates, mask):
        if state.ndim != 2 or state.shape[-1] != STATE_DIM or state.shape[0] < 1:
            raise ValueError(f"Expected nonempty batched state with {STATE_DIM} features")
        if candidates.ndim != 3 or candidates.shape[-1] != CANDIDATE_DIM or candidates.shape[1] < 1:
            raise ValueError(f"Expected batched candidates with {CANDIDATE_DIM} features")
        if candidates.shape[0] != state.shape[0] or mask.shape != candidates.shape[:2]:
            raise ValueError("State, candidate and mask batch dimensions must agree")
        if mask.dtype != torch.bool:
            raise TypeError("Candidate mask must be boolean")
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("Each observation needs at least one legal fallback candidate")
        parameter = next(self.parameters())
        if state.dtype != parameter.dtype or candidates.dtype != parameter.dtype:
            raise TypeError("Observation floating dtype must match the model")
        if state.device != parameter.device or candidates.device != parameter.device or mask.device != parameter.device:
            raise ValueError("Observation and model devices must agree")
        if not bool(torch.isfinite(state).all()) or not bool(torch.isfinite(candidates).all()):
            raise ValueError("Policy observations must be finite")

    def distribution_value(self, state, candidates, mask, hidden=None, episode_start=None):
        self._validate_observation(state, candidates, mask)
        batch_size = state.shape[0]
        if hidden is None:
            hidden = self.initial_state(batch_size, device=state.device)
        if not isinstance(hidden, (tuple, list)) or len(hidden) != 2:
            raise ValueError("hidden must contain separate actor and critic tensors")
        for item in hidden:
            if not isinstance(item, torch.Tensor) or item.shape != (batch_size, self.hidden_size):
                raise ValueError("Hidden state shape must be [batch, hidden_size]")
            if item.device != state.device or item.dtype != state.dtype:
                raise ValueError("Hidden state dtype and device must match observations")
            if not bool(torch.isfinite(item).all()):
                raise ValueError("Hidden state must be finite")
        if episode_start is not None:
            if not isinstance(episode_start, torch.Tensor) or episode_start.shape != (batch_size,):
                raise ValueError("episode_start must be a batched boolean tensor")
            if episode_start.dtype != torch.bool:
                raise TypeError("episode_start must be boolean")
            if episode_start.device != state.device:
                raise ValueError("episode_start device must match observations")
            # masked_fill also prevents gradients crossing an episode boundary.
            hidden = tuple(item.masked_fill(episode_start.unsqueeze(-1), 0.) for item in hidden)
        logits, actor_hidden = self.actor(state, candidates, mask, hidden[0])
        value, critic_hidden = self.critic(state, candidates, mask, hidden[1])
        return Categorical(logits=logits), value, (actor_hidden, critic_hidden)
