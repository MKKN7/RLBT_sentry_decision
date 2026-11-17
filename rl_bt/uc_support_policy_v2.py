"""Independent actor/critic encoders for the versioned UC local task course."""
import torch
from torch import nn
from torch.distributions import Categorical

from rl_bt.ul_curriculum_policy import ULCurriculumActor


class UCSupportV2ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = ULCurriculumActor()
        # V1's value loss also trained actor.state_encoder. This separate
        # encoder makes value regression unable to change deployment logits.
        self.critic_state_encoder = nn.Sequential(nn.Linear(24, 64), nn.Tanh(),
                                                   nn.Linear(64, 64), nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(64+12+12+1, 64), nn.Tanh(), nn.Linear(64, 1))

    def distribution_value(self, state, candidates, mask):
        logits = self.actor(state, candidates, mask)
        encoded = self.critic_state_encoder(state)
        available = mask[..., None]
        mean = (candidates*available).sum(1)/available.sum(1).clamp(min=1)
        maximum = candidates.masked_fill(~available, torch.finfo(candidates.dtype).min).max(1).values
        count = mask.float().mean(1, keepdim=True)
        value = self.critic(torch.cat((encoded, mean, maximum, count), -1)).squeeze(-1)
        return Categorical(logits=logits), value

    def critic_parameters(self):
        return list(self.critic_state_encoder.parameters()) + list(self.critic.parameters())
