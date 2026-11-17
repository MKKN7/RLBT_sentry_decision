"""Small UC candidate actor and candidate-aware training-only critic."""
import torch
from torch import nn
from torch.distributions import Categorical

from rl_bt.ul_curriculum_policy import ULCurriculumActor


class UCSupportActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = ULCurriculumActor()
        # Raw candidate mean/max and legal fraction make availability visible
        # to V, without increasing the deployment actor or using hidden truth.
        self.critic = nn.Sequential(nn.Linear(64 + 12 + 12 + 1, 64),
                                    nn.Tanh(), nn.Linear(64, 1))

    def distribution_value(self, state, candidates, mask):
        logits, encoded = self.actor.encode(state, candidates, mask)
        available = mask[..., None]
        mean = (candidates*available).sum(1)/available.sum(1).clamp(min=1)
        maximum = candidates.masked_fill(~available, torch.finfo(candidates.dtype).min).max(1).values
        count = mask.float().mean(1, keepdim=True)
        value = self.critic(torch.cat((encoded, mean, maximum, count), -1)).squeeze(-1)
        return Categorical(logits=logits), value
