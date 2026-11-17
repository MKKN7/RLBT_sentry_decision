import numpy as np
import pytest
import torch

from rl_bt.uc_support_env import GAMMA, MAX_STEPS, UCSupportEnv
from rl_bt.uc_support_env_v2 import UCSupportEnvV2, uc_support_v2_bt_baseline
from rl_bt.uc_support_policy_v2 import UCSupportV2ActorCritic


def navigation_env(seed=1000):
    env = UCSupportEnvV2(seed=seed, randomization=False)
    assert not env.no_goals
    return env


def test_deadline_is_failure_terminal_with_zero_absorbing_potential():
    env = navigation_env()
    old_phi = -.5*env._remaining()
    for _ in range(MAX_STEPS-1):
        _, _, done, _ = env.step(15)
        assert not done
    obs, reward, done, info = env.step(15)
    assert done and info['terminated'] and not info['truncated']
    assert info['terminal_reason'] == 'finite_task_deadline_failure'
    assert info['reward_components']['potential'] == pytest.approx(-old_phi)
    assert info['reward_components']['deadline_failure'] == -3.
    assert info['episode']['return'] == pytest.approx(env.return_total)
    assert info['episode']['score_100'] == 10.
    # Healthy waiting remains legal, so a learned failure cannot be hidden.
    assert obs['mask'][15] and obs['mask'][:13].any()


def test_candidate_slots_are_permuted_without_changing_task_or_noise_schedule():
    seen = set()
    for seed in range(1000, 1008):
        v1 = UCSupportEnv(seed=seed)
        v2 = UCSupportEnvV2(seed=seed)
        np.testing.assert_array_equal(v1.objectives, v2.objectives)
        np.testing.assert_array_equal(v1.pose_noise, v2.pose_noise)
        np.testing.assert_array_equal(v1.support_received, v2.support_received)
        if not v2.no_goals:
            seen.add(int(np.flatnonzero(v2.obs['candidates'][:, 4])[0]))
    assert len(seen) >= 3


def test_value_gradient_cannot_change_actor():
    model = UCSupportV2ActorCritic()
    state = torch.randn(8,24)
    cand = torch.randn(8,16,12)
    mask = torch.ones(8,16,dtype=torch.bool)
    _, value = model.distribution_value(state,cand,mask)
    value.square().mean().backward()
    assert all(p.grad is None for p in model.actor.parameters())
    assert any(p.grad is not None for p in model.critic_parameters())


def test_v2_checkpoint_replays_candidate_slots_and_diagnostics():
    env = navigation_env()
    obs = env.obs
    for _ in range(5):
        obs,_,_,_=env.step(uc_support_v2_bt_baseline(obs))
    other = navigation_env()
    other.load_state_dict(env.state_dict())
    for _ in range(10):
        action=uc_support_v2_bt_baseline(obs)
        a,b=env.step(action),other.step(action)
        for key in a[0]:
            np.testing.assert_array_equal(a[0][key], b[0][key])
        assert a[1:]==b[1:]
        obs=a[0]
        if a[2]: break
