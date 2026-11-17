"""Transition-level checks for the observed UC support course."""
import copy

import numpy as np
import pytest

from rl_bt.uc_support_env import (DT, GAMMA, MAX_STEPS, UCSupportEnv,
    uc_support_bt_baseline)
from rl_bt.support_costmap import path_support_check


def test_masks_unknown_and_invalid_proposal_cannot_move():
    env = UCSupportEnv()
    index = next(i for i in env.indices if not len(env.pools[i]))
    for seed in range(1000, 1100):
        obs = env.reset(seed)
        if env.sample_index == index:
            break
    assert obs['mask'].tolist() == [False]*15 + [True]
    before = env.pos.copy()
    _, _, _, info = env.step(0)
    assert info['BT_veto'] and info['executed_action'] == 15
    np.testing.assert_array_equal(env.pos, before)


def test_snapshot_replays_actions_noise_faults_and_rewards_exactly():
    env = UCSupportEnv()
    obs = env.reset(1000)
    for _ in range(4):
        obs, _, _, _ = env.step(uc_support_bt_baseline(obs))
    snapshot = env.state_dict()
    twin = UCSupportEnv()
    twin.load_state_dict(snapshot)
    for _ in range(20):
        action = uc_support_bt_baseline(obs)
        a = env.step(action)
        b = twin.step(action)
        for key in a[0]:
            np.testing.assert_array_equal(a[0][key], b[0][key])
        assert a[1:] == b[1:]
        obs = a[0]
        if a[2]:
            break


def test_every_applied_motion_stays_on_observed_continuous_support():
    env = UCSupportEnv()
    rng = np.random.default_rng(441)
    moved = 0
    for seed in range(1000, 1008):
        obs = env.reset(seed)
        for _ in range(MAX_STEPS):
            before = env.pos.copy()
            action = int(rng.choice(np.flatnonzero(obs['mask'])))
            obs, _, done, _ = env.step(action)
            if np.linalg.norm(before-env.pos) > 0:
                moved += 1
                check = path_support_check(env.grids[env.sample_index],
                    [before, env.pos], radius_m=.411)
                assert check['accepted'], check
            if done:
                break
    assert moved > 10


def test_support_staleness_stops_even_an_active_continue():
    env = UCSupportEnv(randomization=False)
    for seed in range(1000, 1100):
        obs = env.reset(seed)
        if not env.no_goals:
            break
    env.scenario = 'support_gap'
    env.fault_begin, env.fault_end = 0., 100.
    env.pending = 100.
    obs, _, _, _ = env.step(0)
    env.pending = 100.
    for _ in range(3):
        obs, _, _, info = env.step(14)
    assert env.active is None and not obs['mask'][:13].any()
    assert info['BT_reason'] == 'support_stale_BT_stop'


def test_wait_is_time_limit_not_success_for_navigation_and_potential_is_fixed():
    env = UCSupportEnv(randomization=False)
    for seed in range(1000, 1100):
        env.reset(seed)
        if not env.no_goals:
            break
    potential = -.5*env._remaining()
    _, _, _, info = env.step(15)
    assert info['reward_components']['potential'] == pytest.approx((GAMMA-1)*potential)
    while not env.done:
        _, _, _, info = env.step(15)
    assert info['truncated'] and not info['terminated']
    assert info['episode']['stage_completed'] == 0
    assert info['episode']['score_100'] == 10.
    with pytest.raises(RuntimeError):
        env.step(15)


def test_source_splits_disjoint_and_baseline_completes_multiple_stages():
    env = UCSupportEnv(randomization=False)
    assert not set(env.splits['sample_indices']['train']) & set(env.splits['sample_indices']['test'])
    obs = env.reset(1000)
    while True:
        obs, _, done, info = env.step(uc_support_bt_baseline(obs))
        if done:
            break
    episode = info['episode']
    assert episode['success'] and episode['stage_completed'] == 3
    assert 0 < episode['score_100'] <= 100
    bad = copy.deepcopy(env.state_dict())
    bad['dataset_sha256'] = 'incorrect'
    with pytest.raises(ValueError):
        env.load_state_dict(bad)
