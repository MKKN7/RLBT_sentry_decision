import numpy as np
import pytest
import torch

from rl_bt.contracts import IntentGate, TaskIntent
from rl_bt.uc_rules import CombatClock, StanceClock, TunnelSequence
from rl_bt.policy import CandidateActor
from scripts.train_phase1 import gae


def test_rejected_intent_cannot_be_resurrected_and_expiry_is_enforced():
    gate = IntentGate()
    stale = TaskIntent(1, 3, 8, 10.)
    assert gate.validate(stale, version=9, now=10., mask=[True, True]) == "candidate_version_mismatch"
    assert gate.validate(stale, version=8, now=10., mask=[True, True]) == "stale_sequence"
    assert gate.validate(TaskIntent(1, 4, 9, 10.), version=9, now=12.01,
                         mask=[True, True]) == "expired_or_future_intent"
    assert gate.validate(TaskIntent(1, 5, 9, 12.), version=9, now=12.,
                         mask=[True, False]) == "unreachable_or_illegal"


def test_out_of_combat_resets_on_actual_shot_damage_and_death():
    clock = CombatClock.from_facts()
    assert clock.update(0)
    assert not clock.update(1, fired=True)
    assert not clock.update(6.999)
    assert clock.update(7)
    assert not clock.update(7.1, damage_taken=True)
    assert not clock.update(13, alive=False)
    assert not clock.update(13.1, alive=True)
    assert not clock.update(19.099)
    assert clock.update(19.1)


def test_stance_cooldown_per_stance_budgets_and_exact_degradation_boundary():
    clock = StanceClock.from_facts()
    assert clock.switch(1, 0)
    clock.advance(5)
    assert not clock.switch(2, 4.999)
    assert clock.switch(2, 5)
    clock.advance(5)
    assert clock.switch(3, 10)
    clock.advance(15)
    assert clock.cumulative_s[0] == 15
    assert clock.remaining_enhanced_s(0) == 0
    assert clock.remaining_enhanced_s(1) == 15
    assert not clock.mask(25)[3]
    with pytest.raises(ValueError):
        clock.advance(.01)
    assert clock.switch(0, 25)
    clock.advance(165)
    assert not clock.degraded(0)
    clock.advance(.001)
    assert clock.degraded(0)


def test_tunnel_sequence_both_directions_and_foreign_card_invalidates():
    clock = TunnelSequence.from_facts("a")
    assert clock.observe("a", "end_a", 0) is None
    assert clock.observe("a", "inside", 1) is None
    assert clock.observe("a", "end_b", 3) == "underpass_sequence_completed_not_buff_ack"
    assert clock.observe("a", "end_b", 4) is None
    assert clock.observe("a", "inside", 5) is None
    assert clock.observe("a", "end_a", 7) == "underpass_sequence_completed_not_buff_ack"
    assert clock.observe("a", "end_a", 8) is None
    assert clock.observe("different", "inside", 9) is None
    assert clock.observe("a", "end_b", 10) is None
    assert clock.observe("a", "inside", 11) is None
    assert clock.observe("a", "end_a", 13.001) is None


def test_terminal_reward_does_not_leak_and_rollout_cut_bootstraps():
    rewards = np.asarray([[1.], [100.]])
    zero = np.zeros_like(rewards)
    adv, ret = gae(rewards, zero, zero, np.asarray([[True], [True]]), np.ones_like(rewards))
    np.testing.assert_equal(adv[:, 0], [1., 100.])
    adv, _ = gae(np.asarray([[1.]]), np.asarray([[2.]]), np.asarray([[5.]]),
                 np.asarray([[False]]), np.asarray([[.2]]), gamma_per_second=.9)
    np.testing.assert_allclose(adv, [[1 + .9**.2 * 5 - 2]])


def test_candidate_permutation_equivariance_and_illegal_choice_mask():
    torch.manual_seed(1)
    actor = CandidateActor()
    state = torch.randn(2, 148)
    candidates = torch.randn(2, 16, 16)
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[:, 5] = False
    first, _, _ = actor.encode(state, candidates, mask)
    order = torch.randperm(16)
    second, _, _ = actor.encode(state, candidates[:, order], mask[:, order])
    torch.testing.assert_close(first[:, order], second)
    assert (first[:, 5] < -1e8).all()


@pytest.fixture(scope="module")
def runtime():
    from rl_bt.alpharm_task_env import AlphaRuntime
    return AlphaRuntime()


def test_environment_replay_observation_separation_and_dead_hold(runtime):
    from rl_bt.alpharm_task_env import AlphaTaskEnv
    a = AlphaTaskEnv(runtime, 42, randomization=False, opponent_style="capture", side=0)
    b = AlphaTaskEnv(runtime, 42, randomization=False, opponent_style="capture", side=0)
    for _ in range(4):
        oa, ra, da, ia = a.step(1)
        ob, rb, db, ib = b.step(1)
        assert ra == rb and da == db
        for key in oa:
            np.testing.assert_array_equal(oa[key], ob[key])
        assert ia["duration_s"] == .5
    before = a.observe()
    # Raw hidden enemy position must not bypass the cached observation channel.
    a.state[runtime.index["pos"]][3] += [1, 1]
    after = a.observe()
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    a.history[-1][0][a.learner, 11] = 0
    dead_obs = a.observe()
    assert dead_obs["mask"].tolist() == [True] + [False] * 15
    assert runtime.navigation.clearance(runtime.navigation.goals.reshape(-1, 2)).min() > .30
