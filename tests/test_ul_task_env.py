import numpy as np
import pytest

from rl_bt.ul_task_env import ULTaskEnv, STATE_DIM, CANDIDATE_DIM, ul_bt_baseline


class OpenGround:
    """Explicit test fixture, not the real map."""
    radius_m = .411

    def is_free(self, xy, team):
        x, y = xy
        if not (.45 <= x <= 11.55 and .45 <= y <= 7.55):
            return False
        return not ((team == 0 and x >= 9.08 and y <= 2.42)
                    or (team == 1 and x <= 2.92 and y >= 5.58))

    def plan(self, start, end, team):
        return np.array([start, end], float) if self.is_free(start, team) and self.is_free(end, team) else None

    def goal_candidates(self, team):
        points = [[1.5, 7.], [6., 4.], [6., 1.5], [5.5, 6.], [7., 2.],
                  [4., 1.], [6., 6.5], [8., 5.5], [2., 2.], [10., 6.],
                  [3.5, 4.8], [8.5, 3.2], [3.5, 7.46], [8.5, .54]]
        if team:
            points = (np.array([12., 8.])-points).tolist()
        kinds = ["resupply", "capture"]+["position"]*8+["highland"]*2+["tunnel"]*2
        return [{"id": i, "name": f"fixture_{i}", "xy": xy, "task_type": kinds[i],
                 "static_legal": bool(i < 10 and self.is_free(xy, team))}
                for i, xy in enumerate(points)]


def make(seed=1, randomization=False):
    return ULTaskEnv(seed=seed, randomization=randomization, nav=OpenGround())


def test_actor_schema_excludes_enemy_truth_and_exact_capture_progress():
    env = make()
    obs = env.reset(seed=2)
    assert obs["state"].shape == (STATE_DIM,)
    assert obs["candidates"].shape == (16, CANDIDATE_DIM)
    assert not obs["mask"][10:14].any() and obs["mask"][15]
    obs_before = {k: v.copy() for k, v in obs.items()}
    # This hidden rule state has no direct feature in the actor projection.
    env.center.progress = [2., 0.]
    after = env._observe()
    for key in obs:
        np.testing.assert_array_equal(after[key], obs_before[key])


def test_BT_completes_both_sides_and_all_curriculum_tasks_without_spending_ammo():
    env = make()
    seen = set()
    for seed in range(1, 30):
        obs = env.reset(seed=seed)
        seen.add((env.objective, env.side))
        for _ in range(121):
            obs, reward, done, info = env.step(ul_bt_baseline(obs))
            assert np.isfinite(reward)
            assert env.ammo.remaining == 750
            if done:
                assert info["episode"]["success"], info
                with pytest.raises(RuntimeError):
                    env.step(15)
                break
        else:
            pytest.fail("Course failed to terminate")
    assert len(seen) == 6


def test_masked_highland_proposal_is_vetoed_and_cannot_teleport_robot():
    env = make()
    before = env.position.copy()
    _, _, _, info = env.step(10)
    assert info["BT_veto"]
    np.testing.assert_equal(env.position, before)


def test_supply_healing_requires_own_supply_and_never_safe_cover_alone():
    env = make()
    env.hp = 100.
    env.position[:] = [6., 1.5]
    env.cached = env._observe()
    env.step(15)
    env.step(15)
    assert env.hp == 100.
    env.position[:] = env.supply
    env._sample_pose(force=True)
    env._deliver_public()
    env.cached = env._observe()
    env.step(15)
    env.step(15)
    assert env.hp == 200.  # official UL 25% of 400 HP / second


def test_incidental_center_crossing_does_not_grant_forced_capture_skill():
    env = make()
    env.require_capture = True
    env.require_heal = False
    env.position[:] = [6., 4.]
    env.goal_id = env.supply_id
    env._sample_pose(force=True); env._deliver_public()
    env.cached = env._observe()
    for _ in range(12):
        env.step(15)
    assert env.center.owner is None
    assert env.center.progress == [0., 0.]


@pytest.mark.parametrize("interrupt", [0, 10, 14, 15])
def test_atomic_capture_cannot_be_abandoned_by_RL_hold_masked_or_new_goal(interrupt):
    env = make()
    env.require_capture, env.require_heal = True, False
    env.position[:] = [6., 4.]
    env._sample_pose(force=True); env._deliver_public()
    env.cached = env._observe()
    env.step(env.center_id)
    assert env.capture_skill_active and 0 < env.center.progress[env.side] < 5.
    position = env.position.copy()
    assert not env.cached["mask"][:14].any()
    for _ in range(12):
        _, _, done, info = env.step(interrupt)
        np.testing.assert_equal(env.position, position)
        if done:
            assert info["episode"]["success"]
            assert not env.capture_skill_active
            break
    else:
        pytest.fail("Atomic capture failed to finish")


def test_independent_packet_validity_and_pose_latency_are_observable():
    env = make()
    env.parameters["pose_delay_s"] = .15
    env.step(15)
    assert env.cached["state"][5] == pytest.approx(.5)
    env.time_s += 2.1
    obs = env._observe()
    assert not obs["state"][[6, 10, 16, 22, 23]].any()


def test_referee_delay_retains_old_self_sample_until_delivery():
    env = make()
    env.parameters["packet_delay_s"] = .2
    env.hp = 123.
    env.step(15)
    packet = env.cache.snapshot(now_s=env.time_s, max_sample_age_s={0x0201: 1.})[0x0201]
    assert packet.fields["own_hp"] != 123
    env.step(15)
    packet = env.cache.snapshot(now_s=env.time_s, max_sample_age_s={0x0201: 1.})[0x0201]
    assert packet.fields["own_hp"] == 123


def test_seed_replays_observations_actions_and_randomization():
    a, b = make(73, True), make(73, True)
    assert a.parameters == b.parameters
    for _ in range(5):
        oa, ra, da, ia = a.step(ul_bt_baseline(a.cached))
        ob, rb, db, ib = b.step(ul_bt_baseline(b.cached))
        for key in oa:
            np.testing.assert_array_equal(oa[key], ob[key])
        assert (ra, da, ia) == (rb, db, ib)
