import copy

import numpy as np
import pytest

from rl_bt.reward_components import (
    REWARD_PROFILES,
    apply_reward_profile,
    per_agent_reward_components,
    reward_config,
    team_reward_components,
)


@pytest.fixture(scope="module")
def runtime():
    from rl_bt.alpharm_task_env import AlphaRuntime
    return AlphaRuntime()


def fixture_state(runtime):
    from scripts.probe_alpharm_onnx import decode
    return {item["name"]: decode(item) for item in runtime.fixture["state"]}


def test_remove_only_selected_home_and_fire_terms(runtime):
    before = fixture_state(runtime)
    after = copy.deepcopy(before)
    cfg = reward_config()
    after["pos"][2] = np.mean(cfg.red_spawn, axis=0)
    after["hp"][2] = .25 * cfg.hp_max
    after["last_fired"][2] = True
    after["reward_dense"][:] = .125
    parts = team_reward_components(before, after, 0)
    assert parts["low_hp_home"] == pytest.approx(cfg.k_low_hp_home_reward / 3)
    assert parts["fire_heat"] == pytest.approx(-cfg.k_fire_heat_penalty / 3)
    baseline = float(np.mean(after["reward_dense"][:3] + after["reward_terminal"][:3]))
    assert apply_reward_profile(baseline, parts) == baseline
    assert apply_reward_profile(baseline, parts, "no_home") == pytest.approx(
        baseline - cfg.k_low_hp_home_reward / 3)
    assert apply_reward_profile(baseline, parts, "no_fire") == pytest.approx(
        baseline + cfg.k_fire_heat_penalty / 3)
    assert apply_reward_profile(baseline, parts, "no_home_no_fire") == pytest.approx(
        baseline - cfg.k_low_hp_home_reward / 3 + cfg.k_fire_heat_penalty / 3)
    with pytest.raises(ValueError, match="Unknown reward profile"):
        apply_reward_profile(baseline, parts, "unsupported")


def test_dead_agent_has_only_death_penalty_and_terminal(runtime):
    before = fixture_state(runtime)
    after = copy.deepcopy(before)
    after["alive"][2] = False
    after["just_died"][2] = True
    after["last_fired"][2] = True  # even a stale marker must not earn a correction
    after["reward_dense"][2] = -.2
    after["reward_terminal"][2] = -6
    parts = per_agent_reward_components(before, after)
    assert parts["low_hp_home"][2] == 0
    assert parts["fire_heat"][2] == 0
    assert parts["death"][2] == -.2
    assert parts["terminal"][2] == -6
    assert parts["contact_and_precision_residual"][2] == pytest.approx(0, abs=1e-8)


def test_own_raw_damage_boost_does_not_change_terminal_or_pay_dead_shooter(runtime):
    from rl_bt.rewards import reward_tick
    before = fixture_state(runtime)
    after = copy.deepcopy(before)
    after["last_hit_damage"][2] = 325
    after["reward_dense"][2] = .05
    after["reward_terminal"][:3] = 6
    base = reward_tick(before, after, team=0, learner=2, config=runtime.fixture["config"])
    boosted = reward_tick(before, after, team=0, learner=2,
                           config=runtime.fixture["config"], profile="own_damage")
    assert boosted["selected_total"] - base["selected_total"] == pytest.approx(.05 * 2/3)
    assert boosted["upstream_terminal"] == base["upstream_terminal"] == 6
    after["alive"][2] = False
    after["just_died"][2] = True
    dead = reward_tick(before, after, team=0, learner=2,
                       config=runtime.fixture["config"], profile="own_damage")
    assert dead["own_hit_raw"] == dead["profile_adjustment"] == 0


def test_respawn_does_not_receive_same_frame_team_kill_credit(runtime):
    before = fixture_state(runtime)
    before["alive"][2] = False
    after = copy.deepcopy(before)
    after["alive"][2] = True
    after["last_kill_score"][0] = reward_config().kill_score
    parts = per_agent_reward_components(before, after)
    np.testing.assert_allclose(parts["kill"][:3], [.1, .1, 0], atol=1e-8)


def test_real_onnx_substeps_keep_baseline_and_reconcile_dense_rewards(runtime):
    """Use real combat states so the residual cannot hide positive missing terms."""
    from scripts.probe_alpharm_onnx import action_masks, decode
    state = [decode(item) for item in runtime.fixture["state"]]
    obs, hidden = decode(runtime.fixture["obs"]), decode(runtime.fixture["hidden"])
    index = runtime.index
    seen_fire = seen_contact = False
    checked = 0
    for _ in range(400):
        before = {name: state[i] for name, i in index.items()}
        tm, fm = action_masks(obs, before, runtime.fixture["config"])
        raw, hidden = runtime.actor.run(None, {"obs": obs, "hidden": hidden})
        actions = np.column_stack((
            np.tanh(3 * np.tanh(raw[:, :4] / 3)),
            np.argmax(np.where(tm, raw[:, 8:12], -np.inf), axis=1),
            np.argmax(np.where(fm, raw[:, 12:14], -np.inf), axis=1),
        )).astype(np.float32)
        feeds = dict(zip(runtime.inputs[:-1], state))
        feeds[runtime.inputs[-1]] = actions
        output = runtime.env.run(None, feeds)
        state, obs, done = output[:-2], output[-2], bool(output[-1])
        after = {name: state[i] for name, i in index.items()}
        parts = per_agent_reward_components(before, after)
        # Contact is solely a cost. Positive residual above rounding tolerance
        # signals an omitted positive term or a bad reconstruction.
        assert max(parts["contact_and_precision_residual"]) < 2e-6
        np.testing.assert_allclose(sum(parts.values()),
                                   after["reward_dense"] + after["reward_terminal"],
                                   atol=2e-7, rtol=1e-6)
        seen_fire |= bool(np.any(after["last_fired"]))
        seen_contact |= bool(np.any(parts["contact_and_precision_residual"] < -1e-4))
        for team in (0, 1):
            mean_parts = team_reward_components(before, after, team)
            upstream = float(np.mean((after["reward_dense"] + after["reward_terminal"])
                                     [team * 3:team * 3 + 3]))
            assert apply_reward_profile(upstream, mean_parts, "upstream") == upstream
            assert all(np.isfinite(apply_reward_profile(upstream, mean_parts, profile))
                       for profile in REWARD_PROFILES)
        checked += 1
        if done:
            break
    assert checked >= 100
    assert seen_fire and seen_contact


def test_reward_profiles_preserve_dynamics_observations_and_switch_cost(runtime):
    from rl_bt.alpharm_task_env import AlphaTaskEnv
    envs = [AlphaTaskEnv(runtime, 771, randomization=True,
                         opponent_style="capture", side=0, reward_profile=profile)
            for profile in REWARD_PROFILES]
    for action in (1, 2, 1, 3, 1, 0):
        results = [env.step(action) for env in envs]
        baseline = results[0]
        for env, result in zip(envs, results):
            for key in result[0]:
                np.testing.assert_array_equal(result[0][key], baseline[0][key])
            for actual, expected in zip(env.state, envs[0].state):
                np.testing.assert_array_equal(actual, expected)
            parts = result[3]["reward_components"]
            assert parts["selected_total"] - parts["switch_cost"] == pytest.approx(result[1])
            assert parts["upstream_total"] - parts["switch_cost"] == pytest.approx(baseline[1])
            assert result[3]["executed_candidate"] == baseline[3]["executed_candidate"]
            assert result[2] == baseline[2]
