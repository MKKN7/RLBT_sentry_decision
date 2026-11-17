"""Live learner mechanics use a labeled mock; no mock counts as Nav2 evidence."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.train_live_nav2 import (duration_gae, parse_args, train, verify_live_artifacts,
    load_live_demonstrations, fit_live_bc_actor, evaluate, goal_schedule)
from rl_bt.uc_support_policy_v2 import UCSupportV2ActorCritic
from scripts.train_uc_support_v2 import observation_tensors


def test_duration_gae_bootstraps_truncation_but_not_terminal_and_scales_time():
    advantages, returns = duration_gae([1., 1., 1.], [2., 2., 2.], [3., 3., 3.],
        [True, False, False], [False, True, False], [1., 1., .5], gamma=.9, lam=.8)
    np.testing.assert_allclose(advantages, [-1., 1.43, 1.7], atol=1e-6)
    np.testing.assert_allclose(returns, [1., 3.43, 3.7], atol=1e-6)


def test_duration_gae_carries_only_within_episode():
    advantage, _ = duration_gae([0., 1.], [0., 0.], [0., 0.],
        [False, True], [False, False], [.25, .5], gamma=.81, lam=.64)
    np.testing.assert_allclose(advantage, [.72, 1.])
    with pytest.raises(ValueError, match="durations"):
        duration_gae([0.], [0.], [0.], [False], [False], [-1.])


class MockLiveEnv:
    """Explicit unit-test fixture: not live ROS or a simulator result."""
    closed = []

    def __init__(self, **kwargs):
        self.run_dir = kwargs["run_dir"]
        self.goal = kwargs["goal"]
        self.steps = 0

    def observation(self):
        state = np.zeros(24, np.float32)
        state[0] = self.steps / 4
        candidate = np.zeros((16, 12), np.float32)
        candidate[:, 0] = np.arange(16)/16
        mask = np.zeros(16, bool)
        mask[[0, 15]] = True
        return {"state": state, "candidates": candidate, "mask": mask}

    def reset(self):
        return self.observation()

    def step(self, action):
        self.steps += 1
        done = self.steps == 4
        return self.observation(), float(action == 0) - .1, done, {
            "elapsed_s": .25 + self.steps*.1, "terminated": done, "truncated": False,
            "nav_status": "mock_only", "request_id": self.steps,
            "episode": {"success": True} if done else {}}

    def close(self):
        self.closed.append(self.run_dir)
        return {"mock_only": True}


def test_live_collector_updates_fresh_weights_and_preserves_actual_durations(tmp_path):
    MockLiveEnv.closed = []
    run = tmp_path / "mock_rl"
    args = parse_args(["--run", str(run), "--episodes", "2", "--max-decisions", "8",
                       "--rollout", "4", "--epochs", "1"])
    summary = train(args, env_factory=MockLiveEnv)
    assert summary["decisions"] == 8 and summary["ppo_updates"] == 2
    assert summary["completed_episodes"] == 2
    assert summary["actor_weight_change_l2"] > 0
    assert summary["critic_weight_change_l2"] > 0
    assert summary["all_parameters_finite"]
    assert not summary["real_nav2_in_loop"]
    assert summary["verified_live_rpc_transitions"] == 0
    assert len(MockLiveEnv.closed) == 2
    rows = [json.loads(row) for row in (run / "transitions.jsonl").read_text().splitlines()]
    assert [row["elapsed_s"] for row in rows] == pytest.approx([.35, .45, .55, .65]*2)
    assert all(row["info"]["nav_status"] == "mock_only" for row in rows)
    initial = torch.load(run / "untrained.pt", weights_only=False)
    final = torch.load(run / "final.pt", weights_only=False)
    assert not final["exact_resume_supported"]
    assert initial["scenario_id"] == "uc_original_live_pointlio_nav2_mppi_task_v1"
    assert final["actor_optimizer"]["state"]
    assert final["critic_optimizer"]["state"]
    assert (run / "source/scripts/train_uc_support_v2.py").is_file()


def test_bt_control_is_separate_from_ppo_and_closes_at_decision_budget(tmp_path):
    args = parse_args(["--run", str(tmp_path / "mock_bt"), "--policy", "bt",
        "--episodes", "1", "--max-decisions", "2"])
    summary = train(args, env_factory=MockLiveEnv)
    assert summary["ppo_updates"] == 0
    assert summary["actor_weight_change_l2"] == 0
    assert summary["episodes"][0]["decision_budget_truncated"]
    assert summary["episodes"][0]["truncated"]
    assert not summary["episodes"][0]["success"]


def test_environment_failure_preserves_checkpoint_and_closes_owned_session(tmp_path):
    class BrokenEnv(MockLiveEnv):
        def reset(self):
            raise RuntimeError("mock transport unavailable")
    run = tmp_path / "broken"
    args = parse_args(["--run", str(run)])
    with pytest.raises(RuntimeError, match="artifacts preserved"):
        train(args, env_factory=BrokenEnv)
    assert (run / "final.pt").is_file()
    assert json.loads((run / "summary.json").read_text())["status"] == "error"
    assert json.loads((run / "sessions.json").read_text())[0]["close_report"]["mock_only"]


def test_runtime_proof_requires_observed_original_field_plugin_and_no_gt(tmp_path):
    execution = tmp_path / "sessions/episode_000/episode_000/execution"
    (execution / "fixture").mkdir(parents=True)
    (execution / "nav2").mkdir()
    fixture = {"GT_localization_input": False, "terrain_prior": "observed",
        "field_source": {"prepared_mesh_sha256":
            "93ff0118af176f388c171b239acfdcac72c17cbe0b523350d5d916d3236fe10b"},
        "commands": [[1., 0., 0.], [1.05, .1, 0.]], "estimated_poses": []}
    (execution / "fixture/result.json").write_text(json.dumps(fixture))
    (execution / "nav2/controller_server.log").write_text(
        "Created controller : FollowPath of type nav2_mppi_controller::MPPIController\n")
    proof = verify_live_artifacts(tmp_path)
    assert proof[0]["runtime_verified"]
    assert proof[0]["nonzero_MPPI_command_samples"] == 1
    fixture["GT_localization_input"] = True
    (execution / "fixture/result.json").write_text(json.dumps(fixture))
    assert not verify_live_artifacts(tmp_path)[0]["runtime_verified"]


def test_mock_bt_traces_never_become_live_bc_demonstrations(tmp_path):
    run = tmp_path / "mock_teacher"
    args = parse_args(["--run", str(run), "--policy", "bt", "--episodes", "1",
                       "--max-decisions", "4"])
    train(args, env_factory=MockLiveEnv)
    with pytest.raises(ValueError, match="never mock/proxy"):
        load_live_demonstrations(run)
    # Self-declaring the summary is insufficient: no actual copied fixtures.
    summary = json.loads((run / "summary.json").read_text())
    summary["real_nav2_in_loop"] = True
    (run / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="runtime evidence"):
        load_live_demonstrations(run)


def test_live_bc_is_actor_only_and_reports_unrepresented_intents():
    torch.set_num_threads(2)
    torch.manual_seed(83)
    model = UCSupportV2ActorCritic()
    initial = copy.deepcopy(model.state_dict())
    observations = [MockLiveEnv(run_dir=Path("unused"), goal=(0., .3)).observation()
                    for _ in range(4)]
    records = {key: np.stack([row[key] for row in observations])
               for key in ("state", "candidates", "mask")}
    records["actions"] = np.asarray([0, 15, 0, 15])
    report = fit_live_bc_actor(model, records, updates=5, seed=991)
    assert report["original_label_counts_and_accuracy"]["continue"]["count"] == 0
    assert report["original_label_counts_and_accuracy"]["continue"]["accuracy"] is None
    assert report["original_label_counts_and_accuracy"]["hold"]["count"] == 2
    assert any(not torch.equal(value, model.state_dict()[key]) for key, value in initial.items()
               if key.startswith("actor."))
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in initial.items()
               if not key.startswith("actor."))


def test_goal_schedule_cycles_and_frozen_greedy_evaluation_never_updates(tmp_path, monkeypatch):
    schedule = tmp_path / "goals.json"
    schedule.write_text(json.dumps({"goals": [[0., .3], [0., .4]]}))
    trained = tmp_path / "mock_checkpoint"
    training = parse_args(["--run", str(trained), "--episodes", "1", "--max-decisions", "4"])
    train(training, env_factory=MockLiveEnv)
    checkpoint_path = trained / "initial.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = UCSupportV2ActorCritic()
    model.load_state_dict(checkpoint["model"])
    calls = []

    def forbidden_optimizer_step(*args, **kwargs):
        raise AssertionError("Frozen greedy evaluation must not call optimizer.step")

    def factory(**kwargs):
        calls.append(kwargs["goal"])
        return MockLiveEnv(**kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", forbidden_optimizer_step)
    args = parse_args(["--run", str(tmp_path / "greedy_eval"), "--episodes", "3",
        "--max-decisions", "12", "--goal-schedule", str(schedule)])
    args.checkpoint = checkpoint_path
    result = evaluate(args, env_factory=factory)
    assert result["evaluation_only"] and result["weights_unchanged"]
    assert result["ppo_updates"] == 0
    assert result["actor_weight_change_l2"] == result["critic_weight_change_l2"] == 0
    assert calls == [(0., .3), (0., .4), (0., .3)]
    assert not (args.run / "updates.jsonl").exists()
    for line in (args.run / "transitions.jsonl").read_text().splitlines():
        row = json.loads(line)
        obs = {key: np.asarray(row["observation"][key], dtype=(bool if key == "mask" else np.float32))
               for key in ("state", "candidates", "mask")}
        with torch.no_grad():
            logits = model.actor(**observation_tensors([obs]))
        assert row["proposed_action"] == int(logits.argmax(-1).item())


def test_goal_schedule_and_evaluation_source_contract_reject_unsafe_inputs(tmp_path):
    schedule = tmp_path / "bad_goals.json"
    schedule.write_text(json.dumps([[0., float("nan")]]))
    args = parse_args(["--run", str(tmp_path / "unused"), "--goal-schedule", str(schedule)])
    with pytest.raises(ValueError, match="finite XY"):
        goal_schedule(args)
    from scripts.train_live_nav2 import validate_checkpoint_contract, OBSERVATION_SCHEMA, SCENARIO_ID
    checkpoint = {"checkpoint_schema": 1, "scenario_id": SCENARIO_ID,
                  "metadata": {"observation_schema": OBSERVATION_SCHEMA}}
    with pytest.raises(ValueError, match="source mismatch"):
        validate_checkpoint_contract(checkpoint)
    checkpoint["scenario_id"] = "uc_support_proxy_v2"
    with pytest.raises(ValueError, match="current live navigation scenario"):
        validate_checkpoint_contract(checkpoint)


def test_mock_warmstart_integration_uses_bc_anchor_but_never_counts_as_live(tmp_path, monkeypatch):
    """Inject mock records only in this test; production proof loader is untouched."""
    import scripts.train_live_nav2 as live_trainer
    observations = [MockLiveEnv(run_dir=Path("unused"), goal=(0., .3)).observation()]
    records = {key: np.stack([row[key] for row in observations])
               for key in ("state", "candidates", "mask")}
    records["actions"] = np.asarray([0])
    monkeypatch.setattr(live_trainer, "load_live_demonstrations",
        lambda path: (records, {"mock_injection_for_unit_test_only": True}))
    args = parse_args(["--run", str(tmp_path / "mock_anchor"), "--episodes", "1",
        "--max-decisions", "4", "--rollout", "4", "--epochs", "1",
        "--warmstart-live", str(tmp_path / "injected_mock"), "--bc-updates", "2"])
    result = train(args, env_factory=MockLiveEnv)
    assert result["ppo_updates"] == 1 and not result["real_nav2_in_loop"]
    initial = torch.load(args.run / "initial.pt", weights_only=False)["model"]
    untrained = torch.load(args.run / "untrained.pt", weights_only=False)["model"]
    assert any(not torch.equal(value, initial[key]) for key, value in untrained.items()
               if key.startswith("actor."))
    assert all(torch.equal(value, initial[key]) for key, value in untrained.items()
               if not key.startswith("actor."))
    update = json.loads((args.run / "updates.jsonl").read_text())
    assert update["auxiliary_CE"] > 0


def test_declared_sensor_seeds_forward_to_each_attempt_without_claiming_physics_pairing(tmp_path):
    seen = []
    def factory(**kwargs):
        seen.append(kwargs["simulation_seed"])
        return MockLiveEnv(**kwargs)
    args = parse_args(["--run", str(tmp_path / "seeded"), "--episodes", "2",
        "--max-decisions", "8", "--simulation-seed-base", "31000"])
    result = train(args, env_factory=factory)
    assert seen == [31000, 31001]
    config = json.loads((args.run / "config.json").read_text())
    assert config["simulation_seed_schedule"] == seen
    assert config["physics_identically_seeded_or_paired"] is False
    assert result["actor_learning_counts"]["on_policy_actor_sample_count"] == 8
    assert result["actor_learning_counts"]["critic_optimizer_steps"] > 0
