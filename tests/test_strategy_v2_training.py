"""New v2 trainer checks: mock transitions only, frozen old sources unchanged."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl_bt.temporal_policy import TemporalActorCritic
from scripts.train_temporal_phase import collect_episode, evaluate_sequence
from scripts.train_strategy_v2 import (SCENARIO_ID, CHECKPOINT_SCHEMA, ROOT, MANIFEST,
    CONTRACT_SOURCES, SOURCE_FILES, SCENE_CONFIG, VALIDATION_SEEDS, TEST_SEEDS,
    build_metadata, load_strategy_checkpoint, normalize_actor_advantages, parse_args,
    ppo_update, training_episode_seed, validate_train_styles, file_sha256, train)


def observation(step=0, *, forced=False):
    state = np.zeros(148, np.float32)
    state[0] = step / 5.
    state[7] = .7
    candidates = np.zeros((16, 16), np.float32)
    candidates[:, 0] = np.linspace(-1, 1, 16)
    candidates[:, 1] = step / 9.
    mask = np.zeros(16, bool)
    mask[:1 if forced else 4] = True
    return {"state": state, "candidates": candidates, "mask": mask}


class FakeEpisode:
    def __init__(self, terminal_after=4, *, forced=False, style="capture"):
        self.done = False
        self.decisions = 0
        self.style = style
        self.team = 0
        self.cached = observation(forced=forced)
        self.terminal_after = terminal_after
        self.forced = forced

    def step(self, action):
        assert self.cached["mask"][action]
        self.decisions += 1
        self.done = self.decisions >= self.terminal_after
        self.cached = observation(self.decisions, forced=self.forced)
        info = {"duration_s": .5, "executed_candidate": action, "override_reason": None}
        if self.done:
            info["episode"] = {"scenario_id": "alpharm_role_balanced_v2", "win": False,
                               "return": .5, "decisions": self.decisions}
        return self.cached, .1 * self.decisions, self.done, info


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def args_for(path, memory="gru"):
    return parse_args(["--run", str(path), "--memory", memory, "--seed", "271",
                       "--decisions", "4096", "--train-styles", "capture", "resupply"])


@pytest.mark.parametrize("memory", ["mlp", "gru"])
def test_reused_collector_and_full_sequence_have_exact_onpolicy_replay(memory):
    torch.manual_seed(18)
    model = TemporalActorCritic(memory=memory)
    sequence = collect_episode(model, FakeEpisode(5), 8, 800001)
    logp, value, _ = evaluate_sequence(model, sequence)
    np.testing.assert_allclose(logp.detach(), sequence["old_logp"], atol=1e-6)
    np.testing.assert_allclose(value.detach(), sequence["values"], atol=1e-6)
    assert sequence["terminated"][-1] and not sequence["truncated"].any()


@pytest.mark.parametrize("signal", [1., -1.])
def test_single_valid_advantage_retains_directional_actor_gradient(signal):
    torch.manual_seed(32)
    model = TemporalActorCritic(memory="gru")
    sequence = collect_episode(model, FakeEpisode(1), 8, 800001)
    # A true terminal residual of +1/-1 is intentionally not centered away.
    sequence["rewards"][:] = sequence["values"] + signal
    before = float(evaluate_sequence(model, sequence)[0][0].detach())
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    metrics = ppo_update(model, optimizer, [sequence], epochs=1, entropy_coef=0.)
    after = float(evaluate_sequence(model, sequence)[0][0].detach())
    assert (after - before) * signal > 1e-6
    assert metrics["actor_decisions"] == 1
    assert metrics["advantage_normalization_mode"] == "raw_single_actor_decision"
    assert metrics["actor_gradient_norm"] > 0
    assert metrics["first_epoch_approx_kl"] < 1e-6


@pytest.mark.parametrize("raw,mode", [([], "no_actor_decisions"), ([2.], "raw_single_actor_decision"),
    ([2., 2.], "raw_low_variance"), ([-2., -2.], "raw_low_variance"),
    ([1., 3., 5.], "mean_std")])
def test_advantage_normalization_modes(raw, mode):
    normalized, info = normalize_actor_advantages(raw)
    assert info["advantage_normalization_mode"] == mode
    if mode == "mean_std":
        assert normalized.mean() == pytest.approx(0., abs=1e-7)
        assert normalized.std() == pytest.approx(1.)
    else:
        np.testing.assert_array_equal(normalized, np.asarray(raw, np.float32))


@pytest.mark.parametrize("memory", ["mlp", "gru"])
def test_finite_updates_with_forced_batch_keep_actor_even_after_adam_momentum(memory):
    torch.manual_seed(52)
    model = TemporalActorCritic(memory=memory)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    normal = collect_episode(model, FakeEpisode(4), 8, 800001)
    metrics = ppo_update(model, optimizer, [normal], epochs=2)
    assert metrics["actor_decisions"] == 4
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    before_actor = [p.detach().clone() for p in model.actor.parameters()]
    before_critic = [p.detach().clone() for p in model.critic.parameters()]
    forced = collect_episode(model, FakeEpisode(3, forced=True), 8, 800002)
    metrics = ppo_update(model, optimizer, [forced], epochs=2)
    assert metrics["advantage_normalization_mode"] == "no_actor_decisions"
    assert metrics["actor_decisions"] == 0
    assert all(torch.equal(a, b) for a, b in zip(before_actor, model.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(before_critic, model.critic.parameters()))


def checkpoint_payload(tmp_path):
    args = args_for(tmp_path / "unused")
    commit = json.loads((ROOT / MANIFEST).read_text())["commit"]
    metadata = build_metadata(args, SimpleNamespace(source_commit=commit))
    model = TemporalActorCritic(memory="gru")
    return {"scenario_id": SCENARIO_ID, "checkpoint_schema": CHECKPOINT_SCHEMA,
            "metadata": metadata, "model": model.state_dict()}


def test_loader_and_dependency_snapshot_schema_cover_new_and_imported_sources(tmp_path):
    payload = checkpoint_payload(tmp_path)
    path = tmp_path / "current.pt"
    torch.save(payload, path)
    model, loaded = load_strategy_checkpoint(path)
    assert model.memory == "gru"
    assert loaded["checkpoint_schema"] == 2
    assert "scripts/train_temporal_phase.py" in CONTRACT_SOURCES
    assert "rl_bt/reward_components.py" in CONTRACT_SOURCES
    assert "rl_bt/strategy_scenarios.py" in CONTRACT_SOURCES
    assert SCENE_CONFIG in CONTRACT_SOURCES
    assert len(payload["metadata"]["upstream_files_sha256"]) == 4
    assert set(SOURCE_FILES) == set(payload["metadata"]["implementation_sha256"])


@pytest.mark.parametrize("damage,match", [
    ("old_schema", "role-balanced"), ("old_scenario", "role-balanced"),
    ("changed_scene", "scene config"), ("changed_source", "source mismatch"),
    ("heldout_style", "TRAIN_STYLES"), ("bad_mode", "support mode"),
    ("live_claim", "real/official"), ("old_initialization", "fresh model"),
    ("seed_role", "seed roles"), ("changed_upstream", "execution assets")])
def test_loader_rejects_old_or_changed_contracts(tmp_path, damage, match):
    payload = checkpoint_payload(tmp_path)
    metadata = payload["metadata"]
    if damage == "old_schema":
        payload["checkpoint_schema"] = 1
    elif damage == "old_scenario":
        payload["scenario_id"] = "alpharm_temporal_global_phase_v1"
    elif damage == "changed_scene":
        metadata["scene_config_sha256"] = "other"
    elif damage == "changed_source":
        metadata["implementation_sha256"]["scripts/train_temporal_phase.py"] = "other"
    elif damage == "heldout_style":
        metadata["train_styles"] = ["pursuit"]
    elif damage == "bad_mode":
        metadata["support_mode"] = "invented"
    elif damage == "live_claim":
        metadata["real_nav2_in_loop"] = True
    elif damage == "old_initialization":
        metadata["historical_model_initialization"] = True
    elif damage == "seed_role":
        metadata["seed_role"] = "test"
    else:
        metadata["upstream_files_sha256"] = {}
    path = tmp_path / "broken.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match=match):
        load_strategy_checkpoint(path)


def test_seed_namespaces_styles_and_defaults():
    assert training_episode_seed(271, 0) == 3510001
    assert training_episode_seed(271, 9999) == 3520000
    assert training_episode_seed(272, 0) == 3520001
    assert not set(VALIDATION_SEEDS + TEST_SEEDS) & {
        training_episode_seed(seed, index) for seed in (0, 271, 809) for index in (0, 1, 9999)}
    for seed, index in [(-1, 0), (True, 0), (2**32, 0), (0, 10000), (0, -1)]:
        with pytest.raises(ValueError):
            training_episode_seed(seed, index)
    assert validate_train_styles(["capture", "pretrained"]) == ("capture", "pretrained")
    for styles in ([], ["capture", "capture"], ["pursuit"]):
        with pytest.raises(ValueError):
            validate_train_styles(styles)
    args = args_for("/tmp/parse-only")
    assert args.lambda_per_second == .995 and args.learning_rate == .001 and args.entropy_coef == .001
    assert args.episodes_per_update == args.epochs == 4


def test_mock_entrypoint_finishes_whole_episode_snapshots_and_refuses_overwrite(tmp_path, monkeypatch):
    import scripts.train_strategy_v2 as trainer
    commit = json.loads((ROOT / MANIFEST).read_text())["commit"]
    monkeypatch.setattr(trainer, "AlphaRuntime", lambda: SimpleNamespace(source_commit=commit))
    styles = []
    def factory(runtime, seed, **kwargs):
        styles.append(kwargs["opponent_style"])
        env = FakeEpisode(3, style=kwargs["opponent_style"])
        env.config_sha256 = file_sha256(ROOT / SCENE_CONFIG)
        return env
    monkeypatch.setattr(trainer, "StrategyTaskEnv", factory)
    args = args_for(tmp_path / "new_mock_run", "mlp")
    args.decisions, args.epochs = 2, 1
    summary = train(args)
    assert summary["decisions"] == 3 and summary["budget_overshoot_decisions"] == 1
    assert summary["completed_training_episodes"] == 1 and summary["truncated_sequences"] == 0
    assert styles == ["resupply"] or styles == ["capture"]
    assert (args.run / "source" / "scripts/train_temporal_phase.py").is_file()
    assert (args.run / "source" / "data/references/alpharm_jax_review/upstream/demo/web/env_step.onnx").is_file()
    _, payload = load_strategy_checkpoint(args.run / "final.pt")
    assert payload["training_episode_seeds"] == [3510001]
    with pytest.raises(FileExistsError):
        train(args)
