"""Verify paired statistics without rerunning expensive game episodes."""
import pytest

from scripts.evaluate_phase1 import cluster_interval, paired_summary, summarize


def episode(policy, seed, style, side, win, return_value, *, draw=False):
    return {"policy": policy, "seed": seed, "opponent_style": style, "team": side,
            "win": win, "draw": draw, "return": return_value,
            "held_out_style": style == "pursuit", "switches": 1, "overrides": 0,
            "recoveries": 0}


def test_pairing_uses_seed_style_and_side_and_preserves_seed_correlation():
    rows = []
    # Two correlated sides per seed must produce 2 independent blocks, not 4.
    for seed, learned_wins in ((100, True), (101, False)):
        for side in (0, 1):
            rows += [episode("bt_baseline", seed, "capture", side, False, 1),
                     episode("learned", seed, "capture", side, learned_wins,
                             3 if learned_wins else -1)]
    result = paired_summary(rows)
    assert result["pairs"] == 4
    assert result["independent_seeds"] == 2
    assert result["learned_wins_bt_does_not"] == 2
    assert result["bt_wins_learned_does_not"] == 0
    assert result["learned_minus_bt"]["win_difference"]["mean"] == .5
    assert result["learned_minus_bt"]["return_difference"]["mean"] == 0
    assert result["learned_minus_bt"]["win_difference"]["seed_bootstrap_95"] == [0., 1.]


def test_draw_scores_half_point_and_missing_pair_is_rejected():
    bt = episode("bt_baseline", 1, "pursuit", 0, False, 0)
    learned = episode("learned", 1, "pursuit", 0, False, 1, draw=True)
    result = paired_summary([learned, bt])
    assert result["learned_minus_bt"]["match_points_difference"]["mean"] == .5
    assert result["learned_minus_bt"]["win_difference"]["seed_bootstrap_95"] == [None, None]
    with pytest.raises(ValueError, match="requires one BT"):
        paired_summary([bt])
    with pytest.raises(ValueError, match="Duplicate"):
        paired_summary([bt, learned, bt])


def test_summary_separates_held_out_styles_and_sides():
    rows = []
    for style, win in (("capture", False), ("pursuit", True)):
        for policy in ("bt_baseline", "learned"):
            rows.append(episode(policy, 1, style, 0, win, 1))
    result = summarize(rows)
    assert result["groups"]["learned/seen_styles"]["win_rate"] == 0
    assert result["groups"]["learned/held_out_style"]["win_rate"] == 1
    assert result["groups"]["learned/all_styles"]["episodes"] == 2
    assert result["by_side"]["1"]["paired"]["pairs"] == 0
    assert cluster_interval([]) == [None, None]
