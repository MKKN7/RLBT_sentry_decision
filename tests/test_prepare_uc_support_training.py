import importlib.util
from pathlib import Path

import numpy as np

from rl_bt.support_costmap import path_support_check

_PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_uc_support_training.py"
_SPEC = importlib.util.spec_from_file_location("prepare_uc_support_training", _PATH)
prepare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prepare)


def test_unknown_padding_preserves_world_coordinates_and_labels():
    source = np.array([[0, -1, 100], [100, 0, -1]], np.int8)
    result, origin, low = prepare.pad_unknown(source, [1., 2.], .15)
    assert result.shape == (121, 121)
    np.testing.assert_array_equal(result[low[0]:low[0] + 2, low[1]:low[1] + 3], source)
    np.testing.assert_allclose(origin + low * .15, [1., 2.])
    assert (result == 0).sum() == (source == 0).sum()
    assert (result == 100).sum() == (source == 100).sum()


def test_estimated_pose_requires_time_alignment():
    assert prepare.matched_estimated_pose([[1., 2., 3.]], 1.3) is None
    pose, stamp, delta = prepare.matched_estimated_pose([[1., 2., 3.]], 1.2)
    np.testing.assert_array_equal(pose, [2., 3.])
    assert stamp == 1. and delta < .25


def test_candidate_extraction_never_uses_unknown_shortcut():
    occupancy = np.full((30, 30), -1, np.int8)
    occupancy[5:25, 5:25] = 0
    occupancy[16, 5:25] = -1
    grid = prepare.make_grid(occupancy, [-1.5, -1.5], .1, 1.)
    start = np.array([-.4, 0.])
    points, methods, robust_n = prepare.extract_goals(grid, start)
    assert len(points) > 0 and robust_n > 0
    for point in points:
        assert path_support_check(grid, [start, point], radius_m=.411)["accepted"]
        assert point[0] < -.3
    assert len(methods) == len(points)


def test_unsupported_current_footprint_has_no_candidates():
    grid = prepare.make_grid(np.full((20, 20), -1, np.int8), [-1., -1.], .1, 1.)
    points, _, _ = prepare.extract_goals(grid, np.zeros(2))
    assert points.shape == (0, 2)


def test_family_split_keeps_retries_together_and_is_deterministic():
    family = prepare.family_name("uc_support_dual_lateral_20261116")
    assert family == prepare.family_name("uc_support_dual_lateral_final_20261116")
    assert prepare.family_name("uc_support_dual_namespaced_forward_20261116") == "uc_support_dual_forward"
    splits = prepare.split_families(["a", "b", "c", family])
    assert splits == prepare.split_families([family, "c", "a", "b", "a"])
    assert set(splits.values()) == {"train", "validation", "test"}


def test_replay_signature_ignores_storage_but_preserves_semantics():
    occupancy = np.zeros((5, 5), np.int8)
    first = prepare.observation_signature(occupancy, [0., 0.], .1, [0., 0.])
    assert first == prepare.observation_signature(occupancy.copy(), [0., 0.], .1, [0., 0.])
    occupancy[0, 0] = -1
    assert first != prepare.observation_signature(occupancy, [0., 0.], .1, [0., 0.])
