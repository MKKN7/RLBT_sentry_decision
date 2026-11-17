import pytest

from rl_bt.nav2_transport import sample_freshness, terminal_status, validate_path


def test_cancel_acceptance_never_becomes_terminal_without_action_result():
    assert terminal_status(4) == "succeeded"
    assert terminal_status(5) == "canceled"
    assert terminal_status(6) == "failed"
    for status in (0, 1, 2, 3, True):
        with pytest.raises(ValueError):
            terminal_status(status)


def test_future_and_stale_samples_fail_in_same_clock_domain():
    assert sample_freshness(now_s=4., sampled_at_s=3.8, max_age_s=.5) == (True, "fresh")
    assert sample_freshness(now_s=4., sampled_at_s=3., max_age_s=.5) == (False, "stale_sample")
    assert sample_freshness(now_s=4., sampled_at_s=4.1, max_age_s=.5) == (False, "future_sample")
    with pytest.raises(ValueError):
        sample_freshness(now_s=float("nan"), sampled_at_s=0., max_age_s=.5)


def test_planner_path_requires_correct_frame_finite_points_and_goal_endpoint():
    options = dict(expected_frame="odom", goal_xy=(1., 2.), endpoint_tolerance_m=.1)
    assert validate_path([("odom", .2, .4), ("odom", 1., 2.)], **options)[0]
    assert validate_path([], **options) == (False, "empty_path")
    assert validate_path([("map", 1., 2.)], **options)[1] == "path_frame_or_position_invalid"
    assert validate_path([("odom", 1., float("nan"))], **options)[1] == "path_frame_or_position_invalid"
    assert validate_path([("odom", 2., 2.)], **options)[1] == "path_endpoint_mismatch"
