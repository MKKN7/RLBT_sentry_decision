import math

import pytest

from rl_bt.estimated_twist import EstimatedPose2D, PoseDifferenceTwistEstimator


def estimator():
    return PoseDifferenceTwistEstimator(
        min_interval_s=0.01, max_interval_s=0.3,
        max_translation_speed_m_s=3.0, max_rotation_speed_rad_s=6.0,
        max_data_age_s=0.2,
    )


def pose(t, x=0.0, y=0.0, yaw=0.0, frame="odom", child="base_footprint"):
    return EstimatedPose2D(t, x, y, yaw, frame, child)


def test_body_velocity_uses_chassis_frame():
    e = estimator()
    assert not e.update(pose(0, yaw=math.pi / 2), now_s=0).valid
    out = e.update(pose(0.1, y=0.1, yaw=math.pi / 2), now_s=0.1)
    assert out.valid
    assert out.vx_m_s == pytest.approx(1)
    assert out.vy_m_s == pytest.approx(0, abs=1e-12)
    assert out.wz_rad_s == 0


def test_wrap_around_pi_has_small_angular_velocity():
    e = estimator()
    e.update(pose(0, yaw=math.pi - 0.01), now_s=0)
    out = e.update(pose(0.1, yaw=-math.pi + 0.01), now_s=0.1)
    assert out.valid
    assert out.wz_rad_s == pytest.approx(0.2)


def test_invalid_values_are_unavailable_not_stationary():
    out = estimator().update(pose(0), now_s=0)
    assert not out.valid
    assert out.vx_m_s is None and out.vy_m_s is None and out.wz_rad_s is None


def test_stale_and_out_of_order_do_not_replace_anchor():
    e = estimator()
    e.update(pose(1, x=1), now_s=1)
    assert e.update(pose(0.9, x=100), now_s=1).reason == "out_of_order"
    assert e.update(pose(0.1, x=100), now_s=1).reason == "stale_sample"
    out = e.update(pose(1.1, x=1.1), now_s=1.1)
    assert out.valid and out.vx_m_s == pytest.approx(1)


def test_short_intervals_accumulate_time():
    e = estimator()
    e.update(pose(0), now_s=0)
    assert e.update(pose(0.005, x=0.005), now_s=0.005).reason == "interval_too_short"
    out = e.update(pose(0.02, x=0.02), now_s=0.02)
    assert out.valid and out.vx_m_s == pytest.approx(1)


@pytest.mark.parametrize("kind", ["frame", "gap", "jump"])
def test_discontinuities_require_a_new_consistent_pair(kind):
    e = estimator()
    e.update(pose(0), now_s=0)
    changed = {
        "frame": pose(0.1, x=10, frame="new_odom"),
        "gap": pose(1, x=10),
        "jump": pose(0.1, x=10),
    }[kind]
    out = e.update(changed, now_s=changed.sampled_at_s)
    assert out.reason == {
        "frame": "frame_changed", "gap": "sample_gap", "jump": "pose_discontinuity",
    }[kind]
    assert not out.valid
    out = e.update(pose(changed.sampled_at_s + 0.1, x=10.1, frame=changed.frame_id),
                   now_s=changed.sampled_at_s + 0.1)
    assert out.valid and out.vx_m_s == pytest.approx(1)


def test_future_timestamp_cannot_initialize_estimator():
    e = estimator()
    assert e.update(pose(10), now_s=0).reason == "future_sample"
    assert e.update(pose(0), now_s=0).reason == "warmup"


def test_validation():
    with pytest.raises(ValueError):
        pose(0, x=float("nan"))
    with pytest.raises(ValueError):
        pose(0, frame="")
    with pytest.raises(ValueError):
        PoseDifferenceTwistEstimator(
            min_interval_s=0.2, max_interval_s=0.1,
            max_translation_speed_m_s=1, max_rotation_speed_rad_s=1,
            max_data_age_s=1,
        )
