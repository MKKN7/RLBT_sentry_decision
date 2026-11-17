import math

import numpy as np
import pytest

from rl_bt.scan_registration import EstimatedPoseHistory


IDENTITY = [0.0, 0.0, 0.0, 1.0]


def yaw_quaternion(yaw):
    return [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]


def register(history, points, times, translation=(0, 0, 0), quaternion=IDENTITY):
    return history.register_scan(points, times, translation, quaternion)


def test_per_point_translation_and_slerp_do_not_use_end_of_scan_pose():
    h = EstimatedPoseHistory()
    h.add(0.0, [0, 0, 0], IDENTITY)
    h.add(0.2, [2, 4, 6], yaw_quaternion(math.pi))
    result = register(h, [[1, 0, 0]] * 3, [0.0, 0.1, 0.2])
    assert result["accepted"]
    np.testing.assert_allclose(result["points_xyz"], [[1, 0, 0], [1, 3, 3], [1, 4, 6]], atol=1e-12)
    assert result["uses_ground_truth"] is False


def test_sensor_rotation_then_translation_then_estimated_base_transform():
    h = EstimatedPoseHistory()
    h.add(2.0, [10, 20, 30], yaw_quaternion(math.pi / 2))
    out = register(h, [[1, 0, 0]], [2.0],
                   translation=(1, 0, 2), quaternion=yaw_quaternion(math.pi / 2))
    # Sensor x -> base y; add sensor origin; base x/y -> odom y/-x.
    assert out["accepted"]
    assert out["points_xyz"][0] == pytest.approx([9, 21, 32])


def test_shortest_arc_slerp_across_pi_and_quaternion_sign():
    h = EstimatedPoseHistory()
    h.add(0, [0, 0, 0], yaw_quaternion(math.radians(170)))
    h.add(0.2, [0, 0, 0], yaw_quaternion(math.radians(-170)))
    out = register(h, [[1, 0, 0]], [0.1])
    assert out["points_xyz"][0] == pytest.approx([-1, 0, 0], abs=1e-12)
    h.clear()
    h.add(0, [0, 0, 0], IDENTITY)
    h.add(0.2, [0, 0, 0], [0, 0, 0, -1])
    assert register(h, [[1, 2, 3]], [0.1])["points_xyz"][0] == pytest.approx([1, 2, 3])


def test_scan_waits_for_future_history_and_rejects_old_history():
    h = EstimatedPoseHistory(max_age_s=0.2)
    assert register(h, [[0, 0, 0]], [0])["reason"] == "no_pose_history"
    h.add(0, [0, 0, 0], IDENTITY)
    assert not h.has_coverage(0, 0.1)
    assert register(h, [[0, 0, 0]], [0.1])["reason"] == "after_pose_history"
    h.add(0.1, [1, 0, 0], IDENTITY)
    assert h.has_coverage(0, 0.1)
    assert register(h, [[0, 0, 0]], [0.05])["accepted"]
    h.add(0.3, [3, 0, 0], IDENTITY)
    assert not h.has_coverage(0)
    assert register(h, [[0, 0, 0]], [0])["reason"] == "before_pose_history"


def test_large_gap_fails_closed_but_exact_estimates_remain_usable():
    h = EstimatedPoseHistory()
    h.add(0, [0, 0, 0], IDENTITY)
    h.add(0.3, [3, 0, 0], IDENTITY)
    assert not h.has_coverage(0, 0.3)
    assert not h.has_coverage(0.1)
    assert register(h, [[0, 0, 0]], [0.1])["reason"] == "pose_interpolation_gap"
    assert register(h, [[0, 0, 0]], [0.3])["accepted"]
    h.add(0.4, [4, 0, 0], IDENTITY)
    assert h.has_coverage(0.3, 0.4)
    assert register(h, [[0, 0, 0]], [0.35])["accepted"]


def test_boundary_tolerance_is_explicit_small_and_never_extrapolates():
    h = EstimatedPoseHistory(timestamp_tolerance_s=1e-4)
    h.add(1, [10, 0, 0], IDENTITY)
    h.add(1.2, [12, 0, 0], IDENTITY)
    out = register(h, [[0, 0, 0], [0, 0, 0]], [1 - 5e-5, 1.2 + 5e-5])
    assert out["accepted"]
    assert out["boundary_clipped_point_count"] == 2
    assert out["points_xyz"][:, 0] == pytest.approx([10, 12])
    assert h.has_coverage(1 - 5e-5) and h.has_coverage(1.2 + 5e-5)
    assert register(h, [[0, 0, 0]], [1 - 2e-4])["reason"] == "before_pose_history"
    assert register(h, [[0, 0, 0]], [1.2 + 2e-4])["reason"] == "after_pose_history"
    strict = EstimatedPoseHistory(timestamp_tolerance_s=0)
    strict.add(1, [0, 0, 0], IDENTITY)
    assert not strict.has_coverage(1 - 1e-10)


@pytest.mark.parametrize("time,position,quaternion,frame,reason", [
    (float("nan"), [0, 0, 0], IDENTITY, "odom", "invalid_timestamp"),
    (2, [float("inf"), 0, 0], IDENTITY, "odom", "invalid_position"),
    (2, [0, 0], IDENTITY, "odom", "invalid_position"),
    (2, [0, 0, 0], [0, 0, 0, 0], "odom", "invalid_quaternion"),
    (2, [0, 0, 0], [0, 0, 0, 2], "odom", "invalid_quaternion"),
    (2, [0, 0, 0], [0, 0, float("nan"), 1], "odom", "invalid_quaternion"),
    (2, [0, 0, 0], IDENTITY, "map", "frame_changed"),
    (0.9, [0, 0, 0], IDENTITY, "odom", "out_of_order"),
    (1, [0, 0, 0], IDENTITY, "odom", "out_of_order"),
])
def test_rejected_pose_does_not_mutate_or_freshen_history(time, position, quaternion, frame, reason):
    h = EstimatedPoseHistory()
    h.add(1, [1, 2, 3], IDENTITY)
    result = h.add(time, position, quaternion, frame_id=frame)
    assert not result["accepted"] and result["reason"] == reason
    assert h.bounds_s == (1, 1) and h.sample_count == 1
    assert register(h, [[0, 0, 0]], [1])["points_xyz"][0] == pytest.approx([1, 2, 3])


@pytest.mark.parametrize("points,times,translation,quaternion,reason", [
    ([[0, 0]], [0], [0, 0, 0], IDENTITY, "invalid_point_shape"),
    ([[0, 0, 0]], [[0]], [0, 0, 0], IDENTITY, "invalid_timestamp_shape"),
    (np.empty((0, 3)), [], [0, 0, 0], IDENTITY, "empty_scan"),
    ([[0, float("nan"), 0]], [0], [0, 0, 0], IDENTITY, "nonfinite_points"),
    ([[0, 0, 0]], [float("nan")], [0, 0, 0], IDENTITY, "nonfinite_point_timestamps"),
    ([[0, 0, 0]], [0], [0, 0], IDENTITY, "invalid_sensor_translation"),
    ([[0, 0, 0]], [0], [0, 0, 0], [0, 0, 0, 0], "invalid_sensor_quaternion"),
])
def test_invalid_scan_is_wholly_rejected(points, times, translation, quaternion, reason):
    h = EstimatedPoseHistory()
    h.add(0, [0, 0, 0], IDENTITY)
    result = register(h, points, times, translation, quaternion)
    assert result["accepted"] is False
    assert result["reason"] == reason
    assert result["points_xyz"] is None


def test_history_storage_is_bounded_and_copies_input_arrays():
    h = EstimatedPoseHistory(max_age_s=0.3, max_states=3)
    position = np.array([1.0, 0, 0])
    quaternion = np.array(IDENTITY)
    h.add(0, position, quaternion)
    position[:] = 100
    quaternion[:] = 0
    assert register(h, [[0, 0, 0]], [0])["points_xyz"][0, 0] == 1
    for time in [0.1, 0.2, 0.3, 0.4, 0.8]:
        h.add(time, [0, 0, 0], IDENTITY)
        assert h.sample_count <= 3
        assert h.bounds_s[1] - h.bounds_s[0] <= 0.3 + 1e-12
    assert h.sample_count == 1 and h.bounds_s == (0.8, 0.8)


def test_unordered_points_preserve_scan_order_and_arbitrary_3d_rotation():
    h = EstimatedPoseHistory()
    # Half-turn about x at the last pose; intermediate rotation is 90 degrees.
    h.add(0, [0, 0, 0], IDENTITY)
    h.add(0.2, [0, 0, 2], [1, 0, 0, 0])
    points = np.array([[0, 1, 0]] * 3, dtype=np.float32)
    saved = points.copy()
    out = register(h, points, [0.2, 0, 0.1])
    np.testing.assert_allclose(out["points_xyz"], [[0, -1, 2], [0, 1, 0], [0, 0, 2]], atol=1e-12)
    np.testing.assert_array_equal(points, saved)


def test_registration_is_equivariant_to_estimated_frame_translation():
    h1, h2 = EstimatedPoseHistory(), EstimatedPoseHistory()
    shift = np.array([123, -57, 19])
    for t in [0, 0.1, 0.2]:
        position = np.array([t, 2 * t, -t])
        q = yaw_quaternion(t)
        h1.add(t, position, q)
        h2.add(t, position + shift, q)
    points, times = [[1, 2, 3], [-2, 4, 5]], [0.03, 0.17]
    np.testing.assert_allclose(register(h2, points, times)["points_xyz"]
                               - register(h1, points, times)["points_xyz"],
                               np.tile(shift, (2, 1)), atol=1e-12)


@pytest.mark.parametrize("kwargs", [
    {"frame_id": ""}, {"max_age_s": 0}, {"max_gap_s": float("inf")},
    {"max_states": 1}, {"max_states": True}, {"timestamp_tolerance_s": -1},
    {"timestamp_tolerance_s": 1e-3},
])
def test_configuration_validation(kwargs):
    with pytest.raises(ValueError):
        EstimatedPoseHistory(**kwargs)
