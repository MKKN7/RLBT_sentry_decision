"""Deskew raw sensor points using timestamped *estimated* base poses.

The input poses and sensor-to-base extrinsic must share a stable estimation
session. No simulator, ground-truth, ROS, or mesh state is read by this module.
Each point uses its own timestamp; translation is linear and rotation is
shortest-arc quaternion SLERP. Missing history and large gaps fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class _Pose:
    sampled_at_s: float
    position_xyz: np.ndarray
    quaternion_xyzw: np.ndarray


def _finite_vector(value: object, size: int) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError, OverflowError):
        return None
    if result.shape != (size,) or not np.isfinite(result).all():
        return None
    return result.copy()


def _unit_quaternion(value: object) -> np.ndarray | None:
    quaternion = _finite_vector(value, 4)
    if quaternion is None:
        return None
    norm = float(np.linalg.norm(quaternion))
    # Permit ordinary serialization/rounding error, not an arbitrary non-unit
    # quaternion or a zero quaternion silently converted into a rotation.
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
        return None
    return quaternion / norm


def _rotate(quaternions_xyzw: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    vector = quaternions_xyzw[..., :3]
    uv = np.cross(vector, points_xyz)
    uuv = np.cross(vector, uv)
    return points_xyz + 2.0 * (quaternions_xyzw[..., 3:4] * uv + uuv)


def _slerp(q0: np.ndarray, q1: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    dot = np.einsum("ij,ij->i", q0, q1)
    q1 = np.where((dot < 0.0)[:, None], -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    result = np.empty_like(q0)
    # Only use the equal-rotation limit at numerical coincidence. A wider
    # small-angle NLERP approximation would change per-point angular velocity.
    linear = dot > 1.0 - 1e-12
    f = fraction[:, None]
    result[linear] = q0[linear] + f[linear] * (q1[linear] - q0[linear])
    curved = ~linear
    if np.any(curved):
        angle = np.arccos(dot[curved])
        denominator = np.sin(angle)
        result[curved] = (
            np.sin((1.0 - fraction[curved]) * angle)[:, None] * q0[curved]
            + np.sin(fraction[curved] * angle)[:, None] * q1[curved]
        ) / denominator[:, None]
    return result / np.linalg.norm(result, axis=1, keepdims=True)


class EstimatedPoseHistory:
    """Bounded estimated base poses in one fixed ``frame_id``.

    ``add`` rejects bad samples without modifying history. Quaternion order is
    XYZW, and rotations map base coordinates into ``frame_id``. Extrinsics map
    sensor coordinates into base coordinates and must be effectively fixed over
    the submitted scan; checking a moving mount is the caller's responsibility.

    ``point_times_s`` are absolute timestamps in the pose clock, not offsets.
    ``has_coverage`` describes continuous time coverage; registration additionally
    checks the actual point timestamps. Calls must be serialized by the caller.
    """

    def __init__(
        self, frame_id: str = "odom", *, max_age_s: float = 3.0,
        max_states: int = 100, max_gap_s: float = 0.25,
        timestamp_tolerance_s: float = 1e-4,
    ) -> None:
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError("frame_id must be a nonempty fixed estimation frame")
        if not all(math.isfinite(v) and v > 0 for v in (max_age_s, max_gap_s)):
            raise ValueError("history age and interpolation gap must be finite and positive")
        if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states < 2:
            raise ValueError("max_states must be an integer of at least two")
        if (not math.isfinite(timestamp_tolerance_s)
                or not 0 <= timestamp_tolerance_s <= 1e-4):
            raise ValueError("timestamp tolerance must be between zero and 1e-4 seconds")
        self.frame_id = frame_id
        self.max_age_s = float(max_age_s)
        self.max_states = max_states
        self.max_gap_s = float(max_gap_s)
        self.timestamp_tolerance_s = float(timestamp_tolerance_s)
        self._poses: list[_Pose] = []

    @property
    def bounds_s(self) -> tuple[float, float] | None:
        if not self._poses:
            return None
        return self._poses[0].sampled_at_s, self._poses[-1].sampled_at_s

    @property
    def sample_count(self) -> int:
        return len(self._poses)

    def clear(self) -> None:
        """Explicitly reset a session; changing frame requires a new history."""
        self._poses.clear()

    def add(
        self, sampled_at_s: float, position_xyz: object,
        quaternion_xyzw: object, frame_id: str = "odom",
    ) -> dict:
        def reject(reason: str) -> dict:
            return {"accepted": False, "reason": reason,
                    "sample_count": self.sample_count, "history_bounds_s": self.bounds_s}

        try:
            sampled_at_s = float(sampled_at_s)
        except (ValueError, TypeError, OverflowError):
            return reject("invalid_timestamp")
        if not math.isfinite(sampled_at_s):
            return reject("invalid_timestamp")
        if frame_id != self.frame_id:
            return reject("frame_changed")
        position = _finite_vector(position_xyz, 3)
        if position is None:
            return reject("invalid_position")
        quaternion = _unit_quaternion(quaternion_xyzw)
        if quaternion is None:
            return reject("invalid_quaternion")
        if self._poses and sampled_at_s <= self._poses[-1].sampled_at_s:
            return reject("out_of_order")
        self._poses.append(_Pose(sampled_at_s, position, quaternion))
        cutoff = sampled_at_s - self.max_age_s
        self._poses = [pose for pose in self._poses if pose.sampled_at_s >= cutoff]
        if len(self._poses) > self.max_states:
            self._poses = self._poses[-self.max_states:]
        return {"accepted": True, "reason": "estimated_pose_added",
                "sample_count": self.sample_count, "history_bounds_s": self.bounds_s}

    def has_coverage(self, start_s: float, end_s: float | None = None) -> bool:
        """Whether every time in the requested interval can be interpolated."""
        if end_s is None:
            end_s = start_s
        try:
            start_s, end_s = float(start_s), float(end_s)
        except (ValueError, TypeError, OverflowError):
            return False
        if not math.isfinite(start_s) or not math.isfinite(end_s) or start_s > end_s:
            return False
        bounds = self.bounds_s
        if bounds is None:
            return False
        tolerance = self.timestamp_tolerance_s
        if start_s < bounds[0] - tolerance or end_s > bounds[1] + tolerance:
            return False
        start_s, end_s = max(start_s, bounds[0]), min(end_s, bounds[1])
        if start_s > end_s:
            # A point within boundary tolerance is clipped to that boundary.
            start_s = end_s = min(max(start_s, bounds[0]), bounds[1])
        times = np.array([pose.sampled_at_s for pose in self._poses])
        if start_s == end_s:
            if np.any(times == start_s):
                return True
            index = int(np.searchsorted(times, start_s))
            return bool(0 < index < len(times)
                        and times[index] - times[index - 1] <= self.max_gap_s + 1e-12)
        gaps = np.diff(times)
        overlaps = (times[:-1] < end_s) & (times[1:] > start_s)
        return not bool(np.any(overlaps & (gaps > self.max_gap_s + 1e-12)))

    def register_scan(
        self, points_xyz: object, point_times_s: object,
        sensor_translation_xyz: object, sensor_quaternion_xyzw: object,
    ) -> dict:
        """Return per-point deskewed coordinates, or ``None`` with a reason.

        Extrapolation is prohibited. The explicitly configured tolerance only
        clips floating point timestamp spill at the oldest/newest pose bounds.
        A scan is rejected as a whole if any requested interpolation is unsafe.
        """
        def reject(reason: str) -> dict:
            return {"accepted": False, "reason": reason, "points_xyz": None,
                    "frame_id": self.frame_id, "history_bounds_s": self.bounds_s}

        try:
            points = np.asarray(points_xyz, dtype=np.float64)
            point_times = np.asarray(point_times_s, dtype=np.float64)
        except (ValueError, TypeError, OverflowError):
            return reject("invalid_scan_arrays")
        if points.ndim != 2 or points.shape[1] != 3:
            return reject("invalid_point_shape")
        if point_times.shape != (len(points),):
            return reject("invalid_timestamp_shape")
        if len(points) == 0:
            return reject("empty_scan")
        if not np.isfinite(points).all():
            return reject("nonfinite_points")
        if not np.isfinite(point_times).all():
            return reject("nonfinite_point_timestamps")
        translation = _finite_vector(sensor_translation_xyz, 3)
        if translation is None:
            return reject("invalid_sensor_translation")
        sensor_quaternion = _unit_quaternion(sensor_quaternion_xyzw)
        if sensor_quaternion is None:
            return reject("invalid_sensor_quaternion")
        bounds = self.bounds_s
        if bounds is None:
            return reject("no_pose_history")
        tolerance = self.timestamp_tolerance_s
        if float(point_times.min()) < bounds[0] - tolerance:
            return reject("before_pose_history")
        if float(point_times.max()) > bounds[1] + tolerance:
            return reject("after_pose_history")
        # This is boundary-only tolerance, not interpolation across missing data.
        query_times = np.clip(point_times, bounds[0], bounds[1])
        times = np.array([pose.sampled_at_s for pose in self._poses])
        positions = np.stack([pose.position_xyz for pose in self._poses])
        quaternions = np.stack([pose.quaternion_xyzw for pose in self._poses])
        right = np.searchsorted(times, query_times, side="left")
        exact = times[right] == query_times
        left = np.maximum(right - 1, 0)
        left[exact] = right[exact]
        gap = times[right] - times[left]
        if np.any((~exact) & (gap > self.max_gap_s + 1e-12)):
            return reject("pose_interpolation_gap")
        fraction = np.zeros(len(points), dtype=np.float64)
        np.divide(query_times - times[left], gap, out=fraction, where=~exact)
        pose_position = positions[left] + fraction[:, None] * (positions[right] - positions[left])
        pose_quaternion = _slerp(quaternions[left], quaternions[right], fraction)
        base_points = _rotate(sensor_quaternion[None, :], points) + translation
        registered = _rotate(pose_quaternion, base_points) + pose_position
        return {"accepted": True, "reason": "estimated_pose_per_point_registration",
                "points_xyz": registered, "frame_id": self.frame_id,
                "point_count": len(points), "history_bounds_s": bounds,
                "scan_bounds_s": (float(point_times.min()), float(point_times.max())),
                "boundary_clipped_point_count": int(np.count_nonzero(query_times != point_times)),
                "max_interpolation_gap_s": float(gap.max()),
                "uses_ground_truth": False}
