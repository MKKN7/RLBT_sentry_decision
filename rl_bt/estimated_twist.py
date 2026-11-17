"""Planar body velocity from timestamped estimated chassis poses, never GT.

Finite differences approximate velocity over the preceding sample interval.
Invalid results are explicitly unavailable rather than reported as zero speed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EstimatedPose2D:
    sampled_at_s: float
    x_m: float
    y_m: float
    yaw_rad: float
    frame_id: str
    child_frame_id: str

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (
            self.sampled_at_s, self.x_m, self.y_m, self.yaw_rad,
        )):
            raise ValueError("pose values must be finite")
        if not self.frame_id or not self.child_frame_id:
            raise ValueError("pose reference and child frames are required")


@dataclass(frozen=True)
class EstimatedBodyTwist2D:
    sampled_at_s: float
    valid: bool
    reason: str
    interval_s: float | None = None
    vx_m_s: float | None = None
    vy_m_s: float | None = None
    wz_rad_s: float | None = None


class PoseDifferenceTwistEstimator:
    """Bounded finite difference in one stable map/odom coordinate session.

    Limits must be chosen for the actual chassis. Discontinuity/gap/frame-change
    results reset the anchor and require a subsequent consistent sample. A stale
    or out-of-order sample cannot reset or freshen an existing estimate.
    """

    def __init__(
        self, *, min_interval_s: float, max_interval_s: float,
        max_translation_speed_m_s: float, max_rotation_speed_rad_s: float,
        max_data_age_s: float,
    ) -> None:
        values = (
            min_interval_s, max_interval_s, max_translation_speed_m_s,
            max_rotation_speed_rad_s, max_data_age_s,
        )
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("estimator limits must be finite and positive")
        if min_interval_s >= max_interval_s:
            raise ValueError("min interval must be less than max interval")
        self.min_interval_s = min_interval_s
        self.max_interval_s = max_interval_s
        self.max_translation_speed_m_s = max_translation_speed_m_s
        self.max_rotation_speed_rad_s = max_rotation_speed_rad_s
        self.max_data_age_s = max_data_age_s
        self._anchor: EstimatedPose2D | None = None

    def reset(self) -> None:
        self._anchor = None

    def update(self, sample: EstimatedPose2D, *, now_s: float) -> EstimatedBodyTwist2D:
        if not math.isfinite(now_s):
            raise ValueError("now_s must be finite and use the pose timestamp clock")

        def invalid(reason: str, interval: float | None = None) -> EstimatedBodyTwist2D:
            return EstimatedBodyTwist2D(sample.sampled_at_s, False, reason, interval)

        age = now_s - sample.sampled_at_s
        if age < -1e-9:
            return invalid("future_sample")
        if age > self.max_data_age_s:
            return invalid("stale_sample")
        old = self._anchor
        if old is None:
            self._anchor = sample
            return invalid("warmup")
        dt = sample.sampled_at_s - old.sampled_at_s
        if dt <= 0:
            return invalid("out_of_order", dt)
        if (old.frame_id, old.child_frame_id) != (sample.frame_id, sample.child_frame_id):
            self._anchor = sample
            return invalid("frame_changed", dt)
        if dt > self.max_interval_s:
            self._anchor = sample
            return invalid("sample_gap", dt)
        if dt < self.min_interval_s:
            # Accumulate time from the last accepted anchor rather than rejecting
            # every high-rate sample indefinitely.
            return invalid("interval_too_short", dt)
        vx_world = (sample.x_m - old.x_m) / dt
        vy_world = (sample.y_m - old.y_m) / dt
        yaw_delta = math.atan2(
            math.sin(sample.yaw_rad - old.yaw_rad),
            math.cos(sample.yaw_rad - old.yaw_rad),
        )
        wz = yaw_delta / dt
        self._anchor = sample
        if (math.hypot(vx_world, vy_world) > self.max_translation_speed_m_s
                or abs(wz) > self.max_rotation_speed_rad_s):
            return invalid("pose_discontinuity", dt)
        c, s = math.cos(sample.yaw_rad), math.sin(sample.yaw_rad)
        return EstimatedBodyTwist2D(
            sample.sampled_at_s, True, "estimated_pose_difference", dt,
            c * vx_world + s * vy_world,
            -s * vx_world + c * vy_world,
            wz,
        )
