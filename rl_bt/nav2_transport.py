"""Pure validation helpers for the ROS 2 Nav2 transport adapter.

No ROS imports, velocity policy or map assumptions live here.
"""
import math


def terminal_status(goal_status):
    """action_msgs/GoalStatus: terminal results only; cancel ACK is not terminal."""
    mapping = {4: "succeeded", 5: "canceled", 6: "failed"}
    if type(goal_status) is not int or goal_status not in mapping:
        raise ValueError("Nav2 result requires terminal action status 4, 5 or 6")
    return mapping[goal_status]


def sample_freshness(*, now_s, sampled_at_s, max_age_s):
    values = (now_s, sampled_at_s, max_age_s)
    if not all(not isinstance(v, bool) and isinstance(v, (int, float))
               and math.isfinite(v) and v >= 0 for v in values) or max_age_s == 0:
        raise ValueError("Finite nonnegative clocks and positive age limit required")
    age = now_s-sampled_at_s
    if age < -1e-6:
        return False, "future_sample"
    if age > max_age_s:
        return False, "stale_sample"
    return True, "fresh"


def validate_path(poses, *, expected_frame, goal_xy, endpoint_tolerance_m):
    """Validate a ComputePath result expressed as [(frame,x,y),...].

    Path geometry still needs the actual Nav2 costmap and task route constraints;
    successful planning is not a promise of physical execution success.
    """
    if not expected_frame or not math.isfinite(endpoint_tolerance_m) or endpoint_tolerance_m <= 0:
        raise ValueError("Frame and finite positive endpoint tolerance required")
    if len(goal_xy) != 2 or not all(math.isfinite(x) for x in goal_xy):
        raise ValueError("Finite goal XY required")
    if not poses:
        return False, "empty_path"
    for frame, x, y in poses:
        if frame != expected_frame or not math.isfinite(x) or not math.isfinite(y):
            return False, "path_frame_or_position_invalid"
    _, x, y = poses[-1]
    if math.hypot(x-goal_xy[0], y-goal_xy[1]) > endpoint_tolerance_m:
        return False, "path_endpoint_mismatch"
    return True, "path_valid"
