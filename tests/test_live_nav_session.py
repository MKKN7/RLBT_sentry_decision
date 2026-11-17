"""RPC admission checks are testable without ROS or simulator truth."""
import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location("nav2_rl_session", Path(__file__).parents[1]/"scripts/nav2_rl_session.py")
session = importlib.util.module_from_spec(spec)
spec.loader.exec_module(session)


def request(**updates):
    value = {"op": "step", "action": 0, "candidate_version": 7, "issued_at_s": 5.}
    value.update(updates)
    return value


def admitted(value, *, now=5.5, mask=None):
    return session.validate_request(value, version=7, issued_at_s=5., now_s=now,
        mask=[True]*16 if mask is None else mask)


def test_admission_rejects_old_and_future_snapshots_without_recomputing_mask():
    assert admitted(request()) is None
    assert admitted(request(candidate_version=6)) == "candidate_version_mismatch"
    assert admitted(request(), now=7.0001) == "expired_or_future_intent"
    assert admitted(request(), now=4.999) == "expired_or_future_intent"
    assert admitted(request(issued_at_s=5.01)) == "issued_time_does_not_match_snapshot"
    assert admitted(request(), mask=[False]*16) == "unreachable_or_illegal"


def test_untrusted_json_ids_and_times_are_not_coerced():
    assert admitted(request(action=True)) == "invalid_action"
    assert admitted(request(action=16)) == "invalid_action"
    assert admitted(request(candidate_version=7.)) == "candidate_version_mismatch"
    assert admitted(request(issued_at_s=float("nan"))) == "invalid_issued_at_s"
    assert admitted(request(issued_at_s=True)) == "invalid_issued_at_s"
    assert admitted([]) == "request_not_object"


def test_numpy_protocol_values_are_json_serializable_and_no_nonfinite_numbers():
    import json
    value = session.json_value({"mask": np.array([True, False]), "score": np.float32(.5),
        "missing_age": float("inf"), "points": (np.array([1., 2.]),)})
    assert value == {"mask": [True, False], "score": .5, "missing_age": None, "points": [[1., 2.]]}
    json.dumps(value, allow_nan=False)
