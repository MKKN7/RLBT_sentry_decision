"""Latest-only bounded spawned terrain execution, independent of ROS."""
import time

import numpy as np
import pytest

from rl_bt.terrain_worker import TerrainWorker
from rl_bt.voxel_terrain import VoxelTerrainConfig


def config():
    return VoxelTerrainConfig(resolution_m=.1, radius_m=.7,
        self_exclusion_radius_m=0., plane_neighborhood_m=.25)


def cloud():
    x, y = np.meshgrid(np.arange(-.575, .6, .05),
                       np.arange(-.575, .6, .05), indexing="ij")
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))


def submit(worker, time_s, points=None, **kwargs):
    return worker.submit(cloud() if points is None else points,
        sampled_at_s=time_s, now_s=kwargs.pop("now_s", time_s),
        robot_position_xyz=(0., 0., 0.), frame_id=kwargs.pop("frame_id", "odom"), **kwargs)


def scan(time_s, points=None, **kwargs):
    return {"points_xyz": cloud() if points is None else points,
        "sampled_at_s": time_s, "now_s": kwargs.pop("now_s", time_s),
        "robot_position_xyz": (0., 0., 0.), "frame_id": "odom", **kwargs}


def wait_result(worker, expected_sequence=1):
    deadline = time.monotonic() + 8.
    while time.monotonic() < deadline:
        result = worker.poll()
        if result is not None and result.sequence >= expected_sequence:
            return result
        assert worker.ready_status()["running"], worker.ready_status()
        time.sleep(.005)
    pytest.fail(f"No terrain result for sequence {expected_sequence}: {worker.ready_status()}")


def test_spawn_worker_returns_real_terrain_and_nonblocking_empty_poll():
    with TerrainWorker(config()) as worker:
        started = time.perf_counter()
        assert worker.poll() is None
        assert time.perf_counter() - started < .1
        assert submit(worker, 1.) == "submitted"
        out = wait_result(worker)
        assert out.sequence == 1 and out.sampled_at_s == 1.
        assert out.update_status == "accepted" and out.error is None
        assert out.grid.diagnostics["free_cells"] > 0
        assert out.grid.diagnostics["source"] == "estimated_registered_scan_only"
        assert out.classification_wall_s > 0
        assert worker.ready_status()["start_method"] == "spawn"
        assert worker.ready_status()["queue_maxsize"] == 1
        assert worker.poll() is None


def test_source_rejection_and_one_hertz_simulation_throttle():
    with TerrainWorker(config()) as worker:
        assert submit(worker, 1.) == "submitted"
        assert submit(worker, 1.2) == "throttled"
        assert submit(worker, 1.3, frame_id="map") == "frame_mismatch"
        assert submit(worker, 2., now_s=1.5) == "future_sample"
        assert submit(worker, .1, now_s=1.) == "stale_sample"
        assert submit(worker, .9) == "out_of_order"
        assert submit(worker, 2.) == "submitted"
        out = wait_result(worker, 2)
        assert out.sampled_at_s == 2. and out.grid.frame_id == "odom"
        assert worker.ready_status()["throttled_samples"] == 1
        assert worker.ready_status()["submitted_samples"] == 2


def test_busy_queue_keeps_latest_sample_without_backlog():
    with TerrainWorker(config()) as worker:
        points = cloud()
        started = time.perf_counter()
        for sequence in range(1, 41):
            assert submit(worker, float(sequence), points) == "submitted"
        # This bound detects accidental synchronous snapshot in the submit
        # loop; classification of forty maps would substantially exceed it.
        assert time.perf_counter() - started < 1.
        out = wait_result(worker, 40)
        assert out.sequence == 40 and out.sampled_at_s == 40.
        assert out.grid.sampled_at_s == 40.
        assert worker.ready_status()["dropped_samples"] > 0
        assert out.dropped_samples > 0


def test_oversize_payload_and_invalid_pose_are_explicit_not_fake_success():
    with TerrainWorker(config(), max_input_bytes=4096) as worker:
        assert submit(worker, 1., np.zeros((500, 3))) == "message_too_large"
        assert submit(worker, 1., np.zeros((5, 2))) == "invalid_cloud_shape"
        assert worker.submit(cloud(), sampled_at_s=1., now_s=1.,
            robot_position_xyz=(np.nan, 0., 0.), frame_id="odom") == "invalid_estimated_pose"
        assert worker.ready_status()["submitted_samples"] == 0
        assert worker.poll() is None


def test_close_is_bounded_idempotent_and_worker_exit_is_explicit():
    worker = TerrainWorker(config())
    assert submit(worker, 1.) == "submitted"
    started = time.monotonic()
    worker.close(timeout_s=.05)
    assert time.monotonic() - started < .8
    assert worker.ready_status()["closed"] is True
    assert worker.ready_status()["running"] is False
    assert submit(worker, 2.) == "closed"
    worker.close()
    assert worker.poll() is None
    with TerrainWorker(config()) as killed:
        killed._process.terminate()
        killed._process.join(timeout=2.)
        assert submit(killed, 1.) == "worker_not_running"
        assert killed.ready_status()["error"].startswith("worker_exited_with_code_")


def test_worker_reports_output_capacity_failure_without_empty_free_grid():
    # A real classified map exceeds the deliberately tiny output capacity.
    large = VoxelTerrainConfig(resolution_m=.05, radius_m=1., self_exclusion_radius_m=0.)
    with TerrainWorker(large, max_output_bytes=4096) as worker:
        assert submit(worker, 1.) == "submitted"
        out = wait_result(worker)
        assert out.update_status == "worker_error"
        assert out.grid is None
        assert "message_too_large" in out.error
        assert worker.ready_status()["error"]


def test_batch_retains_all_scan_observations_and_original_voxel_times():
    # Both observations arrive fresh. They are 2.5 s apart, so interpreting
    # the first against the final receive clock would incorrectly reject it.
    old = np.array([[-.35, .05, .15]])
    recent = np.array([[.35, .05, .15]])
    with TerrainWorker(config()) as worker:
        assert worker.submit_batch([scan(1., old), scan(3.5, recent)]) == "submitted"
        out = wait_result(worker)
        assert out.scan_count == 2
        assert out.sampled_at_s == 3.5 and out.submitted_now_s == 3.5
        assert out.grid.sampled_at_s == 3.5
        assert out.update_status == "accepted" and out.error is None
        assert len(out.grid.obstacle_points_xyz) == 2
        # A new empty observation must expire only the voxel from the first
        # scan, proving the batch did not relabel all voxel times as 3.5 s.
        assert worker.submit_batch([scan(4.6, np.empty((0, 3)))]) == "submitted"
        later = wait_result(worker, 2)
        assert later.scan_count == 1
        np.testing.assert_allclose(later.grid.obstacle_points_xyz, recent, atol=1e-7)
        assert worker.ready_status()["submitted_scans"] == 3
        assert worker.ready_status()["submitted_batches"] == 2


def test_batch_throttle_uses_final_sample_and_preserves_final_pose():
    with TerrainWorker(config()) as worker:
        assert submit(worker, 1.) == "submitted"
        moving = scan(2.1, robot_position_xyz=(.1, .2, 0.))
        assert worker.submit_batch([scan(1.2), moving]) == "submitted"
        out = wait_result(worker, 2)
        assert out.scan_count == 2 and out.sampled_at_s == 2.1
        expected_origin = np.floor((np.array([.1, .2]) - .7) / .1) * .1
        np.testing.assert_allclose(out.grid.origin_xy_m, expected_origin)
        assert worker.submit_batch([scan(2.3), scan(2.6)]) == "throttled"
        assert worker.ready_status()["throttled_samples"] == 1


@pytest.mark.parametrize("changes,expected", [
    ({"frame_id": "map"}, "frame_mismatch"),
    ({"sampled_at_s": np.nan}, "nonfinite_timestamp"),
    ({"now_s": .5}, "future_sample"),
    ({"now_s": 1.6}, "stale_sample"),
    ({"points_xyz": np.zeros((3, 2))}, "invalid_cloud_shape"),
    ({"robot_position_xyz": (0., np.nan, 0.)}, "invalid_estimated_pose"),
])
def test_bad_frame_anywhere_rejects_entire_batch_without_updating_state(changes, expected):
    with TerrainWorker(config()) as worker:
        first = scan(1.)
        first.update(changes)
        assert worker.submit_batch([first, scan(2.)]) == expected
        assert worker.ready_status()["submitted_batches"] == 0
        # Validation failure cannot advance the parent time or enqueue a
        # partially valid scan; the same complete valid interval still works.
        assert worker.submit_batch([scan(1.), scan(2.)]) == "submitted"
        assert wait_result(worker).scan_count == 2


@pytest.mark.parametrize("scans,expected", [
    ([], "empty_batch"),
    ({}, "invalid_batch"),
    ([{}], "invalid_scan_fields"),
    ([scan(2.), scan(1.)], "out_of_order"),
    ([scan(1.), scan(1.)], "out_of_order"),
    ([scan(1., now_s=1.4), scan(1.2)], "out_of_order"),
    ([scan(1.), scan(4.1)], "stale_batch"),
])
def test_invalid_batch_order_and_retention_are_explicit(scans, expected):
    with TerrainWorker(config()) as worker:
        assert worker.submit_batch(scans) == expected
        assert worker.ready_status()["submitted_scans"] == 0


def test_batch_size_limit_applies_to_the_whole_interval():
    with TerrainWorker(config(), max_input_bytes=4096) as worker:
        points = np.zeros((180, 3))
        assert worker.submit_batch([scan(1., points), scan(1.1, points)]) == "message_too_large"
        assert worker.ready_status()["submitted_batches"] == 0
        assert worker.submit_batch([scan(1., points)]) == "submitted"
        assert wait_result(worker).scan_count == 1


def test_busy_latest_slot_replaces_whole_batches_without_backlog():
    with TerrainWorker(config()) as worker:
        points = np.array([[.35, .05, .15]])
        started = time.perf_counter()
        for sequence in range(1, 41):
            assert worker.submit_batch([scan(float(sequence), points),
                scan(sequence + .2, points)]) == "submitted"
        assert time.perf_counter() - started < 1.
        out = wait_result(worker, 40)
        assert out.sequence == 40 and out.scan_count == 2
        assert out.sampled_at_s == 40.2
        assert worker.ready_status()["dropped_batches"] > 0
        assert worker.ready_status()["submitted_scans"] == 80
