"""ROS-free, latest-only spawned worker for observed voxel terrain.

The two multiprocessing queues carry wake-up tokens only and each has size 1.
Cloud/result payloads occupy one bounded shared-memory slot in each direction.
Overwriting a busy slot therefore replaces the pending scan batch rather than
building a queue.  Locks cover memory copies only, never voxel classification.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import multiprocessing as mp
import pickle
from queue import Empty, Full
import time
import traceback

import numpy as np

from .voxel_terrain import TerrainGrid, VoxelTerrainConfig, VoxelTerrainMap


@dataclass(frozen=True)
class TerrainWorkerResult:
    sequence: int
    sampled_at_s: float
    submitted_now_s: float
    update_status: str
    grid: TerrainGrid | None
    classification_wall_s: float
    dropped_samples: int
    error: str | None = None
    scan_count: int = 1


class _LatestSlot:
    """One shared byte payload; queue tokens never contain arrays or history."""

    def __init__(self, context, capacity):
        self.capacity = capacity
        self.buffer = context.RawArray("B", capacity)
        self.length = context.RawValue("Q", 0)
        self.sequence = context.RawValue("Q", 0)
        self.consumed = context.RawValue("Q", 0)
        self.overwritten = context.RawValue("Q", 0)
        self.lock = context.Lock()

    def write(self, payload, sequence):
        if len(payload) > self.capacity:
            raise ValueError("message_too_large")
        with self.lock:
            if self.sequence.value > self.consumed.value:
                self.overwritten.value += 1
            memoryview(self.buffer).cast("B")[:len(payload)] = payload
            self.length.value = len(payload)
            self.sequence.value = sequence

    def read(self, *, blocking=True):
        if not self.lock.acquire(blocking):
            return False
        try:
            sequence = int(self.sequence.value)
            if not sequence or sequence <= self.consumed.value:
                return None
            payload = memoryview(self.buffer).cast("B")[:self.length.value].tobytes()
            self.consumed.value = sequence
        finally:
            self.lock.release()
        return sequence, payload

    def statistics(self):
        # Diagnostics must remain readable even if an explicitly terminated
        # worker died while copying a slot. These counters are advisory only.
        acquired = self.lock.acquire(False)
        try:
            return {"sequence": int(self.sequence.value),
                "consumed_sequence": int(self.consumed.value),
                "overwritten": int(self.overwritten.value)}
        finally:
            if acquired:
                self.lock.release()


def _wake(queue):
    try:
        queue.put_nowait(True)
    except Full:
        # A pending token already requests reading the latest shared slot.
        pass


def _worker_entry(config, frame_id, input_slot, output_slot, input_queue,
                  output_queue, ready_event, stop_event):
    """Importable spawn target.  It owns no ROS node or external process."""
    try:
        terrain = VoxelTerrainMap(config, frame_id=frame_id)
        ready_event.set()
        while not stop_event.is_set():
            try:
                input_queue.get(timeout=.1)
            except Empty:
                continue
            if stop_event.is_set():
                break
            item = input_slot.read()
            if item is None:
                continue
            sequence, payload = item
            started = time.perf_counter()
            sampled, now = float("nan"), float("nan")
            scans = []
            try:
                request = pickle.loads(payload)
                scans = request["scans"]
                sampled, now = scans[-1]["sampled_at_s"], scans[-1]["now_s"]
                status = "accepted"
                for scan in scans:
                    # Preserve the time of observation AND its receive clock.
                    # Reusing the batch-end clock would reject older valid
                    # scans or falsely extend the lifetime of their voxels.
                    status = terrain.update(**scan)
                    if status != "accepted":
                        break
                grid = (terrain.snapshot(robot_position_xyz=scans[-1]["robot_position_xyz"], now_s=now)
                        if status == "accepted" else None)
                error = None
            except Exception:
                status, grid = "worker_error", None
                error = traceback.format_exc(limit=4)
            result = TerrainWorkerResult(sequence, sampled, now, status, grid,
                time.perf_counter() - started, input_slot.statistics()["overwritten"], error,
                len(scans))
            try:
                encoded = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
                output_slot.write(encoded, sequence)
            except Exception:
                # Report a bounded error if e.g. a configured result slot is too
                # small. Never replace failed classification with a free map.
                result = TerrainWorkerResult(sequence, sampled, now, "worker_error", None,
                    time.perf_counter() - started, input_slot.statistics()["overwritten"],
                    traceback.format_exc(limit=4), len(scans))
                output_slot.write(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL), sequence)
            _wake(output_queue)
    finally:
        # Do not wait for a queue feeder while the caller is closing its peer.
        for queue in (input_queue, output_queue):
            queue.cancel_join_thread()


class TerrainWorker:
    """Start one spawn worker and submit at most once per simulation second.

    ``submit_batch`` retains every scan in a bounded interval, while ``submit``
    remains the single-scan compatibility API. Each returns an explicit string
    status and copies data into one shared
    slot; it never runs ``VoxelTerrainMap.snapshot``. ``poll`` is nonblocking and
    returns only a new latest result, or None.  The caller must additionally
    check result sample age against its *current* ROS simulation clock before
    using it: classification can finish after its input has become stale.

    Use from an importable, ``if __name__ == '__main__'`` guarded entry point.
    ``close`` signals/terminates only the process this object created.
    """

    def __init__(self, config=None, *, frame_id="odom", submit_period_s=1.,
                 max_input_bytes=8 * 1024 * 1024, max_output_bytes=8 * 1024 * 1024):
        self.config = VoxelTerrainConfig() if config is None else config
        if not isinstance(self.config, VoxelTerrainConfig) or not frame_id:
            raise ValueError("VoxelTerrainConfig and nonempty frame required")
        if not math.isfinite(submit_period_s) or submit_period_s < 1.:
            raise ValueError("Submission period must be at least one simulation second")
        if (not isinstance(max_input_bytes, int) or not isinstance(max_output_bytes, int)
                or min(max_input_bytes, max_output_bytes) < 4096):
            raise ValueError("IPC byte capacities must be integers of at least 4096")
        self.frame_id = str(frame_id)
        self.submit_period_s = float(submit_period_s)
        self._context = mp.get_context("spawn")
        self._input_slot = _LatestSlot(self._context, max_input_bytes)
        self._output_slot = _LatestSlot(self._context, max_output_bytes)
        self._input_queue = self._context.Queue(maxsize=1)
        self._output_queue = self._context.Queue(maxsize=1)
        self._ready_event = self._context.Event()
        self._stop_event = self._context.Event()
        self._sequence = 0
        self._submitted_scan_count = 0
        self._last_batch_scan_count = 0
        self._last_sample_s = None
        self._last_received_now_s = None
        self._last_submit_status = "not_submitted"
        self._throttled = 0
        self._closed = False
        self._last_error = None
        self._output_notification_pending = False
        self._process = self._context.Process(target=_worker_entry,
            args=(self.config, self.frame_id, self._input_slot, self._output_slot,
                self._input_queue, self._output_queue, self._ready_event, self._stop_event),
            name="rl_bt_voxel_terrain", daemon=True)
        self._process.start()

    @property
    def ready(self):
        return not self._closed and self._ready_event.is_set() and self._process.is_alive()

    def ready_status(self):
        alive = self._process.is_alive()
        if not alive and not self._closed and self._last_error is None:
            self._last_error = f"worker_exited_with_code_{self._process.exitcode}"
        incoming, outgoing = self._input_slot.statistics(), self._output_slot.statistics()
        return {"ready": self.ready, "running": alive, "closed": self._closed,
            "pid": self._process.pid, "start_method": "spawn", "queue_maxsize": 1,
            "submitted_samples": self._sequence, "dropped_samples": incoming["overwritten"],
            "submitted_batches": self._sequence, "submitted_scans": self._submitted_scan_count,
            "last_batch_scan_count": self._last_batch_scan_count,
            "dropped_batches": incoming["overwritten"],
            "consumed_samples_through_sequence": incoming["consumed_sequence"],
            "overwritten_results": outgoing["overwritten"],
            "throttled_samples": self._throttled, "last_submit_status": self._last_submit_status,
            "error": self._last_error}

    def submit(self, points_xyz, *, sampled_at_s, now_s, robot_position_xyz, frame_id):
        return self.submit_batch([{"points_xyz": points_xyz, "sampled_at_s": sampled_at_s,
            "now_s": now_s, "robot_position_xyz": robot_position_xyz, "frame_id": frame_id}])

    def submit_batch(self, scans):
        """Submit ordered scans once per simulation second, classifying once.

        Every scan must be fresh at its original receive clock. The entire
        interval must fit within voxel retention; the final sample controls
        throttling and consumer freshness. Validation is atomic: an invalid
        frame anywhere rejects the whole batch before shared state changes.
        The 8 MiB default input slot bounds the complete serialized batch.
        """
        def reject(reason):
            self._last_submit_status = reason
            return reason
        if self._closed:
            return reject("closed")
        if not self._process.is_alive():
            self._last_error = f"worker_exited_with_code_{self._process.exitcode}"
            return reject("worker_not_running")
        if not isinstance(scans, (list, tuple)):
            return reject("invalid_batch")
        if not scans:
            return reject("empty_batch")
        fields = {"points_xyz", "sampled_at_s", "now_s", "robot_position_xyz", "frame_id"}
        normalized = []
        previous_sample, previous_now = self._last_sample_s, self._last_received_now_s
        for scan in scans:
            if not isinstance(scan, dict) or set(scan) != fields:
                return reject("invalid_scan_fields")
            if scan["frame_id"] != self.frame_id:
                return reject("frame_mismatch")
            try:
                sampled, now = float(scan["sampled_at_s"]), float(scan["now_s"])
            except (TypeError, ValueError, OverflowError):
                return reject("nonfinite_timestamp")
            if not np.isfinite([sampled, now]).all():
                return reject("nonfinite_timestamp")
            if sampled > now + 1e-6:
                return reject("future_sample")
            if now - sampled > self.config.max_sample_age_s:
                return reject("stale_sample")
            if ((previous_sample is not None and sampled <= previous_sample)
                    or (previous_now is not None and now < previous_now)):
                return reject("out_of_order")
            try:
                points = np.asarray(scan["points_xyz"], dtype=np.float32)
            except (TypeError, ValueError, OverflowError):
                return reject("invalid_cloud_shape")
            if points.ndim != 2 or points.shape[1:] != (3,):
                return reject("invalid_cloud_shape")
            try:
                pose = np.asarray(scan["robot_position_xyz"], dtype=float)
            except (TypeError, ValueError, OverflowError):
                return reject("invalid_estimated_pose")
            if pose.shape != (3,) or not np.isfinite(pose).all():
                return reject("invalid_estimated_pose")
            normalized.append({"points_xyz": points, "sampled_at_s": sampled,
                "now_s": now, "robot_position_xyz": pose, "frame_id": self.frame_id})
            previous_sample, previous_now = sampled, now
        sampled_at_s, now_s = normalized[-1]["sampled_at_s"], normalized[-1]["now_s"]
        if now_s - normalized[0]["sampled_at_s"] > self.config.max_age_s:
            return reject("stale_batch")
        if self._last_sample_s is not None:
            delta = sampled_at_s - self._last_sample_s
            if delta < self.submit_period_s - 1e-9:
                self._throttled += 1
                return reject("throttled")
        payload = pickle.dumps({"scans": normalized},
            protocol=pickle.HIGHEST_PROTOCOL)
        sequence = self._sequence + 1
        try:
            self._input_slot.write(payload, sequence)
        except ValueError as error:
            return reject(str(error))
        self._sequence = sequence
        self._last_sample_s = float(sampled_at_s)
        self._last_received_now_s = float(now_s)
        self._submitted_scan_count += len(normalized)
        self._last_batch_scan_count = len(normalized)
        self._last_submit_status = "submitted"
        _wake(self._input_queue)
        return "submitted"

    def poll(self):
        if self._closed:
            return None
        available = False
        while True:
            try:
                self._output_queue.get_nowait()
                available = True
            except Empty:
                break
        self._output_notification_pending |= available
        if not self._output_notification_pending:
            return None
        item = self._output_slot.read(blocking=False)
        if item is False:
            return None
        self._output_notification_pending = False
        if item is None:
            return None
        _, payload = item
        try:
            result = pickle.loads(payload)
            if not isinstance(result, TerrainWorkerResult):
                raise ValueError("unexpected_worker_result_type")
        except Exception:
            self._last_error = traceback.format_exc(limit=3)
            return TerrainWorkerResult(0, float("nan"), float("nan"), "worker_error",
                None, 0., self._input_slot.statistics()["overwritten"], self._last_error)
        if result.error:
            self._last_error = result.error
        return result

    def close(self, timeout_s=1.):
        if self._closed:
            return
        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("Nonnegative finite close timeout required")
        self._closed = True
        self._stop_event.set()
        _wake(self._input_queue)
        deadline = time.monotonic() + timeout_s
        self._process.join(timeout=min(.2, timeout_s))
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=max(0., deadline - time.monotonic()))
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=.1)
        for queue in (self._input_queue, self._output_queue):
            queue.cancel_join_thread()
            queue.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
