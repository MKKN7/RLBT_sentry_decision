#!/usr/bin/env python3
"""Export the independent UC support-course actor; check CPU ONNX parity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import onnxruntime as ort
import torch

from rl_bt.uc_support_env import UC_SUPPORT_SCENARIO_ID
from rl_bt.uc_support_policy import UCSupportActorCritic
from scripts.export_ul_curriculum_actor import ExportActor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    checkpoint = run / "final.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["metadata"]["scenario_id"] != UC_SUPPORT_SCENARIO_ID:
        raise ValueError("UC scenario mismatch: use the checkpoint's frozen source")
    torch.set_num_threads(2)
    model = UCSupportActorCritic().eval()
    model.load_state_dict(payload["model"])
    wrapper = ExportActor(model.actor).eval()
    rng = np.random.default_rng(190735)
    state = rng.normal(size=(1, 24)).astype(np.float32)
    candidates = rng.normal(size=(1, 16, 12)).astype(np.float32)
    mask = np.ones((1, 16), dtype=bool)
    output = run / "actor.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(wrapper, tuple(torch.as_tensor(x) for x in (state, candidates, mask)),
            output, input_names=["state", "candidates", "mask"], output_names=["candidate_logits"],
            opset_version=17, dynamic_axes={name: {0: "batch"} for name in
                ("state", "candidates", "mask", "candidate_logits")})
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    maximum = 0.
    for sample in range(100):
        count = 1 + sample % 4
        state = rng.normal(size=(count, 24)).astype(np.float32)
        candidates = rng.normal(size=(count, 16, 12)).astype(np.float32)
        mask = rng.random((count, 16)) > .35
        mask[:, 15] = True
        with torch.no_grad():
            reference = wrapper(*(torch.as_tensor(x) for x in (state, candidates, mask))).numpy()
        inferred = session.run(None, dict(state=state, candidates=candidates, mask=mask))[0]
        np.testing.assert_allclose(reference, inferred, rtol=1e-5, atol=1e-5)
        if not np.array_equal(reference.argmax(-1), inferred.argmax(-1)):
            raise AssertionError("UC ONNX argmax differs")
        if not mask[np.arange(count), inferred.argmax(-1)].all():
            raise AssertionError("UC ONNX selected a masked action")
        maximum = max(maximum, float(np.abs(reference - inferred).max()))
    one = dict(state=state[:1], candidates=candidates[:1], mask=mask[:1])
    for _ in range(20):
        session.run(None, one)
    timings = []
    for _ in range(500):
        start = time.perf_counter_ns()
        session.run(None, one)
        timings.append((time.perf_counter_ns() - start) / 1e6)
    report = {"scenario_id": UC_SUPPORT_SCENARIO_ID,
        "artifact_type": "UC_support_course_actor_CPU_export_check",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "onnx_bytes": output.stat().st_size,
        "actor_parameters": sum(p.numel() for p in model.actor.parameters()),
        "critic_exported": False, "random_batches_checked": 100,
        "max_abs_logit_error": maximum, "argmax_and_masks_match": True,
        "CPU_provider": session.get_providers(), "warm_latency_samples": 500,
        "workstation_latency_ms_p50_p95_p99": np.percentile(timings, [50, 95, 99]).tolist(),
        "NUC_hardware_benchmarked": False, "live_BT_ROS_adapter_connected": False,
        "metric_is_match_win_rate": False}
    (run / "export_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
