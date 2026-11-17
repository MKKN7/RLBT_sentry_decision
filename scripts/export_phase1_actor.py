#!/usr/bin/env python3
"""Export the small actor alone and compare with ORT CPU, including masks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import onnx
import onnxruntime as ort
import torch

from rl_bt.policy import TaskActorCritic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model = TaskActorCritic()
    model.load_state_dict(checkpoint["model"]); model.eval()
    out = args.checkpoint.parent / "export"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(2701)
    state = torch.from_numpy(rng.normal(size=(1, 148)).astype(np.float32))
    candidates = torch.from_numpy(rng.normal(size=(1, 16, 16)).astype(np.float32))
    mask = torch.ones((1, 16), dtype=torch.bool); mask[:, 3] = False
    stance_mask = torch.zeros((1, 6), dtype=torch.bool); stance_mask[:, 0] = True
    path = out / "actor.onnx"
    torch.onnx.export(model.actor, (state, candidates, mask, stance_mask), str(path),
        input_names=["state", "candidates", "mask", "stance_mask"],
        output_names=["candidate_logits", "stance_logits"], opset_version=17,
        dynamic_axes={"state": {0: "batch"}, "candidates": {0: "batch", 1: "candidates"},
                      "mask": {0: "batch", 1: "candidates"}, "stance_mask": {0: "batch"},
                      "candidate_logits": {0: "batch", 1: "candidates"},
                      "stance_logits": {0: "batch"}}, dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 1; opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    maximum = 0.
    for batch, count in ((1, 16), (3, 9)):
        sample = {"state": rng.normal(size=(batch, 148)).astype(np.float32),
            "candidates": rng.normal(size=(batch, count, 16)).astype(np.float32),
            "mask": rng.random((batch, count)) > .3,
            "stance_mask": np.tile([True, False, False, False, False, False], (batch, 1))}
        sample["mask"][:, 0] = True
        with torch.no_grad():
            expected = model.actor(*(torch.from_numpy(sample[k]) for k in sample))
        actual = session.run(None, sample)
        for a, b in zip(actual, expected):
            maximum = max(maximum, float(np.max(np.abs(a - b.numpy()))))
        assert np.all(sample["mask"][np.arange(batch), np.argmax(actual[0], axis=1)])
    if maximum > 1e-5:
        raise RuntimeError(f"ONNX mismatch {maximum}")
    latencies = []
    feeds = dict(zip(["state", "candidates", "mask", "stance_mask"],
                     [t.numpy() for t in (state, candidates, mask, stance_mask)]))
    for _ in range(220):
        start = time.perf_counter(); session.run(None, feeds)
        latencies.append((time.perf_counter()-start)*1000)
    report = {"status": "PASS", "artifact_type": "phase1_actor_export_not_hardware_acceptance",
        "checkpoint": str(args.checkpoint), "onnx_path": str(path),
        "actor_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
        "actor_parameters": sum(p.numel() for p in model.actor.parameters()),
        "max_abs_error": maximum, "verified_batch_and_candidate_shapes": [[1, 16], [3, 9]],
        "latency_ms_host_CPU_single_thread_warm": dict(zip(["p50", "p95", "p99"],
            np.quantile(latencies[20:], [.5, .95, .99]).tolist())),
        "critic_exported": False, "stance_head": "untrained; phase-I must mask all except stance 0",
        "limits": "Measured on this workstation, not NUC. Inputs still belong to simplified prototype."}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
