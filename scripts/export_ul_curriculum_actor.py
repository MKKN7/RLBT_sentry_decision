#!/usr/bin/env python3
"""Export only the version-matched UL task actor and check CPU ORT parity."""
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
from rl_bt.ul_curriculum_policy import ULCurriculumActorCritic, SCENARIO_ID


class ExportActor(torch.nn.Module):
    """Fixed schema; ingress shape/validity checks belong in the BT adapter."""
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, state, candidates, mask):
        encoded = self.actor.state_encoder(state)
        candidate = self.actor.candidate_encoder(candidates)
        joined = torch.cat((encoded[:, None, :].expand(-1, 16, -1), candidate), -1)
        return self.actor.scorer(joined).squeeze(-1).masked_fill(~mask, -1e9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    torch.set_num_threads(2)
    checkpoint = run / "final.pt"
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if data["metadata"]["scenario_id"] != SCENARIO_ID:
        raise ValueError("Checkpoint/env version mismatch: export its frozen source instead")
    model = ULCurriculumActorCritic().eval()
    model.load_state_dict(data["model"])
    wrapper = ExportActor(model.actor).eval()
    rng = np.random.default_rng(190734)
    state = rng.normal(size=(1, 24)).astype(np.float32)
    candidates = rng.normal(size=(1, 16, 12)).astype(np.float32)
    mask = np.ones((1, 16), bool)
    output = run / "actor.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(wrapper, tuple(torch.as_tensor(x) for x in (state, candidates, mask)),
            output, input_names=["state", "candidates", "mask"], output_names=["candidate_logits"],
            opset_version=17, dynamic_axes={name: {0: "batch"} for name in
                ("state", "candidates", "mask", "candidate_logits")})
    settings = ort.SessionOptions(); settings.intra_op_num_threads = 1
    session = ort.InferenceSession(str(output), settings, providers=["CPUExecutionProvider"])
    max_error = 0.
    for n in range(1, 101):
        batch = 1+n%4
        state = rng.normal(size=(batch, 24)).astype(np.float32)
        candidates = rng.normal(size=(batch, 16, 12)).astype(np.float32)
        mask = rng.random((batch, 16)) > .35
        mask[:, 15] = True
        with torch.no_grad():
            reference = wrapper(*(torch.as_tensor(x) for x in (state, candidates, mask))).numpy()
        inferred = session.run(None, dict(state=state, candidates=candidates, mask=mask))[0]
        np.testing.assert_allclose(reference, inferred, atol=1e-5, rtol=1e-5)
        assert np.array_equal(reference.argmax(-1), inferred.argmax(-1))
        assert mask[np.arange(batch), inferred.argmax(-1)].all()
        max_error = max(max_error, float(abs(reference-inferred).max()))
    one = dict(state=state[:1], candidates=candidates[:1], mask=mask[:1])
    for _ in range(20):
        session.run(None, one)
    latency = []
    for _ in range(500):
        start = time.perf_counter_ns(); session.run(None, one)
        latency.append((time.perf_counter_ns()-start)/1e6)
    result = {"scenario_id": SCENARIO_ID, "artifact_type": "UL_course_actor_CPU_export_check",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "onnx_bytes": output.stat().st_size, "actor_parameters": data["metadata"]["actor_parameters"],
        "critic_exported": False, "random_batches_checked": 100,
        "max_abs_logit_error": max_error, "argmax_and_masks_match": True,
        "CPU_provider": session.get_providers(), "warm_latency_samples": len(latency),
        "workstation_latency_ms_p50_p95_p99": np.percentile(latency, [50, 95, 99]).tolist(),
        "NUC_hardware_benchmarked": False, "live_BT_ROS_adapter_connected": False}
    (run / "export_verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
