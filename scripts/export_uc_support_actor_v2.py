#!/usr/bin/env python3
"""Export a UC v2 actor without critic and verify CPU ONNX parity."""
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

from rl_bt.uc_support_env_v2 import UC_SUPPORT_V2_SCENARIO_ID
from rl_bt.uc_support_policy_v2 import UCSupportV2ActorCritic
from scripts.export_ul_curriculum_actor import ExportActor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    payload = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
    if payload["metadata"]["scenario_id"] != UC_SUPPORT_V2_SCENARIO_ID:
        raise ValueError("Wrong v2 checkpoint")
    torch.set_num_threads(2)
    model = UCSupportV2ActorCritic().eval()
    model.load_state_dict(payload["model"])
    wrapper = ExportActor(model.actor).eval()
    rng = np.random.default_rng(190735)
    state = rng.normal(size=(1, 24)).astype(np.float32)
    candidates = rng.normal(size=(1, 16, 12)).astype(np.float32)
    mask = np.ones((1, 16), dtype=bool)
    output = run / "actor.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(wrapper, tuple(torch.as_tensor(a) for a in (state,candidates,mask)),
            output, input_names=["state","candidates","mask"],
            output_names=["candidate_logits"], opset_version=17,
            dynamic_axes={name:{0:"batch"} for name in ("state","candidates","mask","candidate_logits")})
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    maximum = 0.
    for index in range(100):
        n = 1+index%4
        feed = {"state":rng.normal(size=(n,24)).astype(np.float32),
            "candidates":rng.normal(size=(n,16,12)).astype(np.float32),
            "mask":rng.random((n,16))>.35}
        feed["mask"][:,15] = True
        with torch.no_grad():
            reference = wrapper(*(torch.as_tensor(feed[k]) for k in ("state","candidates","mask"))).numpy()
        actual = session.run(None,feed)[0]
        np.testing.assert_allclose(reference,actual,atol=1e-5,rtol=1e-5)
        assert np.array_equal(reference.argmax(-1),actual.argmax(-1))
        assert feed["mask"][np.arange(n),actual.argmax(-1)].all()
        maximum = max(maximum,float(np.abs(reference-actual).max()))
    one = {k:v[:1] for k,v in feed.items()}
    for _ in range(20): session.run(None,one)
    timings = []
    for _ in range(500):
        start = time.perf_counter_ns()
        session.run(None,one)
        timings.append((time.perf_counter_ns()-start)/1e6)
    digest = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report = {"scenario_id":UC_SUPPORT_V2_SCENARIO_ID,"checkpoint_sha256":digest(run/"final.pt"),
        "onnx_sha256":digest(output),"onnx_bytes":output.stat().st_size,
        "actor_parameters":sum(p.numel() for p in model.actor.parameters()),
        "random_batches_checked":100,"max_abs_logit_error":maximum,
        "argmax_and_masks_match":True,"critic_exported":False,
        "workstation_latency_ms_p50_p95_p99":np.percentile(timings,[50,95,99]).tolist(),
        "NUC_hardware_benchmarked":False,"live_BT_ROS_adapter_connected":False}
    (run/"export_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
