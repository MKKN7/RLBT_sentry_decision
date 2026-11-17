#!/usr/bin/env python3
"""Once-only paired evaluation of a fixed UC v2 final checkpoint."""
import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from scripts.train_uc_support_v2 import (TEST_SEEDS, VALIDATION_SEEDS,
    dataset_hashes, make_model, records_sha256, run_evaluation, sha256_file,
    source_snapshot, write_json)
from rl_bt.uc_support_env_v2 import UC_SUPPORT_V2_SCENARIO_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    args = parser.parse_args()
    run = args.run.resolve()
    output = run / f"evaluation_{args.split}.json"
    if output.exists():
        parser.error(f"Evaluation already recorded: {output}")
    checkpoint = run / "final.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    if metadata["scenario_id"] != UC_SUPPORT_V2_SCENARIO_ID:
        raise ValueError("Wrong v2 scenario")
    if metadata["implementation_sha256"] != source_snapshot():
        raise ValueError("Current implementation differs from frozen checkpoint")
    args.dataset_dir = Path(metadata["dataset_dir"])
    if metadata["dataset_sha256"] != dataset_hashes(args.dataset_dir):
        raise ValueError("Dataset changed")
    if metadata["bc_labels_sha256"] != records_sha256(payload["bc_records"]):
        raise ValueError("BC records changed")
    torch.set_num_threads(2)
    model, teacher = make_model()
    model.load_state_dict(payload["model"])
    teacher.actor.load_state_dict(payload["teacher_actor"])
    seeds = TEST_SEEDS if args.split == "test" else VALIDATION_SEEDS
    result = run_evaluation(run, model.eval(), teacher.eval(), payload["bc_records"],
        args, split=args.split, seeds=seeds)
    result.update({"checkpoint_sha256": sha256_file(checkpoint),
        "ppo_decisions": payload["decisions"], "algorithm": metadata["algorithm"],
        "training_seed": metadata["seed"], "frozen_final_checkpoint": True,
        "evaluation_script_sha256": sha256_file(__file__)})
    (run / "source" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
