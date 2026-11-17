#!/usr/bin/env python3
"""Evaluate a frozen UC course checkpoint on predeclared paired seeds once."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from rl_bt.uc_support_env import UCSupportEnv, UC_SUPPORT_SCENARIO_ID, uc_support_bt_baseline
from rl_bt.uc_support_policy import UCSupportActorCritic
from rl_bt.ul_curriculum_policy import choose
from scripts.train_uc_support import (TEST_SEEDS, VALIDATION_SEEDS, dataset_hashes,
    evaluate_policy, source_hashes, summarize_evaluation, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run = args.run.resolve()
    output = run / f"evaluation_{args.split}.json"
    if output.exists():
        parser.error(f"Evaluation already recorded: {output}; reuse it instead of retesting")
    config = json.loads((run / "config.json").read_text())
    dataset = Path(config["dataset_dir"])
    if config["scenario_id"] != UC_SUPPORT_SCENARIO_ID:
        raise ValueError("Scenario differs from frozen checkpoint")
    if source_hashes() != config["implementation_sha256"]:
        raise ValueError("Current trainer/environment differs from frozen source")
    if dataset_hashes(dataset) != config["dataset_sha256"]:
        raise ValueError("Dataset differs from frozen source")
    seeds = TEST_SEEDS if args.split == "test" else VALIDATION_SEEDS
    groups, hashes = {}, {}
    for label, name in (("untrainedRL", "untrained.pt"),
            ("postBC_RL", "initial.pt"), ("finalPPO_RL", "final.pt")):
        model = UCSupportActorCritic().eval()
        payload = torch.load(run / name, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        hashes[label] = hashlib.sha256((run / name).read_bytes()).hexdigest()
        groups[label] = evaluate_policy(lambda obs: choose(model, obs), UCSupportEnv,
            dataset, split=args.split, seeds=seeds)
    groups["BT"] = evaluate_policy(uc_support_bt_baseline, UCSupportEnv,
        dataset, split=args.split, seeds=seeds)
    result = summarize_evaluation(groups, seeds, args.split, UC_SUPPORT_SCENARIO_ID)
    result.update({"checkpoint_sha256": hashes, "ppo_decisions": payload["decisions"],
        "evaluation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (run / "source" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write_json(output, result)
    print(json.dumps({"output": str(output), "summary": result["summary"],
        "paired_score_differences": result["paired_score_differences"]}, indent=2))


if __name__ == "__main__":
    main()
