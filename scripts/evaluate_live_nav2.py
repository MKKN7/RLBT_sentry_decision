#!/usr/bin/env python3
"""Evaluate one frozen live-nav checkpoint using actual Nav2/MPPI, without SGD."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_live_nav2 import evaluate, parse_args as training_args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=256)
    parser.add_argument("--seed", type=int, default=471)
    parser.add_argument("--simulation-seed-base", type=int)
    parser.add_argument("--goal", type=float, nargs=2, default=(0., .4))
    parser.add_argument("--goal-schedule", type=Path)
    parser.add_argument("--episode-wall-timeout", type=float, default=180.)
    parser.add_argument("--task-timeout-s", type=float, default=20.)
    parsed = parser.parse_args(argv)
    args = training_args(["--run", str(parsed.run)])
    for key, value in vars(parsed).items():
        setattr(args, key, value)
    return args


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), ensure_ascii=False, allow_nan=False), flush=True)
