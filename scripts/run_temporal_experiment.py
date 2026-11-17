#!/usr/bin/env python3
"""Run the registered CPU-only temporal pilot serially, then frozen evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / 'configs/temporal_strategy_experiment_20261117.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, default=REGISTRATION)
    parser.add_argument('--suffix', default='20261117')
    args = parser.parse_args()
    if not args.suffix.replace('_', '').isalnum():
        parser.error('Use a simple suffix to protect output paths')
    registration = json.loads(args.registration.read_text())
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
    sys.path.insert(0, str(ROOT))
    from scripts.train_temporal_phase import load_temporal_checkpoint
    paths = []
    for memory in registration['memory_variants']:
        for seed in registration['training_seeds']:
            label = f'{memory}_s{seed}'
            run = ROOT / 'runs' / f'temporal_{label}_{registration["minimum_decisions_per_run"]}_{args.suffix}'
            paths.append((label, run))
            if run.exists():
                summary_path = run / 'summary.json'
                if not summary_path.is_file():
                    raise RuntimeError(f'Run exists without completion; will not overwrite or pretend to resume: {run}')
                summary = json.loads(summary_path.read_text())
                _, checkpoint = load_temporal_checkpoint(run / 'final.pt')
                metadata = checkpoint['metadata']
                expected = {'memory': memory, 'seed': seed,
                    'target_decisions': registration['minimum_decisions_per_run'],
                    'episodes_per_update': registration['episodes_per_update'], 'epochs': registration['epochs'],
                    'gamma_per_second': registration['gamma_per_second'],
                    'lambda_per_second': registration['lambda_per_second'],
                    'learning_rate': registration['learning_rate'],
                    'entropy_coef': registration['entropy_coefficient'], 'reward_profile': 'upstream',
                    'randomization': True, 'BC_updates': 0, 'device': 'cpu', 'torch_threads': 2}
                if summary['status'] != 'complete' or not summary['actual_budget_meets_target'] or any(metadata.get(k) != v for k,v in expected.items()):
                    raise ValueError(f'Existing run does not match registration: {run}')
                print(f'Validated completed run: {run.name}', flush=True)
                continue
            command = [sys.executable, str(ROOT/'scripts/train_temporal_phase.py'),
                '--run', str(run), '--memory', memory, '--seed', str(seed),
                '--decisions', str(registration['minimum_decisions_per_run']),
                '--episodes-per-update', str(registration['episodes_per_update']),
                '--epochs', str(registration['epochs']),
                '--gamma-per-second', str(registration['gamma_per_second']),
                '--lambda-per-second', str(registration['lambda_per_second']),
                '--learning-rate', str(registration['learning_rate']),
                '--entropy-coef', str(registration['entropy_coefficient']), '--torch-threads', '2']
            print(f'Starting {label}', flush=True)
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
    evaluation = ROOT/'runs'/f'temporal_frozen_evaluation_{args.suffix}'
    if evaluation.exists():
        raise FileExistsError(f'Refusing to overwrite frozen evaluation: {evaluation}')
    command = [sys.executable, str(ROOT/'scripts/evaluate_temporal_phase.py')]
    for label, run in paths:
        command += ['--checkpoint', f'{label}={run / "final.pt"}']
    command += ['--gru-reset-ablation', '--seeds', *map(str,registration['test_seeds']),
        '--styles', *registration['test_styles'], '--sides', *map(str,registration['test_sides']),
        '--split', 'test', '--max-decisions', '400', '--workers', '1', '--out', str(evaluation)]
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    subprocess.run([sys.executable, str(ROOT/'scripts/report_temporal_phase.py'),
        '--runs', *[str(path) for _,path in paths], '--evaluation', str(evaluation),
        '--registration', str(args.registration.resolve()),
        '--out', str(ROOT/'data/reports'/f'temporal_strategy_{args.suffix}')],
        cwd=ROOT, env=environment, check=True)


if __name__ == '__main__':
    main()
