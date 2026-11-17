#!/usr/bin/env python3
"""Run only a registered, development-gated strategy experiment on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_development_gate(registration):
    directory=ROOT/registration['development_gate_run']
    gate=json.loads((directory/'development_gate.json').read_text())
    manifest=json.loads((directory/'manifest.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    if not gate['gates'][registration['support_mode']]['passed'] or summary['status']!='complete' or not summary['sources_unchanged']:
        raise ValueError('The selected scenario has not passed its development gate')
    if sha(directory/'development_gate.json')!=registration['development_gate_sha256']:
        raise ValueError('Development gate differs from registration')
    for relative in ('rl_bt/strategy_scenarios.py','configs/strategy_scenarios_v2.json'):
        if manifest['implementation_sha256'][relative]!=sha(ROOT/relative):
            raise ValueError('Scene changed after accepted development calibration')
    if registration['scene_config_sha256']!=sha(ROOT/'configs/strategy_scenarios_v2.json'):
        raise ValueError('Scene configuration differs from registration')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    args=parser.parse_args()
    registration=json.loads(args.registration.read_text())
    verify_development_gate(registration)
    from scripts.train_strategy_v2 import load_strategy_checkpoint
    environment=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    suffix=registration['run_suffix']
    if not suffix.replace('_','').isalnum():
        raise ValueError('Unsafe run suffix')
    paths=[]
    for memory in registration['memory_variants']:
        for seed in registration['training_seeds']:
            label=f'{memory}_s{seed}'
            run=ROOT/'runs'/f'strategy_v2_{label}_{registration["minimum_decisions_per_run"]}_{suffix}'
            paths.append((label,run))
            expected={'memory':memory,'seed':seed,'support_mode':registration['support_mode'],
                'target_decisions':registration['minimum_decisions_per_run'],
                'train_styles':registration['train_styles'],'epochs':registration['epochs'],
                'episodes_per_update':registration['episodes_per_update'],
                'gamma_per_second':registration['gamma_per_second'],'lambda_per_second':registration['lambda_per_second'],
                'learning_rate':registration['learning_rate'],'entropy_coef':registration['entropy_coefficient'],
                'BC_updates':0,'historical_model_initialization':False,'scene_config_sha256':registration['scene_config_sha256']}
            if run.exists():
                if not (run/'summary.json').is_file():
                    raise RuntimeError(f'Existing incomplete run will not be overwritten or called exact resume: {run}')
                summary=json.loads((run/'summary.json').read_text())
                _,checkpoint=load_strategy_checkpoint(run/'final.pt')
                if summary['status']!='complete' or not summary['actual_budget_meets_target'] or checkpoint['decisions']!=summary['decisions'] or any(checkpoint['metadata'].get(k)!=v for k,v in expected.items()):
                    raise ValueError(f'Existing checkpoint differs from registration: {run}')
                print(f'Validated completed {label}',flush=True)
                continue
            command=[sys.executable,str(ROOT/'scripts/train_strategy_v2.py'),'--run',str(run),
                '--memory',memory,'--seed',str(seed),'--support-mode',registration['support_mode'],
                '--train-styles',*registration['train_styles'],'--decisions',str(registration['minimum_decisions_per_run']),
                '--episodes-per-update',str(registration['episodes_per_update']),'--epochs',str(registration['epochs']),
                '--gamma-per-second',str(registration['gamma_per_second']),'--lambda-per-second',str(registration['lambda_per_second']),
                '--learning-rate',str(registration['learning_rate']),'--entropy-coef',str(registration['entropy_coefficient']),
                '--torch-threads','2']
            print(f'Starting registered {label}',flush=True)
            subprocess.run(command,cwd=ROOT,env=environment,check=True)
    evaluation=ROOT/'runs'/f'strategy_v2_frozen_evaluation_{suffix}'
    if evaluation.exists():
        raise FileExistsError(f'Refusing to overwrite final evaluation: {evaluation}')
    command=[sys.executable,str(ROOT/'scripts/evaluate_strategy_v2.py')]
    for label,run in paths:
        command+=['--checkpoint',f'{label}={run/"final.pt"}']
    command+=['--support-mode',registration['support_mode'],'--seeds',*map(str,registration['test_seeds']),
        '--styles',*registration['test_styles'],'--sides',*map(str,registration['test_sides']),
        '--split','test','--workers','2','--out',str(evaluation)]
    subprocess.run(command,cwd=ROOT,env=environment,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/report_strategy_v2.py'),
        '--registration',str(args.registration.resolve()),'--runs',*[str(run) for _,run in paths],
        '--evaluation',str(evaluation),'--out',str(ROOT/'data/reports'/f'strategy_v2_{suffix}')],
        cwd=ROOT,env=environment,check=True)


if __name__=='__main__':
    main()
