#!/usr/bin/env python3
"""Frozen evaluation in the independently versioned role-balanced scenario."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from rl_bt.alpharm_task_env import AlphaRuntime, TRAIN_STYLES, HELD_OUT_STYLES
from rl_bt.strategy_scenarios import RoleBalancedTaskEnv, SUPPORT_MODES
from scripts.audit_temporal_policy_leverage import run_control_trial
from scripts.evaluate_temporal_phase import make_cases, model_weights_sha256, run_trial, summarize
from scripts.train_strategy_v2 import load_strategy_checkpoint


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def evaluate(args):
    torch.set_num_threads(2)
    cases = make_cases(args.seeds, args.styles, args.sides)
    specs = [{'label':'continue_only','kind':'fixed'}, {'label':'fixed_center','kind':'fixed'},
             {'label':'bt_reactive','kind':'bt_reactive'}, {'label':'bt_memory','kind':'bt_memory'}]
    checkpoints = []
    for item in args.checkpoint:
        label, separator, raw = item.partition('=')
        if not separator or not label.replace('_','').replace('-','').isalnum():
            raise ValueError('Use simple LABEL=PATH checkpoint arguments')
        path = Path(raw).resolve()
        model, payload = load_strategy_checkpoint(path)
        if payload['metadata']['support_mode'] != args.support_mode:
            raise ValueError('Evaluation support mode differs from training')
        model.eval()
        specs.append({'label':label, 'kind':'learned', 'model':model})
        checkpoints.append({'label':label, 'path':str(path), 'sha256':sha(path),
            'memory':payload['metadata']['memory'], 'training_seed':payload['metadata']['seed'],
            'training_decisions':payload['decisions'], 'weights_before_sha256':model_weights_sha256(model)})
        if model.memory == 'gru':
            specs.append({'label':label+'_reset', 'kind':'learned', 'model':model,'reset_each_step':True})
    labels = [row['label'] for row in specs]
    if len(set(labels)) != len(labels):
        raise ValueError('Duplicate policy identities')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    runtime = AlphaRuntime()
    factory = partial(RoleBalancedTaskEnv, support_mode=args.support_mode)
    manifest = {'scenario_id':'alpharm_role_balanced_temporal_v2', 'environment_scenario_id':'alpharm_role_balanced_v2',
        'evaluation_split':args.split, 'cases':cases, 'policies':labels, 'support_mode':args.support_mode,
        'checkpoints':checkpoints, 'expected_episodes':len(cases)*len(specs), 'workers':args.workers,
        'max_decisions_per_episode':args.max_decisions, 'SGD_steps':0, 'randomization':True,
        'real_nav2_in_loop':False, 'official_UL_UC_training':False, 'upstream_commit':runtime.source_commit,
        'action_selection':'frozen_greedy_same_executor_and_mask',
        'observation_note':'Retains upstream HUD including aggregate enemy HP/alive; not certified official legal telemetry.',
        'implementation_sha256':{relative:sha(ROOT/relative) for relative in
            ('scripts/evaluate_strategy_v2.py','scripts/train_strategy_v2.py','rl_bt/strategy_scenarios.py',
             'configs/strategy_scenarios_v2.json','scripts/evaluate_temporal_phase.py','scripts/audit_temporal_policy_leverage.py',
             'rl_bt/temporal_baselines.py','rl_bt/alpharm_task_env.py')},
        'limits':['Simplified planar task benchmark, no actual Nav2 or full UL/UC rules/economy.',
                  'Common exogenous seeds, not head-to-head policies or identical policy-dependent trajectories.',
                  'Only two independent test seeds; no checkpoint selection or tuning on these results.']}
    write_json(out/'manifest.json',manifest)
    trials = [(index,spec,case) for index,case in enumerate(cases) for spec in specs]
    def execute(trial):
        index,spec,case = trial
        if spec['kind'] == 'fixed':
            result = run_control_trial(runtime,spec['label'],case,max_decisions=args.max_decisions,env_factory=factory)
            # A fixed control was registered before this experiment; the v1
            # helper's posthoc labels do not describe this new test design.
            result.update(exploratory_posthoc=False, independent_new_test=args.split=='test')
            trace = None
        else:
            result,trace = run_trial(runtime,spec,case,max_decisions=args.max_decisions,
                save_trace=index==0,env_factory=factory)
        result['support_mode'] = args.support_mode
        return result,trace
    started, results = time.perf_counter(), []
    write_json(out/'status.json',{'status':'running','completed_episodes':0,'expected_episodes':len(trials)})
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool, (out/'episodes.jsonl').open('w') as handle:
            for result,trace in pool.map(execute,trials):
                results.append(result)
                handle.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
                handle.flush()
                if trace is not None:
                    np.savez_compressed(out/f"{result['policy']}_seed{result['seed']}_{result['style']}_side{result['side']}.npz",**trace)
                status={'status':'running','completed_episodes':len(results),'expected_episodes':len(trials),
                    'policy':result['policy'],'wall_s':time.perf_counter()-started}
                write_json(out/'status.json',status)
                print(json.dumps(status),flush=True)
        for entry in checkpoints:
            model=next(spec['model'] for spec in specs if spec['label']==entry['label'])
            entry['weights_after_sha256']=model_weights_sha256(model)
            entry['weights_unchanged']=entry['weights_after_sha256']==entry['weights_before_sha256']
            if not entry['weights_unchanged'] or sha(entry['path'])!=entry['sha256']:
                raise RuntimeError('Frozen checkpoint changed during evaluation')
        if any(not row['completed'] or row['truncated'] for row in results):
            raise RuntimeError('Unexpected truncated evaluation; no invented draws')
        summary={**manifest,**summarize(results,labels),'episodes':results,'actual_episodes':len(results),
            'status':'complete','unexpected_truncations':0,'wall_s':time.perf_counter()-started}
        write_json(out/'summary.json',summary)
        write_json(out/'status.json',{'status':'complete','completed_episodes':len(results),
            'expected_episodes':len(trials),'unexpected_truncations':0})
        return summary
    except BaseException as error:
        write_json(out/'status.json',{'status':'error','completed_episodes':len(results),'error':repr(error)})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',action='append',default=[])
    parser.add_argument('--support-mode',choices=SUPPORT_MODES,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[73001,73002])
    parser.add_argument('--styles',nargs='+',choices=TRAIN_STYLES+HELD_OUT_STYLES,default=['pretrained','pursuit'])
    parser.add_argument('--sides',type=int,nargs='+',default=[0,1])
    parser.add_argument('--split',choices=('validation','test'),default='test')
    parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    parser.add_argument('--max-decisions',type=int,default=400)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.max_decisions<1:parser.error('Decision bound must be positive')
    evaluate(args)


if __name__=='__main__':
    main()
