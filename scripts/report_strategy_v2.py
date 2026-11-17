#!/usr/bin/env python3
"""Validate and report frozen role-balanced strategy experiments, without rollout.

Only recorded training/evaluation artifacts are consumed. Diagnostic inference
on identical saved observations never creates an environment or optimizer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_strategy_v2 import (SCENARIO_ID, ENVIRONMENT_SCENARIO_ID,
    SCENE_CONFIG, load_strategy_checkpoint, observation_tensors, file_sha256)
from scripts.evaluate_temporal_phase import summarize, model_weights_sha256


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def project_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def verify_source_copies(run, config):
    for relative, digest in config['implementation_sha256'].items():
        copied = run / 'source' / relative
        if not copied.is_file() or file_sha256(copied) != digest:
            raise ValueError(f'Training copied source is missing or changed: {copied}')


def gather_training(run):
    run = project_path(run)
    config, summary = read_json(run/'config.json'), read_json(run/'summary.json')
    episodes, updates = read_jsonl(run/'episodes.jsonl'), read_jsonl(run/'updates.jsonl')
    verify_source_copies(run, config)
    initial, initial_payload = load_strategy_checkpoint(run/'initial.pt')
    final, final_payload = load_strategy_checkpoint(run/'final.pt')
    if initial_payload['metadata'] != config or final_payload['metadata'] != config:
        raise ValueError('Checkpoint metadata disagrees with run config')
    if final_payload['decisions'] != summary['decisions'] or final_payload['updates'] != summary['updates']:
        raise ValueError('Final checkpoint counters disagree with summary')
    if initial_payload['decisions'] != 0 or initial_payload['updates'] != 0:
        raise ValueError('Initial checkpoint is not fresh initialization')
    if not updates or not episodes:
        raise ValueError('Training contains no recorded episodes or updates')
    distances = {}
    for name in ('actor', 'critic'):
        left, right = getattr(initial,name).state_dict(), getattr(final,name).state_dict()
        distances[name+'_parameter_change_l2'] = float(np.sqrt(sum(
            float((right[key].double()-value.double()).square().sum()) for key,value in left.items())))
    durations = np.asarray([row['simulated_s'] for row in episodes], float)
    evidence = {**distances,
        'actor_parameters':sum(p.numel() for p in final.actor.parameters()),
        'critic_parameters':sum(p.numel() for p in final.critic.parameters()),
        'total_parameters':sum(p.numel() for p in final.parameters()),
        'unique_actor_decisions':sum(row['actor_decisions'] for row in updates),
        'forced_single_legal_decisions':sum(row['forced_single_legal_decisions'] for row in updates),
        'actor_optimizer_steps':sum(row['epochs'] for row in updates if row['actor_decisions'] > 0),
        'maximum_first_epoch_approx_kl':max(abs(row['first_epoch_approx_kl']) for row in updates),
        'advantage_normalization_modes':dict(Counter(row['advantage_normalization_mode'] for row in updates)),
        'episodes_by_style':dict(Counter(row['opponent_style'] for row in episodes)),
        'episodes_by_side':dict(Counter(str(row['team']) for row in episodes)),
        'actual_simulated_s':float(durations.sum()),
        'duration_range_s':[float(durations.min()),float(durations.max())],
        'duration_median_s':float(np.median(durations)),
        'fraction_episodes_at_least_90s':float(np.mean(durations >= 90.))}
    return {'run':str(run), 'config':config, 'summary':summary, 'episodes':episodes,
        'updates':updates, 'learning_evidence':evidence,
        'source_sha256':{name:file_sha256(run/name) for name in
            ('config.json','summary.json','episodes.jsonl','updates.jsonl','initial.pt','final.pt')}}


def validate_registered_inputs(registration, training, evaluation, episodes, *, gate):
    """Validate identities/accounting before publishing any output files."""
    if registration.get('scenario_id') != SCENARIO_ID:
        raise ValueError('Registration scenario is not strategy v2')
    mode = registration['support_mode']
    selected = gate.get('gates',{}).get(registration.get('development_gate_key',mode),{})
    if selected.get('passed') is not True or not selected.get('criteria') or \
            not all(value is True for value in selected['criteria'].values()):
        raise ValueError('Registered development control gate did not pass')
    expected_training = set(itertools.product(registration['memory_variants'],registration['training_seeds']))
    identities = [(row['config']['memory'],row['config']['seed']) for row in training]
    if len(identities) != len(expected_training) or set(identities) != expected_training:
        raise ValueError('Missing, duplicate or unregistered training identity')
    expected_config = {'scenario_id':SCENARIO_ID, 'environment_scenario_id':ENVIRONMENT_SCENARIO_ID,
        'scene_config_sha256':registration['scene_config_sha256'], 'support_mode':mode,
        'train_styles':registration['train_styles'], 'episodes_per_update':registration['episodes_per_update'],
        'epochs':registration['epochs'], 'gamma_per_second':registration['gamma_per_second'],
        'lambda_per_second':registration['lambda_per_second'], 'learning_rate':registration['learning_rate'],
        'entropy_coef':registration['entropy_coefficient'], 'target_decisions':registration['minimum_decisions_per_run'],
        'BC_updates':0, 'historical_model_initialization':False, 'randomization':True,
        'reward_profile':'upstream', 'real_nav2_in_loop':False, 'official_UL_UC_training':False,
        'seed_role':'train', 'device':'cpu'}
    for row in training:
        config, summary, records, updates = row['config'],row['summary'],row['episodes'],row['updates']
        if any(config.get(key) != value for key,value in expected_config.items()):
            raise ValueError('Training scene/config/hyperparameters differ from registration')
        if not summary or summary.get('status') != 'complete' or \
                summary['decisions'] < registration['minimum_decisions_per_run']:
            raise ValueError('Incomplete training run or insufficient decision budget')
        if not records or not updates or any(not record['terminated'] or record['truncated'] for record in records):
            raise ValueError('Training contains missing or truncated episodes')
        if summary['truncated_sequences'] or summary['completed_training_episodes'] != len(records):
            raise ValueError('Training complete episode counters disagree')
        if sum(record['decisions'] for record in records) != summary['decisions'] or \
                sum(update['batch_decisions'] for update in updates) != summary['decisions']:
            raise ValueError('Training decisions disagree across raw records/updates/summary')
        if len(updates) != summary['updates'] or [u['update'] for u in updates] != list(range(1,len(updates)+1)):
            raise ValueError('Training update counters disagree')
        if sum(u['actor_decisions']+u['forced_single_legal_decisions'] for u in updates) != summary['decisions']:
            raise ValueError('Effective and forced actor decisions do not account for the budget')
        if any(record['seed_role'] != 'train' or record['support_mode'] != mode or
               record['opponent_style'] not in registration['train_styles'] for record in records):
            raise ValueError('Training seed/style/support assignment differs')
        if set(record['seed'] for record in records) & set(registration['test_seeds']):
            raise ValueError('Test seeds entered training')
        if records[-1]['cumulative_decisions'] != summary['decisions']:
            raise ValueError('Final raw cumulative training decision count disagrees')
    labels = registration['test_policies']
    if len(labels) != len(set(labels)):
        raise ValueError('Duplicate registered evaluation policy label')
    cases = set(itertools.product(registration['test_seeds'],registration['test_styles'],registration['test_sides']))
    expected = {(label,*case) for label in labels for case in cases}
    observed = [(row['policy'],row['seed'],row['style'],row['side']) for row in episodes]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError('Missing, duplicate or unregistered evaluation policy/case')
    if evaluation.get('scenario_id') != SCENARIO_ID or \
            evaluation.get('environment_scenario_id') != ENVIRONMENT_SCENARIO_ID or \
            evaluation.get('status') != 'complete' or evaluation['expected_episodes'] != len(expected) or \
            evaluation['actual_episodes'] != len(expected) or evaluation.get('support_mode') != mode:
        raise ValueError('Evaluation scope, mode or completion counters differ')
    if evaluation.get('SGD_steps') != 0 or evaluation.get('real_nav2_in_loop') is not False or \
            evaluation.get('official_UL_UC_training') is not False or evaluation.get('evaluation_split') != 'test':
        raise ValueError('Evaluation must be frozen independent test with prototype scope')
    if any(not row['completed'] or row['truncated'] or row.get('real_nav2_in_loop') is not False or
           row.get('official_UL_UC_training') is not False or row.get('support_mode') != mode for row in episodes):
        raise ValueError('Evaluation contains incomplete or wrong-scope episodes')
    expected_checkpoints = {f"{row['config']['memory']}_s{row['config']['seed']}":row for row in training}
    entries = evaluation['checkpoints']
    actual = {row['label']:row for row in entries}
    if len(entries) != len(expected_checkpoints) or set(actual) != set(expected_checkpoints):
        raise ValueError('Evaluated checkpoint identities differ from training')
    for label,row in expected_checkpoints.items():
        item = actual[label]
        if item['sha256'] != row['source_sha256']['final.pt'] or \
                project_path(item['path']) != project_path(row['run'])/'final.pt':
            raise ValueError('Evaluation did not use each registered final checkpoint')
        if item.get('weights_unchanged') is not True or \
                item.get('weights_before_sha256') != item.get('weights_after_sha256'):
            raise ValueError('Evaluation model weights changed')
        if item.get('memory') != row['config']['memory'] or item.get('training_seed') != row['config']['seed'] or \
                item.get('training_decisions') != row['summary']['decisions']:
            raise ValueError('Evaluation checkpoint metadata disagrees with training')
    return summarize(episodes,labels)


def _trace_path(directory,label,registration):
    return directory/f"{label}_seed{registration['test_seeds'][0]}_{registration['test_styles'][0]}_side{registration['test_sides'][0]}.npz"


def _verified_model(entry):
    path = project_path(entry['path'])
    if file_sha256(path) != entry['sha256']:
        raise ValueError('Checkpoint bytes changed since evaluation')
    model,_ = load_strategy_checkpoint(path)
    if file_sha256(path) != entry['sha256'] or model_weights_sha256(model) != entry['weights_after_sha256']:
        raise ValueError('Loaded checkpoint weights differ from frozen evaluation')
    return model


def _read_trace(path,expected_decisions):
    with np.load(path) as data:
        trace = {key:np.asarray(data[key]).copy() for key in ('state','candidates','mask','action')}
    if any(len(value) != expected_decisions for value in trace.values()) or expected_decisions < 1:
        raise ValueError('Recorded trace length differs from its raw evaluation episode')
    observation_tensors([{key:trace[key][i] for key in ('state','candidates','mask')}
                         for i in range(expected_decisions)])
    actions = trace['action']
    if not np.issubdtype(actions.dtype,np.integer) or actions.shape != (expected_decisions,) or \
            (actions < 0).any() or (actions >= 16).any() or \
            not trace['mask'][np.arange(expected_decisions),actions].all():
        raise ValueError('Recorded trace actions are invalid or masked')
    return trace


@torch.no_grad()
def recorded_diagnostics(directory,evaluation,registration,training,episodes):
    """Saved first-case observations only; no environment rerun or SGD."""
    directory = project_path(directory)
    case = (registration['test_seeds'][0],registration['test_styles'][0],registration['test_sides'][0])
    train_by_label = {f"{row['config']['memory']}_s{row['config']['seed']}":row for row in training}
    results = []
    for entry in evaluation['checkpoints']:
        label = entry['label']
        matching = [row for row in episodes if (row['policy'],row['seed'],row['style'],row['side']) == (label,*case)]
        if len(matching) != 1:
            raise ValueError('First-case replay identity missing or duplicated')
        path = _trace_path(directory,label,registration)
        trace = _read_trace(path,matching[0]['decisions'])
        final = _verified_model(entry)
        initial_path = project_path(train_by_label[label]['run'])/'initial.pt'
        if file_sha256(initial_path) != train_by_label[label]['source_sha256']['initial.pt']:
            raise ValueError('Initial checkpoint bytes changed')
        initial,_ = load_strategy_checkpoint(initial_path)
        final_hidden,initial_hidden = final.initial_state(1),initial.initial_state(1)
        mismatches=changed=memory_changed=0
        divergences,logit_deltas=[] ,[]
        for index in range(len(trace['action'])):
            observation={key:trace[key][index] for key in ('state','candidates','mask')}
            tensors=observation_tensors([observation])
            carried,_,final_hidden=final.distribution_value(**tensors,hidden=final_hidden)
            before,_,initial_hidden=initial.distribution_value(**tensors,hidden=initial_hidden)
            action=int(carried.logits.argmax(-1).item())
            mismatches += action != int(trace['action'][index])
            changed += action != int(before.logits.argmax(-1).item())
            divergences.append(float(torch.distributions.kl_divergence(before,carried).item()))
            if final.memory == 'gru':
                cleared,_,_=final.distribution_value(**tensors,hidden=final.initial_state(1))
                memory_changed += action != int(cleared.logits.argmax(-1).item())
                legal=tensors['mask']
                logit_deltas.append(float((carried.logits[legal]-cleared.logits[legal]).abs().max()))
        if mismatches:
            raise ValueError('Frozen chronological replay does not reproduce recorded greedy actions')
        result={'policy':label,'trace':str(path),'trace_sha256':file_sha256(path),
            'observations':len(trace['action']),'multiple_legal_choices':int((trace['mask'].sum(-1)>1).sum()),
            'final_replay_mismatches':mismatches,'initial_vs_final_greedy_differences':changed,
            'mean_initial_to_final_KL':float(np.mean(divergences)),
            'scope':'same final-policy recorded observations; trajectory-conditional, not performance causality'}
        if final.memory == 'gru':
            result.update(carried_vs_cleared_greedy_differences=memory_changed,
                maximum_legal_normalized_logit_difference=max(logit_deltas))
        results.append(result)
    return results


def _check_summary(raw,recorded):
    for label,stats in raw['groups']['all'].items():
        authoritative=recorded['groups']['all'][label]
        for key,value in stats.items():
            saved=authoritative[key]
            if isinstance(value,(int,float)) and value is not None and not np.isclose(value,saved,rtol=1e-6,atol=1e-8):
                raise ValueError(f'Evaluation summary disagrees with raw episodes: {label}/{key}')


def _number(value,digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def report_markdown(registration,training,stats,diagnostics,gate):
    lines=['# 时序与全局任务决策 v2','',
        '本轮是独立的 role-balanced alphaRM 简化 3v3 任务实验；正式 UL／UC 规则地图、共享经济和真实 Nav2／MPPI 不在本实验中。',
        '',f"场景 support mode：`{registration['support_mode']}`。训练前 development control gate 已通过；测试使用预注册固定 final checkpoint，没有依据测试结果选择模型。",'',
        '| 训练模型 | 实际决策 | 完整局 | PPO 更新 | actor 有效／强制 | actor 参数变化 L2 | 实际局长范围 s | ≥90s 比例 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in training:
        config,summary,e=row['config'],row['summary'],row['learning_evidence']
        lines.append(f"| `{config['memory']}_s{config['seed']}` | {summary['decisions']} | {summary['completed_training_episodes']} | {summary['updates']} | {e['unique_actor_decisions']}／{e['forced_single_legal_decisions']} | {e['actor_parameter_change_l2']:.6f} | {e['duration_range_s'][0]:.1f}–{e['duration_range_s'][1]:.1f} | {e['fraction_episodes_at_least_90s']:.3f} |")
    lines += ['', '| Policy | 胜／局 | 平局 | 团队回报 | 中心占比 | HP 均值 | 伤害／全队 | 换点 | BT 覆盖／决策 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for policy,row in stats['groups']['all'].items():
        lines.append(f"| `{policy}` | {row['wins']}/{row['completed']} | {row['draws']} | {row['mean_return']:.3f} | {_number(row['mean_sampled_center_fraction'])} | {_number(row['mean_sampled_mean_own_hp_fraction'])} | {_number(row['mean_damage_share_of_team'])} | {_number(row['mean_switches'],2)} | {_number(row['mean_overrides_per_decision'])} |")
    lines += ['', '| Policy | pretrained（训练风格）胜／局 | pursuit（训练未见）胜／局 |', '|---|---:|---:|']
    for policy in registration['test_policies']:
        seen,held=(stats['by_style'][style][policy] for style in ('pretrained','pursuit'))
        lines.append(f"| `{policy}` | {seen['wins']}/{seen['completed']} | {held['wins']}/{held['completed']} |")
    lines += ['', '相同 GRU 权重的记忆消融：携带 hidden 减去每步清零；这会改变 hidden 分布，不能独立证明全局规划能力。','',
        '| 模型 | 胜率差 | 团队回报差 | 中心占比差 | 配对场景 |','|---|---:|---:|---:|---:|']
    for row in training:
        if row['config']['memory'] != 'gru':continue
        label=f"gru_s{row['config']['seed']}"
        contrast=stats['paired_contrasts'][f'{label}_minus_{label}_reset']
        m=contrast['metrics']
        lines.append(f"| `{label}` | {m['win']['mean']:+.3f} | {m['return']['mean']:+.3f} | {m['sampled_center_fraction']['mean']:+.3f} | {contrast['completed_pairs']} |")
    lines += ['', '| 首测试场景的记录观测 | 观测数 | initial→final greedy 差异 | KL(initial‖final) | GRU 携带／清零 greedy 差异 |',
        '|---|---:|---:|---:|---:|']
    for item in diagnostics:
        lines.append(f"| `{item['policy']}` | {item['observations']} | {item['initial_vs_final_greedy_differences']} | {item['mean_initial_to_final_KL']:.6f} | {item.get('carried_vs_cleared_greedy_differences','—')} |")
    controls={label:stats['groups']['all'][label] for label in ('continue_only','fixed_center')}
    saturated=any(row['wins']==row['completed'] for row in controls.values())
    lines += ['', f"固定控制诊断：continue_only {controls['continue_only']['wins']}/{controls['continue_only']['completed']}，fixed_center {controls['fixed_center']['wins']}/{controls['fixed_center']['completed']}。"+
        ('至少一个固定控制达到胜率上限；这限制了胜率对学习收益的区分能力。' if saturated else '本测试批次未出现固定控制全胜的胜率上限。'),'',
        '记录观测回放须逐步复现原 greedy 动作；上述 KL、logit 和 greedy 差异仅描述相同记录轨迹上的模型变化，不构成策略收益或因果归因。', '',
        'MLP 为 55,490 参数，GRU 为 113,474 参数，容量不相同。只有两个训练 seed 和两个独立测试扰动 seed；阵营／风格在 seed 内相关。该 pilot 不证明收敛、实战优势或 PPO 必然优于有记忆的 BT。', '',
        '原始物理与规则奖励常量保持不变。支持机器人经固定 decoder 编码，其可表示速度上限约为请求归一化幅值的 99.5%；这是已披露的执行适配误差，不能声称逐位相同的双边控制。真实细节记录在 scene metadata。', '',
        '仍使用上游 HUD 观测语义，其中含敌方全局 HP／alive；本轮不认证正式比赛允许的观测边界。没有共享金币／正式买血买弹状态，没有自动扩大 RL 权限或覆盖 BT 安全门禁。', '',
        '![训练过程](training_progress.png)','']
    return '\n'.join(lines)


def build_report(registration_path,runs,evaluation_directory,out):
    torch.set_num_threads(2)
    out=Path(out).resolve()
    if out.exists():raise FileExistsError(f'Refusing to overwrite report: {out}')
    registration_path=project_path(registration_path)
    registration=read_json(registration_path)
    gate_directory=project_path(registration['development_gate_run'])
    gate_path=gate_directory/'development_gate.json'
    gate=read_json(gate_path)
    training=[gather_training(run) for run in runs]
    evaluation_directory=project_path(evaluation_directory)
    evaluation=read_json(evaluation_directory/'summary.json')
    episodes=read_jsonl(evaluation_directory/'episodes.jsonl')
    stats=validate_registered_inputs(registration,training,evaluation,episodes,gate=gate)
    if registration['scene_config_sha256'] != file_sha256(ROOT/SCENE_CONFIG):
        raise ValueError('Registered scene config differs from current frozen scene')
    for relative,digest in evaluation['implementation_sha256'].items():
        if file_sha256(ROOT/relative) != digest:
            raise ValueError(f'Evaluation source changed: {relative}')
    _check_summary(stats,evaluation)
    diagnostics=recorded_diagnostics(evaluation_directory,evaluation,registration,training,episodes)
    markdown=report_markdown(registration,training,stats,diagnostics,gate)
    result={'artifact_type':'role_balanced_temporal_pilot_not_official_match_acceptance',
        'scenario_id':SCENARIO_ID,'script_sha256':file_sha256(Path(__file__)),
        'registration':{'path':str(registration_path),'sha256':file_sha256(registration_path),'data':registration},
        'development_gate':{'path':str(gate_path),'sha256':file_sha256(gate_path),'data':gate},
        'training':training,'evaluation':{'directory':str(evaluation_directory),'summary':evaluation,
            'recomputed':stats,'source_sha256':{name:file_sha256(evaluation_directory/name) for name in
                ('manifest.json','summary.json','episodes.jsonl')}},
        'recorded_observation_diagnostics':diagnostics,
        'control_saturation':{label:stats['groups']['all'][label]['wins']==stats['groups']['all'][label]['completed']
            for label in ('continue_only','fixed_center')},
        'live_Nav2_MPPI_in_loop':False,'official_UL_UC_match':False,
        'official_observation_legality_claimed':False,'shared_economy_simulated':False,
        'causal_superiority_claim':False,'RL_permissions_expanded':False}
    serialized=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    # All acceptance and deterministic replay checks finish before creating output.
    out.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for row in training:
        label=f"{row['config']['memory']}_s{row['config']['seed']}"
        returns=np.asarray([item['return'] for item in row['episodes']])
        axes[0].plot([item['cumulative_decisions'] for item in row['episodes']],
            np.cumsum(returns)/np.arange(1,len(returns)+1),label=label)
        axes[1].plot([item['decisions'] for item in row['updates']],
            [item['value_loss'] for item in row['updates']],label=label)
    axes[0].set(title='Whole-episode cumulative mean return',xlabel='Actual candidate decisions',ylabel='Upstream team return')
    axes[1].set(title='PPO critic loss',xlabel='Actual candidate decisions',ylabel='Loss')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle('Role-balanced simplified strategic pilot; no live Nav2 or official UL/UC')
    fig.savefig(out/'training_progress.png',dpi=150)
    plt.close(fig)
    (out/'comparison.json').write_text(serialized)
    (out/'report.md').write_text(markdown)
    print(json.dumps({'output':str(out),'training_runs':len(training),'evaluation_episodes':len(episodes)}))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    build_report(args.registration,args.runs,args.evaluation,args.out)


if __name__=='__main__':main()
