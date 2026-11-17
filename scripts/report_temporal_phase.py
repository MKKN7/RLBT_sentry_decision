#!/usr/bin/env python3
"""Report recorded temporal strategic pilots, separate from real Nav2 runs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
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


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def learning_evidence(run, updates):
    initial = torch.load(run / 'initial.pt', map_location='cpu', weights_only=False)['model']
    final = torch.load(run / 'final.pt', map_location='cpu', weights_only=False)['model']
    result = {}
    for prefix in ('actor.', 'critic.'):
        squared = sum(float(np.square(final[key].numpy().astype(np.float64)
            - value.numpy().astype(np.float64)).sum()) for key,value in initial.items() if key.startswith(prefix))
        result[prefix[:-1] + '_parameter_change_l2'] = float(np.sqrt(squared))
    result.update(unique_actor_decisions=sum(row['actor_decisions'] for row in updates),
        forced_single_legal_decisions=sum(row['forced_single_legal_decisions'] for row in updates),
        minimum_actor_decisions_per_update=min(row['actor_decisions'] for row in updates),
        single_actor_decision_updates=sum(row['actor_decisions'] == 1 for row in updates),
        actor_optimizer_steps=sum(row['epochs'] for row in updates if row['actor_decisions'] > 0),
        maximum_first_epoch_approx_kl=max(abs(row['first_epoch_approx_kl']) for row in updates))
    return result


def validate_registered_inputs(registration, training, evaluation, episodes):
    expected_training = set(itertools.product(registration['memory_variants'], registration['training_seeds']))
    identities = [(row['config']['memory'], row['config']['seed']) for row in training]
    if len(identities) != len(expected_training) or set(identities) != expected_training:
        raise ValueError('Training runs differ from the registered architecture/seed identities')
    for row in training:
        summary, records = row['summary'], row['episodes']
        config = row['config']
        expected_config = {'gamma_per_second': registration['gamma_per_second'],
            'lambda_per_second': registration['lambda_per_second'],
            'episodes_per_update': registration['episodes_per_update'], 'epochs': registration['epochs'],
            'learning_rate': registration['learning_rate'], 'entropy_coef': registration['entropy_coefficient'],
            'target_decisions': registration['minimum_decisions_per_run'], 'BC_updates': 0,
            'randomization': True, 'reward_profile': 'upstream', 'real_nav2_in_loop': False,
            'official_UL_UC_training': False}
        if any(config.get(key) != value for key,value in expected_config.items()):
            raise ValueError('Training hyperparameters or scope differ from registration')
        if not summary or summary['status'] != 'complete' or summary['decisions'] < registration['minimum_decisions_per_run']:
            raise ValueError(f"Incomplete training run: {row['run']}")
        if sum(record['decisions'] for record in records) != summary['decisions']:
            raise ValueError('Training raw decisions disagree with summary')
        complete = sum(record['terminated'] and not record['truncated'] for record in records)
        if complete != summary['completed_training_episodes'] or complete != len(records) or summary['truncated_sequences']:
            raise ValueError('Registered complete-episode pilot contains a truncated training sequence')
        if sum(update['batch_decisions'] for update in row['updates']) != summary['decisions']:
            raise ValueError('Training updates do not account for all recorded decisions')
    aliases = {'reactive_BT': 'bt_reactive', 'memory_BT': 'bt_memory',
        **{f'gru_reset_each_step_s{seed}': f'gru_s{seed}_reset' for seed in registration['training_seeds']}}
    labels = [aliases.get(label, label) for label in registration['test_policies']]
    cases = set(itertools.product(registration['test_seeds'], registration['test_styles'], registration['test_sides']))
    expected = {(label, *case) for label in labels for case in cases}
    observed = [(row['policy'], row['seed'], row['style'], row['side']) for row in episodes]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError('Missing, duplicate or unregistered policy/case identity')
    if not evaluation or evaluation['status'] != 'complete' or evaluation['expected_episodes'] != len(expected) or evaluation['actual_episodes'] != len(expected):
        raise ValueError('A complete registered frozen evaluation is required')
    if any(row['truncated'] or not row['completed'] for row in episodes):
        raise ValueError('Unexpected evaluation truncation')
    if evaluation['SGD_steps'] != 0 or any(not item['weights_unchanged'] for item in evaluation['checkpoints']):
        raise ValueError('Evaluation weights were not frozen')
    if evaluation['real_nav2_in_loop'] is not False or evaluation['official_UL_UC_training'] is not False:
        raise ValueError('Proxy evaluation has incorrect scope flags')
    expected_checkpoints = {f"{row['config']['memory']}_s{row['config']['seed']}": row for row in training}
    actual_checkpoints = {row['label']: row for row in evaluation['checkpoints']}
    if len(evaluation['checkpoints']) != len(expected_checkpoints) or set(actual_checkpoints) != set(expected_checkpoints):
        raise ValueError('Evaluated checkpoint identities differ from training runs')
    for label,row in expected_checkpoints.items():
        if actual_checkpoints[label]['sha256'] != row['source_sha256']['final.pt']:
            raise ValueError('Evaluation did not use the registered final checkpoint')
    return aliases


def validate_recorded_trace(data, evaluation, label, case):
    matched = [row for row in evaluation['episodes'] if (row['policy'],row['seed'],row['style'],row['side']) == (label,*case)]
    if len(matched) != 1:
        raise ValueError('Recorded trace does not identify a unique evaluation episode')
    count = matched[0]['decisions']
    if any(len(data[key]) != count for key in ('state','candidates','mask','action')):
        raise ValueError('Recorded trace length disagrees with evaluation decisions')


@torch.no_grad()
def recorded_memory_dependence(evaluation_directory, evaluation, registration):
    from scripts.train_temporal_phase import load_temporal_checkpoint, observation_tensors
    case = (registration['test_seeds'][0], registration['test_styles'][0], registration['test_sides'][0])
    results = []
    for entry in evaluation['checkpoints']:
        if entry['memory'] != 'gru':
            continue
        label = entry['label']
        path = evaluation_directory / f'{label}_seed{case[0]}_{case[1]}_side{case[2]}.npz'
        if sha(Path(entry['path'])) != entry['sha256']:
            raise ValueError('Evaluated checkpoint file changed before report replay')
        with np.load(path) as data:
            validate_recorded_trace(data, evaluation, label, case)
            model,_ = load_temporal_checkpoint(entry['path'])
            hidden = model.initial_state(1)
            differences, replay_mismatches, logit_deltas, choices = 0, 0, [], 0
            for index in range(len(data['action'])):
                observation = {key:data[key][index] for key in ('state', 'candidates', 'mask')}
                tensors = observation_tensors([observation])
                carried,_,hidden = model.distribution_value(**tensors, hidden=hidden)
                cleared,_,_ = model.distribution_value(**tensors, hidden=model.initial_state(1))
                action = int(carried.logits.argmax(-1).item())
                differences += action != int(cleared.logits.argmax(-1).item())
                replay_mismatches += action != int(data['action'][index])
                legal = tensors['mask']
                logit_deltas.append(float((carried.logits[legal]-cleared.logits[legal]).abs().max()))
                choices += int(observation['mask'].sum() > 1)
            if replay_mismatches:
                raise ValueError('Recorded GRU episode does not reproduce with carried memory')
            results.append({'policy':label, 'trace':str(path.resolve()), 'trace_sha256':sha(path),
                'observations':len(data['action']), 'multiple_legal_choices':choices,
                'carried_vs_cleared_greedy_differences':differences,
                'carried_replay_mismatches':replay_mismatches,
                'maximum_legal_normalized_logit_difference':max(logit_deltas),
                'scope':'same recorded observations; no environment rerun or performance attribution'})
    return results


@torch.no_grad()
def recorded_learning_change(evaluation_directory, evaluation, registration):
    from scripts.train_temporal_phase import load_temporal_checkpoint, observation_tensors
    case = (registration['test_seeds'][0], registration['test_styles'][0], registration['test_sides'][0])
    results = []
    for entry in evaluation['checkpoints']:
        label = entry['label']
        path = evaluation_directory / f'{label}_seed{case[0]}_{case[1]}_side{case[2]}.npz'
        if sha(Path(entry['path'])) != entry['sha256']:
            raise ValueError('Evaluated checkpoint file changed before report replay')
        final,_ = load_temporal_checkpoint(entry['path'])
        initial,_ = load_temporal_checkpoint(Path(entry['path']).parent/'initial.pt')
        final_hidden, initial_hidden = final.initial_state(1), initial.initial_state(1)
        changed, replay_mismatches, divergences = 0, 0, []
        with np.load(path) as data:
            validate_recorded_trace(data, evaluation, label, case)
            for index in range(len(data['action'])):
                tensors = observation_tensors([{key:data[key][index] for key in ('state','candidates','mask')}])
                after,_,final_hidden = final.distribution_value(**tensors, hidden=final_hidden)
                before,_,initial_hidden = initial.distribution_value(**tensors, hidden=initial_hidden)
                action = int(after.logits.argmax(-1).item())
                changed += action != int(before.logits.argmax(-1).item())
                replay_mismatches += action != int(data['action'][index])
                divergences.append(float(torch.distributions.kl_divergence(before,after).item()))
            if replay_mismatches:
                raise ValueError('Final checkpoint does not reproduce its recorded evaluation actions')
            results.append({'policy':label,'observations':len(data['action']),
                'trace':str(path.resolve()),'trace_sha256':sha(path),
                'initial_checkpoint_sha256':sha(Path(entry['path']).parent/'initial.pt'),
                'initial_vs_final_greedy_differences':changed,'mean_initial_to_final_KL':float(np.mean(divergences)),
                'final_replay_mismatches':replay_mismatches,
                'scope':'same final-policy recorded observations; not paired performance evaluation'})
    return results


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument('--registration', type=Path, default=ROOT/'configs/temporal_strategy_experiment_20261117.json')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Refusing to overwrite report: {args.out}')
    registration = read_json(args.registration)
    training = []
    for run in args.runs:
        config = read_json(run / "config.json") or {}
        summary = read_json(run / "summary.json")
        episodes = read_jsonl(run / "episodes.jsonl")
        updates = read_jsonl(run / "updates.jsonl")
        training.append({"run": str(run.resolve()), "config": config, "summary": summary,
            "episodes": episodes, "updates": updates,
            "learning_evidence": learning_evidence(run, updates),
            "training_coverage": {'episodes_by_style': dict(Counter(row['opponent_style'] for row in episodes)),
                'episodes_by_side': dict(Counter(str(row['team']) for row in episodes)),
                'actual_simulated_s': sum(row['simulated_s'] for row in episodes),
                'episode_duration_s_range': [min(row['simulated_s'] for row in episodes), max(row['simulated_s'] for row in episodes)]},
            "source_sha256": {name: sha(run / name) for name in
                ("config.json", "summary.json", "episodes.jsonl", "updates.jsonl", "initial.pt", "final.pt")
                if (run / name).is_file()}})
    evaluation_summary = read_json(args.evaluation / "summary.json")
    evaluation_rows = read_jsonl(args.evaluation / "episodes.jsonl")
    aliases = validate_registered_inputs(registration, training, evaluation_summary, evaluation_rows)
    groups = defaultdict(list)
    for row in evaluation_rows:
        groups[str(row.get("policy", row.get("policy_id", "unknown")))].append(row)
    metrics = {}
    for policy, episodes in groups.items():
        complete = [row for row in episodes if row['completed']]
        wins = sum(bool(row['win']) for row in complete)
        metrics[policy] = {"episodes": len(episodes), "completed": len(complete),
            "truncated": len(episodes)-len(complete), "wins": wins,
            "draws": sum(bool(row['draw']) for row in complete),
            "win_rate": wins/len(complete) if complete else None,
            "mean_return": float(np.mean([row['return'] for row in complete])) if complete else None,
            "mean_switches": float(np.mean([row['switches'] for row in complete])) if complete else None}
    for label, row in metrics.items():
        authoritative = evaluation_summary['groups']['all'][label]
        if row['wins'] != authoritative['wins'] or row['completed'] != authoritative['completed'] or not np.isclose(row['mean_return'], authoritative['mean_return']):
            raise ValueError(f'Evaluation summary disagrees with raw episodes: {label}')
    dependence = recorded_memory_dependence(args.evaluation, evaluation_summary, registration)
    learning_change = recorded_learning_change(args.evaluation, evaluation_summary, registration)
    args.out.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for row in training:
        episodes, updates, label = row['episodes'], row['updates'], Path(row['run']).name
        returns = np.asarray([item['return'] for item in episodes])
        axes[0].plot([item['cumulative_decisions'] for item in episodes],
            np.cumsum(returns)/np.arange(1,len(returns)+1), label=label)
        axes[1].plot([item['decisions'] for item in updates], [item['value_loss'] for item in updates], label=label)
    axes[0].set(title='Whole-episode training cumulative mean return', xlabel='Collected decisions', ylabel='Upstream team return')
    axes[1].set(title='PPO critic loss', xlabel='Collected decisions', ylabel='Loss')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=6)
    fig.suptitle('Simplified alphaRM strategic pilot; no live Nav2 or official UL/UC match')
    fig.savefig(args.out/'training_progress.png', dpi=150)
    plt.close(fig)
    report = {"artifact_type": "simplified_whole_episode_temporal_pilot_not_match_acceptance",
        "script_sha256": sha(Path(__file__)), "training": training,
        "registration": {'path': str(args.registration.resolve()), 'sha256': sha(args.registration), 'policy_aliases': aliases},
        "recorded_observation_memory_dependence": dependence,
        "recorded_observation_learning_change": learning_change,
        "evaluation": {"directory": str(args.evaluation.resolve()), "summary": evaluation_summary,
            "recomputed_metrics": metrics, "source_sha256": {name:sha(args.evaluation/name) for name in
                ("manifest.json", "summary.json", "episodes.jsonl") if (args.evaluation/name).is_file()}},
        "live_Nav2_MPPI_in_loop": False, "official_UL_UC_match": False,
        "causal_superiority_claim": False, "RL_permissions_expanded": False}
    (args.out / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    lines = ["# 时序与全局任务决策 pilot", "", "本报告使用已保存的alphaRM简化3v3完整回合；不是正式UL／UC比赛，也未在该环境调用真实Nav2／MPPI。真实导航结果仍单独保存。", "",
        "比较MLP PPO、GRU PPO、reactive BT、带内部状态的Python BT基线，以及GRU每步清空隐藏状态的消融。PPO是训练算法，历史利用能力来自状态设计／循环记忆；不预设优于BT。", "",
        "| Run | Memory | 实际交互 | 完整局数 | 状态 |", "|---|---|---:|---:|---|"]
    for row in training:
        summary = row["summary"] or {}
        config = row["config"]
        lines.append(f"| `{Path(row['run']).name}` | {config['memory']} | {summary['decisions']} | {summary['completed_training_episodes']} | {summary['status']} |")
    lines += ['', '学习有效性（强制单选不计直接actor loss，仍保留时序和critic；参数变化不等于策略收益）：', '',
        '| Run | actor有效决策 | 强制单选 | actor参数变化L2 | 单有效样本更新数 |', '|---|---:|---:|---:|---:|']
    for row in training:
        evidence = row['learning_evidence']
        lines.append(f"| `{Path(row['run']).name}` | {evidence['unique_actor_decisions']} | {evidence['forced_single_legal_decisions']} | {evidence['actor_parameter_change_l2']:.6f} | {evidence['single_actor_decision_updates']} |")
    lines += ['', '实际时间覆盖：', '', '| Run | 累计仿真秒数 | 每局最短—最长秒数 |', '|---|---:|---:|']
    for row in training:
        coverage = row['training_coverage']
        lo,hi = coverage['episode_duration_s_range']
        lines.append(f"| `{Path(row['run']).name}` | {coverage['actual_simulated_s']:.1f} | {lo:.1f}—{hi:.1f} |")
    lines += ["", "固定测试结果（每个checkpoint独立列出，未用测试挑选模型）：", "",
        "| Policy | 胜／完整局 | 平局 | 平均团队回报 | 平均换点 |", "|---|---:|---:|---:|---:|"]
    for policy, row in metrics.items():
        lines.append(f"| `{policy}` | {row['wins']}/{row['completed']} | {row['draws']} | {row['mean_return']:.3f} | {row['mean_switches']:.2f} |")
    lines += ['', '分风格结果：', '', '| Policy | capture 胜/局 | pursuit（训练未见）胜/局 |', '|---|---:|---:|']
    for policy in metrics:
        parts = [evaluation_summary['by_style'][style][policy] for style in ('capture', 'pursuit')]
        lines.append(f"| `{policy}` | {parts[0]['wins']}/{parts[0]['completed']} | {parts[1]['wins']}/{parts[1]['completed']} |")
    lines += ['', '整局行为诊断（占比为决策后采样估计，不是连续真值积分）：', '',
        '| Policy | 平均实际局长s | 存活占比 | 中心占比 | 自身伤害/全队伤害 | BT覆盖/决策 |', '|---|---:|---:|---:|---:|---:|']
    for policy in metrics:
        stats = evaluation_summary['groups']['all'][policy]
        lines.append(f"| `{policy}` | {stats['mean_actual_duration_s']:.1f} | {stats['mean_sampled_alive_fraction']:.3f} | {stats['mean_sampled_center_fraction']:.3f} | {stats['mean_damage_share_of_team']:.3f} | {stats['mean_overrides_per_decision']:.3f} |")
    lines += ['', '同一GRU权重的记忆消融（携带hidden减去每步清零，正值代表携带记忆较高）：', '',
        '| 模型 | 胜率差 | 团队回报差 | 配对场景数 |', '|---|---:|---:|---:|']
    for policy in metrics:
        reset_label = policy + '_reset'
        if reset_label not in metrics:
            continue
        contrast = evaluation_summary['paired_contrasts'].get(f'{policy}_minus_{reset_label}')
        if not contrast:
            raise ValueError('Missing preregistered memory ablation contrast')
        delta = contrast['metrics']
        lines.append(f"| `{policy}` | {delta['win']['mean']:+.3f} | {delta['return']['mean']:+.3f} | {contrast['completed_pairs']} |")
    lines += ['', '相同记录观测的记忆检查（第一测试场景，carried回放须逐步复现原动作）：', '',
        '| GRU | 观测数 | 携带与清零的greedy动作差异数 |', '|---|---:|---:|']
    for item in dependence:
        lines.append(f"| `{item['policy']}` | {item['observations']} | {item['carried_vs_cleared_greedy_differences']} |")
    lines += ['', '相同记录观测的学习前后检查（不用于模型选择或胜率归因）：', '',
        '| 模型 | 观测数 | initial与final的greedy差异数 | 平均KL(initial→final) |', '|---|---:|---:|---:|']
    for item in learning_change:
        lines.append(f"| `{item['policy']}` | {item['observations']} | {item['initial_vs_final_greedy_differences']} | {item['mean_initial_to_final_KL']:.6f} |")
    lines += ["", "两个训练seed和两个测试扰动seed只是小规模工程pilot。部分对手风格、双方阵营在同一seed下相关；均值差不能独立证明收敛、实战优势或普适优于BT。详细分风格、记忆消融及原始结果见evaluation目录。", "",
        "gamma按实际秒折扣.997：180秒后的折扣约.582；lambda按秒.97，二者相乘的GAE trace半衰期约20.7秒，180秒后的trace约.00242。折扣视野不等于实际优势归因视野。GRU采用时间顺序训练、episode重置，仍不能保证记住整局或取得最佳全局策略。", "",
        "MLP共有55,490参数，GRU共有113,474参数，二者容量不相同。每步清零消融保持权重相同，但会改变GRU输入的hidden分布；它用于检查模型对历史的依赖，不能单独证明模型学会了长周期全局规划。", "",
        "复用上游冻结的env_step.onnx、actor.onnx（其他机器人／固定技能）、fixture和旧观测／候选／执行器；本轮未复用历史checkpoint、BC标签或测试数据进行训练。买血买弹与正式UL/UC共享经济不在该不可变简化环境内。", "",
        "每局最长180秒，但可按上游胜负条件提前结束；完整回合不等于必定经历180秒。团队中另两台机器人使用固定上游策略，团队胜负同时受其影响，不能将单局胜利全部归功于待测选点模型。", "",
        "![训练过程](training_progress.png)", ""]
    (args.out / "report.md").write_text("\n".join(lines))
    print(json.dumps({"output":str(args.out.resolve()),"runs":len(training),"evaluation_episodes":len(evaluation_rows)},ensure_ascii=False))


if __name__ == "__main__":
    main()
