#!/usr/bin/env python3
"""Report recorded UC support v2 ablations without running an evaluation.

Inputs are trainer-owned JSON/JSONL artifacts. Reading initial.pt only hashes
the already saved actor tensors; this script never calls an environment or
policy. Navigation episodes and safe-wait episodes have separate denominators.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("BC", "PPO", "BT", "random")
INTENTS = ("navigate", "recover", "continue", "hold")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path, warnings, *, required=False):
    if not path.is_file():
        if required:
            warnings.append(f"Missing {path.name}; corresponding values remain unavailable.")
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        warnings.append(f"Cannot read {path.name}: {error}")
        return None
    if not isinstance(value, dict):
        warnings.append(f"{path.name} must contain a JSON object.")
        return None
    return value


def read_jsonl(path, warnings):
    if not path.is_file():
        warnings.append(f"Missing {path.name}.")
        return []
    rows = []
    for index, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            warnings.append(f"Skipped malformed/incomplete {path.name} line {index}.")
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def mean_field(rows, field):
    values = [row.get(field) for row in rows]
    if not values or any(not isinstance(value, (int, float)) or
                         not math.isfinite(value) for value in values):
        return None
    return float(np.mean(values))


def navigation_rows(rows):
    return [row for row in rows
            if row.get("parameters", {}).get("no_safe_goal") is False]


def wait_rows(rows):
    return [row for row in rows
            if row.get("parameters", {}).get("no_safe_goal") is True]


def metrics(rows):
    """Keep unavailable metrics missing rather than treating them as zero."""
    nav, waiting = navigation_rows(rows), wait_rows(rows)
    result = {"episodes": len(rows),
        "task_successes": sum(row.get("success") is True for row in rows),
        "navigation_episodes": len(nav),
        "navigation_successes": sum(row.get("navigation_success") is True for row in nav),
        "safe_wait_episodes": len(waiting),
        "safe_wait_successes": sum(row.get("wait_success") is True for row in waiting),
        "unclassified_episodes": len(rows) - len(nav) - len(waiting)}
    for field in ("score_100", "time_s", "return", "idle_healthy_steps", "recoveries",
                  "needless_recoveries", "recovery_opportunity_steps", "BT_veto_count",
                  "idle_healthy_with_goal_available_steps",
                  "unsafe_executions", "stage_completed", "stages"):
        result["mean_" + field] = mean_field(rows, field)
        result["navigation_mean_" + field] = mean_field(nav, field)
    component_names = sorted({key for row in rows for key in row.get("score_components", {})})
    result["mean_score_components"] = {
        key: mean_field([row.get("score_components", {}) for row in rows], key)
        for key in component_names}
    return result


def episode_signature(row):
    parameters = row.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        return None
    if row.get("source_run_id") is None or parameters.get("navigation_slot_permutation") is None:
        return None
    return json.dumps({"source_run_id": row["source_run_id"], "parameters": parameters},
                      sort_keys=True, allow_nan=False)


def verify_policy_pairs(groups):
    if not all(isinstance(groups.get(policy), list) and groups[policy] for policy in POLICIES[:3]):
        return {"verified": False, "reason": "BC/PPO/BT episode rows are incomplete"}
    seeds = [[row.get("seed") for row in groups[policy]] for policy in POLICIES[:3]]
    if any(None in sequence for sequence in seeds) or any(sequence != seeds[0] for sequence in seeds[1:]):
        return {"verified": False, "reason": "Policy seed sequences differ or are missing"}
    if len(set(seeds[0])) != len(seeds[0]):
        return {"verified": False, "reason": "Duplicate episode seeds"}
    for i, seed in enumerate(seeds[0]):
        signatures = [episode_signature(groups[policy][i]) for policy in POLICIES[:3]]
        if signatures[0] is None or any(value != signatures[0] for value in signatures[1:]):
            return {"verified": False, "reason": f"Source/task/schedule/permutation differs or is missing at seed {seed}"}
    return {"verified": True, "episodes": len(seeds[0]), "episode_seeds": seeds[0],
            "checked": ["episode seed", "source run", "all recorded task/schedule parameters",
                        "navigation slot permutation"]}


def paired_difference(left, right):
    """Describe paired differences, without pooling episodes as independent trials."""
    if [row.get("seed") for row in left] != [row.get("seed") for row in right]:
        return {"verified": False, "reason": "Different seed order"}
    if not left or len(left) != len(right):
        return {"verified": False, "reason": "Missing or different episode counts"}
    for a, b in zip(left, right):
        signature = episode_signature(a)
        if signature is None or signature != episode_signature(b):
            return {"verified": False, "reason": "Task/source/schedule/permutation differs or is missing"}
    result = {"verified": True, "episodes": len(left), "direction": "left minus right",
        "left_only_successes": sum(a.get("success") is True and b.get("success") is False
                                   for a, b in zip(left, right)),
        "right_only_successes": sum(b.get("success") is True and a.get("success") is False
                                    for a, b in zip(left, right))}
    for field in ("score_100", "time_s", "idle_healthy_steps", "recoveries", "needless_recoveries"):
        if mean_field(left, field) is None or mean_field(right, field) is None:
            result["mean_delta_" + field] = None
            continue
        differences = np.asarray([a[field] - b[field] for a, b in zip(left, right)], dtype=float)
        result["mean_delta_" + field] = float(differences.mean())
        result[field + "_positive_zero_negative_pairs"] = [int((differences > 1e-8).sum()),
            int((np.abs(differences) <= 1e-8).sum()), int((differences < -1e-8).sum())]
    return result


def initial_actor_fingerprint(path, warnings):
    if not path.is_file():
        warnings.append("Missing initial.pt; identical BC initial actor weights cannot be checked.")
        return None
    try:
        import torch
        saved = torch.load(path, map_location="cpu", weights_only=False)
        weights = saved.get("model", {})
        actor = {key: value for key, value in weights.items() if key.startswith("actor.")}
        if not actor:
            raise ValueError("No actor.* tensors in checkpoint model")
        digest = hashlib.sha256()
        for key, tensor in sorted(actor.items()):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(key.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        return digest.hexdigest()
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        warnings.append(f"Cannot hash initial.pt actor tensors: {error}")
        return None


def load_evaluation(path, warnings, *, decisions=None):
    document = read_json(path, warnings)
    if not document:
        return None
    groups = document.get("episodes", {})
    if not isinstance(groups, dict):
        warnings.append(f"{path.name} has no episode groups.")
        groups = {}
    pairing = verify_policy_pairs(groups)
    if not pairing["verified"]:
        warnings.append(f"{path.name}: {pairing['reason']}")
    declared = document.get("seeds_predeclared_before_training")
    if pairing["verified"] and declared != pairing["episode_seeds"]:
        warnings.append(f"{path.name}: recorded seeds differ from declared seed sequence.")
        pairing = {"verified": False, "reason": "Declared/evaluated seed sequence differs"}
    failures = {name: [{key: row.get(key) for key in ("seed", "success", "time_s", "score_100",
        "terminal_reason", "stage_completed", "stages", "idle_healthy_steps", "recoveries",
        "needless_recoveries", "health_stops", "support_gate_rejections", "source_run_id", "parameters")}
        for row in rows if row.get("success") is False]
        for name, rows in groups.items() if isinstance(rows, list)}
    recorded_decisions = document.get("ppo_decisions", decisions)
    if document.get("ppo_decisions") is not None and decisions is not None and recorded_decisions != decisions:
        warnings.append(f"{path.name}: recorded checkpoint decisions differ from inferred round/summary.")
    return {"file": str(path.resolve()), "sha256": file_sha256(path),
        "split": document.get("dataset_split"), "ppo_decisions": recorded_decisions,
        "checkpoint_sha256": document.get("checkpoint_sha256"),
        "frozen_final_checkpoint": document.get("frozen_final_checkpoint"),
        "scenario_id": document.get("scenario_id"), "pairing": pairing,
        "metrics": {name: metrics(rows) for name, rows in groups.items() if isinstance(rows, list)},
        "PPO_minus_BC": paired_difference(groups.get("PPO", []), groups.get("BC", [])),
        "PPO_minus_BT": paired_difference(groups.get("PPO", []), groups.get("BT", [])),
        "failure_cases": failures, "episode_rows": groups}


def load_run(run):
    run = run.resolve()
    warnings = []
    config = read_json(run / "config.json", warnings, required=True) or {}
    summary = read_json(run / "summary.json", warnings) or {}
    status = read_json(run / "status.json", warnings) or {}
    logs = read_jsonl(run / "train.jsonl", warnings)
    validations, tests = [], []
    for path in sorted(run.glob("validation_*.json")):
        match = re.fullmatch(r"validation_(\d+)\.json", path.name)
        if match:
            value = load_evaluation(path, warnings, decisions=int(match.group(1)))
            if value:
                validations.append(value)
    # Only inspect actual saved test artifacts. Never run a policy or simulator.
    for path in sorted(run.glob("*test*.json")):
        document = read_json(path, warnings)
        if document and document.get("dataset_split") == "test" and "episodes" in document:
            value = load_evaluation(path, warnings, decisions=summary.get("ppo_decisions"))
            if value:
                tests.append(value)
    if len(tests) > 1:
        warnings.append("Multiple saved test evaluations exist; each is shown separately, with no best-result selection.")
    latest_decisions = logs[-1].get("decisions") if logs else None
    if status.get("status") == "training":
        warnings.append("Training was still in progress when this snapshot was read.")
    if not validations:
        warnings.append("No saved validation evaluation exists.")
    if not tests:
        warnings.append("No saved test evaluation exists; no test result was generated.")
    if summary.get("status") == "completed" and latest_decisions != summary.get("ppo_decisions"):
        warnings.append("Completed summary and final training log decision counts differ.")
    if tests and summary.get("ppo_decisions") != latest_decisions:
        warnings.append("Test checkpoint decision count is not confirmed against the latest log.")
    final_checkpoint_sha = file_sha256(run / "final.pt") if (run / "final.pt").is_file() else None
    for test in tests:
        test["matches_current_final_checkpoint"] = bool(final_checkpoint_sha and
            test.get("checkpoint_sha256") == final_checkpoint_sha)
        if not test["matches_current_final_checkpoint"]:
            warnings.append(f"{Path(test['file']).name}: test checkpoint fingerprint is missing or differs from final.pt.")
    recall = {"postBC": config.get("bc_report", {}).get("training_label_accuracy_by_group")}
    recall_files = []
    for name in ("policy_bc_recall.json", "bc_recall.json", "bc_diagnostics.json"):
        path = run / name
        if path.is_file():
            recall[name] = read_json(path, warnings)
            recall_files.append(str(path))
    round_differences = []
    for earlier, later in zip(validations, validations[1:]):
        round_differences.append({"earlier_decisions": earlier["ppo_decisions"],
            "later_decisions": later["ppo_decisions"], "direction": "later PPO minus earlier PPO",
            **paired_difference(later["episode_rows"].get("PPO", []),
                                earlier["episode_rows"].get("PPO", []))})
    return {"run": str(run), "algorithm": config.get("algorithm"), "seed": config.get("seed"),
        "scenario_id": config.get("scenario_id"), "status": status.get("status", "unknown"),
        "latest_decisions": latest_decisions, "completed_training_episodes": summary.get("completed_training_episodes"),
        "bc_decisions": config.get("bt_imitation_decisions"),
        "bc_action_groups": config.get("bc_action_groups"),
        "bc_recall": recall, "bc_recall_files": recall_files,
        "bc_labels_sha256": config.get("bc_labels_sha256"),
        "initial_actor_sha256": initial_actor_fingerprint(run / "initial.pt", warnings),
        "dataset_sha256": config.get("dataset_sha256"),
        "implementation_sha256": config.get("implementation_sha256"),
        "optimizer": config.get("optimizer"),
        "anchor_kl_coefficient": config.get("anchor_kl_coefficient"),
        "anchor_ce_coefficient": config.get("anchor_ce_coefficient"),
        "final_checkpoint_sha256": final_checkpoint_sha,
        "validation": validations, "test": tests, "validation_round_differences": round_differences,
        "training_metrics": logs,
        "summary": summary, "warnings": warnings}


def compare_algorithms(runs):
    result = []
    seeds = sorted({run["seed"] for run in runs if isinstance(run["seed"], int)})
    for seed in seeds:
        candidates = {algorithm: [run for run in runs if run["seed"] == seed and run["algorithm"] == algorithm]
                      for algorithm in ("vanilla", "anchored")}
        if not all(len(value) == 1 for value in candidates.values()):
            result.append({"training_seed": seed, "verified": False,
                           "reason": "Need exactly one vanilla and one anchored run per training seed"})
            continue
        vanilla, anchored = candidates["vanilla"][0], candidates["anchored"][0]
        checks = {key: vanilla.get(key) is not None and vanilla.get(key) == anchored.get(key)
                  for key in ("scenario_id", "bc_labels_sha256", "initial_actor_sha256", "dataset_sha256",
                              "implementation_sha256")}
        comparisons = []
        for split in ("validation", "test"):
            v_by_key = {(e["ppo_decisions"], tuple(e["pairing"].get("episode_seeds", []))): e
                        for e in vanilla[split]}
            for evaluation in anchored[split]:
                key = (evaluation["ppo_decisions"], tuple(evaluation["pairing"].get("episode_seeds", [])))
                other = v_by_key.get(key)
                if other is None:
                    comparisons.append({"split": split, "ppo_decisions": key[0], "verified": False,
                                        "reason": "No matching saved evaluation round/seed sequence"})
                    continue
                difference = paired_difference(evaluation["episode_rows"].get("PPO", []),
                                               other["episode_rows"].get("PPO", []))
                comparisons.append({"split": split, "ppo_decisions": key[0],
                    "direction": "anchored PPO minus vanilla PPO", **difference})
        result.append({"training_seed": seed, "verified": all(checks.values()),
                       "checks": checks, "comparisons": comparisons,
                       "causal_limit": "Anchored package also changes actor learning rate; this is not an isolated KL or CE ablation."})
    return result


def fmt(value, digits=2):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and math.isfinite(value) else "—"


def ratio(row, numerator, denominator):
    return f"{row.get(numerator, 0)}/{row.get(denominator, 0)}"


def make_markdown(comparison):
    lines = ["# UC 局部观测支撑课程 v2 对照报告", "", comparison["scope_zh"], "",
        "结果只来自已保存的评估文件；本脚本不执行环境、策略或新测试。导航任务与安全等待任务分别计数。", "",
        "## 已执行训练", "", "| 算法 | 训练 seed | 状态 | PPO 决策 | 训练任务数 | BC 标签数 |",
        "|---|---:|---|---:|---:|---:|"]
    for run in comparison["runs"]:
        lines.append(f"| {run['algorithm']} | {run['seed']} | {run['status']} | {run['latest_decisions']} | "
                     f"{run['completed_training_episodes']} | {run['bc_decisions']} |")
    for split, title in (("validation", "验证集（按预定轮次）"), ("test", "固定最终测试")):
        lines += ["", f"## {title}", "",
            "| 算法 / seed | PPO 决策 | 策略 | 总完成 | 导航完成 | 安全等待 | 平均分 / 100 | 导航用时 s | 健康且无活动目标 WAIT 步数 | 恢复 / 无效恢复 |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        any_results = False
        for run in comparison["runs"]:
            for evaluation in run[split]:
                any_results = True
                for policy in POLICIES:
                    row = evaluation["metrics"].get(policy)
                    if row is None:
                        continue
                    lines.append(f"| {run['algorithm']} / {run['seed']} | {evaluation['ppo_decisions']} | {policy} | "
                        f"{ratio(row, 'task_successes', 'episodes')} | {ratio(row, 'navigation_successes', 'navigation_episodes')} | "
                        f"{ratio(row, 'safe_wait_successes', 'safe_wait_episodes')} | {fmt(row['mean_score_100'])} | "
                        f"{fmt(row['navigation_mean_time_s'])} | {fmt(row['navigation_mean_idle_healthy_steps'])} | "
                        f"{fmt(row['mean_recoveries'])} / {fmt(row['mean_needless_recoveries'])} |")
                if not evaluation["pairing"]["verified"]:
                    lines += ["", f"配对核验未通过：{evaluation['pairing']['reason']}。", ""]
        if not any_results:
            lines.append("| — | — | 尚无保存结果 | — | — | — | — | — | — | — |")
    lines += ["", "WAIT 列仅统计导航 episode；该旧计数未排除恢复冷却等无法选点的状态，不能全部解释为错误等待。分数按全部任务平均，导航用时按导航任务平均。", "",
        "## 同一训练内的预定轮次对照", "",
        "以下只比较同一验证任务上的后轮 PPO 减前轮 PPO；没有选择验证得分最好的 checkpoint。", "",
        "| 算法 / seed | 前轮→后轮决策 | 任务配对 | 分数差 | 用时差 s | 健康无目标 WAIT 步数差 |",
        "|---|---:|---|---:|---:|---:|"]
    for run in comparison["runs"]:
        for delta in run["validation_round_differences"]:
            lines.append(f"| {run['algorithm']} / {run['seed']} | {delta['earlier_decisions']}→{delta['later_decisions']} | "
                f"{delta['verified']} | {fmt(delta.get('mean_delta_score_100'))} | {fmt(delta.get('mean_delta_time_s'))} | "
                f"{fmt(delta.get('mean_delta_idle_healthy_steps'))} |")
    lines += ["", "## BC 训练标签的分意图覆盖", "",
        "下表是原始采样分布上的 recall（teacher 标签内正确比例），不是验证／测试泛化精度；训练使用类别平衡抽样。", "",
        "| 算法 / seed | 意图 | 原始标签数 | postBC recall |", "|---|---|---:|---:|"]
    for run in comparison["runs"]:
        recall = run["bc_recall"].get("postBC") or {}
        for intent in INTENTS:
            row = recall.get(intent, {})
            lines.append(f"| {run['algorithm']} / {run['seed']} | {intent} | {row.get('count', '—')} | {fmt(row.get('accuracy'), 4)} |")
    lines += ["", "最终 PPO 分意图诊断若存在，原始值保存在 `comparison.json` 的 `bc_recall` 字段及其列出的输入文件中；未保存的值不补算。", "",
        "## 同训练 seed 的算法配对", "",
        "检查相同数据哈希、BC 标签哈希、初始 BC actor 权重、环境实现以及每个评估任务的来源、参数、外生故障日程和候选槽位置换。", "",
        "| 训练 seed | 训练输入一致 | split / 决策数 | 评估配对 | anchored−vanilla 分数 | 用时 s | 健康无目标 WAIT 步数 | 单方完成 anchored / vanilla |",
        "|---:|---|---|---|---:|---:|---:|---:|"]
    for paired in comparison["algorithm_pairs"]:
        if not paired.get("comparisons"):
            lines.append(f"| {paired['training_seed']} | {paired['verified']} | — | — | — | — | — | — |")
        for result in paired.get("comparisons", []):
            lines.append(f"| {paired['training_seed']} | {paired['verified']} | {result['split']} / {result['ppo_decisions']} | "
                f"{result['verified']} | {fmt(result.get('mean_delta_score_100'))} | {fmt(result.get('mean_delta_time_s'))} | "
                f"{fmt(result.get('mean_delta_idle_healthy_steps'))} | {result.get('left_only_successes', '—')} / {result.get('right_only_successes', '—')} |")
    lines += ["", "这里逐训练 seed 展示配对差值；不把重复使用相同局部地图和任务种子的结果合并成独立样本。anchored 方案同时改变 actor 学习率，不能将差异单独归因于 KL 或辅助 CE。", "",
        "## 解释边界", "", *[f"- {limit}" for limit in comparison["limits_zh"]], "",
        "## 文件及告警", ""]
    for run in comparison["runs"]:
        lines.append(f"- `{run['run']}`")
        for warning in run["warnings"]:
            lines.append(f"  - {warning}")
    return "\n".join(lines) + "\n"


def plot_training(comparison, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.7))
    fig.subplots_adjust(left=.06, right=.985, bottom=.08, top=.815, wspace=.25, hspace=.38)
    panels = (("recent_score_100", "Recent training score / 100", (0, 102)),
              ("recent_task_success_rate", "Recent task completion", (0, 1.02)),
              ("entropy", "Policy entropy", None),
              ("value_loss", "Critic loss", None),
              ("kl_to_frozen_BC", "Forward KL to frozen BC (anchor batches)", None),
              ("needless_recovery_actions", "Needless recoveries per rollout", None))
    colors = {"vanilla": "#c46a21", "anchored": "#1673a3"}
    seeds = sorted({run["seed"] for run in comparison["runs"] if isinstance(run["seed"], int)})
    for run in comparison["runs"]:
        label = f"{run['algorithm']} seed {run['seed']}"
        color = colors.get(run["algorithm"], "#666666")
        style = "-" if not seeds or run["seed"] == seeds[0] else "--"
        for axis, (field, title, bounds) in zip(axes.flat, panels):
            pairs = [(row.get("decisions"), row.get(field)) for row in run["training_metrics"]]
            pairs = [(x, y) for x, y in pairs if isinstance(x, (int, float)) and isinstance(y, (int, float))
                     and math.isfinite(x) and math.isfinite(y)]
            if pairs:
                x, y = np.asarray(pairs).T
                axis.plot(x, y, label=label, color=color, linestyle=style, linewidth=1.2, alpha=.85)
            axis.set_title(title, fontsize=10)
            axis.set_xlabel("PPO decisions")
            axis.grid(alpha=.22)
            if bounds:
                axis.set_ylim(*bounds)
        validation_points = [(e["ppo_decisions"], e["metrics"].get("PPO", {}).get("mean_score_100"))
                             for e in run["validation"]]
        validation_points = [(x, y) for x, y in validation_points if x is not None and y is not None]
        if validation_points:
            x, y = np.asarray(validation_points).T
            axes[0, 0].scatter(x, y, marker="x", s=55, color=color, zorder=5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .93), ncols=4, fontsize=9)
    fig.suptitle("UC v2 local observed-support proxy: fixed-round optimizer comparison\n"
                 "Lines: training windows. Crosses: recorded validation. Not match win rate.", fontsize=12, y=.985)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/reports/uc_support_v2_20261116")
    args = parser.parse_args()
    runs = [load_run(path) for path in args.runs]
    comparison = {"artifact_type": "UC_support_v2_recorded_optimizer_comparison",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope_zh": "原图实际观测支撑快照上的 UC 局部多段选点、等待与恢复课程；执行器为日志校准代理。",
        "evaluation_executed_by_report": False,
        "limits_zh": ["新验证／测试 seeds 仍复用少量局部观测地图，不能证明新地图泛化。",
            "每段目标由课程指定并带可见目标特征；槽位置换消除了固定下标捷径，但尚不考验多个合理战术目标的取舍。",
            "训练内环没有实时 Point-LIO、Nav2 或 MPPI；导航完成率不是实车导航成功率。",
            "没有对手、交火、正式比赛阶段、基地回血、买弹或团队经济模型；课程分数不是比赛胜率。",
            "固定 100 分仅是任务评估标尺，RL 优化的是训练回报；两者不等价。",
            "同训练 seed 下按相同任务配对，训练 seeds 分开呈现；不作独立样本合并显著性声明。"],
        "runs": runs, "algorithm_pairs": compare_algorithms(runs),
        "report_script_sha256": file_sha256(Path(__file__))}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False,
                                                     indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text(make_markdown(comparison))
    plot_training(comparison, output / "training_curves.png")
    print(json.dumps({"output_dir": str(output), "runs": len(runs),
        "paired_training_seeds": [p["training_seed"] for p in comparison["algorithm_pairs"] if p["verified"]],
        "warnings": {Path(run["run"]).name: run["warnings"] for run in runs}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
