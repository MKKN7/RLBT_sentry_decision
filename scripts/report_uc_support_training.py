#!/usr/bin/env python3
"""Plot and report the UC local frozen-support task proxy without match claims."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


LABELS = ("untrainedRL", "postBC_RL", "finalPPO_RL", "BT")
DISPLAY = {"untrainedRL": "Untrained", "postBC_RL": "Post BC",
    "finalPPO_RL": "Final PPO", "BT": "BT"}


def read_json(path, warnings):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError) as error:
        warnings.append(f"{path.name} could not be read: {error}")
        return {}


def read_metrics(path, warnings):
    if not path.exists():
        warnings.append("train.jsonl is absent; no training samples were invented.")
        return []
    result = []
    for index, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError:
            warnings.append(f"Skipped incomplete/malformed train.jsonl line {index}.")
            continue
        if isinstance(value, dict) and isinstance(value.get("decisions"), (float, int)):
            result.append(value)
    return result


def load_run(run):
    warnings = []
    documents = {name: read_json(run / f"{name}.json", warnings)
        for name in ("status", "summary", "config", "evaluation", "export_verification")}
    metrics = read_metrics(run / "train.jsonl", warnings)
    latest_decisions = int(metrics[-1]["decisions"]) if metrics else None
    evaluated_decisions = documents["summary"].get("ppo_decisions")
    evaluation_available = bool(documents["evaluation"].get("summary"))
    current_evaluation = evaluation_available and evaluated_decisions is not None and evaluated_decisions == latest_decisions
    if evaluation_available and not current_evaluation:
        warnings.append(f"Evaluation belongs to the recorded {evaluated_decisions} decision summary; latest training log is {latest_decisions}. It is not presented as the current checkpoint result.")
    status = documents["status"].get("status", documents["summary"].get("status", "unknown"))
    if status == "training":
        warnings.append("Training was still in progress when this report snapshot was read.")
    return {"run": str(run), **documents, "metrics": metrics,
        "latest_decisions": latest_decisions, "evaluated_decisions": evaluated_decisions,
        "evaluation_matches_latest_log": current_evaluation, "status_name": status,
        "warnings": warnings}


def finite_number(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def formatted(value, precision=3):
    return f"{value:.{precision}f}" if finite_number(value) else "—"


def ratio(group):
    if not group or group.get("episodes", 0) == 0:
        return "—"
    return f"{group.get('successes', 0)}/{group['episodes']}"


def plot_run(run, snapshot, comparison=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(13.8, 7.1), constrained_layout=True)
    panels = [
        ("recent_task_success_rate", "Recent task success fraction", (0., 1.02)),
        ("recent_score_100", "Recent training episode score / 100", (0., 102.)),
        ("recent_return", "Recent episode reward return", None),
        ("entropy", "Policy entropy", None),
        ("value_loss", "Value loss", None),
        ("mean_step_reward", "Mean transition reward", None),
    ]
    for axis, (key, title, bounds) in zip(axes.flat, panels):
        for document, color, label in ((snapshot, "#246393", "Current run"),
                (comparison, "#d58b35", "Comparison")):
            if document is None:
                continue
            points = [(row["decisions"], row[key]) for row in document["metrics"]
                if finite_number(row.get(key))]
            if points:
                x, y = zip(*points)
                axis.plot(x, y, color=color, linewidth=1.3, label=label)
        axis.set(title=title, xlabel="Cumulative PPO decisions")
        axis.grid(alpha=.23)
        if bounds:
            axis.set_ylim(*bounds)
        if not axis.lines:
            axis.text(.5, .5, "No recorded samples", ha="center", va="center", transform=axis.transAxes)
        if comparison is not None and axis.lines:
            axis.legend(fontsize=8)
    figure.suptitle("UC local frozen-support execution proxy | score and reward are separate | not match wins", fontsize=12)
    figure.savefig(run / "training_curves.png", dpi=160)
    plt.close(figure)


def paired_run_difference(current, baseline):
    result = {"available": False,
        "method": "Pair final PPO episodes by seed and source run, with identical recorded scenario and parameter values; do not infer pairing from aggregate means."}
    a_eval, b_eval = current["evaluation"], baseline["evaluation"]
    a_config, b_config = current["config"], baseline["config"]
    if (not a_config.get("dataset_sha256") or not b_config.get("dataset_sha256")
            or a_config["dataset_sha256"] != b_config["dataset_sha256"]
            or a_config.get("implementation_sha256", {}).get("rl_bt/uc_support_env.py")
            != b_config.get("implementation_sha256", {}).get("rl_bt/uc_support_env.py")):
        result["reason"] = "Recorded dataset hashes or executor environment implementation differ or are missing."
        return result
    a = a_eval.get("episodes", {}).get("finalPPO_RL", [])
    b = b_eval.get("episodes", {}).get("finalPPO_RL", [])
    if not a or not b:
        result["reason"] = "One or both final PPO episode ledgers are absent."
        return result
    if a_eval.get("dataset_split") != b_eval.get("dataset_split") or a_eval.get("scenario_id") != b_eval.get("scenario_id"):
        result["reason"] = "Evaluation split or scenario identity differs."
        return result
    indexed = {(row.get("seed"), row.get("source_run_id")): row for row in b}
    if len(indexed) != len(b) or len(a) != len(b):
        result["reason"] = "Episode counts differ or comparison episode identities are duplicated."
        return result
    score, time, success = [], [], []
    for row in a:
        peer = indexed.get((row.get("seed"), row.get("source_run_id")))
        if peer is None or row.get("parameters") != peer.get("parameters"):
            result["reason"] = "Episode identity or exogenous parameter descriptors differ."
            return result
        if not all(finite_number(r.get(k)) for r in (row, peer) for k in ("score_100", "time_s")):
            result["reason"] = "Episode score or time is missing."
            return result
        score.append(row["score_100"] - peer["score_100"])
        time.append(row["time_s"] - peer["time_s"])
        success.append(int(row["success"]) - int(peer["success"]))
    delta = np.asarray(score)
    result.update(available=True, episodes=len(a), mean_score_delta=float(delta.mean()),
        mean_time_delta_s=float(np.mean(time)), success_count_delta=int(np.sum(success)),
        positive=int((delta > 1e-6).sum()), negative=int((delta < -1e-6).sum()),
        tied=int((np.abs(delta) <= 1e-6).sum()))
    return result


def table_lines(snapshot):
    evaluation = snapshot["evaluation"]
    summaries = evaluation.get("summary", {})
    if not summaries:
        return ["尚无 evaluation.json 的已记录评估；本报告不补造结果。", ""]
    lines = [f"评估 split：`{evaluation.get('dataset_split', 'unknown')}`；对应已记录 PPO decisions：`{snapshot['evaluated_decisions']}`。",
        "", "| 策略 | 任务成功 | 导航成功 | 等待成功 | 平均 score/100 | 平均时间/s | 平均 reward return |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name in LABELS:
        row = summaries.get(name)
        if row is None:
            continue
        lines.append(f"| {DISPLAY[name]} | {ratio(row)} | {ratio(row.get('navigation_only'))} | {ratio(row.get('wait_only'))} | {formatted(row.get('mean_score_100'))} | {formatted(row.get('mean_time_s'))} | {formatted(row.get('mean_return'))} |")
    lines += ["", "Score 是预定义的 100 分任务评价，reward return 是 PPO 优化回报；二者不可互换。", ""]
    differences = evaluation.get("paired_score_differences", {})
    if differences:
        lines += ["| 配对比较 | 平均 score 差 | 较高/较低/相同 |", "|---|---:|---:|"]
        for label, row in differences.items():
            lines.append(f"| {label} | {formatted(row.get('mean_score_delta'), 6)} | {row.get('positive', '—')}/{row.get('negative', '—')}/{row.get('tied', '—')} |")
        lines.append("")
    ppo_bc = differences.get("finalPPO_minus_postBC_RL", {})
    ppo_bt = differences.get("finalPPO_minus_BT", {})
    if ppo_bc.get("mean_score_delta") is not None and ppo_bc["mean_score_delta"] <= 1e-6:
        lines += ["当前配对评价未显示 Final PPO 相对 Post BC 的 score 提升。", ""]
    if ppo_bt.get("mean_score_delta") is not None and ppo_bt["mean_score_delta"] <= 1e-6:
        lines += ["当前配对评价未显示 Final PPO 相对 BT 的 score 提升。", ""]
    scenarios = sorted({scenario for row in summaries.values() for scenario in row.get("by_scenario", {})})
    if scenarios:
        lines += ["| 场景 | 策略 | 成功/任务数 | 平均 score/100 |", "|---|---|---:|---:|"]
        for scenario in scenarios:
            for name in LABELS:
                row = summaries.get(name, {}).get("by_scenario", {}).get(scenario)
                if row:
                    lines.append(f"| {scenario} | {DISPLAY[name]} | {row.get('successes', '—')}/{row.get('n', '—')} | {formatted(row.get('mean_score_100'))} |")
        lines.append("")
    return lines


def export_lines(export):
    if not export:
        return ["未发现 export_verification.json；不报告 ONNX 验证通过。", ""]
    return ["导出验证原始记录：", "", "```json", json.dumps(export, ensure_ascii=False, indent=2), "```", ""]


def report(run, comparison_run=None):
    run = run.resolve()
    if not run.is_dir():
        raise ValueError(f"Run directory does not exist: {run}")
    current = load_run(run)
    baseline = load_run(comparison_run.resolve()) if comparison_run is not None else None
    plot_run(run, current, baseline)
    comparison = {"artifact_type": "UC_local_frozen_support_training_report_not_match_result",
        "run": str(run), "status": current["status_name"],
        "latest_logged_ppo_decisions": current["latest_decisions"],
        "evaluation_ppo_decisions": current["evaluated_decisions"],
        "evaluation_matches_latest_log": current["evaluation_matches_latest_log"],
        "within_evaluation_paired_score_differences": current["evaluation"].get("paired_score_differences", {}),
        "comparison_run": None if baseline is None else baseline["run"],
        "between_runs_finalPPO": {"available": False, "reason": "No --comparison-run supplied."}
            if baseline is None else paired_run_difference(current, baseline),
        "warnings": current["warnings"] + ([] if baseline is None else baseline["warnings"]),
        "input_sha256": {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
            for name in ("train.jsonl", "evaluation.json", "summary.json", "export_verification.json")
            if (run / name).exists()},
        "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    lines = ["# UC 支撑观测课程训练报告", "",
        "范围：局部冻结 observed-support 图与校准执行代理；不是完整 UC 比赛训练，也不是在线 Point-LIO/Nav2/MPPI 训练。", "",
        f"Run：`{run}`。状态：`{current['status_name']}`；日志 PPO decisions：`{current['latest_decisions']}`。", "",
        f"已记录训练更新：{len(current['metrics'])}；任务成功率属于课程任务，不是比赛胜率。", "",
        "![训练曲线](training_curves.png)", ""]
    if comparison["warnings"]:
        lines += ["读取快照说明：", ""] + [f"- {warning}" for warning in comparison["warnings"]] + [""]
    lines += table_lines(current)
    test_result = read_json(run / "evaluation_test.json", [])
    if test_result.get("summary"):
        lines += ["预登记最终 checkpoint 的一次性 held-out test：", ""]
        test_snapshot = {**current, "evaluation": test_result,
            "evaluated_decisions": test_result.get("ppo_decisions")}
        lines += table_lines(test_snapshot)
    if baseline is not None:
        lines += [f"比较 run：`{baseline['run']}`。", "", "```json",
            json.dumps(comparison["between_runs_finalPPO"], ensure_ascii=False, indent=2), "```", ""]
    lines += export_lines(current["export_verification"])
    lines += ["数据边界：各 run 使用同一局部场地区域；run split 不构成新地图泛化证据。冻结图代理的零违规计数不能替代实车或 Nav2 物理安全验证。", ""]
    (run / "training_report.md").write_text("\n".join(lines))
    (run / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--comparison-run", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.run, args.comparison_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
