#!/usr/bin/env python3
"""Report copied real navigation artifacts without importing or starting ROS.

Incomplete runs remain explicitly pending. Training demonstrations, stochastic
training, and greedy evaluations are reported separately, never pooled as a
match win rate or treated as identical seeded physical trials.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPECTED_FIELD_HASH = "93ff0118af176f388c171b239acfdcac72c17cbe0b523350d5d916d3236fe10b"
MPPI_PLUGIN_LINE = "Created controller : FollowPath of type nav2_mppi_controller::MPPIController"


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def read_jsonl(path):
    if not path.is_file():
        return []
    rows = []
    for index, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"Malformed JSONL {path}:{index}") from exc
    return rows


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_or_none(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def action_group(action):
    return {13: "recover", 14: "continue", 15: "hold"}.get(action, "navigate")


def copied_runtime(execution):
    fixture_path = execution / "fixture/result.json"
    fixture = read_json(fixture_path, {})
    controller = execution / "nav2/controller_server.log"
    controller_text = controller.read_text() if controller.is_file() else ""
    commands = fixture.get("commands", [])
    valid_commands = [row for row in commands if isinstance(row, list) and len(row) >= 4]
    nonzero = sum(math.hypot(float(row[1]), float(row[2])) > 1e-6 for row in valid_commands)
    loaded = MPPI_PLUGIN_LINE in controller_text
    no_gt = fixture.get("GT_localization_input") is False
    observed = fixture.get("terrain_prior") == "observed"
    field_hash = fixture.get("field_source", {}).get("prepared_mesh_sha256")
    source_files = {}
    source_manifest = execution / "source_sha256.json"
    source_errors = []
    for relative, expected in read_json(source_manifest, {}).items():
        path = execution / "source" / relative
        if not path.is_file() or sha256(path) != expected:
            source_errors.append(relative)
        else:
            source_files[relative] = expected
    return {
        "execution_dir": str(execution),
        "fixture_sha256": sha256(fixture_path) if fixture_path.is_file() else None,
        "controller_log_sha256": sha256(controller) if controller.is_file() else None,
        "runtime_verified": bool(loaded and no_gt and observed and field_hash == EXPECTED_FIELD_HASH),
        "controller_plugin_loaded": loaded, "GT_localization_input": not no_gt,
        "observed_support": observed, "prepared_mesh_sha256": field_hash,
        "MPPI_command_samples": len(commands), "nonzero_MPPI_command_samples": nonzero,
        "estimated_pose_samples": len(fixture.get("estimated_poses", [])),
        "sensor_randomization_seed": fixture.get("sensor_randomization_seed"),
        "source_snapshot_manifest_present": source_manifest.is_file(),
        "source_snapshot_files_verified": len(source_files), "source_snapshot_errors": source_errors,
    }, fixture


def episode_evidence(episode, transitions, runtime, fixture):
    execution = Path(runtime["execution_dir"])
    events_path = execution / "session/events.jsonl"
    result_path = execution / "session/result.json"
    events = read_jsonl(events_path)
    session = read_json(result_path, {})
    controller_path = execution / "nav2/controller_server.log"
    controller_errors = [line for line in controller_path.read_text().splitlines()
                         if "[ERROR]" in line] if controller_path.is_file() else []
    raw = session.get("raw", {})
    stops = [event for event in events if event.get("event") == "safety_stop_latched"]
    terminal_events = [event for event in events if event.get("event") == "navigation_terminal_result"]
    counts = Counter(event.get("event", "missing") for event in events)
    task_error = raw.get("task_goal_distance_m", episode.get("final_estimated_goal_error_m"))
    actual_success = bool(session.get("mission_success") and
                          raw.get("terminal_status") == "succeeded" and
                          task_error is not None and float(task_error) <= .2)
    selected = [row for row in transitions if row.get("episode") == episode.get("episode")]
    action_counts = Counter(action_group(int(row["proposed_action"])) for row in selected)
    forced_hold = [row for row in selected if row.get("observation", {}).get("mask") ==
                   [False] * 15 + [True]]
    navigation_choices = [row for row in selected if
        any(row.get("observation", {}).get("mask", [])[:13]) or
        (len(row.get("observation", {}).get("mask", [])) > 14 and
         row["observation"]["mask"][14])]
    reasons = Counter(str(event.get("reason", "unspecified")) for event in stops)
    cancel_reasons = Counter(str(event.get("reason", "unspecified")) for event in events
                             if event.get("event") == "cancel_requested")
    unsupported = [event for event in events if event.get("event") == "path_support_check" and
                   event.get("accepted") is False]
    footprint_failures = []
    for row in selected:
        info_raw = row.get("info", {}).get("raw", {})
        check = info_raw.get("diagnostics", {}).get("current_footprint_check", {})
        if check.get("accepted") is False:
            footprint_failures.append({"decision": row.get("decision"),
                "sim_time_s": info_raw.get("now_s"), "health": info_raw.get("health"),
                "health_reason": info_raw.get("health_reason"), "check": check,
                "pose_age_s": info_raw.get("pose_age_s"), "tf_age_s": info_raw.get("tf_age_s"),
                "support_age_s": info_raw.get("support_age_s"),
                "obstacle_age_s": info_raw.get("obstacle_age_s")})
    return {
        "episode": episode.get("episode"), "summary_success": episode.get("success"),
        "actual_task_success": actual_success,
        "success_evidence_agrees": bool(episode.get("success")) == actual_success,
        "nav_status": raw.get("terminal_status", episode.get("nav_status")),
        "goal_xy_m": raw.get("task_goal_xy_m", episode.get("goal_xy_m")),
        "time_s": episode.get("time_s"), "wall_s": episode.get("wall_s"),
        "return": episode.get("return"), "steps": len(selected),
        "final_estimated_goal_error_m": task_error,
        "action_group_counts": dict(action_counts), "safety_stop_reasons": dict(reasons),
        "terminal_before_action_admission": sum(row.get("info", {}).get("action_applied") is False
                                                 for row in selected),
        "only_hold_legal_decisions": len(forced_hold),
        "only_hold_legal_elapsed_s": sum(float(row.get("elapsed_s", 0)) for row in forced_hold),
        "at_least_one_executable_navigation_choice": bool(navigation_choices),
        "decisions_with_executable_navigation_choice": len(navigation_choices),
        "decisions_with_multiple_legal_choices": sum(sum(bool(value) for value in
            row.get("observation", {}).get("mask", [])) > 1 for row in selected),
        "cancel_reasons": dict(cancel_reasons),
        "task_timeout": any("time_limit" in reason or "timeout" in reason for reason in reasons),
        "policy_hold_events": counts["RL_hold"],
        "policy_recover_cancellations": cancel_reasons["RL_recover_requested"],
        "policy_hold_cancellations": cancel_reasons["RL_hold_requested"],
        "unsupported_path_count": len(unsupported),
        "unsupported_path_reasons": dict(Counter(str(event.get("reason")) for event in unsupported)),
        "current_footprint_failures": footprint_failures,
        "final_raw_health": {key: selected[-1].get("info", {}).get("raw", {}).get(key)
                             for key in ("now_s", "health", "health_reason", "pose_age_s", "tf_age_s",
                                         "support_age_s", "obstacle_age_s")} if selected else None,
        "final_current_footprint_check": selected[-1].get("info", {}).get("raw", {}).get(
            "diagnostics", {}).get("current_footprint_check") if selected else None,
        "distractor_arrivals": sum(event.get("event") == "engineering_waypoint_arrived" and
                                   not event.get("mission_success") for event in events),
        "navigation_terminal_events": terminal_events,
        "BT_veto_count": session.get("counts", {}).get("BT_veto_count"),
        "session_event_counts": dict(counts), "runtime": runtime,
        "controller_error_log_lines": controller_errors,
        "estimated_trajectory_source": str(execution / "fixture/result.json"),
        "events_source": str(events_path),
        "session_result_sha256": sha256(result_path) if result_path.is_file() else None,
        "events_sha256": sha256(events_path) if events_path.is_file() else None,
        "trajectory": [row[:3] for row in fixture.get("estimated_poses", [])
                       if isinstance(row, list) and len(row) >= 3],
    }


def classify_role(config):
    if not config:
        return "unknown_pending"
    if config.get("evaluation_only"):
        return "greedy_evaluation"
    if config.get("data_role") == "train_demonstration":
        return "BT_training_demonstrations"
    return "PPO_training" if config.get("policy") == "rl" else "BT_control"


def summarize(run):
    run = run.resolve()
    config = read_json(run / "config.json", {})
    summary = read_json(run / "summary.json")
    status = read_json(run / "status.json", {})
    if summary is None:
        return {"run": str(run), "name": run.name, "status": "pending",
                "pending_reason": "summary.json has not been written", "live_status": status,
                "role": classify_role(config), "episodes": [], "integrity_errors": []}
    transitions = read_jsonl(run / "transitions.jsonl")
    updates = read_jsonl(run / "updates.jsonl")
    recorded_runtime = read_json(run / "runtime_evidence.json", [])
    actual_runtime, fixtures = [], {}
    for fixture_path in sorted((run / "sessions").glob("**/fixture/result.json")):
        execution = fixture_path.parent.parent
        row, fixture = copied_runtime(execution)
        actual_runtime.append(row)
        fixtures[str(execution.resolve())] = (row, fixture)
    episodes = []
    unreported_sessions = []
    errors = []
    for episode in summary.get("episodes", []):
        execution = str(Path(episode.get("execution_dir", "")).resolve())
        if execution not in fixtures:
            errors.append(f"Episode {episode.get('episode')} copied fixture is missing")
            continue
        row, fixture = fixtures[execution]
        episodes.append(episode_evidence(episode, transitions, row, fixture))
    reported_executions = {str(Path(episode.get("execution_dir", "")).resolve())
                           for episode in summary.get("episodes", [])}
    for execution, (row, _) in fixtures.items():
        if execution not in reported_executions:
            session = read_json(Path(execution) / "session/result.json", {})
            unreported_sessions.append({"execution_dir": execution,
                "runtime_verified": row["runtime_verified"],
                "mission_success": session.get("mission_success"),
                "terminal_status": session.get("raw", {}).get("terminal_status"),
                "reason": "Actual copied session absent from trainer completed-episode list; not silently excluded"})
    verified_rpc = sum(bool(row.get("remote_evidence", {}).get("verified")) for row in transitions)
    actual_commands = sum(row["MPPI_command_samples"] for row in actual_runtime)
    actual_nonzero = sum(row["nonzero_MPPI_command_samples"] for row in actual_runtime)
    for name, actual in (("decisions", len(transitions)), ("ppo_updates", len(updates)),
                         ("verified_live_rpc_transitions", verified_rpc),
                         ("MPPI_command_samples", actual_commands),
                         ("nonzero_MPPI_command_samples", actual_nonzero)):
        if summary.get(name) != actual:
            errors.append(f"summary {name}={summary.get(name)} differs from artifacts {actual}")
    if [(row.get("fixture_sha256"), row.get("runtime_verified")) for row in recorded_runtime] != [
            (row["fixture_sha256"], row["runtime_verified"]) for row in actual_runtime]:
        errors.append("runtime_evidence.json differs from copied fixture/controller evidence")
    if any(not episode["success_evidence_agrees"] for episode in episodes):
        errors.append("Episode success differs from copied session result")
    if any(row["source_snapshot_errors"] for row in actual_runtime):
        errors.append("Copied execution source snapshot has mismatches")
    success_count = sum(episode["actual_task_success"] for episode in episodes)
    if success_count != summary.get("successes"):
        errors.append("Summary successes differs from copied session task success count")
    copied_source_errors = []
    for relative, expected in config.get("implementation_sha256", {}).items():
        path = run / "source" / relative
        if not path.is_file() or sha256(path) != expected:
            copied_source_errors.append(relative)
    if copied_source_errors:
        errors.append("Training source snapshot has mismatches")
    if config.get("evaluation_only"):
        if updates or summary.get("weights_unchanged") is not True or \
                summary.get("greedy_actions_only") is not True:
            errors.append("Evaluation does not prove frozen greedy weights with zero SGD updates")
    declared_seeds = config.get("simulation_seed_schedule")
    if declared_seeds is not None:
        for episode in episodes:
            index = int(episode["episode"])
            seed = episode["runtime"].get("sensor_randomization_seed")
            if index >= len(declared_seeds) or seed != declared_seeds[index]:
                errors.append(f"Episode {index} runtime sensor seed differs from declared schedule")
    runtime_verified = bool(transitions and verified_rpc == len(transitions) and actual_runtime and
                            all(row["runtime_verified"] for row in actual_runtime) and not errors)
    groups = Counter(action_group(int(row["proposed_action"])) for row in transitions)
    bc_path = run / "bc_report.json"
    bc = read_json(bc_path)
    for candidate in ("bc_summary.json", "BC_summary.json", "bc_metrics.json", "bc_fit.json"):
        if bc is None:
            bc = read_json(run / candidate)
    if bc is None:
        bc = summary.get("BC", summary.get("bc", config.get("BC", config.get("bc"))))
    checkpoints = {name: {"path": str(run / name), "sha256": sha256(run / name)}
                   for name in ("untrained.pt", "initial.pt", "final.pt", "BC_teacher.pt")
                   if (run / name).is_file()}
    update_choice_signal = []
    for update in updates:
        end = int(update["decisions"])
        start = end - int(update["rollout_decisions"])
        rows = [row for row in transitions if start < int(row["decision"]) <= end]
        multi = sum(sum(bool(value) for value in row["observation"]["mask"]) > 1 for row in rows)
        update_choice_signal.append({"update": update["update"], "rollout_decisions": len(rows),
            "multi_action_observations": multi, "single_action_observations": len(rows)-multi,
            "sampled_log_probability_absolute_max": max((abs(float(row.get("log_probability", 0)))
                                                         for row in rows), default=None)})
    return {"run": str(run), "name": run.name, "status": summary.get("status"),
        "role": classify_role(config), "policy": summary.get("policy"),
        "evaluation_only": bool(config.get("evaluation_only", False)),
        "checkpoint_loaded": config.get("checkpoint_loaded", config.get("historical_checkpoint_loaded")),
        "goal_schedule": config.get("goal_schedule_xy_m", config.get("goal_schedule", config.get("goal_xy_m"))),
        "successes": success_count, "completed_episodes": len(episodes),
        "episodes_with_executable_navigation_choice": sum(
            episode["at_least_one_executable_navigation_choice"] for episode in episodes),
        "successes_with_executable_navigation_choice": sum(episode["actual_task_success"] and
            episode["at_least_one_executable_navigation_choice"] for episode in episodes),
        "only_hold_legal_decisions": sum(episode["only_hold_legal_decisions"] for episode in episodes),
        "only_hold_legal_elapsed_s": sum(episode["only_hold_legal_elapsed_s"] for episode in episodes),
        "frozen_evaluation_weights_unchanged": summary.get("weights_unchanged"),
        "mean_time_s_all_episodes": mean_or_none(episode["time_s"] for episode in episodes),
        "mean_success_time_s": mean_or_none(episode["time_s"] for episode in episodes
                                            if episode["actual_task_success"]),
        "mean_return": mean_or_none(episode["return"] for episode in episodes),
        "actual_transition_elapsed_s": sum(float(row.get("elapsed_s", 0)) for row in transitions),
        "decisions": len(transitions), "ppo_updates": len(updates),
        "MPPI_command_samples": actual_commands, "nonzero_MPPI_command_samples": actual_nonzero,
        "runtime_verified": runtime_verified, "verified_live_rpc_transitions": verified_rpc,
        "runtime_verified_session_count": sum(row["runtime_verified"] for row in actual_runtime),
        "action_group_counts": dict(groups), "actor_weight_change_l2": summary.get("actor_weight_change_l2"),
        "terminal_before_action_admission": sum(row.get("info", {}).get("action_applied") is False
                                                 for row in transitions),
        "PPO_actor_weight_change_l2": summary.get("PPO_actor_weight_change_l2"),
        "actor_learning_revision": summary.get("actor_learning_revision"),
        "actor_learning_counts": summary.get("actor_learning_counts"),
        "simulation_seed_schedule": declared_seeds,
        "critic_weight_change_l2": summary.get("critic_weight_change_l2"),
        "BC_training_fit_only": bc, "episodes": episodes, "updates": updates,
        "BC_training_data_proof": read_json(run / "live_BT_demonstrations_proof.json"),
        "checkpoints": checkpoints,
        "PPO_update_choice_signal": update_choice_signal,
        "recorded_physical_sessions": len(actual_runtime),
        "unreported_physical_sessions": unreported_sessions,
        "integrity_errors": errors, "training_source_errors": copied_source_errors,
        "source_sha256": {name: sha256(run / name) for name in
            ("config.json", "summary.json", "transitions.jsonl", "updates.jsonl", "runtime_evidence.json",
             "bc_report.json", "live_BT_demonstrations_proof.json", "live_BT_demonstrations.npz")
            if (run / name).is_file()},
    }


def make_plots(reports, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for report in reports:
        episodes = report["episodes"]
        if not episodes or report["status"] != "complete":
            continue
        label = report["name"].removeprefix("uc_live_").removesuffix("_20261117")
        xs = np.arange(1, len(episodes) + 1)
        cumulative = np.cumsum([episode["actual_task_success"] for episode in episodes]) / xs
        axes[0, 0].plot(xs, cumulative, "o-", label=label)
        axes[0, 1].plot(xs, [episode["time_s"] for episode in episodes], "o-", label=label)
        axes[1, 0].plot(xs, [episode["final_estimated_goal_error_m"] for episode in episodes],
                        "o-", label=label)
        updates = report.get("updates", [])
        if updates:
            axes[1, 1].plot([row["decisions"] for row in updates],
                            [row.get("value_loss", np.nan) for row in updates], "o-", label=label)
    axes[0, 0].set(title="Cumulative actual local task completion", xlabel="Episode", ylabel="Fraction", ylim=(-.05, 1.05))
    axes[0, 1].set(title="Episode duration (success and failure)", xlabel="Episode", ylabel="Simulation seconds")
    axes[1, 0].set(title="Final Point-LIO task goal error", xlabel="Episode", ylabel="Meters")
    axes[1, 1].set(title="Actual PPO value loss", xlabel="Collected live decisions", ylabel="Loss")
    for axis in axes.flat:
        axis.grid(alpha=.2)
        if axis.lines:
            axis.legend(fontsize=7)
    fig.suptitle("Original UC / Point-LIO / real Nav2 + MPPI; short local tasks only", fontsize=11)
    fig.savefig(output_dir / "live_training_progress.png", dpi=150)
    plt.close(fig)
    present = [report for report in reports if report["episodes"] and report["status"] == "complete"]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(4.5 * len(present), 4.5),
                             squeeze=False, constrained_layout=True)
    for axis, report in zip(axes[0], present):
        for episode in report["episodes"]:
            points = np.asarray(episode["trajectory"], float)
            if len(points):
                axis.plot(points[:, 1], points[:, 2], label=f"ep {episode['episode']}: " +
                          ("success" if episode["actual_task_success"] else str(episode["nav_status"])))
            goal = episode["goal_xy_m"]
            if goal is not None:
                axis.scatter(goal[0], goal[1], marker="*", color="black", s=70)
        axis.set(title=report["name"].removeprefix("uc_live_").removesuffix("_20261117"),
                 xlabel="Session estimated x (m)", ylabel="Session estimated y (m)")
        axis.axis("equal"); axis.grid(alpha=.2); axis.legend(fontsize=7)
    fig.suptitle("Actual estimated trajectories; independent noisy physical sessions, not identical seeds")
    fig.savefig(output_dir / "estimated_trajectories.png", dpi=150)
    plt.close(fig)


def write_markdown(report, output_dir):
    lines = ["# 实际 Nav2＋MPPI 训练进展", "", "仅统计原始 UC 地图局部工程任务。BT 示范、PPO 训练和 greedy 评估分开报告；这些数字不是比赛胜率。", "",
        "各组使用相同目标顺序，逐局sensor seed见下文；物理进程、传感器队列与估计器调度并非严格配对，不能解释为完全相同物理随机过程的统计检验。BC 标签准确率仅为训练拟合程度。", "",
        "| Run | 用途 | 完成 | 平均仿真秒（全部任务） | 实际反馈转移 | PPO 更新 | MPPI 指令（非零） | 证据核验 |",
        "|---|---|---:|---:|---:|---:|---:|---|" ]
    for row in report["runs"]:
        if row["status"] == "pending":
            lines.append(f"| `{row['name']}` | {row['role']} | pending | — | — | — | — | 尚未产出 summary |")
            continue
        duration = row["mean_time_s_all_episodes"]
        time_text = "—" if duration is None else f"{duration:.3f}"
        completion = f"{row['successes']}/{row['completed_episodes']}"
        if row["status"] != "complete":
            completion += f"；{row['status']}"
        lines.append(f"| `{row['name']}` | {row['role']} | {completion} | "
            f"{time_text} | {row['decisions']} | {row['ppo_updates']} | {row['MPPI_command_samples']} "
            f"({row['nonzero_MPPI_command_samples']}) | {'通过' if row['runtime_verified'] else '未通过'} |")
    evaluations = [row for row in report["runs"] if row.get("evaluation_only") and row["status"] == "complete"]
    if evaluations:
        cohort = "；".join(f"{row['name']}：{row['episodes_with_executable_navigation_choice']}/{row['completed_episodes']}局存在可执行导航选项"
            for row in evaluations)
        lines += ["", "冻结评估均使用 greedy action，权重未变化且没有 SGD。" + cohort + "。"
                  "当前小样本的物理调度未严格配对，不能仅据完成数差异推断 PPO 带来提升或退化。"]
    for row in report["runs"]:
        if row["status"] == "pending":
            continue
        lines += ["", f"## `{row['name']}`", "",
                  f"动作计数：`{json.dumps(row['action_group_counts'], ensure_ascii=False)}`。"]
        lines += ["", f"完整记录中 {row['episodes_with_executable_navigation_choice']} 局至少一次有可执行导航选项，"
                  f"其中完成 {row['successes_with_executable_navigation_choice']} 局；"
                  f"仅 HOLD 合法共 {row['only_hold_legal_decisions']} 步／{row['only_hold_legal_elapsed_s']:.3f} s。"
                  "该可执行子集仅用于定位门禁瓶颈，不替换上表全部任务分母。"]
        if row["terminal_before_action_admission"]:
            lines += ["", f"其中 {row['terminal_before_action_admission']} 条转移在 action 派发前已进入终态，"
                      "保留终态反馈但未执行所提议动作。"]
        if row["integrity_errors"]:
            lines += ["", "证据不一致："] + [f"- {error}" for error in row["integrity_errors"]]
        if row["unreported_physical_sessions"]:
            lines += ["", f"实际复制了 {row['recorded_physical_sessions']} 个物理会话；"
                      f"另有 {len(row['unreported_physical_sessions'])} 个会话未进入 trainer 完整 episode 列表。"
                      "上表分母仅为 trainer 完整记录，不能忽略这些异常会话或据此声称全部成功。"]
        for episode in row["episodes"]:
            if not episode["actual_task_success"]:
                lines += ["", f"- episode {episode['episode']}：`{episode['nav_status']}`；"
                    f"安全停机原因 `{json.dumps(episode['safety_stop_reasons'], ensure_ascii=False)}`；"
                    f"取消原因 `{json.dumps(episode['cancel_reasons'], ensure_ascii=False)}`；"
                    f"仅 HOLD 合法 {episode['only_hold_legal_decisions']} 步／{episode['only_hold_legal_elapsed_s']:.3f} s；"
                    f"不支持路径 {episode['unsupported_path_count']} 次；"
                    f"非任务点到达 {episode['distractor_arrivals']} 次。"]
                if episode["current_footprint_failures"]:
                    check = episode["current_footprint_failures"][-1]
                    lines += [f"  最后足迹诊断：`{check['check'].get('reason')}`，"
                              f"cell=`{check['check'].get('failure_cell_xy')}`，"
                              f"普通传感器健康状态=`{check['health']}`，"
                              f"odom／TF age=`{check['pose_age_s']}`／`{check['tf_age_s']}` s，"
                              f"support／obstacle age=`{check['support_age_s']}`／`{check['obstacle_age_s']}` s。"]
                    if "localization_invalid" in episode["safety_stop_reasons"]:
                        lines += ["  generic `localization_invalid` 同时覆盖足迹支撑门禁，不能据此认定 Point-LIO 失效。"]
                if episode["controller_error_log_lines"]:
                    lines += ["  实际 controller 错误：`" + episode["controller_error_log_lines"][-1] + "`。"]
        if row.get("BC_training_fit_only") is not None:
            lines += ["", "BC 训练拟合记录：", "", "```json",
                      json.dumps(row["BC_training_fit_only"], ensure_ascii=False, indent=2), "```"]
        if row.get("actor_learning_counts") is not None:
            lines += ["", "实际 actor／critic 学习计数（actor 梯度与 anchor 分开记录）：", "", "```json",
                json.dumps(row["actor_learning_counts"], ensure_ascii=False, indent=2), "```"]
        if row.get("simulation_seed_schedule") is not None:
            lines += ["", f"声明 sensor noise seeds：`{row['simulation_seed_schedule']}`。已逐局核对fixture；"
                "相同sensor seed不保证相同物理调度、Point-LIO或MPPI随机轨迹。"]
        signals = row["PPO_update_choice_signal"]
        if signals:
            forced_updates = [item for item in signals if item["multi_action_observations"] == 0]
            choices = sum(item["multi_action_observations"] for item in signals)
            lines += ["", f"{len(signals)} 次 PPO 更新中 {len(forced_updates)} 次 rollout 全是单一合法动作；"
                      f"累计只有 {choices} 条转移含多选项。单一动作 categorical 的 actor policy gradient 为零，"
                      "此时 actor 变化可能来自显式 BC 辅助项，不能据权重变化声称 PPO 已改善导航。"]
    lines += ["", "实际运行链路由复制的 controller 插件日志、原图 hash、observed 支撑、GT 关闭与 RPC 转移共同核验；命令数只计 fixture 实际记录。", "",
              "checkpoint／源代码 hash、逐局安全停机事件和估计轨迹来源见 `comparison.json`。", "",
              "图仅展示完整批次；中断首尝试保留在上表及异常会话证据中。", "",
              "![实际训练进展](live_training_progress.png)", "", "![Point-LIO 轨迹](estimated_trajectories.png)", ""]
    (output_dir / "report.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize(run) for run in args.runs]
    report = {"artifact_type": "actual_live_nav2_mppi_training_progress_not_match_score",
              "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "physical_runs_identically_seeded": False,
              "full_match_training_validated": False,
              "report_script_sha256": sha256(Path(__file__)), "runs": rows}
    make_plots(rows, args.output_dir)
    # Trajectories are already fully preserved in the copied fixture; retain
    # their provenance and avoid duplicating large raw pose arrays in this report.
    for row in rows:
        for episode in row["episodes"]:
            episode.pop("trajectory", None)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, ensure_ascii=False,
                                                   indent=2, allow_nan=False) + "\n")
    write_markdown(report, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()),
        "runs": [{key: row.get(key) for key in ("name", "status", "successes", "completed_episodes",
                    "decisions", "ppo_updates", "runtime_verified", "integrity_errors")} for row in rows]},
        ensure_ascii=False, allow_nan=False))
    return report


if __name__ == "__main__":
    main()
