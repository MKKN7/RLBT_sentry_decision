# UC 局部任务课程 v2 优化与对照

完成日期：2026-11-17。四组 CPU 实验分别完成 2048 次 train-only BT imitation 和 32,768 次 PPO，两个训练种子 271／809，各比较 vanilla／anchored。合计 131,072 次 PPO、4373 个完整训练任务；同种子的两算法使用完全相同的 BC 标签和初始 actor 权重。原 v1 源码、训练、评分与 checkpoint 保留。

本轮没有完整比赛、自瞄、射击、姿态、回血或经济训练；没有启动 ROS／GPU。仍复用 13 张原始 UC 场地局部支撑观测图和日志校准执行代理，Nav2／MPPI 不在 PPO 内环。

## 改动依据

复现 v1 已知 validation 失败 85024、85025：两例健康且有导航候选时持续 WAIT，累计 153 次错误等待；相同状态下原 BC 会选导航。v1 的 BC 2048 标签中，recover 仅 17 条，BC 和 PPO 对这 17 条均预测为 continue；99.17% 总体 accuracy 掩盖了恢复类完全漏学。

代码还确认：v1 critic 的 value loss 经 actor 的 state encoder 反向传播；观察得到的梯度路径不等于已经证明失败因果。另一个不一致是环境明示有限 60 s 截止，却把失败期限作为 continuing-task truncation 进行 value bootstrap。

v2 独立环境 `rl_bt/uc_support_env_v2.py` 做以下修订：

- deadline 为真正失败终止，bootstrap=0，terminal potential=0，额外 −3×未完成比例。训练回报版本改变，100 分评估和完成标准不变。
- 每局打乱导航候选槽位，目标不再固定下标 0；继续候选加入实际可见的活动目标／剩余路径特征。置换 RNG 独立，不改变任务、观测噪声和故障日程。
- 不屏蔽健康 WAIT，保留 BT 原来的 unknown／足迹／时效门禁；不能靠新增 mask 隐藏策略错误。

v2 模型 `rl_bt/uc_support_policy_v2.py` 分离 actor／critic encoder，critic 的优化器覆盖其所有参数。actor 仍为 12,449 参数的小型候选 MLP，部署体积不增加。

训练器 `scripts/train_uc_support_v2.py` 对两个算法统一做四类均衡 BC：navigate、recover、continue、hold，报告原始分布的各类 recall。种子 271 的四类训练 recall 均 100%；809 的 continue 为 99.38%，其他均 100%。恢复类两组各只有 10 条，拟合通过不能证明广泛恢复能力。

## 对照设置

配置预登记于 `configs/uc_support_training_schedule_v2.json`；固定 16,384／32,768 步验证，固定最终 32,768 步 checkpoint 测试。没有按 test 选择最好 checkpoint。

| 项目 | vanilla | anchored |
|---|---:|---:|
| actor Adam LR | 3e−4 | 1e−4 |
| critic Adam LR | 3e−4 | 3e−4 |
| 冻结 BC 前向 KL 系数 | 0 | 0.05 |
| train-only BT 标签辅助 CE 系数 | 0 | 0.05 |

辅助 minibatch 按上述四类平衡，取自同一份训练 BC 数据。它约束梯度，不替换 actor 提议动作。比较是学习率＋KL＋CE 的组合优化，不能单独归因于 KL 或某一改动。v1 到 v2 也同时改变了终止语义、编码器和 BC，不能用前后分数分配因果贡献。

优化抽样 RNG 与 episode-seed RNG 分开，同种子两算法的第 N 个训练 episode seed 一致；由于策略影响任务长度，固定决策预算下训练 episode 数仍可不同。评估使用逐任务相同的来源、目标序列、槽位置换及外生噪声／故障日程，报告逐训练种子的配对差值，不把重复局部地图当独立样本池。

validation 为 105001～105032，test 为 115001～115064，均排除训练；旧验证／测试种子也排除训练。仍复用 v1 的 run-family 地图划分，新 seed 不等于新地图或新实测数据。

## 实际结果

| 训练 seed | 策略 | 验证完成／32 | 验证分／100 | 测试完成／64 | 测试分／100 |
|---:|---|---:|---:|---:|---:|
| 271 | BC | 31 | 93.41 | 64 | 93.38 |
| 271 | vanilla PPO | 31 | 93.41 | 64 | 93.38 |
| 271 | anchored PPO | 31 | 93.41 | 64 | 93.38 |
| 809 | BC | 31 | 92.46 | 64 | 92.42 |
| 809 | vanilla PPO | 31 | 93.07 | 64 | 93.26 |
| 809 | anchored PPO | 31 | 93.41 | 64 | 93.38 |
| 相同任务 | 纯 BT | 31 | 93.41 | 64 | 93.38 |

种子 809 的测试里，无效恢复次数从 BC 的 17 次降到 vanilla 的 2 次、anchored 的 0 次；总恢复次数分别为 19／9／8，BT 为 8。anchored 相比该 BC 平均分 +0.967，平均任务时间约少 0.25 s；相比 vanilla 仅 +0.126 分，另一个训练 seed 差值为零。证据支持局部调度错误减少，没有证明超过 BT，也不支持普遍优越或统计显著性结论。

四组最终 PPO 在新 validation／test 任务中，健康且无活动目标的 WAIT 计数均为零。四组对已知 v1 失败任务 85024、85025 也均完成 5/5 阶段，分别用时 44.5／51.0 s，健康错误 WAIT 为零；回归另存在 `known_failure_regression.json`，只作已知缺陷检查，不再称独立保留测试。

validation 唯一未完成任务为 105027，所有最终 PPO、BC 和 BT 都完成 4/5 阶段后耗尽 60 s；该任务走廊总长度约 9.79 m，并发生一次支撑失更停车。保存的末段轨迹持续移动，没有健康错误 WAIT。该结果不应被删除或改 deadline 以补成满分。

本轮 validation／test 没有 no-safe-goal 地图样本，安全等待的分母为 0；其门禁仍由 train 与单元测试覆盖，不能报告 held-out 安全等待泛化。

## 产物与复现

四个 run 位于 `runs/uc_support_v2_{vanilla|anchored}_s{271|809}_20261116/`。目录名保留预登记日期，实际完成于 2026-11-17。每组保存 `initial.pt`、`round_00016384.pt`、`round_00032768.pt`、`final.pt`、完整环境/RNG/optimizer 状态、BC 数据、源码与数据 hash；`validation_*.json` 和 `evaluation_test.json` 保存所有逐任务结果。

以下为实际命令形式；已有目录禁止用 fresh run 覆盖：

```bash
cd /home/kellen/桌面/RL_BT_decision
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/kellen/anaconda3/envs/robot/bin/python scripts/train_uc_support_v2.py \
  --algorithm anchored --seed 271 --decisions 32768 --bc-decisions 2048 \
  --run runs/uc_support_v2_anchored_s271_20261116 --eval-split validation
```

同式替换 algorithm 和 seed 完成四组。每组最终 test 已由 `scripts/evaluate_uc_support_v2.py` 一次性记录，该脚本拒绝覆盖已有测试文件。`scripts/export_uc_support_actor_v2.py` 导出两组 anchored actor，并完成各 100 组 ONNX/PyTorch CPU logits／argmax／mask 检查；每个 actor 53,598 bytes，未在 NUC 或实机 BT ROS adapter 验收。

完整回归 **247 passed**，覆盖独立 critic 梯度、稀少类采样、有限截止、实际环境快照和下一段 rollout＋双 optimizer 更新的精确恢复。报告、曲线和全部配对证据见 `data/reports/uc_support_v2_20261117/`；精确文件清单见 `data/reports/uc_support_v2_changes_20261117.json`。

## 下一阶段的有效增量

本课程目标由环境指定，候选中目标位／距离已明确提示正确 waypoint。槽位置换只消除固定下标捷径，尚未考验抢点、撤退、支援等多任务冲突。继续增加该课程步数很难证明全局策略收益。后续应新增真正有时效、机会成本和多个合理选择的任务课程，并复用现有导航执行数据；正式基地、阵地和裁判坐标注册必须先核验，再接入经济和对抗目标。现有 BT 安全权限保持。
