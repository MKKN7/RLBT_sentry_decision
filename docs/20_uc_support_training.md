# UC 实际观测约束的多阶段任务课程

2026-11-16。完成两轮 CPU 训练：2048 次 BT imitation、累计 65,536 次 PPO，2110 个完整训练任务。训练入口为 `scripts/train_uc_support.py`，环境为 `rl_bt/uc_support_env.py`。本课程学习局部候选点、继续、等待和恢复；它不是完整 UC 对抗环境，也未把 Point-LIO／Nav2／MPPI 放入 PPO 内环。

## 数据与执行边界

`data/derived/uc_support_training_v1/` 包含 13 张实际支撑观测图，190 个候选通过半径 0.411 m 连续圆足迹路径检查。原始来源为 `runs/uc_support*`，每张保留运行名、文件 hash、时间、估计位姿、固定起点先验和图来源。图均来自用户指定原始 UC 场地，其 prepared STL SHA256 为 `93ff0118af176f388c171b239acfdcac72c17cbe0b523350d5d916d3236fe10b`。

数据处理没有使用 GT／STL 补充 free cells。unknown 保持 unknown；历史运行中已经显式使用的固定工程起点先验保留并标注。某些末尾估计位姿的足迹不受支撑，课程改从原图中既有的固定工程起点 reset，metadata 明示为 synthetic reset；不会把当前位姿周围重新涂成 free。没有可行候选的一张图只允许安全等待。

同一 run family 的所有快照只属于一个 split；未发现精确重复或跨 split 重复。train 5 张、validation 3 张、test 5 张，仍是同一工程出生邻域，不能证明新区域或全场泛化。validation seeds 为 85001～85032，test 为 95001～95064；训练排除这些种子。held-out 图和执行记录不用于拟合训练速度。

执行代理以训练侧两条成功日志的 Point-LIO 估计位移速度校准，中位数约 0.1851 m/s。接受时延的仿真时间差存在量化为零的情况，不能推断真实通信时延为零。本轮额外设置的 0.1～0.8 s 接受时延、0.8～1.8 s 延迟情景、0.65～1.25 倍速度、8% 支撑更新丢失、8 mm 估计位置特征噪声以及 3～6 s 支撑缺失／卡滞，都是显式工程压力分布，不是比赛统计概率。原始统计和样本数保存在 `execution_calibration.json`。

支撑图为冻结观测，更新缺失按模拟时间采样；不会合成更大范围的 free map。代理中的移动沿已核验的起点—候选点走廊，换点先返回起点再走另一条边，每一段实际移动仍调用连续足迹门禁。位姿噪声只影响 actor 看到的估计特征，不是物理定位误差闭环验证。未运行对手、伤害、经济、买血买弹、回血、姿态、开火和瞄准模型，原机器人与原导航工程没有修改。

## 多阶段任务与评分

每个有导航候选的任务包含 2～5 个有序工程 waypoint，2 Hz 决策，最多 120 步／60 s。目标序列在 reset 时固定。动作 0～12 选候选点、13 请求取消／恢复、14 继续、15 等待。BT 保留候选 mask、支撑新鲜度与连续路径否决权，支撑失更时强制停止。恢复不会清除 unknown。无合法点时等待任务在 5 s 后单独计为完成；评估分开列导航与等待的分母，不能用等待成功代替导航成功。

评分固定为 100 分，供多轮比较，**与训练 reward 分开**：

| 分项 | 上限 | 计算 |
|---|---:|---|
| 任务进度 | 60 | 已完成阶段数／总阶段数 |
| 时间效率 | 20 | 进度 × min(1, 名义时间／实际时间) |
| 恢复纪律 | 10 | 进度／(1 + 无效恢复次数 + 0.2 × 无效切换次数) |
| 代理安全 | 10 | 1／(1 + BT 否决次数 + 已执行违规次数) |

名义时间由固定任务走廊长度、校准速度与每阶段 0.2 s 名义接受时延计算。安全项在正确 mask 下通常恒为 10；它不能解释为物理安全验证。导航任务始终等待只能得 10 分。恢复和等待没有可重复刷取的完成奖励。

训练 reward：每完成阶段 +1.5，整项完成额外 +3，每决策 −0.02，BT 否决 −0.15，切换 −0.06，有用恢复 −0.03、无效恢复 −0.15。距离 shaping 使用固定任务序列的 `Phi=-0.5*剩余走廊距离` 和 `gamma*Phi(next)-Phi(now)`，gamma=0.99／decision。真正完成的 terminal potential=0；时间上限记为 truncation，保留末状态 value bootstrap，GAE 不跨 reset 传播。GAE lambda=0.95。未因测试结果修改奖励或评分。

## 网络、训练和复现

部署 actor 为共享 state MLP、候选 MLP、共享 scorer，输入 state 24、候选 16×12、boolean mask 16；12,449 参数。UC 独立 critic 附加合法候选 mean/max 和合法比例，不导出部署。actor 复用 UL 网络结构，权重、scenario、checkpoint 均独立，没有加载 UL／alphaRM 权重。尚无连续长轨迹支撑 GRU 训练，本轮使用显式任务时龄、上次状态、失更和停滞特征。

```bash
cd /home/kellen/桌面/RL_BT_decision
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/kellen/anaconda3/envs/robot/bin/python scripts/train_uc_support.py \
  --run runs/uc_support_multistage_32768_20261116 \
  --decisions 32768 --bc-decisions 2048 --seed 271 --eval-split validation

# --decisions 是累计目标；同 run 精确恢复，不重做 BC。
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/kellen/anaconda3/envs/robot/bin/python scripts/train_uc_support.py \
  --run runs/uc_support_multistage_32768_20261116 \
  --resume runs/uc_support_multistage_32768_20261116/final.pt \
  --decisions 65536 --bc-decisions 2048 --seed 271 --eval-split validation
```

上述路径已存在，不应再次作为 fresh run 覆写。首轮结果保存在该 run 的 `rounds/32768/`，根目录为累计 65,536 步结果，目录名保留首轮命名。多轮计划预登记在 `configs/uc_support_training_schedule_v1.json`。每 8 次 update 保存 latest；保存 model、optimizer、全部 RNG、环境、当前 observation、episode 计数及源码／数据 hash，恢复时拒绝不匹配。`untrained.pt` 只作评估基线，禁止当训练恢复点跳过 BC。

`source/` 保存本轮环境、policy、trainer、依赖的 PPO 源码和数据 JSON；data NPZ 通过 hash 关联原数据。`tb/` 有 TensorBoard events，本轮未开启服务。`tests/test_uc_support_training.py` 验证下一段 rollout 与下一次 PPO 更新的参数张量精确一致，环境测试另验证实际环境恢复、支撑门禁和 terminal／truncation。

## 实际结果

同一个验证集的固定场景、噪声与故障日程：

| 策略／轮次 | 导航完成 | 平均分／100 | 平均耗时 s |
|---|---:|---:|---:|
| 未训练网络 | 0/32 | 10.00 | 60.00 |
| BC 2048 | 32/32 | 93.09 | 24.25 |
| PPO 32,768 | 32/32 | 93.09 | 24.25 |
| PPO 65,536 | 30/32 | 91.41 | 25.02 |
| 纯 BT | 32/32 | 93.20 | 24.09 |

65,536 是预先指定的最终轮次，没有根据 test 选最好模型。最终模型一次性 held-out test：

| 策略 | 导航完成 | 平均分／100 | 平均耗时 s |
|---|---:|---:|---:|
| 未训练网络 | 0/64 | 10.00 | 60.00 |
| BC | 64/64 | 93.34 | 17.05 |
| PPO 65,536 | 64/64 | 93.34 | 17.05 |
| 纯 BT | 64/64 | 93.63 | 16.75 |

两个 held-out split 均没有 no-safe-goal 样本；等待门禁由 train 和针对性测试覆盖，不能宣称 held-out 等待泛化。最终测试 PPO 与 BC 的每个任务评分均相同；相对 BT 为 3 次较高、13 次较低、48 次相同，平均 −0.294 分。**本轮没有证明 PPO 超过 BC 或 BT**。

验证失败 seeds 85024、85025 的轨迹保存在 `validation_failure_traces.json`：分别完成 4/5、3/5 阶段后，策略在健康且有候选的状态持续选 WAIT。训练集近期满分不能排除该退化。现有数据与任务较窄，BC 已很强，下一版应先在验证侧检查长任务选择、等待退化和恢复差异；不能靠增加训练步数直接推断更强。

`evaluation_test.json` 已保存，不应重复用于调参。后续若参考这些 test 结果修改任务，应另立未用过的测试集。

## 导出与验证

`actor.onnx` 为 53,598 bytes，100 组 CPU ONNX/PyTorch batch 的 logits／argmax／mask 检查通过，最大绝对误差约 5.72e−6。当前工作站 500 次预热推理 p50 约 0.0118 ms、p95 约 0.0190 ms；未在 NUC 测量，尚未接实机 BT ROS adapter。正式部署前仍需要完整比赛任务、地图坐标注册、长路线支撑稳定性与真实闭环验收。

完整回归：229 passed。数据报告、训练日志、评估、导出验证和本轮精确变更清单见 `data/reports/uc_support_training_20261116.json` 与 `data/reports/uc_support_training_changes_20261116.json`。
