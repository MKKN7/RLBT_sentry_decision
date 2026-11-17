# Point-LIO＋Nav2＋MPPI 进入实际训练内环

2026-11-17。此前第 17～19 篇只有独立导航执行验证，第 20～21 篇 PPO 使用观测图与日志校准的执行代理；这些代理分数不能改称真实导航训练成绩。现已新增独立 live 环境，并完成实际 PPO 在线采样与参数更新。

本文保留首次fresh pilot的范围与结果。后续真实BT示范、BC热启动、8局PPO及独立冻结评估见[第23篇](23_live_nav2_continuation.md)，本轮报告位于 `data/reports/live_nav2_continue_20261117/report.md`。

## 当前链路

`scripts/train_live_nav2.py` 在 host CPU 推理／PPO → `rl_bt/live_nav_env.py` JSONL RPC → 容器 `scripts/nav2_rl_session.py` → BT 校验与实际 ComputePathToPose → 实际 NavigateToPose → Nav2 XML BT／Navfn／Omni MPPI → 原 MuJoCo 机器人。Point-LIO 估计位姿、双雷达观测支撑和 action 反馈返回下一状态。

训练转移期间没有使用速度代理、预录轨迹或 GT 位姿移动机器人。当前只做用户指定原始 UC STL 的 `(0,0.4)` 米工程短任务，不是完整比赛。原模型、MPPI 参数、.411 m 圆包络、unknown lethal 配置保持。所有 ROS 节点在私有 domain79／namespace 运行，每个 episode 重启自己持有的 fixture／估计器；不在过程中用 GT teleport。结束只清理自己创建的进程组。

## 新接口与安全边界

- `rl_bt/live_nav_observation.py` 仅从 Point-LIO、当前实际观测支撑和任务／执行状态生成 24 维状态、16×12 候选和 16 位 mask。外形相同不代表语义相同，新 scenario 为 `uc_original_live_pointlio_nav2_mppi_task_v1`，fresh 网络初始化，没有加载代理 checkpoint。
- RL 只提议候选 id、continue、hold、cancel/recover，不输出速度、瞄准、开火或热控。候选 mask 的几何预筛由 BT 侧计算；unknown 不填成 free。
- 导航派发必须收到实际 ComputePathToPose 成功结果，并用最新语义图检查实际估计起点＋完整路径的连续 .411 m 圆足迹。候选版本、时间戳和 2 s TTL 均检查，过期回调不派发。
- 执行中独立监控 support／障碍云／odom／TF 的新鲜度和当前全足迹。host idle wall watchdog 不依赖模拟时钟继续走动；信号、EOF、异常和 close 会取消／锁停。取消 ACK 不等于 terminal，未收到终态不允许替换目标。
- HOLD 会取消活动 action／待规划 generation；CONTINUE 保留当前任务。由于当前恢复动作取消后结束该 episode，尚不是实车完整脱困流程。
- 到达 distractor 不算完成；必须实际 Nav2 `succeeded`、所选目标是任务目标、Point-LIO 估计误差≤.2 m 才算成功。预期安全拒绝也不算导航成功。
- 原有固定工程起点先验仍须显式 `--start-footprint-prior`，不随机器人移动。它不是雷达证明，未扩大起点自由区。支撑图盲区和长路线稳定性问题仍保留。

ROS 会话在 host 推理／PPO 更新时继续运转，属于带实际延迟的交互过程。转移时长按前一观测到后一观测的真实 simulation clock 差计算；另记 RPC 内执行时间。reward／GAE 用实际时长折扣，不能把等待数秒当固定 .5 s。每局有限任务截止为失败 terminal；host 决策预算截断另记 truncated。

## 已实际执行

| Run | 类型 | 决策数 | PPO 更新 | 任务完成 | 实际 MPPI 指令 |
|---|---|---:|---:|---:|---:|
| `runs/uc_live_bt_commission_20261117` | BT 接线对照 | 6 | 0 | 1/1 | 70，69 条非零 |
| `runs/uc_live_ppo_pilot_20261117` | fresh RL 在线 pilot | 49 | 7 | 0/3 | 139，131 条非零 |

BT 对照末尾估计误差 0.14545 m；该 run 没有 PPO 更新，actor／critic 权重变化为零，不能充作 RL 成绩。

RL 三局分别为 canceled、failed 超时、canceled，末尾估计目标误差约 .310／.126／.209 m。其中第二局即便位置落在任务容差内，也未满足实际任务目标 action 完成条件，因此仍计失败。累计实际 transition 时长约 23.545 s；actor 权重 L2 变化 0.04543、critic 0.28729，参数均 finite。仅有 49 个样本，**没有收敛或策略有效性的结论**。

`runtime_evidence.json` 逐局校验实际 copied fixture：原始 prepared map SHA256 `93ff0118af176f388c171b239acfdcac72c17cbe0b523350d5d916d3236fe10b`、observed 支撑、GT localization=false、controller log 确认实际加载 `nav2_mppi_controller::MPPIController`，并从 fixture 计数真实命令。`real_nav2_in_loop` 同时要求这些证据和带 RPC id 的实际转移，不只依据配置开关或进程声明。

## 训练与产物

```bash
cd /home/kellen/桌面/RL_BT_decision
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/kellen/anaconda3/envs/robot/bin/python scripts/train_live_nav2.py \
  --run runs/uc_live_ppo_NEW --episodes 3 --max-decisions 64 \
  --goal 0 .4 --episode-wall-timeout 180 --task-timeout-s 20
```

输出目录必须不存在，domain79 只能同时运行一个会话。`--policy bt` 是单独的接线对照，不参与 PPO。每局启动／重建估计器和物理场景开销比代理大；本轮并行任务没有占用 GPU，也未将多个 episode 偷换为同一冻结运动模型。

trainer 使用 CPU 两线程，actor LR1e−4／critic LR3e−4，rollout8、epochs2，独立 actor／critic。每 .5 s 仿真时间请求一次反馈，实际经过时间写入转移。reward 为 −.04×实际秒数、估计任务距离的 potential shaping、任务成功 +3、终止失败 −1、BT 否决 −.15；potential=`-.5*estimated_distance`，实际 duration 的 gamma 折扣，真正 terminal potential=0。

每个 run 保存：

- `transitions.jsonl`：实际 observation、mask、提议 action/logp、下一 observation、BT 结果、实际时钟、reward 和逐步来源。
- `updates.jsonl`、`untrained.pt`／`latest.pt`／`final.pt`：实际梯度更新与模型／optimizer，保存 RNG 只方便审查，不声称恢复物理状态。
- `sessions/**/execution/`：私有源快照／hash、RPC 请求响应、逐次实际支撑 NPZ、Nav2 日志、Point-LIO 日志、fixture 的实际控制／估计位姿与 GT 诊断分栏。
- `runtime_evidence.json`：实际插件、原图、支撑配置和控制命令证据。

live 物理、估计器、传感器队列不能精确序列化，因此没有 exact physics resume；新会话重新 reset 与权重加载是不同概念。GT 诊断日志不会作为 actor／reward 位置输入。

完整源文件与产物清单见 `data/reports/live_nav2_training_changes_20261117.json`，摘要证据见 `data/reports/live_nav2_training_20261117.json`。后续正式任务／长路线／UL 地图／对手／经济训练仍需继续实现，不能把当前短路程 pilot 解释为赛季决策模型已经训好。
