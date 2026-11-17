# 实际训练中的 actor 有效样本与近场观测诊断

2026-11-17。继续原始UC地图、Point-LIO＋Nav2＋MPPI的局部位置任务。机器人、物理参数、MPPI参数、.411m圆足迹、.45m自车点云过滤、固定起点先验和unknown阻挡保持。

## 本次修正

`rl_bt/live_actor_learning.py`替代live trainer此前复用的proxy updater，共享proxy训练源不改。只有已处理的RL提议且mask含多个合法选项的转移参与on-policy actor loss、熵、KL和优势归一化。仅一条有效选择时保留未归一化优势，避免中心化将其清零。

`action_applied=false`的RPC终态反馈及强制单选仍完整进入按实际simulation duration计算的GAE、reward和critic。延迟失败可沿时间链向之前实际选择传递信用。BT否决仍是已经处理的提议，保留其actor信用。没有实际选择且无有效anchor时跳过actor Adam step，避免旧momentum移动权重。

BC anchor仍只接受经过实际执行证据校验的训练示范。anchor只采有选择空间的标签，CE／KL梯度与on-policy梯度分开记录；actor及critic更新次数分别统计。未将评估数据用于BC或调参。

`--simulation-seed-base`将每局的显式非负seed传至fixture，配置与实际fixture中均存档。它控制原仿真已有的sensor noise随机流，保持原分布及物理参数。ROS调度、Point-LIO和MPPI不因此变成完全配对的确定性物理过程。报告逐局核验实际seed，不能只依据命令行声称已生效。

## 过滤前点云取证

新增可选`--capture-terrain-input`，保存最多80帧经过estimated pose逐点注册、尚未经过.45m过滤的XYZ、样本／接收时间和估计位姿，逐文件hash入fixture。该文件不进入actor或reward，不生成额外训练转移。

诊断run `runs/uc_live_terrain_diagnostic_20261117`保留71帧，实际BT到点成功；明确标记`diagnostic_only_not_training`。启动前5s，之前两块unknown列`(-4,-1)`与`(-4,0)`在过滤前的返回点数都是0。按实际提交批次、接收时间和TTL重放7次分类，基线全部计数吻合；离线将自车过滤半径从.45m改成.411m后所有占用数组仍完全一致。因此不修改过滤半径或坡度门限。

真实点云未观测到的近场地面不能通过调小过滤半径生成。原始雷达方向／安装几何也提示后方近地存在可视范围限制，但该几何推断不替代真实点云证据。起点和长路线支撑缺口、静止漂移仍未解决。不能移动先验或填充unknown来提升完成率。

复现诊断：

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
 /home/kellen/anaconda3/envs/robot/bin/python scripts/report_live_terrain_inputs.py \
 --execution runs/uc_live_terrain_diagnostic_20261117/session/episode_000/execution \
 --output-dir data/reports/live_nav2_validity_20261117
```

## 固定训练与评估设计

计划在采样前写入`configs/live_nav2_validity_20261117.json`：目标交替`(0,.3)`／`(0,.4)`m，原r2真实BT示范（仅6条有选择标签）热启动BC200，PPO8局／最多352次交互，sensor seeds31000～31007，actor seed271。每个启动都计入，未丢弃支撑失败。

冻结评估分别载入固定`initial.pt`与`final.pt`，各4局，双方都用41000～41003这组独立sensor seeds；使用相同目标顺序。两个模型都只greedy推理、0次SGD，评估不选checkpoint。seed顺序相同仍不等于严格物理配对。

实际训练完成8局、108条转移、14个更新batch，5/8到点。26条是有效actor选择，82条强制单选；6个batch有on-policy actor样本，8个只有critic／anchor贡献。28次actor optimizer step中16次仅anchor；全部108条仍用于critic。实际MPPI指令310条，305条非零。postBC到final的actor L2变化约.05597，包含anchor贡献。

5/8不是比赛胜率，也不能直接与上一轮1/8比较并归因于算法：本轮sensor seeds及启动支撑状态不同，任务仍只有.3／.4m。冻结独立评估、逐局支撑与学习计数见`data/reports/live_nav2_validity_20261117/report.md`。

独立冻结评估已完成：BC初始模型3/4、PPO最终模型2/4；两组均4/4至少一次可执行导航，0次SGD，权重hash未变，实际sensor seeds均为41000～41003。失败均触发当前足迹unknown的安全取消。对训练及两组评估合计135条相同已记录观测，两个模型的greedy动作完全一致。概率有微小变化，但当前没有选点行为改善证据，完成数之差也不能归因于PPO退化；未见过的状态不在此结论范围。

零命令漂移复核确认，原执行器持续收到零目标，是P-only速度wrench控制而无位置保持。原始出生点局部mesh坡度约1.55°、接触摩擦系数.02，符合旧日志约8～9mm/s缓慢下滑；这只是具有源码和几何支持的物理机制，未精确分解每个接触贡献。详见`data/reports/live_nav2_validity_20261117/idle_drift_diagnosis.md`。本轮不改控制器，后续应先记录contact／wrench／姿态，再单独验证执行层保持控制及释放，而非重复发零速或使用GT定点。

运行目录：

- `runs/uc_live_actor_validity_train_20261117`：BC＋真实PPO。
- `runs/uc_live_actor_validity_bc_eval_20261117`：冻结BC评估。
- `runs/uc_live_actor_validity_ppo_eval_20261117`：冻结PPO评估。

本次全量回归278项通过，含actor／critic梯度隔离、单动作Adam状态不变、未执行终态时序信用、anchor角色校验及seed透传测试。测试中的mock严格不计真实Nav2证据。未扩大RL权限，尚未完成正式UL／UC对抗训练或NUC验收。

所有正式run的原图hash、插件加载、observed支撑、无GT输入、源快照及实际fixture seeds核验通过；原模型／物理及shared Point-LIO配置保持。结束后domain79节点清单为空。
