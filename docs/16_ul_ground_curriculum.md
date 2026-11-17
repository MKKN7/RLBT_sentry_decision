# UL 地面任务课程与仿真车体参考包络

更新：2026-11-16。用户授权使用桌面导航仿真中的大致形状；原车体模型和共享导航工程均未修改。

## 已接通的执行链

`ULTaskEnv → 16候选及mask → 小MLP选候选 → NavigationExecutor → ULNavigation路径运动代理 → BT状态监控/恢复`。

这是基于官方来源UL参考网格的**独立地面课程**，并非完整3v3比赛环境，也未运行Point-LIO、Nav2、MPPI或Combat XML runtime。UL/UC仍分开：本轮新增UL课程训练，UC已有独立回血/预算模块，但尚无正式UC比赛训练。

- 14个固定导航候选，另有continue和hold。0～9为己方补给、中心及工程地面锚点；10～13为高地/洞口，本轮禁用。工程锚点不是官方RFID测量点。
- 0.1m地面图读取3811面UL STL；禁止取地板底面−0.1m作为支撑，保留侧墙与顶面阻挡。8连通且禁止切角；敌补给区按完整圆包络排除。
- 每0.5s决策，代理内部0.1s积分；到点、导航取消/终态、任务序号、候选版本、地图hash、坐标系、TTL及watchdog均经过独立契约。TTL只控制意图接纳；替换先等待取消终态。
- 已接延迟自状态/弹量/RFID/控制状态payload allowlist；缺失/超龄有独立valid位。定位样本排队、位置噪声与episode固定bias、执行起步延迟、速度差异及临时受阻参与课程。数值为工程初始假设，尚未用实车日志校准。
- actor不读取敌人HP、敌人全局位置、精确占点进度；`0x020B`队友坐标在UL播发及配准核验前保持屏蔽。协议payload解码不是串口驱动。
- 课程没有对手，也没有检测目标，因此**检测漏检/对手风格随机化本轮尚未实现**；不能把报文丢失称为视觉漏检。

## 参考模型净空

`scripts/probe_reference_robot_envelope.py`用拷贝配置原参数构建MuJoCo模型，读取107个机器人geom，双雷达旋转架按2°步长扫整周，变换几何AABB后合并。并未进行动态穿洞。

|量|结果|
|---|---|
|名义长×宽×高包络|0.636643 × 0.445200 × 0.234250 m|
|相对base平面边界|X −0.266～0.370643；Y ±0.222600 m|
|覆盖各geom整周yaw的保守平面半径|0.380938 m|
|规划半径及额外margin|0.391 + 0.020 = 0.411 m|
|规划高度及额外margin|0.245 + 0.010 = 0.255 m|
|参考洞底到顶板下表面|0.249500 m|

原模型名义高度剩约15.25mm；加入10mm静态悬架伸展、向上取整及额外10mm工程余量后，不满足当前保守规划包络。圆包络也可能过度保守，不能从“本轮禁用”推出实车绝对不能穿洞。带姿态/旋转架限制的非圆包络、动态悬架、侧向净空、线缆装甲与实车尺寸仍需另测；不能用原合成小盒探针的通过结果代替这里的结论。

复现：

```bash
/home/kellen/anaconda3/envs/robot/bin/python scripts/probe_reference_robot_envelope.py
/home/kellen/anaconda3/envs/robot/bin/python scripts/probe_ul_navigation.py
```

## 课程、奖励和版本隔离

60秒课程随机抽取capture、resupply、recover_then_capture及红蓝两侧。它是训练任务期限，UL正式比赛仍是300秒。低血量任务初始80～259HP；己补给按UL默认每秒25%×400HP计算，在代理1s结算点处理。回血只来自己补给激活代理；安全位置本身不回血。弹量保持750，购买与补弹全部关闭。

有效回血到360HP记一次里程碑，需占点的任务再完成中心5秒技能。为避免擅自解释中立点未占满后空置衰减，课程限定占点为BT不可由普通RL新目标/hold中断的五秒技能；执行期间仅开放continue/hold，二者均保持技能。发生定位失效/异常离区则终止课程并记录需要裁判语义核对。路径偶然经过中心不自动获得这个技能。这些均是**显式课程边界**，不是完整比赛占点模型。

每个决策的课程奖励：时间−0.005；到当前课程目标的距离减少量×0.10；需要回血时有效回血/400×0.5；首次回血里程碑+1；首次占点+2；课程完成+3；BT否决−0.02；几何失败−2。完整分项进入train.jsonl。距离差奖励不是宣称保持最优策略不变的potential shaping；以后正式比赛奖励需单独设计。

版本记录保留：

- v1 `runs/ul_ground_curriculum_8192_20261116/`：512 BC＋8192 PPO，125训练任务；预登记52001～52024中训练后RL12/24、BT24/24。回满血后反复选补给是已复现失败。此版定位延迟/有效位/原子技能存在缺项，不作当前验收。
- v2 `runs/ul_ground_curriculum_v2_32768_20261116/`：补齐观测有效位和定位延迟、增加BC；4096 BC＋32768 PPO，975训练任务。复核发现hold/BT否决回退可中断技能并遗留partial progress；此结果保留为被替代实验，不能当成v3验证。
- v3 `runs/ul_ground_curriculum_v3_32768_20261116/`：新增独立capture技能latch；4096 BC＋32768 PPO，957训练任务。复核再发现0.1s浮点累加使5秒阈值晚一个积分步，修正后另开v4，不覆盖原结果。
- v4：增加严格阈值数值容差及分步/整步等价回归测试；新schema ID `ul2027_ground_task_curriculum_v4`，新权重初始化，4096 BC＋16384 PPO，训练seed274，评估55001～55024在训练前登记；不加载旧alphaRM或前几版checkpoint。

四个独立run不合称一次连续训练。当前trainer保存RNG、源码hash/快照和optimizer，但未实现精确resume，重开实验不得称为续训。运行命令与结果见各run/config.json、summary.json、evaluation.json；最终v4结果亦汇入README及verification.json。

v4实际完成433个完整PPO训练任务、64次更新。55001～55024的配对评估覆盖红蓝与三类任务：imitation后24/24、PPO后24/24、BT24/24；组合任务为9/9。PPO后平均29.125决策步，BT30.375步，但样本小、课程简单，不能据此推广为策略优势。训练中代理几何失败0次、BT恢复57次；训练与评估种子无交集，源码快照/hash和当前训练源码均一致。

ONNX仅导出12,449参数actor、不含critic；文件53,598 bytes。100组随机batch中PyTorch/ORT的argmax及mask完全一致，最大logit误差5.73e−6以内；工作站CPU warm推理P95约0.012ms，不是NUC测量。导出及核验：

```bash
/home/kellen/anaconda3/envs/robot/bin/python scripts/report_ul_curriculum.py --run runs/ul_ground_curriculum_v4_16384_20261116
/home/kellen/anaconda3/envs/robot/bin/python scripts/export_ul_curriculum_actor.py --run runs/ul_ground_curriculum_v4_16384_20261116
```

本轮完整测试：`/home/kellen/anaconda3/envs/robot/bin/python -m pytest -q`，108 passed。原子技能hold/换点不能中断、0.1s分步与5s整段到达同一控制阈值均有回归测试。

## 可复现入口与文件

```bash
/home/kellen/anaconda3/envs/robot/bin/python scripts/train_ul_curriculum.py \
  --decisions 16384 --seed 274 --bc-decisions 4096 \
  --run runs/ul_ground_curriculum_v4_16384_20261116 \
  --eval-seeds 55001,55002,55003,55004,55005,55006,55007,55008,55009,55010,55011,55012,55013,55014,55015,55016,55017,55018,55019,55020,55021,55022,55023,55024
```

已存在run会拒绝覆盖；复现实验须另选run路径。

新增/修改的核心文件：

- `rl_bt/ul_task_env.py`、`rl_bt/ul_navigation.py`、`rl_bt/ul_curriculum_policy.py`。
- `rl_bt/navigation_contract.py`、`rl_bt/referee_protocol.py`、`rl_bt/ul_observation.py`、`rl_bt/mesh_navigation.py`。
- `rl_bt/ul_rules.py`、`tests/test_ul_rules.py`：修复控制阈值的浮点分步误差。
- `configs/reference_robot_envelope.json`、`configs/training_design.json`。
- `scripts/probe_reference_robot_envelope.py`、`scripts/probe_ul_navigation.py`、`scripts/probe_navigation_contract.py`、`scripts/probe_ul_mesh_navigation.py`。
- `scripts/train_ul_curriculum.py`、`scripts/report_ul_curriculum.py`、`scripts/export_ul_curriculum_actor.py`、`scripts/report_competition_readiness.py`。
- `tests/test_ul_task_env.py`、`tests/test_ul_navigation.py`、`tests/test_ul_curriculum_policy.py`、`tests/test_navigation_contract.py`、`tests/test_referee_protocol.py`、`tests/test_ul_observation.py`、`tests/test_mesh_navigation.py`。
- `docs/15_ul_observation_protocol.md`、本文件、`docs/navigation_execution_contract.md`、`docs/offline_mesh_navigation_reference.md`、`README.md`。
- 官方协议PDF/manifest/raw page与`data/rmu_2027_protocol_facts.json`、`data/rules_text/rmu_2027_protocol.txt`；导航缓存、执行探针、训练run与生成报告详见`data/artifact_inventory.json`。

当前可验证的是任务选择、信息可见性和BT执行边界。正式比赛3v3伤害/复活/对手角色、对手风格训练、UC完整地图语义、实裁判接线和真实Nav2/MPPI回放仍是后续必需工作。课程成功不能证明比赛胜率，也不足以扩大RL在实车上的权限。
