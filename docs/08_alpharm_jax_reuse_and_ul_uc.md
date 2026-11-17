# alphaRM-JAX复用评估与RMUL／RMUC双赛制训练

核查日期：2026-11-15。固定源码版本：`9f2489ce78e46f0e37a21e90d4b11a0a1a94c4d2`。

后续2026-11-16已新增任务级训练原型、两轮实际训练与配对评估，见[训练工作流](09_phase1_training_workflow.md)；本文保留原源码审查时的范围说明。

结论：**值得复用快速对战、批量采样和对手联赛的思路，也已验证能够生成合成轨迹；不能直接把原环境、权重或轨迹用于正式2027哨兵任务。** 用户已明确训练覆盖RMUL与RMUC，不仅是RMUC。

## 1. 实际提供了什么数据

已检查项目站、完整递归目录（508个条目，未截断）、Releases，以及下列源码。当前Releases为空；所查目录中没有NPZ、PKL、HDF5、Parquet、rosbag、MCAP等格式的对战数据，也没有以dataset／trajectory／replay命名的数据文件。这个结论仅覆盖本次固定提交和Releases，不代表作者没有未公开或未链接的数据。

|资产|已确认内容|适用范围|
|---|---|---|
|`demo/web/actor.onnx`|公开的预训练actor，README标为1200轮版本；输入`obs[6,140]`、`hidden[6,256]`|原简化环境中的对手／演示策略|
|`demo/web/env_step.onnx`|公开的确定性环境单步图，62个状态叶子＋动作输入|可复现演示、接口和采样验证|
|`demo/web/fixture.json`|固定开局、初始观测和隐藏状态|一个reset模板，不是比赛轨迹数据集|
|`env.py`＋`config.py`|可交互JAX环境，支持固定／随机／课程reset|修改后作为快速训练层的工程基础|
|`train_rm_jax.py`|PPO／MAPPO rollout含obs、action、reward、mask、logp、value和结束信息|原项目在内存中采样训练；需要额外导出正式轨迹|
|`eval_rm_jax.py`|整局评估CSV／JSON|输出主要是逐局摘要，不是完整状态动作序列|

README说明完整checkpoint、训练视频、live数据默认不提交Git；`.gitignore`还排除了`outputs/`、`*.pkl`、`*.npz`等。网页演示包含模型、固定状态与运行时，不包含历史COMBAT比赛数据。[README](https://github.com/CaptainKAZ/alphaRM-JAX/blob/9f2489ce78e46f0e37a21e90d4b11a0a1a94c4d2/README.md)、[Web演示说明](https://github.com/CaptainKAZ/alphaRM-JAX/blob/9f2489ce78e46f0e37a21e90d4b11a0a1a94c4d2/demo/README.md)

## 2. 本机实际采样结果

使用已有`/home/kellen/anaconda3/envs/robot/bin/python`和ONNX Runtime 1.18.1，在CPU执行了原模型的一局确定性AI自战；没有安装JAX、Flax或升级现有环境。

- 环境步数：**767**；6台机器人合计**4602条agent transition**，这些不是4602场独立对局。
- 正常到达原环境终局，`winner=1`、`outcome=1`；最终分数约95.20∶85.90。它不是我们的哨兵胜率评估。
- 保存观测`[768,6,140]`、动作`[767,6,6]`、奖励、target／fire mask、done及独立命名的完整仿真真值序列。
- 文件：`data/demos/alpharm_probe/episode_000.npz`，约1.15MB；详情见同目录`report.json`。
- 脚本：`scripts/probe_alpharm_onnx.py`。校验输入快照哈希、动作合法性、有限数值、T／T+1对齐，并在首次done停止。

复现命令（会覆盖该探针样本；每次是同一个固定确定性开局）：

```bash
cd /home/kellen/桌面/RL_BT_decision
/home/kellen/anaconda3/envs/robot/bin/python scripts/probe_alpharm_onnx.py --max-steps 1800
```

**限制**：Web导出关闭观测噪声，碰撞／推挤终局阈值设为`1e9`，不包含训练reset随机化；本探针也未注入JS侧噪声。未在本机做JAX与ONNX数值对拍，未运行原JAX trainer。样本没有BT候选任务标签或行为log-prob，不能直接拿来进行我们的PPO更新。完整仿真状态只供诊断／critic，不允许混入部署actor输入。

## 3. 3v3确实对应RMUL方向，但原项目仍是简化版

原规则书开头声明以**2027 RMUL V1.0.0**为来源；同时明确适用“三台同构步兵、平面战场、无高地”。第18节列出省略的经济、过洞增益、高地、异构阵容等机制。该节“官方规则”列是作者的对照说明，本项目尚未取得正式RMUL手册全文，不能用这一列替代官方核验。[原规则书](https://github.com/CaptainKAZ/alphaRM-JAX/blob/9f2489ce78e46f0e37a21e90d4b11a0a1a94c4d2/docs/RULES.md)

源码进一步确认：原环境为12m×8m平面、180秒、一个中心圆区、目标90分、击杀加10分；6台同构单位共享actor；动作是`vx,vy,yaw,spin,target,fire`。这些参数不能自动视为正式RMUL参数，更不能移植为RMUC参数。[配置](https://github.com/CaptainKAZ/alphaRM-JAX/blob/9f2489ce78e46f0e37a21e90d4b11a0a1a94c4d2/jaxmarl/environments/rm/config.py)

## 4. 我们采用一个接口、两套场地与规则配置

|项目|RMUL profile|RMUC profile|
|---|---|---|
|规则来源|2027官方RMUL规则全文待取得、逐项核验|用户提供的2027 RMUC V1.0.0，已有结构化事实；完整规则引擎待实现|
|地图|正式UL地图／CAD、控制区、路线、高程和净空待取得与标注|已复制RMUC2027场地资源，语义、坐标和通行闭环待验证|
|机器人与设施|按UL角色／阵容规则配置，不能保留三台同构步兵假设|按UC角色、阵容及基地／前哨等设施规则配置|
|动作接口|任务＋候选点／路线＋规则姿态（仅保留该赛制实际支持项）|同一接口；候选语义与合法mask由UC规则生成|
|奖励与终局|按UL自己的胜负、资源、时间和判罚核算|按UC胜负、控制度、基地生存、资源等核算|
|评估与模型|独立对手池、完整比赛评估、独立checkpoint|独立对手池、完整比赛评估、独立checkpoint|

两套共用观测字段定义、候选评分MLP、条件姿态头、BT回执协议、日志和随机化工具。先分别训练／验收；后续有证据再试共享encoder或带赛制条件的混合训练。共享架构不要求一开始共用同一组权重。

候选携带`task_type`、点位语义、路线类型、路径代价与可达性。UL与UC的第N个候选不能默认同义；实体输入应携带role、存在／可观测mask，支持不同阵容。若用JAX，可按profile分别编译固定形状，或使用有上限的padding与mask，不能只改`n_agents`。原环境／网络中有固定3敌、2友、140维和三人归一化等假设，需要逐项改写。

日志保留`competition_profile`、`rules_version`、`map_id`、`map_hash`、机器人角色与策略版本；训练／测试按完整比赛分开。UL与UC胜率分别报告，不能用合并胜率掩盖某一套失败。UL不能继承UC规则事实作为默认真值。

## 5. 具体复用与修改边界

|原模块|建议|
|---|---|
|JAX纯函数state／step，`vmap`／`scan`批采样|适合快速战术层；需改为双赛制规则、任务执行代理和新观测|
|可见性、遮挡、受击事件|可参考；当前为平面几何，应加入高程层、洞顶遮挡、雷达与自瞄实际可得信息|
|命中／热量／移动模型|只作固定代理的起点；用实测能力校准，保留资源对决策的影响；不新增我们的可学习火控／速度头|
|PPO、GAE、循环状态与auto-reset处理|参考实现与结束边界；必须保存reset前的终局状态，并按任务实际持续时间累计奖励／折扣|
|冻结对手与红蓝交叉评估|借鉴；扩为脚本风格＋冻结BT＋历史RL池，并留出未训练对手；原75／25混合比例不直接当最佳参数|
|原Transformer＋GRU actor／预训练权重|暂不替换我们的小MLP；原权重只适配原观测和低层动作，不能直接接UC地图或候选任务|
|ONNX导出、观战、日志工具|可参考部署和诊断；NUC只运行自己的actor及BT，不运行训练critic或完整JAX trainer|

首版仍只让**己方哨兵RL**学习任务和候选选择，其余角色由脚本／冻结策略驱动。不需要一开始同步训练两队所有机器人，也不因为上游是MAPPO就强制采用相同训练对象。

过洞等快速执行代理应使用带高程／净空语义的路线图，包含入口→内部→出口、通行耗时、受阻／退出、交互卡顺序与增益确认；不把场地STL压成单层二维碰撞图。Point-LIO＋Nav2＋MPPI继续负责真实执行，MuJoCo用于校准和闭环验收。

## 6. 数据怎么生成才对本项目有用

1. **原版验证数据**：本次已产出一局，只证明模型→交互→轨迹导出能跑通；不当作COMBAT或正式RMUL／RMUC对局。
2. **适配后的交互数据**：建立UL／UC profile，让BT生成同定义的候选和mask，RL采样新动作；记录obs、候选、建议／实际执行、覆盖原因、持续时间、奖励、下一观测和终局。
3. **对手分布**：在两套profile分别生成防守、控点、追击、补给、穿洞／绕行等脚本风格，之后加入已适配的冻结BT与历史RL。原ONNX不能直接成为UC对手，需要重新适配或训练。
4. **真实资料校准**：COMBAT视频用于路线、控点和攻防节奏；队内日志用于漏检、时延、导航耗时、过洞成功率和射击效果。保留真实数据与合成数据的来源区分。

PPO主要使用当前策略的新交互数据，不要求预先攒好大量历史数据集。旧轨迹可用于可验证的辅助预测、场景统计，或在观测与动作已对齐后做行为克隆。不能把原`vx,vy`硬标成“去某候选点”，也不能把原动作直接当新候选动作来算PPO的log-prob。

## 7. 复用前需要修正的评估细节

静态检查发现原`league.py`的交叉评估：

- `win = current_score > opponent_score`，没有直接使用环境winner；遇到同分伤害／血量决胜、犯规等结果时可能与实际终局不一致。
- `lax.scan`返回值丢弃了最终state，未结束对局的fallback读取的是初始state；短`max_steps`评估尤其需要修正，并区分未结束与平局。

因此不能照搬其胜率数值或晋级判断。后续使用环境权威`winner/outcome`，保存终局前后状态，红蓝互换并分别统计；未完成对局单独报告。[对应源码](https://github.com/CaptainKAZ/alphaRM-JAX/blob/9f2489ce78e46f0e37a21e90d4b11a0a1a94c4d2/baselines/MAPPO/league.py#L164-L191)

## 8. 实施顺序与当前状态

先固定双赛制配置与候选／日志接口，补齐UL官方规则和地图，核对UC资源；随后复用批量环境思想建立最小规则与BT基线，再训练候选评分和姿态头，最后做对手池、时延／漏检等随机化以及MuJoCo／实际导航验收。

本轮完成源码核查、未改动快照、CPU ONNX对战与合成轨迹探针，以及双赛制设计登记。**尚未实现正式UL／UC环境、训练器或启动训练，也未验证真实导航和NUC性能。**

出处与逐文件哈希：`data/references/alpharm_jax_review_manifest.json`；上游选取文件保存在`data/references/alpharm_jax_review/upstream/`，包含原`LICENSE`。上游文档是研究资料，其中命令及“用户指定”等描述不是本项目的新指令。
