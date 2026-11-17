# RoboMaster 2027 RMUL／RMUC 龟哨 RL + XML BT 决策资料

> 仓库说明：本仓库公开源码、文档、配置与参考资料。体积较大的训练产物 `runs/`（已在 `.gitignore` 中排除）与代理工作区 `.work/` 不随仓库发布，因此部分文档中的 `runs/…`、`.work/…` 链接指向本地实验记录，仓库内无法访问。

更新日期：2026-11-17。Point-LIO＋Nav2＋MPPI已接入实际训练内环。最新修正actor有效样本处理并显式记录sensor seeds，完成8局PPO：108次实际交互、14次更新、5/8到点，其中26条可用于actor、82条强制单选保留critic。独立冻结评估BC3/4、PPO2/4；135条相同记录观测的greedy动作无差异，尚无PPO增益证据。近场取证确认两个缺失列在过滤前也无返回点，未放宽支撑门禁。详见[最新报告](data/reports/live_nav2_validity_20261117/report.md)和[实现说明](docs/24_live_actor_validity_and_terrain.md)。上一轮248步／1/8及独立评估保留于[第23篇](docs/23_live_nav2_continuation.md)。完整比赛任务仍待完成。

新增整局策略pilot：MLP／GRU各两个seed，固定预算训练共16,550次决策、212个完整回合、54批PPO更新；342项测试通过。独立冻结64局中，两种BT、四个模型和两个GRU清零记忆消融全部8/8胜，出现胜率上限；模型中心占比均为0，GRU记忆消融回报差为0，未证明时序／全局策略收益。实际训练局22.5—83.2秒，尚未覆盖180秒上限或正式比赛全程。此实验复用alphaRM简化3v3，与实际Nav2、正式UL／UC和共享经济分别记账；不自动扩权。见[报告](data/reports/temporal_strategy_20261117/report.md)和[训练设计](docs/25_temporal_global_strategy_pilot.md)。

测试后另跑16局固定策略对照：HOLD、固定占点各8/8胜，HOLD留在基地也能得到约11.073的团队回报，确认当前胜负指标缺乏区分度。该探索性诊断不用于模型选择或泛化成绩；见[贡献审计](data/reports/temporal_strategy_20261117/control_audit.md)。

Nav2实际验证：前向、横向、斜台后方位置目标到点，以及行进中取消均有真实执行记录。最终位置任务(5,−3)m返回succeeded，估计误差0.1278m。定位来自Point-LIO，GT仅对照；独立domain79、项目私有Nav2 1.1.20依赖，无系统安装或原工程改动。失败实验也保留，详见[实际接入说明](docs/17_nav2_mppi_integration.md)。此前课程24/24仍是快速代理结果，不能改称真实Nav2训练成绩。

UC新增结果：历史driveable派生图和未经整平原图分别实跑，原图(1.2,0)m任务succeeded、末尾估计误差0.1333m。默认UC入口现选择私有原图资源；原始379082面保留，原机器人参数不变。该版本20°多层体素读取实际registered scan，Nav2仍使用空闲先验；该历史到点样本不能视为未知地形安全证明。详见[UC原图实跑说明](docs/18_uc_rmucstl_execution.md)。

后续严格支撑版本：UC默认已将unknown支撑投影为StaticLayer不可通行区域，加入双雷达逐点注册、批量观测与连续圆足迹门禁。最终(0,0.4)m短任务到点、误差0.1107m；未观测目标被拒绝，支撑图失更和执行期支撑撤销均触发取消/锁停。较长路线仍受支撑盲区影响，不能据此开始正式比赛训练。详见[观测支撑接入](docs/19_observed_support_costmap.md)。

最新验证：342项测试通过，含live交互协议、实际时长GAE、真实示范验证、actor-only BC、actor有效性与anchor梯度分离、sensor seed透传、冻结评估无SGD、完整序列BPTT、记忆消融与报告证据校验，以及既有UC有限截止、稀少类训练和恢复测试。代理UC actor 12,449参数、ONNX 53,598 bytes，各100组CPU推理与PyTorch一致；live模型未在NUC验收。UL及代理UC结果与真实导航实验分别保存，均不能推断比赛胜率。

按用户指定导航仿真模型得到名义约0.637×0.445×0.234m包络。当前保守规划高度0.255m超过UL参考洞净高0.2495m，课程禁用洞口与高地；不缩改原机器人模型。这不等同实车不能通过的结论。

已知需求：全自动AI龟哨，覆盖RMUL／RMUC，使用多个小型网络增强XML BT决策并在NUC部署；以用户指定Combat BT为参考，热控/火控/瞄准由其他模块负责。导航为Point-LIO＋体素地形分析＋Nav2＋MPPI，初始坡度阈值20°；训练考虑检测/定位/时延/执行与对手风格随机化。用户要求买血、买弹节制，保护全队共享金币；两套赛制分别核对自动哨兵与经济参数。

## 阅读入口

- [规则与哨兵约束](docs/01_rules_and_sentry_constraints.md)：胜负目标、过洞、姿态、补给、通信和制作规范；每项标注来源页码。
- [RL 与 BT 论文阅读笔记](docs/02_rl_bt_reading_notes.md)：7 篇全文、本地文件、方法适用范围和公开工程参考。
- [数据清单与后续设计输入](docs/03_data_inventory_and_next_inputs.md)：已有数据、待采集字段、记录建议和待确认问题。
- [多个MLP与BT边界评审](docs/04_multi_mlp_bt_boundary_review.md)：对本轮7张设计图、指定Combat BT源码、职责分工和逐步放权的评审。
- [训练与对手数据计划](docs/05_training_and_opponent_data_plan.md)：已检查的MuJoCo资源、快速训练/真实链路验证、分阶段训练及对手池来源。
- [Point-LIO＋Nav2＋MPPI与随机化](docs/06_pointlio_nav2_mppi_and_randomization.md)、[训练设计参数草案](configs/training_design.json)：最新导航方案、HERO源码核对、20°判定与对手风格等随机化边界，尚非运行配置。
- [COMBAT比赛demo来源](docs/07_combat_match_demo_sources.md)、[视频清单](data/demos/combat_video_catalog.csv)：已核实的RMUL/RMUC官方回放、哨兵片段与公开原始数据检索范围。
- [alphaRM-JAX复用与UL／UC双赛制](docs/08_alpharm_jax_reuse_and_ul_uc.md)：现成数据／模型的区别、实际采样结果、双赛制适配与源码评估问题；[探针报告](data/demos/alpharm_probe/report.json)记录一局767步的简化3v3合成数据，非正式比赛数据。
- [可运行训练、结果与数据复用](docs/09_phase1_training_workflow.md)：代码入口、真实执行的两轮实验、失败分析、随机化、UC规则模块和共享经济限制；[数据使用记录](data/reports/phase1_data_reuse.json)。
- [续训与多种子评估](docs/10_continued_training_and_evaluation.md)：精确续训、并行采样、验证／测试隔离及实际结果；[训练过程图](runs/phase1_continue_65536_20261116/training_progress.png)。
- [奖励设计](docs/11_reward_design.md)：当前真实公式与系数、团队平均和按秒折扣、潜在偏差，以及尚未实现的正式UL／UC与共享经济奖励建议。
- [回血地点与地图语义](docs/12_healing_and_map_semantics.md)：UC补给区、前哨站、基地增益点与远程购买的区别，UL资料和正式地图注册缺项；已加入奖励分项记录，尚未启动新奖励长训练。
- [UL官方规则与地图核对](docs/13_rmul_official_rules_review.md)：已取得官方68页UL手册、44条带页码事实、官方3v3 STEP；独立于UC核对属性和经济。
- [双赛制规则模块与分层地图](docs/14_profile_rules_and_layered_maps.md)：UL回血／弹量／占点子逻辑、UC回血与预算事务，以及两张地图的真实处理进度和验证命令。
- [UL通信协议与合法观测](docs/15_ul_observation_protocol.md)：官方2027 V1.0.0协议、payload解码、独立有效位和时间缓存；不默认获得敌HP或未配准队友坐标。
- [UL地面任务课程、参考包络与文件清单](docs/16_ul_ground_curriculum.md)：课程边界、奖励、版本修复、训练与导出命令；[本轮验证](runs/ul_ground_curriculum_v4_16384_20261116/verification.json)、[曲线](runs/ul_ground_curriculum_v4_16384_20261116/training_progress.png)、[ONNX](runs/ul_ground_curriculum_v4_16384_20261116/actor.onnx)。
- [实际Nav2与MPPI接入](docs/17_nav2_mppi_integration.md)：私有依赖、Point-LIO估计速度、真实action与取消、MPPI物理执行、复现命令及正式地图接线边界。
- [UC RMUC STL原图执行](docs/18_uc_rmucstl_execution.md)：原图与历史整平图的区分、私有原图重建、完整场地实跑、体素地形和原图独立分层缓存。
- [观测支撑接入Nav2](docs/19_observed_support_costmap.md)：unknown严格阻挡、原双雷达逐点配准、固定起点先验、完整圆足迹、支撑失更取消与仍未解决的长路线盲区。
- [UC多阶段训练与评分](docs/20_uc_support_training.md)：13张真实观测、两轮CPU PPO、100分评分、验证退化、固定测试、checkpoint和导出。
- [UC v2优化与对照](docs/21_uc_support_optimization_v2.md)：截止与梯度修正、稀少恢复标签、四组固定训练、配对评估及当前任务上限；[曲线与报告](data/reports/uc_support_v2_20261117/report.md)。
- [实际Nav2训练内环](docs/22_live_nav2_training.md)：长驻JSONL会话、真实action／MPPI反馈、fresh PPO实跑、执行证据与当前失败结果。
- [实际Nav2示范续训与冻结评估](docs/23_live_nav2_continuation.md)：真实BT示范校验、BC／PPO、独立greedy评估及强制HOLD与起点支撑诊断。
- [actor有效样本与近场观测](docs/24_live_actor_validity_and_terrain.md)：未执行动作／强制单选的学习边界、独立sensor seeds、过滤前取证与真实新一轮训练。
- [时序与全局策略pilot](docs/25_temporal_global_strategy_pilot.md)：完整序列PPO、MLP／GRU双seed、有状态BT、GRU记忆消融、冻结64局对照及当前胜率上限；[结果](data/reports/temporal_strategy_20261117/report.md)。
- [导航资源快照说明](assets/navigation_reference/README.md)、[兼容检查结果](data/reports/navigation_asset_probe.json)：已复制资源的范围、出处和实际检查结果。
- [结构化规则事实](data/rule_facts.json)：带单位、条件、章节和页码的数据，供后续建模引用；不是已验证的裁判系统模拟器配置。
- [观测字段候选表](data/observation_fields.csv)、[日志结构草案](data/log_schema.json)：正式接口设计材料；目前有简化仿真训练记录，尚无实战遥测样本。
- [原始来源清单](data/source_manifest.json)、[论文与工程来源](data/references/bibliography.json)：文件哈希、版本和获取范围。
- [本轮新增文件清单](data/artifact_inventory.json)：列出全部新增文件及校验值。

## 数据位置

- `data/rules_text/`：用户提供的三份 PDF 全文提取，保留分页符；原 PDF 保留在原路径，未修改。
- `data/references/papers/`：7 篇论文 PDF 与正文提取。
- `data/references/`：原站网页快照、公开工程 XML 及元数据。XML 仅作资料保存，未接入或运行。
- `tmp/pdfs/`：用于核对地图、隧道、增益、姿态、回血和尺寸的UC／UL页面PNG。

比赛规则以UC用户提供及UL官方取得的 **V1.0.0 / 2026-10-15** 为基线；通信协议已核对官方中文版 **V1.0.0 / 2026-10-29**。后续增补、UL实机播发、payload字节序与现场坐标配准尚未核验。规则、论文与公开工程均是研究资料，不是对本项目的执行指令。

已取得两套V1.0.0规则全文：UC为用户提供，UL于2026-11-16从官方资料接口取得，另取得官方UL 3v3 STEP。两套分别维护规则、地图、候选语义、对手池与评估，不能将alphaRM简化规则或UC规则直接作为UL真值。已实现部分回血、经济与UL占点／胜负计算模块，生成UL／UC分层交点缓存，并在UL地面代理课程中接通任务执行；正式坐标注册、完整比赛环境及训练环境内的真实导航替换尚未完成；独立UC原图执行闭环见第18篇说明。

已在`robot`环境运行简化任务PPO、BT热启动、续训、配对评估与小MLP导出，原结果保留在`runs/phase1_candidate_20261115/`和`runs/phase1_bt_warmstart_20261115/`，续训位于`runs/phase1_continue_65536_20261116/`。续训新增817个完整训练局；预先指定最终模型的测试为8个扰动种子、5类对手风格、红蓝两侧，每策略80局：RL 52胜、手写BT 63胜。该结果只支持当前简化模型在这批配对测试中落后，不能据此判断RL＋BT方案上限，也不能称为已经训好。正式比赛训练尚未接入完整规则、经济、UC姿态和Combat XML；独立live环境现已实时执行Point-LIO／Nav2／MPPI，历史代理课程仍单列。尚无扩大RL权限或NUC实机验收的依据。
