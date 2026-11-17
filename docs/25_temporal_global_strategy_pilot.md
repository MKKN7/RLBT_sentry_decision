# 时序与全局任务决策训练

日期：2026-11-17。

本轮针对整局任务选择，而不是将短距离到点成功率解释为全局策略能力。原有Point-LIO→观测支撑→Nav2 MPPI→MuJoCo闭环与第24篇所有结果保留。新增实验复用冻结alphaRM简化3v3完整回合；不包含正式UL／UC完整规则、共享金币、哨兵姿态和过洞，也未在该代理环境调用Nav2。

## 模型与职责

RL输入仍为现有估计state148、16个候选点各16维特征和BT合法mask；输出只选任务／候选，固定执行器共享。BT处理合法性、覆盖和恢复；速度、火控、瞄准、热控不是模型输出。各策略都使用相同候选、mask、路径与技能；当前新增BT是Python研究基线，不是已经接入的Combat XML。

MLP PPO与GRU PPO使用同样的候选集合编码和打分结构。GRU的actor／critic各保留64维hidden，跨决策传递、局间清零；actor和critic参数独立，避免value回归直接改变actor。MLP总参数55,490，GRU总参数113,474，容量不等，因此两者差异不能全归因于记忆。

基线包括原reactive BT及新增有状态BT：HP进入／退出阈值0.30／0.80、4秒任务承诺、路径与观测威胁评分。这样不会把BT人为限制成无记忆系统。每个GRU final checkpoint还用同一权重进行每步清空hidden的消融；消融反映对历史的依赖，同时存在hidden分布改变的影响。

## 固定实验

训练前登记于[实验配置](../configs/temporal_strategy_experiment_20261117.json)。MLP、GRU各训练seed271、809；每组至少4,096次交互，完成最后一局，记录超出预算部分。每次最多4个完整序列更新2个epoch，Adam LR0.0003、entropy0.01、hidden64、CPU最多2线程、禁用CUDA。没有BC、旧checkpoint初始化、测试数据回流或按测试挑选checkpoint。

奖励保持旧环境的upstream团队平均dense＋terminal回报，并使用旧任务切换代价0.002。原始分项记录在每局episode.reward_components中；没有新增“停留更久就是更好”的奖励。gamma按实际秒为0.997，180秒后的折扣约0.582；lambda按秒为0.97。每步实际duration参与GAE，不将长短任务当作等长。gamma延长回报视野，不保证获得全局最优策略。

gamma与lambda共同作用的GAE trace半衰期约20.7秒，180秒后的trace约0.00242；理论回报折扣与实用优势归因视野不同。已完成MLP训练的完整局实际仅22.7—68秒，尚未覆盖180秒，更未覆盖正式比赛全程。以上作为结果解释边界记录，固定实验中不临时改参数。

采样期沿时间顺序保存state、candidates、mask、action、old_logp、value、reward、duration和终止信息。每个PPO epoch从局初零hidden重新计算当前策略，整段BPTT，不打乱单个时间步。真实终局bootstrap为零，显式400决策安全截断保留下一状态value；预算达到时完成当前局。强制单选不计该步直接actor loss，但输入仍更新GRU记忆，后续actor loss可以通过这些历史计算反传；真实奖励与时长仍进入critic和之前选择的回报。actor／critic分别裁剪梯度。

一个已知边界是整批恰好只有一个有效actor决策时，中心化advantage会清除该样本奖励梯度；本轮固定源码保持不变，训练日志需核对是否触发。只读审核见`.work/temporal_training_review.md`。不能将该情况等同于正常多选择更新。

训练对手风格固定为pretrained、capture、defense、resupply；pursuit只在测试出现。检测漏检、观测／命令延迟、位置噪声与速度偏差沿用旧环境，均为工程假设，尚非比赛遥测标定值。

测试seed51001、51002 × capture／pursuit × 红蓝侧，共8个场景／策略。比较2种BT、4个final checkpoint及2个GRU清零消融，共64个完整回合。各策略共用外生seed、风格、阵营，不是彼此直接对战；策略改变轨迹后，观测和受伤自然可以不同。所有评估采用greedy、no_grad、SGD0，前后权重hash必须相同。截断单列，不计为平局。

测试输出胜／平、团队回报、换点、BT覆盖、伤害、采样存活／中心／基地占比与HP。采样诊断读取仿真真值只用于事后统计，不进入actor。弹量和共享经济在此不可变简化模型中不存在，输出null，不伪造支出成绩。两个训练seed和两个外生测试seed只支持工程pilot，不能证明收敛、正式比赛胜率或PPO普遍强于BT。

上游回合最长180秒，达到胜负条件可以提前结束；完整局不代表一定经历180秒，更不是正式赛季420秒比赛。其余队友使用冻结上游策略，团队胜利还受队友表现影响，因此同时记录待测机器人伤害占比与占点等诊断，避免将全部胜利归于选点模型。

## 数据复用和执行

继续使用本地冻结alphaRM上游commit `9f2489ce78e46f0e37a21e90d4b11a0a1a94c4d2`的env_step.onnx、actor.onnx、fixture，以及原观测、候选、reward和共享执行器。冻结manifest记录上游逐文件hash，checkpoint保存manifest hash；已列出的Python合同源码另存source快照。这不是完整运行环境的封装。没有下载新比赛数据，也没有把历史代理胜率、公开视频和实机导航记录当作本轮训练样本。

统一入口会串行完成四组训练、冻结评估和报告，已完整且配置一致的训练run可验证后跳过；未完成run拒绝覆盖，也不声称支持精确resume：

```bash
cd /home/kellen/桌面/RL_BT_decision
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  /home/kellen/anaconda3/envs/robot/bin/python scripts/run_temporal_experiment.py
```

默认正式run为`runs/temporal_{mlp,gru}_s{271,809}_4096_20261117`；小规模`runs/temporal_smoke_gru_20261117`仅验证实现，排除于正式对照。每组保存initial/latest/final.pt、config／status／summary、逐局与更新JSONL、压缩原始序列和源码快照。

首次MLP seed271进程在第26局记录后消失，原因未确认，没有final.pt；保存在`runs/temporal_mlp_s271_4096_20261117_interrupted_1881`。其中1,881条已采样、1,702条已完成更新，不计入正式四组预算和测试。随后以同一配置从初始seed重跑，采用独立进程串行入口；这是重新执行，不是精确resume。运行日志`.work/temporal_registered_experiment.log`及PID文件用于核验进程。

最终数据位于`runs/temporal_frozen_evaluation_20261117`，汇总位于`data/reports/temporal_strategy_20261117`。是否存在策略收益必须根据已完成的结果文件判断；训练状态文件不是验收成绩。

## 已完成结果

四组实际决策分别4,116／4,124／4,133／4,177，完整回合56／51／55／50，共16,550决策、212局、54批更新，11,558个有效actor决策、4,992个强制单选。所有参数有限、无截断、没有仅一个有效actor样本的更新。实际训练局范围22.5—83.2秒。342项全量测试通过；最终四个checkpoint评估前后hash不变。

固定64局中，两种BT、4个final模型与2个清空记忆消融全部8/8胜。因此这批场景的胜率已经饱和，不能区分策略质量。平均团队回报：reactive BT11.375、有状态BT11.060、MLP两seed10.656／11.012、GRU两seed11.073／10.656。learned模型采样中心占比均为0，而两种BT约0.510／0.485；这些诊断不足以证明RL学到有效全局任务选择。

两个GRU逐步清零hidden后的8局回报与携带hidden完全一致；第一场景71／73条相同记录观测中，greedy动作差异均为0。GRU seed271的initial／final在该场景greedy差异也为0。其他模型可有greedy差异，但KL仍很小，接近打分并列时的标签变化不等于实用收益。当前结论是训练器确实更新，但尚未验证记忆或长期策略收益；不是RL方案上限的结论。

详见[逐项数据与曲线](../data/reports/temporal_strategy_20261117/report.md)。测试后补充固定continue／center控制诊断，单独作为探索性审计，不能回填成新的独立测试，也不用于模型选择。它用于检查冻结队友和测试配置是否使选点策略缺乏可识别的胜负影响。

该诊断16局已完成：两种固定策略均8/8胜。continue_only中心占比0、基地占比1，平均回报11.073，与GRU seed271在每个已见场景的回报与动作计数一致；fixed_center中心占比约0.669、平均回报10.498。它们行为明显不同而胜率完全一致，进一步确认当前胜负指标缺乏区分度。没有单独干预队友／对手，不能精确作因果归因。见[控制审计](../data/reports/temporal_strategy_20261117/control_audit.md)。下一轮应先增加有策略影响且更长的场景、预注册新的验证／测试seed，再扩大预算；继续保留公平的BT基线和原安全边界。

## 后续接入边界

该实验用于判断时间记忆、训练信号与策略比较是否工作。正式UL／UC还需分别采用已核对的合法观测、地图／规则与经济模块，并接入已有实际Nav2执行接口。正式买血买弹仍应通过BT事务校验与团队预算门禁。任何代理实验分数都不自动扩大RL权限，不改变unknown支撑门禁、原机器人和原地图。
