# 实际 Nav2＋MPPI 的示范、续训与冻结评估

2026-11-17。本轮继续第22篇的实际执行链路，目标是检查真实 BT 示范热启动后能否改善短任务决策。范围仍为原始 UC 场地的局部位置任务，未加入正式比赛、对手、射击、共享经济或 UL 实际执行。

## 固定实验设计

`configs/live_nav2_continue_20261117.json` 在采集前声明目标循环 `(0,.3)`、`(0,.4)` m、BT示范4局、actor-only BC 200次更新、PPO最多8局且总决策上限256次、初始BC／最终PPO各4局独立greedy评估。达到总决策上限会提前结束，实际完成局数以summary为准。未使用历史代理checkpoint。

PPO rollout8、epochs2，训练标签 CE／冻结BC KL anchor系数各.05。BC按有标签的navigate／continue／hold／recover组均衡抽样，但并不存在的类别不补造。critic不参加BC；PPO转移来自实时RPC。`untrained.pt`是BC前初始化，`initial.pt`是BC后权重，`final.pt`是固定预算结束权重。

独立评估载入指定权重，argmax选动作，不执行BC／PPO／SGD。评估数据不回流训练，也不用评估结果选择checkpoint。各策略目标顺序一致，但物理进程及估计器独立重启，不能声称严格配对随机种子测试。

## 本轮接口修正

- active期间屏蔽执行器不支持的无关目标替换；planning期间保留目标，使CONTINUE可用。healthy HOLD及取消保持合法。
- 截止恰在两次RPC之间发生时，返回真实终态及`action_applied=false`，避免把任务超时报成通信错误。真实成功标准不变。
- 示范加载校验完整BT训练记录、相同observation／session源码hash、原图及实际插件／命令／RPC证据，拒绝proxy、RL、评估或出错run。

上述observation／session修正后冻结源码再采集。保留首个失败示范run `runs/uc_live_bt_demonstrations_20261117`，其status=error，不用于BC。正式来源为 `runs/uc_live_bt_demonstrations_r2_20261117`。

## 示范中的有效选择与支撑瓶颈

正式BT4局仅1局完成，共128条转移：navigate1、continue5、hold122、recover0。122条HOLD全是mask仅有一个合法动作的强制等待，失败三局在起始估计位置即出现当前圆足迹支撑unknown，不能算策略主动放弃导航。

因此128条记录只有6条提供动作选择。单动作mask的CE／策略梯度／熵为零；BC训练标签准确率100%仅表示对这批稀少标签的拟合，不代表泛化能力。纯强制HOLD rollout中非零actor梯度可能只来自BC anchor；actor权重变化本身不能证明PPO学到了有效选点。

记录`action_applied=false`是为完整保留实际经过时间与终态奖励。本批示范的该类记录均为强制HOLD；未来若该竞态出现在多动作状态，需单独屏蔽未执行动作的actor loss，保留critic与时间转移，不能把选择当成已经执行。

支撑unknown仍阻挡；圆足迹半径.411m、固定工程起点先验、机器人模型、MPPI参数保持。不会通过填充unknown、扩大先验或缩小机器人来提高完成率。

逐格复核发现，失败示范起始估计位置约`(-.061409,.000578)`m，两格unknown位于x约`[-.60,-.45]`m、y约`[-.15,.15]`m。其距固定先验中心最近.45m，距当前估计圆心最近约.38859m，因此在先验之外、机器人圆足迹之内；门禁拒绝正确，非数组转置或浮点误差。原模型未接收MPPI命令时，GT诊断仍显示约.172m静止漂移，不能把全部偏移归于定位噪声。GT仅用于这项事后诊断。后续应核验初始odom／base坐标和静止接触状态，不通过筛掉失败启动来改变统计。

## 产物与复现

固定预算实际完成8局、248次交互、31次PPO更新、1/8到点，120条MPPI指令（119条非零）。其中240步仅HOLD合法；前30个更新batch没有on-policy actor选择信号，最后1个batch包含8条多动作选择记录。postBC到final的actor L2变化约.095598，含anchor贡献，不能单独作为PPO增益证据。

冻结独立greedy评估：BC初始模型4/4到点（21步、260条MPPI指令），PPO最终模型0/4（125步、45条MPPI指令）。两组均0次SGD且权重hash不变。BC评估4局均有可执行导航选项；PPO评估3局全程强制HOLD，1局运动后因足迹unknown取消。该小样本且未配对的启动条件差异不能隔离策略效果，结论为尚无PPO增益证据。

PPO训练的另一次运动失败，通用执行器取消原因记作`localization_invalid`，但该时刻详细足迹检查为`unknown_support`、odom／TF及支撑时间戳均新鲜。该通用原因来自组合health检查，不能解读为Point-LIO单独失效。

正式运行目录分别为：

- `runs/uc_live_bt_demonstrations_r2_20261117`：真实BT训练示范。
- `runs/uc_live_ppo_bc_continue_20261117`：BC及实时PPO。
- `runs/uc_live_bc_evaluation_20261117`：冻结`initial.pt`评估。
- `runs/uc_live_ppo_evaluation_20261117`：冻结`final.pt`评估。

逐次转移、实际时长、mask、执行状态、模型、源快照及实际插件／原图／命令证据分别保存在各run。详细对比由 `scripts/report_live_nav2_progress.py` 从这些记录重算，见 `data/reports/live_nav2_continue_20261117/report.md`。指标为局部导航任务完成率，不能解释为比赛胜率。

```bash
cd /home/kellen/桌面/RL_BT_decision
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  /home/kellen/anaconda3/envs/robot/bin/python scripts/train_live_nav2.py \
  --run runs/uc_live_ppo_bc_NEW --episodes 8 --max-decisions 256 \
  --goal-schedule configs/live_nav2_continue_20261117.json \
  --warmstart-live runs/uc_live_bt_demonstrations_r2_20261117 \
  --bc-updates 200 --anchor-ce .05 --anchor-kl .05 \
  --episode-wall-timeout 180 --task-timeout-s 20
```

所有真实会话必须串行使用私有domain79；输出目录必须不存在。host CPU两线程，无GPU训练。模型快照不能恢复物理与传感器队列，不支持exact physics resume。

本轮全量回归267项通过；各正式run均核对原图hash、实际MPPI插件、observed支撑、GT不进入策略和RPC来源。结束后容器中无本轮持有的ROS／fixture进程残留。
