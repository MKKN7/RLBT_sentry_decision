# BT 与导航的离线执行契约

日期：2026-11-16。实现：`rl_bt/navigation_contract.py`。该模块是可执行的离线状态机，尚未连接 ROS、Combat XML、Point-LIO、Nav2 或 MPPI，也不修改原导航工程。

同日后续进展：状态机保持ROS-free，已由`scripts/nav2_action_probe.py`接入实际Nav2 action并完成Point-LIO/MuJoCo点到点、绕行和取消测试。Combat任务XML、正式地图候选服务和训练环境尚未替换；实际接线见`docs/17_nav2_mppi_integration.md`。下文“未来适配器”描述保留了契约边界，现已实现其中测试场景的action/反馈/安全停车适配。

## 分工与输入

候选服务提供 `CandidateSnapshot`，携带赛制、地图标识与哈希、坐标系、版本、生成时间和有效期。每个导航候选另含 x/y/z/yaw、支撑层、任务语义、路线类型，以及明确的合法性／可达性；同一 XY 的洞底与洞顶不能共用一个未标层的目标。`continue` 与 `hold` 不携带目标位姿。

RL 仅给出原有 `TaskIntent` 的候选编号、序号、候选版本、发送时刻与 TTL。BT 状态机再次校验版本、地图与坐标系、候选有效期、合法性／可达性、存活、定位有效性和 TF 就绪状态。候选的合法性必须来自实际允许的观测和地图，不能使用仿真隐藏真值。

TTL 仅限制新意图能否被接纳；已被导航接受的任务可以运行超过 TTL。任务总时限、接收超时、反馈超时和取消超时分别配置，没有把图片中的 2 秒 TTL 当成每个导航任务的执行上限。所有计时采用同一个单调时间域，比赛倒计时和 ROS 消息时戳需由未来适配器明确转换；时间回退要求新建会话，不能静默沿用旧任务。

## 替换、取消与回退

状态为 `idle → awaiting_acceptance → active`。选择同一目标不重复发送；选新点后转 `canceling`，先取消旧 action。

**Nav2 接受取消请求不等于旧 action 已停止。** 收到 `cancel_accepted` 仍等待，只有旧 action 的终态 `canceled`／`succeeded`／`failed`／`rejected` 才释放槽位。然后 `dispatch_pending` 必须使用当前候选快照重新校验，过期、版本改变或已不可达的待发意图直接拒绝，交回 BT 决定回落。

每个 goal 有独立 session／task ID。旧 action 的迟到结果不能覆盖新 action；乱序／未来反馈被忽略。进度消息的原采样时间用于监控新鲜度，不能用延后接收时刻把旧反馈刷新成正常。

接受、反馈、总执行或取消超时后输出取消及 `require_safety_stop` 请求，保留 `uncertain` 中的旧任务，不因超时擅自认定已经停车。`require_safety_stop` 是对未来确定性安全适配器的请求，当前离线模块没有执行停车。定位失效或死亡同样清空待发目标并请求接管。BT／安全层仍负责恢复、停机确认与重新规划；状态机没有内置无限重试。

## 尚未实现的适配

- 把 `send_goal`／`cancel_goal` 接到固定版本的 Nav2 action client；确认 action UUID、终态、取消回执和采样时钟的映射。
- 将 task／支撑层／路线与正式地图注册、ComputePath 结果、地形与车体包络检查绑定。导航 action 只有一个 Pose 时，`route_type=underpass` 本身不能强制 Nav2 过洞，需要明确路由约束、途经点或分层代价地图。
- 把 Point-LIO 定位与 TF 健康、Nav2 反馈、新的不可达结果接到 BT 的 watchdog／stop，并由独立安全链路落实停车。
- 将任务结果转成 XML BT 的 SUCCESS／RUNNING／FAILURE，统一导航层和任务层的恢复预算；日志保留建议、发送、接受、取消、终态及实际执行的任务 ID。

## 已运行验证

```bash
/home/kellen/anaconda3/envs/robot/bin/python -m pytest -q tests/test_navigation_contract.py
/home/kellen/anaconda3/envs/robot/bin/python scripts/probe_navigation_contract.py \
  --out data/reports/navigation_contract_smoke_20261116.json
```

13 项测试覆盖：版本与跨赛制隔离、重复目标、TTL 与任务时限分离、取消接受／终态区分、替换后重验、迟到结果、取消／接受超时、定位失效／死亡、反馈时序与采样时间新鲜度。smoke 使用明确标记的合成地图和工程超时输入，保留事件及源码哈希，不能当作正式地图导航或实机验收。
