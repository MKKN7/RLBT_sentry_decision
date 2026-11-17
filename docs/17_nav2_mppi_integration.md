# 实际 Nav2 / MPPI 接入

日期：2026-11-16。前一阶段的UL课程用栅格路径运动代理，不能代替用户指定的Point-LIO＋Nav2＋MPPI。此前未接的实际原因是宿主没有ROS，现有容器虽有Point-LIO，但尚无Nav2/MPPI；桌面“导航仿真”的启动文件实际只启动SLAM。

后续进展：用户指定UC为RMUC STL后，已另行准备未经整平原图并在完整场地实跑1.2米路线，结果succeeded、误差0.1333米；新增实际点云20°分层体素模块。地图来源、默认入口与当前未知支撑限制见[UC原图执行说明](18_uc_rmucstl_execution.md)。下文的“当前仍未完成”描述本文件原平面验证阶段，UC后续状态以新说明为准。

本轮已经运行真实执行链：

```text
原 MuJoCo 车体 / 双MID360 / IMU
  → 原 Point-LIO 内核
  → lio_display 估计底盘 pose / TF
  → 估计位姿差分得到 body vx/vy/wz
  → 点云障碍层 / Nav2 costmap
  → ComputePathToPose / Navfn
  → TaskIntent / NavigationExecutor 校验
  → NavigateToPose / 简单导航 XML
  → FollowPath / Omni MPPI
  → cmd_vel / 原 MuJoCo wrench drive
  → feedback / action 终态返回任务 BT
```

本轮使用原机器人平面测试场景，有围墙和一个斜台；**没有把UL/UC地图训练悄悄切换到这个场景**。定位只来自真实Point-LIO估计，GT仅写入独立对照日志，不广播GT世界TF或替代导航odom。

## 运行环境与隔离

- 共享容器`Combat_Sentry2026`的ROS Humble继续作为基础；没有修改其dpkg状态、apt源、原工程或桌面入口。
- 项目私有overlay：`runtime/nav2/prefix`，46个明确版本包、约84MiB，Nav2 1.1.20；容器临时副本`/tmp/rl_bt_nav2_overlay/`。读取apt索引、下载并校验deb，仅解包，不执行系统安装/maintainer scripts。
- 依赖清单与验证：`data/reports/nav2_overlay_20261116.json`。重建：`bash scripts/prepare_nav2_overlay.sh`。
- ROS_DOMAIN_ID=79，namespace=`rl_bt_nav_probe`，与原工程61/65分开；启动器发现该域有其他节点会拒绝启动，只结束自己创建的进程组。
- 原ROS bridge只在私有内存模块中把domain guard 65改为79，配置namespace改为独立测试namespace；原文件不变。
- 原机器人几何、质量、悬架、驱动参数和20万点/秒/雷达的传感器配置保留。使用CPU射线、小场景，不占用GPU训练资源。

## 适配内容

原`lio_display`的odometry未填写twist。新增`PoseDifferenceTwistEstimator`使用同一采样时钟的估计底盘位姿差分，转换到body frame；缺失、乱序、超龄、跳变、帧改变和时间间隔异常均显式无效，不能把无效当零速度。fixture给节点开启`use_sim_time`，避免用墙钟与仿真样本相减。

Nav2 controller为实际加载的`nav2_mppi_controller::MPPIController`，Omni模型，20Hz、30×0.05s时域、500条采样，最大vx0.5/vy0.4/wz0.8。任务只要求XY到点，因此最终配置yaw容差为π，关闭GoalAngle/PathAngle critic；指定车头朝向的靠站/过洞任务需另配目标检查器，不能复用位置任务验收。Humble这版未发现可直接使用的加速度约束参数；原MuJoCo drive执行加速度限制1m/s²，不能声称MPPI内部也已约束。圆包络0.411m与课程配置一致。

Action适配先等待三个生命周期节点active，调用真实ComputePath，检查结果frame/有限值/目标末端，再由BT契约接纳意图并提交NavigateToPose。记录action UUID和task ID；取消ACK与取消终态分开；过期/乱序反馈不可刷新watchdog。定位/TF失效或导航超时会输出取消与独立`navigation_safety_stop`，fixture接收后锁住零指令，后续MPPI速度不能解除此锁。

当前导航XML为PipelineSequence，以1Hz重算路径并并行FollowPath，未叠加导航级自动恢复；重试/恢复预算仍归任务BT。此XML是导航测试树，**不是已经集成用户Combat XML的声明**。

## 实际验证结果

|测试|实际结果|记录|
|---|---|---|
|前向(1,0)m|Nav2 succeeded；末尾估计约(0.877,0.010)m，GT对照约(0.890,0.021)m|`runs/nav2_pointlio_forward_20261116/`|
|横向组合(1,1)m|Nav2 succeeded；估计目标误差0.1297m，符合配置0.15m容差；166条MPPI速度，93组有效估计速度|`runs/nav2_pointlio_lateral_20261116/verification.json`|
|行进中取消|取消接受后收到独立canceled终态；原始取消回放不把终态直接当作已测停车|`runs/nav2_pointlio_cancel_20261116/action/result.json`|
|取消后继续观测1秒|末尾估计平移速度约0.0068m/s，独立于canceled终态记录|`runs/nav2_pointlio_cancel_settle_20261116/verification.json`|
|最终配置取消回归|canceled后安全停车锁已触发；额外1秒后估计平移速度约0.0094m/s；保存所有接线源码hash/快照|`runs/nav2_pointlio_final_cancel_20261116/verification.json`|
|斜台方向目标(4,−3)m|一次规划版本60s执行超时，BT取消并触发安全锁；1Hz重规划版本23.7s返回failed、规划器报告无法到达目标。两条均保留为失败，不能计入到点成功率|`runs/nav2_pointlio_obstacle_20261116/`、`runs/nav2_pointlio_replan_obstacle_20261116/`|
|斜台后方(5,−3)m，固定yaw旧版|位置误差已到0.025m，因未满足固定车头朝向，进度检查失败。失败终态触发独立停车锁|`runs/nav2_pointlio_replan_clear_goal_20261116/`|
|斜台后方(5,−3)m，按XY任务完成|修正位置任务完成条件后succeeded；末尾估计XY误差0.1278m；566条MPPI命令、302组有效估计速度|`runs/nav2_pointlio_position_goal_obstacle_20261116/verification.json`|

本轮完整测试126 passed，涵盖原规则/训练模块及新增Nav2运输校验、MuJoCo执行fixture、估计速度。到点误差与运行次数只描述这些实际样本，不是导航可靠性统计或NUC性能验收。

可重复运行，输出目录不能已存在：

```bash
/home/kellen/anaconda3/envs/robot/bin/python scripts/run_nav2_closed_loop.py \
  --out runs/nav2_pointlio_NEW --goal 1 1
/home/kellen/anaconda3/envs/robot/bin/python scripts/run_nav2_closed_loop.py \
  --out runs/nav2_pointlio_cancel_NEW --goal 2 0 --cancel-after-s 1
```

启动器自动复制本项目必要脚本/配置到容器私有目录，不调用原desktop启动/stop脚本。每次保留模型、Point-LIO配置、节点日志、BT事件、实际命令、估计轨迹、GT对照与汇总。后续版本在终态后继续观察1秒，并单独记录最终估计速度；失败/取消终态另触发锁存停车，避免把终态等同已停车。

## 当前仍未完成的部分

- 当前terrain_obstacles是**平面测试专用的高度过滤**，不是完整HERO体素地图、20°坡面分类、分层支撑和洞顶净空；障碍来自registered_scan，测试平面未知区的空闲先验已显式记录。
- 正式UL/UC地图到Point-LIO odom的注册、场地语义与RFID实测尚未接通。平面场景以odom作全局坐标，不伪造官方map→odom。
- 当前接线用测试候选TaskIntent；之前UL课程训练的`ULTaskEnv`仍是快速运动代理。**不能把旧24/24课程结果标成真实Nav2训练结果**。下一步是正式地图与候选服务通过ComputePath/地形校验后，接入同一个导航适配层，并以较慢的真实执行回放校验训练代理。
- 另一份`rl_bt/nav2_mujoco.py` / `scripts/run_nav2_mujoco_harness.py`是domain86的UL栅格物理执行fixture，明确使用GT定位，必须与本轮domain79真实Point-LIO链路分开报告。

本轮改动入口：`rl_bt/nav2_transport.py`、`rl_bt/estimated_twist.py`、`rl_bt/nav2_mujoco.py`；`scripts/nav2_action_probe.py`、`scripts/nav2_mujoco_fixture.py`、`scripts/launch_nav2_probe.py`、`scripts/run_nav2_closed_loop.py`、`scripts/run_nav2_mujoco_harness.py`、`scripts/prepare_nav2_overlay.sh`、`scripts/audit_nav2_runtime.py`；`configs/nav2_probe.yaml`、`configs/nav2_probe_bt.xml`；对应3份新增测试、运行记录及`runtime/nav2/`。精确文件哈希清单见`data/artifact_inventory.json`。
