# UC RMUC STL 原图导航执行验证

2026-11-16。按用户明确指定，UC 使用桌面导航仿真的 RMUC STL。现已在完整场地运行原 MuJoCo 车体、双 MID360、IMU、Point-LIO、Navfn、Omni MPPI 和任务 BT action 回执；未将旧 UL 课程结果改称 UC 或真实导航训练成绩。

后续版本已将UC默认切为严格观测支撑，接入原双雷达点云逐点配准与StaticLayer。下文的1.2米成功是旧clear先验执行测试，当前严格模式会拒绝该方向的支撑缺口；最新复现与结果见[观测支撑说明](19_observed_support_costmap.md)，不能沿用旧命令预期本段相同结果。

## 地图版本与原参数

原工程当前 `field-current` 指向历史整平的 `rmuc_2027_driveable.stl` 派生资源，不能把它与原始 `rmuc_2027.stl` 等同。先用派生资源接线，随后从原始 STL 在容器私有目录重新准备场地。

|资源|三角面|SHA256|
|---|---:|---|
|原始 `rmuc_2027.stl`|379082|`2835c0e755c255f757d810d260283c623fbd4a49a8fd986487c70ad2a89e9d10`|
|原图 prepared `field.stl`|379082|`93ff0118af176f388c171b239acfdcac72c17cbe0b523350d5d916d3236fe10b`|
|历史 driveable prepared 图|385026|`3aea8445574b08fa2bc3ae571deb50a2d39befd07f3a48e19f694906526c4685`|

原图 prepared 资源：容器 `/tmp/rl_bt_nav_probe/fields/rmuc2027_original`。原 STL 仅按已有 world pose 平移一次，逐顶点最大误差约 0.73 微米；不能重复平移。原工具生成20649个碰撞片，保留全图三角形用于射线，沿用1毫米接触壳和局部凸包近似。这不等于实车净空认证。没有修改原 `field-current`、原工程、机器人质量/几何/悬架/驱动、共享系统包或桌面入口。

重建/核验：

```bash
/home/kellen/anaconda3/envs/robot/bin/python scripts/prepare_uc_original_field.py
```

私有资源在 `/tmp`，容器重建或清理后需用脚本重新准备。核验报告为 `data/reports/uc_original_field_20261116.json`。项目保存了原图 prepared STL 至 `assets/navigation_reference/uc_original/field.stl`；旧图资源和缓存均保留，未覆盖。

新缓存 `data/derived/uc_original_surface_layers_010m/surfaces.npz`：0.1米分辨率，280×161列，135051个交点，单列最多30层。旧派生图是126751个交点，二者不能混用。交点缓存用于离线几何查询，不直接向 actor 提供全知观测，也不是已认证可通行地图。

## 实际运行

|运行记录|地图|任务及结果|
|---|---|---|
|`runs/uc_rmuc_driveable_sensor_20261116/`|历史派生图|12.18秒墙钟推进1.904秒仿真；有效估计速度17组，实际registered scan驱动体素分类|
|`runs/uc_rmuc_driveable_nav2_forward_20261116/`|历史派生图|local目标(1.2,0)m，Nav2 succeeded，末尾估计误差0.14823m|
|`runs/uc_rmuc_original_nav2_forward_20261116/`|未经整平原图|local目标(1.2,0)m，Nav2 succeeded，末尾估计误差0.13330m，335条MPPI命令、75组估计速度|
|`runs/uc_rmuc_original_nav2_worker_forward_20261116/`|未经整平原图，异步分类|同一目标succeeded、误差0.13344m；10组地形结果，worker零丢样、无积压、正常关闭|
|`runs/uc_rmuc_original_nav2_worker_cancel_20261116/`|未经整平原图，异步分类|行进中取消，收到独立canceled终态和安全停车锁；后续估计平移速度约0.0312m/s，不把取消终态等同绝对静止|

每次运行包含场地来源、源码快照/hash、Point-LIO/导航日志、BT事件、实际命令、估计轨迹及独立GT对照。`scripts/report_uc_navigation.py`验证保存的源码快照hash，生成轨迹图和汇总 `data/reports/uc_nav2_execution_20261116.json`。到点依据配置的0.15米XY容差；该任务不约束车头朝向，不可直接当作过洞或精确停靠验收。终态后的等待使用墙钟，汇总另算实际推进的仿真时间，不混用两种秒。

复测（输出目录必须不存在）：

```bash
/home/kellen/anaconda3/envs/robot/bin/python scripts/run_nav2_closed_loop.py \
  --scene rmuc --goal 1.2 0 --wall-timeout-s 180 \
  --out runs/uc_rmuc_original_nav2_NEW
```

`--scene rmuc`现默认选择私有原图；可用`--field-dir`明确指定另一已审资源，fixture与action使用同一目录和prepared hash。原测试spawn世界(3.4,9.5,0.31)、yaw0，仅用于本次工程验证；Point-LIO建立session odom，local目标不是已配准的官方裁判坐标。GT只写入对照记录，未进入定位TF、体素分类或策略观测。

## 体素地形与尚未验证的范围

`rl_bt/voxel_terrain.py`只接收估计坐标系registered scan。体素保留多Z层，对局部支撑面拟合，20°坡面阈值包含边界，检查连续台阶和0.255米参考高度净空；过期/乱序/未来/不同frame观测拒绝，稀疏或不连通支撑保持unknown。台阶阈值、噪声容差和体素大小是显式工程初值，尚非实机标定。

最终入口使用`rl_bt/terrain_worker.py`独立spawn进程，至多每1秒仿真时间提交一帧，输入/输出各保留最新payload与容量1的通知队列；主循环不运行地形snapshot，忙时覆盖待处理旧样本。消费结果时另检查当前仿真时钟，超过2.5秒不发布；任务BT也要求地形样本新鲜。错误显式报告，不生成伪自由地图；退出只关闭自建worker。原图异步到点样本实际推进10.131秒仿真、耗时13.129秒墙钟，10次分类耗时约0.022～0.065秒；只描述本次工作站CPU样本，不是NUC性能验收。异步1Hz与旧同步10Hz保留样本不同，不能只按耗时比声称算法性能提升。

最终完整测试**146 passed**，包括地形分层/20°边界/台阶/低洞顶、Z原点平移后的分类一致性、worker限频/最新样本覆盖/退出/错误、地图来源门禁与已有规则/训练测试。高度估计允许毫米量化差，不将其误称逐浮点值不变。

本次Nav2仍只接收分类后的障碍点，rolling costmap沿用执行测试的空闲先验；体素输出unknown尚未作为costmap不可通行区域。这些成功样本不能证明悬空区、缺失支撑或全场未知区安全，不能直接生成正式比赛的可达mask。当前脚下自滤孔、稀疏点云支撑连通和低频分类时效还需处理。

后续任务训练应使用原图独立几何数据与候选配置，补齐观测支撑约束和候选服务，再把真实Nav2执行时延/失败分布校准到快速训练代理。当前没有启动新的正式UC PPO训练，没有用这两次到点提高RL权限，也没有改变用户要求的全队共享经济节制策略。
