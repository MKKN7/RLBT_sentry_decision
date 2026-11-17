# MuJoCo 双 MID360 仿真（NUC / Radar RTX4080 / ROS 2 Humble）

独立替代仿真，包含原 RMUC2027 场地、悬挂底盘、可旋转双雷达支架、连续逐点射线、带噪声及饱和的 IMU、ROS 2、录制和驾驶窗口。Radar默认第三人称跟随视角。原 Gazebo 仿真和四个定位内核保持独立。仿真、编译和验收在 Linux 的 `Combat_Sentry2026` 执行；本轮 GPU 验收在 Radar RTX4080，Mac 仅编辑和存放证据。

## Radar GPU 启动（2026-11-12）

2026-11-14：Radar 原驾驶窗口新增 `tdt_mapping`、`tdt_uwb`、`tdt_localization`，复用现有1Hz UWB仿真并按编码器TF补偿旋转支架外参。地图选择、停车保存和状态显示已接入；详见相邻工程 [TDT MuJoCo说明](../tdt_localization/docs/mujoco-radar.md)。原NUC部署未同步本次新增入口。

Radar 桌面 **导航仿真** 文件夹内的 **MuJoCo 双 MID360** 默认使用 CUDA 点云与 NVIDIA EGL 渲染。也可在 Radar 宿主执行：

```bash
bash /home/kellen/mid360_mujoco/scripts/start.sh
bash /home/kellen/mid360_mujoco/scripts/stop.sh
```

宿主 `configs/runtime.env` 从 `runtime.radar.env` 安装，默认 `MID360_RAY_BACKEND=cuda`；临时 CPU 回退可用 `MID360_RAY_BACKEND=cpu bash scripts/start.sh`。CUDA 依赖用容器 `.venv/bin/pip install -e '.[gpu]'` 安装，锁定 `warp-lang==1.19.0`。项目 `configs/nvidia-egl.json` 显式选择 NVIDIA EGL，避免退回 Mesa llvmpipe；不是修改系统全局 EGL。

CUDA 使用 Warp BVH 计算全场逐点射线，MuJoCo 物理仍在 CPU。每 100 ms 批量提交射线，但保留 101 个相邻 1 ms 姿态，逐点按原 5 µs 时刻插值，不把整帧当瞬时扫描。CPU/GPU 噪声模型相同，但批量随机数调用顺序不同，不保证两后端同种子的量测逐位一致。

CPU 物理复用已计算的状态，避免重复接触/位置求解；静态接触采用半径 4 m 的保守窗口，保留原始接触凸包，完整视觉网格和激光网格始终保留。独立常驻子进程预编译下一窗口，XML只解析一次并线性筛选，避免Python线程争用执行锁导致移动卡顿。若下一窗口来不及，物理在安全边界前暂缓，GUI继续处理停车、按键与绘制，不能跨进缺失地形。ROS 发布和落盘使用有界、保序队列；队列满时阻塞，不静默丢传感器帧。`/clock` 随 200 Hz 传感器状态及 UWB 接收更新，物理步长仍是 1 ms。

验收数据和已知边界见工作区 `测试数据/测试结果/mid360_mujoco/20261112_radar-gpu-realtime/报告.md`。这是单机器人 GUI/建图的实时率验收，不是整套物理迁移至 GPU，也不保证所有全场路线及任意并行负载的最坏帧耗时。

移动卡顿回归使用 `tools/check_motion_latency.py NEW --speed 1 --seconds 90 --leg-seconds 6`，同时检查墙钟实时率、10ms界面心跳的最大间隔和最慢单步。不要只凭平均RTF约1.00认定界面不卡。修复与对照证据见工作区 `测试数据/测试结果/mid360_mujoco/20261112_motion-stutter/报告.md`。

## NUC 启动

NUC 桌面新增 **MuJoCo 双 MID360** 和 **停止 MuJoCo**。也可在 NUC 宿主执行：

```bash
bash /home/sentry/mid360_mujoco/scripts/start.sh
bash /home/sentry/mid360_mujoco/scripts/stop.sh
```

启动会打开原容器和驾驶窗口；关闭窗口会结束其建图进程并保存录制。停止脚本只针对本程序 PID，不停止 Gazebo 或关闭整个共享容器。重复启动由进程锁拒绝，启动日志位于宿主 `logs/gui-*.log`。

- WASD 平移；Q/E 或左右方向键转底盘；Z/C 切换支架旋转；空格停车，失焦停车。
- 可选择 `pointlio`、`pointlio_deskew`、`duet_lio`、`fastlio_multi`，点击“启动建图 + RViz”。`pointlio_deskew` 的副雷达仅作去畸变显示，不参与定位；`duet_lio` 为双雷达联合估计。
- “开始录制”停止建图并在下一个 100 ms 扫描边界开始；结束录制也等待完整帧。录制保存在容器 `/opt/mid360_mujoco/recordings/<时间>/`，同时包含原生数据和 ROS bag，不覆盖旧记录。用 `docker cp` 导出后再分析。
- 复位会停止建图、保存进行中的录制，恢复车辆初始位置；仿真时钟单调连续。需要重新点击启动建图。
- ROS 使用隔离域 **65**，命名空间仍为 `red_standard_robot1`。不要与 Gazebo 同域运行。

完整场地、双雷达各 20 万点/秒及 1 ms 物理步长下，原 NUC 的 CPU 版本尚未达到实时：4 秒场景同时 ROS 发布及落盘耗时约 13 秒。传感器频率按仿真时间保持正确，不能把墙钟收包频率称为 200 Hz；GUI 状态栏显示实际实时率。

## Radar桌面入口整理（2026-11-14）

`/home/kellen/桌面/导航仿真`集中6个启动入口：MuJoCo、Gazebo四种LIO和录包。TDT模式仍在MuJoCo驾驶窗口内选择。桌面顶层原8个启停图标已移除；停止功能保留在应用菜单和原`stop.sh`，窗口可正常退出。Isaac Sim/Isaac Lab独立程序入口不变。

两个工程的`bash scripts/install-desktop.sh`均安装到此文件夹，避免重新散落桌面；运行前自动备份所管理的旧入口到各自`backups/desktop-时间-PID`。本轮仅部署Radar，NUC现有桌面未调整。

## 第三人称跟随相机与刷新（2026-11-14）

Radar驾驶窗口和CLI快照默认采用第三人称，正常距离2.5m、俯视25°、观察点在底盘原点上方0.18m。相机跟随底盘位置和航向，保持世界竖直方向；不跟随支架旋转，不把车体俯仰/侧倾直接放大到画面。车体、四轮和双雷达实体可见。相机遮挡查询排除底盘及全部子部件（含旋转支架、雷达与轮子），雷达转动不会拉近镜头。视线遇到墙或顶棚时自动拉近，避免相机落到遮挡物后面；狭窄位置实际距离可能小于2.5m。

在`configs/robot.yaml`的`view`调整`distance_m`、`elevation_deg`、`target_height_m`。`mode: first_person`可显式恢复旧视角；原`fpv`安装参数保留。相机只读物理状态并改变渲染，不参与定位或传感器测量。

旧GUI在仿真落后墙钟时连续使用`after(0)`，会饿死Tk的idle画面刷新，而输入和物理仍可继续。现在每轮固定让出1ms，使过载追赶期间也能刷新。没有通过减小点数、IMU频率或物理精度换取流畅；也不把窗口重绘等同于定位状态。

本轮按用户要求**不启动可视化窗口**。回归工具：`tools/check_ui_scheduler.py NEW`用无窗口Tcl调用真实GUI调度；`tools/check_chase_view.py NEW`用NVIDIA EGL离屏验证相机/遮挡/车体可见性；`tools/check_tdt_headless.py NEW --seconds 180`用真实GUI tick、Tcl idle探针、完整场地、离屏渲染和`rviz:=false`的TDT UWB联测。三者均须`unset DISPLAY`；后两者配置`MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`及NVIDIA EGL JSON。`tools/check_mount_camera.py NEW`额外用真实模型对支架整圈721角度、两种底盘航向和近墙场景共2163姿态检查相机距离与几何不变；该工具只跳过绘图，不开窗口。详细数据与限制见工作区`测试数据/测试结果/mid360_mujoco/20261114_tdt-stutter-third-person/报告.md`。此测试不替代远程桌面最终显示链路验收。

## 旧第一人称相机近墙显示

驾驶视点位于底盘碰撞盒内部 `[0.19, 0, 0.13] m`，随底盘真实俯仰/侧倾运动。`fpv.near_clip_m=0.002`按米指定，编译后换算为MuJoCo的`znear/extent`比例，避免完整场地将默认近裁剪放大到约49cm而切掉近墙/坡面。FPV渲染仅隐藏包住虚拟视点的底盘外壳和方向标；物理碰撞、LiDAR自遮挡、RViz车体均保留。它是虚拟驾驶视点，不是新增实物相机或传感器外参。

相机修复验证：`tools/check_fpv_clipping.py NEW`对照近墙渲染深度，`tools/check_fpv_motion.py NEW`测试撞墙、上下坡及视点间隙；输出必须新目录。证据见工作区`测试数据/测试结果/mid360_mujoco/20261113_fpv-clipping/`。

## 传感器契约

| 项目 | 实现 |
|---|---|
| IMU 频率与单位 | 每台 200 Hz；加速度 m/s²，角速度 rad/s，机体传感器坐标系；无方向测量 |
| IMU 真值 | MuJoCo 物理 site 的 accelerometer / gyro，包含重力比力与旋转支架/杆臂效应 |
| 加速度饱和 | 逐轴、带正负号的 ±4 g = ±39.2266 m/s²；不是向量模限幅 |
| 角速度饱和 | 逐轴 ±2000°/s = ±34.906585 rad/s |
| 噪声 | 白噪声密度、初始零偏、随机游走、40 Hz 一阶低通；噪声和偏置后再次硬限幅 |
| 白噪声密度参考 | 100 µg/√Hz、4.5 mdps/√Hz，来自 ICM40609-D 芯片参考，不是 MID360 整机标定 |
| 点云 | 每台每仿真秒 200,000 次发射，间隔 5 µs；10 Hz 合帧，回波数取决于遮挡和距离 |
| 采样姿态 | 每点在两个物理状态间位置插值、旋转 SLERP；移动物体也插值；不把整帧绑定到单个位姿 |
| 求交 | CPU：MuJoCo 动态几何 + Embree 静态场地；CUDA：动态解析几何 + Warp 原始静态三角面 BVH，合并最近回波 |
| 量测误差 | 默认距离 σ=1 cm、角度 σ=0.03°；40 m@10% 与 70 m@80% 端点之间采用明确的近似反射率模型 |
| 时间 | 整数仿真纳秒；header 为帧首，`offset_time` 为逐点纳秒，`time` 为逐点秒；无回波点不挤压时间 |

默认配置在 `configs/simulation.yaml`，物理车体/安装参数在 `configs/robot.yaml`。种子固定，双 IMU/双雷达分别使用独立随机子流。`noise_enabled: false` 可关闭随机噪声，但带宽滤波与量程仍生效。

ROS 点云类型 `sensor_msgs/PointCloud2`，28 字节：`x/y/z/intensity` float32、`offset_time` uint32、`time` float32、`line/tag` uint8、`ring` uint16。`line=ring=0`、`tag=0x10` 为兼容字段，不宣称恢复真实激光通道或置信度编码。

话题保留 `livox/lidar`、`livox_right/lidar`、`livox/imu`、`livox_right/imu`。直接发布 `lidar1/points`、`lidar2/points`，不能再经过旧瞬时扫描 `lidar_format`。IMU 协方差描述当前白噪声模型，不代表偏置或饱和期间的总误差。`orientation_covariance[0]=-1`。

`chassis_odometry_gt` 仅供评估；不广播世界真值 TF、不向定位内核注入真值。TF 仅发布车体至传感器的安装/支架状态。控制订阅为 `cmd_vel`、`lidar_yaw/cmd_vel`，两类指令分别检查超时。

## 数据与回放

每个录制目录包含 `simulation.yaml`、`robot.yaml`、`model.xml`、`imu.csv`、`state.csv`、`clouds/*.npz`、`session.json`；启用 ROS 时另有 `rosbag/`。NPZ 中 `points` 为传感器量测，`truth_world` 仅用于测试；ROS bag 不含这个逐点真值数组。元数据记载发射/回波数、饱和数、耗时与是否完整结束，非整数帧结束时明确列出未刷出的扫描发射数。

四模式相同输入验收（仅回放双格式云、左 IMU、安装 TF 和原 `/clock`）：

```bash
# 以下在容器内执行
cd /opt/mid360_mujoco
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=65 ROS_LOCALHOST_ONLY=1
.venv/bin/python tools/replay_check.py recordings/NEW/rosbag pointlio evidence/NEW-check
```

`tools/prepare_lio.py` 为新工程复制原后端包，只在运输适配层读取逐点 `offset_time`，定位 vendor 源码保持 73 文件 SHA 一致；`prepare_launch.py` 生成独立配置，绕过旧格式转换。原部署文件不会被覆盖。

## 构建与复现

已部署：NUC 宿主 `/home/sentry/mid360_mujoco`；容器 `/opt/mid360_mujoco`；Linux x86_64、Python 3.10、MuJoCo 3.17.0、ROS Humble。配置/资产按源码目录读取，使用 editable 安装。

```bash
# 需要容器原有 ROS Humble、g++、python3-venv、python3-tk
cd /opt/mid360_mujoco
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e '.[dev,field]'
.venv/bin/python tools/prepare_field.py \
 /opt/pb2025_nav_ws/src/pb2025_light_sim/worlds/rmuc_2027_world.sdf generated/NEW-field
ln -s NEW-field generated/field-current  # 首次创建；不要覆盖历史转换
.venv/bin/python tools/prepare_lio.py \
 /opt/pb2025_nav_ws/src/pb2025_lio_backends lio_ws
.venv/bin/python tools/prepare_launch.py \
 /opt/pb2025_nav_ws/src/pb2025_light_sim /opt/mid360_mujoco
source /opt/ros/humble/setup.bash
source /opt/pb2025_nav_ws/install/setup.bash
cd lio_ws
colcon build --packages-select pb2025_lio_backends --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
```

容器 DNS 不可用时，使用证据目录中的 Linux CP310 wheels 离线安装；不要直接使用系统旧 numpy/scipy。C++ 射线层首次启动自动对当前 MuJoCo wheel 编译，输出在 `build/`。

验证命令均在 NUC 容器执行，输出路径必须尚不存在：

```bash
cd /opt/mid360_mujoco
.venv/bin/python -m pytest -q tests
.venv/bin/python tools/acceptance.py evidence/NEW-sensors --field generated/field-current
.venv/bin/python tools/check_tunnels.py generated/field-current evidence/NEW-tunnels
export MUJOCO_GL=egl ROS_DOMAIN_ID=65 ROS_LOCALHOST_ONLY=1
source /opt/ros/humble/setup.bash
.venv/bin/python -m mid360_mujoco.cli --field generated/field-current \
 --duration 4 --motion combined --ros --record evidence/NEW-field
.venv/bin/python tools/check_bag.py evidence/NEW-field/rosbag evidence/NEW-bag.json
```

## 已知近似与边界

- 扫描轨迹来自 MuJoCo-LiDAR 的 800,000 行表，每 4 秒循环，是有限时长扫描模式近似；不是无限非重复或厂商光学机构精确模型。表的俯仰边缘约 −7.21°～52.16°，接近标称 −7°～52°。来源与许可见 `THIRD_PARTY.md`。
- MID360 内部滤波/温漂、每台实物的 IMU 零偏/噪声、光学中心至 IMU 外参尚未标定；默认 IMU 与光学框架共点。提供可配置位置，不能据此宣称与实物逐样本一致。
- 反射率为当前统一参数，强度为简化值；未模拟多回波、雨雾、太阳干扰、玻璃、多径、真实通道编码或设备网络丢包。
- 原始 385,026 个场地三角面用于视觉和射线，接触面单独使用 1 mm 壳与局部凸包。平面重三角化保留凹轮廓/孔洞，平面容差 0.2 mm、边界量化/简化 0.01 mm。小部件局部凸化可能改变细小凹槽；四条指定隧道物理通行通过不等于全场接触严格等价。
- 车轮为圆滑接触近似、底盘由接触门控的有限力/力矩驱动；并非真实麦轮滚子/电机/地面摩擦标定。物理步长间做连续插值，不声称 5 µs 物理积分。
- 当前完整场地性能不足实时。所有频率与速度按仿真时钟解释；实车精度、长路线定位和高速复杂地形仍需另外验收。

详细证据入口：工作区 `测试数据/测试结果/mid360_mujoco/20261105_implementation/报告.md`。

## 灰色场地、车体和双雷达显示（2026-11-06）

完整赛场使用纯灰无纹理材质，RViz累计地图统一灰色；副雷达去畸变对照图仍保留必要的分组色。此前MuJoCo已经有车体碰撞体积，RViz缺失是桥接没有发布`robot_description`。现在从同一个编译后的MuJoCo模型生成车体、四轮、支架和左右MID360的URDF，以transient-local发布，并随bag保存；完整父子TF随悬挂/支架更新。底盘实体尺寸0.48×0.36×0.10m，左右光心初始坐标为(0.16, ±0.18, 0.218)m，外倾30°；旋转后的坐标由TF给出。雷达保留橙/蓝辨识色，各自坐标轴和文字标签显示实际安装位置。URDF支架胶囊以圆柱近似显示，物理胶囊未改。

`base_footprint`沿用原命名，但在本模型中实际是底盘参考坐标；没有把世界真值作为地图TF广播。打开“启动建图 + RViz”即可显示；只有传感器运行而未启动LIO时，RViz固定帧应选`base_footprint`查看局部车体。

## 2027 UWB位置输入

默认发布`/uwb_pose`，1Hz、二维天线位置、接收时间戳，含白噪声、慢偏置、延迟和丢包，支持正常/恶劣链路及可选NLOS压力区。z/yaw不作为观测输出，不生成伪造的测速模块角度。原生录制新增`uwb.csv`，ROS bag包含位置、协方差、仿真诊断；当前四种LIO未融合UWB。

完整官方依据、东北大学接口对应、参数假设及CAD坐标注册限制见[UWB设计](docs/uwb-2027.md)。**当前CAD到官方场地坐标的identity变换仍待角点/方向核对；噪声未做实物标定。** 旧录制配置缺少`uwb`时加载为关闭，避免改变历史传感器行为。

NUC增量验证（source Humble，域65，使用新输出目录）：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python tools/check_uwb_statistics.py evidence/NEW-uwb-statistics
DISPLAY=:0 MUJOCO_GL=egl .venv/bin/python tools/check_display_uwb.py evidence/NEW-display
.venv/bin/python tools/replay_check.py evidence/NEW-display/recording/rosbag pointlio evidence/NEW-replay --rviz
```

本轮13项测试、正常/恶劣各20000次位置模型统计、完整场地12秒GUI/Point-LIO/RViz/录制联测通过；统计是合成模型验证，不是实测无线性能。真实截图检查后修正MarkerArray的Humble主题属性，并限制RViz远程桌面字体尺寸。详见工作区`测试数据/测试结果/mid360_mujoco/20261106_display-uwb/报告.md`。

## 4080 / Radar 部署（2026-11-11）

宿主 `/home/kellen/mid360_mujoco`，容器仍为 `Combat_Sentry2026`，内部 `/opt/mid360_mujoco`。用户要求只部署，不启动仿真；部署检查仅安装依赖、编译和导入模块，没有运行仿真或RViz。独立虚拟环境使用Python3.10/Linux x86_64，LIO构建在 `lio_ws`，不覆盖原Gazebo后端。桌面新增同名MuJoCo启动/停止入口。

启动脚本从自身目录定位宿主工程，继承现有容器的DISPLAY和XAUTHORITY：Radar当前`:1`与`/tmp/pb2025.xauthority`，NUC通常`:0`。实际启动时仍需图形会话有效；本次没有做4080图形/性能验收。录制保存在容器`/opt/mid360_mujoco/recordings`，未对既有容器增加挂载；导出可用`docker cp`，重建容器前需另存录制。

部署日志/安装清单/SHA与未启动进程检查见工作区`测试数据/测试结果/mid360_mujoco/20261111_radar-deploy/`。11月11日NUC回放修复已验证，但之后直连断开，NUC最终容器关闭状态无法确认；不会将Radar部署完成写成NUC已清理。
