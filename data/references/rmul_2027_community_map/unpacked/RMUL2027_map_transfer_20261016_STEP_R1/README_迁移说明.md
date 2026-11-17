> **STEP R1：已按官方 STEP 更新实际几何，保留初版迁移包。关键修正：围挡420mm、工作台顶面403mm、中央控制区顶面10mm、隧道实际净空249.5mm。详见 docs/RMUL_2027_STEP_AUDIT.md。包内 source_cad/ 提供原始官方场地 STEP；它不是机器人 CAD。**

# RMUL 2027 新地图迁移包

版本：2026-10-16。适用于 Ubuntu 24.04 / WSL2 Ubuntu 24.04、ROS 2 Jazzy、Gazebo Harmonic。

这是**地图增量包**，含新场地完整模型、碰撞网格、纹理、SDF 世界、PGM/YAML、参数证据、
生成与检查工具、安装脚本、迁移提示词和实际仿真截图。
不含机器人 CAD/传感器模型、完整 ROS 包、上位机、训练数据、算法权重或 ROS/Gazebo 软件本身。
不需要原 PDF 才能运行已生成的场地；原版权及手册局限见 `docs/RMUL_2027.md`。

`src/guodong_sentry_sim` 是增量目录，不是可独立 colcon 构建的完整包。
没有原机器人工程时，可以先独立查看地图；要连同哨兵运行，需先恢复完整仿真仓库。

## 1. 复制并校验

把整个 ZIP 复制到新电脑。在 WSL 的 Linux 主目录下解压，例如 `~/projects/`。
请解压为一个新目录，不要直接覆盖已有仓库。路径可以与旧电脑完全不同。

进入压缩包解出的 `RMUL2027_map_transfer_20261016_STEP_R1` 目录：

```bash
python3 verify_package.py
```

该命令只需 Python 3 标准库，会检查每个文件大小与 SHA-256。
压缩包外另提供 `.zip.sha256`；在 ZIP 所在目录可以执行 `sha256sum -c RMUL2027_map_transfer_20261016_STEP_R1.zip.sha256`。

## 2. 仅查看新场地（没有原机器人也可）

前提：已安装 Gazebo Harmonic，ROS 版 Gazebo 可先 `source /opt/ros/jazzy/setup.bash`。

```bash
bash scripts/view_field.sh
```

此模式打开场地，不包含机器人、ROS 桥接、SLAM 或 Nav2。
无窗口模式可用 `bash scripts/view_field.sh -s --headless-rendering`，退出按 Ctrl+C。
模型和纹理全部来自本地，不需要 Fuel 下载场地资源。

## 3. 安装到已有的完整哨兵仿真仓库

目标仓库需要含 `src/guodong_sentry_sim/package.xml`、原 CAD/preview 机器人、
`depot`/`sentry_lab` 旧世界、可选择 world/visual_quality 的 launch、正常的模型目录安装规则。
适配原项目 `f1dc095` 的 launch 布局，以及已安装本次地图的版本；其他布局会先报错，交给迁移助手合并。
脚本不会改写未知格式的 launch，也不会复制参考 launch 去强行覆盖目标电脑的新修改。

假设完整仓库位于 `~/projects/guodong-sentry-sim`，从**迁移包目录**运行：

```bash
python3 install_into_sim.py --repo "$HOME/projects/guodong-sentry-sim" --check
python3 install_into_sim.py --repo "$HOME/projects/guodong-sentry-sim"
bash scripts/build_integrated.sh "$HOME/projects/guodong-sentry-sim"
bash "$HOME/projects/guodong-sentry-sim/scripts/rmul_2027.sh"
```

第一条命令为只读兼容性检查。安装会新增场地，合并已知格式的 world 选项并把默认世界设为
`rmul_2027`；**旧地图不移动、不删除，安装前后核对哈希**。已有文件的备份与安装记录位于
目标仓库的 `output/validation/rmul_transfer_backups/<时间>/`。
若目标仓库有迁移清单 `manifest.json`，脚本同步更新本次涉及条目的大小和哈希。

构建默认使用目标仓库的 `output/validation/`。
使用自定义构建位置时，构建、启动和检查须设置同一个 `SENTRY_WS`：

```bash
SENTRY_WS="$HOME/rmul_ws" bash scripts/build_integrated.sh "$HOME/projects/guodong-sentry-sim"
SENTRY_WS="$HOME/rmul_ws" bash "$HOME/projects/guodong-sentry-sim/scripts/rmul_2027.sh"
```

构建脚本不安装软件或执行 sudo。缺少 ROS、colcon、Gazebo、ros_gz、RViz、rclpy 等时，
请先按完整仿真仓库的新电脑部署说明安装；Python 工具依赖见 `DEPENDENCIES.md`。

## 4. 回切、外观和 WSL 图形选项

```bash
bash "$HOME/projects/guodong-sentry-sim/scripts/rmul_2027.sh" world:=depot
bash "$HOME/projects/guodong-sentry-sim/scripts/rmul_2027.sh" world:=sentry_lab
bash "$HOME/projects/guodong-sentry-sim/scripts/rmul_2027.sh" visual_quality:=cad
bash "$HOME/projects/guodong-sentry-sim/scripts/rmul_2027.sh" headless:=true rviz:=false
```

脚本不自动强制使用 NVIDIA。旧电脑曾使用 D3D12 加速；新电脑若已确认支持，可在启动命令前加：
`GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`。
非 NVIDIA 或非 WSL 电脑不要照搬旧电脑显卡配置。

## 5. 在新电脑验收

安装后在**完整仓库根目录**运行：

```bash
python3 tools/test_rmul_field.py
python3 tools/test_step_field.py
python3 tools/verify_bundle.py  # 如果完整仓库有这个工具和 manifest.json
source /opt/ros/jazzy/setup.bash
source output/validation/install/setup.bash
python3 tools/check_rmul_runtime.py
```

如使用自定义工作空间，将最后的 source 路径替换成对应的 `install/setup.bash`。
运行检查会独立启动仿真（ROS 域 97、随机 Gazebo 分区），捕获实际图像、检查扫描与低速西墙碰撞，
退出时清理它自己启动的进程。结果写入新电脑完整仓库的 `output/rmul_runtime/<编号>/`。
不要同时运行两个本检查。`RMUL_CHECK_GUI=1 python3 tools/check_rmul_runtime.py` 可连同 GUI 检查。

包内 `docs/validation/source_machine.json` 和 `previews/` 是**旧电脑历史证据**，
不能代替新电脑实际验收。初版历史：35 项静态检查通过；有/无 GUI 运行通过；当前 STEP R1 的检查以 source_machine.json 为准。初版
720 束扫描 95 分位误差约 21.54 mm，最大掠射误差约 7.884 m 保留，不能说所有射线精确对齐。

## 6. 必须保留的场地限制

地图内场 12×8 m，以内侧西南角为 (0,0)，红方在西北、蓝方在东南。
当前CAD尺寸见 `step_verified` 与 `step_measurements.json`，原手册参数已归档至 `parameters.json` 的 `manual_reference`。
当前几何以包内官方 STEP 为准；manual_reference 保存旧手册重建输入。CAD面颜色不等于真实材质测量，CAD与实物仍可能存在制造调整。

现有底盘是关闭重力的平面速度控制，尚无登高/地形跟随；水平雷达高约 279.62 mm，
能扫过 200 mm 高地和 270 mm 隧道顶。附带 PGM 是保守底盘通行投影，
**不是 SLAM 地图，也不是经过新场地 AMCL 验收的扫描地图**。
不要直接把旧 Depot 巡逻坐标迁过来；旧上位机导航 demo 仍显式选择 Depot。

本次包只用于迁移和场地检查，不开始毕业设计全套测试、训练或强化学习。
如后续继续开发，请先读取用户原计划并另建新场地验收矩阵；旧场地通过的数字不可当作新场地结果。

## 7. 提示词

将压缩包上传到另一台电脑的 Codex，再把 `迁移提示词.txt` 全文发给它。
提示词要求先检查环境和完整工程、再安装和实测，并明确上述边界。

## STEP 原始面颜色核对

`bash "$HOME/projects/guodong-sentry-sim/scripts/rmul_2027.sh" world:=rmul_2027_native`

该世界与默认场景的几何一致，显示官方 STEP 的 AP214 面颜色；默认场景保留手册地胶纹理风格。
运行不需要 OCCT；只有重新从源 STEP 转换才需要 OpenCascade Python OCP 7.7.2。
