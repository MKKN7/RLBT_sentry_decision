# 导航仿真资源快照

来源核对与复制日期：2026-11-15。仅供当前项目本地研究使用，来源工程未改动。

- `field/field.stl`：从正在使用的容器 `Combat_Sentry2026:/opt/mid360_mujoco/generated/rmuc2027-v5/field.stl` 复制，385,026三角面，SHA-256见项目的资源清单。
- `field/source_manifest.summary.json`：原转换manifest的摘要，保留源文件、坐标平移、数量与近似说明。它不是原manifest全文。
- `source/`：原样复制的机器人模型生成器及最小配置加载模块；不是完整传感器仿真或ROS工程。
- `configs/`：原配置快照；其中机器人尺寸、动力学和噪声是该导航仿真的配置，不代表队伍实机标定。
- `source_notes/`：原工程README与依赖声明，按历史资料保存，不自动执行其中命令。
- `THIRD_PARTY.md`：原工程来源与许可说明。场地没有声明公共分发许可，不能把本地复制解释成可公开重新分发。

未复制：24,325个分解碰撞体、逐点LiDAR扫描表、ROS/LIO工作区、录包、容器镜像、日志与备份。完整MuJoCo接触验证仍需对接原场地碰撞资源；不得把这里的静态STL直接当作等价的单个凸碰撞体，否则可能封闭隧道。

`field.stl`已经处于原导航仿真的准备后世界坐标，不能再重复应用源CAD平移。其包围盒约28×16m，包含外围，不代表有效比赛区宽度是16m，也不能为匹配规则尺寸擅自缩放。到官方坐标的注册仍待核对。

已执行的检查命令：

```bash
/home/kellen/anaconda3/envs/robot/bin/python "/home/kellen/桌面/RL_BT_decision/scripts/probe_navigation_assets.py" \
  --output "/home/kellen/桌面/RL_BT_decision/data/reports/navigation_asset_probe.json"
```

检查范围为文件哈希、STL结构、复制的机器人生成器、简单测试地面1000步CPU物理和qpos复位。未验证全场接触、过洞、传感器、ROS、GPU多世界或RL训练。
# UC 原始场地增补（2026-11-16）

`field/field.stl` 是原导航工程历史整平的 driveable 派生图。用户指定 RMUC STL 后，另保存未经整平原图经原 world 平移的 prepared 副本 `uc_original/field.stl`，SHA256 `93ff0118af176f388c171b239acfdcac72c17cbe0b523350d5d916d3236fe10b`，379082 面。没有覆盖派生图。原图碰撞片留在容器私有目录，可用 `scripts/prepare_uc_original_field.py` 重建；完整来源核验见 `data/reports/uc_original_field_20261116.json`。两套缓存与实验记录分别维护。
