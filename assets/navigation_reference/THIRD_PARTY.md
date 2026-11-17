# 来源和许可

- `assets/mid360.npy` 原样来自 [discoverse-dev/MuJoCo-LiDAR](https://github.com/discoverse-dev/MuJoCo-LiDAR/tree/e6b879df4b96daaed30ef10fe02650d29048fa9b)，commit `e6b879df4b96daaed30ef10fe02650d29048fa9b`。SHA256 `9fa0165576f0060488254f04bd647a891004d4f4faa6e9b03b69d830c117c9a3`。保留上游 MIT 许可 `assets/MuJoCo-LiDAR-LICENSE`；仅复用扫描表，本工程逐点求交/时序/IMU/ROS 实现独立编写。
- 物理运行库 [MuJoCo](https://github.com/google-deepmind/mujoco) 3.17.0，Apache-2.0，通过 Python wheel 使用，不复制其源代码。
- 静态求交 [embreex](https://github.com/trimesh/embreex) 4.4.0.post1 / Intel Embree，经 wheel 安装，许可证随依赖分发。
- 场地准备依赖 trimesh、Shapely、fast-simplification；记录实际安装版本见交付证据 `runtime-freeze.txt`。
- 四个 LIO 后端由本工作区现有 `pb2025_lio_backends` 复制构建，保留原文件及其许可；73 个 vendor 文件 SHA 清单在 `lio_ws/source-manifest.json`，不重新授权第三方内核。
- 场地来自已有授权派生 `rmuc_2027_driveable.stl`，SHA256 `97faa279776cc7b00a635dde51b1bc3dcf8ea91cb949fc73e88e28ea7b94905f`，不宣称公共分发许可。转换输出 manifest 保留原路径、坐标平移、接触近似与几何来源。

规格参考：[Livox MID360 FAQ](https://www.livoxtech.com/mid-360/faq)、[产品规格](https://www.livoxtech.com/mid-360/specs)、ICM40609-D datasheet。2026-11-05 的下载快照、TDK 链接和源码审查见工作区 `测试数据/测试结果/mid360_sim_research/20261105_candidates/`。芯片噪声参数不是整机实测标定值。
