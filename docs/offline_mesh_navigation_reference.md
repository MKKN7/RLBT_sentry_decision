# UL 离线车体包络与网格表面检查

日期：2026-11-16。

`rl_bt/mesh_navigation.py` 补充了之前竖直分层采样无法检查的侧壁与车体横向包络：

- 任意 yaw 的直立长方体与全部三角面进行 13 轴 SAT 相交检查，包含洞顶与竖直侧壁。
- 支撑点只接受调用方明确给出的支撑层高度、坡度与高度容差，不会自动跳到洞顶或场地下表面。
- 用包含四角的 3×3 模板检查支撑样本；沿线段按显式平移／转角间距采样，检查途中阻挡。

本模块没有默认实机尺寸。实验命令使用 **0.48×0.40×0.20m、底部离支撑层0.01m的合成包络**，这是诊断输入，不是已测量龟哨参数。已有导航仿真的 chassis、轮组及传感器几何并不等同这个合成包络。

```bash
cd /home/kellen/桌面/RL_BT_decision
/home/kellen/anaconda3/envs/robot/bin/python -m pytest -q tests/test_mesh_navigation.py
/home/kellen/anaconda3/envs/robot/bin/python scripts/probe_ul_mesh_navigation.py \
  --length-m 0.48 --width-m 0.40 --body-height-m 0.20 \
  --belly-clearance-m 0.01 --support-tolerance-m 0.003 \
  --translation-step-m 0.025 \
  --out data/reports/ul_mesh_navigation_reference_20261116.json
```

在来源哈希已核对的 UL STL 上，示例低包络通过两个洞底姿态及红侧58／蓝侧57个采样姿态的穿洞线段检查；0.30m高包络被洞顶拒绝，贴侧壁姿态和错误支撑层姿态被拒绝。补给区、中心外平地与高地控制面各选一个示例姿态检查。结果只描述这些姿态及采样间距，不能推广为全场可达性。

报告：`data/reports/ul_mesh_navigation_reference_20261116.json`。

仍有边界：三角面相交没有证明实体内部／外部；有限模板没有证明连续支撑；有限路径采样没有证明连续扫掠安全。未模拟车轮、俯仰侧倾、悬架、动态障碍或定位不确定性，也没有连接 Nav2／MPPI、MuJoCo 接触或实机 TF。`sampled_reference_pass` 因而不能直接成为正式 RL 合法动作 mask；目前可作离线否决反例与导航服务输入检查。
