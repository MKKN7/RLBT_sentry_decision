# 实际观测支撑接入 Nav2

2026-11-16。上一轮 UC 原图已跑通真实 Point-LIO/Nav2/MPPI，但 rolling costmap 仍有空闲先验。本轮默认 UC 改为严格观测支撑模式：没有观察到连续地面的栅格不能直接成为候选或路径。

## 实现边界

- `rl_bt/support_costmap.py` 保留语义图 `0=观测支撑，100=障碍，-1=unknown`，另向导航发布 `unknown→100` 的严格图。两套 costmap 使用 StaticLayer 首插件、顶层 `use_maximum=false`，ObstacleLayer 只按 max 合并，关闭支撑和障碍层的 footprint clearing 以及分类障碍云的射线清空。
- 完整 OccupancyGrid 发布到独立 namespace；严格图外围追加约2.1米 lethal padding，发布时检查覆盖当前全局rolling窗口。源图、local/global costmap 都使用0.15米分辨率。只让底盘在源语义图中有支撑的区域运动，不借原 STL 或 GT 补未知地面。
- Humble 1.1.20 的 MPPI 对 unknown 和圆模式 footprint 有特殊处理，已按官方固定版本源码核对。观察模式使用包住0.411米圆的16边外切polygon，再启用 `consider_footprint=true`，不缩小原圆包络；Navfn保持 `allow_unknown=false`。源码与hash存于 `data/references/nav2_support_review/`。
- BT在ComputePath返回后，使用连续线段扫掠圆足迹检查全部相交格，拒绝 unknown、障碍和图外；执行时还检查当前位置全足迹，以及支撑图、障碍云、odom、TF各自采样时间。支撑图缺失或超过2.5秒时取消并锁停，障碍云持续更新也不能替代支撑图。
- `--start-footprint-prior` 是显式工程先验，只把固定session起点(0,0)、半径0.411米相交的unknown格假定有支撑，既有occupied不清除。先验中心不可随车移动；默认不开启。它处理当前脚下雷达自滤孔，不是传感器证明。当前成功样本均明确启用了它，不能推广为任意未知起点可启动。

## 保留真实扫描与双雷达配准

旧worker每秒只取单帧，严格模式暴露支撑稀疏。现在 `TerrainWorker.submit_batch` 以最多1Hz执行分类，但保留这段时间内的实际扫描；每帧沿用原始采样/接收时钟，逐帧更新体素。旧观测不会全部改成批末时间。队列仍latest-only、有字节/数量/年龄界限，不积压分类。

原 Point-LIO transport 只订阅前雷达。两个 MID360 原来都实际仿真并发布，现新增 `rl_bt/scan_registration.py`，以Point-LIO估计底盘位姿history逐点插值平移和SLERP旋转，结合安装/encoder TF，将两台原始扫描注册到同一session odom供地形模块使用。原Point-LIO估计内核仍使用前雷达，未偷换成双雷达SLAM。逐点时间来自cloud header+`offset_time`，无外推、覆盖不足或间隔过大则拒绝。每对扫描用最早return作采样时间，保守计算年龄。

当前雷达座没有运动指令；首末外参变化超过阈值拒绝，但尚未实现任意运动雷达座逐点外参补偿。注册成功和支撑图发布分别计数，不能混淆。GT仅记录诊断，没有进入pose history、注册、体素或候选。

另外修复了两项工程问题：每次运行stage到独立目录，避免并发覆盖运行源码；Point-LIO的 `effective_config` 显式指定私有run路径，记录共享默认文件前后hash。先前版本未显式指定该参数，不能将其“共享文件不改”的旧声明当成已核验事实。本轮新记录验证了共享effective文件不变。

## 实际运行记录

|运行|结果及含义|
|---|---|
|`runs/uc_support_unknown_reject_20261116/`|首个严格层验证：未观测目标被Navfn拒绝，0条MPPI命令，BT锁停。预期为拒绝，不能计到点成功。|
|`runs/uc_support_observed_forward_20261116/`|单雷达严格模式1.2米目标被拒绝；起点足迹合法，但前向约1.9厘米接触unknown，记录保留。|
|`runs/uc_support_dual_forward_20261116/`|双雷达首轮TF命名空间错误，没有支撑图，最终wall timeout。失败记录保留；已改为订阅独立namespace的encoder TF。|
|`runs/uc_support_dual_namespaced_forward_20261116/`|双雷达注册生效、支撑覆盖增加，正前方1.2米仍有unknown缺口，Navfn拒绝、0命令；没有放宽unknown使其通过。|
|`runs/uc_support_dual_lateral_20261116/`|严格模式已观测(0,0.5)米侧移通过完整圆足迹门禁，Nav2 succeeded，末尾估计误差0.10462米。|
|`runs/uc_support_dual_unknown_reject_20261116/`|同一双雷达/严格配置下，(5,5)米未观测目标被拒绝，未下发导航速度。|
|`runs/uc_support_map_dropout_20261116/`|执行侧移时只注入支撑图失更，障碍云仍更新；6.6秒BT报告support_map_missing_or_stale，6.615秒收到canceled终态，安全锁触发；末尾估计平移约0.0297米/秒。|
|`runs/uc_support_dual_lateral_final_20261116/`|在派发前也要求当前足迹有支撑的实验版持续等待至wall timeout；该版本未下发命令。随后将派发前判断交给包含实际起点的完整路径足迹检查，执行期仍逐tick检查当前足迹。|
|`runs/uc_support_active_footprint_lateral_20261116/`|0.5米任务已执行，但约y=0.292米时新图把当前足迹边缘归为unknown；BT取消、锁停，尚未到目标容差。该结果是动态支撑撤销的实际停车记录，不是到点成功。|
|`runs/uc_support_active_footprint_short_20261116/`|最终当前足迹监控版在已观测短区域执行(0,0.4)米任务succeeded，末尾估计误差0.11069米；未降低0.411米包络或放宽unknown。短任务不能替代较长路线稳定性验证。|
|`runs/uc_support_map_dropout_final_20261116/`|最后版本复测掉图场景时，BT先发现实际估计起点足迹接触unknown，返回unsupported_path并锁停，未派发导航；未进入故障注入的active取消阶段，不能计作再次通过掉图验收。|

完整记录汇总：`data/reports/uc_support_execution_20261116.json`，每个run保存源码快照/hash、实际costmap、支撑图、BT事件、注册/worker日志。`success_for_reported_scope=true` 可以表示“按预期拒绝/取消”，另用 `navigation_succeeded` 明确区分到点。最终版本样本为 `uc_support_active_footprint_short`；不能将先前不含执行期当前足迹检查的0.5米成功替代最终较长路线验收。

复现已观测侧移（输出目录必须不存在）：

```bash
/home/kellen/anaconda3/envs/robot/bin/python scripts/run_nav2_closed_loop.py \
  --scene rmuc --goal 0 0.4 --start-footprint-prior --warmup-s 5 \
  --wall-timeout-s 120 --out runs/uc_support_lateral_NEW
```

复现“支撑图停止、障碍云继续”故障注入：使用目标`--goal 0 0.5`，增加 `--support-dropout-after-s 5 --expected-status canceled`。原图资源若被 `/tmp` 清理，先运行 `scripts/prepare_uc_original_field.py`。显式选择 `--terrain-prior clear` 才能回到旧执行测试先验；不能把该模式结果计入严格支撑验证。

完整测试210 passed；最后修正派发前等待/实际起点检查后，31项相关回归及实际短路线再次通过。严格模式暴露自遮挡与3秒体素保留下的支撑不稳定：底盘到达已观测位置后，盲区和支撑连通变化可使边缘重新变unknown。下一步应研究带原始时间与来源的静态支撑记忆和连续候选更新，不能直接扩大起点先验、移动画自由盘或取消unknown检查。相同分辨率仍不能证明rolling origin重采样对所有细洞完全保守，源图连续足迹门禁和执行期当前足迹检查均须保留。

本轮未启动正式UC PPO训练。候选mask可以复用同一支撑/路径足迹门禁，但当前图仍受自滤盲区、支撑连通、定位误差、栅格化和遮挡影响，尚不是完整比赛导航验收。原图前向1.2米的旧成功来自clear先验，不能覆盖本轮严格模式的失败；过洞、任意旋转雷达座、长路线、实机NUC和真实Combat XML仍需独立验证。
