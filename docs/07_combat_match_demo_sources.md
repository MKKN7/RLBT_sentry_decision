# COMBAT 的 RMUL / RMUC 比赛 demo 来源

检索与核验日期：2026-11-15。目标队伍为南京理工大学江阴校区 COMBAT，不是南京理工大学 Alliance。

本轮已实际打开视频详情页，核对标题、上传者、上传时间、显示时长与合集归属；未完成逐局观看、战术标注、坐标提取或原始视频下载。下列日期均为平台上传时间，不据此推断实际比赛日期。

## 1. 官方身份与入口

- [南京理工Combat战队](https://space.bilibili.com/3546890614344273)：账号简介写明“南京理工大学江阴校区RoboMaster战队官方账号”。
- [RoboMaster机甲大师](https://space.bilibili.com/20554233)：下列RMUL/RMUC赛事回放上传者。
- [COMBAT公开代码组织](https://github.com/Combat-Algorithm-Team)：用户此前提供的BT仓库所属组织。仓库包含继承的上游文档，不能把所有演示链接都归为COMBAT。

## 2. 已核实的视频

|类别|视频|上传者|上传日期|显示时长|用途|
|---|---|---|---|---|---|
|RMUL 2026 江苏站第4场|[COMBAT vs 浙江大学 Hello World](https://www.bilibili.com/video/BV1oNXTB2EoE/)|RoboMaster机甲大师|2026-03-27|08:32|比赛级上下文，后续标注接敌/支援/回补等可见行为|
|RMUC 2026 东部赛区第14场|[COMBAT vs 上海工程技术大学 木鸢Birdiebot](https://www.bilibili.com/video/BV1bSLt6qEf2/)|RoboMaster机甲大师|2026-05-21|26:23|超抗攻防、前哨/基地与路径决策的可见场景|
|哨兵第一视角片段|[沉浸式体验烧饼速推前哨](https://www.bilibili.com/video/BV1hYV86uEGP/)|呼吸好累|2026-05-29|03:12|可见敌情、交战顺序与感知受限片段；需后续标注|
|战队官方哨兵片段|[哨兵得了MVP——操作手绝望时刻](https://www.bilibili.com/video/BV1nMEBzTEEY/)|南京理工Combat战队|2025-05-14|00:36|短片段参考，不作为完整对局经验|

第一视角片段的关联依据：详情页标签包含Combat、哨兵、第一视角、东部赛区，且战队官方账号在5月29日动态中转发，并评论“哨兵不是区”。已通过[官方账号的哨兵动态检索](https://space.bilibili.com/3546890614344273/search?keyword=%E5%93%A8%E5%85%B5)看到该关联。它不是战队官方账号原创上传；精确对手、场次和局次仍未核实，不能直接把它拼接到上述RMUC第14场。

## 3. 可继续定位其他场次的官方合集

- [RMUL 2026 江苏站比赛回放](https://space.bilibili.com/20554233/channel/collectiondetail?sid=7763972)：已核对合集58条、当前视频为4/58。
- [RMUC 2026 区域赛·东部赛区比赛回放](https://space.bilibili.com/20554233/channel/collectiondetail?sid=8156146)：已核对合集89条、当前视频为14/89。

RMUL播放页还显示两条同校条目，作为后续线索保存，尚未逐条打开详情：

- 江苏站第35场：三江学院破浪 vs 南京理工大学江阴校区，标题将队名拼为“Combot”，时长09:27。
- 江苏站第48场：南京理工大学江阴校区Combat vs 江南大学霞客湾校区SHARK，时长07:30。

这些条目目前没有在清单中伪造独立BV号，也不将标题拼写错误改写为已确认的队伍身份；后续可从合集精确定位。

## 4. 公开原始数据的检索结果

已通过GitHub API查看COMBAT组织当前14个公开仓库列表，重点检查了：

- `combat_sentry_behavior`：递归目录与Releases。
- `combat_sentry_nav`：递归目录、README与Releases。
- `sentry_nav_ws`：递归目录与Releases。
- `Combat_vision2026`：递归目录与顶层README。

在上述范围内未定位到可确认属于COMBAT比赛的rosbag、MCAP、DB3、状态—动作轨迹数据包或下载入口；上述三个导航/BT相关仓库的Releases返回空列表。这是限定范围内的检索结论，不代表队伍没有私有日志，也不代表整个互联网不存在相关数据。

`sentry_nav_ws`有记录导航日志的工具源码，但工具本身不是历史比赛数据。

需要排除的混淆来源：

- COMBAT导航仓库README保留北极熊导航演示和FlowUs先验地图链接。这些不是COMBAT的比赛demo或比赛轨迹。
- COMBAT雷达仓库README包含港科大ENTERPRIZE数据集/第一视角共享链接。其来源已在文档注明，不作为COMBAT对局数据。
- “raycast demo”或检测训练集不等同于哨兵战术决策状态—动作数据。

## 5. 对当前训练的使用方式

先将完整回放作为比赛背景，将第一视角作为局部观测参考。逐片段建立任务、接敌、目标丢失、前进/驻守/撤离、补给、受阻/恢复等标签；只标画面能支持的事实，并标注置信度。

路线坐标需要地图、机位或第一视角定位依据；画面没有的候选集合、BT原因、真实敌方状态、精确控制输入不能补造。视频上传时间、剪辑时间轴与比赛倒计时也不能直接视为同一时钟。

由此可以建立场景库和对手风格假设，再做可交互脚本复现。要用于行为克隆或离线RL，仍需额外获得可靠的观测/动作/时间对齐，不能仅凭这四条视频就宣称已有可直接训练的数据集。2025/2026规则与2027不同，机制和数值不能照搬。

机器可读条目见`data/demos/combat_video_catalog.json`与CSV；公开仓库检索依据见`data/references/combat_demo_search_manifest.json`。

## 6. 获取限制

通用搜索接口返回404，部分搜索结果无效；B站直接HTTP接口返回412/验证页，浏览器访问公开搜索、主页和详情页成功。Google后续出现异常流量验证，未处理验证码。未登录、未联系上传者、未下载私有资料。通过公开页面已完成上述四条视频的元数据核验。
