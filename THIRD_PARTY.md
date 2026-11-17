# 本地复用来源

`data/references/alpharm_jax_review/upstream/`保留CaptainKAZ/alphaRM-JAX提交`9f2489ce78e46f0e37a21e90d4b11a0a1a94c4d2`的未修改选取文件及原Apache-2.0 LICENSE。来源与每个文件的Git blob SHA1、SHA256见`data/references/alpharm_jax_review_manifest.json`。

本项目的`rl_bt/alpharm_task_env.py`和`rl_bt/geometry.py`新增候选任务、执行代理、观测扰动和对手脚本，使用原ONNX图、fixture与静态地图配置；动作解码／掩码逻辑沿用此前新增的`scripts/probe_alpharm_onnx.py`对原`demo/web/engine.js`及`env.py`接口的适配。它们不是上游原文件，也不代表正式RMUL／RMUC规则。新增训练、规则计时、数据处理和测试代码位于本项目独立目录；上游快照没有被修改。

`assets/navigation_reference/`保留用户已有导航项目的来源说明与`THIRD_PARTY.md`。本轮只读取场地STL、配置和仿真录制，新增派生统计与低坡度面片缓存，没有修改原导航工程或共享容器文件。

COMBAT、HERO导航和论文材料用于接口与研究参考。真实Combat XML／C++／ROS工作区尚未接入；视频目录仅保存已核验来源，不包含视频动作标注或队伍私有遥测。

2027 UL官方手册、STEP和API页面来源存于`data/references/official_rmul_2027/`及`assets/rmul_2027/source_cad/`。官方及社区资源均保留原权利归属，不因本地研究下载推定有公开重新分发许可。社区转换包保存于`data/references/rmul_2027_community_map/`，内含迁移说明与脚本均未执行；仅独立解析JSON三角面和测量信息生成研究派生STL，来源哈希和边界检查见`assets/rmul_2027/derived/manifest.json`。未把社区转换等同为独立重现官方CAD转换。
