# RL + BT 调研笔记

整理日期：2026-11-15。以下7篇均已获取原站PDF和正文，保存于 `data/references/papers/`；阅读重点为问题定义、方法、实验范围与局限。本轮未复现实验，未验证作者的所有证明。预印本与论文版本信息见 `bibliography.json`。

各条“本项目启发”是工程推断，不是论文在RoboMaster上的验证结果。

## 1. 切换边界价值：最直接对应“全局性”

**Improving the performance of Learned Controllers in Behavior Trees using Value Function Estimates at Switching Boundaries**，Mart Kartašev、Petter Ögren，2023预印本 / 本次PDF v3为2024-02-22。

[原站](https://arxiv.org/abs/2305.18903) · [本地PDF](../data/references/papers/2305.18903.pdf)

论文针对“各子控制器局部最优，组合后整体仍低效”的问题，在前一控制器的切换边界引入后一控制器的价值估计。其全局最优结论依赖文中的MDP、区域划分、控制器顺序与价值等假设，不能直接外推到部分可观测、动态对抗的RM比赛。重点阅读Introduction、方法与Discussion。

**本项目启发**：到洞口最快，不一定让“过洞—到达有利位置—控点/交战”的总收益最高；评价到达状态时还应考虑出口风险、后续路径、热量、姿态与剩余时间。这比只给每个叶节点单独设计成功奖励更贴近你的需求。

## 2. Progress Constraints：避免子任务互相拆台

**Progress Constraints for Reinforcement Learning in Behavior Trees**，Finn Rietz、Mart Kartašev、Petter Ögren、Johannes A. Stork，2026。

[原站](https://arxiv.org/abs/2602.06525) · [本地PDF](../data/references/papers/2602.06525.pdf)

论文基于BT收敛条件，学习可行性估计并约束允许动作，减少后续控制器破坏已完成子目标的问题；实验为2D概念环境及仓库移动操作环境。作者明确指出，学习可行性估计的数据覆盖可能不足。重点阅读§III–VI。

**本项目启发**：过洞、补给、控点不能仅各自贪心；需要明确哪些进度应保留、何时允许放弃。学习得到的可行性估计可辅助筛选，但不能替代已知禁区、射击权限和机械包络等确定约束。

## 3. Active Constraint Conditions：较早的进度保持思路

**Improving the Performance of Backward Chained Behavior Trees that use Reinforcement Learning**，Mart Kartašev、Justin Salér、Petter Ögren，2021预印本 / 本次PDF v3为2022-02-02。

[原站](https://arxiv.org/abs/2112.13744) · [本地PDF](../data/references/papers/2112.13744.pdf)

论文从后向链式BT的收敛条件出发，让RL控制器在完成当前子目标的同时避免违反Active Constraint Conditions（ACCs），减少重复完成先前子目标的成本。改善程度取决于原先违反约束的频率和重做代价；其结构假设不能忽略。

**本项目启发**：可以把“任务成功条件”和“执行时不得破坏的条件”分开描述。例如等待兑换到账期间，是否应维持脱战/安全条件，需要按对应规则与官方协议确认，不能凭任务名称臆定。

## 4. 在保留BT结构的情况下学习参数

**On the Fly Adaptation of Behavior Tree-Based Policies through Reinforcement Learning**，Marco Iannotta等，2025-03-08。

[原站](https://arxiv.org/abs/2503.06309) · [本地PDF](../data/references/papers/2503.06309.pdf)

上层策略根据上下文选择参数，下层BT执行控制；在仿真和Franka Panda上验证障碍避让、物体操作变化。论文重点是局部任务变化下的参数适配，并非比赛级全局调度；条件节点不可参数化也是其局限之一。

**本项目启发**：可比较学习目标点偏移、交战距离或任务阈值与直接学习子树选择的收益和成本。但参数适配本身不自动解决长时信用分配，仍需比赛级评估。

## 5. 学习模块接入现有BT的保护结构

**Adding Neural Network Controllers to Behavior Trees without Destroying Performance Guarantees**，Christopher Iliffe Sprague、Petter Ögren，2018预印本 / 本次PDF v3为2022-07-25。

[原站](https://arxiv.org/abs/1809.10283) · [本地PDF](../data/references/papers/1809.10283.pdf)

论文给出在满足明确条件时，组合安全控制器、模型控制器与学习控制器并保留原有安全/收敛性质的结构，以倒立摆为例。保护性质来自被验证的条件和控制器能力，不是只要套上Fallback节点就成立。

**本项目启发**：设计学习输出超时、观测无效、动作被拒绝、导航失败时的确定回退入口；同时记录实际采用了哪个控制分支。

## 6. Options：对应“时序延伸动作”

**The Option-Critic Architecture**，Pierre-Luc Bacon、Jean Harb、Doina Precup，2016预印本 / AAAI 2017；本次PDF v2为2016-12-03。

[原站](https://arxiv.org/abs/1609.05140) · [本地PDF](../data/references/papers/1609.05140.pdf)

Options把动作扩展为带启动集合、内部策略与终止条件的行为，Option-Critic学习内部策略和终止机制。它不是XML BT框架，使用其思想也不意味着必须采用原算法。

**本项目启发**：过洞、回补给、驻守、占点可看作持续时间不同的任务。日志应记录任务持续时长及终止原因；若在任务边界更新价值，应考虑时长对应的折扣，例如按基础时间步累计回报并使用`γ^k`，不能把1秒与20秒任务都视作同样的一步而不说明。

## 7. Temporal Behavior Trees：规格、奖励、课程和诊断

**Temporal Behavior Trees for Reinforcement Learning: Specifications as Rewards, Curricula, and Diagnostics**，Vladimir Krsmanović、Bernd Finkbeiner，2026-11-10，预印本v1。

[原站](https://arxiv.org/abs/2611.13993) · [本地PDF](../data/references/papers/2611.13993.pdf)

TBTCL把时序规格用于稠密奖励、课程DAG与轨迹分段诊断。作者说明其评估为两个小型领域中的六个规格，长轨迹分段计算可能昂贵，课程结构结果不保证学习收敛或最优顺序。重点阅读§3、§6。

**本项目启发**：适合研究“按顺序过卡、规定时间内完成、持续保持控点”的规格与失败分析。TBT时序逻辑语义与BehaviorTree.CPP运行树语义不同，不能把现有XML换个名字就获得这些能力；该工作主要组织奖励与课程，也不能当作已经实现了部署时的宏任务调度。

## 8. 已获取的公开工程

### SMBU PolarBear 2025 哨兵BT

[仓库](https://github.com/SMBU-PolarBear-Robotics-Team/pb2025_sentry_behavior)；本次快照commit：`d111b635d326775c6c19aa12500a0bbd6a27a588`。

已读取作者README，获取仓库目录树，并通过GitHub API保存、解析了该commit的 `behavior_trees/rmul_2025.xml`。本地文件为 `data/references/rm_bt_xml.xml`，根属性为`BTCPP_format="4"`，含比赛阶段检查、状态检查、补给判断、导航目标发布等节点。

工程自述使用Ubuntu22.04、ROS2 Humble、BehaviorTree.CPP 4.6.2；这些是该参考工程的环境，不是本项目已选依赖。XML里的`heat_max=350`和旧赛季注释不能套到2027自动哨兵热量上限260。

作者README还记录过ROS Action halt问题和topic替代方案；这是该工程的历史说明，本轮未核验其在当前上游版本是否仍存在。后续若参考其导航节点，要验证取消、执行反馈与目标到达语义，而不能把“已发布目标”当作“已完成导航”。

未运行其启动或安装命令，未验证该树适用于2027。

### BehaviorTree.CPP

[官方源码](https://github.com/BehaviorTree/BehaviorTree.CPP)的仓库页面已获取；文档站多次连接重置。后续选择版本后，应按对应源码核对Sequence、ReactiveSequence、Fallback、异步Action、`halt()`、Blackboard及SubTree端口语义。本轮不声称已审计这些实现。

## 9. 后续对照你的方案时的评估问题

1. RL到底学习任务选择、持续/终止、参数，还是价值估计？先定义接口再选算法。
2. 模型能否看到姿态预算、增益剩余时间、进行中的请求以及敌情时效？历史应如何保留？
3. BT的紧急抢占和RL的任务承诺如何共存？导航取消和真实停止如何确认？
4. 是否优化与胜负一致的团队长期收益？每个叶节点的成功是否会牺牲后续任务？
5. 相对于带记忆/计时器的纯BT，RL增加了多少收益？如何通过相同场景、相同观测和多个随机种子验证？

这些是调研形成的比较维度，不是已敲定的设计。
