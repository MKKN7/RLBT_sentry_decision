# 依赖与包范围

只校验/安装文件：Python 3.10 或更新版本的标准库，不联网、不需 ROS。

只查看地图：Gazebo Harmonic / Gazebo Sim 8，Ogre2 渲染支持。ROS 2 Jazzy 安装中的
Gazebo vendor 可通过 `source /opt/ros/jazzy/setup.bash` 暴露 `gz` 命令。

运行生成器和五项场地静态测试：Python 3 + Pillow（Ubuntu 中通常为 python3-pil）。

构建并运行完整哨兵：Ubuntu 24.04、ROS 2 Jazzy、Gazebo Harmonic、colcon、ament_cmake、
ros_gz_sim、ros_gz_bridge、launch/launch_ros、ament_index_python、robot_state_publisher、RViz2、
rclpy、geometry_msgs、nav_msgs、sensor_msgs、tf2_msgs、rosgraph_msgs、diagnostic_msgs，
以及原项目完整的机器人网格、SDF、URDF、桥接配置和节点。

运行真实仿真场地检查：上述完整运行环境，加 NumPy、Pillow。
不要用 Windows Python 安装的 rclpy 去代替 WSL 中匹配 ROS 2 Jazzy 的 Python 环境。

本包不包含 ROS/Gazebo 安装包，不宣称可以在完全离线且无任何依赖的新系统直接运行。
恢复原项目及系统依赖后，场地的模型、纹理不需要在线下载。

未包含：完整机器人工程、上位机/下位机源码、LLM 权重、强化学习策略、训练数据或比赛裁判系统。
