



  







  
    
  
  
  
  
  
  













  


  

  

    
    
    
    
  


  

  











  
  




































  GitHub - SMBU-PolarBear-Robotics-Team/pb2025_sentry_behavior: SMBU Polarbear Robotics Team's behavior tree for sentry robot decision-making in RoboMaster 2025, based on BehaviorTree.CPP. · GitHub



  
  
  
  

    
  


  


    


  
  

  
  

    







  

  




    

  

    

    

      

      

    
    
    

      
  
  



      



        


  
  
  
  

  

      

    

  

  
  



    

    
  


  


  

  
  

  
  
  





  

  

  
    

      

      

    

      Skip to content

      
    
      
      




  
  
  







      

        
            





  
  
  




            

                











  
  
  


Navigation Menu




Sign in



Products



Platform
GitHubRepository hosting, code review, and collaboration
CopilotAI coding agent for tasks, projects, and automated work

Build & Ship
ActionsCI/CD and automation
Code QualityQuality analysis, fixes, and enforcement
CodespacesCloud development environments

Secure
Code SecurityVulnerability detection and remediation
Secret ProtectionSecret scanning and leak prevention
All Features

Developers



Learn
Documentation
Training
Certifications

Build
Build with Copilot
Customize Copilot
Integrations

Explore
Trending repos
GitHub Sponsors
Developer articles

Changelog
ChangelogYour source for the latest features, improvements, and fixes across GitHub products, all in one place.

Businesses



By Stage
Enterprises
Team
Startups
Nonprofits

By Industry
Healthcare
Financial services
Manufacturing
Government
See all industries

Evaluate GitHub
Why GitHub
Customer Stories
Business insights
Trust Center

Add-ons
GitHub Advanced SecurityEnterprise-grade code security
Copilot for businessEnterprise-grade AI features
Premium SupportEnterprise-grade 24/7 support

Resources



Learn & Connect
Documentation
Resource center
Blog
Events & webinars
Community Forum
Customer Stories
Integrations

Support
Customer support
Training
GitHub Status

Company
About
Partners
Changelog

GitHub Universe
GitHub UniverseThe event for developers, by developers — talks, workshops, and what’s next for GitHub.
Pricing

Search/


Sign inSign up
Appearance settings






      
        

  
          
    

          You signed in with another tab or window. Reload to refresh your session.
          You signed out in another tab or window. Reload to refresh your session.
          You switched accounts on another tab or window. Reload to refresh your session.

      
    

Dismiss alert


  

    

  









    






  
    


  

    
      
    

    
    

      
      
{{ message }}

    
  

  



    






  

        

    
      
      
    


    








    














  
  
  



SMBU-PolarBear-Robotics-Team
pb2025_sentry_behavior

Repository navigation

Code
Issues1 (1)
Pull requests1 (1)
Actions
Projects
Security and quality
Insights

More items




  




    

    



    
      
    









  
  
  
   
















pb2025_sentry_behaviorPublic

NotificationsYou must be signed in to change notification settings
Fork9 (9)
Star47 (47)You must be signed in to star a repository


About
SMBU Polarbear Robotics Team's behavior tree for sentry robot decision-making in RoboMaster 2025, based on BehaviorTree.CPP.
Topics
behavior-treerobomasterros2
Resources
Readme
Apache-2.0 license
Contributing
Contributing


Activity
Custom properties

Stars
47 stars
Watchers
1 watching
Forks
9 forks
Report repository










main

BranchesTags






Go to fileCode
Open more actions menu





Latest commit
 



History28 Commits

28 Commits
Folders and files
NameName
Last commit message
Last commit date



.github/workflows



.github/workflows
 

 



behavior_trees



behavior_trees
 

 



include/pb2025_sentry_behavior



include/pb2025_sentry_behavior
 

 



launch



launch
 

 



params



params
 

 



plugins



plugins
 

 



src



src
 

 



.clang-format



.clang-format
 

 



.clang-tidy



.clang-tidy
 

 



.gitignore



.gitignore
 

 



.pre-commit-config.yaml



.pre-commit-config.yaml
 

 



CMakeLists.txt



CMakeLists.txt
 

 



CONTRIBUTING.md



CONTRIBUTING.md
 

 



LICENSE



LICENSE
 

 



README.md



README.md
 

 



dependencies.repos



dependencies.repos
 

 



package.xml



package.xml
 

 
View all files



Repository files navigation

README
Contributing
Apache-2.0 license

More items



pb2025_sentry_behavior







1. Overview


开发中，不考虑向前兼容性，仅供参考，请谨慎使用。文档可能不会及时更新以反映代码的最新变化。


基于 BehaviorTree.CPP 和 BehaviorTree.ROS2 的行为树框架与插件，用于 RoboMaster 2025 赛季哨兵机器人。


2. Quick Start


2.1 Setup Environment


Ubuntu 22.04

ROS2 Humble

BehaviorTree.CPP (Developed with release 4.6.2)



2.2 Create Workspace

mkdir -p ~/ros_ws
cd ~/ros_ws

pip install vcstool2

git clone https://github.com/SMBU-PolarBear-Robotics-Team/pb2025_sentry_behavior.git src/pb2025_sentry_behavior

vcs import --recursive src < src/pb2025_sentry_behavior/dependencies.repos


2.3 Build

rosdepc install -r --from-paths src --ignore-src --rosdistro $ROS_DISTRO -y

colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=release


2.4 Running

ros2 launch pb2025_sentry_behavior pb2025_sentry_behavior_launch.py


3. Behaviors


3.1 Action

CalculateAttackPose



CalculateAttackPose 动作节点用于计算并输出进攻坐标，该节点基于敌方坐标和全局代价地图生成候选点，过滤出可行点，并选择最佳攻击点。

Input Ports:


costmap_port：类型为 nav_msgs::msg::OccupancyGrid，表示全局代价地图。

tracker_port：类型为 auto_aim_interfaces::msg::Target，表示视觉估计得到的目标状态信息。Related repo: armor_tracker


Output Ports:


goal：类型为 geometry_msgs::msg::PoseStamped，表示发送给导航系统的目标姿态。


Parameters:


attack_radius：以目标位置为圆心的半径，default: 3.0。

num_sectors：划分扇区数量，default: 36。

cost_threshold：代价阈值，default: 50。

robot_base_frame：机器人基坐标系，default: "chassis"。

transform_tolerance：查询 tf 超时时间，default: 0.5。

visualize：是否启用可视化，default: false。


功能描述


transformToCostmapFrame：将敌方位置转换到代价地图框架。

generateCandidatePoints：生成候选攻击点。

filterFeasiblePoints：过滤出可行的攻击点。

selectBestPoint：选择最佳攻击点。

createAttackPose：创建攻击姿态。

createVisualizationMarkers：创建可视化标记。


PubTwist

以 geometry_msgs/msg/Twist 的形式发布速度，用于控制底盘运动。

SendNav2Goal

创建 Client，以 nav2_msgs/action/NavigateToPose 的形式发送 Navigation2 目标点。


Caution
BehaviorTree.ROS2 中存在 bug，导致继承自 RosActionNode 的节点无法被正确 halt，最终导致行为树被 shutdown。因此，下面提供了一个以 topic 代替 action 发布 goal_pose 的临时方案 PubNav2Goal，坏处是无法实时获取到 action 的 feedback 和 result。

Related Issue:  #18 Error when canceling action during halt()


PubNav2Goal

以 geometry_msgs/msg/pose_stamped 的形式发布 Navigation2 目标点。


3.2 Condition

IsAttacked

通过 GlobalBlackboard 获取实时的 pb_rm_interfaces::msg::RobotStatus 类型数据，判断机器人是否受到攻击，并根据裁判系统装甲模块反馈的信息输出敌方可能的角度位置。该条件节点会根据输入端口的配置，检查以下几个条件：


key_port：从 GlobalBlackboard 获取 RobotStatus 消息

gimbal_pitch：输出固定的云台俯仰角度（0.0）

gimbal_yaw：输出敌方可能的角度位置


如果检测到装甲板被击中，则返回 SUCCESS，并输出相应的云台角度；否则返回 FAILURE。

IsDetectEnemy

通过视觉模块判断是否感知到敌方。


armor_id：预期的装甲 ID 列表。可以是多个数字，用 ; 分隔

max_distance：敌方目标的最大距离


如果视觉模块感知到的敌方在 armor_id 列表中，且距离小于 max_distance，则返回 SUCCESS，否则返回 FAILURE。

IsGameStatus

通过 GlobalBlackboard 获取实时的 pb_rm_interfaces::msg::GameStatus 类型数据，判断当前比赛状态是否在输入的时间范围内且处于预期的比赛阶段。该条件节点会根据输入端口的配置，检查以下几个条件：


key_port：从 GlobalBlackboard 获取 GameStatus 消息

expected_game_progress：预期的比赛阶段

min_remain_time：最小剩余时间（秒）

max_remain_time：最大剩余时间（秒）


如果比赛阶段和剩余时间都符合预期，则返回 SUCCESS，否则返回 FAILURE。

IsRfidDetected

通过 GlobalBlackboard 获取实时的 pb_rm_interfaces::msg::RfidStatus 类型数据，判断机器人是否检测到指定的 RFID 标签。该条件节点会根据输入端口的配置，检查以下几个位置的 RFID 状态：


key_port：从 GlobalBlackboard 获取 RfidStatus 消息

friendly_fortress_gain_point：己方堡垒增益点

friendly_supply_zone_non_exchange：己方与兑换区不重叠的补给区 / RMUL 补给区

friendly_supply_zone_exchange：己方与兑换区重叠的补给区

center_gain_point：中心增益点（仅 RMUL 适用）


如果任意一个配置为 true 的位置检测到 RFID 标签，则返回 SUCCESS，否则返回 FAILURE。

IsStatusOK

通过 GlobalBlackboard 获取实时的 pb_rm_interfaces::msg::RobotStatus 类型数据，判断机器人的状态是否正常。该条件节点会根据输入端口的配置，检查以下几个条件：


key_port：从 GlobalBlackboard 获取 RobotStatus 消息

hp_min：最低血量

heat_max：最大发射机构的射击热量

ammo_min：最小弹丸允许发弹量


如果机器人的 HP、热量和弹药量都在预期范围内，则返回 SUCCESS，否则返回 FAILURE。






About
SMBU Polarbear Robotics Team's behavior tree for sentry robot decision-making in RoboMaster 2025, based on BehaviorTree.CPP.
Topics
behavior-treerobomasterros2
Resources
Readme
Apache-2.0 license
Contributing
Contributing


Activity
Custom properties

Stars
47 stars
Watchers
1 watching
Forks
9 forks
Report repository

Releases



Packages


Used by


Contributors


Languages




   





  



    
  

  

  
            
  
Footer

  


  

    

      
        
    


      
        © 2026 GitHub, Inc.
      
    

    
      
Footer navigation

      


          

            Terms
          

          

            Privacy
          


            

              Security
            

            

              Status
            

          

            Community
          

          

            Docs
          

          

            Contact
          

          


  
    
      Manage cookies
    
  


  

    
      
        Do not share my personal information
      
    
  

      
    
  




    




  

    
    

    
      
    

    
    You can’t perform that action at this time.
  

    
  
    
    
      
        
    

      
      

    
  


    

  

  


    
  

    
      
    

      
    

    
  


  

    
      
    

      
    

    
  





    
    

    

  


