"""Standalone sentry world; does not start or reconfigure Nav2."""
import os
import shlex
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def visual_suffix(context):
    quality=LaunchConfiguration('visual_quality').perform(context)
    if quality not in ('cad', 'preview'):
        raise ValueError('visual_quality must be cad or preview')
    return '_preview' if quality == 'preview' else ''

def generate_launch_description():
    share=Path(get_package_share_directory('guodong_sentry_sim'))
    gz_share=Path(get_package_share_directory('ros_gz_sim'))
    paths=[str(share/'models'),str(share.parent)]
    if os.environ.get('GZ_SIM_RESOURCE_PATH'):
        paths.append(os.environ['GZ_SIM_RESOURCE_PATH'])
    def gazebo(context):
        headless=LaunchConfiguration('headless').perform(context).lower() in ('true','1')
        world=LaunchConfiguration('world').perform(context)
        if world not in ('sentry_lab', 'depot', 'rmul_2027', 'rmul_2027_native'):
            raise ValueError('world must be sentry_lab, depot, rmul_2027 or rmul_2027_native')
        args=('-s --headless-rendering ' if headless else '')+'-r -v 3 '+shlex.quote(str(share/'worlds'/f'{world}{visual_suffix(context)}.sdf'))
        return [IncludeLaunchDescription(PythonLaunchDescriptionSource(str(gz_share/'launch/gz_sim.launch.py')),
                    launch_arguments={'gz_args':args}.items())]
    def robot(context):
        return [Node(package='robot_state_publisher',executable='robot_state_publisher',
             parameters=[{'robot_description':(share/'urdf'/f'sentry{visual_suffix(context)}.urdf').read_text(encoding='utf-8'),
                          'use_sim_time':True}],output='screen')]
    return LaunchDescription([
        DeclareLaunchArgument('rviz',default_value='true'),
        DeclareLaunchArgument('headless',default_value='false'),
        DeclareLaunchArgument('health',default_value='true'),
        DeclareLaunchArgument('world',default_value='rmul_2027'),
        DeclareLaunchArgument('visual_quality',default_value='cad'),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH',os.pathsep.join(paths)),
        OpaqueFunction(function=gazebo),
        Node(package='ros_gz_bridge',executable='parameter_bridge',name='sentry_bridge',
             parameters=[{'config_file':str(share/'config/bridge.yaml'),'use_sim_time':True}],output='screen'),
        OpaqueFunction(function=robot),
        Node(package='guodong_sentry_sim',executable='velocity_guard.py',output='screen'),
        Node(package='guodong_sentry_sim',executable='sensor_health.py',
             parameters=[{'use_sim_time':True}],condition=IfCondition(LaunchConfiguration('health')),
             output='screen'),
        Node(package='rviz2',executable='rviz2',arguments=['-d',str(share/'rviz/sentry.rviz')],
             parameters=[{'use_sim_time':True}],condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
