from launch import LaunchDescription 
from launch_ros.actions import Node 
from launch.actions import IncludeLaunchDescription 
from launch.launch_description_sources import PythonLaunchDescriptionSource 
import os 

def generate_launch_description()-> LaunchDescription: 
    
    franka_sim_node = Node(
        package='franka_sim',
        executable='franka_sim_node',
        name='franka_sim_node',
        output='screen',
        parameters=[
            {'update_frequency': 10.0},  
            {'camera_namespace': 'camera'},
            {'robot_namespace': 'robot'}
        ]
    )

    franka_sim_point_cloud_node = Node(
        package='franka_sim',
        executable='franka_sim_point_cloud_node',
        name='franka_sim_point_cloud_node',
        output='screen',
    )

    model_node = Node(
        package='franka_common',
        executable='model_node',
        name='model_node',
        output='screen',
    )
    
    affordance_pose_plot_node = Node(
        package='franka_common',
        executable='affordance_pose_plot_node',
        name='affordance_pose_plot_node',
        output='screen',
    )
    
    return LaunchDescription(
        [
            franka_sim_node,
            franka_sim_point_cloud_node,
            model_node,
            affordance_pose_plot_node,
        ]
    )
