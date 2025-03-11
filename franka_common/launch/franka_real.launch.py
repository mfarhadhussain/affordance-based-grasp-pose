from launch import LaunchDescription 
from launch_ros.actions import Node 
from launch.actions import IncludeLaunchDescription 
from launch.launch_description_sources import PythonLaunchDescriptionSource 
import os 

def generate_launch_description()-> LaunchDescription: 
    
    franka_real_camera_node = Node(
        package='franka_real',
        executable='franka_real_camera_node',
        name='franka_real_camera_node',
        output='screen',
        parameters=[
            {'update_frequency': 10.0},  
            {'camera_namespace': 'camera'},
            {'robot_namespace': 'robot'}
        ]
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
            franka_real_camera_node,
            model_node,
            affordance_pose_plot_node,
        ]
    )
