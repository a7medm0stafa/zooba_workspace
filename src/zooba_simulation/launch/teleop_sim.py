"""
Full simulation stack launch file for manual teleop control.

Launches:
    - gazebo_ackermann_steering_vehicle vehicle.launch.py (Gazebo + vehicle model)
    - sim_bridge_node (VehicleCmd → Float64 conversion)
    - teleop_keyboard_node (keyboard → /vehicle/cmd)

Usage:
    ros2 launch zooba_simulation full_sim.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    # ---- Launch arguments ----
    world_arg = DeclareLaunchArgument(
        'world', default_value='empty.sdf',
        description='Gazebo world file'
    )

    # ---- Include the upstream vehicle launch ----
    gazebo_pkg = get_package_share_directory('gazebo_ackermann_steering_vehicle')
    vehicle_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg, 'launch', 'vehicle.launch.py')
        ),
        launch_arguments={
            'world': LaunchConfiguration('world'),
        }.items()
    )

    # ---- Simulation bridge node ----
    sim_bridge_node = Node(
        package='zooba_simulation',
        executable='sim_bridge_node',
        name='sim_bridge_node',
        output='screen',
        parameters=[{
            'input_topic': '/vehicle/cmd',
            'steering_topic': '/steering_angle',
            'velocity_topic': '/velocity',
        }],
    )

    # ---- Teleop node (publishes directly to /vehicle/cmd) ----
    teleop_node = Node(
        package='mid_level_controller',
        executable='teleop_keyboard_node',
        name='teleop_keyboard_node',
        output='screen',
        prefix='xterm -e',
        parameters=[{
            'output_topic': '/vehicle/cmd',
            'publish_rate': 10.0,
            'velocity_step': 0.1,
            'heading_step': 5.0,
            'max_velocity': 2.0,
            'max_heading': 35.0,
        }],
    )

    return LaunchDescription([
        world_arg,
        vehicle_launch,
        sim_bridge_node,
        teleop_node,
    ])
