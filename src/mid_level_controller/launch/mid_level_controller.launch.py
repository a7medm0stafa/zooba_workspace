"""
Launch file for the full mid-level controller stack.

Launches:
    - teleop_keyboard_node   (keyboard input → /teleop/raw_cmd)
    - nonholonomic_constraints_node (/teleop/raw_cmd → /vehicle/cmd)

Usage:
    ros2 launch mid_level_controller mid_level_controller.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    pkg_share = get_package_share_directory('mid_level_controller')
    default_config = os.path.join(pkg_share, 'config', 'vehicle_constraints.yaml')

    # ---- Launch arguments ----
    config_file_arg = DeclareLaunchArgument(
        'config_file', default_value=default_config,
        description='Path to vehicle constraints YAML config'
    )

    # ---- Teleop node ----
    teleop_node = Node(
        package='mid_level_controller',
        executable='teleop_keyboard_node',
        name='teleop_keyboard_node',
        output='screen',
        prefix='xterm -e',
        parameters=[{
            'output_topic': '/teleop/raw_cmd',
            'publish_rate': 10.0,
            'velocity_step': 0.1,
            'heading_step': 5.0,
            'max_velocity': 2.0,
            'max_heading': 35.0,
        }],
    )

    # ---- Constraints node ----
    constraints_node = Node(
        package='mid_level_controller',
        executable='nonholonomic_constraints_node',
        name='nonholonomic_constraints_node',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
    )

    return LaunchDescription([
        config_file_arg,
        teleop_node,
        constraints_node,
    ])
