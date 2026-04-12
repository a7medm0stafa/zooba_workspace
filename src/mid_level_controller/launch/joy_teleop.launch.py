"""
Launch file for joystick teleoperation and mid-level constraints.

Launches:
    - joy_node (standard block reading from /dev/input/js0 -> /joy)
    - teleop_joy_node (/joy -> /teleop/raw_cmd)
    - nonholonomic_constraints_node (/teleop/raw_cmd -> /vehicle/cmd)

Usage:
    ros2 launch mid_level_controller joy_teleop.launch.py
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

    output_topic_arg = DeclareLaunchArgument(
        'output_topic', default_value='/teleop/raw_cmd',
        description='Topic to publish raw teleop commands'
    )

    # ---- Standard ROS 2 Joy Node ----
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[{
            'deadzone': 0.05,
            'autorepeat_rate': 20.0,
        }]
    )

    # ---- Joy Teleop Node ----
    teleop_joy_node = Node(
        package='mid_level_controller',
        executable='teleop_joy_node',
        name='teleop_joy_node',
        output='screen',
        parameters=[{
            'output_topic': LaunchConfiguration('output_topic'),
            'max_velocity': 6.0,
            'max_heading': 45.0,
            'axis_steering': 0,
            'axis_forward': 5, # R2/RT
            'axis_reverse': 2, # L2/LT
            'button_estop': 0, # X/A button
            'button_unestop': 1 # Circle/B Button
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
        output_topic_arg,
        joy_node,
        teleop_joy_node,
        constraints_node,
    ])
