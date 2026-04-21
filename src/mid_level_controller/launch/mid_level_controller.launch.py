"""
Launch file for the full mid-level controller stack.

Launches:
    - teleop_keyboard_node or joy_node+teleop_joy_node (depending on teleop_type)
    - nonholonomic_constraints_node (/teleop/raw_cmd → /vehicle/cmd)

Usage:
    ros2 launch mid_level_controller mid_level_controller.launch.py teleop_type:=joy
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():

    pkg_share = get_package_share_directory('mid_level_controller')
    default_config = os.path.join(pkg_share, 'config', 'vehicle_constraints.yaml')

    # ---- Launch arguments ----
    config_file_arg = DeclareLaunchArgument(
        'config_file', default_value=default_config,
        description='Path to vehicle constraints YAML config'
    )
    
    teleop_type_arg = DeclareLaunchArgument(
        'teleop_type', default_value='keyboard',
        description='Type of teleop to run (keyboard or joy)'
    )

    # ---- Conditionally Launch Keyboard Teleop node ----
    teleop_keyboard_node = Node(
        package='mid_level_controller',
        executable='teleop_keyboard_node',
        name='teleop_keyboard_node',
        output='screen',
        prefix='xterm -e',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('teleop_type'), "' == 'keyboard'"])),
        parameters=[{
            'output_topic': '/teleop/raw_cmd',
            'publish_rate': 10.0,
            'velocity_step': 0.05,   # smaller step suits low-speed vehicle
            'heading_step': 5.0,
            'max_velocity': 0.249,   # physical max
            'max_heading': 45.0,
        }],
    )

    # ---- Conditionally Launch Joy Teleop node ----
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('teleop_type'), "' == 'joy'"])),
        parameters=[{
            'deadzone': 0.05,
            'autorepeat_rate': 20.0,
        }]
    )

    teleop_joy_node = Node(
        package='mid_level_controller',
        executable='teleop_joy_node',
        name='teleop_joy_node',
        output='screen',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('teleop_type'), "' == 'joy'"])),
        parameters=[{
            'output_topic': '/teleop/raw_cmd',
            'max_velocity': 0.249,   # physical max
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
        teleop_type_arg,
        teleop_keyboard_node,
        joy_node,
        teleop_joy_node,
        constraints_node,
    ])
