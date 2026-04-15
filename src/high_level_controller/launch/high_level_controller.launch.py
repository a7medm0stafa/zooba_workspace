"""
Launch file for the full autonomous driving stack (traffic-light reactive).

Launches:
    1. traffic_light_controller_node  (high-level — this package)
    2. nonholonomic_constraints_node  (mid-level — from mid_level_controller)
    3. low_level_controller_node      (low-level — from low_level_controller)

Optionally also launches the traffic light detector (perception).

Usage:
    # Full stack (perception + control):
    ros2 launch high_level_controller high_level_controller.launch.py with_perception:=true

    # Control only (detector running separately):
    ros2 launch high_level_controller high_level_controller.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():

    hlc_share = get_package_share_directory('high_level_controller')
    hlc_config = os.path.join(hlc_share, 'config', 'high_level_controller.yaml')

    mlc_share = get_package_share_directory('mid_level_controller')
    mlc_config = os.path.join(mlc_share, 'config', 'vehicle_constraints.yaml')

    # ---- Launch arguments ----
    hlc_config_arg = DeclareLaunchArgument(
        'hlc_config', default_value=hlc_config,
        description='Path to high-level controller YAML config'
    )

    mlc_config_arg = DeclareLaunchArgument(
        'mlc_config', default_value=mlc_config,
        description='Path to mid-level controller (constraints) YAML config'
    )

    with_perception_arg = DeclareLaunchArgument(
        'with_perception', default_value='false',
        description='Also launch the traffic light detector node'
    )

    # ---- 1. Traffic Light Controller (High-Level) ----
    traffic_light_controller = Node(
        package='high_level_controller',
        executable='traffic_light_controller_node',
        name='traffic_light_controller_node',
        output='screen',
        parameters=[LaunchConfiguration('hlc_config')],
    )

    # ---- 2. Non-Holonomic Constraints (Mid-Level) ----
    constraints_node = Node(
        package='mid_level_controller',
        executable='nonholonomic_constraints_node',
        name='nonholonomic_constraints_node',
        output='screen',
        parameters=[LaunchConfiguration('mlc_config')],
    )

    # ---- 3. Low-Level Controller ----
    low_level_node = Node(
        package='low_level_controller',
        executable='low_level_controller_node',
        name='low_level_controller_node',
        output='screen',
    )

    # ---- 4. Traffic Light Detector (optional) ----
    traffic_light_detector = Node(
        package='perception',
        executable='traffic_light_detector_node',
        name='traffic_light_detector_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('with_perception')),
    )

    return LaunchDescription([
        hlc_config_arg,
        mlc_config_arg,
        with_perception_arg,
        traffic_light_controller,
        constraints_node,
        low_level_node,
        traffic_light_detector,
    ])
