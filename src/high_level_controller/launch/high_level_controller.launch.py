"""
Launch file for the full autonomous driving stack.

Launches the complete pipeline:
    1. camera_publisher_node       (perception — single camera owner)
    2. sign_detection_node         (perception — subscribes to camera topic)
    3. traffic_light_detector_node (perception — subscribes to camera topic)
    4. traffic_light_controller_node (high-level — publishes to /teleop/auto_cmd)
    5. nonholonomic_constraints_node (mid-level — /teleop/raw_cmd → /vehicle/cmd)
    6. low_level_controller_node   (low-level — /vehicle/cmd → hardware)

Usage:
    # Full stack:
    ros2 launch high_level_controller high_level_controller.launch.py

    # Without perception (e.g. for testing controllers only):
    ros2 launch high_level_controller high_level_controller.launch.py with_perception:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():

    hlc_share = get_package_share_directory('high_level_controller')
    hlc_config = os.path.join(hlc_share, 'config', 'high_level_controller.yaml')

    mlc_share = get_package_share_directory('mid_level_controller')
    mlc_config = os.path.join(mlc_share, 'config', 'vehicle_params.yaml')

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
        'with_perception', default_value='true',
        description='Launch camera + perception nodes (set false for controller-only testing)'
    )

    show_dash_arg = DeclareLaunchArgument(
        'show_dash', default_value='false',
        description='Launch dashboard node'
    )

    show_traffic_arg = DeclareLaunchArgument(
        'show_traffic', default_value='false',
        description='Show traffic light debug display'
    )

    show_sign_arg = DeclareLaunchArgument(
        'show_sign', default_value='false',
        description='Show sign detection debug display'
    )

    show_both_arg = DeclareLaunchArgument(
        'show_both', default_value='false',
        description='Show both traffic light and sign detection debug displays'
    )

    # ---- 1. Camera Publisher (Perception) ----
    camera_publisher = Node(
        package='perception',
        executable='camera_publisher',
        name='camera_publisher_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('with_perception')),
        parameters=[{
            'camera_id': 0,
            'frame_width': 640,
            'frame_height': 480,
            'fps': 20.0,
            'flip_code': -1,          # 180° rotation for inverted mount
            'output_topic': '/camera/image_raw',
        }],
    )

    # ---- 2. Sign Detection (Perception) ----
    sign_detection = Node(
        package='perception',
        executable='sign_detection_node',
        name='sign_detection_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('with_perception')),
        parameters=[{
            'camera_topic': '/camera/image_raw',
            'output_topic': '/sign/command',
            'show_gui': PythonExpression(["'true' if ('", LaunchConfiguration('show_sign'), "'.lower() == 'true' or '", LaunchConfiguration('show_both'), "'.lower() == 'true') else 'false'"]),
        }],
    )

    # ---- 3. Traffic Light Detector (Perception) ----
    traffic_light_detector = Node(
        package='perception',
        executable='traffic_light_detector_node',
        name='traffic_light_detector_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('with_perception')),
        parameters=[{
            'camera_topic': '/camera/image_raw',
            'show_debug_display': PythonExpression(["'true' if ('", LaunchConfiguration('show_traffic'), "'.lower() == 'true' or '", LaunchConfiguration('show_both'), "'.lower() == 'true') else 'false'"]),
        }],
    )

    # ---- 4. Traffic Light Controller (High-Level) ----
    traffic_light_controller = Node(
        package='high_level_controller',
        executable='traffic_light_controller_node',
        name='traffic_light_controller_node',
        output='screen',
        parameters=[LaunchConfiguration('hlc_config')],
    )

    # ---- 5. Non-Holonomic Constraints (Mid-Level) ----
    constraints_node = Node(
        package='mid_level_controller',
        executable='nonholonomic_constraints_node',
        name='nonholonomic_constraints_node',
        output='screen',
        parameters=[LaunchConfiguration('mlc_config')],
    )

    # ---- 6. Low-Level Controller ----
    low_level_node = Node(
        package='low_level_controller',
        executable='low_level_controller_node',
        name='low_level_controller_node',
        output='screen',
    )

    # ---- 7. Dashboard HUD (optional, when GUI is enabled) ----
    dashboard = Node(
        package='perception',
        executable='dashboard_node',
        name='dashboard_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('show_dash')),
        parameters=[{
            'camera_topic': '/camera/image_raw',
            'window_width': 800,
            'window_height': 480,
        }],
    )

    return LaunchDescription([
        hlc_config_arg,
        mlc_config_arg,
        with_perception_arg,
        show_dash_arg,
        show_traffic_arg,
        show_sign_arg,
        show_both_arg,
        # Perception
        camera_publisher,
        sign_detection,
        traffic_light_detector,
        # High-level
        traffic_light_controller,
        # Mid-level
        constraints_node,
        # Low-level
        low_level_node,
        # Dashboard
        dashboard,
    ])
