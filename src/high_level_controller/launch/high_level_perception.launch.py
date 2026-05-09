"""
Launch file for the full autonomous driving stack (Hardware).

Launches the complete closed-loop pipeline:
    Perception → HLC (setpoints) → MLC (PI + Stanley + merger) → Constraints → LLC

Pipeline:
    1. camera_publisher_node        (perception — camera owner)
    2. sign_detection_node          (perception — sign detection)
    3. traffic_light_detector_node  (perception — traffic light detection)
    4. perception_controller_node   (HLC — publishes /hlc/desired_speed & /hlc/desired_heading)
    5. odometry_node                (localization — dead-reckoning → /vehicle/state)
    6. speed_control_node           (MLC — PI speed controller)
    7. lateral_control_node         (MLC — Stanley lateral controller)
    8. control_merger_node          (MLC — merges speed + lateral → VehicleCmd)
    9. nonholonomic_constraints_node(MLC — rate limiting + clamping)
   10. low_level_controller_node    (LLC — hardware serial)

Usage:
    # Full stack:
    ros2 launch high_level_controller high_level_perception.launch.py

    # Without perception (controller-only testing):
    ros2 launch high_level_controller high_level_perception.launch.py with_perception:=false
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
    constraints_config = os.path.join(mlc_share, 'config', 'vehicle_constraints.yaml')

    # ---- Launch arguments ----
    hlc_config_arg = DeclareLaunchArgument(
        'hlc_config', default_value=hlc_config,
        description='Path to high-level controller YAML config'
    )

    mlc_config_arg = DeclareLaunchArgument(
        'mlc_config', default_value=constraints_config,
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

    # ---- 4. Perception Controller (HLC) ----
    # Publishes /hlc/desired_speed and /hlc/desired_heading (Float64).
    # Does NOT publish VehicleCmd — the MLC handles closed-loop control.
    perception_controller = Node(
        package='high_level_controller',
        executable='perception_controller_node',
        name='perception_controller_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('with_perception')),
        parameters=[LaunchConfiguration('hlc_config')],
    )

    # ---- 5. Low-Level Controller ----
    low_level_node = Node(
        package='low_level_controller',
        executable='low_level_controller_node',
        name='low_level_controller_node',
        output='screen',
    )

    # ---- 6. Odometry (Localization — IMU + Encoder Dead-Reckoning) ----
    odometry_node = Node(
        package='localization',
        executable='odometry_node',
        name='odometry_node',
        output='screen',
        parameters=[{
            'feedback_topic': '/vehicle/feedback',
            'imu_topic': '/vehicle/imu',
            'state_topic': '/vehicle/state',
            'publish_rate': 20.0,
            'wheel_radius': 0.033,
            'ticks_per_rev': 1968,
            'initial_x': 0.0,
            'initial_y': 0.0,
            'initial_yaw': 0.0,
        }],
    )

    # ---- 7. Speed Control Node (PI — bypass mode, Arduino handles PI) ----
    speed_control_node = Node(
        package='mid_level_controller',
        executable='speed_control_node',
        name='speed_control_node',
        output='screen',
        parameters=[{
            'desired_speed': 0.0,
            'control_rate': 20.0,
            'state_topic': '/vehicle/state',
            'output_topic': '/teleop/speed_cmd',
            'desired_speed_topic': '/hlc/desired_speed',
            'bypass_pi': True,   # Arduino handles PI speed control
        }],
    )

    # ---- 8. Lateral Control Node (Stanley) ----
    lateral_control_node = Node(
        package='mid_level_controller',
        executable='lateral_control_node',
        name='lateral_control_node',
        output='screen',
        parameters=[{
            'desired_heading': 0.0,
            'desired_y': 0.0,
            'k_heading': 1.5,
            'k_stanley': 2.5,
            'k_soft': 1.0,
            'k_d_heading': 0.3,
            'max_steering_angle': 35.0,
            'control_rate': 20.0,
            'invert_steering_output': False,
            'state_topic': '/vehicle/state',
            'output_topic': '/teleop/lateral_cmd',
            'desired_heading_topic': '/hlc/desired_heading',
        }],
    )

    # ---- 9. Control Merger Node ----
    merger_node = Node(
        package='mid_level_controller',
        executable='control_merger_node',
        name='control_merger_node',
        output='screen',
        parameters=[{
            'speed_topic': '/teleop/speed_cmd',
            'lateral_topic': '/teleop/lateral_cmd',
            'output_topic': '/teleop/raw_cmd',
            'publish_rate': 20.0,
        }],
    )

    # ---- 10. Non-Holonomic Constraints ----
    constraints_node = Node(
        package='mid_level_controller',
        executable='nonholonomic_constraints_node',
        name='nonholonomic_constraints_node',
        output='screen',
        parameters=[LaunchConfiguration('mlc_config')],
    )

    # ---- 11. Dashboard HUD (optional) ----
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
        # High-level (publishes setpoints — no VehicleCmd)
        perception_controller,
        # Low-level
        low_level_node,
        # Localization
        odometry_node,
        # Mid-level controllers
        speed_control_node,
        lateral_control_node,
        merger_node,
        # Constraints
        constraints_node,
        # Dashboard
        dashboard,
    ])
