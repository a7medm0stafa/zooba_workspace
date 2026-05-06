"""
Launch file for closed-loop control on hardware (Raspberry Pi + Arduino).
==========================================================================
FILE: high_level_controller/launch/closed_loop_hw.launch.py

WHAT THIS LAUNCHES:
    1. Low-level controller node (serial to Arduino — PI speed control on Arduino)
    2. EKF Localization node OR dead-reckoning odometry (toggle via use_ekf arg)
    3. Speed control node (bypass mode — forwards desired speed to Arduino PI)
    4. Lateral control node (Stanley controller → steering angle)
    5. Control merger node (speed + lateral → /teleop/raw_cmd)
    6. Non-holonomic constraints node (/teleop/raw_cmd → /vehicle/cmd)

CONFIGURATION:
    All EKF tuning parameters are loaded from:
        localization/config/ekf_localization.yaml

    All speed + lateral control parameters are loaded from:
        mid_level_controller/config/closed_loop_control.yaml

USAGE:
    # Default (EKF + Arduino PI):
    ros2 launch high_level_controller closed_loop_hw.launch.py

    # Use dead-reckoning instead of EKF:
    ros2 launch high_level_controller closed_loop_hw.launch.py use_ekf:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node


def generate_launch_description():

    # ---- Launch arguments ----
    use_ekf_arg = DeclareLaunchArgument(
        'use_ekf', default_value='true',
        description='Use EKF localization instead of dead-reckoning'
    )
    serial_port_arg = DeclareLaunchArgument(
        'serial_port', default_value='/dev/ttyACM0',
        description='Arduino serial port'
    )
    desired_speed_arg = DeclareLaunchArgument(
        'desired_speed', default_value='0.15',
        description='Goal speed [m/s]'
    )
    desired_y_arg = DeclareLaunchArgument(
        'desired_y', default_value='0.0',
        description='Goal lateral position [m]'
    )
    track_arg = DeclareLaunchArgument(
        'track', default_value='track_1',
        description='Track for path planner (track_1, track_2, track_3)'
    )
    use_planner_arg = DeclareLaunchArgument(
        'use_planner', default_value='false',
        description='Enable path planner (overrides desired_speed/desired_y)'
    )

    # ---- Config file paths ----
    ekf_config = os.path.join(
        get_package_share_directory('localization'),
        'config', 'ekf_localization.yaml'
    )
    control_config = os.path.join(
        get_package_share_directory('mid_level_controller'),
        'config', 'closed_loop_control.yaml'
    )
    mid_pkg = get_package_share_directory('mid_level_controller')
    constraints_config = os.path.join(mid_pkg, 'config', 'vehicle_constraints.yaml')
    hlc_pkg = get_package_share_directory('high_level_controller')
    planner_config = os.path.join(hlc_pkg, 'config', 'path_planner_config.yaml')

    # ---- Low-level controller node (Arduino PI for speed) ----
    low_level_node = Node(
        package='low_level_controller',
        executable='low_level_controller_node',
        name='low_level_controller_node',
        output='screen',
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
            'baud_rate': 115200,
            'max_velocity': 0.25,       # max ~ 71.95 RPM × 2π×0.033/60
            'wheel_radius': 0.033,       # 33 mm wheel
            'servo_center': 85,
            'servo_min': 40,
            'servo_max': 130,
            'max_steering_angle': 45.0,
            'cmd_topic': '/vehicle/cmd',
            'feedback_topic': '/vehicle/feedback',
            'imu_topic': '/vehicle/imu',
            'use_pi_mode': True,         # Arduino handles PI speed control
            'gear_ratio': 124.333,
        }],
    )

    # ---- EKF Localization node ----
    ekf_node = Node(
        package='localization',
        executable='ekf_localization_node',
        name='ekf_localization_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_ekf')),
        parameters=[
            ekf_config,
            {
                'source': 'hardware',
                'wheelbase': 0.22,
                'wheel_radius': 0.033,
                'encoder_cpr': 5471,
                'feedback_topic': '/vehicle/feedback',
                'imu_topic': '/vehicle/imu',
                'state_topic': '/vehicle/state',
                'publish_rate': 50.0,
            }
        ],
    )

    # ---- Classic Odometry node (dead-reckoning fallback) ----
    odometry_node = Node(
        package='localization',
        executable='odometry_node',
        name='odometry_node',
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('use_ekf')),
        parameters=[{
            'source': 'hardware',
            'wheelbase': 0.22,
            'wheel_radius': 0.033,
            'encoder_cpr': 5471,
            'use_imu_heading': True,
            'feedback_topic': '/vehicle/feedback',
            'imu_topic': '/vehicle/imu',
            'state_topic': '/vehicle/state',
            'publish_rate': 20.0,
        }],
    )

    # ---- Speed control node (bypass — Arduino handles PI) ----
    speed_control_node = Node(
        package='mid_level_controller',
        executable='speed_control_node',
        name='speed_control_node',
        output='screen',
        parameters=[
            control_config,
            {
                'desired_speed': LaunchConfiguration('desired_speed'),
                'control_rate': 20.0,
                'state_topic': '/vehicle/state',
                'output_topic': '/teleop/speed_cmd',
                'bypass_pi': True,           # Arduino handles PI, ROS just forwards desired speed
            }
        ],
    )

    # ---- Lateral control node (Stanley) ----
    lateral_control_node = Node(
        package='mid_level_controller',
        executable='lateral_control_node',
        name='lateral_control_node',
        output='screen',
        parameters=[
            control_config,
            {
                'desired_y': LaunchConfiguration('desired_y'),
                'control_rate': 20.0,
                'invert_steering_output': False,
                'state_topic': '/vehicle/state',
                'output_topic': '/teleop/lateral_cmd',
            }
        ],
    )

    # ---- Control merger node ----
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

    # ---- Non-holonomic constraints node ----
    constraints_node = Node(
        package='mid_level_controller',
        executable='nonholonomic_constraints_node',
        name='nonholonomic_constraints_node',
        output='screen',
        parameters=[constraints_config],
    )

    # ---- Path planner node (optional — enabled via use_planner:=true) ----
    path_planner_node = Node(
        package='high_level_controller',
        executable='path_planner_node',
        name='path_planner_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_planner')),
        parameters=[
            planner_config,
            {
                'track_name':    LaunchConfiguration('track'),
                'state_topic':   '/vehicle/state',
                'start_delay':   10.0,
            }
        ],
    )

    return LaunchDescription([
        # Arguments
        use_ekf_arg,
        serial_port_arg,
        desired_speed_arg,
        desired_y_arg,
        track_arg,
        use_planner_arg,
        # Hardware
        low_level_node,
        ekf_node,
        odometry_node,
        # Path planner (conditional)
        path_planner_node,
        # Controllers
        speed_control_node,
        lateral_control_node,
        merger_node,
        # Constraints
        constraints_node,
    ])
