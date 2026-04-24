"""
Launch file for closed-loop control on hardware (Raspberry Pi + Arduino).
==========================================================================
FILE: high_level_controller/launch/closed_loop_hw.launch.py
STATUS: MODIFIED — replaced dead-reckoning odometry with EKF localization
MODIFIED: 2026-04-24

CHANGES MADE:
    - REPLACED: mid_level_controller/odometry_node (dead-reckoning, yaw drift)
    - WITH:     localization/ekf_localization_node (EKF, gyro bias + ZUPT)
    - The EKF node runs at 50 Hz (was 20 Hz) for better estimation
    - All EKF tuning parameters are set inline below

WHAT THIS LAUNCHES:
    1. Low-level controller node (serial to Arduino — PI or open-loop)
    2. EKF Localization node (encoder + IMU → VehicleState, drift-corrected)
    3. Speed control node (PI controller → /teleop/speed_cmd)
    4. Lateral control node (Stanley controller → /teleop/lateral_cmd)
    5. Control merger node (speed + lateral → /teleop/raw_cmd)
    6. Non-holonomic constraints node (/teleop/raw_cmd → /vehicle/cmd)

SIGNAL FLOW:
    Arduino → /vehicle/feedback + /vehicle/imu
           → EKF Localization Node → /vehicle/state
           → Speed Control + Lateral Control → /teleop/speed_cmd + /teleop/lateral_cmd
           → Control Merger → /teleop/raw_cmd
           → Non-Holonomic Constraints → /vehicle/cmd
           → Low-Level Controller → Arduino serial

USAGE:
    # Default (PI mode on Arduino):
    ros2 launch high_level_controller closed_loop_hw.launch.py

    # Custom speed goal:
    ros2 launch high_level_controller closed_loop_hw.launch.py desired_speed:=0.8

    # Custom lateral target:
    ros2 launch high_level_controller closed_loop_hw.launch.py desired_y:=1.0

    # Open-loop mode (skip Arduino PI):
    ros2 launch high_level_controller closed_loop_hw.launch.py use_pi_mode:=false

ROLLBACK (revert to old dead-reckoning):
    Change the odometry_node definition below from:
        package='localization', executable='ekf_localization_node'
    to:
        package='mid_level_controller', executable='odometry_node'
    And remove the EKF-specific parameters.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ---- Launch arguments ----
    serial_port_arg = DeclareLaunchArgument(
        'serial_port', default_value='/dev/ttyACM0',
        description='Arduino serial port'
    )
    use_pi_mode_arg = DeclareLaunchArgument(
        'use_pi_mode', default_value='true',
        description='Use PI control mode on Arduino'
    )
    desired_speed_arg = DeclareLaunchArgument(
        'desired_speed', default_value='0.5',
        description='Goal speed [m/s]'
    )
    desired_y_arg = DeclareLaunchArgument(
        'desired_y', default_value='0.0',
        description='Goal lateral position [m]'
    )
    desired_heading_arg = DeclareLaunchArgument(
        'desired_heading', default_value='0.0',
        description='Goal heading [degrees]'
    )
    k_heading_arg = DeclareLaunchArgument(
        'k_heading', default_value='1.5',
        description='Heading proportional gain'
    )
    kp_arg = DeclareLaunchArgument('kp', default_value='1.0',
                                   description='PI proportional gain')
    ki_arg = DeclareLaunchArgument('ki', default_value='0.1',
                                   description='PI integral gain')
    k_stanley_arg = DeclareLaunchArgument(
        'k_stanley', default_value='2.5',
        description='Stanley cross-track gain'
    )
    k_d_heading_arg = DeclareLaunchArgument(
        'k_d_heading', default_value='0.3',
        description='Heading derivative damping gain'
    )

    # ---- Low-level controller node ----
    low_level_node = Node(
        package='low_level_controller',
        executable='low_level_controller_node',
        name='low_level_controller_node',
        output='screen',
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
            'baud_rate': 115200,
            'max_velocity': 0.249,       # max ~ 71.95 RPM × 2π×0.033/60
            'wheel_radius': 0.033,       # 33 mm wheel
            'servo_center': 90,
            'servo_min': 45,
            'servo_max': 135,
            'max_steering_angle': 45.0,
            'cmd_topic': '/vehicle/cmd',
            'feedback_topic': '/vehicle/feedback',
            'imu_topic': '/vehicle/imu',
            'use_pi_mode': LaunchConfiguration('use_pi_mode'),
            'gear_ratio': 124.333,       # 44.727 (internal) × 2.7798 (herringbone 45.45/16.35)
        }],
    )

    # ---- EKF Localization node (replaces dead-reckoning odometry) ----
    odometry_node = Node(
        package='localization',
        executable='ekf_localization_node',
        name='ekf_localization_node',
        output='screen',
        parameters=[{
            'source': 'hardware',
            'wheelbase': 0.22,
            'wheel_radius': 0.033,
            'encoder_cpr': 5471,
            'feedback_topic': '/vehicle/feedback',
            'imu_topic': '/vehicle/imu',
            'state_topic': '/vehicle/state',
            'publish_rate': 50.0,
            # EKF tuning (defaults are good starting points)
            'process_noise_x': 0.01,
            'process_noise_y': 0.01,
            'process_noise_yaw': 0.005,
            'process_noise_vel': 0.1,
            'process_noise_gyro_bias': 0.0001,
            'encoder_velocity_noise': 0.05,
            'gyro_rate_noise': 0.01,
            'imu_yaw_noise': 0.15,
            'zupt_velocity_threshold': 0.02,
            'zupt_noise': 0.001,
            'imu_settle_time': 2.5,
        }],
    )

    # ---- Speed control node (PI) ----
    speed_control_node = Node(
        package='mid_level_controller',
        executable='speed_control_node',
        name='speed_control_node',
        output='screen',
        parameters=[{
            'desired_speed': LaunchConfiguration('desired_speed'),
            'kp': LaunchConfiguration('kp'),
            'ki': LaunchConfiguration('ki'),
            'max_velocity': 0.21,       # physical max ~0.249 m/s
            'control_rate': 20.0,
            'state_topic': '/vehicle/state',
            'output_topic': '/teleop/speed_cmd',
            'bypass_pi': True,           # Hardware uses Arduino PI, bypass ROS PI
        }],
    )

    # ---- Lateral control node (Stanley) ----
    lateral_control_node = Node(
        package='mid_level_controller',
        executable='lateral_control_node',
        name='lateral_control_node',
        output='screen',
        parameters=[{
            'desired_y': LaunchConfiguration('desired_y'),
            'desired_heading': LaunchConfiguration('desired_heading'),
            'k_heading': LaunchConfiguration('k_heading'),
            'k_stanley': LaunchConfiguration('k_stanley'),
            'k_soft': 1.0,
            'k_d_heading': LaunchConfiguration('k_d_heading'),
            'max_steering_angle': 45.0,
            'control_rate': 20.0,
            'state_topic': '/vehicle/state',
            'output_topic': '/teleop/lateral_cmd',
        }],
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
    mid_pkg = get_package_share_directory('mid_level_controller')
    constraints_config = os.path.join(mid_pkg, 'config', 'vehicle_constraints.yaml')

    constraints_node = Node(
        package='mid_level_controller',
        executable='nonholonomic_constraints_node',
        name='nonholonomic_constraints_node',
        output='screen',
        parameters=[constraints_config],
    )

    return LaunchDescription([
        # Arguments
        serial_port_arg,
        use_pi_mode_arg,
        desired_speed_arg, desired_y_arg, desired_heading_arg,
        k_heading_arg, kp_arg, ki_arg, k_stanley_arg, k_d_heading_arg,
        # Hardware
        low_level_node,
        odometry_node,
        # Controllers
        speed_control_node,
        lateral_control_node,
        merger_node,
        # Constraints
        constraints_node,
    ])
