"""
Launch file for closed-loop control on hardware (Raspberry Pi + Arduino).

Launches:
    - Low-level controller node (serial to Arduino — PI or open-loop)
    - Odometry node (encoder + IMU → VehicleState)
    - Speed control node (PI controller)
    - Lateral control node (Stanley controller)
    - Control merger node (speed + lateral → VehicleCmd)
    - Non-holonomic constraints node

Usage:
    # Default (PI mode on Arduino):
    ros2 launch low_level_controller closed_loop_hw.launch.py

    # Custom speed goal:
    ros2 launch low_level_controller closed_loop_hw.launch.py desired_speed:=0.8

    # Open-loop mode (skip Arduino PI):
    ros2 launch low_level_controller closed_loop_hw.launch.py use_pi_mode:=false
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
        description='Goal heading [rad]'
    )
    kp_arg = DeclareLaunchArgument('kp', default_value='1.0',
                                   description='PI proportional gain')
    ki_arg = DeclareLaunchArgument('ki', default_value='0.1',
                                   description='PI integral gain')
    k_stanley_arg = DeclareLaunchArgument(
        'k_stanley', default_value='2.5',
        description='Stanley cross-track gain'
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
            'max_velocity': 1.0,
            'wheel_radius': 0.033,
            'servo_center': 90,
            'servo_min': 45,
            'servo_max': 135,
            'max_steering_angle': 30.0,
            'cmd_topic': '/vehicle/cmd',
            'feedback_topic': '/vehicle/feedback',
            'imu_topic': '/vehicle/imu',
            'use_pi_mode': LaunchConfiguration('use_pi_mode'),
            'gear_ratio': 134.181,
        }],
    )

    # ---- Odometry node ----
    odometry_node = Node(
        package='mid_level_controller',
        executable='odometry_node',
        name='odometry_node',
        output='screen',
        parameters=[{
            'wheelbase': 0.22,
            'wheel_radius': 0.033,
            'encoder_cpr': 5904,
            'use_imu_heading': True,
            'source': 'hardware',
            'feedback_topic': '/vehicle/feedback',
            'imu_topic': '/vehicle/imu',
            'state_topic': '/vehicle/state',
            'publish_rate': 20.0,
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
            'max_velocity': 1.0,
            'control_rate': 20.0,
            'state_topic': '/vehicle/state',
            'output_topic': '/teleop/speed_cmd',
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
            'k_stanley': LaunchConfiguration('k_stanley'),
            'k_soft': 1.0,
            'max_steering_angle': 30.0,
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
        kp_arg, ki_arg, k_stanley_arg,
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
