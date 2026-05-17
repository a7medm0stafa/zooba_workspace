"""Launch file for closed-loop control on hardware (Raspberry Pi + Arduino).
==========================================================================
FILE: high_level_controller/launch/closed_loop_hw.launch.py

WHAT THIS LAUNCHES:
    1. Low-level controller node (serial to Arduino — PI speed + Arduino EKF)
    2. Speed control node (bypass mode — forwards desired speed to Arduino PI)
    3. Lateral control node (Stanley controller → steering angle)
    4. Control merger node (speed + lateral → /teleop/raw_cmd)
    5. Non-holonomic constraints node (/teleop/raw_cmd → /vehicle/cmd)

NOTE: The Arduino runs a 4-state EKF internally and sends x,y,theta,v
      as part of its feedback. The LLC node parses this and publishes
      VehicleState on /vehicle/state. No Pi-side EKF is needed.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():

    # ---- Launch arguments ----
    serial_port_arg = DeclareLaunchArgument(
        'serial_port', default_value='/dev/ttyACM0',
        description='Arduino serial port'
    )
    desired_speed_arg = DeclareLaunchArgument(
        'desired_speed', default_value='0.0',
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
    control_config = os.path.join(
        get_package_share_directory('mid_level_controller'),
        'config', 'controller_params.yaml'
    )
    mid_pkg = get_package_share_directory('mid_level_controller')
    constraints_config = os.path.join(mid_pkg, 'config', 'vehicle_params.yaml')
    hlc_pkg = get_package_share_directory('high_level_controller')
    planner_config = os.path.join(hlc_pkg, 'config', 'path_planner_config.yaml')

    # ---- Low-level controller node (Arduino PI + EKF, publishes /vehicle/state) ----
    low_level_node = Node(
        package='low_level_controller',
        executable='low_level_controller_node',
        name='low_level_controller_node',
        output='screen',
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
            'baud_rate': 115200,
            'max_velocity': 0.25,
            'wheel_radius': 0.033,
            'servo_center': 84,          # calibrated for rightward drift
            'servo_min': 37,
            'servo_max': 127,
            'max_steering_angle': 45.0,
            'cmd_topic': '/vehicle/cmd',
            'feedback_topic': '/vehicle/feedback',
            'imu_topic': '/vehicle/imu',
            'state_topic': '/vehicle/state',
            'use_pi_mode': True,
            'gear_ratio': 124.333,
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
        serial_port_arg,
        desired_speed_arg,
        desired_y_arg,
        track_arg,
        use_planner_arg,
        # Hardware (LLC publishes /vehicle/state from Arduino EKF)
        low_level_node,
        # Path planner (conditional)
        path_planner_node,
        # Controllers
        speed_control_node,
        lateral_control_node,
        merger_node,
        # Constraints
        constraints_node,
    ])
