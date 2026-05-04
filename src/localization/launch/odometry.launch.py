"""
Localization Launch File — Odometry Mode
==========================================
Launches the odometry node for dead-reckoning localization
using IMU heading + encoder distance. This is the mode used
on the real car.

Usage:
    ros2 launch localization odometry.launch.py
    ros2 launch localization odometry.launch.py initial_yaw:=90.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ---- Arguments ----
    feedback_topic_arg = DeclareLaunchArgument(
        'feedback_topic', default_value='/vehicle/feedback',
        description='Encoder feedback topic (VehicleFeedback)')
    imu_topic_arg = DeclareLaunchArgument(
        'imu_topic', default_value='/imu/data',
        description='IMU data topic (ImuData)')
    state_topic_arg = DeclareLaunchArgument(
        'state_topic', default_value='/vehicle/state',
        description='Output vehicle state topic (VehicleState)')
    wheel_radius_arg = DeclareLaunchArgument(
        'wheel_radius', default_value='0.04',
        description='Wheel radius [m]')
    ticks_per_rev_arg = DeclareLaunchArgument(
        'ticks_per_rev', default_value='1968',
        description='Encoder ticks per wheel revolution')
    initial_x_arg = DeclareLaunchArgument(
        'initial_x', default_value='0.0',
        description='Initial X position [m]')
    initial_y_arg = DeclareLaunchArgument(
        'initial_y', default_value='0.0',
        description='Initial Y position [m]')
    initial_yaw_arg = DeclareLaunchArgument(
        'initial_yaw', default_value='0.0',
        description='Initial heading [degrees]')

    # ---- Node ----
    odometry_node = Node(
        package='localization',
        executable='odometry_node',
        name='odometry_node',
        output='screen',
        parameters=[{
            'feedback_topic': LaunchConfiguration('feedback_topic'),
            'imu_topic':      LaunchConfiguration('imu_topic'),
            'state_topic':    LaunchConfiguration('state_topic'),
            'publish_rate':   20.0,
            'wheel_radius':   LaunchConfiguration('wheel_radius'),
            'ticks_per_rev':  LaunchConfiguration('ticks_per_rev'),
            'initial_x':      LaunchConfiguration('initial_x'),
            'initial_y':      LaunchConfiguration('initial_y'),
            'initial_yaw':    LaunchConfiguration('initial_yaw'),
        }],
    )

    return LaunchDescription([
        feedback_topic_arg,
        imu_topic_arg,
        state_topic_arg,
        wheel_radius_arg,
        ticks_per_rev_arg,
        initial_x_arg,
        initial_y_arg,
        initial_yaw_arg,
        odometry_node,
    ])
