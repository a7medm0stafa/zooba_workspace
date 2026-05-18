"""
EKF Localization Launch File
==============================
FILE:    localization/launch/ekf_localization.launch.py
STATUS:  NOT NEEDED FOR HARDWARE ANYMORE

MODES:
    Hardware:   ros2 launch localization ekf_localization.launch.py
    Simulation: ros2 launch localization ekf_localization.launch.py source:=simulation
    Custom pose: ros2 launch localization ekf_localization.launch.py initial_x:=1.0 initial_yaw:=90.0
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    pkg_dir = get_package_share_directory('localization')
    default_config = os.path.join(pkg_dir, 'config', 'ekf_localization.yaml')

    # ---- Arguments ----
    source_arg = DeclareLaunchArgument(
        'source', default_value='hardware',
        description='Sensor source: "hardware" or "simulation"')
    feedback_topic_arg = DeclareLaunchArgument(
        'feedback_topic', default_value='/vehicle/feedback',
        description='Encoder feedback topic (VehicleFeedback)')
    imu_topic_arg = DeclareLaunchArgument(
        'imu_topic', default_value='/vehicle/imu',
        description='IMU data topic (ImuData)')
    state_topic_arg = DeclareLaunchArgument(
        'state_topic', default_value='/vehicle/state',
        description='Output vehicle state topic (VehicleState)')
    initial_x_arg = DeclareLaunchArgument(
        'initial_x', default_value='0.0',
        description='Initial X position [m]')
    initial_y_arg = DeclareLaunchArgument(
        'initial_y', default_value='0.0',
        description='Initial Y position [m]')
    initial_yaw_arg = DeclareLaunchArgument(
        'initial_yaw', default_value='0.0',
        description='Initial heading [degrees]')

    # ---- EKF Node ----
    ekf_node = Node(
        package='localization',
        executable='ekf_localization_node',
        name='ekf_localization_node',
        output='screen',
        parameters=[
            default_config,
            {
                'source':         LaunchConfiguration('source'),
                'feedback_topic': LaunchConfiguration('feedback_topic'),
                'imu_topic':      LaunchConfiguration('imu_topic'),
                'state_topic':    LaunchConfiguration('state_topic'),
                'initial_x':      LaunchConfiguration('initial_x'),
                'initial_y':      LaunchConfiguration('initial_y'),
                'initial_yaw':    LaunchConfiguration('initial_yaw'),
            },
        ],
    )

    return LaunchDescription([
        source_arg,
        feedback_topic_arg,
        imu_topic_arg,
        state_topic_arg,
        initial_x_arg,
        initial_y_arg,
        initial_yaw_arg,
        ekf_node,
    ])
