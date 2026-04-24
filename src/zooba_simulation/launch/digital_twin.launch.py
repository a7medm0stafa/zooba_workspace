"""
Digital Twin Launch — RViz2 Vehicle Visualization
===================================================
Launches:
    - rviz_vehicle_node   (subscribes to /vehicle/state, publishes markers)
    - rviz2               (opens preconfigured view)

Usage:
    # View the physical car in real time (run alongside closed_loop_hw):
    ros2 launch zooba_simulation digital_twin.launch.py

    # View the simulation in real time (run alongside closed_loop_sim):
    ros2 launch zooba_simulation digital_twin.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    pkg_share = get_package_share_directory('zooba_simulation')
    default_rviz = os.path.join(pkg_share, 'config', 'digital_twin.rviz')

    # ---- Arguments ----
    state_topic_arg = DeclareLaunchArgument(
        'state_topic', default_value='/vehicle/state',
        description='VehicleState topic to visualize'
    )

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config', default_value=default_rviz,
        description='Path to RViz2 config file'
    )

    # ---- RViz vehicle visualization node ----
    rviz_vehicle_node = Node(
        package='zooba_simulation',
        executable='rviz_vehicle_node',
        name='rviz_vehicle_node',
        output='screen',
        parameters=[{
            'state_topic': LaunchConfiguration('state_topic'),
            'marker_topic': '/vehicle/viz_markers',
            'path_topic': '/vehicle/path',
            'odom_topic': '/vehicle/odom_viz',
            'car_length': 0.30,
            'car_width': 0.20,
            'car_height': 0.10,
            'trail_max_points': 500,
            'publish_rate': 20.0,
        }],
    )

    # ---- RViz2 ----
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
    )

    return LaunchDescription([
        state_topic_arg,
        rviz_config_arg,
        rviz_vehicle_node,
        rviz2_node,
    ])
