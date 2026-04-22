"""
Launch file for teleop keyboard node only.

Usage:
    ros2 launch mid_level_controller teleop.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    output_topic_arg = DeclareLaunchArgument(
        'output_topic', default_value='/teleop/raw_cmd',
        description='Topic to publish raw teleop commands'
    )
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate', default_value='10.0',
        description='Publishing rate in Hz'
    )
    velocity_step_arg = DeclareLaunchArgument(
        'velocity_step', default_value='0.1',
        description='Velocity increment per key press (m/s)'
    )
    heading_step_arg = DeclareLaunchArgument(
        'heading_step', default_value='5.0',
        description='Heading increment per key press (degrees)'
    )

    teleop_node = Node(
        package='mid_level_controller',
        executable='teleop_keyboard_node',
        name='teleop_keyboard_node',
        output='screen',
        parameters=[{
            'output_topic': LaunchConfiguration('output_topic'),
            'publish_rate': LaunchConfiguration('publish_rate'),
            'velocity_step': LaunchConfiguration('velocity_step'),
            'heading_step': LaunchConfiguration('heading_step'),
        }],
    )

    return LaunchDescription([
        output_topic_arg,
        publish_rate_arg,
        velocity_step_arg,
        heading_step_arg,
        teleop_node,
    ])
