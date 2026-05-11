"""
Launch file for the low-level controller node.

Usage:
  ros2 launch low_level_controller low_level_controller.launch.py

Override parameters:
  ros2 launch low_level_controller low_level_controller.launch.py serial_port:=/dev/ttyUSB0 baud_rate:=9600
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # Declare launch arguments (overridable from CLI)
    serial_port_arg = DeclareLaunchArgument(
        'serial_port', default_value='/dev/ttyACM0',
        description='Serial port for Arduino connection'
    )
    baud_rate_arg = DeclareLaunchArgument(
        'baud_rate', default_value='115200',
        description='Serial baud rate'
    )
    max_velocity_arg = DeclareLaunchArgument(
        'max_velocity', default_value='0.25',
        description='Maximum velocity in m/s (physical max ~0.25 m/s)'
    )
    wheel_radius_arg = DeclareLaunchArgument(
        'wheel_radius', default_value='0.033',
        description='Wheel radius in meters (33 mm wheel)'
    )
    servo_center_arg = DeclareLaunchArgument(
        'servo_center', default_value='82',
        description='Servo center angle (straight)'
    )
    servo_min_arg = DeclareLaunchArgument(
        'servo_min', default_value='37',
        description='Servo minimum angle'
    )
    servo_max_arg = DeclareLaunchArgument(
        'servo_max', default_value='127',
        description='Servo maximum angle'
    )
    max_steering_arg = DeclareLaunchArgument(
        'max_steering_angle', default_value='45.0',
        description='Maximum steering angle in degrees (+/-)'
    )
    watchdog_arg = DeclareLaunchArgument(
        'watchdog_timeout', default_value='0.5',
        description='Watchdog timeout in seconds'
    )
    cmd_topic_arg = DeclareLaunchArgument(
        'cmd_topic', default_value='/vehicle/cmd',
        description='Topic name for velocity/heading commands'
    )
    feedback_topic_arg = DeclareLaunchArgument(
        'feedback_topic', default_value='/vehicle/feedback',
        description='Topic name for encoder feedback'
    )
    use_pi_mode_arg = DeclareLaunchArgument(
        'use_pi_mode', default_value='true',
        description='Set to false to run the node in open-loop (PWM) mode'
    )

    # Node
    low_level_node = Node(
        package='low_level_controller',
        executable='low_level_controller_node',
        name='low_level_controller_node',
        output='screen',
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
            'baud_rate': LaunchConfiguration('baud_rate'),
            'max_velocity': LaunchConfiguration('max_velocity'),
            'wheel_radius': LaunchConfiguration('wheel_radius'),
            'servo_center': LaunchConfiguration('servo_center'),
            'servo_min': LaunchConfiguration('servo_min'),
            'servo_max': LaunchConfiguration('servo_max'),
            'max_steering_angle': LaunchConfiguration('max_steering_angle'),
            'watchdog_timeout': LaunchConfiguration('watchdog_timeout'),
            'cmd_topic': LaunchConfiguration('cmd_topic'),
            'feedback_topic': LaunchConfiguration('feedback_topic'),
            'use_pi_mode': LaunchConfiguration('use_pi_mode'),
        }],
    )

    return LaunchDescription([
        serial_port_arg,
        baud_rate_arg,
        max_velocity_arg,
        wheel_radius_arg,
        servo_center_arg,
        servo_min_arg,
        servo_max_arg,
        max_steering_arg,
        watchdog_arg,
        cmd_topic_arg,
        feedback_topic_arg,
        use_pi_mode_arg,
        low_level_node,
    ])
