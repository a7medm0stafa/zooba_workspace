"""
Localization Launch File — Ground Truth Mode (Simulation Only)
================================================================
Launches the ground truth node that reads the Gazebo model's
world-frame pose. Used for controller debugging and validation.

Also launches the ros_gz_bridge for the model pose topic.

Usage:
    ros2 launch localization ground_truth.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ---- Arguments ----
    model_name_arg = DeclareLaunchArgument(
        'model_name', default_value='ackermann_steering_vehicle',
        description='Gazebo model name (used to construct pose topic)')
    state_topic_arg = DeclareLaunchArgument(
        'state_topic', default_value='/vehicle/state',
        description='Output vehicle state topic (VehicleState)')
    wheel_radius_arg = DeclareLaunchArgument(
        'wheel_radius', default_value='0.033',
        description='Wheel radius [m]')
    wheelbase_arg = DeclareLaunchArgument(
        'wheelbase', default_value='0.22',
        description='Wheelbase [m]')

    model_name = LaunchConfiguration('model_name')

    # ---- ROS-Gazebo Pose Bridge ----
    # Bridges /model/<name>/pose from Gazebo (gz.msgs.Pose) to ROS2 (PoseStamped)
    pose_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='pose_bridge',
        output='screen',
        arguments=[
            ['/model/', model_name, '/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose'],
        ],
    )

    # ---- Ground Truth Node ----
    ground_truth_node = Node(
        package='localization',
        executable='ground_truth_node',
        name='ground_truth_node',
        output='screen',
        parameters=[{
            'pose_topic':    ['/model/', model_name, '/pose'],
            'state_topic':   LaunchConfiguration('state_topic'),
            'publish_rate':  20.0,
            'wheel_radius':  LaunchConfiguration('wheel_radius'),
            'wheelbase':     LaunchConfiguration('wheelbase'),
        }],
    )

    return LaunchDescription([
        model_name_arg,
        state_topic_arg,
        wheel_radius_arg,
        wheelbase_arg,
        pose_bridge,
        ground_truth_node,
    ])
