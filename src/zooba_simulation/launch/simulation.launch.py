"""
Launch file for the Gazebo simulation with the bridge node.

Launches:
    - gazebo_ackermann_steering_vehicle vehicle.launch.py (Gazebo + vehicle model)
    - sim_bridge_node (VehicleCmd → Float64 conversion)

Usage:
    ros2 launch zooba_simulation simulation.launch.py
    ros2 launch zooba_simulation simulation.launch.py world:=/path/to/world.sdf
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    # ---- Launch arguments ----
    world_arg = DeclareLaunchArgument(
        'world', default_value='empty.sdf',
        description='Gazebo world file'
    )
    x_arg = DeclareLaunchArgument('x', default_value='0.0',
                                  description='Initial X position')
    y_arg = DeclareLaunchArgument('y', default_value='0.0',
                                  description='Initial Y position')
    z_arg = DeclareLaunchArgument('z', default_value='0.1',
                                  description='Initial Z position')
    roll_arg = DeclareLaunchArgument('R', default_value='0.0',
                                     description='Initial Roll')
    pitch_arg = DeclareLaunchArgument('P', default_value='0.0',
                                      description='Initial Pitch')
    yaw_arg = DeclareLaunchArgument('Y', default_value='0.0',
                                    description='Initial Yaw')

    # ---- Include the upstream vehicle launch ----
    gazebo_pkg = get_package_share_directory('gazebo_ackermann_steering_vehicle')
    vehicle_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg, 'launch', 'vehicle.launch.py')
        ),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'x': LaunchConfiguration('x'),
            'y': LaunchConfiguration('y'),
            'z': LaunchConfiguration('z'),
            'R': LaunchConfiguration('R'),
            'P': LaunchConfiguration('P'),
            'Y': LaunchConfiguration('Y'),
        }.items()
    )

    # ---- Simulation bridge node ----
    sim_bridge_node = Node(
        package='zooba_simulation',
        executable='sim_bridge_node',
        name='sim_bridge_node',
        output='screen',
        parameters=[{
            'input_topic': '/vehicle/cmd',
            'steering_topic': '/steering_angle',
            'velocity_topic': '/velocity',
        }],
    )

    return LaunchDescription([
        world_arg,
        x_arg,
        y_arg,
        z_arg,
        roll_arg,
        pitch_arg,
        yaw_arg,
        vehicle_launch,
        sim_bridge_node,
    ])
