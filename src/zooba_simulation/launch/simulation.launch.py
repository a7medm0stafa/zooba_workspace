"""
Launch file for the Gazebo simulation with the bridge node and open-loop control.

Launches:
    - gazebo_ackermann_steering_vehicle vehicle.launch.py (Gazebo + vehicle model)
    - sim_bridge_node (VehicleCmd → Float64 conversion)
    - open_loop_node  (constant VehicleCmd publisher)
    - nonholonomic_constraints_node (raw → constrained commands)

Usage:
    ros2 launch zooba_simulation simulation.launch.py
    ros2 launch zooba_simulation simulation.launch.py open_loop_velocity:=1.0 open_loop_heading:=10.0
    ros2 launch zooba_simulation simulation.launch.py open_loop_duration:=5.0
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

    # ---- Open-loop response arguments ----
    open_loop_velocity_arg = DeclareLaunchArgument(
        'open_loop_velocity', default_value='0.8',
        description='Open-loop velocity command [m/s]'
    )
    open_loop_heading_arg = DeclareLaunchArgument(
        'open_loop_heading', default_value='10.0',
        description='Open-loop steering heading [degrees]'
    )
    open_loop_rate_arg = DeclareLaunchArgument(
        'open_loop_rate', default_value='10.0',
        description='Open-loop publish rate [Hz]'
    )
    open_loop_duration_arg = DeclareLaunchArgument(
        'open_loop_duration', default_value='0.0',
        description='Open-loop duration in seconds (0 = infinite)'
    )

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

    # ---- Open-loop response node ----
    open_loop_node = Node(
        package='mid_level_controller',
        executable='open_loop_node',
        name='open_loop_node',
        output='screen',
        parameters=[{
            'velocity': LaunchConfiguration('open_loop_velocity'),
            'heading': LaunchConfiguration('open_loop_heading'),
            'publish_rate': LaunchConfiguration('open_loop_rate'),
            'duration': LaunchConfiguration('open_loop_duration'),
            'output_topic': '/teleop/raw_cmd',
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
        world_arg,
        x_arg,
        y_arg,
        z_arg,
        roll_arg,
        pitch_arg,
        yaw_arg,
        open_loop_velocity_arg,
        open_loop_heading_arg,
        open_loop_rate_arg,
        open_loop_duration_arg,
        vehicle_launch,
        sim_bridge_node,
        open_loop_node,
        constraints_node,
    ])

