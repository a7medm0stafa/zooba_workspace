"""
Launch file for closed-loop control in Gazebo simulation.

Launches:
    - Gazebo with empty world + Ackermann vehicle
    - Sim bridge (extended — publishes VehicleState + VehicleFeedback)
    - Speed control node (PI controller)
    - Lateral control node (Stanley controller)
    - Control merger node (speed + lateral → VehicleCmd)
    - Non-holonomic constraints node

Usage:
    # Default:
    ros2 launch zooba_simulation closed_loop_sim.launch.py

    # Custom initial pose and goals:
    ros2 launch zooba_simulation closed_loop_sim.launch.py \\
        x:=1.0 y:=0.0 Y:=0.0 \\
        desired_speed:=0.8 desired_y:=2.0

    # Custom PI gains:
    ros2 launch zooba_simulation closed_loop_sim.launch.py \\
        kp:=2.0 ki:=0.3 k_stanley:=3.0
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    # ---- Launch arguments: initial pose ----
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
                                    description='Initial Yaw (theta)')

    # ---- Launch arguments: control goals ----
    desired_speed_arg = DeclareLaunchArgument(
        'desired_speed', default_value='0.5',
        description='Goal speed [m/s]'
    )
    desired_y_arg = DeclareLaunchArgument(
        'desired_y', default_value='0.0',
        description='Goal lateral position (lane) [m]'
    )
    desired_heading_arg = DeclareLaunchArgument(
        'desired_heading', default_value='0.0',
        description='Goal heading [rad]'
    )

    # ---- Launch arguments: PI gains ----
    kp_arg = DeclareLaunchArgument('kp', default_value='1.0',
                                   description='PI proportional gain')
    ki_arg = DeclareLaunchArgument('ki', default_value='0.1',
                                   description='PI integral gain')

    # ---- Launch arguments: Stanley gain ----
    k_stanley_arg = DeclareLaunchArgument(
        'k_stanley', default_value='2.5',
        description='Stanley cross-track gain'
    )

    # ---- Include Gazebo vehicle launch ----
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

    # ---- Simulation bridge node (extended) ----
    sim_bridge_node = Node(
        package='zooba_simulation',
        executable='sim_bridge_node',
        name='sim_bridge_node',
        output='screen',
        parameters=[{
            'input_topic': '/vehicle/cmd',
            'steering_topic': '/steering_angle',
            'velocity_topic': '/velocity',
            'state_topic': '/vehicle/state',
            'feedback_topic': '/vehicle/feedback',
            'wheel_radius': 0.04,
            'wheelbase': 0.22,
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
            'max_velocity': 2.0,
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
            'max_steering_angle': 35.0,
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

    # # ---- Non-holonomic constraints node ----
    # mid_pkg = get_package_share_directory('mid_level_controller')
    # constraints_config = os.path.join(mid_pkg, 'config', 'vehicle_constraints.yaml')

    # constraints_node = Node(
    #     package='mid_level_controller',
    #     executable='nonholonomic_constraints_node',
    #     name='nonholonomic_constraints_node',
    #     output='screen',
    #     parameters=[constraints_config],
    # )

    return LaunchDescription([
        # Arguments
        world_arg,
        x_arg, y_arg, z_arg,
        roll_arg, pitch_arg, yaw_arg,
        desired_speed_arg, desired_y_arg, desired_heading_arg,
        kp_arg, ki_arg, k_stanley_arg,
        # Gazebo
        vehicle_launch,
        # Bridge
        sim_bridge_node,
        # Controllers
        speed_control_node,
        lateral_control_node,
        merger_node,
        # Constraints
        # constraints_node,
    ])
