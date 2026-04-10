"""
Full simulation stack launch file.

Launches everything needed to teleop the vehicle in Gazebo:
    - Gazebo simulation with Ackermann vehicle model
    - Simulation bridge node (VehicleCmd → Gazebo topics)
    - Teleop keyboard node (keyboard → raw commands)
    - Non-holonomic constraints node (raw → constrained commands)

Usage:
    ros2 launch zooba_simulation full_sim.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    # ---- Launch arguments ----
    world_arg = DeclareLaunchArgument(
        'world', default_value='empty.sdf',
        description='Gazebo world file'
    )

    # ---- Include simulation launch (Gazebo + bridge) ----
    sim_pkg = get_package_share_directory('zooba_simulation')
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sim_pkg, 'launch', 'simulation.launch.py')
        ),
        launch_arguments={
            'world': LaunchConfiguration('world'),
        }.items()
    )

    # ---- Include mid-level controller launch (teleop + constraints) ----
    mid_pkg = get_package_share_directory('mid_level_controller')
    mid_level_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(mid_pkg, 'launch', 'mid_level_controller.launch.py')
        )
    )

    return LaunchDescription([
        world_arg,
        simulation_launch,
        mid_level_launch,
    ])
