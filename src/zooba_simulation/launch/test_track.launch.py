import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression

def generate_launch_description():
    # Argument to select the track
    track_arg = DeclareLaunchArgument(
        'track', default_value='track_1',
        description='Which track to load (e.g., track_1, track_2, track_3)'
    )

    # Paths
    vehicle_pkg = get_package_share_directory('gazebo_ackermann_steering_vehicle')
    zooba_sim_pkg = get_package_share_directory('zooba_simulation')

    # Construct the world path based on the track argument
    world_path = PathJoinSubstitution([
        zooba_sim_pkg,
        'worlds',
        [LaunchConfiguration('track'), '.world']
    ])

    # Include the vehicle launch file, overriding the world and initial pose
    vehicle_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(vehicle_pkg, 'launch', 'vehicle.launch.py')
        ),
        launch_arguments={
            'world': world_path,
            'x': '0.0',
            'y': PythonExpression(["'1.75' if '", LaunchConfiguration('track'), "' == 'track_3' else '0.0'"]),
            'z': '0.1',
            'Y': '0.0', # theta = 0 (Yaw)
        }.items()
    )

    return LaunchDescription([
        track_arg,
        vehicle_launch
    ])
