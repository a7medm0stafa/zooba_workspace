"""
closed_loop_sim_track.launch.py — Track Simulation with Path Planner
=====================================================================
FILE: zooba_simulation/launch/closed_loop_sim_track.launch.py

Launches a full closed-loop simulation for a specified track with:
    1. Gazebo + Ackermann vehicle model
    2. Pose bridge (Gazebo world-frame pose)
    3. Ground Truth localization ONLY (no EKF, no dead-reckoning)
    4. Simulation bridge (VehicleCmd ↔ Gazebo topics)
    5. Path Planner node (cubic spline trajectory for the selected track)
    6. Speed control node (PI)
    7. Lateral control node (Extended Stanley)
    8. Control merger node (speed + lateral → /vehicle/cmd)

SIGNAL FLOW:
    Gazebo → /model/.../pose + /joint_states → Ground Truth → /vehicle/state
    Path Planner reads /vehicle/state → updates desired_speed, desired_y, desired_heading
    Speed Control + Lateral Control → speed_cmd + lateral_cmd
    Control Merger → /vehicle/cmd
    Simulation Bridge → /steering_angle + /velocity → Gazebo

USAGE:
    # Track 1 (lane keeping):
    ros2 launch zooba_simulation closed_loop_sim_track.launch.py track:=track_1

    # Track 2 (obstacle avoidance):
    ros2 launch zooba_simulation closed_loop_sim_track.launch.py track:=track_2

    # Track 3 (closed circuit):
    ros2 launch zooba_simulation closed_loop_sim_track.launch.py track:=track_3

    # With custom speed:
    ros2 launch zooba_simulation closed_loop_sim_track.launch.py track:=track_2 cruise_speed:=0.20
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    # ================================================================
    # ---- Launch arguments ------------------------------------------
    # ================================================================
    track_arg = DeclareLaunchArgument(
        'track', default_value='track_1',
        description='Track to load (track_1, track_2, track_3)'
    )

    # Path planner tuning
    cruise_speed_arg = DeclareLaunchArgument(
        'cruise_speed', default_value='1.5',
        description='Cruise speed on straights [m/s]'
    )
    curve_speed_arg = DeclareLaunchArgument(
        'curve_speed', default_value='1.0',
        description='Speed in tight curves [m/s]'
    )
    lookahead_arg = DeclareLaunchArgument(
        'lookahead', default_value='0.20',
        description='Lookahead distance on trajectory [m]'
    )

    # Stanley gains (simulation-tuned defaults)
    k_heading_arg = DeclareLaunchArgument(
        'k_heading', default_value='3.0',
        description='Heading proportional gain'
    )
    k_stanley_arg = DeclareLaunchArgument(
        'k_stanley', default_value='5.0',
        description='Cross-track gain'
    )
    k_soft_arg = DeclareLaunchArgument(
        'k_soft', default_value='1.0',
        description='Softening constant'
    )
    k_d_heading_arg = DeclareLaunchArgument(
        'k_d_heading', default_value='0.3',
        description='Heading derivative damping'
    )

    # PI speed gains
    kp_arg = DeclareLaunchArgument(
        'kp', default_value='0.5',
        description='PI proportional gain'
    )
    ki_arg = DeclareLaunchArgument(
        'ki', default_value='0.1',
        description='PI integral gain'
    )

    # ================================================================
    # ---- Package paths ---------------------------------------------
    # ================================================================
    zooba_sim_pkg = get_package_share_directory('zooba_simulation')
    gazebo_pkg = get_package_share_directory('gazebo_ackermann_steering_vehicle')
    hlc_pkg = get_package_share_directory('high_level_controller')

    planner_config = os.path.join(hlc_pkg, 'config', 'path_planner_config.yaml')

    # ================================================================
    # ---- 1. Gazebo + Vehicle Model ---------------------------------
    # ================================================================
    # Build world path from track argument
    world_path = PathJoinSubstitution([
        zooba_sim_pkg, 'worlds',
        [LaunchConfiguration('track'), '.world']
    ])

    # Determine initial Y based on track (Track 3 starts at Y=1.75)
    initial_y = PythonExpression([
        "'1.75' if '", LaunchConfiguration('track'), "' == 'track_3' else '0.1875'"
    ])

    vehicle_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg, 'launch', 'vehicle.launch.py')
        ),
        launch_arguments={
            'world': world_path,
            'x': '0.0',
            'y': initial_y,
            'z': '0.1',
            'Y': '0.0',
        }.items()
    )

    # ================================================================
    # ---- 2. Gazebo Pose Bridge (for ground truth) ------------------
    # ================================================================
    pose_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='pose_bridge',
        output='screen',
        arguments=[
            '/model/ackermann_steering_vehicle/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose',
        ],
    )

    # ================================================================
    # ---- 3. Ground Truth Node (ONLY localization source) -----------
    # ================================================================
    ground_truth = Node(
        package='localization',
        executable='ground_truth_node',
        name='ground_truth_node',
        output='screen',
        parameters=[{
            'pose_topic':   '/model/ackermann_steering_vehicle/pose',
            'state_topic':  '/vehicle/state',  # Primary state topic
            'publish_rate': 50.0,
            'wheel_radius': 0.04,
            'wheelbase':    0.22,
        }],
    )

    # ================================================================
    # ---- 4. Simulation Bridge (VehicleCmd ↔ Gazebo) ----------------
    # ================================================================
    sim_bridge = Node(
        package='zooba_simulation',
        executable='sim_bridge_node',
        name='sim_bridge_node',
        output='screen',
        parameters=[{
            'input_topic':    '/vehicle/cmd',
            'steering_topic': '/steering_angle',
            'velocity_topic': '/velocity',
            'feedback_topic': '/vehicle/feedback',
            'wheel_radius':   0.04,
            'publish_rate':   20.0,
        }],
    )

    # ================================================================
    # ---- 5. Path Planner Node --------------------------------------
    # ================================================================
    path_planner = Node(
        package='high_level_controller',
        executable='path_planner_node',
        name='path_planner_node',
        output='screen',
        parameters=[
            planner_config,
            {
                'track_name':         LaunchConfiguration('track'),
                'cruise_speed':       LaunchConfiguration('cruise_speed'),
                'curve_speed':        LaunchConfiguration('curve_speed'),
                'lookahead_distance': LaunchConfiguration('lookahead'),
                'state_topic':        '/vehicle/state',
                'start_delay':        3.0,  # Wait for Gazebo to settle
            }
        ],
    )

    # ================================================================
    # ---- 6. PI Speed Control Node ----------------------------------
    # ================================================================
    speed_control = Node(
        package='mid_level_controller',
        executable='speed_control_node',
        name='speed_control_node',
        output='screen',
        parameters=[{
            'desired_speed': 0.0,  # Will be set by path planner
            'kp':            LaunchConfiguration('kp'),
            'ki':            LaunchConfiguration('ki'),
            'max_velocity':  2.0,
            'control_rate':  20.0,
            'state_topic':   '/vehicle/state',
            'output_topic':  '/teleop/speed_cmd',
        }],
    )

    # ================================================================
    # ---- 7. Extended Stanley Lateral Control Node -------------------
    # ================================================================
    lateral_control = Node(
        package='mid_level_controller',
        executable='lateral_control_node',
        name='lateral_control_node',
        output='screen',
        parameters=[{
            'desired_x':          0.0,    # Will be set by path planner
            'desired_y':          0.0,    # Will be set by path planner
            'desired_heading':    0.0,    # Will be set by path planner
            'k_heading':          LaunchConfiguration('k_heading'),
            'k_stanley':          LaunchConfiguration('k_stanley'),
            'k_soft':             LaunchConfiguration('k_soft'),
            'k_d_heading':        LaunchConfiguration('k_d_heading'),
            'max_steering_angle': 45.0,
            'control_rate':       20.0,
            'invert_steering_output': True,  # Simulation requires inversion
            'state_topic':        '/vehicle/state',
            'output_topic':       '/teleop/lateral_cmd',
        }],
    )

    # ================================================================
    # ---- 8. Control Merger Node ------------------------------------
    # ================================================================
    cmd_merger = Node(
        package='mid_level_controller',
        executable='control_merger_node',
        name='control_merger_node',
        output='screen',
        parameters=[{
            'speed_topic':   '/teleop/speed_cmd',
            'lateral_topic': '/teleop/lateral_cmd',
            'output_topic':  '/vehicle/cmd',
            'publish_rate':  20.0,
        }],
    )

    # ================================================================
    return LaunchDescription([
        # Arguments
        track_arg,
        cruise_speed_arg,
        curve_speed_arg,
        lookahead_arg,
        k_heading_arg,
        k_stanley_arg,
        k_soft_arg,
        k_d_heading_arg,
        kp_arg,
        ki_arg,
        # Gazebo
        vehicle_launch,
        pose_bridge,
        # Localization (ground truth ONLY)
        ground_truth,
        # Simulation bridge
        sim_bridge,
        # Path planner
        path_planner,
        # Mid-level controllers
        speed_control,
        lateral_control,
        cmd_merger,
    ])
