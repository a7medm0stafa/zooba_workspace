"""
closed_loop_sim_track_noisy.launch.py — Track Simulation with Sensor Noise + EKF
==================================================================================
FILE: zooba_simulation/launch/closed_loop_sim_track_noisy.launch.py

Identical to closed_loop_sim_track.launch.py EXCEPT the localization pipeline
is replaced with a realistic noisy-sensor chain:

    CLEAN PIPELINE  (closed_loop_sim_track.launch.py):
        Gazebo ──▶ ground_truth_node ──▶ /vehicle/state ──▶ controllers

    NOISY PIPELINE  (this file):
        Gazebo ──▶ ground_truth_node ──▶ /vehicle/state_gt
                                          │
                                    sensor_noise_node   (adds Gaussian noise)
                                          │
                                  /vehicle/state_noisy
                                          │
                                     ekf_sim_node       (EKF filtering)
                                          │
                                    /vehicle/state ──▶ controllers

This lets you study how sensor noise affects path-tracking performance and
how well the EKF can attenuate that noise.

SIGNAL FLOW:
    Gazebo → /model/.../pose + /joint_states
        → ground_truth_node → /vehicle/state_gt  (perfect, Gazebo frame)
        → sensor_noise_node → /vehicle/state_noisy (corrupted with Gaussian noise)
        → ekf_sim_node → /vehicle/state (EKF-filtered estimate)
    Path Planner reads /vehicle/state → updates desired waypoints
    Speed Control + Lateral Control → speed_cmd + lateral_cmd
    Control Merger → /vehicle/cmd
    Simulation Bridge → /steering_angle + /velocity → Gazebo

NOISE PARAMETERS (tunable at launch time):
    sigma_position  — std-dev for x, y noise [m]           (default 0.05 m)
    sigma_yaw       — std-dev for heading noise [rad]       (default 0.02 rad)
    sigma_velocity  — std-dev for velocity noise [m/s]      (default 0.05 m/s)
    sigma_yaw_rate  — std-dev for yaw-rate noise [rad/s]    (default 0.01 rad/s)
    sigma_steering  — std-dev for steering noise [rad]      (default 0.01 rad)

USAGE:
    # Default noise, Track 3:
    ros2 launch zooba_simulation closed_loop_sim_track_noisy.launch.py

    # High position noise:
    ros2 launch zooba_simulation closed_loop_sim_track_noisy.launch.py sigma_position:=0.15

    # Different track + custom speed:
    ros2 launch zooba_simulation closed_loop_sim_track_noisy.launch.py \\
        track:=track_2 cruise_speed:=0.20 sigma_position:=0.08
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import TextSubstitution


def generate_launch_description():

    # ================================================================
    # ---- Launch arguments ------------------------------------------
    # ================================================================
    track_arg = DeclareLaunchArgument(
        'track', default_value='track_3',
        description='Track to load (track_1, track_2, track_3)'
    )

    # Path planner tuning (simulation overrides)
    cruise_speed_arg = DeclareLaunchArgument(
        'cruise_speed', default_value='2.5',
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

    # PI speed gains (simulation-tuned)
    kp_arg = DeclareLaunchArgument(
        'kp', default_value='0.5',
        description='PI proportional gain'
    )
    ki_arg = DeclareLaunchArgument(
        'ki', default_value='0.1',
        description='PI integral gain'
    )

    # ----------------------------------------------------------------
    # ---- Sensor Noise parameters ------------------------------------
    # ----------------------------------------------------------------
    sigma_position_arg = DeclareLaunchArgument(
        'sigma_position', default_value='0.01',
        description='Std-dev of Gaussian position noise [m]  (x, y)'
    )
    sigma_yaw_arg = DeclareLaunchArgument(
        'sigma_yaw', default_value='0.1',
        description='Std-dev of Gaussian heading noise [rad]  (~1.1°)'
    )
    sigma_velocity_arg = DeclareLaunchArgument(
        'sigma_velocity', default_value='0.01',
        description='Std-dev of Gaussian velocity noise [m/s]'
    )
    sigma_yaw_rate_arg = DeclareLaunchArgument(
        'sigma_yaw_rate', default_value='0.01',
        description='Std-dev of Gaussian yaw-rate noise [rad/s]'
    )
    sigma_steering_arg = DeclareLaunchArgument(
        'sigma_steering', default_value='0.01',
        description='Std-dev of Gaussian steering-angle noise [rad]  (~0.6°)'
    )

    # ================================================================
    # ---- Package paths ---------------------------------------------
    # ================================================================
    zooba_sim_pkg = get_package_share_directory('zooba_simulation')
    gazebo_pkg    = get_package_share_directory('gazebo_ackermann_steering_vehicle')
    hlc_pkg       = get_package_share_directory('high_level_controller')

    planner_config = os.path.join(hlc_pkg, 'config', 'path_planner_config.yaml')

    # ================================================================
    # ---- Vehicle constants (from vehicle_params.yaml) --------------
    # ================================================================
    WHEELBASE          = 0.22
    WHEEL_RADIUS       = 0.033
    MAX_STEERING_ANGLE = 45.0
    CONTROL_RATE       = 20.0

    # Simulation-specific overrides
    MAX_VELOCITY_SIM = 2.0

    # ================================================================
    # ---- 1. Gazebo + Vehicle Model ---------------------------------
    # ================================================================
    world_path = PathJoinSubstitution([
        zooba_sim_pkg, 'worlds',
        [LaunchConfiguration('track'), '.world']
    ])

    vehicle_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg, 'launch', 'vehicle.launch.py')
        ),
        launch_arguments={
            'world': world_path,
            'x': '0.0',
            'y': '0.0',
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
    # ---- 3a. Ground Truth Node (publishes PERFECT state to _gt) ----
    # ================================================================
    # Key difference: state_topic is /vehicle/state_gt, NOT /vehicle/state.
    # The EKF sim node will publish the final /vehicle/state after filtering.
    ground_truth = Node(
        package='localization',
        executable='ground_truth_node',
        name='ground_truth_node',
        output='screen',
        parameters=[{
            'pose_topic':   '/model/ackermann_steering_vehicle/pose',
            'state_topic':  '/vehicle/state_gt',   # ← perfect, unfiltered
            'publish_rate': 50.0,
            'wheel_radius': WHEEL_RADIUS,
            'wheelbase':    WHEELBASE,
        }],
    )

    # ================================================================
    # ---- 3b. Sensor Noise Node (GT → Noisy) -----------------------
    # ================================================================
    sensor_noise = Node(
        package='localization',
        executable='sensor_noise_node',
        name='sensor_noise_node',
        output='screen',
        parameters=[{
            'input_topic':    '/vehicle/state_gt',
            'output_topic':   '/vehicle/state_noisy',
            'sigma_position': ParameterValue(LaunchConfiguration('sigma_position'), value_type=float),
            'sigma_yaw':      ParameterValue(LaunchConfiguration('sigma_yaw'),      value_type=float),
            'sigma_velocity': ParameterValue(LaunchConfiguration('sigma_velocity'), value_type=float),
            'sigma_yaw_rate': ParameterValue(LaunchConfiguration('sigma_yaw_rate'), value_type=float),
            'sigma_steering': ParameterValue(LaunchConfiguration('sigma_steering'), value_type=float),
            'seed':           -1,  # non-deterministic by default
        }],
    )

    # ================================================================
    # ---- 3c. EKF Simulation Node (Noisy → Filtered /vehicle/state) -
    # ================================================================
    ekf_sim = Node(
        package='localization',
        executable='ekf_sim_node',
        name='ekf_sim_node',
        output='screen',
        parameters=[{
            'wheelbase':           WHEELBASE,
            'wheel_radius':        WHEEL_RADIUS,
            'publish_rate':        50.0,
            'noisy_state_topic':   '/vehicle/state_noisy',
            'state_topic':         '/vehicle/state',   # ← controllers read this
            # EKF measurement noise — match sensor_noise_node sigmas
            'meas_noise_position': ParameterValue(LaunchConfiguration('sigma_position'), value_type=float),
            'meas_noise_yaw':      ParameterValue(LaunchConfiguration('sigma_yaw'),      value_type=float),
            'meas_noise_velocity': ParameterValue(LaunchConfiguration('sigma_velocity'), value_type=float),
            # Process noise
            'process_noise_x':          0.05,
            'process_noise_y':          0.001,
            'process_noise_yaw':        0.01,
            'process_noise_vel':        0.1,
            'process_noise_gyro_bias':  0.0001,
            # ZUPT
            'zupt_velocity_threshold':  0.02,
            'zupt_noise':               0.001,
            # Initial pose (origin)
            'initial_x':   0.0,
            'initial_y':   0.0,
            'initial_yaw': 0.0,
        }],
    )

    # ================================================================
    # ---- 3d. State Comparison Printer (Terminal output) ------------
    # ================================================================
    state_comparison = Node(
        package='localization',
        executable='state_comparison_node',
        name='state_comparison_node',
        output='screen',
        parameters=[{
            'gt_topic':    '/vehicle/state_gt',
            'noisy_topic': '/vehicle/state_noisy',
            'ekf_topic':   '/vehicle/state',
            'print_rate':  2.0,   # Hz — prints twice per second
            'use_color':   True,
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
            'wheel_radius':   WHEEL_RADIUS,
            'publish_rate':   CONTROL_RATE,
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
                'start_delay':        3.0,
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
            'desired_speed': 0.0,
            'kp':            LaunchConfiguration('kp'),
            'ki':            LaunchConfiguration('ki'),
            'max_velocity':  MAX_VELOCITY_SIM,
            'control_rate':  CONTROL_RATE,
            'state_topic':   '/vehicle/state',
            'output_topic':  '/teleop/speed_cmd',
        }],
    )

    # ================================================================
    # ---- 7. Extended Stanley Lateral Control Node ------------------
    # ================================================================
    lateral_control = Node(
        package='mid_level_controller',
        executable='lateral_control_node',
        name='lateral_control_node',
        output='screen',
        parameters=[{
            'desired_x':          0.0,
            'desired_y':          0.0,
            'desired_heading':    0.0,
            'k_heading':          LaunchConfiguration('k_heading'),
            'k_stanley':          LaunchConfiguration('k_stanley'),
            'k_soft':             LaunchConfiguration('k_soft'),
            'k_d_heading':        LaunchConfiguration('k_d_heading'),
            'max_steering_angle': MAX_STEERING_ANGLE,
            'control_rate':       CONTROL_RATE,
            'invert_steering_output': False,
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
            'publish_rate':  CONTROL_RATE,
        }],
    )

    # ================================================================
    return LaunchDescription([
        # ---- Arguments ----
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
        # Noise arguments
        sigma_position_arg,
        sigma_yaw_arg,
        sigma_velocity_arg,
        sigma_yaw_rate_arg,
        sigma_steering_arg,
        # ---- Gazebo ----
        vehicle_launch,
        pose_bridge,
        # ---- Localization chain (noisy pipeline) ----
        ground_truth,      # GT → /vehicle/state_gt
        sensor_noise,      # /vehicle/state_gt → /vehicle/state_noisy
        ekf_sim,           # /vehicle/state_noisy → /vehicle/state
        state_comparison,  # Prints GT | Noisy | EKF comparison table
        # ---- Simulation bridge ----
        sim_bridge,
        # ---- Path planner ----
        path_planner,
        # ---- Mid-level controllers ----
        speed_control,
        lateral_control,
        cmd_merger,
    ])
