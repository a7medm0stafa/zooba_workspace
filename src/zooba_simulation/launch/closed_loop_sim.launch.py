"""
closed_loop_sim.launch.py — Full Closed-Loop Simulation
=========================================================
FILE: zooba_simulation/launch/closed_loop_sim.launch.py
STATUS: MODIFIED — replaced ground_truth_node with EKF localization
MODIFIED: 2026-04-24

CHANGES MADE:
    - REPLACED: localization/ground_truth_node (perfect Gazebo pose, no IMU sim)
    - WITH:     localization/ekf_localization_node (EKF, uses /joint_states)
    - The EKF runs in SIMULATION MODE (source='simulation')
    - It subscribes to /joint_states published by Gazebo JointStatePublisher plugin
    - Lower noise parameters since Gazebo joints are cleaner than real hardware

GAZEBO ACKERMANN COMPATIBILITY:
    The EKF node reads /joint_states from the gazebo_ackermann_steering_vehicle model.
    Joint names used (from vehicle.xacro):
        rear_left_wheel_joint       → velocity → linear speed v
        rear_right_wheel_joint      → velocity → linear speed v
        front_left_steering_joint   → position → steering angle δ → yaw rate ω
    Yaw rate is computed from bicycle kinematics: ω = v·tan(δ)/wheelbase

WHAT THIS LAUNCHES:
    1. Gazebo + Ackermann vehicle model (via vehicle.launch.py)
    2. Pose bridge (still available for ground truth comparison)
    3. EKF Localization node (simulation mode → /joint_states → /vehicle/state)
    4. Simulation bridge node (/vehicle/cmd ↔ Gazebo + /vehicle/feedback)
    5. Speed control node (PI → /teleop/speed_cmd)
    6. Lateral control node (Extended Stanley → /teleop/lateral_cmd)
    7. Control merger node (merges → /vehicle/cmd)

SIGNAL FLOW:
    Gazebo → /joint_states → EKF Localization Node → /vehicle/state
          → Speed Control + Lateral Control → speed_cmd + lateral_cmd
          → Control Merger → /vehicle/cmd
          → Simulation Bridge → /steering_angle + /velocity → Gazebo

USAGE:
    # Defaults (0.3 m/s, lane y=1.0, spawn at origin):
    ros2 launch zooba_simulation closed_loop_sim.launch.py

    # Custom initial pose and goals:
    ros2 launch zooba_simulation closed_loop_sim.launch.py \\
        x:=1.0 y:=0.5 Y:=0.0 \\
        desired_speed:=0.8 desired_y:=2.0

    # Tune PI and Stanley gains:
    ros2 launch zooba_simulation closed_loop_sim.launch.py \\
        kp:=2.0 ki:=0.3 k_stanley:=3.0 k_d_heading:=1.0

ROLLBACK (revert to ground truth):
    1. Replace the ekf_localization Node definition with ground_truth_node
    2. Change 'ekf_localization' to 'ground_truth' in LaunchDescription list
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    # ================================================================
    # ---- Launch arguments: initial vehicle pose --------------------
    # ================================================================
    world_arg = DeclareLaunchArgument(
        'world', default_value='empty.sdf',
        description='Gazebo world file'
    )
    x_arg = DeclareLaunchArgument(
        'x', default_value='0.0',
        description='Initial X position [m]'
    )
    y_arg = DeclareLaunchArgument(
        'y', default_value='0.0',
        description='Initial Y position [m]'
    )
    z_arg = DeclareLaunchArgument(
        'z', default_value='0.1',
        description='Initial Z (height) [m]'
    )
    roll_arg = DeclareLaunchArgument(
        'R', default_value='0.0',
        description='Initial Roll [rad]'
    )
    pitch_arg = DeclareLaunchArgument(
        'P', default_value='0.0',
        description='Initial Pitch [rad]'
    )
    yaw_arg = DeclareLaunchArgument(
        'Y', default_value='0.0',
        description='Initial Yaw / heading [rad]'
    )

    # ================================================================
    # ---- Launch arguments: speed controller (PI) -------------------
    # ================================================================
    desired_speed_arg = DeclareLaunchArgument(
        'desired_speed', default_value='0.3',
        description='Target speed [m/s]'
    )
    kp_arg = DeclareLaunchArgument(
        'kp', default_value='0.5',
        description='PI proportional gain'
    )
    ki_arg = DeclareLaunchArgument(
        'ki', default_value='0.1',
        description='PI integral gain'
    )
    max_velocity_arg = DeclareLaunchArgument(
        'max_velocity', default_value='2.0',
        description='Speed output saturation [m/s]'
    )

    # ================================================================
    # ---- Launch arguments: lateral controller (Extended Stanley) ---
    # ================================================================
    desired_y_arg = DeclareLaunchArgument(
        'desired_y', default_value='1.0',
        description='Target lateral lane/distance [m]'
    )
    desired_heading_arg = DeclareLaunchArgument(
        'desired_heading', default_value='0.0',
        description='Target heading [degrees] (0 = +X, 90 = +Y, 180 = -X)'
    )
    k_heading_arg = DeclareLaunchArgument(
        'k_heading', default_value='1.0',
        description='Heading proportional gain (> 1 for aggressive heading alignment)'
    )
    k_stanley_arg = DeclareLaunchArgument(
        'k_stanley', default_value='1.2',
        description='Stanley cross-track gain (lower = smoother lane change)'
    )
    k_soft_arg = DeclareLaunchArgument(
        'k_soft', default_value='1.0',
        description='Stanley softening constant (avoids div-by-zero)'
    )
    k_d_heading_arg = DeclareLaunchArgument(
        'k_d_heading', default_value='0.2',
        description='Heading derivative damping gain (prevents heading overshoot)'
    )
    max_steering_arg = DeclareLaunchArgument(
        'max_steering_angle', default_value='45.0',
        description='Steering output saturation [degrees]'
    )

    # ================================================================
    # ---- 1. Gazebo + vehicle model ---------------------------------
    # ================================================================
    gazebo_pkg = get_package_share_directory('gazebo_ackermann_steering_vehicle')
    vehicle_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg, 'launch', 'vehicle.launch.py')
        ),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'x':     LaunchConfiguration('x'),
            'y':     LaunchConfiguration('y'),
            'z':     LaunchConfiguration('z'),
            'R':     LaunchConfiguration('R'),
            'P':     LaunchConfiguration('P'),
            'Y':     LaunchConfiguration('Y'),
        }.items()
    )

    # ================================================================
    # ---- 2a. Gazebo model pose bridge (world-frame ground truth) ---
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
    # ---- 2b. EKF Localization (simulation mode — uses joint_states) -
    # ================================================================
    ekf_localization = Node(
        package='localization',
        executable='ekf_localization_node',
        name='ekf_localization_node',
        output='screen',
        parameters=[{
            'source': 'simulation',
            'wheelbase': 0.22,
            'wheel_radius': 0.04,
            'state_topic': '/vehicle/state',
            'publish_rate': 50.0,
            # Simulation: lower noise since Gazebo joints are clean
            'process_noise_x': 0.005,
            'process_noise_y': 0.005,
            'process_noise_yaw': 0.002,
            'process_noise_vel': 0.05,
            'process_noise_gyro_bias': 0.0001,
            'encoder_velocity_noise': 0.02,
            'gyro_rate_noise': 0.005,
            'imu_yaw_noise': 0.1,
            'zupt_velocity_threshold': 0.01,
        }],
    )

    # ================================================================
    # ---- 2b_gt. Ground Truth Localization (for comparison) ---------
    # ================================================================
    ground_truth = Node(
        package='localization',
        executable='ground_truth_node',
        name='ground_truth_node',
        output='screen',
        parameters=[{
            'pose_topic':   '/model/ackermann_steering_vehicle/pose',
            'state_topic':  '/vehicle/state_gt',  # DIFFERENT TOPIC!
            'publish_rate': 50.0,
            'wheel_radius': 0.04,
            'wheelbase':    0.22,
        }],
    )

    # ================================================================
    # ---- 2c. Simulation bridge node (commands + encoder feedback) --
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
    # ---- 3. PI speed control node ----------------------------------
    # ================================================================
    speed_control = Node(
        package='mid_level_controller',
        executable='speed_control_node',
        name='speed_control_node',
        output='screen',
        parameters=[{
            'desired_speed': LaunchConfiguration('desired_speed'),
            'kp':            LaunchConfiguration('kp'),
            'ki':            LaunchConfiguration('ki'),
            'max_velocity':  LaunchConfiguration('max_velocity'),
            'control_rate':  20.0,
            'state_topic':   '/vehicle/state',
            'output_topic':  '/teleop/speed_cmd',
            # bypass_pi defaults to False — simulation uses ROS PI
        }],
    )

    # ================================================================
    # ---- 4. Extended Stanley lateral control node ------------------
    # ================================================================
    lateral_control = Node(
        package='mid_level_controller',
        executable='lateral_control_node',
        name='lateral_control_node',
        output='screen',
        parameters=[{
            'desired_y':          LaunchConfiguration('desired_y'),
            'desired_heading':    LaunchConfiguration('desired_heading'),
            'k_heading':          LaunchConfiguration('k_heading'),
            'k_stanley':          LaunchConfiguration('k_stanley'),
            'k_soft':             LaunchConfiguration('k_soft'),
            'k_d_heading':        LaunchConfiguration('k_d_heading'),
            'max_steering_angle': LaunchConfiguration('max_steering_angle'),
            'control_rate':       20.0,
            'invert_steering_output': True,
            'state_topic':        '/vehicle/state',
            'output_topic':       '/teleop/lateral_cmd',
        }],
    )

    # ================================================================
    # ---- 5. Command merger node ------------------------------------
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

    # # ================================================================
    # # ---- 6. Non-holonomic constraints node -------------------------
    # # ================================================================
    # mid_pkg = get_package_share_directory('mid_level_controller')
    # constraints_config = os.path.join(mid_pkg, 'config', 'vehicle_constraints.yaml')

    # constraints_node = Node(
    #     package='mid_level_controller',
    #     executable='nonholonomic_constraints_node',
    #     name='nonholonomic_constraints_node',
    #     output='screen',
    #     parameters=[constraints_config],
    # )

    # ================================================================
    return LaunchDescription([
        # --- declare all args first ---
        world_arg,
        x_arg, y_arg, z_arg, roll_arg, pitch_arg, yaw_arg,
        desired_speed_arg, kp_arg, ki_arg, max_velocity_arg,
        desired_y_arg, desired_heading_arg,
        k_heading_arg, k_stanley_arg, k_soft_arg, k_d_heading_arg, max_steering_arg,
        # --- then launch everything ---
        vehicle_launch,
        pose_bridge,
        ekf_localization,
        ground_truth,
        sim_bridge,
        speed_control,
        lateral_control,
        cmd_merger,
        # constraints_node,
    ])
