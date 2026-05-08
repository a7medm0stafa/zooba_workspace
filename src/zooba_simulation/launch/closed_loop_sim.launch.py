"""
closed_loop_sim.launch.py — Full Closed-Loop Simulation
=========================================================
FILE: zooba_simulation/launch/closed_loop_sim.launch.py

LOCALIZATION:
    Simulation uses ground_truth_node (perfect Gazebo pose) for state estimation.
    EKF localization is reserved for HARDWARE only (see closed_loop_hw.launch.py).

WHAT THIS LAUNCHES:
    1. Gazebo + Ackermann vehicle model (via vehicle.launch.py)
    2. Pose bridge (Gazebo world-frame pose → ROS)
    3. Ground Truth Localization node (Gazebo pose → /vehicle/state)
    4. Simulation bridge node (/vehicle/cmd ↔ Gazebo + /vehicle/feedback)
    5. Speed control node (PI → /teleop/speed_cmd)
    6. Lateral control node (Extended Stanley → /teleop/lateral_cmd)
    7. Control merger node (merges → /vehicle/cmd)
    8. Path planner node (optional — enabled via use_planner:=true)

SIGNAL FLOW:
    Gazebo → /model/.../pose → Ground Truth Node → /vehicle/state
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

    # Path planning with track 3:
    ros2 launch zooba_simulation closed_loop_sim.launch.py \\
        use_planner:=true track:=track_3
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    # ================================================================
    # ---- Vehicle constants (from vehicle_params.yaml) --------------
    # ================================================================
    WHEELBASE = 0.22
    WHEEL_RADIUS = 0.033
    MAX_STEERING_ANGLE = 45.0
    CONTROL_RATE = 20.0

    # Simulation-specific overrides
    MAX_VELOCITY_SIM = 2.0

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
        'k_d_heading', default_value='0.3',
        description='Heading derivative damping gain (prevents heading overshoot)'
    )

    # ================================================================
    # ---- Launch arguments: path planner ----------------------------
    # ================================================================
    track_arg = DeclareLaunchArgument(
        'track', default_value='track_1',
        description='Track for path planner (track_1, track_2, track_3)'
    )
    use_planner_arg = DeclareLaunchArgument(
        'use_planner', default_value='false',
        description='Enable path planner (overrides desired_speed/desired_y)'
    )

    # ================================================================
    # ---- Config file paths -----------------------------------------
    # ================================================================
    hlc_pkg = get_package_share_directory('high_level_controller')
    planner_config = os.path.join(hlc_pkg, 'config', 'path_planner_config.yaml')

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
    # ---- 2b. Ground Truth Localization (Gazebo pose → /vehicle/state)
    # ================================================================
    ground_truth = Node(
        package='localization',
        executable='ground_truth_node',
        name='ground_truth_node',
        output='screen',
        parameters=[{
            'pose_topic':   '/model/ackermann_steering_vehicle/pose',
            'state_topic':  '/vehicle/state',
            'publish_rate': CONTROL_RATE,
            'wheel_radius': WHEEL_RADIUS,
            'wheelbase':    WHEELBASE,
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
            'wheel_radius':   WHEEL_RADIUS,
            'publish_rate':   CONTROL_RATE,
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
            'max_velocity':  MAX_VELOCITY_SIM,
            'control_rate':  CONTROL_RATE,
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
            'max_steering_angle': MAX_STEERING_ANGLE,
            'control_rate':       CONTROL_RATE,
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
            'publish_rate':  CONTROL_RATE,
        }],
    )

    # ================================================================
    # ---- 6. Path planner node (optional — use_planner:=true) -------
    # ================================================================
    path_planner = Node(
        package='high_level_controller',
        executable='path_planner_node',
        name='path_planner_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_planner')),
        parameters=[
            planner_config,
            {
                'track_name':    LaunchConfiguration('track'),
                'state_topic':   '/vehicle/state',
                'start_delay':   3.0,
            }
        ],
    )

    # ================================================================
    return LaunchDescription([
        # --- declare all args first ---
        world_arg,
        x_arg, y_arg, z_arg, roll_arg, pitch_arg, yaw_arg,
        desired_speed_arg, kp_arg, ki_arg,
        desired_y_arg, desired_heading_arg,
        k_heading_arg, k_stanley_arg, k_soft_arg, k_d_heading_arg,
        track_arg, use_planner_arg,
        # --- then launch everything ---
        vehicle_launch,
        pose_bridge,
        ground_truth,
        sim_bridge,
        # Path planner (conditional)
        path_planner,
        # Controllers
        speed_control,
        lateral_control,
        cmd_merger,
    ])
