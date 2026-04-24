"""
closed_loop_sim.launch.py
==========================
Full closed-loop simulation launch file.

Launches:
    1. Gazebo (via gazebo_ackermann_steering_vehicle/launch/vehicle.launch.py)
    2. sim_bridge_node        — bridges /vehicle/cmd ↔ Gazebo topics,
                                publishes /vehicle/state ground-truth
    3. speed_control_node     — PI controller → /teleop/speed_cmd  (Float64)
    4. lateral_control_node   — Extended Stanley → /teleop/lateral_cmd (Float64)
    5. control_merger_node    — merges both into /teleop/raw_cmd (VehicleCmd)
    6. nonholonomic_constraints_node — applies kinematic limits → /vehicle/cmd

All controller nodes come from the mid_level_controller package.

Topic graph:
    speed_control_node  ──► /teleop/speed_cmd   ──►┐
                                                     control_merger_node ──► /teleop/raw_cmd
    lateral_control_node ──► /teleop/lateral_cmd ──►┘         │
                                                              ▼
                                              nonholonomic_constraints_node
                                                              │
                                                              ▼
                                                        /vehicle/cmd
                                                              │
                                                              ▼
                                                      sim_bridge_node ──► Gazebo

Usage:
    # Defaults (0.5 m/s, straight lane y=0, spawn at origin):
    ros2 launch zooba_simulation closed_loop_sim.launch.py

    # Custom initial pose and goals:
    ros2 launch zooba_simulation closed_loop_sim.launch.py \\
        x:=1.0 y:=0.5 Y:=0.0 \\
        desired_speed:=0.8 desired_y:=2.0

    # Tune PI and Stanley gains:
    ros2 launch zooba_simulation closed_loop_sim.launch.py \\
        kp:=2.0 ki:=0.3 k_stanley:=3.0 k_d_heading:=1.0
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
    # The PosePublisher plugin in the vehicle xacro publishes the model's
    # world-frame pose on /model/ackermann_steering_vehicle/pose (gz.msgs.Pose).
    # Bridge it to ROS2 as geometry_msgs/PoseStamped.
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
    # ---- 2b. Simulation bridge node --------------------------------
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
            'state_topic':    '/vehicle/state',
            'feedback_topic': '/vehicle/feedback',
            'pose_topic':     '/model/ackermann_steering_vehicle/pose',
            'wheel_radius':   0.04,
            'wheelbase':      0.22,
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
        sim_bridge,
        speed_control,
        lateral_control,
        cmd_merger,
        # constraints_node,
    ])
