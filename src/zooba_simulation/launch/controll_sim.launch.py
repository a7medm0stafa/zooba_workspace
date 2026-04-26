"""
controll_sim.launch.py
=======================
Full closed-loop simulation launch file.

Launches:
    1. Gazebo (via gazebo_ackermann_steering_vehicle/launch/vehicle.launch.py)
    2. sim_bridge_node          — bridges /vehicle/cmd ↔ Gazebo topics,
                                  publishes /vehicle/state ground-truth
    3. sim_speed_control_node   — PI controller → /sim/speed_cmd  (Float64)
    4a. sim_lateral_control_node     — Stanley controller (lateral_controller:=stanley)
    4b. sim_smc_lateral_control_node — SMC controller    (lateral_controller:=smc)
    5. sim_cmd_merger_node      — merges both into /vehicle/cmd (VehicleCmd)

No mid_level_controller node is used.

Topic graph:
    speed_control_node   ──► /sim/speed_cmd   ──►┌
                                                   sim_cmd_merger_node ──► /vehicle/cmd ──► sim_bridge_node ──► Gazebo
    lateral_control_node ──► /sim/lateral_cmd ──►┘
    sim_bridge_node ──► /vehicle/state ──► speed_control_node
                                     ──► lateral_control_node

Usage:
    # Default: Stanley controller, two-lane world:
    ros2 launch zooba_simulation controll_sim.launch.py

    # SMC lateral controller:
    ros2 launch zooba_simulation controll_sim.launch.py lateral_controller:=smc

    # Stanley with custom gains:
    ros2 launch zooba_simulation controll_sim.launch.py \\
        lateral_controller:=stanley \\
        k_stanley:=1.5 k_d_heading:=0.8

    # SMC with custom gains:
    ros2 launch zooba_simulation controll_sim.launch.py \\
        lateral_controller:=smc \\
        k_smc:=4.0 lambda_smc:=2.0 phi:=0.2

    # Empty world with Stanley:
    ros2 launch zooba_simulation controll_sim.launch.py \\
        use_two_lane:=false lateral_controller:=stanley

    # Tune PI speed gains:
    ros2 launch zooba_simulation controll_sim.launch.py \\
        kp:=2.0 ki:=0.3 desired_speed:=0.8
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    # ================================================================
    # ---- Resolve world path ----------------------------------------
    # ================================================================
    sim_pkg = get_package_share_directory('zooba_simulation')
    two_lane_world_path = os.path.join(sim_pkg, 'worlds', 'two_lane.world')

    # ================================================================
    # ---- Launch arguments: world / pose ----------------------------
    # ================================================================
    use_two_lane_arg = DeclareLaunchArgument(
        'use_two_lane', default_value='false',
        description='Set true to load the two-lane road world instead of empty.sdf'
    )
    world_arg = DeclareLaunchArgument(
        'world', default_value='empty.sdf',
        description='Gazebo world file (ignored when use_two_lane:=true)'
    )
    x_arg = DeclareLaunchArgument('x', default_value='0.0',  description='Initial X position [m]')
    y_arg = DeclareLaunchArgument('y', default_value='0.0',  description='Initial Y position [m]')
    z_arg = DeclareLaunchArgument('z', default_value='0.1',  description='Initial Z (height) [m]')
    roll_arg  = DeclareLaunchArgument('R', default_value='0.0', description='Initial Roll [rad]')
    pitch_arg = DeclareLaunchArgument('P', default_value='0.0', description='Initial Pitch [rad]')
    yaw_arg   = DeclareLaunchArgument('Y', default_value='0.0', description='Initial Yaw [rad]')

    # ================================================================
    # ---- Launch argument: lateral controller selector --------------
    # ================================================================
    lateral_controller_arg = DeclareLaunchArgument(
        'lateral_controller', default_value='stanley',
        description=(
            'Which lateral controller to use. '
            'Options: "stanley" (default) or "smc" (Sliding Mode Control)'
        )
    )

    # ================================================================
    # ---- Launch arguments: speed controller (PI) -------------------
    # ================================================================
    desired_speed_arg = DeclareLaunchArgument(
        'desired_speed', default_value='0.2',
        description='Target speed [m/s]'
    )
    kp_arg = DeclareLaunchArgument(
        'kp', default_value='0.4',
        description='PI proportional gain'
    )
    ki_arg = DeclareLaunchArgument(
        'ki', default_value='0.07',
        description='PI integral gain'
    )
    max_velocity_arg = DeclareLaunchArgument(
        'max_velocity', default_value='2.0',
        description='Speed output saturation [m/s]'
    )

    # ================================================================
    # ---- Launch arguments: common lateral parameters ---------------
    # ================================================================
    desired_y_arg = DeclareLaunchArgument(
        'desired_y', default_value='1.0',
        description='Target lateral lane/distance [m] (both controllers)'
    )
    desired_heading_arg = DeclareLaunchArgument(
        'desired_heading', default_value='0.0',
        description='Target heading [rad], 0 = straight (both controllers)'
    )
    max_steering_arg = DeclareLaunchArgument(
        'max_steering_angle', default_value='35.0',
        description='Steering output saturation [degrees] (both controllers)'
    )

    # ================================================================
    # ---- Launch arguments: Stanley-specific gains ------------------
    # ================================================================
    k_stanley_arg = DeclareLaunchArgument(
        'k_stanley', default_value='1.0',
        description='[Stanley] Cross-track gain'
    )
    k_soft_arg = DeclareLaunchArgument(
        'k_soft', default_value='1.0',
        description='[Stanley] Softening constant (avoids div-by-zero at low speed)'
    )
    k_d_heading_arg = DeclareLaunchArgument(
        'k_d_heading', default_value='0.6',
        description='[Stanley] Heading derivative damping gain (prevents heading overshoot)'
    )

    # ================================================================
    # ---- Launch arguments: SMC-specific gains ----------------------
    # ================================================================
    lambda_smc_arg = DeclareLaunchArgument(
        'lambda_smc', default_value='0.2',
        description='[SMC] Sliding surface CTE weighting gain (blends CTE vs heading)'
    )
    k_smc_arg = DeclareLaunchArgument(
        'k_smc', default_value='1.0',
        description='[SMC] Switching gain (drives sliding surface to zero)'
    )
    k_heading_arg = DeclareLaunchArgument(
        'k_heading', default_value='0.5',
        description='[SMC] Proportional heading feedback gain'
    )
    phi_arg = DeclareLaunchArgument(
        'phi', default_value='0.1',
        description='[SMC] Boundary layer thickness [rad] — suppresses chattering near s=0'
    )

    # ================================================================
    # ---- Condition helpers -----------------------------------------
    # ================================================================
    use_stanley = IfCondition(
        PythonExpression(["'", LaunchConfiguration('lateral_controller'), "' == 'stanley'"])
    )
    use_smc = IfCondition(
        PythonExpression(["'", LaunchConfiguration('lateral_controller'), "' == 'smc'"])
    )

    # ================================================================
    # ---- 1a. Gazebo — EMPTY world (use_two_lane:=false) ------------
    # ================================================================
    gazebo_pkg = get_package_share_directory('gazebo_ackermann_steering_vehicle')

    vehicle_launch_empty = IncludeLaunchDescription(
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
        }.items(),
        condition=UnlessCondition(LaunchConfiguration('use_two_lane'))
    )

    # ================================================================
    # ---- 1b. Gazebo — TWO-LANE world (use_two_lane:=true) ----------
    # ================================================================
    vehicle_launch_two_lane = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg, 'launch', 'vehicle.launch.py')
        ),
        launch_arguments={
            'world': two_lane_world_path,
            'x':     LaunchConfiguration('x'),
            'y':     LaunchConfiguration('y'),
            'z':     LaunchConfiguration('z'),
            'R':     LaunchConfiguration('R'),
            'P':     LaunchConfiguration('P'),
            'Y':     LaunchConfiguration('Y'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_two_lane'))
    )

    # ================================================================
    # ---- 2. Simulation bridge node ---------------------------------
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
        }],
    )

    # ================================================================
    # ---- 3. PI speed control node ----------------------------------
    # ================================================================
    speed_control = Node(
        package='zooba_simulation',
        executable='sim_speed_control_node',
        name='speed_control_node',
        output='screen',
        parameters=[{
            'desired_speed': LaunchConfiguration('desired_speed'),
            'kp':            LaunchConfiguration('kp'),
            'ki':            LaunchConfiguration('ki'),
            'max_velocity':  LaunchConfiguration('max_velocity'),
            'control_rate':  20.0,
            'state_topic':   '/vehicle/state',
            'output_topic':  '/sim/speed_cmd',
        }],
    )

    # ================================================================
    # ---- 4a. Stanley lateral control node (lateral_controller:=stanley)
    # ================================================================
    lateral_control_stanley = Node(
        package='zooba_simulation',
        executable='sim_lateral_control_node',
        name='lateral_control_node',
        output='screen',
        parameters=[{
            'desired_y':          LaunchConfiguration('desired_y'),
            'desired_heading':    LaunchConfiguration('desired_heading'),
            'k_stanley':          LaunchConfiguration('k_stanley'),
            'k_soft':             LaunchConfiguration('k_soft'),
            'k_d_heading':        LaunchConfiguration('k_d_heading'),
            'max_steering_angle': LaunchConfiguration('max_steering_angle'),
            'control_rate':       20.0,
            'state_topic':        '/vehicle/state',
            'output_topic':       '/sim/lateral_cmd',
        }],
        condition=use_stanley,
    )

    # ================================================================
    # ---- 4b. SMC lateral control node (lateral_controller:=smc) ----
    # ================================================================
    lateral_control_smc = Node(
        package='zooba_simulation',
        executable='sim_smc_lateral_control_node',
        name='lateral_control_node',
        output='screen',
        parameters=[{
            'desired_y':          LaunchConfiguration('desired_y'),
            'desired_heading':    LaunchConfiguration('desired_heading'),
            'lambda_smc':         LaunchConfiguration('lambda_smc'),
            'k_smc':              LaunchConfiguration('k_smc'),
            'k_heading':          LaunchConfiguration('k_heading'),
            'phi':                LaunchConfiguration('phi'),
            'max_steering_angle': LaunchConfiguration('max_steering_angle'),
            'control_rate':       20.0,
            'state_topic':        '/vehicle/state',
            'output_topic':       '/sim/lateral_cmd',
        }],
        condition=use_smc,
    )

    # ================================================================
    # ---- 5. Command merger node ------------------------------------
    # ================================================================
    cmd_merger = Node(
        package='zooba_simulation',
        executable='sim_cmd_merger_node',
        name='sim_cmd_merger_node',
        output='screen',
        parameters=[{
            'speed_topic':   '/sim/speed_cmd',
            'lateral_topic': '/sim/lateral_cmd',
            'output_topic':  '/vehicle/cmd',
            'publish_rate':  20.0,
        }],
    )

    # ================================================================
    return LaunchDescription([
        # --- declare all args first ---
        use_two_lane_arg,
        world_arg,
        x_arg, y_arg, z_arg, roll_arg, pitch_arg, yaw_arg,
        lateral_controller_arg,
        desired_speed_arg, kp_arg, ki_arg, max_velocity_arg,
        desired_y_arg, desired_heading_arg, max_steering_arg,
        # Stanley gains
        k_stanley_arg, k_soft_arg, k_d_heading_arg,
        # SMC gains
        lambda_smc_arg, k_smc_arg, k_heading_arg, phi_arg,
        # --- then launch everything ---
        vehicle_launch_empty,
        vehicle_launch_two_lane,
        sim_bridge,
        speed_control,
        lateral_control_stanley,
        lateral_control_smc,
        cmd_merger,
    ])
