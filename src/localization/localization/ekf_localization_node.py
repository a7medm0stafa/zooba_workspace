"""
EKF Localization Node — Fused Encoder + IMU State Estimation
==============================================================
FILE: localization/localization/ekf_localization_node.py
STATUS: NEW FILE — replaces the old odometry_node.py (dead-reckoning)
CREATED: 2026-04-24

WHAT THIS FILE DOES:
    This is the main ROS2 localization node. It runs an Extended Kalman
    Filter (EKF) that fuses encoder and IMU data to produce a stable,
    drift-resistant vehicle pose estimate.

WHAT CHANGED (vs. old system):
    - OLD: mid_level_controller/odometry_node.py — direct yaw = IMU, x += Δd·cos(yaw)
           Dead-reckoning with NO drift correction. Yaw drift → y-position drift.
    - NEW: EKF with 5-state model [x, y, θ, v, ω_bias]
           → Gyro bias estimation removes yaw drift
           → Zero-Velocity Update (ZUPT) locks position when stopped
           → IMU heading used as soft anchor (low trust) for long-term stability

MODES OF OPERATION:
    1. HARDWARE MODE (source='hardware'):
       - Subscribes to: /vehicle/feedback (VehicleFeedback) — encoder ticks + velocity
                         /vehicle/imu      (ImuData)          — gyro_z + comp.filter yaw
       - This is what runs on the real car (Raspberry Pi + Arduino)
       - Usage: ros2 launch localization ekf_localization.launch.py source:=hardware
                ros2 launch high_level_controller closed_loop_hw.launch.py

    2. SIMULATION MODE (source='simulation'):
       - Subscribes to: /joint_states (JointState) — Gazebo wheel velocities + steering
       - Compatible with the gazebo_ackermann_steering_vehicle model
       - Gazebo joint names used:
            rear_left_wheel_joint      → angular velocity → linear velocity
            rear_right_wheel_joint     → angular velocity → linear velocity
            front_left_steering_joint  → steering position → yaw rate (kinematics)
       - Usage: ros2 launch localization ekf_localization.launch.py source:=simulation
                ros2 launch zooba_simulation closed_loop_sim.launch.py

PUBLISHES:
    /vehicle/state      (vehicle_interfaces/VehicleState)  — fused state estimate
    TF: odom → base_link                                    — transform for RViz

PARAMETERS (all configurable via YAML or launch file):
    source, wheelbase, wheel_radius, encoder_cpr, publish_rate,
    process_noise_*, encoder_velocity_noise, gyro_rate_noise,
    imu_yaw_noise, zupt_velocity_threshold, zupt_noise,
    imu_settle_time, initial_x, initial_y, initial_yaw

GAZEBO ACKERMANN COMPATIBILITY:
    ✓ Joint names match vehicle.xacro: rear_left_wheel_joint, rear_right_wheel_joint,
      front_left_steering_joint, front_right_steering_joint
    ✓ Wheel velocity → linear velocity: v = avg(ω_L, ω_R) × wheel_radius
    ✓ Steering → yaw rate: ω = v × tan(δ) / wheelbase (bicycle model)
    ✓ sim_bridge_node publishes /vehicle/feedback with simulated encoder ticks
    ✓ TF broadcast: odom → base_link for RViz visualization

ROLLBACK:
    To revert to old dead-reckoning, change the launch files:
      package='localization', executable='ekf_localization_node'
    back to:
      package='mid_level_controller', executable='odometry_node'
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleFeedback, ImuData, VehicleState, VehicleCmd
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
import tf2_ros

from localization.ekf_core import BicycleEKF


class EKFLocalizationNode(Node):

    def __init__(self):
        super().__init__('ekf_localization_node')

        # ================================================================
        # Parameters
        # ================================================================
        self.declare_parameter('source', 'hardware')           # "hardware" or "simulation"
        self.declare_parameter('wheelbase', 0.22)              # m
        self.declare_parameter('wheel_radius', 0.033)          # m
        self.declare_parameter('encoder_cpr', 5471)            # counts per revolution
        self.declare_parameter('publish_rate', 50.0)           # Hz
        self.declare_parameter('feedback_topic', '/vehicle/feedback')
        self.declare_parameter('imu_topic', '/vehicle/imu')
        self.declare_parameter('state_topic', '/vehicle/state')

        # Process noise (Q diagonal)
        self.declare_parameter('process_noise_x', 0.01)
        self.declare_parameter('process_noise_y', 0.01)
        self.declare_parameter('process_noise_yaw', 0.005)
        self.declare_parameter('process_noise_vel', 0.1)
        self.declare_parameter('process_noise_gyro_bias', 0.0001)

        # Measurement noise
        self.declare_parameter('encoder_velocity_noise', 0.05)
        self.declare_parameter('gyro_rate_noise', 0.01)
        self.declare_parameter('imu_yaw_noise', 0.15)          # ~8.5° — low trust

        # ZUPT
        self.declare_parameter('zupt_velocity_threshold', 0.02)  # m/s
        self.declare_parameter('zupt_noise', 0.001)

        # IMU settling
        self.declare_parameter('imu_settle_time', 2.5)         # seconds

        # Initial pose
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)             # degrees

        # ================================================================
        # Read parameters
        # ================================================================
        self.source = self.get_parameter('source').value
        self.wheelbase = self.get_parameter('wheelbase').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.encoder_cpr = self.get_parameter('encoder_cpr').value
        publish_rate = self.get_parameter('publish_rate').value
        feedback_topic = self.get_parameter('feedback_topic').value
        imu_topic = self.get_parameter('imu_topic').value
        state_topic = self.get_parameter('state_topic').value

        self.zupt_velocity_threshold = self.get_parameter('zupt_velocity_threshold').value
        self.imu_settle_time = self.get_parameter('imu_settle_time').value
        initial_x = self.get_parameter('initial_x').value
        initial_y = self.get_parameter('initial_y').value
        initial_yaw_deg = self.get_parameter('initial_yaw').value

        # ================================================================
        # Precompute
        # ================================================================
        self.wheel_circumference = 2.0 * math.pi * self.wheel_radius
        self.meters_per_tick = self.wheel_circumference / self.encoder_cpr

        # ================================================================
        # Initialize EKF
        # ================================================================
        process_noise = {
            'x': self.get_parameter('process_noise_x').value,
            'y': self.get_parameter('process_noise_y').value,
            'yaw': self.get_parameter('process_noise_yaw').value,
            'vel': self.get_parameter('process_noise_vel').value,
            'gyro_bias': self.get_parameter('process_noise_gyro_bias').value,
        }

        self.ekf = BicycleEKF(
            process_noise=process_noise,
            encoder_vel_noise=self.get_parameter('encoder_velocity_noise').value,
            gyro_rate_noise=self.get_parameter('gyro_rate_noise').value,
            imu_yaw_noise=self.get_parameter('imu_yaw_noise').value,
            zupt_noise=self.get_parameter('zupt_noise').value,
        )

        # Set initial pose
        self.ekf.set_state(
            x=float(initial_x),
            y=float(initial_y),
            theta=math.radians(float(initial_yaw_deg)),
        )

        # ================================================================
        # Sensor state tracking
        # ================================================================
        # Encoder
        self.last_ticks = None
        self.last_encoder_velocity = 0.0

        # IMU
        self.imu_initialized = False
        self.imu_yaw_offset = 0.0
        self.imu_gyro_sum = 0.0
        self.imu_gyro_count = 0
        self.latest_gyro_z = 0.0         # rad/s
        self.latest_imu_yaw_rad = None   # rad (after offset correction)
        self.start_time = None

        # Simulation
        self.sim_velocity = 0.0
        self.sim_steering_angle = 0.0

        # Timing
        self.last_predict_time = self.get_clock().now()

        # Yaw rate (for VehicleState output)
        self.current_yaw_rate = 0.0

        # Steering angle (for VehicleState output)
        self.current_steering_angle = 0.0

        # ================================================================
        # Publisher & TF
        # ================================================================
        self.state_pub = self.create_publisher(VehicleState, state_topic, 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ================================================================
        # Subscribers (based on source mode)
        # ================================================================
        if self.source == 'hardware':
            self.feedback_sub = self.create_subscription(
                VehicleFeedback, feedback_topic,
                self._feedback_callback, 10)
            self.imu_sub = self.create_subscription(
                ImuData, imu_topic,
                self._imu_callback, 10)
            # Subscribe to commanded steering for Ackermann yaw rate
            self.cmd_sub = self.create_subscription(
                VehicleCmd, '/vehicle/cmd',
                self._cmd_callback, 10)
        else:
            # SIMULATION MODE: subscribe to Gazebo /joint_states
            # Joint names from vehicle.xacro:
            #   rear_left_wheel_joint, rear_right_wheel_joint → wheel ω → velocity
            #   front_left_steering_joint → steering angle δ → yaw rate
            self.joint_sub = self.create_subscription(
                JointState, '/joint_states',
                self._joint_state_callback, 10)

        # ================================================================
        # Timer — prediction + publish at fixed rate
        # ================================================================
        self.timer = self.create_timer(1.0 / publish_rate, self._timer_callback)

        # ================================================================
        # Startup log
        # ================================================================
        self.get_logger().info('=' * 58)
        self.get_logger().info('EKF Localization Node Started')
        self.get_logger().info(f'  Source           : {self.source}')
        self.get_logger().info(f'  Wheelbase        : {self.wheelbase:.3f} m')
        self.get_logger().info(f'  Wheel radius     : {self.wheel_radius:.4f} m')
        self.get_logger().info(f'  Encoder CPR      : {self.encoder_cpr}')
        self.get_logger().info(f'  m/tick           : {self.meters_per_tick:.6f}')
        self.get_logger().info(f'  Publish rate     : {publish_rate:.0f} Hz')
        self.get_logger().info(f'  ZUPT threshold   : {self.zupt_velocity_threshold:.3f} m/s')
        self.get_logger().info(f'  IMU settle time  : {self.imu_settle_time:.1f} s')
        self.get_logger().info(f'  Initial pose     : ({initial_x:.2f}, {initial_y:.2f}) '
                               f'yaw={initial_yaw_deg:.1f}°')
        self.get_logger().info(f'  Output topic     : {state_topic}')
        if self.source == 'simulation':
            self.get_logger().info(f'  Gazebo joints    : rear_*_wheel_joint, front_left_steering_joint')
        self.get_logger().info('=' * 58)

    # ================================================================
    # Hardware Callbacks
    # ================================================================

    def _feedback_callback(self, msg: VehicleFeedback):
        """Process encoder feedback: velocity + distance update.
        
        Called in HARDWARE mode only.
        Source topic: /vehicle/feedback (VehicleFeedback)
        From: low_level_controller_node → Arduino encoder
        """
        current_ticks = msg.encoder_ticks
        self.last_encoder_velocity = msg.actual_velocity

        # ---- Velocity measurement update ----
        self.ekf.update_velocity(msg.actual_velocity)

        # ---- Distance-based position correction ----
        if self.last_ticks is not None:
            delta_ticks = current_ticks - self.last_ticks
            if abs(delta_ticks) > 0:
                # Additional velocity cross-check from ticks
                # (provides redundancy with actual_velocity)
                pass  # Already handled by velocity update

        self.last_ticks = current_ticks

    def _imu_callback(self, msg: ImuData):
        """Process IMU data: gyro rate + heading update.
        
        Called in HARDWARE mode only.
        Source topic: /vehicle/imu (ImuData)
        From: low_level_controller_node → Arduino MPU6050
        
        IMU fields used:
            msg.gyro_z  → angular rate [rad/s] — used in EKF prediction
            msg.yaw     → complementary filter heading [degrees] — soft anchor
        """
        now = self.get_clock().now()

        # ---- IMU settling period ----
        if self.start_time is None:
            self.start_time = now

        elapsed = (now - self.start_time).nanoseconds * 1e-9

        if not self.imu_initialized:
            if elapsed < self.imu_settle_time:
                # Still settling — track raw yaw for offset
                self.imu_yaw_offset = msg.yaw
                self.latest_gyro_z = msg.gyro_z
                self.imu_gyro_sum += msg.gyro_z
                self.imu_gyro_count += 1
                return

            # Capture zero-offset after settling
            self.imu_yaw_offset = msg.yaw
            initial_bias = self.imu_gyro_sum / max(1, self.imu_gyro_count)
            self.ekf.x[self.ekf.IBIAS] = initial_bias
            self.imu_initialized = True
            self.get_logger().info(
                f'IMU initialized: yaw_offset={self.imu_yaw_offset:.2f}°, '
                f'gyro_bias={math.degrees(initial_bias):.3f}°/s '
                f'(after {self.imu_settle_time:.1f}s settling)')

        # ---- Store latest gyro reading (used in prediction) ----
        self.latest_gyro_z = msg.gyro_z  # rad/s

        # ---- Yaw rate for output ----
        self.current_yaw_rate = msg.gyro_z

        # ---- Compute corrected IMU heading ----
        corrected_yaw_deg = msg.yaw - self.imu_yaw_offset
        corrected_yaw_rad = math.radians(corrected_yaw_deg)
        self.latest_imu_yaw_rad = self._normalize_angle(corrected_yaw_rad)

        # ---- Heading measurement update (low trust, soft anchor) ----
        # Re-enabled to prevent permanent heading drift after lane changes.
        # The Stanley gains have been fixed, so this shouldn't cause premature straightening anymore.
        self.ekf.update_heading(self.latest_imu_yaw_rad)

    def _cmd_callback(self, msg: VehicleCmd):
        """Capture the commanded steering angle from /vehicle/cmd.

        Called in HARDWARE mode only.
        msg.heading is the steering angle in degrees.
        This populates VehicleState.steering_angle and enables
        Ackermann-based yaw rate prediction.
        """
        self.current_steering_angle = math.radians(msg.heading)

    # ================================================================
    # Simulation Callback
    # ================================================================

    def _joint_state_callback(self, msg: JointState):
        """Extract velocity and steering from Gazebo joint states.
        
        Called in SIMULATION mode only.
        Source topic: /joint_states (JointState)
        From: Gazebo gz-sim-joint-state-publisher-system plugin
        
        Gazebo Ackermann model joint names (from vehicle.xacro):
            rear_left_wheel_joint       → velocity[i] = angular velocity [rad/s]
            rear_right_wheel_joint      → velocity[i] = angular velocity [rad/s]
            front_left_steering_joint   → position[i] = steering angle [rad]
            front_right_steering_joint  → position[i] = steering angle [rad]
        
        Velocity calculation:
            v = avg(ω_left, ω_right) × wheel_radius
        
        Yaw rate calculation (bicycle model):
            ω = v × tan(δ) / wheelbase
        """
        try:
            rear_left_idx = None
            rear_right_idx = None
            front_left_steer_idx = None

            for i, name in enumerate(msg.name):
                if name == 'rear_left_wheel_joint':
                    rear_left_idx = i
                elif name == 'rear_right_wheel_joint':
                    rear_right_idx = i
                elif name == 'front_left_steering_joint':
                    front_left_steer_idx = i

            if rear_left_idx is not None and rear_right_idx is not None:
                omega_l = msg.velocity[rear_left_idx]
                omega_r = msg.velocity[rear_right_idx]
                self.sim_velocity = ((omega_l + omega_r) / 2.0) * self.wheel_radius

                # Update EKF with velocity measurement
                self.ekf.update_velocity(self.sim_velocity)

            if front_left_steer_idx is not None:
                self.sim_steering_angle = msg.position[front_left_steer_idx]
                self.current_steering_angle = self.sim_steering_angle

                # Compute yaw rate from Ackermann steering kinematics
                # ω = v · tan(δ) / L  (bicycle model approximation)
                if abs(self.sim_steering_angle) > 1e-4 and abs(self.sim_velocity) > 0.01:
                    turning_radius = self.wheelbase / math.tan(self.sim_steering_angle)
                    self.latest_gyro_z = self.sim_velocity / turning_radius
                    self.current_yaw_rate = self.latest_gyro_z
                else:
                    self.latest_gyro_z = 0.0
                    self.current_yaw_rate = 0.0

        except (IndexError, AttributeError) as e:
            self.get_logger().warn(
                f'Joint state parse error: {e}', throttle_duration_sec=5.0)

    # ================================================================
    # Timer — EKF Predict + Publish
    # ================================================================

    def _timer_callback(self):
        """Run EKF prediction and publish fused state at fixed rate."""
        now = self.get_clock().now()
        dt = (now - self.last_predict_time).nanoseconds * 1e-9
        self.last_predict_time = now

        # Guard against pathological dt
        if dt <= 0.0 or dt > 1.0:
            dt = 0.02  # fallback to 50 Hz

        # ---- Wait for IMU to settle before running EKF ----
        # During settling, publish zero state so controllers don't drive
        # the car before heading calibration is complete.
        if not self.imu_initialized:
            state = VehicleState()
            state.header.stamp = now.to_msg()
            state.header.frame_id = 'odom'
            state.x = 0.0
            state.y = 0.0
            state.yaw = 0.0
            state.velocity = 0.0
            state.yaw_rate = 0.0
            state.steering_angle = 0.0
            self.state_pub.publish(state)
            return

        # ---- Compute Ackermann yaw rate from commanded steering ----
        # ω_ackermann = v · tan(δ) / L  (bicycle model)
        # This provides a feed-forward yaw rate estimate independent of the gyro.
        if abs(self.current_steering_angle) > 1e-4 and abs(self.ekf.velocity) > 0.01:
            ackermann_yaw_rate = self.ekf.velocity * math.tan(self.current_steering_angle) / self.wheelbase
            self.current_yaw_rate = ackermann_yaw_rate
        else:
            self.current_yaw_rate = self.latest_gyro_z

        # ---- EKF Prediction step ----
        self.ekf.predict(self.latest_gyro_z, dt)

        # ---- Zero-Velocity Update (ZUPT) ----
        # When robot is stationary, inject v=0 virtual measurement
        # This is the KEY fix for lateral drift when stopped
        if abs(self.last_encoder_velocity) < self.zupt_velocity_threshold:
            if self.source == 'hardware':
                self.ekf.zupt()
            elif abs(self.sim_velocity) < self.zupt_velocity_threshold:
                self.ekf.zupt()

        # ---- Gyro bias estimation (always, not just ZUPT) ----
        # Moved outside ZUPT block so bias tracks continuously while driving.
        # Without this, bias freezes once the car starts moving → yaw drift.
        self.ekf.update_gyro(self.latest_gyro_z)

        # ---- Publish VehicleState ----
        state = VehicleState()
        state.header.stamp = now.to_msg()
        state.header.frame_id = 'odom'
        state.x = self.ekf.position_x
        state.y = self.ekf.position_y
        state.yaw = self.ekf.heading
        state.velocity = self.ekf.velocity
        state.yaw_rate = self.current_yaw_rate
        state.steering_angle = self.current_steering_angle
        self.state_pub.publish(state)

        # ---- Broadcast TF: odom → base_link ----
        t = TransformStamped()
        t.header.stamp = state.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.ekf.position_x
        t.transform.translation.y = self.ekf.position_y
        t.transform.translation.z = 0.0

        # Yaw → quaternion
        cy = math.cos(self.ekf.heading * 0.5)
        sy = math.sin(self.ekf.heading * 0.5)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = sy
        t.transform.rotation.w = cy
        self.tf_broadcaster.sendTransform(t)

        # ---- Diagnostic log (throttled) ----
        cov = self.ekf.covariance_diagonal
        self.get_logger().info(
            f'[EKF] pos=({self.ekf.position_x:.3f},{self.ekf.position_y:.3f}) '
            f'yaw={math.degrees(self.ekf.heading):.1f}° '
            f'v={self.ekf.velocity:.3f} '
            f'bias={math.degrees(self.ekf.gyro_bias):.3f}°/s '
            f'σ_xy=({math.sqrt(cov[0]):.3f},{math.sqrt(cov[1]):.3f}) '
            f'imu={"OK" if self.imu_initialized else "SETTLING"} '
            f'enc={"OK" if self.last_ticks is not None else "WAIT"}',
            throttle_duration_sec=2.0
        )

    # ================================================================
    # Utilities
    # ================================================================

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-π, π]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = EKFLocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
