"""
EKF Localization Node — Fused Encoder + IMU State Estimation
==============================================================
FILE:    localization/localization/ekf_localization_node.py
STATUS:  COMPLETE REWRITE — replaces the broken 5-state EKF node
CREATED: 2026-05-17

PURPOSE:
    ROS2 node that runs a 4-state EKF fusing encoder velocity and
    IMU gyroscope data to produce a drift-resistant vehicle pose.

ARCHITECTURE:
    - Prediction (timer-driven, 50 Hz): propagates state using gyro ω_z
    - Encoder update (event-driven): velocity measurement on callback
    - ZUPT (conditional): when encoder reports near-zero velocity
    - Heading update (DISABLED by default): Arduino yaw is pure gyro
      integration — double-counts gyro data. Gated behind parameter.

MODES:
    1. HARDWARE: subscribes to /vehicle/feedback + /vehicle/imu
    2. SIMULATION: subscribes to /joint_states (Gazebo Ackermann model)

PUBLISHES:
    /vehicle/state  (VehicleState)  — fused state estimate
    TF: odom → base_link            — transform for RViz

ALL INTERNAL UNITS: meters, radians, seconds.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleFeedback, ImuData, VehicleState, VehicleCmd
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
import tf2_ros

from localization.ekf_core import EKF2D


class EKFLocalizationNode(Node):

    def __init__(self):
        super().__init__('ekf_localization_node')

        # ================================================================
        # Parameters
        # ================================================================
        self.declare_parameter('source', 'hardware')
        self.declare_parameter('wheelbase', 0.265)
        self.declare_parameter('wheel_radius', 0.033)
        self.declare_parameter('encoder_cpr', 5471)
        self.declare_parameter('gear_ratio', 124.333)
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('feedback_topic', '/vehicle/feedback')
        self.declare_parameter('imu_topic', '/vehicle/imu')
        self.declare_parameter('state_topic', '/vehicle/state')

        # Process noise (continuous-time standard deviations)
        self.declare_parameter('sigma_velocity', 0.1)
        self.declare_parameter('sigma_yaw_rate', 0.02)
        self.declare_parameter('sigma_bias', 0.005)

        # Measurement noise (standard deviations)
        self.declare_parameter('sigma_encoder', 0.05)
        self.declare_parameter('sigma_imu_heading', 0.20)

        # ZUPT
        self.declare_parameter('zupt_velocity_threshold', 0.02)
        self.declare_parameter('sigma_zupt', 0.001)

        # IMU
        self.declare_parameter('imu_settle_time', 5.0)
        self.declare_parameter('use_imu_heading', False)
        self.declare_parameter('heading_update_divisor', 10)

        # Ackermann heading correction (fights dynamic gyro bias)
        self.declare_parameter('use_ackermann_heading', True)
        self.declare_parameter('sigma_ackermann_heading', 0.3)
        self.declare_parameter('ackermann_min_velocity', 0.03)

        # Initial pose
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)  # degrees (user convenience)

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
        self.use_imu_heading = self.get_parameter('use_imu_heading').value
        self.heading_update_divisor = self.get_parameter('heading_update_divisor').value

        # Ackermann heading correction
        self.use_ackermann_heading = self.get_parameter('use_ackermann_heading').value
        self.R_ackermann = self.get_parameter('sigma_ackermann_heading').value ** 2
        self.ackermann_min_velocity = self.get_parameter('ackermann_min_velocity').value

        sigma_v = self.get_parameter('sigma_velocity').value
        sigma_omega = self.get_parameter('sigma_yaw_rate').value
        sigma_bias = self.get_parameter('sigma_bias').value
        self.R_encoder = self.get_parameter('sigma_encoder').value ** 2
        self.R_heading = self.get_parameter('sigma_imu_heading').value ** 2
        self.R_zupt = self.get_parameter('sigma_zupt').value ** 2

        initial_x = float(self.get_parameter('initial_x').value)
        initial_y = float(self.get_parameter('initial_y').value)
        initial_yaw_deg = float(self.get_parameter('initial_yaw').value)

        # ================================================================
        # Precompute
        # ================================================================
        self.wheel_circumference = 2.0 * math.pi * self.wheel_radius
        self.meters_per_tick = self.wheel_circumference / self.encoder_cpr

        # ================================================================
        # Initialize EKF
        # ================================================================
        self.ekf = EKF2D(sigma_v=sigma_v, sigma_omega=sigma_omega,
                         sigma_bias=sigma_bias)
        self.ekf.set_initial_state(
            x=initial_x,
            y=initial_y,
            theta=math.radians(initial_yaw_deg),
        )

        # ================================================================
        # Sensor state tracking
        # ================================================================
        # Encoder
        self.last_ticks = None
        self.last_encoder_velocity = 0.0

        # IMU — all stored in SI units (rad/s, rad)
        self.imu_initialized = False
        self.imu_yaw_offset_rad = 0.0
        self.latest_gyro_z = 0.0          # rad/s (raw from LLC, before bias correction)
        self.latest_imu_yaw_rad = None    # rad (after offset correction)
        self.start_time = None

        # Gyro bias estimation
        # Static calibration during 5-second settle period only
        self.gyro_z_settle_samples = []   # accumulate gyro_z during settling
        self.gyro_z_bias = 0.0            # estimated bias [rad/s]

        # Simulation
        self.sim_velocity = 0.0
        self.sim_steering_angle = 0.0

        # Timing
        self.last_predict_time = self.get_clock().now()
        self.timer_tick_count = 0

        # Output fields
        self.current_yaw_rate = 0.0
        self.current_steering_angle = 0.0
        self.commanded_steering_rad = 0.0  # raw commanded steering for Ackermann

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
            self.cmd_sub = self.create_subscription(
                VehicleCmd, '/vehicle/cmd',
                self._cmd_callback, 10)
        else:
            self.joint_sub = self.create_subscription(
                JointState, '/joint_states',
                self._joint_state_callback, 10)
            # Simulation doesn't need IMU settling
            self.imu_initialized = True

        # ================================================================
        # Timer — prediction + publish at fixed rate
        # ================================================================
        self.timer = self.create_timer(1.0 / publish_rate, self._timer_callback)

        # ================================================================
        # Startup log
        # ================================================================
        self.get_logger().info('=' * 60)
        self.get_logger().info('EKF Localization Node v3.0 (5-state, bias estimation)')
        self.get_logger().info(f'  Source            : {self.source}')
        self.get_logger().info(f'  Wheelbase         : {self.wheelbase:.3f} m')
        self.get_logger().info(f'  Wheel radius      : {self.wheel_radius:.4f} m')
        self.get_logger().info(f'  Encoder CPR       : {self.encoder_cpr}')
        self.get_logger().info(f'  m/tick            : {self.meters_per_tick:.6f}')
        self.get_logger().info(f'  Publish rate      : {publish_rate:.0f} Hz')
        self.get_logger().info(f'  σ_velocity        : {sigma_v}')
        self.get_logger().info(f'  σ_yaw_rate        : {sigma_omega}')
        self.get_logger().info(f'  σ_bias            : {sigma_bias}')
        self.get_logger().info(f'  σ_encoder         : {math.sqrt(self.R_encoder):.4f}')
        self.get_logger().info(f'  ZUPT threshold    : {self.zupt_velocity_threshold:.3f} m/s')
        self.get_logger().info(f'  IMU heading update: {"ENABLED" if self.use_imu_heading else "DISABLED"}')
        self.get_logger().info(f'  Ackermann heading : {"ENABLED" if self.use_ackermann_heading else "DISABLED"}')
        if self.use_ackermann_heading:
            self.get_logger().info(f'    σ_ackermann     : {math.sqrt(self.R_ackermann):.3f} rad')
            self.get_logger().info(f'    min velocity    : {self.ackermann_min_velocity:.3f} m/s')
        self.get_logger().info(f'  IMU settle time   : {self.imu_settle_time:.1f} s')
        self.get_logger().info(f'  Initial pose      : ({initial_x:.2f}, {initial_y:.2f}) '
                               f'yaw={initial_yaw_deg:.1f}°')
        self.get_logger().info(f'  Output topic      : {state_topic}')
        self.get_logger().info('=' * 60)

    # ================================================================
    # Hardware Callbacks
    # ================================================================

    def _feedback_callback(self, msg: VehicleFeedback):
        """Process encoder feedback: velocity measurement update.

        Called in HARDWARE mode only.
        Source: /vehicle/feedback (VehicleFeedback) from LLC → Arduino encoder.
        """
        self.last_encoder_velocity = msg.actual_velocity

        # Event-driven velocity update — happens immediately on arrival
        if self.imu_initialized:
            self.ekf.update_velocity(msg.actual_velocity, self.R_encoder)

        self.last_ticks = msg.encoder_ticks

    def _imu_callback(self, msg: ImuData):
        """Process IMU data: store gyro_z and heading.

        Called in HARDWARE mode only.
        Source: /vehicle/imu (ImuData) from LLC → Arduino MPU6050.

        Units from firmware:
            msg.gyro_z  → rad/s (already bias-corrected by Arduino)
            msg.yaw     → degrees (pure gyro integration on Arduino)
        """
        now = self.get_clock().now()

        # ---- IMU settling period ----
        if self.start_time is None:
            self.start_time = now

        elapsed = (now - self.start_time).nanoseconds * 1e-9

        if not self.imu_initialized:
            if elapsed < self.imu_settle_time:
                # Still settling — accumulate gyro_z samples for bias estimation
                self.imu_yaw_offset_rad = math.radians(msg.yaw)
                self.latest_gyro_z = msg.gyro_z
                self.gyro_z_settle_samples.append(msg.gyro_z)
                return

            # Settling complete — compute gyro bias + yaw offset
            self.imu_yaw_offset_rad = math.radians(msg.yaw)
            if len(self.gyro_z_settle_samples) > 10:
                self.gyro_z_bias = float(np.mean(self.gyro_z_settle_samples))
            self.imu_initialized = True
            self.get_logger().info(
                f'IMU initialized: yaw_offset={math.degrees(self.imu_yaw_offset_rad):.2f}° '
                f'gyro_z_bias={math.degrees(self.gyro_z_bias):.3f}°/s '
                f'({len(self.gyro_z_settle_samples)} samples) '
                f'(after {self.imu_settle_time:.1f}s settling)')
            self.gyro_z_settle_samples = []  # free memory

        # ---- Store latest gyro reading (used in prediction) ----
        # Raw from LLC — we subtract our own bias in the timer callback
        self.latest_gyro_z = msg.gyro_z

        # ---- Yaw rate for VehicleState output (bias-corrected) ----
        self.current_yaw_rate = msg.gyro_z - self.gyro_z_bias

        # ---- Compute corrected IMU heading (single deg→rad conversion) ----
        raw_yaw_rad = math.radians(msg.yaw)
        corrected_yaw_rad = raw_yaw_rad - self.imu_yaw_offset_rad
        self.latest_imu_yaw_rad = EKF2D._normalize_angle(corrected_yaw_rad)

    def _cmd_callback(self, msg: VehicleCmd):
        """Capture commanded steering angle from /vehicle/cmd.

        Called in HARDWARE mode only.
        msg.heading is in degrees — convert to radians at input boundary.
        """
        self.current_steering_angle = math.radians(msg.heading)
        self.commanded_steering_rad = self.current_steering_angle

    # ================================================================
    # Simulation Callback
    # ================================================================

    def _joint_state_callback(self, msg: JointState):
        """Extract velocity and steering from Gazebo joint states.

        Called in SIMULATION mode only.
        Source: /joint_states (JointState) from Gazebo.

        Joint names (from vehicle.xacro):
            rear_left_wheel_joint       → velocity = angular vel [rad/s]
            rear_right_wheel_joint      → velocity = angular vel [rad/s]
            front_left_steering_joint   → position = steering angle [rad]
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
                self.last_encoder_velocity = self.sim_velocity

                # Event-driven velocity update
                self.ekf.update_velocity(self.sim_velocity, self.R_encoder)

            if front_left_steer_idx is not None:
                self.sim_steering_angle = msg.position[front_left_steer_idx]
                self.current_steering_angle = self.sim_steering_angle
                self.commanded_steering_rad = self.sim_steering_angle

                # Compute yaw rate from Ackermann kinematics: ω = v·tan(δ)/L
                if abs(self.sim_steering_angle) > 1e-4 and abs(self.sim_velocity) > 0.01:
                    self.latest_gyro_z = (self.sim_velocity
                                          * math.tan(self.sim_steering_angle)
                                          / self.wheelbase)
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

        # ---- Wait for IMU to settle ----
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

        self.timer_tick_count += 1

        # ---- Gyro for prediction (EKF subtracts its own bias estimate) ----
        # Pass the LLC-bias-corrected gyro. The EKF has a 5th state (b_ω)
        # that estimates and removes any remaining dynamic bias from
        # motor vibrations. No need for manual bias correction here.
        corrected_gyro_z = self.latest_gyro_z - self.gyro_z_bias

        # ---- EKF Prediction step ----
        # Heading: θ' = θ + (ω_gyro - b_ω_estimated) · dt
        # The EKF now estimates b_ω online, so the gyro detects real
        # turns while vibration bias is automatically removed.
        self.ekf.predict(corrected_gyro_z, dt)

        # ---- Zero-Velocity Update (ZUPT) ----
        is_stationary = False
        if self.source == 'hardware':
            is_stationary = abs(self.last_encoder_velocity) < self.zupt_velocity_threshold
        else:
            is_stationary = abs(self.sim_velocity) < self.zupt_velocity_threshold

        if is_stationary:
            self.ekf.zupt(self.R_zupt)

        # ---- Ackermann heading measurement ----
        # Cross-checks gyro heading against steering geometry.
        # The innovation drives BOTH heading correction AND bias learning.
        # This is what makes the gyro bias observable.
        if self.use_ackermann_heading:
            self.ekf.update_heading_from_ackermann(
                steering_angle=self.commanded_steering_rad,
                wheelbase=self.wheelbase,
                dt=dt,
                R=self.R_ackermann,
            )

        # ---- IMU heading update (DISABLED by default) ----
        if (self.use_imu_heading
                and self.latest_imu_yaw_rad is not None
                and self.timer_tick_count % self.heading_update_divisor == 0):
            self.ekf.update_heading(self.latest_imu_yaw_rad, self.R_heading)

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

        cy = math.cos(self.ekf.heading * 0.5)
        sy = math.sin(self.ekf.heading * 0.5)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = sy
        t.transform.rotation.w = cy
        self.tf_broadcaster.sendTransform(t)

        # ---- Diagnostic log (throttled to 1 Hz) ----
        cov = self.ekf.covariance_diagonal
        imu_yaw_deg = (math.degrees(self.latest_imu_yaw_rad)
                       if self.latest_imu_yaw_rad is not None else 0.0)

        self.get_logger().info(
            f'[RAW ] gyro_z={math.degrees(self.latest_gyro_z):.3f}°/s '
            f'bias={math.degrees(self.gyro_z_bias):.3f}°/s '
            f'corrected={math.degrees(self.latest_gyro_z - self.gyro_z_bias):.3f}°/s '
            f'imu_yaw={imu_yaw_deg:.2f}° '
            f'enc_vel={self.last_encoder_velocity:.3f}m/s '
            f'steer={math.degrees(self.current_steering_angle):.2f}°',
            throttle_duration_sec=1.0
        )

        self.get_logger().info(
            f'[EKF ] pos=({self.ekf.position_x:.3f},{self.ekf.position_y:.3f}) '
            f'yaw={math.degrees(self.ekf.heading):.2f}° '
            f'v={self.ekf.velocity:.3f} '
            f'gyro_bias={math.degrees(self.ekf.gyro_bias):.3f}°/s '
            f'σ=({math.sqrt(max(0, cov[0])):.4f},'
            f'{math.sqrt(max(0, cov[1])):.4f},'
            f'{math.sqrt(max(0, cov[2])):.4f})',
            throttle_duration_sec=1.0
        )


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
