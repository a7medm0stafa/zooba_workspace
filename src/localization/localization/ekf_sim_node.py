"""
EKF Simulation Node — Kalman Filter for Noisy Simulated Sensors
================================================================
FILE: localization/localization/ekf_sim_node.py

PURPOSE:
    Extended Kalman Filter wrapper for the *simulation* noisy-sensor
    pipeline.  Unlike ekf_localization_node.py (which either fuses
    hardware sensors OR reads Gazebo joint states), this node:

        1. Subscribes to /vehicle/state_noisy  — the corrupted pose
           produced by sensor_noise_node.py
        2. Subscribes to /joint_states         — for clean velocity +
           steering angle (the "actuator" model input)
        3. Runs the BicycleEKF predict/update cycle at a fixed rate
        4. Publishes the filtered estimate to /vehicle/state

    This mimics what an EKF would do on the real robot when all it has
    is a GPS/IMU (position + yaw, noisy) and wheel encoders (velocity).

SIGNAL FLOW:
    Gazebo   ──▶  ground_truth_node  ──▶  /vehicle/state_gt
    /vehicle/state_gt  ──▶  sensor_noise_node  ──▶  /vehicle/state_noisy
    /joint_states (Gazebo)  ──┐
    /vehicle/state_noisy    ──┴──▶  EKFSimNode  ──▶  /vehicle/state

EKF MEASUREMENT SOURCES:
    • Velocity  : from /joint_states (wheel ω → v, low noise)
    • Position  : from /vehicle/state_noisy .x / .y  (GPS-like, σ_pos)
    • Heading   : from /vehicle/state_noisy .yaw      (compass, σ_yaw)
    • Gyro rate : computed from Ackermann kinematics (joint_states)

STATE VECTOR (5×1):
    [x, y, θ, v, ω_bias]  — same as BicycleEKF

PARAMETERS:
    wheelbase            (float) — vehicle wheelbase [m]
    wheel_radius         (float) — wheel radius [m]
    publish_rate         (float) — EKF update rate [Hz]
    noisy_state_topic    (str)   — input: noisy state topic
    state_topic          (str)   — output: filtered state topic

    # Process noise (Q diagonal)
    process_noise_x      (float)
    process_noise_y      (float)
    process_noise_yaw    (float)
    process_noise_vel    (float)
    process_noise_gyro_bias (float)

    # Measurement noise (R values — tune to match SensorNoiseNode sigmas)
    meas_noise_position  (float) — x, y measurement noise [m]
    meas_noise_yaw       (float) — heading measurement noise [rad]
    meas_noise_velocity  (float) — velocity measurement noise [m/s]

    # ZUPT
    zupt_velocity_threshold (float) — speed below which ZUPT activates [m/s]
    zupt_noise           (float)

    # Initial pose
    initial_x            (float)
    initial_y            (float)
    initial_yaw          (float) — degrees
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleState
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
import tf2_ros

from localization.ekf_core import BicycleEKF


class EKFSimNode(Node):
    """EKF for simulation: fuses noisy state + Gazebo joint velocities."""

    def __init__(self):
        super().__init__('ekf_sim_node')

        # ================================================================
        # Parameters
        # ================================================================
        self.declare_parameter('wheelbase',           0.22)
        self.declare_parameter('wheel_radius',        0.033)
        self.declare_parameter('publish_rate',        50.0)
        self.declare_parameter('noisy_state_topic',   '/vehicle/state_noisy')
        self.declare_parameter('state_topic',         '/vehicle/state')

        # Process noise (Q diagonal)
        self.declare_parameter('process_noise_x',          0.001)
        self.declare_parameter('process_noise_y',          0.001)
        self.declare_parameter('process_noise_yaw',        0.001)
        self.declare_parameter('process_noise_vel',        0.001)
        self.declare_parameter('process_noise_gyro_bias',  0.0001)

        # Measurement noise (tune to match SensorNoiseNode sigmas)
        self.declare_parameter('meas_noise_position',  0.001)   # m
        self.declare_parameter('meas_noise_yaw',       0.01)   # rad
        self.declare_parameter('meas_noise_velocity',  0.01)   # m/s

        # ZUPT
        self.declare_parameter('zupt_velocity_threshold', 0.02)
        self.declare_parameter('zupt_noise',               0.001)

        # Initial pose
        self.declare_parameter('initial_x',   0.0)
        self.declare_parameter('initial_y',   0.0)
        self.declare_parameter('initial_yaw', 0.0)  # degrees

        # ================================================================
        # Read parameters
        # ================================================================
        self.wheelbase    = self.get_parameter('wheelbase').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        publish_rate      = self.get_parameter('publish_rate').value
        noisy_topic       = self.get_parameter('noisy_state_topic').value
        state_topic       = self.get_parameter('state_topic').value

        self.zupt_threshold = self.get_parameter('zupt_velocity_threshold').value

        meas_pos  = self.get_parameter('meas_noise_position').value
        meas_yaw  = self.get_parameter('meas_noise_yaw').value
        meas_vel  = self.get_parameter('meas_noise_velocity').value

        initial_x   = self.get_parameter('initial_x').value
        initial_y   = self.get_parameter('initial_y').value
        initial_yaw = math.radians(self.get_parameter('initial_yaw').value)

        # ================================================================
        # Build extra measurement noise matrices (position + yaw)
        # We extend the existing BicycleEKF interface with custom updates
        # ================================================================
        self.R_position = np.eye(2) * (meas_pos ** 2)   # 2×2 for [x, y]
        self.R_yaw      = np.array([[meas_yaw ** 2]])    # 1×1 for [θ]
        self.R_velocity = np.array([[meas_vel ** 2]])    # 1×1 for [v]

        # ================================================================
        # Initialize BicycleEKF
        # ================================================================
        process_noise = {
            'x':          self.get_parameter('process_noise_x').value,
            'y':          self.get_parameter('process_noise_y').value,
            'yaw':        self.get_parameter('process_noise_yaw').value,
            'vel':        self.get_parameter('process_noise_vel').value,
            'gyro_bias':  self.get_parameter('process_noise_gyro_bias').value,
        }

        self.ekf = BicycleEKF(
            process_noise=process_noise,
            encoder_vel_noise=meas_vel,
            gyro_rate_noise=0.01,           # kinematic gyro — quite accurate
            imu_yaw_noise=meas_yaw,
            zupt_noise=self.get_parameter('zupt_noise').value,
        )
        self.ekf.set_state(x=initial_x, y=initial_y, theta=initial_yaw)

        # ================================================================
        # Runtime state
        # ================================================================
        self.sim_velocity       = 0.0
        self.sim_steering_angle = 0.0
        self.current_yaw_rate   = 0.0
        self.latest_gyro_z      = 0.0

        self.noisy_state_received = False
        self.last_noisy: VehicleState | None = None

        self.last_predict_time = self.get_clock().now()

        # ================================================================
        # Publisher + TF
        # ================================================================
        self.state_pub      = self.create_publisher(VehicleState, state_topic, 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ================================================================
        # Subscribers
        # ================================================================
        # Noisy state — provides position, heading as pseudo-GPS/compass
        self.noisy_sub = self.create_subscription(
            VehicleState, noisy_topic,
            self._noisy_state_callback, 10)

        # Gazebo joint states — provides clean wheel velocity + steering
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states',
            self._joint_state_callback, 10)

        # ================================================================
        # Timer
        # ================================================================
        self.timer = self.create_timer(
            1.0 / publish_rate, self._timer_callback)

        # ================================================================
        # Startup log
        # ================================================================
        self.get_logger().info('=' * 62)
        self.get_logger().info('EKF Simulation Node Started')
        self.get_logger().info(f'  Noisy input  : {noisy_topic}')
        self.get_logger().info(f'  State output : {state_topic}')
        self.get_logger().info(f'  Wheelbase    : {self.wheelbase:.3f} m')
        self.get_logger().info(f'  Wheel radius : {self.wheel_radius:.4f} m')
        self.get_logger().info(f'  Pub rate     : {publish_rate:.0f} Hz')
        self.get_logger().info(f'  Meas σ_pos   : {meas_pos:.4f} m')
        self.get_logger().info(f'  Meas σ_yaw   : {math.degrees(meas_yaw):.2f}°')
        self.get_logger().info(f'  Meas σ_vel   : {meas_vel:.4f} m/s')
        self.get_logger().info(f'  ZUPT thresh  : {self.zupt_threshold:.3f} m/s')
        self.get_logger().info(f'  Init pose    : ({initial_x:.2f},{initial_y:.2f}) '
                               f'yaw={math.degrees(initial_yaw):.1f}°')
        self.get_logger().info('=' * 62)

    # ================================================================
    # Callbacks
    # ================================================================

    def _noisy_state_callback(self, msg: VehicleState):
        """Store the latest noisy state for use in the next EKF update."""
        self.last_noisy = msg

        if not self.noisy_state_received:
            self.get_logger().info(
                f'[EKF-Sim] First noisy state: '
                f'({msg.x:.3f},{msg.y:.3f}) yaw={math.degrees(msg.yaw):.1f}°')
            # Warm-start EKF from the first noisy measurement
            self.ekf.set_state(x=float(msg.x), y=float(msg.y),
                               theta=float(msg.yaw))
            self.noisy_state_received = True

    def _joint_state_callback(self, msg: JointState):
        """Extract velocity and steering angle from Gazebo joint states."""
        try:
            rear_left_idx       = None
            rear_right_idx      = None
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

                # Velocity measurement update — wheel odometry is relatively clean
                self.ekf.update_velocity(self.sim_velocity)

            if front_left_steer_idx is not None:
                self.sim_steering_angle = msg.position[front_left_steer_idx]

                # Compute kinematic yaw rate (bicycle model)
                if (abs(self.sim_steering_angle) > 1e-4
                        and abs(self.sim_velocity) > 0.01):
                    turning_radius = (self.wheelbase
                                      / math.tan(self.sim_steering_angle))
                    self.latest_gyro_z = self.sim_velocity / turning_radius
                    self.current_yaw_rate = self.latest_gyro_z
                else:
                    self.latest_gyro_z  = 0.0
                    self.current_yaw_rate = 0.0

        except (IndexError, AttributeError) as e:
            self.get_logger().warn(
                f'Joint state parse error: {e}', throttle_duration_sec=5.0)

    # ================================================================
    # Timer — EKF predict + update + publish
    # ================================================================

    def _timer_callback(self):
        """Main EKF loop: predict, update from noisy state, publish."""
        if not self.noisy_state_received:
            return

        now = self.get_clock().now()
        dt  = (now - self.last_predict_time).nanoseconds * 1e-9
        self.last_predict_time = now

        # Guard against bad dt
        if dt <= 0.0 or dt > 1.0:
            dt = 0.02  # 50 Hz fallback

        # ---- EKF Prediction (bicycle kinematics, driven by kinematic gyro) ----
        self.ekf.predict(self.latest_gyro_z, dt)

        # ---- Position + Heading updates from noisy state (GPS/compass model) ----
        if self.last_noisy is not None:
            self._update_position(self.last_noisy.x, self.last_noisy.y)
            self._update_heading_noisy(self.last_noisy.yaw)

        # ---- ZUPT when nearly stopped ----
        if abs(self.sim_velocity) < self.zupt_threshold:
            self.ekf.zupt()

        # ---- Ackermann yaw rate for output ----
        if (abs(self.sim_steering_angle) > 1e-4
                and abs(self.ekf.velocity) > 0.01):
            self.current_yaw_rate = (self.ekf.velocity
                                     * math.tan(self.sim_steering_angle)
                                     / self.wheelbase)
        else:
            self.current_yaw_rate = self.latest_gyro_z

        # ---- Publish filtered VehicleState ----
        state = VehicleState()
        state.header.stamp    = now.to_msg()
        state.header.frame_id = 'odom'
        state.x               = self.ekf.position_x
        state.y               = self.ekf.position_y
        state.yaw             = self.ekf.heading
        state.velocity        = self.ekf.velocity
        state.yaw_rate        = self.current_yaw_rate
        state.steering_angle  = self.sim_steering_angle
        self.state_pub.publish(state)

        # ---- Broadcast TF odom → base_link ----
        t = TransformStamped()
        t.header.stamp    = state.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id  = 'base_link'
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

        # ---- Diagnostic log (throttled to 0.5 Hz) ----
        cov = self.ekf.covariance_diagonal
        self.get_logger().info(
            f'[EKF-Sim] pos=({self.ekf.position_x:.3f},{self.ekf.position_y:.3f}) '
            f'yaw={math.degrees(self.ekf.heading):.1f}° '
            f'v={self.ekf.velocity:.3f} '
            f'σ_xy=({math.sqrt(max(cov[0],0)):.3f},'
            f'{math.sqrt(max(cov[1],0)):.3f})',
            throttle_duration_sec=2.0)

    # ================================================================
    # Custom EKF Measurement Updates
    # ================================================================

    def _update_position(self, x_meas: float, y_meas: float):
        """Position measurement update from noisy GPS-like source.

        Measurement model: z = [x_meas, y_meas], h(x) = [x, y]
        """
        H = np.zeros((2, self.ekf.N_STATES))
        H[0, self.ekf.IX] = 1.0
        H[1, self.ekf.IY] = 1.0

        z      = np.array([x_meas, y_meas])
        z_pred = np.array([self.ekf.position_x, self.ekf.position_y])

        self.ekf._ekf_update(z, z_pred, H, self.R_position)

    def _update_heading_noisy(self, theta_meas: float):
        """Heading measurement update from noisy compass source.

        Measurement model: z = [θ_meas], h(x) = [θ]
        Uses EKF's built-in heading update with the noisy yaw noise.
        """
        # Temporarily patch the EKF's R_yaw so we use our configured value
        original_R_yaw = self.ekf.R_yaw
        self.ekf.R_yaw = self.R_yaw
        self.ekf.update_heading(theta_meas)
        self.ekf.R_yaw = original_R_yaw

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
    node = EKFSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
