"""
Sensor Noise Node — Gaussian Noise Injection for Simulation
=============================================================
FILE: localization/localization/sensor_noise_node.py

PURPOSE:
    Simulates realistic sensor noise on the ground-truth vehicle state
    produced by Gazebo.  The clean perfect state from /vehicle/state_gt
    (or any source topic) is corrupted with independent Gaussian noise
    on every field and re-published as /vehicle/state_noisy.

    The noisy topic is then consumed by the EKF simulation node
    (ekf_sim_node.py) to produce a filtered, realistic state estimate.

SIGNAL FLOW:
    Gazebo perfect pose  ──▶  ground_truth_node  ──▶  /vehicle/state_gt
    /vehicle/state_gt    ──▶  SensorNoiseNode     ──▶  /vehicle/state_noisy
    /vehicle/state_noisy ──▶  EKFSimNode          ──▶  /vehicle/state

NOISE MODEL:
    Each VehicleState field is corrupted independently with zero-mean
    additive Gaussian noise:

        x_noisy         = x_gt    + N(0, σ_pos²)
        y_noisy         = y_gt    + N(0, σ_pos²)
        yaw_noisy       = yaw_gt  + N(0, σ_yaw²)
        velocity_noisy  = v_gt    + N(0, σ_vel²)
        yaw_rate_noisy  = ω_gt    + N(0, σ_yaw_rate²)
        steering_noisy  = δ_gt    + N(0, σ_steer²)

    Defaults match typical low-cost automotive sensors:
        σ_pos       ≈ 0.05 m    (5 cm GPS-grade position noise)
        σ_yaw       ≈ 0.02 rad  (≈ 1.1° heading noise)
        σ_vel       ≈ 0.05 m/s  (encoder noise)
        σ_yaw_rate  ≈ 0.01 rad/s (gyroscope noise)
        σ_steer     ≈ 0.01 rad  (≈ 0.6° steering encoder noise)

PARAMETERS:
    input_topic    (str)   — clean ground-truth state topic
    output_topic   (str)   — noisy state topic
    sigma_position (float) — std dev for x, y [m]
    sigma_yaw      (float) — std dev for yaw [rad]
    sigma_velocity (float) — std dev for velocity [m/s]
    sigma_yaw_rate (float) — std dev for yaw_rate [rad/s]
    sigma_steering (float) — std dev for steering_angle [rad]
    seed           (int)   — RNG seed (-1 = random)

USAGE:
    Loaded automatically by closed_loop_sim_track_noisy.launch.py.
    Can also be run standalone:
        ros2 run localization sensor_noise_node \\
            --ros-args -p sigma_position:=0.05 -p sigma_yaw:=0.02
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleState


class SensorNoiseNode(Node):
    """Adds configurable Gaussian noise to ground-truth VehicleState."""

    def __init__(self):
        super().__init__('sensor_noise_node')

        # ================================================================
        # Parameters
        # ================================================================
        self.declare_parameter('input_topic',    '/vehicle/state_gt')
        self.declare_parameter('output_topic',   '/vehicle/state_noisy')

        # Noise standard deviations (σ)
        self.declare_parameter('sigma_position', 0.05)   # m  — x, y
        self.declare_parameter('sigma_yaw',      0.01)   # rad — heading
        self.declare_parameter('sigma_velocity', 0.05)   # m/s
        self.declare_parameter('sigma_yaw_rate', 0.01)   # rad/s
        self.declare_parameter('sigma_steering', 0.01)   # rad

        # RNG seed (-1 = non-deterministic)
        self.declare_parameter('seed', -1)

        # ================================================================
        # Read parameters
        # ================================================================
        input_topic  = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.sigma_pos    = self.get_parameter('sigma_position').value
        self.sigma_yaw    = self.get_parameter('sigma_yaw').value
        self.sigma_vel    = self.get_parameter('sigma_velocity').value
        self.sigma_yr     = self.get_parameter('sigma_yaw_rate').value
        self.sigma_steer  = self.get_parameter('sigma_steering').value

        seed = self.get_parameter('seed').value
        if seed < 0:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(int(seed))

        # ================================================================
        # Pub / Sub
        # ================================================================
        self.sub = self.create_subscription(
            VehicleState, input_topic,
            self._state_callback, 10)

        self.pub = self.create_publisher(
            VehicleState, output_topic, 10)

        # ================================================================
        # Startup log
        # ================================================================
        self.get_logger().info('=' * 60)
        self.get_logger().info('Sensor Noise Node Started')
        self.get_logger().info(f'  Input  : {input_topic}')
        self.get_logger().info(f'  Output : {output_topic}')
        self.get_logger().info(f'  σ_pos  : {self.sigma_pos:.4f} m')
        self.get_logger().info(f'  σ_yaw  : {math.degrees(self.sigma_yaw):.2f}°  '
                               f'({self.sigma_yaw:.4f} rad)')
        self.get_logger().info(f'  σ_vel  : {self.sigma_vel:.4f} m/s')
        self.get_logger().info(f'  σ_yr   : {self.sigma_yr:.4f} rad/s')
        self.get_logger().info(f'  σ_steer: {math.degrees(self.sigma_steer):.2f}°  '
                               f'({self.sigma_steer:.4f} rad)')
        self.get_logger().info(f'  RNG seed: {"random" if seed < 0 else seed}')
        self.get_logger().info('=' * 60)

    # ================================================================
    # Callback
    # ================================================================

    def _state_callback(self, msg: VehicleState):
        """Corrupt the incoming perfect state and re-publish it."""
        noisy = VehicleState()
        noisy.header.stamp    = msg.header.stamp   # preserve original timestamp
        noisy.header.frame_id = msg.header.frame_id

        # Independent Gaussian noise on each field
        noisy.x               = msg.x + self._noise(self.sigma_pos)
        noisy.y               = msg.y + self._noise(self.sigma_pos)
        noisy.yaw             = msg.yaw + self._noise(self.sigma_yaw)
        noisy.velocity        = msg.velocity + self._noise(self.sigma_vel)
        noisy.yaw_rate        = msg.yaw_rate + self._noise(self.sigma_yr)
        noisy.steering_angle  = msg.steering_angle + self._noise(self.sigma_steer)

        # Normalize yaw to [-π, π]
        noisy.yaw = float(self._normalize_angle(noisy.yaw))

        self.pub.publish(noisy)

        self.get_logger().debug(
            f'[Noise] gt=({msg.x:.3f},{msg.y:.3f}) '
            f'→ noisy=({noisy.x:.3f},{noisy.y:.3f}) '
            f'Δpos=({noisy.x - msg.x:.3f},{noisy.y - msg.y:.3f})')

    # ================================================================
    # Helpers
    # ================================================================

    def _noise(self, sigma: float) -> float:
        """Sample one zero-mean Gaussian with std-dev sigma."""
        return float(self.rng.normal(0.0, sigma))

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
    node = SensorNoiseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
