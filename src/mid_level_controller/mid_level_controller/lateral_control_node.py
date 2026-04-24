"""
Lateral Control Node — Stanley Controller
===========================================
Closed-loop lateral controller using the Stanley method to steer the
vehicle towards a desired lateral position (lane).

The Stanley controller operates at the front axle and combines:
  1. Heading error correction
  2. Cross-track error correction (proportional to atan of lateral offset)

Stanley Law:
    δ = (ψ_desired - ψ) + atan2(k * e_cte, k_soft + v)

Where:
    ψ_desired = desired heading [rad]
    ψ         = current heading [rad]
    e_cte     = cross-track error (desired_y - y) [m]
    k         = cross-track gain
    k_soft    = softening constant to avoid div-by-zero at low speed
    v         = current longitudinal velocity [m/s]

Reference:
    Comparison of lateral controllers for autonomous vehicle
    https://hal.archives-ouvertes.fr/hal-02459398/document

Subscribes:
    /vehicle/state  (vehicle_interfaces/VehicleState)

Publishes:
    /teleop/lateral_cmd  (std_msgs/Float64)  — steering angle [degrees]
"""

import math

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleState
from std_msgs.msg import Float64


class LateralControlNode(Node):

    def __init__(self):
        super().__init__('lateral_control_node')

        # ---- Parameters ----
        self.declare_parameter('desired_y', 0.0)               # target lateral position [m]
        self.declare_parameter('desired_heading', 0.0)         # target heading [rad]
        self.declare_parameter('k_stanley', 2.5)               # cross-track gain
        self.declare_parameter('k_soft', 1.0)                  # softening constant
        self.declare_parameter('max_steering_angle', 35.0)     # degrees
        self.declare_parameter('control_rate', 20.0)           # Hz
        self.declare_parameter('state_topic', '/vehicle/state')
        self.declare_parameter('output_topic', '/teleop/lateral_cmd')

        self.desired_y = self.get_parameter('desired_y').value
        self.desired_heading = self.get_parameter('desired_heading').value
        self.k_stanley = self.get_parameter('k_stanley').value
        self.k_soft = self.get_parameter('k_soft').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        control_rate = self.get_parameter('control_rate').value
        state_topic = self.get_parameter('state_topic').value
        output_topic = self.get_parameter('output_topic').value

        # ---- State ----
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_velocity = 0.0

        # ---- Subscriber ----
        self.state_sub = self.create_subscription(
            VehicleState, state_topic, self._state_callback, 10)

        # ---- Publisher ----
        self.cmd_pub = self.create_publisher(Float64, output_topic, 10)

        # ---- Timer ----
        self.timer = self.create_timer(1.0 / control_rate, self._control_callback)

        # ---- Dynamic parameter update ----
        self.add_on_set_parameters_callback(self._param_callback)

        self.get_logger().info('=' * 55)
        self.get_logger().info('Lateral Control Node Started (Stanley)')
        self.get_logger().info(f'  Desired Y       : {self.desired_y:.2f} m')
        self.get_logger().info(f'  Desired heading : {self.desired_heading:.3f} rad')
        self.get_logger().info(f'  k_stanley       : {self.k_stanley}')
        self.get_logger().info(f'  k_soft          : {self.k_soft}')
        self.get_logger().info(f'  Max steering    : ±{self.max_steering_angle:.1f}°')
        self.get_logger().info(f'  Control rate    : {control_rate:.0f} Hz')
        self.get_logger().info(f'  State topic     : {state_topic}')
        self.get_logger().info(f'  Output topic    : {output_topic}')
        self.get_logger().info('=' * 55)

    def _param_callback(self, params):
        """Handle dynamic parameter updates."""
        from rcl_interfaces.msg import SetParametersResult
        for param in params:
            if param.name == 'desired_y':
                self.desired_y = param.value
                self.get_logger().info(f'[Stanley] desired_y updated: {param.value:.2f} m')
            elif param.name == 'desired_heading':
                self.desired_heading = param.value
                self.get_logger().info(f'[Stanley] desired_heading updated: {param.value:.3f} rad')
            elif param.name == 'k_stanley':
                self.k_stanley = param.value
            elif param.name == 'k_soft':
                self.k_soft = param.value
        return SetParametersResult(successful=True)

    def _state_callback(self, msg: VehicleState):
        """Receive current vehicle state."""
        self.current_x = msg.x
        self.current_y = msg.y
        self.current_yaw = msg.yaw
        self.current_velocity = msg.velocity

    def _control_callback(self):
        """Stanley control loop — compute and publish steering command."""
        # --- Heading error ---
        heading_error = self._normalize_angle(self.desired_heading - self.current_yaw)

        # --- Cross-track error ---
        # For straight-line following along X axis with desired lateral offset Y:
        cross_track_error = self.desired_y - self.current_y

        # --- Stanley law ---
        # δ = heading_error + atan2(k * e_cte, k_soft + |v|)
        v = abs(self.current_velocity) if abs(self.current_velocity) > 0.01 else 0.01
        cross_track_term = math.atan2(
            self.k_stanley * cross_track_error,
            self.k_soft + v
        )

        # Total steering angle (radians)
        steering_rad = heading_error + cross_track_term

        # Convert to degrees and negate to match VehicleCmd convention (+right, -left)
        steering_deg = -math.degrees(steering_rad)

        # Saturate
        steering_deg = max(-self.max_steering_angle,
                           min(self.max_steering_angle, steering_deg))

        # Publish
        msg = Float64()
        msg.data = steering_deg
        self.cmd_pub.publish(msg)

        # Log (throttled)
        self.get_logger().info(
            f'[Stanley] cte={cross_track_error:.3f}m '
            f'he={math.degrees(heading_error):.1f}° '
            f'ct={math.degrees(cross_track_term):.1f}° '
            f'δ={steering_deg:.1f}° '
            f'pos=({self.current_x:.2f},{self.current_y:.2f}) '
            f'ψ={math.degrees(self.current_yaw):.1f}°',
            throttle_duration_sec=2.0
        )

    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-π, π]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)

    node = LateralControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
