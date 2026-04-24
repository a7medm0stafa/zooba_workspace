"""
Lateral Control Node — Extended Stanley Controller
====================================================
Closed-loop lateral controller using the extended Stanley method to steer the
vehicle towards a desired lateral position (lane).

Extended Stanley Law (with heading proportional gain and derivative damping):
    δ = k_heading * heading_error + k_d * d(heading_error)/dt + atan2(k_stanley * e_cte, k_soft + |v|)

Where:
    e_cte              = perpendicular distance from car to desired path line
    heading_error      = desired_heading - current_yaw  (normalized to [-π, π])
    d(heading_error)/dt= rate of change of heading error (damping term)
    k_heading          = heading proportional gain (> 1 for aggressive heading alignment)
    k_stanley          = cross-track gain
    k_soft             = softening constant (avoids div-by-zero at low speed)
    k_d_heading        = heading derivative damping gain
    v                  = current longitudinal speed [m/s]

The desired path is a line through (0, desired_y) in the direction of
desired_heading.  The cross-track error is the signed perpendicular distance
from the vehicle to this line.

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
        self.declare_parameter('desired_heading', 0.0)         # target heading [degrees]
        self.declare_parameter('k_heading', 1.5)               # heading proportional gain
        self.declare_parameter('k_stanley', 2.5)               # cross-track gain
        self.declare_parameter('k_soft', 1.0)                  # softening constant
        self.declare_parameter('k_d_heading', 0.3)             # heading derivative damping
        self.declare_parameter('max_steering_angle', 35.0)     # degrees
        self.declare_parameter('control_rate', 20.0)           # Hz
        self.declare_parameter('state_topic', '/vehicle/state')
        self.declare_parameter('output_topic', '/teleop/lateral_cmd')

        self.desired_y = self.get_parameter('desired_y').value
        self.desired_heading = self.get_parameter('desired_heading').value          # degrees
        self.desired_heading_rad = math.radians(self.desired_heading)               # radians (internal)
        self.k_heading = self.get_parameter('k_heading').value
        self.k_stanley = self.get_parameter('k_stanley').value
        self.k_soft = self.get_parameter('k_soft').value
        self.k_d_heading = self.get_parameter('k_d_heading').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        control_rate = self.get_parameter('control_rate').value
        state_topic = self.get_parameter('state_topic').value
        output_topic = self.get_parameter('output_topic').value

        # ---- State ----
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_velocity = 0.0

        # ---- Derivative state for heading damping ----
        self.prev_heading_error = 0.0
        self.last_time = self.get_clock().now()

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
        self.get_logger().info('Lateral Control Node Started (Extended Stanley)')
        self.get_logger().info(f'  Desired Y       : {self.desired_y:.2f} m')
        self.get_logger().info(f'  Desired heading : {self.desired_heading:.1f}° ({self.desired_heading_rad:.4f} rad)')
        self.get_logger().info(f'  k_heading       : {self.k_heading}')
        self.get_logger().info(f'  k_stanley       : {self.k_stanley}')
        self.get_logger().info(f'  k_soft          : {self.k_soft}')
        self.get_logger().info(f'  k_d_heading     : {self.k_d_heading}')
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
                self.prev_heading_error = 0.0   # reset derivative on setpoint change
                self.get_logger().info(f'[Stanley] desired_y updated: {param.value:.2f} m')
            elif param.name == 'desired_heading':
                self.desired_heading = param.value
                self.desired_heading_rad = math.radians(param.value)
                self.prev_heading_error = 0.0
                self.get_logger().info(f'[Stanley] desired_heading updated: {param.value:.1f}° ({self.desired_heading_rad:.4f} rad)')
            elif param.name == 'k_heading':
                self.k_heading = param.value
            elif param.name == 'k_stanley':
                self.k_stanley = param.value
            elif param.name == 'k_soft':
                self.k_soft = param.value
            elif param.name == 'k_d_heading':
                self.k_d_heading = param.value
        return SetParametersResult(successful=True)

    def _state_callback(self, msg: VehicleState):
        """Receive current vehicle state."""
        self.current_x = msg.x
        self.current_y = msg.y
        self.current_yaw = msg.yaw
        self.current_velocity = msg.velocity

    def _control_callback(self):
        """Extended Stanley control loop — compute and publish steering command."""
        # --- dt for derivative term ---
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0.0 or dt > 1.0:
            dt = 0.05

        # --- Heading error (normalised to [-π, π]) ---
        heading_error = self._normalize_angle(self.desired_heading_rad - self.current_yaw)

        # --- Derivative of heading error (damping) ---
        d_heading = self._normalize_angle(heading_error - self.prev_heading_error) / dt
        self.prev_heading_error = heading_error

        # --- Cross-track error (perpendicular distance to desired path line) ---
        # Path: line through (0, desired_y) in direction desired_heading_rad
        # Positive CTE = car needs to steer left to reach the path
        dx = self.current_x
        dy = self.current_y - self.desired_y
        cross_track_error = dx * math.sin(self.desired_heading_rad) \
                          - dy * math.cos(self.desired_heading_rad)

        # --- Extended Stanley law ---
        #   k_heading * heading_error   — proportional heading correction
        #   k_d_heading * d_heading     — heading derivative damping
        #   atan2(k_stanley * CTE, ...) — cross-track correction
        v = max(abs(self.current_velocity), 0.01)
        cross_track_term = math.atan2(self.k_stanley * cross_track_error, self.k_soft + v)
        heading_term = self.k_heading * heading_error
        heading_damp_term = self.k_d_heading * d_heading
        steering_rad = heading_term + heading_damp_term + cross_track_term

        # Convert to degrees and negate (VehicleCmd: +right, Stanley: +left)
        steering_deg = -math.degrees(steering_rad)
        steering_deg = max(-self.max_steering_angle,
                           min(self.max_steering_angle, steering_deg))

        # Publish
        msg = Float64()
        msg.data = float(steering_deg)
        self.cmd_pub.publish(msg)

        # Log (throttled)
        self.get_logger().info(
            f'[Stanley] cte={cross_track_error:.3f}m '
            f'he={math.degrees(heading_error):.1f}° '
            f'ht={math.degrees(heading_term):.1f}° '
            f'damp={math.degrees(heading_damp_term):.1f}° '
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
