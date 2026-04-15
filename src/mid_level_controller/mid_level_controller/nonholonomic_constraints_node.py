"""
Non-Holonomic Constraints Node for Ackermann Vehicle
====================================================
Subscribes to raw vehicle commands (e.g. from teleop) and enforces the
physical constraints of an Ackermann-steered vehicle before republishing.

Constraints enforced:
    1. Velocity clamping:        |v| ≤ max_velocity
    2. Steering angle clamping:  |δ| ≤ max_steering_angle
    3. Velocity rate limiting:   |dv/dt| ≤ max_velocity_rate
    4. Steering rate limiting:   |dδ/dt| ≤ max_steering_rate
    5. Minimum turning radius:   R_min = wheelbase / tan(max_steering_angle)
       (informational — the steering clamp implicitly guarantees this)

Subscribes to:
    /teleop/raw_cmd  (vehicle_interfaces/VehicleCmd)

Publishes:
    /vehicle/cmd          (vehicle_interfaces/VehicleCmd)       — constrained commands
    /vehicle/constraints  (vehicle_interfaces/VehicleConstraints) — diagnostics
"""

import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vehicle_interfaces.msg import VehicleCmd, VehicleConstraints


class NonHolonomicConstraintsNode(Node):

    def __init__(self):
        super().__init__('nonholonomic_constraints_node')

        # ---- Parameters ----
        self.declare_parameter('wheelbase', 0.22)              # meters
        self.declare_parameter('track_width', 0.20)            # meters
        self.declare_parameter('max_velocity', 2.0)            # m/s
        self.declare_parameter('max_steering_angle', 35.0)     # degrees
        self.declare_parameter('max_velocity_rate', 1.0)       # m/s per second
        self.declare_parameter('max_steering_rate', 45.0)      # degrees per second
        self.declare_parameter('wheel_radius', 0.04)           # meters
        self.declare_parameter('input_topic', '/teleop/raw_cmd')
        self.declare_parameter('output_topic', '/vehicle/cmd')
        self.declare_parameter('constraints_topic', '/vehicle/constraints')
        self.declare_parameter('publish_rate', 20.0)

        # Sign detection parameters
        self.declare_parameter('sign_command_topic', '/sign/command')
        self.declare_parameter('stop_velocity', 0.0)
        self.declare_parameter('slow_velocity', 0.75)
        self.declare_parameter('turn_velocity', 1.0)
        self.declare_parameter('turn_heading', 20.0)           # Hz

        self.wheelbase = self.get_parameter('wheelbase').value
        self.track_width = self.get_parameter('track_width').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.max_velocity_rate = self.get_parameter('max_velocity_rate').value
        self.max_steering_rate = self.get_parameter('max_steering_rate').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        constraints_topic = self.get_parameter('constraints_topic').value
        publish_rate = self.get_parameter('publish_rate').value

        # Sign detection parameters
        sign_command_topic = self.get_parameter('sign_command_topic').value
        self.stop_velocity = self.get_parameter('stop_velocity').value
        self.slow_velocity = self.get_parameter('slow_velocity').value
        self.turn_velocity = self.get_parameter('turn_velocity').value
        self.turn_heading = self.get_parameter('turn_heading').value

        # ---- Derived constants ----
        max_steer_rad = math.radians(self.max_steering_angle)
        if max_steer_rad > 0:
            self.min_turning_radius = self.wheelbase / math.tan(max_steer_rad)
        else:
            self.min_turning_radius = float('inf')

        # ---- State ----
        self.desired_velocity = 0.0
        self.desired_heading = 0.0
        self.current_velocity = 0.0
        self.current_heading = 0.0
        self.last_update_time = self.get_clock().now()

        # ---- Sign detection state ----
        self.sign_command = 'NO_SIGNAL'  # Current active sign command
        self.sign_active = False          # Whether a sign is overriding teleop
        self.last_sign_time = time.time() # Timestamp of last sign command received

        # ---- Subscriber (teleop) ----
        self.raw_cmd_sub = self.create_subscription(
            VehicleCmd,
            input_topic,
            self._raw_cmd_callback,
            10
        )

        # ---- Subscriber (sign detection) ----
        self.sign_cmd_sub = self.create_subscription(
            String,
            sign_command_topic,
            self._sign_cmd_callback,
            10
        )

        # ---- Publishers ----
        self.cmd_pub = self.create_publisher(VehicleCmd, output_topic, 10)
        self.constraints_pub = self.create_publisher(
            VehicleConstraints, constraints_topic, 10
        )

        # ---- Timer for rate-limited output ----
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self._timer_callback)

        # ---- Log startup info ----
        self.get_logger().info('=' * 58)
        self.get_logger().info('Non-Holonomic Constraints Node Started')
        self.get_logger().info(f'  Input topic      : {input_topic}')
        self.get_logger().info(f'  Sign cmd topic   : {sign_command_topic}')
        self.get_logger().info(f'  Output topic     : {output_topic}')
        self.get_logger().info(f'  Wheelbase        : {self.wheelbase:.3f} m')
        self.get_logger().info(f'  Track width      : {self.track_width:.3f} m')
        self.get_logger().info(f'  Max velocity     : {self.max_velocity:.2f} m/s')
        self.get_logger().info(f'  Max steering     : ±{self.max_steering_angle:.1f}°')
        self.get_logger().info(f'  Min turn radius  : {self.min_turning_radius:.3f} m')
        self.get_logger().info(f'  Vel rate limit   : {self.max_velocity_rate:.2f} m/s²')
        self.get_logger().info(f'  Steer rate limit : {self.max_steering_rate:.1f} °/s')
        self.get_logger().info(f'  Output rate      : {publish_rate:.0f} Hz')
        self.get_logger().info(f'  Sign speeds      : STOP={self.stop_velocity}, SLOW={self.slow_velocity}, TURN={self.turn_velocity} m/s')
        self.get_logger().info(f'  Turn heading     : ±{self.turn_heading}°')
        self.get_logger().info('=' * 58)

    # ==================== Callbacks ====================

    def _raw_cmd_callback(self, msg: VehicleCmd):
        """Store the latest desired (unconstrained) command from teleop."""
        # Only use teleop if no sign is actively overriding
        if not self.sign_active:
            self.desired_velocity = msg.velocity
            self.desired_heading = msg.heading

    def _sign_cmd_callback(self, msg: String):
        """Handle sign detection commands — override teleop when a sign is active."""
        command = msg.data.strip()
        self.sign_command = command
        self.last_sign_time = time.time()

        if command == 'STOP':
            self.sign_active = True
            self.desired_velocity = self.stop_velocity
            self.desired_heading = 0.0
            self.get_logger().info(f'SIGN CMD: STOP → vel={self.stop_velocity}')

        elif command == 'SLOW_DOWN':
            self.sign_active = True
            self.desired_velocity = self.slow_velocity
            self.desired_heading = 0.0
            self.get_logger().info(f'SIGN CMD: SLOW_DOWN → vel={self.slow_velocity}')

        elif command == 'TURN_LEFT':
            self.sign_active = True
            self.desired_velocity = self.turn_velocity
            self.desired_heading = -self.turn_heading  # Negative = left
            self.get_logger().info(f'SIGN CMD: TURN_LEFT → vel={self.turn_velocity}, hdg={-self.turn_heading}°')

        elif command == 'TURN_RIGHT':
            self.sign_active = True
            self.desired_velocity = self.turn_velocity
            self.desired_heading = self.turn_heading  # Positive = right
            self.get_logger().info(f'SIGN CMD: TURN_RIGHT → vel={self.turn_velocity}, hdg={self.turn_heading}°')

        elif command == 'NO_SIGNAL':
            self.sign_active = False
            # Teleop resumes control (next _raw_cmd_callback will set desired values)

    def _timer_callback(self):
        """Apply constraints and publish at fixed rate."""
        now = self.get_clock().now()
        dt = (now - self.last_update_time).nanoseconds * 1e-9
        self.last_update_time = now

        # Guard against zero or absurd dt
        if dt <= 0.0 or dt > 1.0:
            dt = 0.05  # fallback to ~20 Hz

        # ---- 1. Clamp desired values ----
        target_velocity = max(-self.max_velocity,
                              min(self.max_velocity, self.desired_velocity))
        target_heading = max(-self.max_steering_angle,
                             min(self.max_steering_angle, self.desired_heading))

        # ---- 2. Rate-limit velocity ----
        vel_diff = target_velocity - self.current_velocity
        max_vel_change = self.max_velocity_rate * dt
        if abs(vel_diff) > max_vel_change:
            vel_diff = math.copysign(max_vel_change, vel_diff)
        self.current_velocity += vel_diff

        # ---- 3. Rate-limit steering ----
        hdg_diff = target_heading - self.current_heading
        max_hdg_change = self.max_steering_rate * dt
        if abs(hdg_diff) > max_hdg_change:
            hdg_diff = math.copysign(max_hdg_change, hdg_diff)
        self.current_heading += hdg_diff

        # ---- 4. Final clamp (safety) ----
        self.current_velocity = max(-self.max_velocity,
                                    min(self.max_velocity, self.current_velocity))
        self.current_heading = max(-self.max_steering_angle,
                                   min(self.max_steering_angle, self.current_heading))

        # ---- Compute instantaneous turning radius for diagnostics ----
        steer_rad = math.radians(abs(self.current_heading))
        if steer_rad > 1e-3:
            instant_radius = self.wheelbase / math.tan(steer_rad)
        else:
            instant_radius = float('inf')

        # ---- Publish constrained command ----
        cmd_msg = VehicleCmd()
        cmd_msg.header.stamp = now.to_msg()
        cmd_msg.header.frame_id = 'base_link'
        cmd_msg.velocity = self.current_velocity
        cmd_msg.heading = self.current_heading
        self.cmd_pub.publish(cmd_msg)

        # ---- Publish constraints diagnostic ----
        diag_msg = VehicleConstraints()
        diag_msg.max_velocity = self.max_velocity
        diag_msg.max_steering_angle_deg = self.max_steering_angle
        diag_msg.min_turning_radius = self.min_turning_radius
        diag_msg.wheelbase = self.wheelbase
        diag_msg.constraints_active = True
        self.constraints_pub.publish(diag_msg)


def main(args=None):
    rclpy.init(args=args)

    node = NonHolonomicConstraintsNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
