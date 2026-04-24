"""
Speed Control Node — PI Controller
====================================
Closed-loop speed controller subscribing to the vehicle's current state
and publishing a velocity command to reach a desired speed.

Uses a discrete PI (Proportional-Integral) controller with anti-windup.

Subscribes:
    /vehicle/state  (vehicle_interfaces/VehicleState)  — current velocity

Publishes:
    /teleop/speed_cmd  (std_msgs/Float64)  — commanded velocity [m/s]

Parameters (configurable from launch file):
    desired_speed:  goal speed [m/s]
    kp:             proportional gain
    ki:             integral gain
    max_velocity:   output saturation [m/s]
    control_rate:   Hz
"""

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleState
from std_msgs.msg import Float64


class SpeedControlNode(Node):

    def __init__(self):
        super().__init__('speed_control_node')

        # ---- Parameters ----
        self.declare_parameter('desired_speed', 0.5)       # m/s
        self.declare_parameter('kp', 1.0)
        self.declare_parameter('ki', 0.1)
        self.declare_parameter('max_velocity', 2.0)        # m/s saturation
        self.declare_parameter('control_rate', 20.0)       # Hz
        self.declare_parameter('state_topic', '/vehicle/state')
        self.declare_parameter('output_topic', '/teleop/speed_cmd')
        self.declare_parameter('bypass_pi', False)         # Bypass ROS PI logic (pass-through)

        self.desired_speed = self.get_parameter('desired_speed').value
        self.kp = self.get_parameter('kp').value
        self.ki = self.get_parameter('ki').value
        self.max_velocity = self.get_parameter('max_velocity').value
        control_rate = self.get_parameter('control_rate').value
        state_topic = self.get_parameter('state_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.bypass_pi = self.get_parameter('bypass_pi').value

        # ---- PI State ----
        self.integral = 0.0
        self.current_velocity = 0.0
        self.last_time = self.get_clock().now()

        # Anti-windup: integral saturation
        self.integral_max = self.max_velocity / max(self.ki, 0.001)

        # ---- Subscriber ----
        self.state_sub = self.create_subscription(
            VehicleState, state_topic, self._state_callback, 10)

        # ---- Publisher ----
        self.cmd_pub = self.create_publisher(Float64, output_topic, 10)

        # ---- Timer ----
        self.timer = self.create_timer(1.0 / control_rate, self._control_callback)

        # ---- Dynamic parameter update ----
        self.add_on_set_parameters_callback(self._param_callback)

        self.get_logger().info('=' * 50)
        self.get_logger().info('Speed Control Node Started (PI)')
        self.get_logger().info(f'  Desired speed : {self.desired_speed:.2f} m/s')
        self.get_logger().info(f'  Kp            : {self.kp}')
        self.get_logger().info(f'  Ki            : {self.ki}')
        self.get_logger().info(f'  Max velocity  : {self.max_velocity:.2f} m/s')
        self.get_logger().info(f'  Control rate  : {control_rate:.0f} Hz')
        self.get_logger().info(f'  State topic   : {state_topic}')
        self.get_logger().info(f'  Output topic  : {output_topic}')
        self.get_logger().info(f'  Bypass PI     : {self.bypass_pi}')
        self.get_logger().info('=' * 50)

    def _param_callback(self, params):
        """Handle dynamic parameter updates."""
        from rcl_interfaces.msg import SetParametersResult
        for param in params:
            if param.name == 'desired_speed':
                self.desired_speed = param.value
                self.get_logger().info(f'[PI] desired_speed updated: {param.value:.2f} m/s')
            elif param.name == 'kp':
                self.kp = param.value
            elif param.name == 'ki':
                self.ki = param.value
                self.integral_max = self.max_velocity / max(self.ki, 0.001)
        return SetParametersResult(successful=True)

    def _state_callback(self, msg: VehicleState):
        """Receive current vehicle velocity."""
        self.current_velocity = msg.velocity

    def _control_callback(self):
        """PI control loop — compute and publish velocity command."""
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        if dt <= 0.0 or dt > 1.0:
            dt = 0.05

        if self.bypass_pi:
            output = self.desired_speed
            self.integral = 0.0  # Keep integral zeroed
            
            # Publish
            msg = Float64()
            msg.data = output
            self.cmd_pub.publish(msg)

            # Log (throttled)
            self.get_logger().info(
                f'[PI BYPASS] Forwarding des={self.desired_speed:.3f} directly to output',
                throttle_duration_sec=2.0
            )
            return

        # Error
        error = self.desired_speed - self.current_velocity

        # Integral with anti-windup
        self.integral += error * dt
        self.integral = max(-self.integral_max, min(self.integral_max, self.integral))

        # PI output
        output = self.kp * error + self.ki * self.integral

        # Saturate
        output = max(-self.max_velocity, min(self.max_velocity, output))

        # Publish
        msg = Float64()
        msg.data = output
        self.cmd_pub.publish(msg)

        # Log (throttled)
        self.get_logger().info(
            f'[PI] err={error:.3f} int={self.integral:.3f} '
            f'out={output:.3f} cur={self.current_velocity:.3f} '
            f'des={self.desired_speed:.3f}',
            throttle_duration_sec=2.0
        )


def main(args=None):
    rclpy.init(args=args)

    node = SpeedControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
