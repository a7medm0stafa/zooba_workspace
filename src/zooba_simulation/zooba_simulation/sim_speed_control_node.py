"""
Speed Control Node (Simulation) — PI Controller
=================================================
Closed-loop speed controller for Gazebo simulation.
Reads the vehicle's ground-truth velocity from /vehicle/state and
drives the vehicle towards a desired speed using a discrete PI controller
with anti-windup.

Subscribes:
    /vehicle/state  (vehicle_interfaces/VehicleState)  — ground-truth velocity

Publishes:
    /sim/speed_cmd  (std_msgs/Float64)  — commanded velocity [m/s]

Parameters (all configurable from the launch file):
    desired_speed   : goal speed [m/s]
    kp              : proportional gain
    ki              : integral gain
    max_velocity    : output saturation [m/s]
    control_rate    : controller frequency [Hz]
"""

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleState
from std_msgs.msg import Float64


class SimSpeedControlNode(Node):

    def __init__(self):
        super().__init__('speed_control_node')

        # ---- Declare & read parameters ----
        self.declare_parameter('desired_speed', 0.5)
        self.declare_parameter('kp', 1.5)
        self.declare_parameter('ki', 0.2)
        self.declare_parameter('max_velocity', 2.0)
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('state_topic', '/vehicle/state')
        self.declare_parameter('output_topic', '/sim/speed_cmd')

        self.desired_speed = self.get_parameter('desired_speed').value
        self.kp            = self.get_parameter('kp').value
        self.ki            = self.get_parameter('ki').value
        self.max_velocity  = self.get_parameter('max_velocity').value
        control_rate       = self.get_parameter('control_rate').value
        state_topic        = self.get_parameter('state_topic').value
        output_topic       = self.get_parameter('output_topic').value

        # ---- PI state ----
        self.integral         = 0.0
        self.current_velocity = 0.0
        self.last_time        = self.get_clock().now()
        self.integral_max     = self.max_velocity / max(self.ki, 0.001)

        # ---- Subscriber ----
        self.state_sub = self.create_subscription(
            VehicleState, state_topic, self._state_callback, 10)

        # ---- Publisher ----
        self.cmd_pub = self.create_publisher(Float64, output_topic, 10)

        # ---- Control timer ----
        self.timer = self.create_timer(1.0 / control_rate, self._control_callback)

        # ---- Dynamic parameter updates ----
        self.add_on_set_parameters_callback(self._param_callback)

        # ---- Startup banner ----
        self.get_logger().info('')
        self.get_logger().info('╔══════════════════════════════════════╗')
        self.get_logger().info('║    SPEED CONTROL NODE  (PI)  STARTED ║')
        self.get_logger().info('╠══════════════════════════════════════╣')
        self.get_logger().info(f'║  Desired speed : {self.desired_speed:>6.2f} m/s           ║')
        self.get_logger().info(f'║  Kp            : {self.kp:>6.3f}               ║')
        self.get_logger().info(f'║  Ki            : {self.ki:>6.3f}               ║')
        self.get_logger().info(f'║  Max velocity  : {self.max_velocity:>6.2f} m/s           ║')
        self.get_logger().info(f'║  Control rate  : {control_rate:>6.1f} Hz            ║')
        self.get_logger().info(f'║  State topic   : {state_topic:<20s}║')
        self.get_logger().info(f'║  Output topic  : {output_topic:<20s}║')
        self.get_logger().info('╚══════════════════════════════════════╝')
        self.get_logger().info('')

    # ------------------------------------------------------------------
    def _param_callback(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == 'desired_speed':
                self.desired_speed = p.value
                self.get_logger().info(f'[PI] desired_speed → {p.value:.3f} m/s')
            elif p.name == 'kp':
                self.kp = p.value
            elif p.name == 'ki':
                self.ki = p.value
                self.integral_max = self.max_velocity / max(self.ki, 0.001)
        return SetParametersResult(successful=True)

    def _state_callback(self, msg: VehicleState):
        self.current_velocity = msg.velocity

    def _control_callback(self):
        now = self.get_clock().now()
        dt  = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        if dt <= 0.0 or dt > 1.0:
            dt = 0.05

        # --- PI computation ---
        error = self.desired_speed - self.current_velocity

        self.integral += error * dt
        self.integral  = max(-self.integral_max,
                             min(self.integral_max, self.integral))

        output = self.kp * error + self.ki * self.integral
        output = max(0, min(self.max_velocity, output))

        # --- Publish ---
        msg      = Float64()
        msg.data = output
        self.cmd_pub.publish(msg)

        # --- Pretty-print output (throttled to 1 Hz) ---
        self.get_logger().info(
            f'\n'
            f'  ╔══════════ SPEED CONTROL ══════════╗\n'
            f'  ║  Target    : {self.desired_speed:>+7.3f} m/s          ║\n'
            f'  ║  True vel  : {self.current_velocity:>+7.3f} m/s          ║\n'
            f'  ║  Error     : {error:>+7.3f} m/s          ║\n'
            f'  ║  Ctrl eff  : {output:>+7.3f} m/s          ║\n'
            f'  ╚═══════════════════════════════════╝',
            throttle_duration_sec=1.0
        )


# ──────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = SimSpeedControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
