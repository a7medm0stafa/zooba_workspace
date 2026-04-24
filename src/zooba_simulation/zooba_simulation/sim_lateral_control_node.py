"""
Lateral Control Node (Simulation) — Stanley Controller
========================================================
Closed-loop lateral controller for Gazebo simulation.
Reads the vehicle's ground-truth pose from /vehicle/state and
steers the vehicle to a desired lateral lane position using the
Stanley method.

Extended Stanley Law (with heading derivative damping):
    δ = heading_error + k_d * d(heading_error)/dt + atan2(k_stanley * e_cte, k_soft + |v|)

Where:
    e_cte              = desired_y - current_y   (cross-track error)
    heading_error      = desired_heading - current_yaw
    d(heading_error)/dt= rate of change of heading error (damping term)
    k_stanley          = cross-track gain
    k_soft             = softening constant (avoids div-by-zero at low speed)
    k_d_heading        = heading derivative damping gain
    v                  = current longitudinal speed [m/s]

The derivative term prevents heading overshoot during lane changes: when the
heading error is decreasing quickly (approaching 0°) the derivative produces
a counter-steering correction that slows the heading rate before it crosses 0°.

Subscribes:
    /vehicle/state  (vehicle_interfaces/VehicleState)

Publishes:
    /sim/lateral_cmd  (std_msgs/Float64)  — steering angle [degrees]

Parameters (all configurable from the launch file):
    desired_y           : target lateral position / lane [m]
    desired_heading     : target heading [rad]
    k_stanley           : cross-track gain
    k_soft              : softening constant
    k_d_heading         : heading derivative damping gain (new)
    max_steering_angle  : output saturation [degrees]
    control_rate        : controller frequency [Hz]
"""

import math

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleState
from std_msgs.msg import Float64


class SimLateralControlNode(Node):

    def __init__(self):
        super().__init__('lateral_control_node')

        # ---- Declare & read parameters ----
        self.declare_parameter('desired_y',           0.0)
        self.declare_parameter('desired_heading',     0.0)
        self.declare_parameter('k_stanley',           2.5)
        self.declare_parameter('k_soft',              1.0)
        self.declare_parameter('k_d_heading',         0.5)   # heading derivative damping
        self.declare_parameter('max_steering_angle', 35.0)
        self.declare_parameter('control_rate',       20.0)
        self.declare_parameter('state_topic',  '/vehicle/state')
        self.declare_parameter('output_topic', '/sim/lateral_cmd')

        self.desired_y           = self.get_parameter('desired_y').value
        self.desired_heading     = self.get_parameter('desired_heading').value
        self.k_stanley           = self.get_parameter('k_stanley').value
        self.k_soft              = self.get_parameter('k_soft').value
        self.k_d_heading         = self.get_parameter('k_d_heading').value
        self.max_steering_angle  = self.get_parameter('max_steering_angle').value
        control_rate             = self.get_parameter('control_rate').value
        state_topic              = self.get_parameter('state_topic').value
        output_topic             = self.get_parameter('output_topic').value

        # ---- State cache ----
        self.current_x        = 0.0
        self.current_y        = 0.0
        self.current_yaw      = 0.0
        self.current_velocity = 0.0

        # ---- Derivative state for heading damping ----
        self.prev_heading_error = 0.0
        self.last_time          = self.get_clock().now()

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
        self.get_logger().info('╔══════════════════════════════════════════╗')
        self.get_logger().info('║  LATERAL CONTROL NODE (Stanley)  STARTED ║')
        self.get_logger().info('╠══════════════════════════════════════════╣')
        self.get_logger().info(f'║  Desired Y      : {self.desired_y:>6.2f} m              ║')
        self.get_logger().info(f'║  Desired heading: {math.degrees(self.desired_heading):>6.2f} deg            ║')
        self.get_logger().info(f'║  k_stanley      : {self.k_stanley:>6.3f}               ║')
        self.get_logger().info(f'║  k_soft         : {self.k_soft:>6.3f}               ║')
        self.get_logger().info(f'║  k_d_heading    : {self.k_d_heading:>6.3f}               ║')
        self.get_logger().info(f'║  Max steering   : ±{self.max_steering_angle:.1f} deg              ║')
        self.get_logger().info(f'║  Control rate   : {control_rate:>6.1f} Hz             ║')
        self.get_logger().info(f'║  State topic    : {state_topic:<22s}║')
        self.get_logger().info(f'║  Output topic   : {output_topic:<22s}║')
        self.get_logger().info('╚══════════════════════════════════════════╝')
        self.get_logger().info('')

    # ------------------------------------------------------------------
    def _param_callback(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == 'desired_y':
                self.desired_y = p.value
                self.prev_heading_error = 0.0   # reset derivative state on setpoint change
                self.get_logger().info(f'[Stanley] desired_y → {p.value:.3f} m')
            elif p.name == 'desired_heading':
                self.desired_heading = p.value
                self.prev_heading_error = 0.0
                self.get_logger().info(f'[Stanley] desired_heading → {math.degrees(p.value):.2f}°')
            elif p.name == 'k_stanley':
                self.k_stanley = p.value
            elif p.name == 'k_soft':
                self.k_soft = p.value
            elif p.name == 'k_d_heading':
                self.k_d_heading = p.value
        return SetParametersResult(successful=True)

    def _state_callback(self, msg: VehicleState):
        self.current_x        = msg.x
        self.current_y        = msg.y
        self.current_yaw      = msg.yaw
        self.current_velocity = msg.velocity

    def _control_callback(self):
        # --- dt for derivative term ---
        now = self.get_clock().now()
        dt  = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0.0 or dt > 1.0:
            dt = 0.05

        # --- Heading error (normalised to [-π, π]) ---
        heading_error = self._normalise(self.desired_heading - self.current_yaw)

        # --- Derivative of heading error (damping) ---
        # Detects how fast the heading error is changing. When the car is
        # turning back toward the desired heading too quickly the derivative
        # is negative and counter-steers to slow down before overshooting 0°.
        d_heading = self._normalise(heading_error - self.prev_heading_error) / dt
        self.prev_heading_error = heading_error

        # --- Cross-track error ---
        cte = self.desired_y - self.current_y

        # --- Extended Stanley law (+ heading derivative damping) ---
        v = max(abs(self.current_velocity), 0.01)
        cross_track_term  = math.atan2(self.k_stanley * cte, self.k_soft + v)
        heading_damp_term = self.k_d_heading * d_heading
        steering_rad      = heading_error + heading_damp_term + cross_track_term

        # Convert to degrees and negate (VehicleCmd: +right, Stanley: +left)
        steering_deg = -math.degrees(steering_rad)
        steering_deg  = max(-self.max_steering_angle,
                            min(self.max_steering_angle, steering_deg))

        # --- Publish ---
        msg      = Float64()
        msg.data = steering_deg
        self.cmd_pub.publish(msg)

        # --- Pretty-print output (throttled to 1 Hz) ---
        heading_err_deg  = math.degrees(heading_error)
        ct_deg           = math.degrees(cross_track_term)
        damp_deg         = math.degrees(heading_damp_term)
        self.get_logger().info(
            f'\n'
            f'  ╔══════════ LATERAL CONTROL ════════════╗\n'
            f'  ║  Target Y   : {self.desired_y:>+7.3f} m              ║\n'
            f'  ║  Actual Y   : {self.current_y:>+7.3f} m   '
            f'X: {self.current_x:>+6.3f} m  ║\n'
            f'  ║  Heading ψ  : {math.degrees(self.current_yaw):>+7.2f}°  '
            f'Target: {math.degrees(self.desired_heading):>+5.2f}°  ║\n'
            f'  ║  CTE        : {cte:>+7.3f} m              ║\n'
            f'  ║  Head. err  : {heading_err_deg:>+7.2f}°              ║\n'
            f'  ║  Head. damp : {damp_deg:>+7.2f}°              ║\n'
            f'  ║  CTE term   : {ct_deg:>+7.2f}°              ║\n'
            f'  ║  Ctrl eff δ : {steering_deg:>+7.2f}°              ║\n'
            f'  ╚═══════════════════════════════════════╝',
            throttle_duration_sec=1.0
        )

    @staticmethod
    def _normalise(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


# ──────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = SimLateralControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
