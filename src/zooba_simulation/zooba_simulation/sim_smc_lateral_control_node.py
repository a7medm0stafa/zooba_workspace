"""
Lateral Control Node (Simulation) — Sliding Mode Controller (SMC)
==================================================================
Closed-loop lateral controller for Gazebo simulation using a
Sliding Mode Control (SMC) strategy.

The sliding surface combines cross-track error (CTE) and heading error:

    s = lambda * e_cte + e_heading

Where:
    e_cte      = desired_y - current_y        (cross-track error)
    e_heading  = desired_heading - current_yaw (heading error, normalised)
    lambda     = weighting gain that blends position vs heading error

The discontinuous control law (with boundary layer for chattering reduction):

    delta = -k_smc * sat(s / phi) - k_heading * e_heading

Where:
    k_smc      = SMC switching gain (drives s → 0)
    k_heading  = proportional heading feedback term
    phi        = boundary layer thickness (replaces sign() with sat()
                 to suppress chattering near s=0)

The sat() function (saturation inside boundary layer):
    sat(x) = x          if |x| <= 1  (PD-like region near surface)
    sat(x) = sign(x)    if |x| >  1  (pure switching outside)

Benefits over Stanley:
  - Robust to model uncertainty and disturbances (guaranteed sliding mode)
  - Boundary layer eliminates high-frequency chattering in actuators
  - No velocity division — safe at standstill (no k_soft workaround needed)
  - lambda gives explicit trade-off tuning between CTE and heading

Subscribes:
    /vehicle/state  (vehicle_interfaces/VehicleState)

Publishes:
    /sim/lateral_cmd  (std_msgs/Float64)  — steering angle [degrees]

Parameters (all configurable from the launch file):
    desired_y           : target lateral position / lane [m]
    desired_heading     : target heading [rad]
    lambda_smc          : sliding surface CTE weighting gain
    k_smc               : SMC switching gain
    k_heading           : proportional heading feedback gain
    phi                 : boundary layer thickness [rad] (chattering suppression)
    max_steering_angle  : output saturation [degrees]
    control_rate        : controller frequency [Hz]
"""

import math

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleState
from std_msgs.msg import Float64


class SimSmcLateralControlNode(Node):

    def __init__(self):
        super().__init__('lateral_control_node')

        # ---- Declare & read parameters ----
        self.declare_parameter('desired_y',           0.0)
        self.declare_parameter('desired_heading',     0.0)
        self.declare_parameter('lambda_smc',          1.5)   # sliding surface blend (CTE weight)
        self.declare_parameter('k_smc',               3.0)   # switching gain
        self.declare_parameter('k_heading',           1.2)   # proportional heading term
        self.declare_parameter('phi',                 0.3)   # boundary layer thickness [rad]
        self.declare_parameter('max_steering_angle', 35.0)
        self.declare_parameter('control_rate',       20.0)
        self.declare_parameter('state_topic',  '/vehicle/state')
        self.declare_parameter('output_topic', '/sim/lateral_cmd')

        self.desired_y          = self.get_parameter('desired_y').value
        self.desired_heading    = self.get_parameter('desired_heading').value
        self.lambda_smc         = self.get_parameter('lambda_smc').value
        self.k_smc              = self.get_parameter('k_smc').value
        self.k_heading          = self.get_parameter('k_heading').value
        self.phi                = self.get_parameter('phi').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        control_rate            = self.get_parameter('control_rate').value
        state_topic             = self.get_parameter('state_topic').value
        output_topic            = self.get_parameter('output_topic').value

        # ---- State cache ----
        self.current_x        = 0.0
        self.current_y        = 0.0
        self.current_yaw      = 0.0
        self.current_velocity = 0.0

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
        self.get_logger().info('║   LATERAL CONTROL NODE  (SMC)  STARTED   ║')
        self.get_logger().info('╠══════════════════════════════════════════╣')
        self.get_logger().info(f'║  Desired Y      : {self.desired_y:>6.2f} m              ║')
        self.get_logger().info(f'║  Desired heading: {math.degrees(self.desired_heading):>6.2f} deg            ║')
        self.get_logger().info(f'║  lambda_smc     : {self.lambda_smc:>6.3f}               ║')
        self.get_logger().info(f'║  k_smc          : {self.k_smc:>6.3f}               ║')
        self.get_logger().info(f'║  k_heading      : {self.k_heading:>6.3f}               ║')
        self.get_logger().info(f'║  phi (BL)       : {self.phi:>6.3f} rad            ║')
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
                self.get_logger().info(f'[SMC] desired_y → {p.value:.3f} m')
            elif p.name == 'desired_heading':
                self.desired_heading = p.value
                self.get_logger().info(f'[SMC] desired_heading → {math.degrees(p.value):.2f}°')
            elif p.name == 'lambda_smc':
                self.lambda_smc = p.value
            elif p.name == 'k_smc':
                self.k_smc = p.value
            elif p.name == 'k_heading':
                self.k_heading = p.value
            elif p.name == 'phi':
                self.phi = max(p.value, 1e-6)   # phi must never be zero
        return SetParametersResult(successful=True)

    def _state_callback(self, msg: VehicleState):
        self.current_x        = msg.x
        self.current_y        = msg.y
        self.current_yaw      = msg.yaw
        self.current_velocity = msg.velocity

    def _control_callback(self):
        # --- Errors ---
        e_cte     = self.desired_y - self.current_y
        e_heading = self._normalise(self.desired_heading - self.current_yaw)

        # --- Sliding surface ---
        # s > 0  ⟹ vehicle is to the right of (or under-turned toward) the target
        # s < 0  ⟹ vehicle is to the left  of (or over-turned toward) the target
        s = self.lambda_smc * e_cte + e_heading

        # --- Boundary-layer saturation (replaces discontinuous sign function) ---
        # Inside |s| < phi  → proportional (smooth) action
        # Outside           → full switching action
        sat_s = self._sat(s / max(self.phi, 1e-6))

        # --- SMC steering law ---
        # Positive s → steer left (positive steering in our convention before negation)
        steering_rad = self.k_smc * sat_s + self.k_heading * e_heading

        # Convert to degrees and negate (VehicleCmd convention: +right, control: +left)
        steering_deg = -math.degrees(steering_rad)
        steering_deg = max(-self.max_steering_angle,
                           min(self.max_steering_angle, steering_deg))

        # --- Publish ---
        msg      = Float64()
        msg.data = steering_deg
        self.cmd_pub.publish(msg)

        # --- Pretty-print (throttled to 1 Hz) ---
        self.get_logger().info(
            f'\n'
            f'  ╔══════════ LATERAL CONTROL (SMC) ══════════╗\n'
            f'  ║  Target Y   : {self.desired_y:>+7.3f} m              ║\n'
            f'  ║  Actual Y   : {self.current_y:>+7.3f} m   '
            f'X: {self.current_x:>+6.3f} m  ║\n'
            f'  ║  Heading ψ  : {math.degrees(self.current_yaw):>+7.2f}°  '
            f'Target: {math.degrees(self.desired_heading):>+5.2f}°  ║\n'
            f'  ║  e_cte      : {e_cte:>+7.3f} m              ║\n'
            f'  ║  e_heading  : {math.degrees(e_heading):>+7.2f}°              ║\n'
            f'  ║  Sliding s  : {s:>+7.4f}               ║\n'
            f'  ║  sat(s/φ)   : {sat_s:>+7.4f}               ║\n'
            f'  ║  Ctrl eff δ : {steering_deg:>+7.2f}°              ║\n'
            f'  ╚═══════════════════════════════════════════╝',
            throttle_duration_sec=1.0
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _sat(x: float) -> float:
        """Saturation function — boundary layer approximation of sign()."""
        if x > 1.0:
            return 1.0
        if x < -1.0:
            return -1.0
        return x

    @staticmethod
    def _normalise(angle: float) -> float:
        """Wrap angle to [-π, π]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


# ──────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = SimSmcLateralControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
