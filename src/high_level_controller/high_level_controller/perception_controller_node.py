"""
Perception Controller Node (High-Level Controller)
=====================================================
Subscribes to traffic light and traffic sign states from perception,
maintains stateful velocity and turning logic, and publishes desired
speed and heading setpoints for the mid-level controllers.

This node does NOT publish VehicleCmd.  It publishes two Float64 topics:
    /hlc/desired_speed   → consumed by speed_control_node (PI)
    /hlc/desired_heading  → consumed by lateral_control_node (Stanley)

The MLC nodes compare state feedback from the localization system and
command the LLC until the target is reached (closed-loop).

Velocity State Logic:
    RED or STOP                     → STOP state
    YELLOW or SLOW_DOWN             → SLOW state
    GREEN                           → FAST state
    (Maintains current state if NO_SIGNAL or UNKNOWN)

Turning State Logic (RHR convention: CCW = positive):
    TURN_LEFT                       → target_yaw = current_yaw + π/2
    TURN_RIGHT                      → target_yaw = current_yaw - π/2
    (Locks turning state until abs(current_yaw - target_yaw) < tolerance)

Sign Latch (One-Shot):
    Once a turn sign is detected and acted on, the command is latched.
    Even if the sign remains in view, it will NOT re-trigger.
    The latch clears only after perception reports NO_SIGNAL for at
    least `sign_clear_frames` consecutive frames.
"""

import os
import yaml
import time
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
from vehicle_interfaces.msg import VehicleState
from ament_index_python.packages import get_package_share_directory


class PerceptionControllerNode(Node):
    """High-level controller that reacts to traffic lights and signs."""

    def __init__(self):
        super().__init__('perception_controller_node')

        # -- Declare parameters ----------------------------------------
        self._declare_parameters()

        # -- Read parameters -------------------------------------------
        self.cruise_velocity = self._p('cruise_velocity')
        self.slow_velocity = self._p('slow_velocity')
        self.heading = self._p('heading')
        self.unknown_timeout = self._p('unknown_timeout')
        self.turn_tolerance_deg = self._p('turn_tolerance_deg')
        self.sign_clear_frames = self._p('sign_clear_frames')

        state_topic = self._p('state_topic')
        sign_topic = self._p('sign_topic')
        vehicle_state_topic = self._p('vehicle_state_topic')
        desired_speed_topic = self._p('desired_speed_topic')
        desired_heading_topic = self._p('desired_heading_topic')
        publish_rate = self._p('publish_rate')

        # -- State tracking --------------------------------------------
        self.current_light_state = 'UNKNOWN'
        self.last_light_time = time.time()

        self.current_sign_state = 'NO_SIGNAL'
        self.last_sign_time = time.time()

        self.current_yaw = 0.0

        # Velocity state machine
        self.velocity_state = 'FAST'     # 'FAST', 'SLOW', 'STOP'

        # Turning state machine
        self.turning_state = 'STRAIGHT'  # 'STRAIGHT', 'TURNING_LEFT', 'TURNING_RIGHT'
        self.target_yaw = 0.0

        # Sign latch (one-shot) — prevents re-triggering while sign is visible
        self.sign_latch_active = False
        self.sign_cleared = True
        self.no_signal_consecutive = 0

        # Track the heading the car should hold when STRAIGHT.
        # Updated after each turn completes so the car continues
        # in the new direction (dead-reckoning, no revert).
        self.straight_heading_deg = self.heading

        # -- Subscribers -----------------------------------------------
        self.state_sub = self.create_subscription(
            String, state_topic, self._light_callback, 10
        )
        self.sign_sub = self.create_subscription(
            String, sign_topic, self._sign_callback, 10
        )
        self.vehicle_state_sub = self.create_subscription(
            VehicleState, vehicle_state_topic, self._vehicle_state_callback, 10
        )

        # -- Publishers: desired setpoints for MLC ---------------------
        self.desired_speed_pub = self.create_publisher(
            Float64, desired_speed_topic, 10
        )
        self.desired_heading_pub = self.create_publisher(
            Float64, desired_heading_topic, 10
        )

        # -- Timer for periodic decision publishing --------------------
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self._timer_callback)

        # -- Startup log -----------------------------------------------
        self.get_logger().info('=' * 58)
        self.get_logger().info('Perception Controller Node Started')
        self.get_logger().info(f'  Light topic      : {state_topic}')
        self.get_logger().info(f'  Sign topic       : {sign_topic}')
        self.get_logger().info(f'  Vehicle state    : {vehicle_state_topic}')
        self.get_logger().info(f'  Desired speed  → : {desired_speed_topic}')
        self.get_logger().info(f'  Desired heading→ : {desired_heading_topic}')
        self.get_logger().info(f'  Cruise velocity  : {self.cruise_velocity:.2f} m/s')
        self.get_logger().info(f'  Slow velocity    : {self.slow_velocity:.2f} m/s')
        self.get_logger().info(f'  Turn tolerance   : {self.turn_tolerance_deg:.1f}°')
        self.get_logger().info(f'  Sign clear count : {self.sign_clear_frames}')
        self.get_logger().info(f'  Mode             : Topic pub → MLC (closed-loop)')
        self.get_logger().info('=' * 58)

    def _declare_parameters(self):
        yaml_params = self._load_yaml_params()

        def _d(name, fallback):
            self.declare_parameter(name, yaml_params.get(name, fallback))

        _d('cruise_velocity', 0.25)
        _d('slow_velocity', 0.15)
        _d('heading', 0.0)
        _d('publish_rate', 20.0)
        _d('state_topic', '/traffic_light/state')
        _d('sign_topic', '/sign/command')
        _d('vehicle_state_topic', '/vehicle/state')
        _d('desired_speed_topic', '/hlc/desired_speed')
        _d('desired_heading_topic', '/hlc/desired_heading')
        _d('unknown_timeout', 2.0)
        _d('turn_tolerance_deg', 5.0)
        _d('sign_clear_frames', 10)

    def _load_yaml_params(self) -> dict:
        try:
            this_file = os.path.abspath(__file__)
            pkg_src_dir = os.path.dirname(os.path.dirname(this_file))
            local_yaml = os.path.join(
                pkg_src_dir, 'config', 'high_level_controller.local.yaml'
            )
            if os.path.isfile(local_yaml):
                with open(local_yaml, 'r') as f:
                    raw = yaml.safe_load(f)
                return raw.get('perception_controller_node', {}).get('ros__parameters', {})
        except Exception:
            pass

        try:
            share_dir = get_package_share_directory('high_level_controller')
            yaml_path = os.path.join(share_dir, 'config', 'high_level_controller.yaml')
            with open(yaml_path, 'r') as f:
                raw = yaml.safe_load(f)
            return raw.get('perception_controller_node', {}).get('ros__parameters', {})
        except Exception:
            return {}

    def _p(self, name):
        return self.get_parameter(name).value

    # ===================================================================
    # Callbacks
    # ===================================================================

    def _light_callback(self, msg: String):
        state = msg.data.strip().upper()
        if state not in ('RED', 'YELLOW', 'GREEN', 'UNKNOWN'):
            state = 'UNKNOWN'
        self.current_light_state = state
        self.last_light_time = time.time()

    def _sign_callback(self, msg: String):
        state = msg.data.strip().upper()
        valid_signs = ('STOP', 'SLOW_DOWN', 'TURN_LEFT', 'TURN_RIGHT', 'NO_SIGNAL')
        if state not in valid_signs:
            state = 'NO_SIGNAL'
        self.current_sign_state = state
        self.last_sign_time = time.time()

    def _vehicle_state_callback(self, msg: VehicleState):
        self.current_yaw = msg.yaw

    # ===================================================================
    # Helper Math
    # ===================================================================

    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-π, π]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    # ===================================================================
    # Decision Logic
    # ===================================================================

    def _timer_callback(self):
        now = time.time()

        # 1. Handle Timeouts (if perception nodes crash/freeze)
        if (now - self.last_light_time) > self.unknown_timeout:
            eff_light = 'UNKNOWN'
        else:
            eff_light = self.current_light_state

        if (now - self.last_sign_time) > self.unknown_timeout:
            eff_sign = 'NO_SIGNAL'
        else:
            eff_sign = self.current_sign_state

        # 2. Velocity State Machine (persists — not reset by NO_SIGNAL)
        if eff_light == 'RED' or eff_sign == 'STOP':
            self.velocity_state = 'STOP'
        elif eff_light == 'YELLOW' or eff_sign == 'SLOW_DOWN':
            self.velocity_state = 'SLOW'
        elif eff_light == 'GREEN':
            self.velocity_state = 'FAST'
        # else: keep current velocity_state (UNKNOWN light / NO_SIGNAL sign)

        # Determine target speed from state
        if self.velocity_state == 'STOP':
            target_speed = 0.0
        elif self.velocity_state == 'SLOW':
            target_speed = self.slow_velocity
        else:  # 'FAST'
            target_speed = self.cruise_velocity

        # Publish desired speed
        speed_msg = Float64()
        speed_msg.data = float(target_speed)
        self.desired_speed_pub.publish(speed_msg)

        # 3. Sign Latch Logic — track consecutive NO_SIGNAL frames
        if eff_sign == 'NO_SIGNAL':
            self.no_signal_consecutive += 1
            if self.no_signal_consecutive >= self.sign_clear_frames:
                if self.sign_latch_active:
                    self.get_logger().info(
                        f'[HLC] Sign latch CLEARED after {self.no_signal_consecutive} '
                        f'NO_SIGNAL frames')
                self.sign_latch_active = False
                self.sign_cleared = True
        else:
            self.no_signal_consecutive = 0

        # 4. Turning State Machine (RHR: +ve = left, -ve = right)
        if self.turning_state == 'STRAIGHT':
            # Only accept a new turn command if the latch is not active
            if eff_sign in ('TURN_LEFT', 'TURN_RIGHT') and not self.sign_latch_active:
                if eff_sign == 'TURN_LEFT':
                    self.turning_state = 'TURNING_LEFT'
                    # +90 degrees = +π/2 (left in RHR convention)
                    self.target_yaw = self._normalize_angle(
                        self.current_yaw + math.pi / 2.0)
                else:
                    self.turning_state = 'TURNING_RIGHT'
                    # -90 degrees = -π/2 (right in RHR convention)
                    self.target_yaw = self._normalize_angle(
                        self.current_yaw - math.pi / 2.0)

                # Engage the latch — prevent re-triggering
                self.sign_latch_active = True
                self.sign_cleared = False
                self.no_signal_consecutive = 0

                self.get_logger().info(
                    f'[HLC] TURN initiated: {self.turning_state} | '
                    f'current_yaw={math.degrees(self.current_yaw):.1f}° → '
                    f'target_yaw={math.degrees(self.target_yaw):.1f}° | '
                    f'LATCH ENGAGED')

            # Straight path: use the current straight heading
            target_heading_deg = self.straight_heading_deg
        else:
            # We are currently turning. Check for completion.
            heading_error = self._normalize_angle(
                self.target_yaw - self.current_yaw)

            if abs(math.degrees(heading_error)) < self.turn_tolerance_deg:
                # Turn is complete — update straight heading to the NEW direction
                self.straight_heading_deg = math.degrees(self.target_yaw)
                self.turning_state = 'STRAIGHT'

                self.get_logger().info(
                    f'[HLC] TURN COMPLETE | '
                    f'New straight heading: {self.straight_heading_deg:.1f}° | '
                    f'Latch still active (waiting for sign to leave view)')

                target_heading_deg = self.straight_heading_deg
            else:
                # Still turning — set desired heading to the target yaw
                target_heading_deg = math.degrees(self.target_yaw)

        # Publish desired heading (degrees)
        heading_msg = Float64()
        heading_msg.data = float(target_heading_deg)
        self.desired_heading_pub.publish(heading_msg)

        # 5. Terminal log
        light_icon = {
            'RED': '🔴', 'YELLOW': '🟡', 'GREEN': '🟢', 'UNKNOWN': '⚪'
        }.get(eff_light, '?')
        sign_icon = {
            'STOP': '🛑', 'SLOW_DOWN': '⚠️', 'TURN_LEFT': '⬅️',
            'TURN_RIGHT': '➡️', 'NO_SIGNAL': '➖'
        }.get(eff_sign, '?')

        hdg_err_deg = math.degrees(
            self._normalize_angle(self.target_yaw - self.current_yaw))
        latch_str = '🔒LATCH' if self.sign_latch_active else '🔓free'

        self.get_logger().info(
            f'[HLC] {light_icon}{eff_light:7s} | {sign_icon}{eff_sign:10s} || '
            f'Vel: {self.velocity_state} ({target_speed:.2f}m/s) | '
            f'Turn: {self.turning_state} '
            f'(tg={math.degrees(self.target_yaw):.0f}° '
            f'err={hdg_err_deg:.0f}°) → '
            f'hdg={target_heading_deg:.0f}° | {latch_str}',
            throttle_duration_sec=0.5
        )

    def destroy_node(self):
        self.get_logger().info('Shutting down HLC — sending stop...')

        # Publish zero speed
        stop_msg = Float64()
        stop_msg.data = 0.0
        self.desired_speed_pub.publish(stop_msg)

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
