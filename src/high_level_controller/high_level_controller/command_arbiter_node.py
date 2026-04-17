"""
Command Arbiter Node
====================
Merges joystick (manual) and autonomous driving commands with
perception-based safety overrides.

Behaviour:
    - If joystick is active (received within `joy_timeout`), use joystick
      velocity/heading as the base command.
    - BUT perception safety overrides always apply:
        • STOP or RED traffic light  → force velocity to 0.0
        • SLOW_DOWN or YELLOW        → clamp velocity to slow_velocity
    - If joystick is inactive, pass through autonomous commands directly.

Subscribes:
    /teleop/joy_cmd              (VehicleCmd)  – raw joystick commands
    /teleop/auto_cmd             (VehicleCmd)  – autonomous commands from HLC
    /traffic_light/state         (String)      – traffic light state
    /sign/command                (String)      – sign detection state

Publishes:
    /teleop/raw_cmd              (VehicleCmd)  – merged output to mid-level
"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vehicle_interfaces.msg import VehicleCmd


class CommandArbiterNode(Node):
    """Merges joystick + autonomous commands with perception safety."""

    def __init__(self):
        super().__init__('command_arbiter_node')

        # -- Parameters ------------------------------------------------
        self.declare_parameter('joy_topic', '/teleop/joy_cmd')
        self.declare_parameter('auto_topic', '/teleop/auto_cmd')
        self.declare_parameter('output_topic', '/teleop/raw_cmd')
        self.declare_parameter('state_topic', '/traffic_light/state')
        self.declare_parameter('sign_topic', '/sign/command')
        self.declare_parameter('joy_timeout', 0.5)
        self.declare_parameter('slow_velocity', 0.3)
        self.declare_parameter('publish_rate', 20.0)

        joy_topic = self.get_parameter('joy_topic').value
        auto_topic = self.get_parameter('auto_topic').value
        output_topic = self.get_parameter('output_topic').value
        state_topic = self.get_parameter('state_topic').value
        sign_topic = self.get_parameter('sign_topic').value
        self.joy_timeout = self.get_parameter('joy_timeout').value
        self.slow_velocity = self.get_parameter('slow_velocity').value
        publish_rate = self.get_parameter('publish_rate').value

        # -- State -----------------------------------------------------
        self.joy_cmd = VehicleCmd()
        self.auto_cmd = VehicleCmd()
        self.last_joy_time = 0.0
        self.last_auto_time = 0.0

        self.current_light_state = 'UNKNOWN'
        self.current_sign_state = 'NO_SIGNAL'

        # -- Subscribers -----------------------------------------------
        self.joy_sub = self.create_subscription(
            VehicleCmd, joy_topic, self._joy_callback, 10
        )
        self.auto_sub = self.create_subscription(
            VehicleCmd, auto_topic, self._auto_callback, 10
        )
        self.light_sub = self.create_subscription(
            String, state_topic, self._light_callback, 10
        )
        self.sign_sub = self.create_subscription(
            String, sign_topic, self._sign_callback, 10
        )

        # -- Publisher -------------------------------------------------
        self.cmd_pub = self.create_publisher(VehicleCmd, output_topic, 10)

        # -- Timer -----------------------------------------------------
        self.timer = self.create_timer(1.0 / publish_rate,
                                       self._timer_callback)

        # -- Startup log -----------------------------------------------
        self.get_logger().info('=' * 58)
        self.get_logger().info('Command Arbiter Node Started')
        self.get_logger().info(f'  Joy input  : {joy_topic}')
        self.get_logger().info(f'  Auto input : {auto_topic}')
        self.get_logger().info(f'  Output     : {output_topic}')
        self.get_logger().info(f'  Joy timeout: {self.joy_timeout:.2f}s')
        self.get_logger().info(f'  Slow vel   : {self.slow_velocity:.2f} m/s')
        self.get_logger().info('=' * 58)

    # ==================================================================
    # Callbacks
    # ==================================================================

    def _joy_callback(self, msg: VehicleCmd):
        self.joy_cmd = msg
        self.last_joy_time = time.time()

    def _auto_callback(self, msg: VehicleCmd):
        self.auto_cmd = msg
        self.last_auto_time = time.time()

    def _light_callback(self, msg: String):
        state = msg.data.strip().upper()
        if state not in ('RED', 'YELLOW', 'GREEN', 'UNKNOWN'):
            state = 'UNKNOWN'
        self.current_light_state = state

    def _sign_callback(self, msg: String):
        state = msg.data.strip().upper()
        valid = ('STOP', 'SLOW_DOWN', 'TURN_LEFT', 'TURN_RIGHT', 'NO_SIGNAL')
        if state not in valid:
            state = 'NO_SIGNAL'
        self.current_sign_state = state

    # ==================================================================
    # Arbiter logic
    # ==================================================================

    def _timer_callback(self):
        now = time.time()

        # Determine if joystick is active
        joy_active = (now - self.last_joy_time) < self.joy_timeout

        # Select base command source
        if joy_active:
            base_velocity = self.joy_cmd.velocity
            base_heading = self.joy_cmd.heading
            source = 'JOY'
        else:
            base_velocity = self.auto_cmd.velocity
            base_heading = self.auto_cmd.heading
            source = 'AUTO'

        # -- Apply perception safety overrides -------------------------
        # These ALWAYS apply regardless of command source
        light = self.current_light_state
        sign = self.current_sign_state

        override = ''

        if light == 'RED' or sign == 'STOP':
            base_velocity = 0.0
            override = 'STOP_OVERRIDE'
        elif sign == 'SLOW_DOWN' or light == 'YELLOW':
            if abs(base_velocity) > self.slow_velocity:
                base_velocity = self.slow_velocity if base_velocity > 0 else -self.slow_velocity
                override = 'SLOW_OVERRIDE'

        # -- Publish merged command ------------------------------------
        out = VehicleCmd()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_link'
        out.velocity = base_velocity
        out.heading = base_heading
        self.cmd_pub.publish(out)

        # -- Log (throttled) -------------------------------------------
        light_icon = {'RED': '🔴', 'YELLOW': '🟡', 'GREEN': '🟢',
                      'UNKNOWN': '⚪'}.get(light, '?')
        sign_icon = {'STOP': '🛑', 'SLOW_DOWN': '⚠️', 'TURN_LEFT': '⬅️',
                     'TURN_RIGHT': '➡️', 'NO_SIGNAL': '➖'}.get(sign, '?')

        override_str = f' [{override}]' if override else ''
        self.get_logger().info(
            f'[ARB] {source} | {light_icon}{light:7s} | {sign_icon}{sign:10s} || '
            f'v={base_velocity:.2f} h={base_heading:.0f}°{override_str}',
            throttle_duration_sec=1.0
        )

    def destroy_node(self):
        self.get_logger().info('Shutting down — sending stop command...')
        msg = VehicleCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.velocity = 0.0
        msg.heading = 0.0
        self.cmd_pub.publish(msg)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CommandArbiterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
