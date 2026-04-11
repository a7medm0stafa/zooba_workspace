"""
PS4/PS5 Controller Teleoperation Node
======================================
Reads PS4/PS5 gamepad input via the ROS2 joy topic and publishes
VehicleCmd messages on /teleop/raw_cmd.

Controls:
    R2 Trigger         : forward throttle (progressive)
    L2 Trigger         : reverse throttle (progressive)
    Left Stick X-axis  : steering (heading angle)
    X (✕) Button       : emergency stop
    Triangle (△)       : toggle speed mode (slow / normal / fast)
    L1 / R1 Bumpers    : fine-tune heading left / right
    Circle (○) Button  : center steering (heading = 0)

Requires: ros2 joy package (sudo apt install ros2-${ROS_DISTRO}-joy)
Run joy_node first: ros2 run joy joy_node
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from vehicle_interfaces.msg import VehicleCmd


# PS4/PS5 button mapping (may vary — use `ros2 topic echo /joy` to verify)
# These are the default mappings for ds4/ds5 on Linux with the joy package
class PS4Buttons:
    CROSS = 0        # X / ✕
    CIRCLE = 1       # ○
    TRIANGLE = 2     # △
    SQUARE = 3       # □
    L1 = 4
    R1 = 5
    L2_BUTTON = 6    # L2 as digital button
    R2_BUTTON = 7    # R2 as digital button
    SHARE = 8
    OPTIONS = 9
    PS = 10
    L3 = 11          # Left stick press
    R3 = 12          # Right stick press


class PS4Axes:
    LEFT_STICK_X = 0   # Left/Right  (-1.0 = right, 1.0 = left)
    LEFT_STICK_Y = 1   # Up/Down     (-1.0 = down,  1.0 = up)
    L2_TRIGGER = 2     # 1.0 = released, -1.0 = fully pressed
    RIGHT_STICK_X = 3  # Left/Right
    RIGHT_STICK_Y = 4  # Up/Down
    R2_TRIGGER = 5     # 1.0 = released, -1.0 = fully pressed
    DPAD_X = 6         # -1.0 = left, 1.0 = right
    DPAD_Y = 7         # -1.0 = down, 1.0 = up


# Speed mode presets
SPEED_MODES = [
    {'name': 'SLOW',   'max_velocity': 0.5, 'max_heading': 20.0},
    {'name': 'NORMAL', 'max_velocity': 1.0, 'max_heading': 30.0},
    {'name': 'FAST',   'max_velocity': 2.0, 'max_heading': 30.0},
]


class TeleopPSNode(Node):

    def __init__(self):
        super().__init__('teleop_ps_node')

        # ---- Parameters ----
        self.declare_parameter('publish_rate', 20.0)         # Hz
        self.declare_parameter('max_velocity', 2.0)          # m/s
        self.declare_parameter('max_heading', 30.0)          # degrees
        self.declare_parameter('deadzone', 0.05)             # stick deadzone
        self.declare_parameter('heading_fine_step', 2.0)     # degrees per L1/R1 press
        self.declare_parameter('output_topic', '/vehicle/cmd')
        self.declare_parameter('joy_topic', '/joy')

        self.publish_rate = self.get_parameter('publish_rate').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.max_heading = self.get_parameter('max_heading').value
        self.deadzone = self.get_parameter('deadzone').value
        self.heading_fine_step = self.get_parameter('heading_fine_step').value
        output_topic = self.get_parameter('output_topic').value
        joy_topic = self.get_parameter('joy_topic').value

        # ---- State ----
        self.current_velocity = 0.0
        self.current_heading = 0.0
        self.speed_mode_idx = 1  # Start in NORMAL
        self.e_stop = False
        self.joy_connected = False

        # Button edge detection (for toggle buttons)
        self.prev_buttons = {}

        # ---- Publisher ----
        self.cmd_pub = self.create_publisher(VehicleCmd, output_topic, 10)

        # ---- Subscriber (joy topic) ----
        self.joy_sub = self.create_subscription(Joy, joy_topic, self._joy_callback, 10)

        # ---- Timer for periodic publishing ----
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self._publish_cmd)

        self.get_logger().info('=' * 55)
        self.get_logger().info('PS Controller Teleop Node Started')
        self.get_logger().info(f'  Joy topic    : {joy_topic}')
        self.get_logger().info(f'  Output topic : {output_topic}')
        self.get_logger().info(f'  Publish rate : {self.publish_rate} Hz')
        self.get_logger().info(f'  Speed mode   : {SPEED_MODES[self.speed_mode_idx]["name"]}')
        self.get_logger().info(f'  Deadzone     : {self.deadzone}')
        self.get_logger().info('=' * 55)
        self.get_logger().info('Waiting for PS controller input on /joy...')
        self.get_logger().info('  R2=forward  L2=reverse  LeftStick X=steering')
        self.get_logger().info('  X=E-STOP  △=speed mode  ○=center steering')

    # ==================== Joy Callback ====================

    def _joy_callback(self, msg: Joy):
        """Process incoming joystick message from the joy_node."""
        if not self.joy_connected:
            self.joy_connected = True
            self.get_logger().info('PS Controller connected!')
            # Debug: log raw axes/buttons on first message to help verify mapping
            self.get_logger().info(f'  Axes ({len(msg.axes)}): {[round(a, 2) for a in msg.axes]}')
            self.get_logger().info(f'  Buttons ({len(msg.buttons)}): {list(msg.buttons)}')

        axes = msg.axes
        buttons = msg.buttons

        # Safety check
        if len(axes) < 2 or len(buttons) < 3:
            self.get_logger().warn(f'Unexpected joy message: {len(axes)} axes, {len(buttons)} buttons')
            return

        mode = SPEED_MODES[self.speed_mode_idx]

        # ---- Handle buttons (edge detection) ----

        # Emergency stop (X / Cross button)
        if self._button_pressed(buttons, PS4Buttons.CROSS):
            self.e_stop = not self.e_stop
            if self.e_stop:
                self.current_velocity = 0.0
                self.current_heading = 0.0
                self.get_logger().warn('*** EMERGENCY STOP ACTIVATED ***  (press X again to release)')
            else:
                self.get_logger().info('Emergency stop released — controller active')

        # Speed mode toggle (Triangle button)
        if self._button_pressed(buttons, PS4Buttons.TRIANGLE):
            self.speed_mode_idx = (self.speed_mode_idx + 1) % len(SPEED_MODES)
            mode = SPEED_MODES[self.speed_mode_idx]
            self.get_logger().info(f'Speed mode: {mode["name"]} (max_vel={mode["max_velocity"]}m/s, max_hdg={mode["max_heading"]}°)')

        # Center steering (Circle button)
        if self._button_pressed(buttons, PS4Buttons.CIRCLE):
            self.current_heading = 0.0
            self.get_logger().info('Steering centered')

        # Save current button state for edge detection
        self.prev_buttons = {i: b for i, b in enumerate(buttons)}

        # If e-stop is active, don't process movement
        if self.e_stop:
            self.current_velocity = 0.0
            self.current_heading = 0.0
            return

        # ---- Throttle (R2 = forward, L2 = reverse) ----
        # Triggers: 1.0 = released, -1.0 = fully pressed
        # Remap to 0.0 (released) → 1.0 (fully pressed)
        r2_raw = axes[PS4Axes.R2_TRIGGER]
        l2_raw = axes[PS4Axes.L2_TRIGGER]
        r2_value = self._trigger_to_throttle(r2_raw)
        l2_value = self._trigger_to_throttle(l2_raw)

        # Net velocity: R2 forward, L2 reverse
        velocity = (r2_value - l2_value) * mode['max_velocity']
        self.current_velocity = max(-mode['max_velocity'], min(mode['max_velocity'], velocity))

        # ---- Steering (Left Stick X-axis) ----
        stick_x = axes[PS4Axes.LEFT_STICK_X]

        # Apply deadzone
        if abs(stick_x) < self.deadzone:
            stick_x = 0.0

        # Left stick: positive = left, but our heading: positive = right
        # So invert: heading = -stick_x * max_heading
        heading = -stick_x * mode['max_heading']

        # Fine-tune with bumpers (L1 = left, R1 = right)
        if len(buttons) > PS4Buttons.R1:
            if buttons[PS4Buttons.L1]:
                heading -= self.heading_fine_step
            if buttons[PS4Buttons.R1]:
                heading += self.heading_fine_step

        self.current_heading = max(-mode['max_heading'], min(mode['max_heading'], heading))

        # ---- Debug log (throttled to avoid spam) ----
        if abs(self.current_velocity) > 0.01 or abs(self.current_heading) > 0.5:
            self.get_logger().info(
                f'R2={r2_raw:.2f}→{r2_value:.2f}  L2={l2_raw:.2f}→{l2_value:.2f}  '
                f'vel={self.current_velocity:.2f}m/s  hdg={self.current_heading:.1f}°'
            )

    # ==================== Helpers ====================

    def _button_pressed(self, buttons, idx):
        """Detect rising edge (button just pressed, not held)."""
        if idx >= len(buttons):
            return False
        current = buttons[idx]
        previous = self.prev_buttons.get(idx, 0)
        return current == 1 and previous == 0

    def _trigger_to_throttle(self, trigger_value):
        """
        Convert trigger axis value to throttle 0.0-1.0.
        Joy trigger: 1.0 = released, -1.0 = fully pressed.
        Output:      0.0 = released,  1.0 = fully pressed.
        """
        throttle = (1.0 - trigger_value) / 2.0
        if throttle < self.deadzone:
            throttle = 0.0
        return throttle

    # ==================== Publish ====================

    def _publish_cmd(self):
        """Timer callback: publish current velocity and heading."""
        msg = VehicleCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.velocity = self.current_velocity
        msg.heading = self.current_heading
        self.cmd_pub.publish(msg)

    # ==================== Cleanup ====================

    def destroy_node(self):
        """Send stop command before shutting down."""
        msg = VehicleCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.velocity = 0.0
        msg.heading = 0.0
        self.cmd_pub.publish(msg)
        self.get_logger().info('Teleop PS node shutdown — stop command sent')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = TeleopPSNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
