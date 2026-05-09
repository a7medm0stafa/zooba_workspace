"""
Keyboard Teleoperation Node for Ackermann Vehicle
==================================================
Reads keyboard input and publishes VehicleCmd messages on /teleop/raw_cmd.

Controls:
    W / ↑   : increase velocity
    S / ↓   : decrease velocity
    A / ←   : steer left  (decrease heading)
    D / →   : steer right (increase heading)
    Space   : emergency stop (velocity=0, heading=0)
    Q       : quit

Publishes raw (unconstrained) commands. The nonholonomic_constraints_node
downstream will enforce physical limits before forwarding to /vehicle/cmd.
"""

import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleCmd


# Key constants for arrow keys (escape sequences)
ARROW_PREFIX = '\x1b'

HELP_TEXT = """
╔══════════════════════════════════════════╗
║     Keyboard Teleop — Ackermann Vehicle  ║
╠══════════════════════════════════════════╣
║                                          ║
║        W / ↑   : accelerate              ║
║        S / ↓   : decelerate / reverse    ║
║        A / ←   : steer left              ║
║        D / →   : steer right             ║
║        SPACE   : emergency stop           ║
║        Q       : quit                    ║
║                                          ║
╚══════════════════════════════════════════╝
"""


class TeleopKeyboardNode(Node):

    def __init__(self):
        super().__init__('teleop_keyboard_node')

        # ---- Parameters ----
        self.declare_parameter('publish_rate', 10.0)         # Hz
        self.declare_parameter('velocity_step', 0.1)         # m/s per key press
        self.declare_parameter('heading_step', 5.0)          # degrees per key press
        self.declare_parameter('max_velocity', 2.0)          # m/s (soft limit for display)
        self.declare_parameter('max_heading', 35.0)          # degrees (soft limit for display)
        self.declare_parameter('output_topic', '/teleop/raw_cmd')

        self.publish_rate = self.get_parameter('publish_rate').value
        self.velocity_step = self.get_parameter('velocity_step').value
        self.heading_step = self.get_parameter('heading_step').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.max_heading = self.get_parameter('max_heading').value
        output_topic = self.get_parameter('output_topic').value

        # ---- State ----
        self.current_velocity = 0.0
        self.current_heading = 0.0

        # ---- Publisher ----
        self.cmd_pub = self.create_publisher(VehicleCmd, output_topic, 10)

        # ---- Timer for periodic publishing ----
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self._publish_cmd)

        # ---- Terminal settings for raw input ----
        self.old_settings = termios.tcgetattr(sys.stdin)

        self.get_logger().info('Teleop Keyboard Node started')
        self.get_logger().info(f'  Publishing on: {output_topic}')
        self.get_logger().info(f'  Rate: {self.publish_rate} Hz')
        print(HELP_TEXT)
        self._print_status()

    def _get_key(self):
        """Non-blocking key read from stdin."""
        if select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1)
            # Handle arrow key escape sequences
            if key == ARROW_PREFIX:
                # Read the remaining two characters of the escape sequence
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    key += sys.stdin.read(1)
                    if select.select([sys.stdin], [], [], 0.05)[0]:
                        key += sys.stdin.read(1)
            return key
        return None

    def _publish_cmd(self):
        """Timer callback: read key and publish command."""
        key = self._get_key()

        if key is not None:
            self._process_key(key)

        # Always publish current state
        msg = VehicleCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.velocity = self.current_velocity
        msg.heading = self.current_heading
        self.cmd_pub.publish(msg)

    def _process_key(self, key):
        """Update velocity/heading based on key input."""
        changed = False

        # Forward / accelerate
        if key in ('w', 'W', '\x1b[A'):  # W or Up arrow
            self.current_velocity = min(
                self.current_velocity + self.velocity_step,
                self.max_velocity
            )
            changed = True

        # Reverse / decelerate
        elif key in ('s', 'S', '\x1b[B'):  # S or Down arrow
            self.current_velocity = max(
                self.current_velocity - self.velocity_step,
                -self.max_velocity
            )
            changed = True

        # Steer left (positive heading)
        elif key in ('a', 'A', '\x1b[D'):  # A or Left arrow
            self.current_heading = min(
                self.current_heading + self.heading_step,
                self.max_heading
            )
            changed = True

        # Steer right (negative heading)
        elif key in ('d', 'D', '\x1b[C'):  # D or Right arrow
            self.current_heading = max(
                self.current_heading - self.heading_step,
                -self.max_heading
            )
            changed = True

        # Emergency stop
        elif key == ' ':
            self.current_velocity = 0.0
            self.current_heading = 0.0
            changed = True
            print('\r\n  *** EMERGENCY STOP ***')

        # Quit
        elif key in ('q', 'Q'):
            self.current_velocity = 0.0
            self.current_heading = 0.0
            # Publish stop command before quitting
            msg = VehicleCmd()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.velocity = 0.0
            msg.heading = 0.0
            self.cmd_pub.publish(msg)
            print('\r\n  Quitting teleop...')
            raise SystemExit

        if changed:
            self._print_status()

    def _print_status(self):
        """Print current velocity and heading to terminal."""
        direction = 'FWD' if self.current_velocity >= 0 else 'REV'
        bar_len = 20
        vel_frac = abs(self.current_velocity) / self.max_velocity if self.max_velocity > 0 else 0
        vel_bars = int(vel_frac * bar_len)
        hdg_frac = abs(self.current_heading) / self.max_heading if self.max_heading > 0 else 0
        hdg_bars = int(hdg_frac * bar_len)

        hdg_dir = 'L' if self.current_heading < 0 else ('R' if self.current_heading > 0 else '-')

        vel_bar = '█' * vel_bars + '░' * (bar_len - vel_bars)
        hdg_bar = '█' * hdg_bars + '░' * (bar_len - hdg_bars)

        print(
            f'\r  Vel: {self.current_velocity:+6.2f} m/s [{vel_bar}] {direction}'
            f'  |  Hdg: {self.current_heading:+6.1f}° [{hdg_bar}] {hdg_dir}   ',
            end='', flush=True
        )

    def destroy_node(self):
        """Restore terminal settings on shutdown."""
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        except Exception:
            pass
        print('\r\n')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = TeleopKeyboardNode()

    # Switch terminal to raw mode for non-blocking key reads
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
