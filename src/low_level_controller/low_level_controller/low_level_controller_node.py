"""
Low-Level Controller Node (Open-Loop + Encoder Feedback)
========================================================
Subscribes to /vehicle/cmd (VehicleCmd) with velocity (m/s) and heading (degrees).
Maps velocity → PWM (0–255) and heading → servo angle (open-loop).
Sends serial frames to Arduino: <direction>,<pwm>,<servo_angle>\n
Reads encoder feedback from Arduino and publishes VehicleFeedback.
Includes watchdog safety: stops vehicle if no command received within timeout.
"""

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleCmd, VehicleFeedback
import serial
import serial.tools.list_ports
import time
import math
import threading


class LowLevelControllerNode(Node):

    def __init__(self):
        super().__init__('low_level_controller_node')

        # --------------- Declare ROS2 Parameters ---------------
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('max_velocity', 1.0)        # m/s
        self.declare_parameter('wheel_radius', 0.033)      # meters (for feedback conversion)
        self.declare_parameter('servo_center', 90)          # degrees (straight)
        self.declare_parameter('servo_min', 45)             # degrees (full right)
        self.declare_parameter('servo_max', 135)            # degrees (full left)
        self.declare_parameter('max_steering_angle', 30.0)  # degrees (+/- from center)
        self.declare_parameter('cmd_topic', '/vehicle/cmd')
        self.declare_parameter('feedback_topic', '/vehicle/feedback')

        # --------------- Read Parameters ---------------
        self.serial_port_name = self.get_parameter('serial_port').value
        self.baud_rate = self.get_parameter('baud_rate').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.servo_center = self.get_parameter('servo_center').value
        self.servo_min = self.get_parameter('servo_min').value
        self.servo_max = self.get_parameter('servo_max').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        cmd_topic = self.get_parameter('cmd_topic').value
        feedback_topic = self.get_parameter('feedback_topic').value

        # Precompute wheel circumference (for converting encoder RPM → m/s in feedback)
        self.wheel_circumference = 2.0 * math.pi * self.wheel_radius

        # --------------- Serial Connection ---------------
        self.serial_conn = None
        self._connect_serial()

        # --------------- Subscriber ---------------
        self.subscription = self.create_subscription(
            VehicleCmd,
            cmd_topic,
            self._cmd_callback,
            10
        )

        # --------------- Publisher (encoder feedback) ---------------
        self.feedback_pub = self.create_publisher(
            VehicleFeedback,
            feedback_topic,
            10
        )

        # --------------- Serial Reader Thread ---------------
        self._serial_thread_running = True
        self._serial_thread = threading.Thread(target=self._serial_reader, daemon=True)
        self._serial_thread.start()

        # --------------- State Tracking ---------------
        self.last_pwm = 0
        self.last_servo = self.servo_center
        self.last_direction = 1

        self.get_logger().info('=' * 55)
        self.get_logger().info('Low-Level Controller Node Started (Open-Loop)')
        self.get_logger().info(f'  Serial port   : {self.serial_port_name}')
        self.get_logger().info(f'  Baud rate     : {self.baud_rate}')
        self.get_logger().info(f'  Cmd topic     : {cmd_topic}')
        self.get_logger().info(f'  Feedback topic: {feedback_topic}')
        self.get_logger().info(f'  Max velocity  : {self.max_velocity} m/s → PWM 255')
        self.get_logger().info(f'  Wheel radius  : {self.wheel_radius} m (for feedback)')
        self.get_logger().info(f'  Servo range   : {self.servo_min}°–{self.servo_max}° (center: {self.servo_center}°)')
        self.get_logger().info(f'  Max steering  : ±{self.max_steering_angle}°')
        self.get_logger().info('=' * 55)

    # ==================== Serial Connection ====================

    def _connect_serial(self):
        """Attempt to open the serial port."""
        if self.serial_conn is not None and self.serial_conn.is_open:
            return True

        try:
            self.serial_conn = serial.Serial(
                port=self.serial_port_name,
                baudrate=self.baud_rate,
                timeout=1
            )
            time.sleep(2.0)  # Arduino reset delay
            self.get_logger().info(f'Serial connected: {self.serial_port_name} @ {self.baud_rate}')
            return True

        except serial.SerialException as e:
            self.get_logger().error(f'Failed to open serial port {self.serial_port_name}: {e}')
            self.serial_conn = None

            available = [p.device for p in serial.tools.list_ports.comports()]
            if available:
                self.get_logger().warn(f'Available serial ports: {available}')
            else:
                self.get_logger().warn('No serial ports detected')
            return False

    def _reconnect_serial(self):
        """Try to reconnect if serial connection was lost."""
        if self.serial_conn is not None:
            try:
                self.serial_conn.close()
            except Exception:
                pass
            self.serial_conn = None
        return self._connect_serial()

    # ==================== Serial Reader Thread ====================

    def _serial_reader(self):
        """Background thread: reads feedback from Arduino and publishes VehicleFeedback."""
        while self._serial_thread_running:
            if self.serial_conn is None or not self.serial_conn.is_open:
                time.sleep(0.5)
                continue

            try:
                line = self.serial_conn.readline().decode('ascii', errors='ignore').strip()
                if not line:
                    continue

                if line.startswith('FB:'):
                    self._parse_feedback(line)
                elif line == 'OK':
                    pass
                elif line == 'WATCHDOG_STOP':
                    self.get_logger().warn('Arduino watchdog triggered — motor stopped')
                elif line.startswith('ERR_PARSE:'):
                    self.get_logger().error(f'Arduino parse error: {line}')
                elif line == 'LOW_LEVEL_READY':
                    self.get_logger().info('Arduino ready')
                else:
                    self.get_logger().debug(f'Arduino: {line}')

            except serial.SerialException:
                self.get_logger().error('Serial read error — will attempt reconnect')
                time.sleep(1.0)
            except Exception as e:
                self.get_logger().error(f'Serial reader error: {e}')
                time.sleep(0.1)

    def _parse_feedback(self, line):
        """Parse 'FB:<rpm>,<ticks>' and publish VehicleFeedback."""
        try:
            data = line[3:]  # Strip "FB:"
            parts = data.split(',')
            if len(parts) != 2:
                return

            actual_rpm = float(parts[0])
            encoder_ticks = int(parts[1])

            # Convert RPM to velocity: v = (RPM * circumference) / 60
            actual_velocity = (actual_rpm * self.wheel_circumference) / 60.0

            fb_msg = VehicleFeedback()
            fb_msg.actual_velocity = actual_velocity
            fb_msg.actual_rpm = actual_rpm
            fb_msg.encoder_ticks = encoder_ticks
            self.feedback_pub.publish(fb_msg)

        except (ValueError, IndexError) as e:
            self.get_logger().warn(f'Failed to parse feedback: {line} ({e})')

    # ==================== Command Callback ====================

    def _cmd_callback(self, msg: VehicleCmd):
        """
        Callback for /vehicle/cmd topic.
        msg.velocity: m/s (positive = forward, negative = reverse)
        msg.heading:  degrees from center (positive = right, negative = left)
        """
        velocity = msg.velocity
        heading = msg.heading

        # Update watchdog
        self.last_cmd_time = time.time()

        # --- Map velocity to direction + PWM (open-loop) ---
        direction = 1 if velocity >= 0 else 0
        abs_velocity = min(abs(velocity), self.max_velocity)

        # Linear mapping: 0 m/s → PWM 0, max_velocity → PWM 255
        pwm = int((abs_velocity / self.max_velocity) * 255.0)
        pwm = max(0, min(255, pwm))

        # --- Map heading to servo angle ---
        heading = max(-self.max_steering_angle, min(self.max_steering_angle, heading))

        if heading >= 0:
            servo_angle = self.servo_center + (heading / self.max_steering_angle) * (self.servo_max - self.servo_center)
        else:
            servo_angle = self.servo_center + (heading / self.max_steering_angle) * (self.servo_center - self.servo_min)

        servo_angle = int(round(servo_angle))
        servo_angle = max(self.servo_min, min(self.servo_max, servo_angle))

        # --- Send to Arduino ---
        self._send_command(direction, pwm, servo_angle)

        # --- Log ---
        dir_str = 'FWD' if direction == 1 else 'REV'
        self.is_stopped = (pwm == 0)

        self.get_logger().info(
            f'CMD: vel={velocity:.2f}m/s heading={heading:.1f}° → '
            f'{dir_str} PWM={pwm} Servo={servo_angle}°'
        )

    # ==================== Send Serial Command ====================

    def _send_command(self, direction: int, pwm: int, servo_angle: int):
        """
        Send frame to Arduino: <direction>,<pwm>,<servo_angle>\n
        """
        self.last_direction = direction
        self.last_pwm = pwm
        self.last_servo = servo_angle

        if self.serial_conn is None or not self.serial_conn.is_open:
            if not self._reconnect_serial():
                return

        frame = f'{direction},{pwm},{servo_angle}\n'

        try:
            self.serial_conn.write(frame.encode('ascii'))
            self.serial_conn.flush()
        except serial.SerialException as e:
            self.get_logger().error(f'Serial write failed: {e}')
            self._reconnect_serial()

    # ==================== Cleanup ====================

    def destroy_node(self):
        """Clean shutdown: stop vehicle and close serial."""
        self.get_logger().info('Shutting down — sending stop command...')
        self._serial_thread_running = False
        self._send_command(1, 0, self.servo_center)

        if self.serial_conn is not None and self.serial_conn.is_open:
            self.serial_conn.close()
            self.get_logger().info('Serial port closed')

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = LowLevelControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
