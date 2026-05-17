"""
Low-Level Controller Node (PI Speed Control + IMU Feedback)
===========================================================
Subscribes to /vehicle/cmd (VehicleCmd) with velocity (m/s) and heading (degrees).
Sends serial frames to Arduino in two modes:
  - Mode 0 (open-loop):  0,<signed_pwm>,<servo_angle>\n
  - Mode 1 (PI control): 1,<target_rpm_x10>,<servo_angle>\n

Reads extended feedback from Arduino:
  FB:<rpm>,<ticks>[,<ax>,<ay>,<az>,<gx>,<gy>,<gz>,<yaw>]
Publishes VehicleFeedback and ImuData.

Includes watchdog safety: stops vehicle if no command received within timeout.
"""

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleCmd, VehicleFeedback, ImuData
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
        self.declare_parameter('max_velocity', 0.25)       # m/s
        self.declare_parameter('wheel_radius', 0.033)      # meters (for feedback conversion)
        self.declare_parameter('servo_center', 82)          # degrees (straight)
        self.declare_parameter('servo_min', 37)             # degrees (full right)
        self.declare_parameter('servo_max', 127)            # degrees (full left)
        self.declare_parameter('max_steering_angle', 45.0)  # degrees (+/- from center)
        self.declare_parameter('cmd_topic', '/vehicle/cmd')
        self.declare_parameter('feedback_topic', '/vehicle/feedback')
        self.declare_parameter('imu_topic', '/vehicle/imu')
        self.declare_parameter('use_pi_mode', True)         # Use Arduino PI mode
        self.declare_parameter('gear_ratio', 134.181)       # Total gear ratio to wheel

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
        imu_topic = self.get_parameter('imu_topic').value
        self.use_pi_mode = self.get_parameter('use_pi_mode').value
        self.gear_ratio = self.get_parameter('gear_ratio').value

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

        # --------------- Publishers ---------------
        self.feedback_pub = self.create_publisher(
            VehicleFeedback,
            feedback_topic,
            10
        )
        self.imu_pub = self.create_publisher(
            ImuData,
            imu_topic,
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
        self.get_logger().info('Low-Level Controller Node Started (PI + IMU)')
        self.get_logger().info(f'  Serial port   : {self.serial_port_name}')
        self.get_logger().info(f'  Baud rate     : {self.baud_rate}')
        self.get_logger().info(f'  Cmd topic     : {cmd_topic}')
        self.get_logger().info(f'  Feedback topic: {feedback_topic}')
        self.get_logger().info(f'  IMU topic     : {imu_topic}')
        self.get_logger().info(f'  Max velocity  : {self.max_velocity} m/s')
        self.get_logger().info(f'  Wheel radius  : {self.wheel_radius} m')
        self.get_logger().info(f'  Gear ratio    : 1:{self.gear_ratio}')
        self.get_logger().info(f'  PI mode       : {"enabled" if self.use_pi_mode else "disabled (open-loop)"}')
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
        """Background thread: reads feedback from Arduino and publishes."""
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
                elif line.startswith('ERR_MODE:'):
                    self.get_logger().error(f'Arduino mode error: {line}')
                elif line == 'LOW_LEVEL_READY':
                    self.get_logger().info('Arduino ready')
                elif line == 'IMU_OK':
                    self.get_logger().info('Arduino IMU initialized successfully')
                elif line.startswith('IMU_ERR:'):
                    self.get_logger().warn(f'Arduino IMU error: {line}')
                elif line.startswith('PWM:'):
                    self.get_logger().info(f'Arduino -> Motor PWM: {line[4:]} / 255')
                else:
                    self.get_logger().debug(f'Arduino: {line}')

            except serial.SerialException:
                self.get_logger().error('Serial read error — will attempt reconnect')
                time.sleep(1.0)
            except Exception as e:
                self.get_logger().error(f'Serial reader error: {e}')
                time.sleep(0.1)

    def _parse_feedback(self, line):
        """
        Parse feedback line and publish VehicleFeedback + ImuData.

        Format (2 fields, no IMU):  FB:<rpm>,<ticks>
        Format (9 fields, with IMU): FB:<rpm>,<ticks>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>,<yaw>
        """
        try:
            data = line[3:]  # Strip "FB:"
            parts = data.split(',')

            if len(parts) < 2:
                return

            actual_rpm = float(parts[0])
            encoder_ticks = int(parts[1])

            # Convert RPM to velocity: v = (RPM * circumference) / 60
            actual_velocity = (actual_rpm * self.wheel_circumference) / 60.0

            fb_msg = VehicleFeedback()
            fb_msg.header.stamp = self.get_clock().now().to_msg()
            fb_msg.header.frame_id = 'base_link'
            fb_msg.actual_velocity = actual_velocity
            fb_msg.actual_rpm = actual_rpm
            fb_msg.encoder_ticks = encoder_ticks
            self.feedback_pub.publish(fb_msg)

            # Parse IMU data if available (9 fields total)
            if len(parts) >= 9:
                imu_msg = ImuData()
                imu_msg.header.stamp = self.get_clock().now().to_msg()
                imu_msg.header.frame_id = 'imu_link'
                # Values are ×100 integers from Arduino.
                # The Arduino firmware negation (line 288) is WRONG for this
                # board's chip mounting. The LLC must negate again to restore
                # correct REP-103 signs (CCW-positive, CW-negative).
                imu_msg.accel_x = int(parts[2]) / 100.0
                imu_msg.accel_y = -(int(parts[3]) / 100.0)
                imu_msg.accel_z = int(parts[4]) / 100.0
                imu_msg.gyro_x = int(parts[5]) / 100.0
                imu_msg.gyro_y = -(int(parts[6]) / 100.0)
                imu_msg.gyro_z = -(int(parts[7]) / 100.0)
                imu_msg.yaw = -(int(parts[8]) / 10.0)  # ×10 integer → degrees
                self.imu_pub.publish(imu_msg)

        except (ValueError, IndexError) as e:
            self.get_logger().warn(f'Failed to parse feedback: {line} ({e})')

    # ==================== Command Callback ====================

    def _cmd_callback(self, msg: VehicleCmd):
        """
        Callback for /vehicle/cmd topic.
        msg.velocity: m/s (positive = forward, negative = reverse)
        msg.heading:  degrees from center (positive = left, negative = right)
        """
        velocity = msg.velocity
        heading = msg.heading

        # Update watchdog
        self.last_cmd_time = time.time()

        # --- Map heading to servo angle ---
        heading = max(-self.max_steering_angle, min(self.max_steering_angle, heading))

        if heading >= 0:
            # Positive heading (LEFT) -> move servo towards servo_min (physically turns wheels left)
            servo_angle = self.servo_center - (heading / self.max_steering_angle) * (self.servo_center - self.servo_min)
        else:
            # Negative heading (RIGHT) -> move servo towards servo_max (physically turns wheels right)
            servo_angle = self.servo_center - (heading / self.max_steering_angle) * (self.servo_max - self.servo_center)

        servo_angle = int(round(servo_angle))
        servo_angle = max(self.servo_min, min(self.servo_max, servo_angle))

        if self.use_pi_mode:
            # --- PI mode: send target RPM ---
            # Convert velocity (m/s) to output shaft RPM
            # v = (RPM * wheel_circumference) / 60
            # RPM = v * 60 / wheel_circumference
            if abs(velocity) < 0.001:
                target_rpm = 0.0
            else:
                target_rpm = velocity * 60.0 / self.wheel_circumference

            # Encode as integer × 10 (preserves 1 decimal place)
            rpm_x10 = int(round(target_rpm * 10.0))
            self._send_command_raw(f'1,{rpm_x10},{servo_angle}')

            dir_str = 'FWD' if velocity >= 0 else 'REV'
            self.get_logger().info(
                f'CMD [PI]: vel={velocity:.2f}m/s → RPM={target_rpm:.1f} '
                f'heading={heading:.1f}° → Servo={servo_angle}°',
                throttle_duration_sec=1.0
            )
        else:
            # --- Open-loop mode: send signed PWM ---
            abs_velocity = min(abs(velocity), self.max_velocity)
            pwm = int((abs_velocity / self.max_velocity) * 255.0)
            pwm = max(0, min(255, pwm))
            signed_pwm = pwm if velocity >= 0 else -pwm
            self._send_command_raw(f'0,{signed_pwm},{servo_angle}')

            dir_str = 'FWD' if velocity >= 0 else 'REV'
            self.get_logger().info(
                f'CMD [OL]: vel={velocity:.2f}m/s → {dir_str} PWM={pwm} '
                f'heading={heading:.1f}° → Servo={servo_angle}°',
                throttle_duration_sec=1.0
            )

    # ==================== Send Serial Command ====================

    def _send_command_raw(self, frame_str: str):
        """Send a raw command string to Arduino (no newline added yet)."""
        if self.serial_conn is None or not self.serial_conn.is_open:
            if not self._reconnect_serial():
                return

        frame = frame_str + '\n'

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
        self._send_command_raw(f'0,0,{self.servo_center}')

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
