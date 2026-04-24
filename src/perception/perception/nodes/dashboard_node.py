"""
Dashboard Node — Unified HUD Overlay
=====================================
Subscribes to the camera feed and all system topics, then renders a
single OpenCV window that looks like a vehicle FOV with all telemetry
overlaid.

Subscribes:
    /camera/image_raw          (sensor_msgs/Image)      – live camera feed
    /traffic_light/state       (std_msgs/String)         – traffic light state
    /sign/command              (std_msgs/String)         – sign detection
    /vehicle/cmd               (vehicle_interfaces/VehicleCmd)   – constrained cmd
    /vehicle/feedback          (vehicle_interfaces/VehicleFeedback) – encoder feedback
    /teleop/raw_cmd            (vehicle_interfaces/VehicleCmd)   – arbiter output

Displays:
    - Full camera feed as background
    - Traffic light state indicator (coloured circle)
    - Sign detection state
    - Current velocity / target velocity
    - Steering angle (visual indicator)
    - Encoder RPM
    - Command source (JOY / AUTO)
    - FPS counter
"""

import time
import math

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vehicle_interfaces.msg import VehicleCmd, VehicleFeedback


class DashboardNode(Node):
    """Unified HUD overlay on the camera feed."""

    def __init__(self):
        super().__init__('dashboard_node')

        # -- Parameters ------------------------------------------------
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('window_width', 800)
        self.declare_parameter('window_height', 480)

        camera_topic = self.get_parameter('camera_topic').value
        self.win_w = self.get_parameter('window_width').value
        self.win_h = self.get_parameter('window_height').value

        # -- State -----------------------------------------------------
        self.bridge = CvBridge()
        self.latest_frame = None

        self.light_state = 'UNKNOWN'
        self.sign_state = 'NO_SIGNAL'

        self.cmd_velocity = 0.0
        self.cmd_heading = 0.0
        self.raw_velocity = 0.0
        self.raw_heading = 0.0

        self.actual_velocity = 0.0
        self.actual_rpm = 0.0
        self.encoder_ticks = 0

        self.fps = 0.0
        self._t_last = time.time()

        # -- Subscribers -----------------------------------------------
        self.create_subscription(
            Image, camera_topic, self._image_cb, 10)
        self.create_subscription(
            String, '/traffic_light/state', self._light_cb, 10)
        self.create_subscription(
            String, '/sign/command', self._sign_cb, 10)
        self.create_subscription(
            VehicleCmd, '/vehicle/cmd', self._cmd_cb, 10)
        self.create_subscription(
            VehicleCmd, '/teleop/raw_cmd', self._raw_cmd_cb, 10)
        self.create_subscription(
            VehicleFeedback, '/vehicle/feedback', self._feedback_cb, 10)

        # -- Render timer (30 FPS) -------------------------------------
        self.create_timer(1.0 / 30.0, self._render)

        self.get_logger().info('Dashboard node started')

    # ==================================================================
    # Callbacks
    # ==================================================================

    def _image_cb(self, msg: Image):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
        except Exception:
            pass

    def _light_cb(self, msg: String):
        self.light_state = msg.data.strip().upper()

    def _sign_cb(self, msg: String):
        self.sign_state = msg.data.strip().upper()

    def _cmd_cb(self, msg: VehicleCmd):
        self.cmd_velocity = msg.velocity
        self.cmd_heading = msg.heading

    def _raw_cmd_cb(self, msg: VehicleCmd):
        self.raw_velocity = msg.velocity
        self.raw_heading = msg.heading

    def _feedback_cb(self, msg: VehicleFeedback):
        self.actual_velocity = msg.actual_velocity
        self.actual_rpm = msg.actual_rpm
        self.encoder_ticks = msg.encoder_ticks

    # ==================================================================
    # Render HUD
    # ==================================================================

    def _render(self):
        # FPS calculation
        now = time.time()
        dt = now - self._t_last
        self._t_last = now
        self.fps = 0.9 * self.fps + 0.1 / max(dt, 1e-4)

        # Base frame
        if self.latest_frame is not None:
            canvas = cv2.resize(self.latest_frame,
                                (self.win_w, self.win_h))
        else:
            canvas = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)
            cv2.putText(canvas, 'WAITING FOR CAMERA...',
                        (self.win_w // 2 - 180, self.win_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)

        h, w = canvas.shape[:2]

        # ---- Top bar (semi-transparent) ----
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

        # Traffic light indicator (top-left)
        light_colors = {
            'RED': (0, 0, 255), 'YELLOW': (0, 220, 255),
            'GREEN': (0, 255, 0), 'UNKNOWN': (80, 80, 80)
        }
        lc = light_colors.get(self.light_state, (80, 80, 80))
        cv2.circle(canvas, (30, 30), 18, lc, -1)
        cv2.circle(canvas, (30, 30), 18, (200, 200, 200), 2)
        cv2.putText(canvas, self.light_state, (55, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Sign detection (top-center)
        sign_colors = {
            'STOP': (0, 0, 255), 'SLOW_DOWN': (0, 200, 255),
            'TURN_LEFT': (255, 150, 0), 'TURN_RIGHT': (255, 100, 0),
            'NO_SIGNAL': (100, 100, 100)
        }
        sign_icons = {
            'STOP': '[ STOP ]', 'SLOW_DOWN': '[ SLOW ]',
            'TURN_LEFT': '[ << LEFT ]', 'TURN_RIGHT': '[ RIGHT >> ]',
            'NO_SIGNAL': '[ -- ]'
        }
        sc = sign_colors.get(self.sign_state, (100, 100, 100))
        sign_text = sign_icons.get(self.sign_state, self.sign_state)
        cv2.putText(canvas, sign_text,
                    (w // 2 - 60, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 2)

        # FPS (top-right)
        cv2.putText(canvas, f'{self.fps:.0f} FPS', (w - 100, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ---- Bottom bar (semi-transparent) ----
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, h - 90), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

        bar_y = h - 80

        # Velocity gauge (bottom-left)
        vel_label = f'VEL: {self.cmd_velocity:.2f} m/s'
        cv2.putText(canvas, vel_label, (15, bar_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Velocity bar
        max_bar_w = 180
        vel_frac = min(abs(self.cmd_velocity) / 2.0, 1.0)
        bar_filled = int(max_bar_w * vel_frac)
        bar_color = (0, 255, 0) if self.cmd_velocity >= 0 else (0, 100, 255)
        cv2.rectangle(canvas, (15, bar_y + 30),
                      (15 + max_bar_w, bar_y + 45), (60, 60, 60), -1)
        if bar_filled > 0:
            cv2.rectangle(canvas, (15, bar_y + 30),
                          (15 + bar_filled, bar_y + 45), bar_color, -1)
        cv2.rectangle(canvas, (15, bar_y + 30),
                      (15 + max_bar_w, bar_y + 45), (120, 120, 120), 1)

        # Actual velocity from encoder
        cv2.putText(canvas, f'ACT: {self.actual_velocity:.2f} m/s  RPM: {self.actual_rpm:.0f}',
                    (15, bar_y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # ---- Steering visualisation (bottom-center) ----
        steer_cx = w // 2
        steer_cy = bar_y + 35

        # Steering arc
        cv2.ellipse(canvas, (steer_cx, steer_cy + 10), (60, 30),
                    0, 180, 360, (80, 80, 80), 2)

        # Steering needle
        angle_rad = math.radians(-self.cmd_heading)  # negative because screen coords
        needle_len = 50
        nx = int(steer_cx + needle_len * math.sin(angle_rad))
        ny = int(steer_cy - needle_len * math.cos(angle_rad) * 0.5 + 10)
        needle_color = (0, 255, 255) if abs(self.cmd_heading) > 5 else (200, 200, 200)
        cv2.line(canvas, (steer_cx, steer_cy + 10), (nx, ny),
                 needle_color, 3)
        cv2.circle(canvas, (steer_cx, steer_cy + 10), 5,
                   (255, 255, 255), -1)

        # Steering angle text
        cv2.putText(canvas, f'{self.cmd_heading:.1f} deg',
                    (steer_cx - 35, bar_y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # ---- Right side: raw command info ----
        rx = w - 220
        cv2.putText(canvas, f'RAW v={self.raw_velocity:.2f}',
                    (rx, bar_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(canvas, f'RAW h={self.raw_heading:.1f} deg',
                    (rx, bar_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(canvas, f'ENC: {self.encoder_ticks}',
                    (rx, bar_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # ---- Safety override flash ----
        if self.light_state == 'RED' or self.sign_state == 'STOP':
            # Flashing red border
            if int(time.time() * 4) % 2 == 0:
                cv2.rectangle(canvas, (2, 2), (w - 3, h - 3),
                              (0, 0, 255), 4)
                cv2.putText(canvas, 'SAFETY STOP',
                            (w // 2 - 90, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow('Zooba Dashboard', canvas)
        cv2.waitKey(1)

    # ==================================================================
    # Cleanup
    # ==================================================================

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
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
