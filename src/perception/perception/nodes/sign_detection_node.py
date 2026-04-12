"""
Sign Detection ROS2 Node
========================
Detects action signs (STOP, SLOW_DOWN, TURN_LEFT, TURN_RIGHT) using
the laptop/Pi camera and publishes VehicleCmd to /vehicle/cmd.

Uses the same SignDetector pipeline as test_sign_detection.py with
temporal voting for robust, jitter-free commands.

Topics Published:
    /vehicle/cmd              (vehicle_interfaces/VehicleCmd)  — velocity + heading
    /perception/sign_detected (std_msgs/String)                — current sign label

Parameters:
    cruise_velocity   : forward speed when no sign or turn (default 0.5 m/s)
    slow_velocity     : speed during SLOW_DOWN (default 0.3 m/s)
    turn_heading      : steering angle for TURN signs (default 20.0°)
    camera_index      : OpenCV camera index (default 0)
    flip_horizontal   : mirror the camera feed (default True)
    publish_rate      : detection + publish rate in Hz (default 20.0)
    output_topic      : vehicle command topic (default /vehicle/cmd)
"""

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleCmd
from std_msgs.msg import String

import cv2
import numpy as np
import time
from collections import deque


# ═══════════════════════════════════════════════════════════
#  HSV Color Ranges  (must match test_sign_detection.py)
# ═══════════════════════════════════════════════════════════

RED_RANGES = [
    (np.array([0,   120, 70]),  np.array([10,  255, 255])),
    (np.array([170, 120, 70]),  np.array([179, 255, 255])),
]
YELLOW_RANGE = (
    np.array([20, 120, 120]),
    np.array([30, 255, 255])
)
BLUE_RANGE = (np.array([100, 80, 60]), np.array([130, 255, 255]))


# ═══════════════════════════════════════════════════════════
#  Sign Detector  (same logic as test_sign_detection.py)
# ═══════════════════════════════════════════════════════════

class SignDetector:

    def __init__(self):
        self.min_area = 1500
        self.max_area = 120000
        self.epsilon_factor = 0.02

        # Temporal voting
        self.vote_window = 15
        self.vote_threshold = 11
        self.history = deque(maxlen=self.vote_window)

    # ── preprocessing ────────────────────────────────────

    def preprocess(self, frame):
        img = cv2.resize(frame, (640, 480))

        # Brightness boost via V channel
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.clip(cv2.add(v, 20), 0, 255).astype(np.uint8)
        img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

        # CLAHE on L channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab = cv2.merge((clahe.apply(l), a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Gentle blur
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    # ── main pipeline ────────────────────────────────────

    def detect(self, frame):
        processed = self.preprocess(frame)
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)

        detections = []
        detections += self._find_red(hsv)
        detections += self._find_yellow(hsv)
        detections += self._find_blue(hsv, processed)
        return detections, processed, hsv

    # ── red → STOP ───────────────────────────────────────

    def _find_red(self, hsv):
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in RED_RANGES:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        mask = self._clean_mask(mask)
        return self._classify_contours(mask, 'STOP',
                                       min_vertices=6, max_vertices=12,
                                       min_circularity=0.55)

    # ── yellow → SLOW_DOWN ───────────────────────────────

    def _find_yellow(self, hsv):
        lo, hi = YELLOW_RANGE
        mask = cv2.inRange(hsv, lo, hi)
        mask = self._clean_mask(mask)
        return self._classify_contours(mask, 'SLOW_DOWN',
                                       min_vertices=3, max_vertices=8,
                                       min_circularity=0.3)

    # ── blue → TURN_LEFT / TURN_RIGHT ────────────────────

    def _find_blue(self, hsv, processed):
        lo, hi = BLUE_RANGE
        mask = cv2.inRange(hsv, lo, hi)
        mask = self._clean_mask(mask)

        detections = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            ar = w / h if h else 0
            if not (0.5 < ar < 2.0):
                continue

            roi = processed[y:y+h, x:x+w]
            direction = self._arrow_direction(roi)
            if direction:
                detections.append((f'TURN_{direction}', 0.70, (x, y, w, h)))
        return detections

    # ── arrow direction ──────────────────────────────────

    def _arrow_direction(self, roi):
        if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, white = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kern)

        h, w = white.shape
        left_pixels = cv2.countNonZero(white[:, :w//2])
        right_pixels = cv2.countNonZero(white[:, w//2:])
        total = left_pixels + right_pixels
        if total == 0:
            return None
        ratio = (left_pixels - right_pixels) / total
        if ratio > 0.05:
            return 'LEFT'
        elif ratio < -0.05:
            return 'RIGHT'
        return None

    # ── shared helpers ───────────────────────────────────

    def _clean_mask(self, mask):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        return mask

    def _classify_contours(self, mask, label,
                           min_vertices=3, max_vertices=12,
                           min_circularity=0.3):
        detections = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue

            approx = cv2.approxPolyDP(cnt, self.epsilon_factor * peri, True)
            verts = len(approx)
            circ = 4 * np.pi * area / (peri * peri)

            x, y, w, h = cv2.boundingRect(cnt)
            ar = w / h if h else 0

            if (min_vertices <= verts <= max_vertices
                    and circ >= min_circularity
                    and 0.5 < ar < 2.0):
                conf = min(1.0, circ + 0.2 * (area / self.max_area))
                detections.append((label, round(conf, 2), (x, y, w, h)))
        return detections

    # ── temporal voting ──────────────────────────────────

    def vote(self, detections):
        if detections:
            best = max(detections, key=lambda d: d[1])
            self.history.append(best[0])
        else:
            self.history.append('NO_SIGN')

        if len(self.history) < 5:
            return 'NO_SIGN', 0.0

        counts = {}
        for v in self.history:
            counts[v] = counts.get(v, 0) + 1

        winner = max(counts, key=counts.get)
        frac = counts[winner] / len(self.history)

        if winner != 'NO_SIGN' and counts[winner] >= self.vote_threshold:
            return winner, frac
        return 'NO_SIGN', 0.0


# ═══════════════════════════════════════════════════════════
#  ROS2 Node
# ═══════════════════════════════════════════════════════════

class SignDetectionNode(Node):

    def __init__(self):
        super().__init__('sign_detection_node')

        # ── Parameters ──────────────────────────────────
        self.declare_parameter('cruise_velocity', 0.5)       # m/s
        self.declare_parameter('slow_velocity', 0.3)         # m/s
        self.declare_parameter('turn_heading', 20.0)         # degrees
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('flip_horizontal', True)
        self.declare_parameter('publish_rate', 20.0)         # Hz
        self.declare_parameter('output_topic', '/vehicle/cmd')
        self.declare_parameter('sign_topic', '/perception/sign_detected')
        self.declare_parameter('show_gui', False)            # GUI windows (off on Pi)

        self.cruise_velocity = self.get_parameter('cruise_velocity').value
        self.slow_velocity = self.get_parameter('slow_velocity').value
        self.turn_heading = self.get_parameter('turn_heading').value
        camera_index = self.get_parameter('camera_index').value
        self.flip_horizontal = self.get_parameter('flip_horizontal').value
        publish_rate = self.get_parameter('publish_rate').value
        output_topic = self.get_parameter('output_topic').value
        sign_topic = self.get_parameter('sign_topic').value
        self.show_gui = self.get_parameter('show_gui').value

        # ── Sign → command mapping ──────────────────────
        # Each entry: (velocity, heading)
        self.sign_commands = {
            'STOP':       (0.0,                  0.0),
            'SLOW_DOWN':  (self.slow_velocity,   0.0),
            'TURN_LEFT':  (self.cruise_velocity, -self.turn_heading),
            'TURN_RIGHT': (self.cruise_velocity,  self.turn_heading),
            'NO_SIGN':    (self.cruise_velocity,  0.0),
        }

        # ── Camera ──────────────────────────────────────
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.get_logger().error(f'Cannot open camera index {camera_index}')
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # ── Detector ────────────────────────────────────
        self.detector = SignDetector()

        # ── Publishers ──────────────────────────────────
        self.cmd_pub = self.create_publisher(VehicleCmd, output_topic, 10)
        self.sign_pub = self.create_publisher(String, sign_topic, 10)

        # ── Timer ───────────────────────────────────────
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self._timer_callback)

        # ── State ───────────────────────────────────────
        self.prev_command = ''
        self.fps = 0.0

        # ── Startup log ─────────────────────────────────
        self.get_logger().info('=' * 58)
        self.get_logger().info('Sign Detection Node Started')
        self.get_logger().info(f'  Camera         : index {camera_index}')
        self.get_logger().info(f'  Flip horizontal: {self.flip_horizontal}')
        self.get_logger().info(f'  Publish rate   : {publish_rate} Hz')
        self.get_logger().info(f'  Output topic   : {output_topic}')
        self.get_logger().info(f'  Sign topic     : {sign_topic}')
        self.get_logger().info(f'  Cruise velocity: {self.cruise_velocity} m/s')
        self.get_logger().info(f'  Slow velocity  : {self.slow_velocity} m/s')
        self.get_logger().info(f'  Turn heading   : ±{self.turn_heading}°')
        self.get_logger().info(f'  GUI windows    : {self.show_gui}')
        self.get_logger().info('=' * 58)
        self.get_logger().info('Command mapping:')
        for sign, (vel, hdg) in self.sign_commands.items():
            self.get_logger().info(f'  {sign:12s} → vel={vel:.2f} m/s, hdg={hdg:+.1f}°')
        self.get_logger().info('=' * 58)

    # ==================== Main Loop ====================

    def _timer_callback(self):
        """Capture frame → detect → vote → publish."""
        t0 = time.time()

        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn('Camera read failed', throttle_duration_sec=5.0)
            return

        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)

        # ── Detection + voting ──
        detections, processed, hsv = self.detector.detect(frame)
        command, conf = self.detector.vote(detections)

        # ── FPS tracking ──
        dt = time.time() - t0
        self.fps = 0.9 * self.fps + 0.1 / max(dt, 1e-4)

        # ── Publish VehicleCmd ──
        vel, hdg = self.sign_commands.get(command, (self.cruise_velocity, 0.0))

        cmd_msg = VehicleCmd()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.header.frame_id = 'base_link'
        cmd_msg.velocity = vel
        cmd_msg.heading = hdg
        self.cmd_pub.publish(cmd_msg)

        # ── Publish sign label ──
        sign_msg = String()
        sign_msg.data = command
        self.sign_pub.publish(sign_msg)

        # ── Log command changes ──
        if command != self.prev_command:
            if command != 'NO_SIGN':
                self.get_logger().info(
                    f'SIGN: {command} (conf={conf:.0%}) → '
                    f'vel={vel:.2f} m/s, hdg={hdg:+.1f}°  [{self.fps:.0f} FPS]'
                )
            else:
                self.get_logger().info(
                    f'SIGN: NO_SIGN → cruise vel={vel:.2f} m/s, hdg={hdg:+.1f}°'
                )
        self.prev_command = command

        # ── Optional GUI (for desktop debugging) ──
        if self.show_gui:
            self._show_debug(frame, detections, command, conf)

    # ==================== Debug GUI ====================

    def _show_debug(self, frame, detections, command, conf):
        """Show OpenCV windows for debugging (desktop only)."""
        COLORS = {
            'STOP':       (0,   0,   255),
            'SLOW_DOWN':  (0,   200, 255),
            'TURN_LEFT':  (255, 150, 0),
            'TURN_RIGHT': (255, 100, 0),
            'NO_SIGN':    (100, 100, 100),
        }

        display = cv2.resize(frame, (640, 480)).copy()

        for sign, c, (x, y, w, h) in detections:
            col = COLORS.get(sign, (255, 255, 255))
            cv2.rectangle(display, (x, y), (x+w, y+h), col, 2)
            lbl = f'{sign} {c:.0%}'
            cv2.putText(display, lbl, (x+2, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        col = COLORS.get(command, (100, 100, 100))
        cv2.rectangle(display, (0, 0), (640, 48), (25, 25, 25), -1)
        cv2.putText(display, f'CMD: {command}', (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        cv2.putText(display, f'{self.fps:.0f} FPS', (560, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.imshow('Sign Detection (ROS2)', display)
        cv2.waitKey(1)

    # ==================== Cleanup ====================

    def destroy_node(self):
        """Send stop command, release camera, close GUI."""
        self.get_logger().info('Shutting down — sending stop command...')

        # Send stop
        cmd_msg = VehicleCmd()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.header.frame_id = 'base_link'
        cmd_msg.velocity = 0.0
        cmd_msg.heading = 0.0
        self.cmd_pub.publish(cmd_msg)

        # Release camera
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        # Close GUI
        if self.show_gui:
            cv2.destroyAllWindows()

        super().destroy_node()


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)

    node = SignDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()