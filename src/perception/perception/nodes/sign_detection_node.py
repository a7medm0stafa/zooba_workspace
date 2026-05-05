import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import csv
import time
from collections import deque
import os
# ═══════════════════════════════════════════════════════════
#  HSV Color Ranges
# ═══════════════════════════════════════════════════════════

RED_RANGES = [
    (np.array([0, 130, 110]), np.array([10, 255, 255])),   
    (np.array([170, 120, 100]), np.array([180, 255, 255])) 
]

YELLOW_RANGE = (
    np.array([20, 90, 80]), 
    np.array([35, 255, 255])
)
BLUE_RANGE   = (np.array([100, 80, 60]),  np.array([130, 255, 255]))

# ═══════════════════════════════════════════════════════════
#  Visualisation helpers
# ═══════════════════════════════════════════════════════════

COLORS = {
    'STOP':       (0,   0,   255),
    'SLOW_DOWN':  (0,   200, 255),
    'TURN_LEFT':  (255, 150, 0),
    'TURN_RIGHT': (255, 100, 0),
    'NO_SIGN':    (100, 100, 100),
}

def draw_gui(frame, detections, command, conf, fps):
    out = frame.copy()

    # Bounding boxes
    for sign, c, (x, y, w, h) in detections:
        col = COLORS.get(sign, (255, 255, 255))
        cv2.rectangle(out, (x, y), (x+w, y+h), col, 2)
        lbl = f'{sign} {c:.0%}'
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x, y-th-10), (x+tw+4, y), col, -1)
        cv2.putText(out, lbl, (x+2, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

    # Header bar
    col = COLORS.get(command, (100, 100, 100))
    cv2.rectangle(out, (0, 0), (640, 48), (25, 25, 25), -1)
    cv2.putText(out, f'CMD: {command}', (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)

    # Confidence bar
    bw = int(180 * conf)
    cv2.rectangle(out, (440, 12), (440+bw, 38), col, -1)
    cv2.rectangle(out, (440, 12), (620, 38), (80, 80, 80), 1)
    cv2.putText(out, f'{conf:.0%}', (445, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    # FPS
    cv2.putText(out, f'{fps:.0f} FPS', (560, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return out

# ═══════════════════════════════════════════════════════════
#  ROS2 Node Wrapper for SignDetector
# ═══════════════════════════════════════════════════════════

class SignDetectionNode(Node):

    def __init__(self):
        super().__init__('sign_detection_node')

        # Declare parameters with default values
        self.declare_parameter('min_area', 1500)
        self.declare_parameter('max_area', 120000)
        self.declare_parameter('epsilon_factor', 0.02)
        self.declare_parameter('vote_window', 15)
        self.declare_parameter('vote_threshold', 11)
        self.declare_parameter('show_gui', True)
        self.declare_parameter('output_topic', '/sign/command')
        self.declare_parameter('camera_topic', '/camera/image_raw')
        
        # Load parameters
        self.min_area = self.get_parameter('min_area').value
        self.max_area = self.get_parameter('max_area').value
        self.epsilon_factor = self.get_parameter('epsilon_factor').value
        self.vote_window = self.get_parameter('vote_window').value
        self.vote_threshold = self.get_parameter('vote_threshold').value
        self.show_gui = self.get_parameter('show_gui').value
        output_topic = self.get_parameter('output_topic').value
        camera_topic = self.get_parameter('camera_topic').value

        # Log active params
        self.get_logger().info(f"Loaded params - min_area: {self.min_area}, show_gui: {self.show_gui}")

        # Temporal voting
        self.history = deque(maxlen=self.vote_window)
        self.lower_blue = np.array([100, 150, 40])
        self.upper_blue = np.array([130, 255, 255])
        self.last_debug_time = time.time()
        
        # Performance tracking
        self.fps = 0.0
        self.t_start_fps = time.time()
        self.prev_cmd = ''

        # Subscribe to shared camera topic (instead of opening camera directly)
        self.bridge = CvBridge()
        self.latest_frame = None
        self.image_sub = self.create_subscription(
            Image, camera_topic, self._image_callback, 10
        )
        self.get_logger().info(f"Subscribing to camera on: {camera_topic}")

        self.test_images = [ "Turnn.png","SlowDown.png", "Stop.png"]
        self.img_idx = 0

        # ROS Publisher for sign detection commands
        self.command_publisher = self.create_publisher(
            String,
            output_topic,
            10
        )
        self.get_logger().info(f"Publishing sign commands on: {output_topic}")

        # Timer running at ~20 FPS
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info("Sign detection node started")

        # -- KPI logging -----------------------------------------------
        self._kpi_init()

    # ── preprocessing ────────────────────────────────────
    def preprocess(self, frame):
        img = cv2.resize(frame, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.clip(cv2.add(v, 20), 0, 255).astype(np.uint8)
        img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab = cv2.merge((clahe.apply(l), a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        return img

    # ── main pipeline ────────────────────────────────────
    def detect(self, frame):
        processed = self.preprocess(frame)
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in RED_RANGES:
            red_mask = cv2.bitwise_or(red_mask, cv2.inRange(hsv, lo, hi))
        yellow_mask = cv2.inRange(hsv, *YELLOW_RANGE)
        blue_mask   = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        debug_masks = {
            "red": self._clean_mask(red_mask),
            "yellow": self._clean_mask(yellow_mask),
            "blue": self._clean_mask(blue_mask)
        }
        all_found = []
        all_found += self._find_blue(hsv, processed)
        all_found += self._find_yellow(hsv)
        all_found += self._find_red(hsv)
        best_detection = []
        if all_found:
            all_found.sort(key=lambda x: x[2][2] * x[2][3], reverse=True)
            best_detection = [all_found[0]]
        return best_detection, processed, hsv, debug_masks

    def _find_red(self, hsv):
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in RED_RANGES:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        mask = self._clean_mask(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_area:
                peri = cv2.arcLength(cnt, True)
                circ = (4 * np.pi * area) / (peri**2) if peri > 0 else 0
                approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                verts = len(approx)
                print(f"[RED DETECTED] Area: {area:.0f} | Circ: {circ:.2f} | Verts: {verts} ")
                
                # Basic filters
                if (0.6 <= circ <= 0.95 and 6 <= verts <= 10):
                    x, y, w, h_rect = cv2.boundingRect(cnt)
                    print(f"✓ STOP SIGN CONFIRMED")
                    return [('STOP', 1.0, (x, y, w, h_rect))]

                    # continue
                
                # HEXAGON VERIFICATION - use angle check
                # if self._verify_hexagon_angles(approx):
                   
        return []

    def _verify_hexagon_angles(self, approx):
        """Verify hexagonal shape by checking internal angles"""
        if len(approx) < 6:
            return False
        
        angles = []
        for i in range(len(approx)):
            p1 = approx[i-1][0]
            p2 = approx[i][0]
            p3 = approx[(i+1) % len(approx)][0]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            angles.append(angle)
        
        angle_std = np.std(angles)
        angle_mean = np.mean(angles)
        
        print(f"  Angles μ={angle_mean:.1f}° σ={angle_std:.1f}°")
        
        # Hexagon: ~120° with low variance
        # Circle: varying angles or all ~equal but not 120°
        return angle_std < 20 and 105 < angle_mean < 135
    # ── yellow  →  SLOW_DOWN ─────────────────────────────
    def _find_yellow(self, hsv):
        lo, hi = YELLOW_RANGE
        mask = cv2.inRange(hsv, lo, hi)
        mask = self._clean_mask(mask)       
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)      
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_area:
                peri = cv2.arcLength(cnt, True)
                circ = (4 * np.pi * area) / (peri**2) if peri > 0 else 0
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                verts = len(approx)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    h, s, v = hsv[cy, cx]
                    print(f"[YELLOW DETECTED] Area: {area:.0f} | Circ: {circ:.2f} | Verts: {verts} | HSV: ({h},{s},{v})")
        return self._classify_contours(mask, 'SLOW_DOWN', 
                                       min_vertices=3, max_vertices=7, 
                                       min_circularity=0.6)

    # ── blue  →  TURN_LEFT / TURN_RIGHT ──────────────────
    def _find_blue(self, hsv, frame):
        blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)        
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_detections = []
        max_area_found = 0
        best_circularity = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area_found: max_area_found = area           
            perimeter = cv2.arcLength(cnt, True)
            circ = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            if circ > best_circularity: best_circularity = circ
            if area > self.min_area:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    h, s, v = hsv[cy, cx]
                    print(f"[BLUE DETECTED] Area: {area:.0f} | Circ: {circ:.2f} | HSV: ({h},{s},{v})")
            if self.min_area < area < self.max_area:
                if self._is_circular(cnt):
                    x, y, w, h = cv2.boundingRect(cnt)
                    roi = self.extract_roi(cnt, frame)
                    direction = self._arrow_direction(roi)   
                    print(f"[ARROW DEBUG] Circularity passed! Direction sensed: {direction}")              
                    if direction:
                        label = direction if "TURN_" in direction else f"TURN_{direction}"
                        blue_detections.append((label, 1.0, (x, y, w, h)))
        return blue_detections

    # ── arrow direction ─────────────────────────────────
    def _arrow_direction(self, roi):
        if roi.size == 0 or roi.shape[0] < 15 or roi.shape[1] < 15:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, white = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)       
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kern)
        h, w = white.shape
        margin = int(h * 0.1)
        focused_zone = white[margin:h-margin, :]
        left_pixels  = cv2.countNonZero(focused_zone[:, :w//2])
        right_pixels = cv2.countNonZero(focused_zone[:, w//2:])
        total = left_pixels + right_pixels     
        if total == 0: return None     
        ratio = (left_pixels - right_pixels) / total
        print(f"[ARROW] L: {left_pixels} | R: {right_pixels} | Ratio: {ratio:.3f}")
        if ratio > 0.065:
            return 'LEFT'
        elif ratio < -0.065:
            return 'RIGHT'
        return None

    def extract_roi(self, contour, frame):
        x, y, w, h = cv2.boundingRect(contour)
        y_start, y_end = max(0, y), min(frame.shape[0], y + h)
        x_start, x_end = max(0, x), min(frame.shape[1], x + w)
        return frame[y_start:y_end, x_start:x_end]

    def _is_circular(self, contour):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0: return False
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        print(f"[GEOMETRY] Area: {area:.0f} | Calculated Circularity: {circularity:.3f}")
        return 0.7 < circularity < 1.2

    # ── shared helpers ───────────────────────────────────
    def _clean_mask(self, mask):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        return mask

    def _classify_contours(self, mask, label, min_vertices=3, max_vertices=12, min_circularity=0.3):
        detections = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            approx = cv2.approxPolyDP(cnt, self.epsilon_factor * peri, True)
            verts  = len(approx)
            circ   = 4 * np.pi * area / (peri * peri)
            x, y, w, h = cv2.boundingRect(cnt)
            ar = w / h if h else 0
            if (min_vertices <= verts <= max_vertices and circ >= min_circularity and 0.5 < ar < 2.0):
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

    # ── camera subscription callback ──────────────────────
    def _image_callback(self, msg: Image):
        """Store the latest camera frame from the shared camera topic."""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge error: {e}')

    # ── KPI logging infrastructure ────────────────────────
    def _kpi_init(self):
        """Initialise KPI CSV logging.

        Opens the CSV in write mode so it clears on every launch.
        """
        self.kpi_total_frames = 0
        self.kpi_candidate_frames = 0
        self.kpi_confirmed_frames = 0
        self._kpi_file = None
        self._kpi_writer = None

        csv_path = os.path.expanduser('~/zooba_workspace/zooba_kpi/sign_detection_kpi.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        self._kpi_file = open(csv_path, 'w', newline='')
        self._kpi_writer = csv.writer(self._kpi_file)
        self._kpi_writer.writerow([
            'timestamp', 'latency_ms',
            'raw_detection', 'voted_command', 'vote_confidence',
            'num_contours_found', 'candidates_found', 'detection_rate_pct',
        ])
        self._kpi_file.flush()
        self.get_logger().info(f'KPI CSV logging to {csv_path}')

    def _kpi_write_row(self, latency_ms, raw_detection, voted_command,
                       vote_confidence, num_contours_found):
        """Append one KPI row to the CSV."""
        if self._kpi_writer is None:
            return

        self.kpi_total_frames += 1

        # A "candidate frame" is one where contours passed geometric filters
        candidates_found = 1 if num_contours_found > 0 else 0
        if candidates_found:
            self.kpi_candidate_frames += 1

        # A "confirmed frame" is one where voting returned a real command
        if voted_command != 'NO_SIGN' and voted_command != 'NO_SIGNAL':
            self.kpi_confirmed_frames += 1

        detection_rate_pct = (
            (self.kpi_confirmed_frames / self.kpi_candidate_frames * 100.0)
            if self.kpi_candidate_frames > 0 else 0.0
        )

        self._kpi_writer.writerow([
            f'{time.time():.4f}',
            f'{latency_ms:.2f}',
            raw_detection,
            voted_command,
            f'{vote_confidence:.4f}',
            num_contours_found,
            candidates_found,
            f'{detection_rate_pct:.2f}',
        ])
        self._kpi_file.flush()

    def _kpi_close(self):
        """Close the KPI CSV file handle."""
        if self._kpi_file is not None:
            self._kpi_file.close()
            self._kpi_file = None
            self._kpi_writer = None

    # ── ros timer callback ───────────────────────────────
    def timer_callback(self):
        frame = self.latest_frame
        if frame is None:
            return

        proc_start = time.time()
        detections, processed, hsv, debug_masks = self.detect(frame)
        command, conf = self.vote(detections)
        proc_ms = (time.time() - proc_start) * 1000

        # Determine raw detection label for KPI
        raw_det = detections[0][0] if detections else 'NO_SIGN'

        msg = String()
        msg.data = "NO_SIGNAL" if command == 'NO_SIGN' else command
        self.command_publisher.publish(msg)
        dt = time.time() - self.t_start_fps
        self.fps = 0.9 * self.fps + 0.1 / max(dt, 1e-4)
        self.t_start_fps = time.time()
        if command != self.prev_cmd and command != 'NO_SIGN':
            self.get_logger().info(f'>>> {command:10} | Conf: {conf:.0%} | {proc_ms:>4.1f}ms | {self.fps:.0f} FPS')
        self.prev_cmd = command

        # -- KPI CSV logging -------------------------------------------
        self._kpi_write_row(
            latency_ms=proc_ms,
            raw_detection=raw_det,
            voted_command=command,
            vote_confidence=conf,
            num_contours_found=len(detections),
        )

        if self.show_gui:
            display = draw_gui(cv2.resize(frame, (640, 480)), detections, command, conf, self.fps)            
            mask_h, mask_w = 160, 640 // 3
            vis = [cv2.cvtColor(debug_masks[k], cv2.COLOR_GRAY2BGR) for k in ["red", "yellow", "blue"]]
            vis[0] = cv2.resize(vis[0], (mask_w, mask_h))
            vis[1] = cv2.resize(vis[1], (mask_w, mask_h))
            vis[2] = cv2.resize(vis[2], (640 - 2*mask_w, mask_h))
            combined = np.vstack((cv2.resize(display, (640, 320)), np.hstack(vis)))            
            cv2.putText(combined, "RED", (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.putText(combined, "YELLOW", (230, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            cv2.putText(combined, "BLUE", (460, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            cv2.imshow('Sign Detection Debug', combined)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = SignDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down sign detection node...')
    finally:
        node._kpi_close()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()