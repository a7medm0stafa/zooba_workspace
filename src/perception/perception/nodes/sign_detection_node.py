import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import numpy as np
import time
from collections import deque

# ═══════════════════════════════════════════════════════════
#  HSV Color Ranges
# ═══════════════════════════════════════════════════════════

RED_RANGES = [
    (np.array([0,   150, 120]),  np.array([8,   255, 255])),
    (np.array([172, 150, 120]),  np.array([179, 255, 255]))
]

YELLOW_RANGE = (
    np.array([20, 120, 120]),
    np.array([30, 255, 255])
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
        self.declare_parameter('show_gui', False)
        
        # Load parameters
        self.min_area = self.get_parameter('min_area').value
        self.max_area = self.get_parameter('max_area').value
        self.epsilon_factor = self.get_parameter('epsilon_factor').value
        self.vote_window = self.get_parameter('vote_window').value
        self.vote_threshold = self.get_parameter('vote_threshold').value
        self.show_gui = self.get_parameter('show_gui').value

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

        # Initialize camera (0 for Pi Camera or USB Camera)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error('Cannot open camera')
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # ROS Publisher for vehicle commands
        self.command_publisher = self.create_publisher(
            String,
            'vehicle/command',
            10
        )

        # Timer running at ~20 FPS
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info("Sign detection node started")

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
        """Returns (detections_list, processed_frame, hsv_frame, debug_masks)."""
        processed = self.preprocess(frame)
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)

        # --- CREATE MASKS FOR DEBUG ---
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

        # --- ORIGINAL DETECTION (UNCHANGED) ---
        detections = []
        detections += self._find_red(hsv)
        detections += self._find_yellow(hsv)
        detections += self._find_blue(hsv, processed)

        return detections, processed, hsv, debug_masks

    # ── red  →  STOP ─────────────────────────────────────
    def _find_red(self, hsv):
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in RED_RANGES:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        mask = self._clean_mask(mask)
        return self._classify_contours(mask, 'STOP',
                                       min_vertices=6, max_vertices=12,
                                       min_circularity=0.55)

    # ── yellow  →  SLOW_DOWN ─────────────────────────────
    def _find_yellow(self, hsv):
        lo, hi = YELLOW_RANGE
        mask = cv2.inRange(hsv, lo, hi)
        mask = self._clean_mask(mask)
        return self._classify_contours(mask, 'SLOW_DOWN',
                                       min_vertices=3, max_vertices=8,
                                       min_circularity=0.3)

    # ── blue  →  TURN_LEFT / TURN_RIGHT ──────────────────
    def _find_blue(self, hsv, frame):
        blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blue_detections = []
        max_area_found = 0
        best_circularity = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area_found:
                max_area_found = area
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circ = (4 * np.pi * area) / (perimeter ** 2)
                if circ > best_circularity:
                    best_circularity = circ

            if self.min_area < area < self.max_area:
                if self._is_circular(cnt):
                    x, y, w, h = cv2.boundingRect(cnt)
                    roi = self.extract_roi(cnt, frame)
                    direction = self._arrow_direction(roi)
                    
                    if direction:
                        label = direction if "TURN_" in direction else f"TURN_{direction}"
                        blue_detections.append((label, 1.0, (x, y, w, h)))

        return blue_detections

    # ── arrow direction ─────────────────────────────────
    def _arrow_direction(self, roi):
        if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
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

        if ratio > 0.08:
            return 'LEFT'
        elif ratio < -0.08:
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

    # ── ros timer callback ───────────────────────────────
    def timer_callback(self):
        t_start = time.time()
        ok, frame = self.cap.read()
        if not ok:
            return
        
        # Rotate 180 degrees (-1) or remove completely if no flip is needed
        frame = cv2.flip(frame, -1)
        
        proc_start = time.time()
        detections, processed, hsv, debug_masks = self.detect(frame)
        command, conf = self.vote(detections)
        proc_ms = (time.time() - proc_start) * 1000 

        # Only publish actual signs or NO_SIGN? Publish command
        msg = String()
        # For compatibility with earlier setups, let's map NO_SIGN -> NO_SIGNAL if needed.
        # But 'NO_SIGN' is ok too. Let's just output command.
        if command == 'NO_SIGN':
            msg.data = "NO_SIGNAL"
        else:
            msg.data = command
            
        self.command_publisher.publish(msg)

        # FPS Tracking
        dt = time.time() - self.t_start_fps
        self.fps = 0.9 * self.fps + 0.1 / max(dt, 1e-4)
        self.t_start_fps = time.time()

        # Log significant changes
        if command != self.prev_cmd and command != 'NO_SIGN':
            self.get_logger().info(f'>>> {command:10} | Conf: {conf:.0%} | Latency: {proc_ms:>4.1f}ms | {self.fps:.0f} FPS')
        self.prev_cmd = command

        # GUI Display
        if self.show_gui:
            display = draw_gui(cv2.resize(frame, (640, 480)), detections, command, conf, self.fps)

            # Convert masks to 3-channel
            red_vis    = cv2.cvtColor(debug_masks["red"], cv2.COLOR_GRAY2BGR)
            yellow_vis = cv2.cvtColor(debug_masks["yellow"], cv2.COLOR_GRAY2BGR)
            blue_vis   = cv2.cvtColor(debug_masks["blue"], cv2.COLOR_GRAY2BGR)

            # --- FIXED RESIZING ---
            mask_h = 160
            mask_w = 640 // 3

            red_vis    = cv2.resize(red_vis, (mask_w, mask_h))
            yellow_vis = cv2.resize(yellow_vis, (mask_w, mask_h))
            blue_vis   = cv2.resize(blue_vis, (640 - 2*mask_w, mask_h))

            # Stack masks
            masks_row = np.hstack((red_vis, yellow_vis, blue_vis))

            # Resize main display
            display_small = cv2.resize(display, (640, 320))

            # Combine safely
            combined = np.vstack((display_small, masks_row))

            # Labels
            cv2.putText(combined, "RED", (10, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.putText(combined, "YELLOW", (230, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            cv2.putText(combined, "BLUE", (460, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

            cv2.imshow('Sign Detection Debug', combined)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = SignDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()