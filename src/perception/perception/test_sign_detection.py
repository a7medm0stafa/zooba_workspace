"""
Sign Detection Test Script (Standalone — No ROS2)
==================================================
Detects action signs using your laptop camera:
  - STOP        → Red octagonal/circular sign
  - SLOW_DOWN   → Yellow triangular/diamond sign
  - TURN_LEFT   → Blue sign with left arrow
  - TURN_RIGHT  → Blue sign with right arrow

Usage:  python test_sign_detection.py

Controls:
    Q       Quit
    D       Toggle debug mask windows
    Space   Clear vote history
    +/-     Adjust min contour area (sensitivity)
"""

import cv2
import numpy as np
import time
from collections import deque


# ═══════════════════════════════════════════════════════════
#  HSV Color Ranges
# ═══════════════════════════════════════════════════════════

# Red wraps around in HSV, so we need two ranges
RED_RANGES = [
    (np.array([0,   120, 70]),  np.array([10,  255, 255])),
    (np.array([170, 120, 70]),  np.array([179, 255, 255])),
]
YELLOW_RANGE = (
    np.array([20, 120, 120]),
    np.array([30, 255, 255])
)
BLUE_RANGE   = (np.array([100, 80, 60]),  np.array([130, 255, 255]))


# ═══════════════════════════════════════════════════════════
#  Sign Detector
# ═══════════════════════════════════════════════════════════

class SignDetector:

    def __init__(self):
        # Contour area bounds (pixels² at 640×480)
        self.min_area = 1500
        self.max_area = 120000

        # Shape approximation factor for cv2.approxPolyDP
        self.epsilon_factor = 0.02

        # Temporal voting
        self.vote_window = 15
        self.vote_threshold = 11        # need 11/15 agreeing frames
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
        """Returns (detections_list, processed_frame, hsv_frame)."""
        processed = self.preprocess(frame)
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)

        detections = []
        detections += self._find_red(hsv)
        detections += self._find_yellow(hsv)
        detections += self._find_blue(hsv, processed)
        return detections, processed, hsv

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

            # Extract ROI and determine arrow direction
            roi = processed[y:y+h, x:x+w]
            direction = self._arrow_direction(roi)
            if direction:
                detections.append((f'TURN_{direction}', 0.70, (x, y, w, h)))
        return detections

    # ── arrow direction ──────────────────────────────────

    def _arrow_direction(self, roi):
        """Determine LEFT or RIGHT based strictly on pixel mass distribution."""
        if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
            return None

        # 1. Pre-processing
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 140 is a "sweet spot" to keep the arrow solid without picking up blue noise
        _, white = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        
        # Smooth the arrow to fill any gaps
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kern)

        h, w = white.shape

        left_pixels  = cv2.countNonZero(white[:, :w//2])
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

    @staticmethod
    def _hull_angle(pts, idx, n):
        """Interior angle at pts[idx] on the convex hull."""
        p = pts[(idx - 1) % n].astype(float)
        q = pts[idx].astype(float)
        r = pts[(idx + 1) % n].astype(float)
        v1 = p - q
        v2 = r - q
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    @staticmethod
    def _arrow_by_mass(white_mask):
        """Fallback: compare white-pixel mass in left vs right half."""
        h, w = white_mask.shape
        # Use np.sum or cv2.countNonZero (countNonZero is faster on Pi)
        left_mass  = np.sum(white_mask[:, :w//2].astype(float))
        right_mass = np.sum(white_mask[:, w//2:].astype(float))
        total = left_mass + right_mass
        
        if total == 0:
            return None
            
        ratio = (left_mass - right_mass) / total
        
        # --- ADJUSTED THRESHOLD ---
        # 0.05 means a 5% difference is enough to trigger a turn
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
            verts  = len(approx)
            circ   = 4 * np.pi * area / (peri * peri)

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
        """Smooth detections over time. Returns (command, confidence)."""
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
        frac   = counts[winner] / len(self.history)

        if winner != 'NO_SIGN' and counts[winner] >= self.vote_threshold:
            return winner, frac
        return 'NO_SIGN', 0.0


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


def draw(frame, detections, command, conf, fps):
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


def debug_masks(hsv):
    """Return a stacked image showing Red / Yellow / Blue masks."""
    r = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in RED_RANGES:
        r = cv2.bitwise_or(r, cv2.inRange(hsv, lo, hi))
    y = cv2.inRange(hsv, *YELLOW_RANGE)
    b = cv2.inRange(hsv, *BLUE_RANGE)

    row = np.hstack([
        cv2.cvtColor(r, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(y, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(b, cv2.COLOR_GRAY2BGR),
    ])
    cv2.putText(row, 'RED',    (10,  25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),   2)
    cv2.putText(row, 'YELLOW', (650, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(row, 'BLUE',   (1290,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),   2)
    return row


# ═══════════════════════════════════════════════════════════
#  Main loop
# ═══════════════════════════════════════════════════════════

def main():
    print('=' * 55)
    print('  Sign Detection — Test Script')
    print('  STOP | SLOW_DOWN | TURN_LEFT | TURN_RIGHT')
    print('=' * 55)
    print('  Q = quit   D = debug masks   Space = clear votes')
    print('  +/- = adjust sensitivity')
    print('=' * 55)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: cannot open camera')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    det = SignDetector()
    show_dbg = False
    fps = 0.0
    prev_cmd = ''

    print('\nCamera ready — hold signs in front of the camera.\n')

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ok:
            break

        detections, processed, hsv = det.detect(frame)
        command, conf = det.vote(detections)

        dt = time.time() - t0
        fps = 0.9 * fps + 0.1 / max(dt, 1e-4)

        display = draw(cv2.resize(frame, (640, 480)),
                       detections, command, conf, fps)
        cv2.imshow('Sign Detection', display)

        if show_dbg:
            cv2.imshow('Color Masks  (R | Y | B)', debug_masks(hsv))
            cv2.imshow('Processed', processed)

        # Log command changes
        if command != prev_cmd and command != 'NO_SIGN':
            print(f'  >>> {command}  (confidence {conf:.0%},  {fps:.0f} FPS)')
        prev_cmd = command

        # ── key handling ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_dbg = not show_dbg
            if not show_dbg:
                cv2.destroyWindow('Color Masks  (R | Y | B)')
                cv2.destroyWindow('Processed')
            print(f'  Debug masks: {"ON" if show_dbg else "OFF"}')
        elif key == ord(' '):
            det.history.clear()
            print('  Vote history cleared')
        elif key == ord('+') or key == ord('='):
            det.min_area = max(200, det.min_area - 300)
            print(f'  Min area ↓ {det.min_area}')
        elif key == ord('-'):
            det.min_area += 300
            print(f'  Min area ↑ {det.min_area}')

    cap.release()
    cv2.destroyAllWindows()
    print('\nDone.')


if __name__ == '__main__':
    main()
