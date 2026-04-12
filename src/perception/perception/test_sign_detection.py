"""
Sign Detection + Tracking Script (Standalone — No ROS2)
========================================================
Detects action signs using your laptop camera:
  - STOP        → Red octagonal/circular sign
  - SLOW_DOWN   → Yellow triangular/diamond sign
  - TURN_LEFT   → Blue circular sign with left arrow
  - TURN_RIGHT  → Blue circular sign with right arrow

Tracking strategy
─────────────────
  • Full detection (expensive)  runs every DETECT_INTERVAL frames,
    OR immediately when no trackers are active.
  • Between detection frames, each tracked sign is updated with a
    KCF/CSRT tracker (very cheap — just a bounding-box prediction).
  • New detections are matched to existing trackers via IoU so we
    never create duplicate trackers for the same sign.
  • A tracker is dropped when:
      – the OpenCV tracker reports failure for MAX_LOST consecutive frames, or
      – the bounding box has drifted fully outside the frame.

Usage:  python test_sign_detection.py

Controls:
    Q        Quit
    D        Toggle debug mask windows
    Space    Clear vote history + all trackers
    +/-      Adjust min contour area (detection sensitivity)
    T        Cycle tracker type  KCF → CSRT → MOSSE
"""

import cv2
import numpy as np
import time
from collections import deque


# ═══════════════════════════════════════════════════════════
#  Tunable constants
# ═══════════════════════════════════════════════════════════

DETECT_INTERVAL  = 15    # run full detection every N frames
IOU_MATCH_THRESH = 0.30  # minimum IoU to link a detection to an existing tracker
MAX_LOST         = 4     # consecutive missed frames before a tracker is dropped
FRAME_W, FRAME_H = 640, 480


# ═══════════════════════════════════════════════════════════
#  HSV Color Ranges  (tune for your lighting)
# ═══════════════════════════════════════════════════════════

RED_RANGES = [
    (np.array([0,   120, 70]),  np.array([10,  255, 255])),
    (np.array([170, 120, 70]),  np.array([179, 255, 255])),
]
YELLOW_RANGE = (
    np.array([20, 100, 100]),
    np.array([35, 255, 255]),
)
BLUE_RANGE = (
    np.array([100, 80, 60]),
    np.array([130, 255, 255]),
)


# ═══════════════════════════════════════════════════════════
#  Tracker factory  (handles OpenCV API differences)
# ═══════════════════════════════════════════════════════════

TRACKER_TYPES = ['KCF', 'CSRT', 'MOSSE']


def create_tracker(tracker_type: str = 'KCF'):
    """
    Create an OpenCV tracker.
    Tries the modern API first (OpenCV 4.5+), then the legacy module
    (opencv-contrib-python ≥ 4.5).
    """
    builders = {
        'KCF':   [lambda: cv2.TrackerKCF_create(),
                  lambda: cv2.legacy.TrackerKCF_create()],
        'CSRT':  [lambda: cv2.TrackerCSRT_create(),
                  lambda: cv2.legacy.TrackerCSRT_create()],
        'MOSSE': [lambda: cv2.legacy.TrackerMOSSE_create(),
                  lambda: cv2.TrackerMOSSE_create()],
    }
    for build in builders.get(tracker_type, builders['KCF']):
        try:
            return build()
        except AttributeError:
            continue
    raise RuntimeError(
        f'Cannot create tracker "{tracker_type}". '
        'Make sure opencv-contrib-python is installed.'
    )


# ═══════════════════════════════════════════════════════════
#  TrackedSign — one sign being actively tracked
# ═══════════════════════════════════════════════════════════

class TrackedSign:
    """Wraps a single OpenCV tracker for one detected sign."""

    def __init__(self, label: str, conf: float, bbox: tuple,
                 frame: np.ndarray, tracker_type: str = 'KCF'):
        self.label   = label
        self.conf    = conf
        self.bbox    = bbox     # (x, y, w, h) kept up to date every frame
        self.lost    = 0        # consecutive failed-update counter
        self._ttype  = tracker_type
        self._tracker = create_tracker(tracker_type)
        self._tracker.init(frame, bbox)

    # ── advance by one frame ─────────────────────────────

    def update(self, frame: np.ndarray) -> bool:
        """
        Advance the tracker by one frame.
        Returns True  → keep this track alive.
        Returns False → remove this track (failed or left the frame).
        """
        ok, raw = self._tracker.update(frame)

        if ok:
            x, y, w, h = (int(v) for v in raw)
            # Remove if box has wandered completely off-screen
            if x + w <= 0 or y + h <= 0 or x >= FRAME_W or y >= FRAME_H:
                return False
            self.bbox = (x, y, w, h)
            self.lost = 0
            return True
        else:
            self.lost += 1
            return self.lost < MAX_LOST   # tolerate a few bad frames

    # ── re-anchor to a fresh detection bbox ─────────────

    def reinit(self, label: str, conf: float, bbox: tuple,
               frame: np.ndarray):
        """Snap the tracker to a newly confirmed bounding box."""
        self.label  = label
        self.conf   = conf
        self.bbox   = bbox
        self.lost   = 0
        self._tracker = create_tracker(self._ttype)
        self._tracker.init(frame, bbox)

    def as_detection(self) -> tuple:
        """Return (label, conf, bbox) matching the format used elsewhere."""
        return (self.label, self.conf, self.bbox)


# ═══════════════════════════════════════════════════════════
#  TrackingManager — owns all active TrackedSign objects
# ═══════════════════════════════════════════════════════════

class TrackingManager:

    def __init__(self, tracker_type: str = 'KCF'):
        self.tracker_type           = tracker_type
        self._tracks: list[TrackedSign] = []

    # ── update all trackers each frame (fast path) ───────

    def update(self, frame: np.ndarray) -> list[tuple]:
        """
        Advance every tracker by one frame.
        Stale or off-screen trackers are removed.
        Returns the current list of (label, conf, bbox) detections.
        """
        self._tracks = [t for t in self._tracks if t.update(frame)]
        return [t.as_detection() for t in self._tracks]

    # ── register fresh detector output (slow path) ───────

    def register(self, detections: list[tuple], frame: np.ndarray):
        """
        Match each fresh detection to an existing tracker by IoU.
          IoU ≥ threshold  →  re-anchor that tracker (keeps it sharp).
          No match found   →  spawn a new TrackedSign.
        This ensures one tracker per physical sign even if detection
        fires multiple times.
        """
        for label, conf, bbox in detections:
            idx = self._best_match(bbox)
            if idx is not None:
                self._tracks[idx].reinit(label, conf, bbox, frame)
            else:
                self._tracks.append(
                    TrackedSign(label, conf, bbox, frame, self.tracker_type)
                )

    def clear(self):
        self._tracks.clear()

    @property
    def has_tracks(self) -> bool:
        return bool(self._tracks)

    # ── IoU helpers ──────────────────────────────────────

    @staticmethod
    def _iou(b1: tuple, b2: tuple) -> float:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        ix    = max(0, min(x1+w1, x2+w2) - max(x1, x2))
        iy    = max(0, min(y1+h1, y2+h2) - max(y1, y2))
        inter = ix * iy
        union = w1*h1 + w2*h2 - inter
        return inter / union if union > 0 else 0.0

    def _best_match(self, bbox: tuple):
        best_idx, best_iou = None, IOU_MATCH_THRESH
        for i, t in enumerate(self._tracks):
            iou = self._iou(bbox, t.bbox)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        return best_idx


# ═══════════════════════════════════════════════════════════
#  SignDetector — color / shape detection (unchanged logic)
# ═══════════════════════════════════════════════════════════

class SignDetector:

    def __init__(self):
        self.min_area    = 1500
        self.max_area    = 120_000
        self.epsilon_factor = 0.03

        self.stop_min_circularity  = 0.65
        self.slow_min_circularity  = 0.35
        self.turn_min_circularity  = 0.75

        self.vote_window    = 7
        self.vote_threshold = 4
        self.history        = deque(maxlen=self.vote_window)

    # ── preprocessing ────────────────────────────────────

    def preprocess(self, frame):
        img = cv2.resize(frame, (FRAME_W, FRAME_H))

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v   = np.clip(cv2.add(v, 20), 0, 255).astype(np.uint8)
        img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img   = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)),
                              cv2.COLOR_LAB2BGR)
        return cv2.GaussianBlur(img, (5, 5), 0)

    # ── full detection pipeline ───────────────────────────

    def detect(self, frame):
        processed = self.preprocess(frame)
        hsv       = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        det       = self._find_red(hsv)
        det      += self._find_yellow(hsv)
        det      += self._find_blue(hsv, processed)
        return det, processed, hsv

    # ── red → STOP ───────────────────────────────────────

    def _find_red(self, hsv):
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in RED_RANGES:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        return self._classify_contours(
            self._clean(mask), 'STOP',
            min_vertices=6, max_vertices=14,
            min_circularity=self.stop_min_circularity)

    # ── yellow → SLOW_DOWN ───────────────────────────────

    def _find_yellow(self, hsv):
        lo, hi = YELLOW_RANGE
        return self._classify_contours(
            self._clean(cv2.inRange(hsv, lo, hi)), 'SLOW_DOWN',
            min_vertices=3, max_vertices=6,
            min_circularity=self.slow_min_circularity)

    # ── blue → TURN_LEFT / TURN_RIGHT ────────────────────

    def _find_blue(self, hsv, processed):
        lo, hi = BLUE_RANGE
        mask   = self._clean(cv2.inRange(hsv, lo, hi))
        dets   = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            if 4 * np.pi * area / (peri * peri) < self.turn_min_circularity:
                continue                         # must be a circle
            x, y, w, h = cv2.boundingRect(cnt)
            if not (0.6 < w / h < 1.6):
                continue
            direction = self._arrow_direction(processed[y:y+h, x:x+w])
            if direction:
                circ = 4 * np.pi * area / (peri * peri)
                dets.append((f'TURN_{direction}',
                              round(min(1.0, circ + 0.1), 2),
                              (x, y, w, h)))
        return dets

    # ── arrow direction ──────────────────────────────────

    def _arrow_direction(self, roi):
        if roi is None or roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
            return None
        hsv_roi   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_roi, *BLUE_RANGE)
        gray      = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        am        = cv2.bitwise_and(bright, cv2.bitwise_not(blue_mask))
        k         = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        am        = cv2.morphologyEx(cv2.morphologyEx(am, cv2.MORPH_CLOSE, k),
                                     cv2.MORPH_OPEN,  k)
        nw = cv2.countNonZero(am)
        total = am.shape[0] * am.shape[1]
        if nw < total * 0.05 or nw > total * 0.80:
            return None
        contours, _ = cv2.findContours(am, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        hull = cv2.convexHull(max(contours, key=cv2.contourArea)).squeeze()
        if hull.ndim != 2 or len(hull) < 5:
            return self._mass_dir(am)
        n  = len(hull)
        li = int(np.argmin(hull[:, 0]))
        ri = int(np.argmax(hull[:, 0]))
        la = self._angle(hull, li, n)
        ra = self._angle(hull, ri, n)
        if abs(ra - la) > 10:
            return 'LEFT' if la < ra else 'RIGHT'
        return self._mass_dir(am)

    @staticmethod
    def _angle(pts, idx, n):
        p  = pts[(idx-1) % n].astype(float)
        q  = pts[idx].astype(float)
        r  = pts[(idx+1) % n].astype(float)
        v1, v2 = p - q, r - q
        cos_a  = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    @staticmethod
    def _mass_dir(mask):
        h, w = mask.shape
        lm = float(np.sum(mask[:, :w//2]))
        rm = float(np.sum(mask[:, w//2:]))
        t  = lm + rm
        if t == 0:
            return None
        r = (lm - rm) / t
        if   r >  0.15: return 'LEFT'
        elif r < -0.15: return 'RIGHT'
        return None

    # ── shared helpers ───────────────────────────────────

    def _clean(self, mask):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        return cv2.morphologyEx(
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k),
            cv2.MORPH_OPEN, k)

    def _classify_contours(self, mask, label, min_vertices=3, max_vertices=12,
                           min_circularity=0.3):
        dets = []
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
            if (min_vertices <= verts <= max_vertices
                    and circ >= min_circularity
                    and 0.5 < (w / h if h else 0) < 2.0):
                conf = min(1.0, circ + 0.2 * (area / self.max_area))
                dets.append((label, round(conf, 2), (x, y, w, h)))
        return dets

    # ── temporal voting ──────────────────────────────────

    def vote(self, detections):
        if detections:
            self.history.append(max(detections, key=lambda d: d[1])[0])
        else:
            self.history.append('NO_SIGN')

        if len(self.history) < 3:
            return 'NO_SIGN', 0.0

        counts: dict[str, int] = {}
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
    'TURN_LEFT':  (255, 150,   0),
    'TURN_RIGHT': (255, 100,   0),
    'NO_SIGN':    (100, 100, 100),
}


def draw(frame, tracked, command, conf, fps, mode_label, tracker_type):
    out = frame.copy()
    for sign, c, (x, y, w, h) in tracked:
        col = COLORS.get(sign, (255, 255, 255))
        cv2.rectangle(out, (x, y), (x+w, y+h), col, 2)
        cv2.circle(out, (x + w//2, y + h//2), 4, col, -1)
        lbl = f'{sign} {c:.0%}'
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x, y-th-10), (x+tw+4, y), col, -1)
        cv2.putText(out, lbl, (x+2, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    col = COLORS.get(command, (100, 100, 100))
    cv2.rectangle(out, (0, 0), (FRAME_W, 48), (25, 25, 25), -1)
    cv2.putText(out, f'CMD: {command}', (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    bw = int(180 * conf)
    cv2.rectangle(out, (440, 12), (440+bw, 38), col, -1)
    cv2.rectangle(out, (440, 12), (620, 38), (80, 80, 80), 1)
    cv2.putText(out, f'{conf:.0%}', (445, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.rectangle(out, (0, FRAME_H-28), (FRAME_W, FRAME_H), (25, 25, 25), -1)
    cv2.putText(out,
                f'{fps:.0f} FPS   mode:{mode_label}   tracker:{tracker_type}',
                (8, FRAME_H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
    return out


def debug_masks(hsv):
    r = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in RED_RANGES:
        r = cv2.bitwise_or(r, cv2.inRange(hsv, lo, hi))
    y = cv2.inRange(hsv, *YELLOW_RANGE)
    b = cv2.inRange(hsv, *BLUE_RANGE)
    row = np.hstack([cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x in (r, y, b)])
    for txt, xpos, col in [('RED', 10, (0,0,255)),
                            ('YELLOW', 650, (0,255,255)),
                            ('BLUE', 1290, (255,100,0))]:
        cv2.putText(row, txt, (xpos, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return row


# ═══════════════════════════════════════════════════════════
#  Main loop
# ═══════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('  Sign Detection + Tracking')
    print('  STOP | SLOW_DOWN | TURN_LEFT | TURN_RIGHT')
    print('=' * 60)
    print('  Q        quit')
    print('  D        toggle debug mask windows')
    print('  Space    clear votes + all trackers')
    print('  +/-      adjust detection sensitivity')
    print('  T        cycle tracker  KCF → CSRT → MOSSE')
    print('=' * 60)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: cannot open camera')
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    detector    = SignDetector()
    tracker_idx = 0
    tracker_mgr = TrackingManager(TRACKER_TYPES[tracker_idx])

    show_dbg  = False
    fps       = 0.0
    prev_cmd  = ''
    frame_idx = 0
    last_hsv  = None

    print(f'\nCamera ready.  Tracker: {TRACKER_TYPES[tracker_idx]}\n')

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        # ── detection gate ────────────────────────────────
        # Run full detection when:  (a) no trackers exist, or
        #                           (b) we have hit the interval boundary.
        run_det = (frame_idx % DETECT_INTERVAL == 0
                   or not tracker_mgr.has_tracks)

        if run_det:
            new_dets, processed, hsv = detector.detect(frame)
            last_hsv = hsv
            tracker_mgr.register(new_dets, processed)
            mode_label = 'DETECT'
        else:
            mode_label = 'TRACK '

        # ── tracker update (runs every frame — very fast) ─
        tracked = tracker_mgr.update(frame)

        command, conf = detector.vote(tracked)

        dt  = time.time() - t0
        fps = 0.9 * fps + 0.1 / max(dt, 1e-4)

        display = draw(frame, tracked, command, conf, fps,
                       mode_label, TRACKER_TYPES[tracker_idx])
        cv2.imshow('Sign Detection + Tracking', display)

        if show_dbg and last_hsv is not None:
            cv2.imshow('Color Masks  (R | Y | B)', debug_masks(last_hsv))
            if run_det:
                cv2.imshow('Processed', processed)

        if command != prev_cmd and command != 'NO_SIGN':
            print(f'  >>> {command}  '
                  f'(conf {conf:.0%},  {fps:.0f} FPS,  '
                  f'active tracks: {len(tracker_mgr._tracks)})')
        prev_cmd  = command
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_dbg = not show_dbg
            if not show_dbg:
                for wn in ('Color Masks  (R | Y | B)', 'Processed'):
                    try: cv2.destroyWindow(wn)
                    except: pass
            print(f'  Debug: {"ON" if show_dbg else "OFF"}')
        elif key == ord(' '):
            detector.history.clear()
            tracker_mgr.clear()
            frame_idx = 0
            print('  Cleared votes + trackers')
        elif key in (ord('+'), ord('=')):
            detector.min_area = max(200, detector.min_area - 300)
            print(f'  Min area ↓ {detector.min_area}')
        elif key == ord('-'):
            detector.min_area += 300
            print(f'  Min area ↑ {detector.min_area}')
        elif key == ord('t'):
            tracker_idx = (tracker_idx + 1) % len(TRACKER_TYPES)
            new_type    = TRACKER_TYPES[tracker_idx]
            tracker_mgr = TrackingManager(new_type)
            detector.history.clear()
            frame_idx = 0
            print(f'  Tracker → {new_type}  (trackers reset)')

    cap.release()
    cv2.destroyAllWindows()
    print('\nDone.')


if __name__ == '__main__':
    main()