"""
test_gtsrb_live.py
==================
Run this on the Raspberry Pi BEFORE integrating into the robot.

Shows a live camera feed with:
  - Raw GTSRB class prediction (all 43 classes)
  - Mapped label (STOP / SLOW_DOWN / TURN_LEFT / TURN_RIGHT / NO_SIGN)
  - Confidence score
  - Color ROI bounding box

This lets you hold each sign in front of the camera and verify:
  ✓ Does the ROI filter find the sign?
  ✓ Does the classifier label it correctly?
  ✓ Is confidence consistently above 0.85?

If any sign fails → move to fine-tuning (only 30 images needed at that point)

Usage:
  python3 test_gtsrb_live.py
  python3 test_gtsrb_live.py --model /path/to/model.tflite
  python3 test_gtsrb_live.py --no-roi   (skip color filter, classify full frame crop)

Press Q to quit.
Press S to save a snapshot of the current frame (useful for collecting failures).
"""

import cv2
import numpy as np
import argparse
import time
import os

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# ─────────────────────────────────────────────────────────────────────────────
# GTSRB CLASS MAPPING
# Maps all 43 GTSRB class indices to your 5 robot commands.
# Anything not listed defaults to NO_SIGN.
# ─────────────────────────────────────────────────────────────────────────────
GTSRB_CLASSES = {
    0: "Speed limit 20",    1: "Speed limit 30",
    2: "Speed limit 50",    3: "Speed limit 60",
    4: "Speed limit 70",    5: "Speed limit 80",
    6: "End speed limit 80",7: "Speed limit 100",
    8: "Speed limit 120",   9: "No passing",
    10:"No passing (trucks)",11:"Priority road ahead",
    12:"Priority road",     13:"Yield",
    14:"Stop",              15:"No vehicles",
    16:"No trucks",         17:"No entry",
    18:"General caution",   19:"Curve left",
    20:"Curve right",       21:"Double curve",
    22:"Bumpy road",        23:"Slippery road",
    24:"Road narrows right",25:"Road work",
    26:"Traffic signals",   27:"Pedestrians",
    28:"Children crossing", 29:"Bicycles crossing",
    30:"Ice/snow",          31:"Wild animals",
    32:"End restrictions",  33:"Turn right ahead",
    34:"Turn left ahead",   35:"Go straight",
    36:"Straight or right", 37:"Straight or left",
    38:"Keep right",        39:"Keep left",
    40:"Roundabout",        41:"End no passing",
    42:"End no passing (trucks)"
}

# ── YOUR 5-CLASS MAPPING ──────────────────────────────────────────────────────
# Edit this if your "SLOW DOWN" sign looks more like a specific GTSRB class.
# For example: if your slow-down sign is a yellow warning triangle, add class 18.
GTSRB_TO_ROBOT = {
    # STOP
    14: "STOP",

    # TURN RIGHT
    33: "TURN_RIGHT",
    36: "TURN_RIGHT",    # straight-or-right (close enough)
    38: "TURN_RIGHT",    # keep right

    # TURN LEFT
    34: "TURN_LEFT",
    37: "TURN_LEFT",     # straight-or-left
    39: "TURN_LEFT",     # keep left

    # SLOW DOWN — speed limit signs + general warning
    0:  "SLOW_DOWN",     # 20 km/h
    1:  "SLOW_DOWN",     # 30 km/h
    2:  "SLOW_DOWN",     # 50 km/h
    3:  "SLOW_DOWN",     # 60 km/h
    4:  "SLOW_DOWN",     # 70 km/h
    5:  "SLOW_DOWN",     # 80 km/h
    7:  "SLOW_DOWN",     # 100 km/h
    8:  "SLOW_DOWN",     # 120 km/h
    18: "SLOW_DOWN",     # General caution triangle
    25: "SLOW_DOWN",     # Road work
}

FRAME_W, FRAME_H = 320, 240
CROP_SIZE        = 64
CONFIDENCE_THRESHOLD = 0.75    # lower threshold for testing (raise to 0.85 in prod)

# ─────────────────────────────────────────────────────────────────────────────
# COLOR ROI FILTER (same as sign_node.py)
# ─────────────────────────────────────────────────────────────────────────────
class ColorROIFilter:
    # 1. Increased Vibrancy (Saturation and Value 90-100 min)
    RED = [
        (np.array([0, 120, 120]), np.array([10, 255, 255])),
        (np.array([170,120,120]), np.array([179,255,255]))
    ]
    YELLOW =  (np.array([18, 100, 100]),  np.array([38, 255, 255]))
    BLUE   =  (np.array([100,100,80]),  np.array([130,255, 255]))
    MIN_AREA = 1500
    MAX_AREA = 150000
    IGNORE_BOTTOM_PERCENT = 0.40 # Discard bottom 40% of frame

    def find_roi(self, frame):
        frame = cv2.GaussianBlur(frame, (5,5), 0)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        h, s, v = cv2.split(hsv)
        hsv = cv2.merge([h, s, clahe.apply(v)])

        red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self.RED:
            red_mask |= cv2.inRange(hsv, lo, hi)
        yellow_mask = cv2.inRange(hsv, *self.YELLOW)
        blue_mask   = cv2.inRange(hsv, *self.BLUE)
        
        # Completely zero out the bottom 40% of the masks to ignore the floor
        h_mask = red_mask.shape[0]
        cutoff = int(h_mask * (1.0 - self.IGNORE_BOTTOM_PERCENT))
        red_mask[cutoff:, :] = 0
        yellow_mask[cutoff:, :] = 0
        blue_mask[cutoff:, :] = 0

        best_area, best_crop, best_box = 0, None, None

        for mask in [red_mask, yellow_mask, blue_mask]:
            k = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                area = cv2.contourArea(c)
                if not (self.MIN_AREA < area < self.MAX_AREA):
                    continue
                x, y, w, h = cv2.boundingRect(c)
                if not (0.65 < w/float(h) < 1.45):
                    continue
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    continue
                if area / hull_area < 0.73:
                    continue
                    
                # Polygon approximation to ensure it's a harsh geometric shape (triangle, rectangle, octagon, circle)
                epsilon = 0.04 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                if not (3 <= len(approx) <= 12):
                    continue
                if area > best_area:
                    best_area = area
                    pad = int(max(w, h) * 0.10)
                    x1 = max(0, x - pad);  y1 = max(0, y - pad)
                    x2 = min(frame.shape[1], x+w+pad)
                    y2 = min(frame.shape[0], y+h+pad)
                    best_crop = frame[y1:y2, x1:x2]
                    best_box  = (x1, y1, x2-x1, y2-y1)

        return best_crop, best_box, red_mask, yellow_mask, blue_mask


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
class GTSRBClassifier:

    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path, num_threads=2)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()
        self.out = self.interpreter.get_output_details()
        self.is_int8 = self.inp[0]["dtype"] in (np.int8, np.uint8)
        print(f"Model loaded: {model_path}")
        print(f"Input shape:  {self.inp[0]['shape']}")
        print(f"Input dtype:  {self.inp[0]['dtype']}")
        print(f"INT8 mode:    {self.is_int8}")

    def classify(self, crop):
        """Returns (gtsrb_class_id, gtsrb_class_name, robot_label, confidence)"""
        img = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.is_int8:
            inp = (img.astype(np.float32) - 128).astype(np.int8)
        else:
            inp = img.astype(np.float32) / 255.0

        self.interpreter.set_tensor(self.inp[0]["index"], np.expand_dims(inp, 0))
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.out[0]["index"])[0].astype(np.float32)

        if self.out[0]["dtype"] == np.int8:
            scale, zp = self.out[0]["quantization"]
            output = (output - zp) * scale

        # softmax
        output = np.exp(output - np.max(output))
        output /= output.sum()

        class_id   = int(np.argmax(output))
        confidence = float(output[class_id])
        if len(output) == 5:
            # Custom Model (5 classes directly)
            CUSTOM_CLASSES = ["NO_SIGN", "SLOW_DOWN", "STOP", "TURN_LEFT", "TURN_RIGHT"]
            gtsrb_name = "Custom Model"
            if confidence >= CONFIDENCE_THRESHOLD:
                robot_label = CUSTOM_CLASSES[class_id]
            else:
                robot_label = "NO_SIGN"
        else:
            # Original GTSRB Model (43 classes)
            gtsrb_name = GTSRB_CLASSES.get(class_id, f"Class {class_id}")
            robot_label = GTSRB_TO_ROBOT.get(class_id, "NO_SIGN") if confidence >= CONFIDENCE_THRESHOLD else "NO_SIGN"

        return class_id, gtsrb_name, robot_label, confidence


# ─────────────────────────────────────────────────────────────────────────────
# LIVE TEST LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_live_test(model_path, use_roi):
    roi_filter = ColorROIFilter()
    classifier = GTSRBClassifier(model_path)

    import sys
    if sys.platform == 'darwin':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    os.makedirs("snapshots", exist_ok=True)
    snap_count = 0
    fps_timer  = time.time()
    fps        = 0.0
    frame_count = 0

    print("\nLive test running...")
    print("  Hold each sign in front of the camera")
    print("  Press S to save a snapshot")
    print("  Press Q to quit")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # frame = cv2.flip(frame, -1)   # flip if camera is upside-down
        display = frame.copy()
        frame_count += 1

        # ── FPS counter ───────────────────────────────────────────────────────
        if frame_count % 30 == 0:
            fps = 30.0 / (time.time() - fps_timer)
            fps_timer = time.time()

        # ── Find ROI ──────────────────────────────────────────────────────────
        m_red, m_yel, m_blu = None, None, None
        if use_roi:
            crop, box, m_red, m_yel, m_blu = roi_filter.find_roi(frame)
        else:
            # fallback: use centre 60% of frame as crop
            h, w = frame.shape[:2]
            m = 0.20
            crop = frame[int(h*m):int(h*(1-m)), int(w*m):int(w*(1-m))]
            box  = (int(w*m), int(h*m), int(w*0.6), int(h*0.6))

        # ── Classify ──────────────────────────────────────────────────────────
        if crop is not None and crop.size > 0:
            class_id, gtsrb_name, robot_label, confidence = classifier.classify(crop)

            # draw bounding box
            if box:
                x, y, bw, bh = box
                box_color = (0,255,0) if robot_label != "NO_SIGN" else (100,100,100)
                cv2.rectangle(display, (x,y), (x+bw, y+bh), box_color, 2)

            # overlay text
            _text(display, f"GTSRB: {gtsrb_name} ({class_id})", (5, 20),  (200,200,0))
            _text(display, f"Robot: {robot_label}",              (5, 45),
                  (0,255,0) if robot_label != "NO_SIGN" else (100,100,100))
            _text(display, f"Conf:  {confidence:.2f}",           (5, 70),
                  (0,255,0) if confidence >= CONFIDENCE_THRESHOLD else (0,100,255))

            # big warning if confidence is low
            if confidence < CONFIDENCE_THRESHOLD and robot_label != "NO_SIGN":
                _text(display, "LOW CONFIDENCE", (5, 100), (0, 0, 255), scale=0.6)

        else:
            _text(display, "No ROI found", (5, 20), (100, 100, 100))

        _text(display, f"FPS: {fps:.1f}", (FRAME_W-80, 20), (200,200,200))
        _text(display, f"ROI: {'ON' if use_roi else 'OFF'}", (FRAME_W-80, 45), (200,200,200))

        cv2.imshow("GTSRB Live Test", display)
        if use_roi:
            if m_red is not None: cv2.imshow("Mask: RED",    m_red)
            if m_yel is not None: cv2.imshow("Mask: YELLOW", m_yel)
            if m_blu is not None: cv2.imshow("Mask: BLUE",   m_blu)
            
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            path = f"snapshots/snap_{snap_count:04d}.jpg"
            cv2.imwrite(path, frame)
            print(f"Snapshot saved → {path}")
            snap_count += 1

    cap.release()
    cv2.destroyAllWindows()


def _text(img, text, pos, color, scale=0.55):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0),   3)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,     1)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="sign_classifier_gtsrb_int8.tflite")
    parser.add_argument("--no-roi", action="store_true",
                        help="Skip color filter, use centre crop instead")
    args = parser.parse_args()

    run_live_test(args.model, use_roi=not args.no_roi)
