"""
sign_detection_node.py  (GTSRB TFLite version)
================================================
ROS2 Traffic Sign Detection Node — uses a fine-tuned GTSRB TFLite model
(5 custom classes: NO_SIGN, SLOW_DOWN, STOP, TURN_LEFT, TURN_RIGHT).

Pipeline:
  1. Capture frame from Pi camera
  2. Color ROI filter (red/yellow/blue HSV masks → crop candidate)
  3. TFLite classifier on crop → robot label + confidence
  4. Temporal smoother → stabilised label
  5. Map label → VehicleCmd and publish

Publishes:
  /teleop/raw_cmd           (vehicle_interfaces/VehicleCmd)  — velocity + heading
  /perception/sign_detected (std_msgs/String)                — label string

Setup on Pi:
  pip install tflite-runtime
  taskset -c 3 ros2 launch perception sign_detection.launch.py

DEBUG CONTROLS (env vars you can set before launching):
  SIGN_DEBUG=1          → verbose per-frame logs (ROI stats, raw scores, conf)
  SIGN_SAVE_CROPS=1     → saves every crop to /tmp/sign_crops/ for inspection
  SIGN_SAVE_INTERVAL=30 → how often (in classified frames) to save a crop (default 30)
"""

import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vehicle_interfaces.msg import VehicleCmd
import cv2
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    _TFLITE_SOURCE = "tflite_runtime"
except ImportError:
    import tensorflow.lite as tflite
    _TFLITE_SOURCE = "tensorflow.lite"

# ── Debug flags (set via environment variables) ───────────────────────────────
_DEBUG         = os.environ.get("SIGN_DEBUG", "0") == "1"
_SAVE_CROPS    = os.environ.get("SIGN_SAVE_CROPS", "0") == "1"
_SAVE_INTERVAL = int(os.environ.get("SIGN_SAVE_INTERVAL", "30"))
_CROP_DIR      = "/tmp/sign_crops"

if _SAVE_CROPS:
    os.makedirs(_CROP_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# COLOR ROI FILTER  (ported from test_gtsrb_live.py)
# ─────────────────────────────────────────────────────────────────────────────
class ColorROIFilter:
    """Finds the most prominent sign-coloured region in the frame."""

    def __init__(self, params: dict, logger=None):
        self.RED = [
            (np.array(params['red_range1_low']),  np.array(params['red_range1_high'])),
            (np.array(params['red_range2_low']),  np.array(params['red_range2_high'])),
        ]
        self.YELLOW = (np.array(params['yellow_range_low']), np.array(params['yellow_range_high']))
        self.BLUE   = (np.array(params['blue_range_low']),   np.array(params['blue_range_high']))

        self.MIN_AREA = params.get('min_area', 1500)
        self.MAX_AREA = params.get('max_area', 150000)
        self.IGNORE_BOTTOM_PERCENT = params.get('ignore_bottom_percent', 0.40)
        self.MIN_ASPECT  = params.get('min_aspect_ratio', 0.65)
        self.MAX_ASPECT  = params.get('max_aspect_ratio', 1.45)
        self.MIN_SOLIDITY = params.get('min_solidity', 0.73)
        self.logger = logger

        if logger:
            logger.info("[ROI] ColorROIFilter initialized")
            logger.info(f"[ROI]   RED range1  : {params['red_range1_low']} → {params['red_range1_high']}")
            logger.info(f"[ROI]   RED range2  : {params['red_range2_low']} → {params['red_range2_high']}")
            logger.info(f"[ROI]   YELLOW range: {params['yellow_range_low']} → {params['yellow_range_high']}")
            logger.info(f"[ROI]   BLUE range  : {params['blue_range_low']} → {params['blue_range_high']}")
            logger.info(f"[ROI]   area        : {self.MIN_AREA} – {self.MAX_AREA}")
            logger.info(f"[ROI]   aspect ratio: {self.MIN_ASPECT} – {self.MAX_ASPECT}")
            logger.info(f"[ROI]   min solidity: {self.MIN_SOLIDITY}")
            logger.info(f"[ROI]   ignore bottom: {self.IGNORE_BOTTOM_PERCENT*100:.0f}%")

    def find_roi(self, frame, debug=False):
        """
        Returns (crop, bounding_box, red_mask, yellow_mask, blue_mask).

        When debug=True, also logs per-contour rejection reasons so you can
        see exactly why a candidate is being discarded.
        """
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # CLAHE on V channel — normalises brightness
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
        h, s, v = cv2.split(hsv)
        hsv = cv2.merge([h, s, clahe.apply(v)])

        # Build colour masks
        red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self.RED:
            red_mask |= cv2.inRange(hsv, lo, hi)
        yellow_mask = cv2.inRange(hsv, *self.YELLOW)
        blue_mask   = cv2.inRange(hsv, *self.BLUE)

        # Zero-out bottom portion (floor / chassis)
        h_mask = red_mask.shape[0]
        cutoff = int(h_mask * (1.0 - self.IGNORE_BOTTOM_PERCENT))
        red_mask[cutoff:, :] = 0
        yellow_mask[cutoff:, :] = 0
        blue_mask[cutoff:, :] = 0

        if debug and self.logger:
            self.logger.debug(
                f"[ROI] mask pixel counts → red={int(red_mask.sum()//255)} "
                f"yellow={int(yellow_mask.sum()//255)} "
                f"blue={int(blue_mask.sum()//255)}"
            )

        best_area, best_crop, best_box = 0, None, None
        color_names = ["RED", "YELLOW", "BLUE"]

        for color_name, mask in zip(color_names, [red_mask, yellow_mask, blue_mask]):
            k = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if debug and self.logger and contours:
                self.logger.debug(f"[ROI]   {color_name}: {len(contours)} contour(s) found")

            for idx, c in enumerate(contours):
                area = cv2.contourArea(c)

                # ── area gate ────────────────────────────────────────────────
                if not (self.MIN_AREA < area < self.MAX_AREA):
                    if debug and self.logger:
                        self.logger.debug(
                            f"[ROI]     {color_name}[{idx}] REJECTED area={area:.0f} "
                            f"(need {self.MIN_AREA}–{self.MAX_AREA})"
                        )
                    continue

                x, y, w, bh = cv2.boundingRect(c)
                aspect = w / float(bh)

                # ── aspect gate ──────────────────────────────────────────────
                if not (self.MIN_ASPECT < aspect < self.MAX_ASPECT):
                    if debug and self.logger:
                        self.logger.debug(
                            f"[ROI]     {color_name}[{idx}] REJECTED aspect={aspect:.2f} "
                            f"(need {self.MIN_ASPECT}–{self.MAX_ASPECT})"
                        )
                    continue

                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    continue
                solidity = area / hull_area

                # ── solidity gate ────────────────────────────────────────────
                if solidity < self.MIN_SOLIDITY:
                    if debug and self.logger:
                        self.logger.debug(
                            f"[ROI]     {color_name}[{idx}] REJECTED solidity={solidity:.2f} "
                            f"(need ≥{self.MIN_SOLIDITY})"
                        )
                    continue

                # ── polygon gate ─────────────────────────────────────────────
                epsilon = 0.04 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                if not (3 <= len(approx) <= 12):
                    if debug and self.logger:
                        self.logger.debug(
                            f"[ROI]     {color_name}[{idx}] REJECTED vertices={len(approx)} "
                            f"(need 3–12)"
                        )
                    continue

                if debug and self.logger:
                    self.logger.debug(
                        f"[ROI]     {color_name}[{idx}] PASSED "
                        f"area={area:.0f} aspect={aspect:.2f} "
                        f"solidity={solidity:.2f} verts={len(approx)}"
                    )

                if area > best_area:
                    best_area = area
                    pad = int(max(w, bh) * 0.10)
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(frame.shape[1], x + w + pad)
                    y2 = min(frame.shape[0], y + bh + pad)
                    best_crop = frame[y1:y2, x1:x2]
                    best_box  = (x1, y1, x2 - x1, y2 - y1)

        return best_crop, best_box, red_mask, yellow_mask, blue_mask


# ─────────────────────────────────────────────────────────────────────────────
# GTSRB TFLITE CLASSIFIER  (ported from test_gtsrb_live.py)
# ─────────────────────────────────────────────────────────────────────────────
class GTSRBClassifier:
    """Runs inference on a cropped sign image using a TFLite model."""

    # Custom 5-class order (must match training folder alphabetical order)
    CUSTOM_CLASSES = ["NO_SIGN", "SLOW_DOWN", "STOP", "TURN_LEFT", "TURN_RIGHT"]

    def __init__(self, model_path: str, crop_size: int, num_threads: int,
                 confidence_threshold: float, logger=None):
        self.crop_size = crop_size
        self.confidence_threshold = confidence_threshold
        self.logger = logger
        self._classify_count = 0  # used to throttle crop saves

        self.interpreter = tflite.Interpreter(
            model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()
        self.out = self.interpreter.get_output_details()
        self.is_int8 = self.inp[0]["dtype"] in (np.int8, np.uint8)

        # ── Read quantization params for proper de-quantization ──────────────
        # Input quantization
        inp_quant = self.inp[0].get("quantization_parameters", {})
        self.inp_scale  = float(inp_quant.get("scales",       [1.0])[0]) if inp_quant else 1.0
        self.inp_zp     = int(  inp_quant.get("zero_points",  [0]  )[0]) if inp_quant else 0

        # Fallback: use legacy "quantization" tuple (scale, zero_point)
        if self.inp_scale == 1.0 and self.inp_zp == 0:
            legacy = self.inp[0].get("quantization", (1.0, 0))
            self.inp_scale, self.inp_zp = float(legacy[0]), int(legacy[1])

        # Output quantization
        out_quant = self.out[0].get("quantization_parameters", {})
        self.out_scale  = float(out_quant.get("scales",       [1.0])[0]) if out_quant else 1.0
        self.out_zp     = int(  out_quant.get("zero_points",  [0]  )[0]) if out_quant else 0

        if self.out_scale == 1.0 and self.out_zp == 0:
            legacy = self.out[0].get("quantization", (1.0, 0))
            self.out_scale, self.out_zp = float(legacy[0]), int(legacy[1])

        if logger:
            logger.info("=" * 60)
            logger.info(f"[CLASSIFIER] TFLite source  : {_TFLITE_SOURCE}")
            logger.info(f"[CLASSIFIER] Model path     : {model_path}")
            logger.info(f"[CLASSIFIER] Model size     : {os.path.getsize(model_path)//1024} KB")
            logger.info(f"[CLASSIFIER] Input  shape   : {self.inp[0]['shape']}")
            logger.info(f"[CLASSIFIER] Input  dtype   : {self.inp[0]['dtype']}")
            logger.info(f"[CLASSIFIER] Input  scale   : {self.inp_scale}  zero_pt: {self.inp_zp}")
            logger.info(f"[CLASSIFIER] Output shape   : {self.out[0]['shape']}")
            logger.info(f"[CLASSIFIER] Output dtype   : {self.out[0]['dtype']}")
            logger.info(f"[CLASSIFIER] Output scale   : {self.out_scale}  zero_pt: {self.out_zp}")
            logger.info(f"[CLASSIFIER] INT8 mode      : {self.is_int8}")
            logger.info(f"[CLASSIFIER] Confidence thr : {confidence_threshold}")
            logger.info(f"[CLASSIFIER] Crop size      : {crop_size}×{crop_size}")
            logger.info(f"[CLASSIFIER] Num threads    : {num_threads}")
            logger.info(f"[CLASSIFIER] Save crops     : {_SAVE_CROPS}  (dir: {_CROP_DIR})")
            logger.info("=" * 60)

            # ── WARN if quantization params look suspicious ───────────────────
            if self.is_int8 and self.inp_scale == 1.0 and self.inp_zp == 0:
                logger.warn(
                    "[CLASSIFIER] ⚠  INT8 model but input scale=1.0 / zp=0 — "
                    "quantization params may not have been read correctly. "
                    "Check that your .tflite file was exported with full integer quantization."
                )
            if self.is_int8 and self.out_scale == 1.0 and self.out_zp == 0:
                logger.warn(
                    "[CLASSIFIER] ⚠  INT8 model but output scale=1.0 / zp=0 — "
                    "output de-quantization will be a no-op. Probabilities may be garbage."
                )

    def classify(self, crop, debug=False):
        """
        Returns (robot_label, confidence).

        When debug=True, logs the full probability vector so you can see
        what the model is actually predicting even when confidence is too low.
        """
        self._classify_count += 1

        # ── Pre-process ───────────────────────────────────────────────────────
        orig_h, orig_w = crop.shape[:2]
        img = cv2.resize(crop, (self.crop_size, self.crop_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.is_int8:
            # Correct INT8 quantization: float → quantized int8
            # Formula: q = round(float_val / scale) + zero_point
            # For a [0, 255] uint8 image normalised to [0, 1]:
            #   float_val = pixel / 255.0
            #   q = round((pixel/255.0) / inp_scale) + inp_zp
            if self.inp_scale > 0:
                inp = np.round(img.astype(np.float32) / 255.0 / self.inp_scale
                               + self.inp_zp).astype(np.int8)
            else:
                # Fallback to old hardcoded method if scale is zero (shouldn't happen)
                inp = (img.astype(np.float32) - 128).astype(np.int8)
                if debug and self.logger:
                    self.logger.warn(
                        "[CLASSIFIER] ⚠  inp_scale is 0 — using legacy (img-128) quantization!"
                    )
        else:
            inp = img.astype(np.float32) / 255.0

        # ── Save crop for visual inspection ───────────────────────────────────
        if _SAVE_CROPS and (self._classify_count % _SAVE_INTERVAL == 0):
            fname = os.path.join(_CROP_DIR, f"crop_{self._classify_count:06d}.jpg")
            cv2.imwrite(fname, crop)  # save the original (unreized) crop
            if self.logger:
                self.logger.info(
                    f"[CLASSIFIER] 📷 Saved crop {orig_w}×{orig_h} → {fname}"
                )

        # ── Run inference ─────────────────────────────────────────────────────
        self.interpreter.set_tensor(self.inp[0]["index"], np.expand_dims(inp, 0))
        self.interpreter.invoke()
        raw_output = self.interpreter.get_tensor(self.out[0]["index"])[0]

        if debug and self.logger:
            self.logger.debug(
                f"[CLASSIFIER] Raw tensor dtype={raw_output.dtype}  "
                f"values={raw_output.tolist()}"
            )

        output = raw_output.astype(np.float32)

        # ── De-quantize output ────────────────────────────────────────────────
        # Must de-quant for BOTH int8 AND uint8 output dtypes
        if raw_output.dtype in (np.int8, np.uint8):
            if self.out_scale != 1.0 or self.out_zp != 0:
                output = (output - self.out_zp) * self.out_scale
                if debug and self.logger:
                    self.logger.debug(
                        f"[CLASSIFIER] After de-quant (scale={self.out_scale}, "
                        f"zp={self.out_zp}): {output.tolist()}"
                    )
            else:
                if debug and self.logger:
                    self.logger.warn(
                        "[CLASSIFIER] ⚠  Output is int8/uint8 but scale=1.0 / zp=0 — "
                        "skipping de-quant.  If results look wrong, your model's "
                        "quantization metadata may be missing."
                    )

        # ── Softmax ───────────────────────────────────────────────────────────
        shifted = output - np.max(output)
        exp_out = np.exp(shifted)
        probs   = exp_out / exp_out.sum()

        class_id   = int(np.argmax(probs))
        confidence = float(probs[class_id])

        if debug and self.logger:
            prob_str = "  ".join(
                f"{self.CUSTOM_CLASSES[i]}={probs[i]:.3f}"
                for i in range(len(self.CUSTOM_CLASSES))
            )
            self.logger.debug(f"[CLASSIFIER] Softmax probs → {prob_str}")
            self.logger.debug(
                f"[CLASSIFIER] Top class: {self.CUSTOM_CLASSES[class_id]} "
                f"conf={confidence:.3f}  threshold={self.confidence_threshold}"
            )

        # ── Apply confidence threshold ────────────────────────────────────────
        if confidence >= self.confidence_threshold:
            robot_label = self.CUSTOM_CLASSES[class_id]
        else:
            robot_label = "NO_SIGN"
            # Always log when a crop is found but confidence is too low —
            # this is the most common silent failure mode on the Pi
            if self.logger:
                prob_str = "  ".join(
                    f"{self.CUSTOM_CLASSES[i]}={probs[i]:.3f}"
                    for i in range(len(self.CUSTOM_CLASSES))
                )
                self.logger.warn(
                    f"[CLASSIFIER] ROI found but confidence too low "
                    f"({confidence:.3f} < {self.confidence_threshold}) → NO_SIGN  |  {prob_str}"
                )

        return robot_label, confidence


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL SMOOTHER  (ported from sign_node.py)
# ─────────────────────────────────────────────────────────────────────────────
class TemporalSmoother:
    """Requires N consecutive identical detections before committing."""

    def __init__(self, confirm_frames=3, clear_frames=6):
        self.confirm_frames  = confirm_frames
        self.clear_frames    = clear_frames
        self.candidate       = "NO_SIGN"
        self.candidate_count = 0
        self.stable          = "NO_SIGN"

    def update(self, detection):
        if detection == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate       = detection
            self.candidate_count = 1

        threshold = (self.clear_frames if detection == "NO_SIGN"
                     else self.confirm_frames)

        if self.candidate_count >= threshold:
            self.stable = self.candidate

        return self.stable


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 NODE
# ─────────────────────────────────────────────────────────────────────────────
class SignDetectionNode(Node):

    def __init__(self):
        super().__init__('sign_detection_node')

        # ── Declare ALL parameters ────────────────────────────────────────────
        # Model
        self.declare_parameter('model_path', '')
        self.declare_parameter('crop_size', 64)
        self.declare_parameter('num_threads', 2)
        self.declare_parameter('confidence_threshold', 0.75)

        # Camera
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('frame_width', 320)
        self.declare_parameter('frame_height', 240)
        self.declare_parameter('flip_camera', True)
        self.declare_parameter('publish_rate', 20.0)

        # Display
        self.declare_parameter('show_gui', False)

        # Temporal smoother
        self.declare_parameter('confirm_frames', 3)
        self.declare_parameter('clear_frames', 6)

        # Topics
        self.declare_parameter('cmd_output_topic', '/teleop/raw_cmd')
        self.declare_parameter('sign_topic', '/perception/sign_detected')

        # Velocity / heading mappings for sign → VehicleCmd
        self.declare_parameter('cruise_velocity', 0.5)
        self.declare_parameter('slow_velocity', 0.3)
        self.declare_parameter('stop_velocity', 0.0)
        self.declare_parameter('turn_velocity', 0.35)
        self.declare_parameter('turn_heading', 20.0)
        self.declare_parameter('stop_duration', 3.0)

        # ROI filter HSV ranges (tuned values from test_gtsrb_live.py)
        self.declare_parameter('red_range1_low',  [0, 150, 150])
        self.declare_parameter('red_range1_high', [8, 255, 255])
        self.declare_parameter('red_range2_low',  [172, 150, 150])
        self.declare_parameter('red_range2_high', [179, 255, 255])
        self.declare_parameter('yellow_range_low',  [20, 150, 150])
        self.declare_parameter('yellow_range_high', [35, 255, 255])
        self.declare_parameter('blue_range_low',  [105, 150, 120])
        self.declare_parameter('blue_range_high', [125, 255, 255])

        # ROI filter geometry
        self.declare_parameter('min_area', 1500)
        self.declare_parameter('max_area', 150000)
        self.declare_parameter('ignore_bottom_percent', 0.40)
        self.declare_parameter('min_aspect_ratio', 0.65)
        self.declare_parameter('max_aspect_ratio', 1.45)
        self.declare_parameter('min_solidity', 0.73)

        # ── Read parameters ───────────────────────────────────────────────────
        model_path_param = self.get_parameter('model_path').value
        crop_size        = self.get_parameter('crop_size').value
        num_threads      = self.get_parameter('num_threads').value
        conf_thresh      = self.get_parameter('confidence_threshold').value
        camera_index     = self.get_parameter('camera_index').value
        self.frame_w     = self.get_parameter('frame_width').value
        self.frame_h     = self.get_parameter('frame_height').value
        self.flip_camera = self.get_parameter('flip_camera').value
        publish_rate     = self.get_parameter('publish_rate').value
        self.show_gui    = self.get_parameter('show_gui').value
        confirm_frames   = self.get_parameter('confirm_frames').value
        clear_frames     = self.get_parameter('clear_frames').value
        cmd_topic        = self.get_parameter('cmd_output_topic').value
        sign_topic       = self.get_parameter('sign_topic').value

        self.cruise_vel  = self.get_parameter('cruise_velocity').value
        self.slow_vel    = self.get_parameter('slow_velocity').value
        self.stop_vel    = self.get_parameter('stop_velocity').value
        self.turn_vel    = self.get_parameter('turn_velocity').value
        self.turn_hdg    = self.get_parameter('turn_heading').value
        self.stop_dur    = self.get_parameter('stop_duration').value

        # ── Resolve model path ────────────────────────────────────────────────
        if not model_path_param or not os.path.isfile(model_path_param):
            pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_path = os.path.join(pkg_dir, 'model', 'custom_sign_classifier_int8.tflite')
            self.get_logger().warn(
                f"[INIT] model_path param='{model_path_param}' not found — "
                f"trying default: {default_path}"
            )
            if os.path.isfile(default_path):
                model_path_param = default_path
            else:
                self.get_logger().fatal(
                    f"[INIT] Model not found at '{model_path_param}' or '{default_path}'. "
                    "Set the model_path parameter or place the .tflite file in "
                    "perception/model/"
                )
                raise FileNotFoundError("TFLite model not found")
        self.get_logger().info(f"[INIT] Using model: {model_path_param}")

        # ── Build ROI filter params dict ──────────────────────────────────────
        roi_params = {
            'red_range1_low':  self._get_int_list('red_range1_low'),
            'red_range1_high': self._get_int_list('red_range1_high'),
            'red_range2_low':  self._get_int_list('red_range2_low'),
            'red_range2_high': self._get_int_list('red_range2_high'),
            'yellow_range_low':  self._get_int_list('yellow_range_low'),
            'yellow_range_high': self._get_int_list('yellow_range_high'),
            'blue_range_low':  self._get_int_list('blue_range_low'),
            'blue_range_high': self._get_int_list('blue_range_high'),
            'min_area':   self.get_parameter('min_area').value,
            'max_area':   self.get_parameter('max_area').value,
            'ignore_bottom_percent': self.get_parameter('ignore_bottom_percent').value,
            'min_aspect_ratio': self.get_parameter('min_aspect_ratio').value,
            'max_aspect_ratio': self.get_parameter('max_aspect_ratio').value,
            'min_solidity': self.get_parameter('min_solidity').value,
        }

        # ── Instantiate pipeline stages ───────────────────────────────────────
        self.roi_filter = ColorROIFilter(roi_params, logger=self.get_logger())
        self.classifier = GTSRBClassifier(
            model_path_param, crop_size, num_threads, conf_thresh,
            logger=self.get_logger())
        self.smoother = TemporalSmoother(confirm_frames, clear_frames)

        # ── Camera ────────────────────────────────────────────────────────────
        self.get_logger().info(f"[INIT] Opening camera index={camera_index} via V4L2")
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.frame_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)
        self.cap.set(cv2.CAP_PROP_FPS,          publish_rate)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        # Confirm what the camera actually accepted
        actual_w   = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h   = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(
            f"[INIT] Camera opened — requested {self.frame_w}×{self.frame_h}@{publish_rate}fps  "
            f"actual {actual_w:.0f}×{actual_h:.0f}@{actual_fps:.1f}fps"
        )

        if not self.cap.isOpened():
            self.get_logger().error("[INIT] Camera not available!")
            raise RuntimeError("Camera not available")

        # ── Warm-up read — catch camera issues early ──────────────────────────
        for _ in range(5):
            ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            self.get_logger().error(
                "[INIT] Camera opened but initial frame read FAILED. "
                "Check /dev/video* permissions and V4L2 device index."
            )
        else:
            self.get_logger().info(
                f"[INIT] Camera warm-up OK — frame shape: {test_frame.shape}"
            )

        # ── Publishers ────────────────────────────────────────────────────────
        self.cmd_pub  = self.create_publisher(VehicleCmd, cmd_topic, 10)
        self.sign_pub = self.create_publisher(String, sign_topic, 10)

        # ── State ─────────────────────────────────────────────────────────────
        self.frame_count    = 0
        self.roi_found_count = 0   # how many frames had a valid ROI
        self.classified_count = 0  # how many frames ran the classifier
        self.cached_label   = "NO_SIGN"
        self.cached_conf    = 0.0
        self.cached_box     = None
        self.cached_masks   = (None, None, None)
        self.stop_start_time = None
        self.fps            = 0.0
        self.fps_timer      = time.time()

        # ── Timer loop ────────────────────────────────────────────────────────
        self.timer = self.create_timer(1.0 / publish_rate, self._loop)

        self.get_logger().info('=' * 60)
        self.get_logger().info('Sign Detection Node (GTSRB TFLite) READY')
        self.get_logger().info(f'  Cmd topic  : {cmd_topic}')
        self.get_logger().info(f'  Sign topic : {sign_topic}')
        self.get_logger().info(f'  FPS        : {publish_rate}')
        self.get_logger().info(f'  Show GUI   : {self.show_gui}')
        self.get_logger().info(f'  Cruise vel : {self.cruise_vel} m/s')
        self.get_logger().info(f'  Slow vel   : {self.slow_vel} m/s')
        self.get_logger().info(f'  Turn hdg   : ±{self.turn_hdg}°')
        self.get_logger().info(f'  SIGN_DEBUG : {_DEBUG}  (set env var SIGN_DEBUG=1 to enable)')
        self.get_logger().info(f'  SAVE_CROPS : {_SAVE_CROPS}  (set env var SIGN_SAVE_CROPS=1 to enable)')
        self.get_logger().info('=' * 60)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _get_int_list(self, name):
        return [int(v) for v in self.get_parameter(name).value]

    # ── main loop ─────────────────────────────────────────────────────────────
    def _loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("[LOOP] cap.read() returned False — skipping frame")
            return

        if self.flip_camera:
            frame = cv2.flip(frame, -1)

        self.frame_count += 1

        # FPS counter (every 30 frames)
        if self.frame_count % 30 == 0:
            self.fps = 30.0 / max(time.time() - self.fps_timer, 1e-6)
            self.fps_timer = time.time()

        # ── Classify every 2nd frame, cache in between ────────────────────────
        if self.frame_count % 2 == 0:
            self.classified_count += 1
            crop, box, m_red, m_yel, m_blu = self.roi_filter.find_roi(
                frame, debug=_DEBUG)

            if crop is not None and crop.size > 0:
                self.roi_found_count += 1
                crop_h, crop_w = crop.shape[:2]

                if _DEBUG:
                    self.get_logger().debug(
                        f"[LOOP] ROI found — box={box}  crop={crop_w}×{crop_h}"
                    )

                label, conf = self.classifier.classify(crop, debug=_DEBUG)

            else:
                label, conf = "NO_SIGN", 0.0

                if _DEBUG:
                    self.get_logger().debug("[LOOP] No ROI found this frame")

            self.cached_label = label
            self.cached_conf  = conf
            self.cached_box   = box
            self.cached_masks = (m_red, m_yel, m_blu)
        else:
            label = self.cached_label
            conf  = self.cached_conf
            box   = self.cached_box

        # ── Temporal smoothing ────────────────────────────────────────────────
        stable = self.smoother.update(label)

        # ── Publish sign label ────────────────────────────────────────────────
        sign_msg = String()
        sign_msg.data = stable
        self.sign_pub.publish(sign_msg)

        # ── Map sign → VehicleCmd and publish ─────────────────────────────────
        cmd_msg = VehicleCmd()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.header.frame_id = 'base_link'

        if stable == "STOP":
            if self.stop_start_time is None:
                self.stop_start_time = time.time()
                self.get_logger().info("[CMD] STOP sign detected — stopping")
            cmd_msg.velocity = self.stop_vel
            cmd_msg.heading  = 0.0

            elapsed = time.time() - self.stop_start_time
            if elapsed >= self.stop_dur:
                self.get_logger().info(
                    f"[CMD] STOP duration ({self.stop_dur}s) elapsed — resuming cruise"
                )
                cmd_msg.velocity = self.cruise_vel
                cmd_msg.heading  = 0.0
                self.smoother.stable = "NO_SIGN"
                self.stop_start_time = None

        elif stable == "SLOW_DOWN":
            self.stop_start_time = None
            cmd_msg.velocity = self.slow_vel
            cmd_msg.heading  = 0.0

        elif stable == "TURN_LEFT":
            self.stop_start_time = None
            cmd_msg.velocity = self.turn_vel
            cmd_msg.heading  = -self.turn_hdg

        elif stable == "TURN_RIGHT":
            self.stop_start_time = None
            cmd_msg.velocity = self.turn_vel
            cmd_msg.heading  = self.turn_hdg

        else:  # NO_SIGN → cruise
            self.stop_start_time = None
            cmd_msg.velocity = self.cruise_vel
            cmd_msg.heading  = 0.0

        self.cmd_pub.publish(cmd_msg)

        # ── Optional GUI ──────────────────────────────────────────────────────
        if self.show_gui:
            display = frame.copy()
            if box:
                x, y, bw, bh = box
                clr = (0, 255, 0) if stable != "NO_SIGN" else (100, 100, 100)
                cv2.rectangle(display, (x, y), (x + bw, y + bh), clr, 2)

            cv2.putText(display, f"{stable} ({conf:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if stable != "NO_SIGN" else (100, 100, 100), 2)
            cv2.putText(display, f"FPS: {self.fps:.1f}", (self.frame_w - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display, f"v={cmd_msg.velocity:.2f} h={cmd_msg.heading:.1f}",
                        (10, self.frame_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            cv2.imshow("Sign Detection", display)

            m_red, m_yel, m_blu = self.cached_masks
            if m_red is not None:
                cv2.imshow("Mask: RED", m_red)
            if m_yel is not None:
                cv2.imshow("Mask: YELLOW", m_yel)
            if m_blu is not None:
                cv2.imshow("Mask: BLUE", m_blu)

            cv2.waitKey(1)

        # ── Periodic log (every 60 frames = ~3s at 20fps) ─────────────────────
        if self.frame_count % 60 == 0:
            roi_rate = (self.roi_found_count / max(self.classified_count, 1)) * 100
            self.get_logger().info(
                f"[STATUS] frame={self.frame_count}  fps={self.fps:.1f}  "
                f"roi_rate={roi_rate:.0f}%  "
                f"raw={label}({conf:.2f})  stable={stable}  "
                f"cmd=(v={cmd_msg.velocity:.2f}, h={cmd_msg.heading:.1f})"
            )
            # Reset rolling counters
            self.roi_found_count  = 0
            self.classified_count = 0

    # ── cleanup ───────────────────────────────────────────────────────────────
    def destroy_node(self):
        self.cap.release()
        if self.show_gui:
            cv2.destroyAllWindows()
        super().destroy_node()


# ─────────────────────────────────────────────────────────────────────────────
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