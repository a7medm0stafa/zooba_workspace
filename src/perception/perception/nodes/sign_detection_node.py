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
except ImportError:
    import tensorflow.lite as tflite


# ─────────────────────────────────────────────────────────────────────────────
# COLOR ROI FILTER  (ported from test_gtsrb_live.py)
# ─────────────────────────────────────────────────────────────────────────────
class ColorROIFilter:
    """Finds the most prominent sign-coloured region in the frame."""

    def __init__(self, params: dict):
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

    def find_roi(self, frame):
        """Returns (crop, bounding_box, red_mask, yellow_mask, blue_mask)."""
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # CLAHE on V channel — normalises brightness
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
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

        best_area, best_crop, best_box = 0, None, None

        for mask in [red_mask, yellow_mask, blue_mask]:
            k = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                area = cv2.contourArea(c)
                if not (self.MIN_AREA < area < self.MAX_AREA):
                    continue

                x, y, w, bh = cv2.boundingRect(c)
                if not (self.MIN_ASPECT < w / float(bh) < self.MAX_ASPECT):
                    continue

                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    continue
                if area / hull_area < self.MIN_SOLIDITY:
                    continue

                # Polygon approximation — sign must be a geometric shape
                epsilon = 0.04 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                if not (3 <= len(approx) <= 12):
                    continue

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

        self.interpreter = tflite.Interpreter(
            model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()
        self.out = self.interpreter.get_output_details()
        self.is_int8 = self.inp[0]["dtype"] in (np.int8, np.uint8)

        if logger:
            logger.info(f"Model loaded: {model_path}")
            logger.info(f"Input shape : {self.inp[0]['shape']}")
            logger.info(f"Input dtype : {self.inp[0]['dtype']}")
            logger.info(f"INT8 mode   : {self.is_int8}")
            logger.info(f"Num outputs : {self.out[0]['shape']}")

    def classify(self, crop):
        """Returns (robot_label, confidence)."""
        img = cv2.resize(crop, (self.crop_size, self.crop_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.is_int8:
            inp = (img.astype(np.float32) - 128).astype(np.int8)
        else:
            inp = img.astype(np.float32) / 255.0

        self.interpreter.set_tensor(self.inp[0]["index"], np.expand_dims(inp, 0))
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.out[0]["index"])[0].astype(np.float32)

        # De-quantize output if needed
        if self.out[0]["dtype"] == np.int8:
            scale, zp = self.out[0]["quantization"]
            output = (output - zp) * scale

        # Softmax
        output = np.exp(output - np.max(output))
        output /= output.sum()

        class_id   = int(np.argmax(output))
        confidence = float(output[class_id])

        # 5-class custom model
        if confidence >= self.confidence_threshold:
            robot_label = self.CUSTOM_CLASSES[class_id]
        else:
            robot_label = "NO_SIGN"

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
            # Default: look inside this package's model/ folder
            pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_path = os.path.join(pkg_dir, 'model', 'custom_sign_classifier_int8.tflite')
            if os.path.isfile(default_path):
                model_path_param = default_path
            else:
                self.get_logger().fatal(
                    f"Model not found at '{model_path_param}' or '{default_path}'. "
                    "Set the model_path parameter or place the .tflite file in "
                    "perception/model/")
                raise FileNotFoundError(f"TFLite model not found")
        self.get_logger().info(f"Using model: {model_path_param}")

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
        self.roi_filter = ColorROIFilter(roi_params)
        self.classifier = GTSRBClassifier(
            model_path_param, crop_size, num_threads, conf_thresh,
            logger=self.get_logger())
        self.smoother = TemporalSmoother(confirm_frames, clear_frames)

        # ── Camera ────────────────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.frame_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)
        self.cap.set(cv2.CAP_PROP_FPS,          publish_rate)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        if not self.cap.isOpened():
            self.get_logger().error("Camera not available!")
            raise RuntimeError("Camera not available")

        # ── Publishers ────────────────────────────────────────────────────────
        self.cmd_pub  = self.create_publisher(VehicleCmd, cmd_topic, 10)
        self.sign_pub = self.create_publisher(String, sign_topic, 10)

        # ── State ─────────────────────────────────────────────────────────────
        self.frame_count   = 0
        self.cached_label  = "NO_SIGN"
        self.cached_conf   = 0.0
        self.cached_box    = None
        self.cached_masks  = (None, None, None)
        self.stop_start_time = None   # Track how long we've been stopped
        self.fps           = 0.0
        self.fps_timer     = time.time()

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
        self.get_logger().info('=' * 60)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _get_int_list(self, name):
        """Read a parameter that was declared as a list of ints."""
        return [int(v) for v in self.get_parameter(name).value]

    # ── main loop ─────────────────────────────────────────────────────────────
    def _loop(self):
        ret, frame = self.cap.read()
        if not ret:
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
            crop, box, m_red, m_yel, m_blu = self.roi_filter.find_roi(frame)
            if crop is not None and crop.size > 0:
                label, conf = self.classifier.classify(crop)
            else:
                label, conf = "NO_SIGN", 0.0
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
            # On first STOP detection, record the time
            if self.stop_start_time is None:
                self.stop_start_time = time.time()
                self.get_logger().info("STOP sign detected — stopping")
            cmd_msg.velocity = self.stop_vel
            cmd_msg.heading  = 0.0

            # After stop_duration seconds, resume cruising
            elapsed = time.time() - self.stop_start_time
            if elapsed >= self.stop_dur:
                self.get_logger().info(f"STOP duration ({self.stop_dur}s) elapsed — resuming cruise")
                cmd_msg.velocity = self.cruise_vel
                cmd_msg.heading  = 0.0
                # Reset smoother so it doesn't keep issuing STOP
                self.smoother.stable = "NO_SIGN"
                self.stop_start_time = None

        elif stable == "SLOW_DOWN":
            self.stop_start_time = None
            cmd_msg.velocity = self.slow_vel
            cmd_msg.heading  = 0.0

        elif stable == "TURN_LEFT":
            self.stop_start_time = None
            cmd_msg.velocity = self.turn_vel
            cmd_msg.heading  = -self.turn_hdg   # negative = left

        elif stable == "TURN_RIGHT":
            self.stop_start_time = None
            cmd_msg.velocity = self.turn_vel
            cmd_msg.heading  = self.turn_hdg    # positive = right

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

            # Show color masks (same as test_gtsrb_live.py)
            m_red, m_yel, m_blu = self.cached_masks
            if m_red is not None:
                cv2.imshow("Mask: RED", m_red)
            if m_yel is not None:
                cv2.imshow("Mask: YELLOW", m_yel)
            if m_blu is not None:
                cv2.imshow("Mask: BLUE", m_blu)

            cv2.waitKey(1)

        # ── Periodic log ──────────────────────────────────────────────────────
        if self.frame_count % 60 == 0:
            self.get_logger().info(
                f"[FPS={self.fps:.1f}] sign={stable}  conf={conf:.2f}  "
                f"cmd=(v={cmd_msg.velocity:.2f}, h={cmd_msg.heading:.1f})")

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
