"""
sign_detection_node.py  (YOLOv8 TFLite version)
================================================
ROS2 Traffic Sign Detection Node — uses a YOLOv8 INT8 TFLite model
(Custom classes: NO_SIGN, SLOW_DOWN, STOP, TURN_LEFT, TURN_RIGHT).

Pipeline:
  1. Capture frame from Pi camera
  2. Resize entire frame to YOLO input size (e.g., 320x320)
  3. TFLite Object Detection → Bounding boxes + Confidences
  4. Non-Maximum Suppression (NMS) → Best detection
  5. Temporal smoother → Stabilised label
  6. Map label → VehicleCmd and publish

Publishes:
  /teleop/raw_cmd           (vehicle_interfaces/VehicleCmd)
  /perception/sign_detected (std_msgs/String)
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

_DEBUG = os.environ.get("SIGN_DEBUG", "0") == "1"

# ─────────────────────────────────────────────────────────────────────────────
# YOLOv8 TFLITE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
class YOLOv8Detector:
    """Runs single-stage object detection on the full frame."""

    # Must match the order of classes in your Roboflow dataset/data.yaml
    CUSTOM_CLASSES = ["SLOW_DOWN", "STOP", "TURN_LEFT", "TURN_RIGHT"]

    def __init__(self, model_path: str, imgsz: int, num_threads: int,
                 conf_threshold: float, iou_threshold: float, logger=None):
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.logger = logger

        # Try to load with XNNPACK delegate for ARM speed boost
        try:
            self.interpreter = tflite.Interpreter(
                model_path=model_path, 
                num_threads=num_threads,
                experimental_delegates=[tflite.load_delegate('libXNNPACK.so')]
            )
            if logger: logger.info("[YOLO] Loaded with XNNPACK hardware acceleration!")
        except ValueError:
            self.interpreter = tflite.Interpreter(
                model_path=model_path, num_threads=num_threads)
            if logger: logger.warn("[YOLO] XNNPACK not found. Running standard CPU execution.")

        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]
        self.is_int8 = self.inp["dtype"] in (np.int8, np.uint8)

        # Quantization params
        quant = self.inp.get("quantization_parameters", {})
        self.inp_scale = float(quant.get("scales", [1.0])[0]) if quant and quant.get("scales") else 1.0
        self.inp_zp    = int(quant.get("zero_points", [0])[0]) if quant and quant.get("zero_points") else 0

        out_quant = self.out.get("quantization_parameters", {})
        self.out_scale = float(out_quant.get("scales", [1.0])[0]) if out_quant and out_quant.get("scales") else 1.0
        self.out_zp    = int(out_quant.get("zero_points", [0])[0]) if out_quant and out_quant.get("zero_points") else 0

        if logger:
            logger.info("=" * 60)
            logger.info(f"[YOLO] TFLite source  : {_TFLITE_SOURCE}")
            logger.info(f"[YOLO] Model path     : {model_path}")
            logger.info(f"[YOLO] Input  shape   : {self.inp['shape']} ({self.inp['dtype']})")
            logger.info(f"[YOLO] Output shape   : {self.out['shape']} ({self.out['dtype']})")
            logger.info(f"[YOLO] Conf threshold : {self.conf_threshold}")
            logger.info(f"[YOLO] IOU threshold  : {self.iou_threshold}")
            logger.info("=" * 60)

    def detect(self, frame_bgr):
        """Returns (best_label, confidence, bounding_box_xywh) relative to input frame size."""
        orig_h, orig_w = frame_bgr.shape[:2]
        
        # 1. Preprocess
        img = cv2.resize(frame_bgr, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.is_int8:
            # Handle standard float32 to int8 quantization
            inp = np.round((img.astype(np.float32) / 255.0) / self.inp_scale + self.inp_zp).astype(np.int8)
        else:
            inp = img.astype(np.float32) / 255.0

        # 2. Inference
        self.interpreter.set_tensor(self.inp["index"], np.expand_dims(inp, 0))
        self.interpreter.invoke()
        raw_output = self.interpreter.get_tensor(self.out["index"])[0]

        # 3. De-quantize if necessary
        if raw_output.dtype in (np.int8, np.uint8):
            output = (raw_output.astype(np.float32) - self.out_zp) * self.out_scale
        else:
            output = raw_output.astype(np.float32)

        # 4. Parse YOLOv8 Output Shape
        # YOLOv8 outputs [4 + num_classes, num_anchors] (e.g., [8, 2100])
        # We need to transpose it to [num_anchors, features]
        if output.shape[0] < output.shape[1]:
            output = output.T 

        boxes_cxcywh = output[:, :4]
        scores_matrix = output[:, 4:]

        # Get highest score for each bounding box
        class_ids = np.argmax(scores_matrix, axis=1)
        confidences = np.max(scores_matrix, axis=1)

        # 5. Filter by confidence threshold
        mask = confidences > self.conf_threshold
        filtered_boxes = boxes_cxcywh[mask]
        filtered_confs = confidences[mask]
        filtered_class_ids = class_ids[mask]

        if len(filtered_boxes) == 0:
            return "NO_SIGN", 0.0, None

        # 6. Convert Center-X, Center-Y, W, H to X_min, Y_min, W, H for OpenCV NMS
        boxes_xywh = np.zeros_like(filtered_boxes)
        boxes_xywh[:, 0] = filtered_boxes[:, 0] - (filtered_boxes[:, 2] / 2) # X_min
        boxes_xywh[:, 1] = filtered_boxes[:, 1] - (filtered_boxes[:, 3] / 2) # Y_min
        boxes_xywh[:, 2] = filtered_boxes[:, 2]                              # Width
        boxes_xywh[:, 3] = filtered_boxes[:, 3]                              # Height

        # 7. Non-Maximum Suppression (Removes overlapping boxes)
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(), 
            filtered_confs.tolist(), 
            self.conf_threshold, 
            self.iou_threshold
        )

        if len(indices) > 0:
            # Get the best detection
            best_idx = indices[0]
            best_class_id = filtered_class_ids[best_idx]
            best_conf = filtered_confs[best_idx]
            best_box = boxes_xywh[best_idx] # [x, y, w, h] in imgsz scale

            # 8. Scale bounding box back to original frame resolution
            scale_x = orig_w / self.imgsz
            scale_y = orig_h / self.imgsz
            
            final_box = (
                int(best_box[0] * scale_x),
                int(best_box[1] * scale_y),
                int(best_box[2] * scale_x),
                int(best_box[3] * scale_y)
            )

            # Safety check just in case model detects something outside CUSTOM_CLASSES bounds
            if best_class_id < len(self.CUSTOM_CLASSES):
                label = self.CUSTOM_CLASSES[best_class_id]
            else:
                label = "NO_SIGN"

            return label, float(best_conf), final_box

        return "NO_SIGN", 0.0, None


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL SMOOTHER
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

        # ── Declare Parameters ────────────────────────────────────────────
        # Model (Updated for YOLO)
        self.declare_parameter('model_path', '')
        self.declare_parameter('imgsz', 320)  # Replaces crop_size
        self.declare_parameter('num_threads', 4) # Increased to 4 for Pi
        self.declare_parameter('confidence_threshold', 0.60)
        self.declare_parameter('iou_threshold', 0.45) # New for YOLO NMS

        # Camera
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('frame_width', 320)
        self.declare_parameter('frame_height', 240)
        self.declare_parameter('flip_camera', True)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('show_gui', False)

        # Temporal smoother
        self.declare_parameter('confirm_frames', 3)
        self.declare_parameter('clear_frames', 6)

        # Topics
        self.declare_parameter('cmd_output_topic', '/teleop/raw_cmd')
        self.declare_parameter('sign_topic', '/perception/sign_detected')

        # Velocity / heading mappings
        self.declare_parameter('cruise_velocity', 0.5)
        self.declare_parameter('slow_velocity', 0.3)
        self.declare_parameter('stop_velocity', 0.0)
        self.declare_parameter('turn_velocity', 0.35)
        self.declare_parameter('turn_heading', 20.0)
        self.declare_parameter('stop_duration', 3.0)

        # ── Read parameters ───────────────────────────────────────────────────
        model_path_param = self.get_parameter('model_path').value
        imgsz            = self.get_parameter('imgsz').value
        num_threads      = self.get_parameter('num_threads').value
        conf_thresh      = self.get_parameter('confidence_threshold').value
        iou_thresh       = self.get_parameter('iou_threshold').value
        
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
            default_path = os.path.join(pkg_dir, 'model', 'best_int8.tflite')
            self.get_logger().warn(
                f"[INIT] model_path not found — trying default: {default_path}"
            )
            if os.path.isfile(default_path):
                model_path_param = default_path
            else:
                self.get_logger().fatal("YOLO TFLite model not found!")
                raise FileNotFoundError("YOLO TFLite model not found")

        # ── Instantiate pipeline stages ───────────────────────────────────────
        self.detector = YOLOv8Detector(
            model_path_param, imgsz, num_threads, conf_thresh, iou_thresh,
            logger=self.get_logger())
        self.smoother = TemporalSmoother(confirm_frames, clear_frames)

        # ── Camera ────────────────────────────────────────────────────────────
        self.get_logger().info(f"[INIT] Opening camera index={camera_index} via V4L2")
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.frame_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)
        self.cap.set(cv2.CAP_PROP_FPS,          publish_rate)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        if not self.cap.isOpened():
            self.get_logger().error("[INIT] Camera not available!")
            raise RuntimeError("Camera not available")

        # ── Publishers ────────────────────────────────────────────────────────
        self.cmd_pub  = self.create_publisher(VehicleCmd, cmd_topic, 10)
        self.sign_pub = self.create_publisher(String, sign_topic, 10)

        # ── State ─────────────────────────────────────────────────────────────
        self.frame_count    = 0
        self.stop_start_time = None
        self.fps            = 0.0
        self.fps_timer      = time.time()

        # ── Timer loop ────────────────────────────────────────────────────────
        self.timer = self.create_timer(1.0 / publish_rate, self._loop)
        self.get_logger().info('Sign Detection Node (YOLOv8 TFLite) READY')

    def _loop(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        if self.flip_camera:
            frame = cv2.flip(frame, -1)

        self.frame_count += 1

        # FPS counter
        if self.frame_count % 30 == 0:
            self.fps = 30.0 / max(time.time() - self.fps_timer, 1e-6)
            self.fps_timer = time.time()

        # ── 1. Detect Sign (Replaces ROI + Classification) ────────────────────
        label, conf, box = self.detector.detect(frame)

        # ── 2. Temporal smoothing ─────────────────────────────────────────────
        stable = self.smoother.update(label)

        # ── 3. Publish sign label ─────────────────────────────────────────────
        sign_msg = String()
        sign_msg.data = stable
        self.sign_pub.publish(sign_msg)

        # ── 4. Map sign → VehicleCmd and publish ──────────────────────────────
        cmd_msg = VehicleCmd()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.header.frame_id = 'base_link'

        if stable == "STOP":
            if self.stop_start_time is None:
                self.stop_start_time = time.time()
                self.get_logger().info("[CMD] STOP sign detected — stopping")
            cmd_msg.velocity = self.stop_vel
            cmd_msg.heading  = 0.0

            if (time.time() - self.stop_start_time) >= self.stop_dur:
                cmd_msg.velocity = self.cruise_vel
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

        # ── 5. Optional GUI ───────────────────────────────────────────────────
        if self.show_gui:
            display = frame.copy()
            if box:
                x, y, bw, bh = box
                clr = (0, 255, 0) if stable != "NO_SIGN" else (0, 0, 255)
                # Ensure box constraints
                x, y = max(0, x), max(0, y)
                cv2.rectangle(display, (x, y), (x + bw, y + bh), clr, 2)
                cv2.putText(display, f"{label} {conf:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)

            cv2.putText(display, f"Stable: {stable}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, f"FPS: {self.fps:.1f}", (self.frame_w - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("YOLO Sign Detection", display)
            cv2.waitKey(1)

        # ── Periodic log ──────────────────────────────────────────────────────
        if self.frame_count % 60 == 0:
            self.get_logger().info(
                f"[STATUS] fps={self.fps:.1f} | raw={label}({conf:.2f}) "
                f"stable={stable} | cmd=(v={cmd_msg.velocity:.2f}, h={cmd_msg.heading:.1f})"
            )

    def destroy_node(self):
        self.cap.release()
        if self.show_gui:
            cv2.destroyAllWindows()
        super().destroy_node()


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