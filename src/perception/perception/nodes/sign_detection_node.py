"""
Sign Detection ROS2 Node
========================
Detects action signs (STOP, SLOW_DOWN, TURN_LEFT, TURN_RIGHT) using
the laptop/Pi camera and publishes VehicleCmd to /vehicle/cmd.

This iteration uses a highly optimized 6-step geometric isolation algorithm.

Topics Published:
    /vehicle/cmd              (vehicle_interfaces/VehicleCmd)  — velocity + heading
    /perception/sign_detected (std_msgs/String)                — current sign label
"""

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleCmd
from std_msgs.msg import String

import cv2
import numpy as np
import time

FRAME_W, FRAME_H = 640, 480

# ═══════════════════════════════════════════════════════════
#  Sign Detector Core Engine
# ═══════════════════════════════════════════════════════════

class SignDetector:
    def __init__(self, p):
        # p is a dictionary of parameters grabbed from the ROS config node
        self.min_area = p['min_area']
        self.max_area = p['max_area']
        self.epsilon_factor = 0.02
        self.min_purity = p['min_purity']
        self.crop_top_percentage = p['crop_top_percentage']

        self.stop_min_circularity  = p['stop_min_circularity']
        self.slow_min_circularity  = p['slow_min_circularity']
        self.turn_min_circularity  = p['turn_min_circularity']

        self.RED_RANGES = [
            (np.array(p['red_range1_low']), np.array(p['red_range1_high'])),
            (np.array(p['red_range2_low']), np.array(p['red_range2_high'])),
        ]
        self.YELLOW_RANGE = (np.array(p['yellow_range_low']), np.array(p['yellow_range_high']))
        self.BLUE_RANGE   = (np.array(p['blue_range_low']),   np.array(p['blue_range_high']))

    def detect(self, original_frame, show_dbg=False):
        import time
        timings = {}
        t_start = time.perf_counter()

        img = cv2.resize(original_frame, (FRAME_W, FRAME_H))
        y_end = int(FRAME_H * self.crop_top_percentage)
        img_bgr = img[0:y_end, :]

        t_resize = time.perf_counter()
        timings['Resize/Crop'] = t_resize - t_start

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        t_hsv = time.perf_counter()
        timings['HSV'] = t_hsv - t_resize

        detections = []
        if show_dbg:
            debug_shapes = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            debug_shapes = cv2.cvtColor(debug_shapes, cv2.COLOR_GRAY2BGR)
            debug_edges = np.zeros_like(img_bgr)
        else:
            debug_shapes = None
            debug_edges = None

        # Red Pipeline
        mask_r = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self.RED_RANGES:
            mask_r = cv2.bitwise_or(mask_r, cv2.inRange(hsv, lo, hi))
        clean_r = self._clean(mask_r)
        det_r = self._process_pipeline('STOP', clean_r, img_bgr, 6, 14, self.stop_min_circularity, debug_shapes, debug_edges, (0, 0, 255))
        detections.extend(det_r)

        t_red = time.perf_counter()
        timings['Red'] = t_red - t_hsv

        # Yellow Pipeline
        mask_y = cv2.inRange(hsv, *self.YELLOW_RANGE)
        clean_y = self._clean(mask_y)
        det_y = self._process_pipeline('SLOW_DOWN', clean_y, img_bgr, 3, 8, self.slow_min_circularity, debug_shapes, debug_edges, (0, 255, 255))
        detections.extend(det_y)

        t_yellow = time.perf_counter()
        timings['Yellow'] = t_yellow - t_red

        # Blue Pipeline
        mask_b = cv2.inRange(hsv, *self.BLUE_RANGE)
        clean_b = self._clean(mask_b)
        det_b = self._process_blue_pipeline(clean_b, img_bgr, debug_shapes, debug_edges)
        detections.extend(det_b)

        t_blue = time.perf_counter()
        timings['Blue'] = t_blue - t_yellow

        clean_masks = (clean_r, clean_y, clean_b) if show_dbg else None

        best_cmd = 'NO_SIGN'
        best_conf = 0.0
        best_bbox = (0, 0, 0, 0)
        
        if detections:
            best_det = max(detections, key=lambda d: d[1])
            best_cmd, best_conf, best_bbox = best_det

        return best_cmd, best_conf, best_bbox, img_bgr, clean_masks, debug_shapes, debug_edges, timings

    def _clean(self, mask):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)

    def _process_pipeline(self, label, clean_mask, img_bgr, min_v, max_v, min_circ, debug_shapes, debug_edges, color):
        dets = []
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area: continue
            x, y, w, h = cv2.boundingRect(cnt)
            
            ar = w / float(h) if h > 0 else 0
            if not (0.5 < ar < 2.0): continue
                
            roi_mask = clean_mask[y:y+h, x:x+w]
            purity = cv2.countNonZero(roi_mask) / float(w * h)
            if purity < self.min_purity: continue

            if debug_shapes is not None:
                cv2.rectangle(debug_shapes, (x, y), (x+w, y+h), (50, 200, 50), 1)

            pad = 10
            y1, y2 = max(0, y-pad), min(img_bgr.shape[0], y+h+pad)
            x1, x2 = max(0, x-pad), min(img_bgr.shape[1], x+w+pad)
            roi_bgr = img_bgr[y1:y2, x1:x2]

            passed, circ, verts = self._roi_shape_detection(roi_bgr, min_v, max_v, min_circ, debug_shapes, debug_edges, x1, y1, color)
            if passed:
                conf = min(1.0, circ + 0.2 * (area / self.max_area))
                dets.append((label, round(conf, 2), (x, y, w, h)))
                
        return dets

    def _process_blue_pipeline(self, clean_mask, img_bgr, debug_shapes, debug_edges):
        dets = []
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area: continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            if not (0.6 < w / float(h) < 1.6): continue
            
            roi_mask = clean_mask[y:y+h, x:x+w]
            purity = cv2.countNonZero(roi_mask) / float(w * h)
            if purity < self.min_purity: continue

            if debug_shapes is not None:
                cv2.rectangle(debug_shapes, (x, y), (x+w, y+h), (50, 200, 50), 1)

            pad = 10
            y1, y2 = max(0, y-pad), min(img_bgr.shape[0], y+h+pad)
            x1, x2 = max(0, x-pad), min(img_bgr.shape[1], x+w+pad)
            roi_bgr = img_bgr[y1:y2, x1:x2]
            
            passed, circ, verts = self._roi_shape_detection(roi_bgr, 6, 20, self.turn_min_circularity, debug_shapes, debug_edges, x1, y1, (255, 100, 0))
            if passed:
                direction = self._arrow_direction(roi_bgr)
                if direction:
                    conf = min(1.0, circ + 0.1)
                    dets.append((f'TURN_{direction}', round(conf, 2), (x, y, w, h)))
        return dets

    def _roi_shape_detection(self, roi_bgr, min_v, max_v, min_circ, debug_shapes, debug_edges, offset_x, offset_y, color):
        if roi_bgr.shape[0] < 10 or roi_bgr.shape[1] < 10: 
            return False, 0.0, 0
            
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)
        
        edge_map = cv2.Canny(roi_gray, 50, 150)
        edge_map = cv2.dilate(edge_map, None, iterations=1)
        
        contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if debug_edges is not None:
            edge_bgr = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
            eh, ew = edge_bgr.shape[:2]
            if offset_y+eh <= debug_edges.shape[0] and offset_x+ew <= debug_edges.shape[1]:
                debug_edges[offset_y:offset_y+eh, offset_x:offset_x+ew] = cv2.bitwise_or(
                    debug_edges[offset_y:offset_y+eh, offset_x:offset_x+ew], 
                    edge_bgr
                )
        
        if not contours: 
            return False, 0.0, 0

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if peri == 0: return False, 0.0, 0
            
        circ = 4 * np.pi * area / (peri * peri)
        approx = cv2.approxPolyDP(cnt, self.epsilon_factor * peri, True)
        verts = len(approx)

        passed = (min_v <= verts <= max_v) and (circ >= min_circ)

        if debug_shapes is not None:
            approx_global = approx + np.array([offset_x, offset_y])
            cnt_global = cnt + np.array([offset_x, offset_y])
            cv2.drawContours(debug_shapes, [cnt_global], -1, (80, 80, 80), 1)
            
            if passed:
                cv2.drawContours(debug_shapes, [approx_global], -1, color, 2)
                for pt in approx_global.squeeze():
                    cv2.circle(debug_shapes, tuple(pt), 4, (255,255,255), -1)
                
            text_color = color if passed else (100, 100, 100)
            cv2.putText(debug_shapes, f'v={verts} c={circ:.2f}', (offset_x, offset_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

        return passed, circ, verts

    def _arrow_direction(self, roi):
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        b_mask = cv2.inRange(hsv_roi, *self.BLUE_RANGE)
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        am = cv2.bitwise_and(bright, cv2.bitwise_not(b_mask))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        am = cv2.morphologyEx(cv2.morphologyEx(am, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
        
        nw, total = cv2.countNonZero(am), am.shape[0]*am.shape[1]
        if nw < total * 0.02 or nw > total * 0.80: return None
        
        contours, _ = cv2.findContours(am, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        
        hull = cv2.convexHull(max(contours, key=cv2.contourArea)).squeeze()
        if hull.ndim != 2 or len(hull) < 5: return self._mass_dir(am)
        
        n = len(hull)
        li, ri = int(np.argmin(hull[:, 0])), int(np.argmax(hull[:, 0]))
        la, ra = self._angle(hull, li, n), self._angle(hull, ri, n)
        if abs(ra - la) > 10: return 'LEFT' if la < ra else 'RIGHT'
        return self._mass_dir(am)

    @staticmethod
    def _angle(pts, idx, n):
        p, q, r = pts[(idx-1)%n].astype(float), pts[idx].astype(float), pts[(idx+1)%n].astype(float)
        v1, v2 = p - q, r - q
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6), -1, 1)))

    @staticmethod
    def _mass_dir(m):
        h, w = m.shape
        lm, rm = float(np.sum(m[:, :w//2])), float(np.sum(m[:, w//2:]))
        if lm+rm == 0: return None
        r = (lm-rm)/(lm+rm)
        return 'RIGHT' if r > 0.02 else ('LEFT' if r < -0.02 else None)


# ═══════════════════════════════════════════════════════════
#  ROS2 Node Definition
# ═══════════════════════════════════════════════════════════

class SignDetectionNode(Node):

    def __init__(self):
        super().__init__('sign_detection_node')

        # ── Parameters ──────────────────────────────────
        self.declare_parameter('cruise_velocity', 0.5)
        self.declare_parameter('slow_velocity', 0.3)
        self.declare_parameter('turn_heading', 20.0)
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('flip_horizontal', True)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('output_topic', '/vehicle/cmd')
        self.declare_parameter('sign_topic', '/perception/sign_detected')
        self.declare_parameter('show_gui', False)

        self.declare_parameter('crop_top_percentage', 0.60)
        self.declare_parameter('min_area', 1200)
        self.declare_parameter('max_area', 120000)
        self.declare_parameter('min_purity', 0.50)

        self.declare_parameter('stop_min_circularity', 0.75)
        self.declare_parameter('slow_min_circularity', 0.52)
        self.declare_parameter('turn_min_circularity', 0.75)

        self.declare_parameter('red_range1_low', [0, 120, 70])
        self.declare_parameter('red_range1_high', [10, 255, 255])
        self.declare_parameter('red_range2_low', [170, 120, 70])
        self.declare_parameter('red_range2_high', [179, 255, 255])
        self.declare_parameter('yellow_range_low', [18, 90, 90])
        self.declare_parameter('yellow_range_high', [38, 255, 255])
        self.declare_parameter('blue_range_low', [100, 130, 60])
        self.declare_parameter('blue_range_high', [130, 255, 255])

        # Extract values
        self.cruise_velocity = self.get_parameter('cruise_velocity').value
        self.slow_velocity = self.get_parameter('slow_velocity').value
        self.turn_heading = self.get_parameter('turn_heading').value
        camera_index = self.get_parameter('camera_index').value
        self.flip_horizontal = self.get_parameter('flip_horizontal').value
        publish_rate = self.get_parameter('publish_rate').value
        output_topic = self.get_parameter('output_topic').value
        sign_topic = self.get_parameter('sign_topic').value
        self.show_gui = self.get_parameter('show_gui').value

        params_dict = {
            'crop_top_percentage': self.get_parameter('crop_top_percentage').value,
            'min_area': self.get_parameter('min_area').value,
            'max_area': self.get_parameter('max_area').value,
            'min_purity': self.get_parameter('min_purity').value,
            'stop_min_circularity': self.get_parameter('stop_min_circularity').value,
            'slow_min_circularity': self.get_parameter('slow_min_circularity').value,
            'turn_min_circularity': self.get_parameter('turn_min_circularity').value,
            'red_range1_low': self.get_parameter('red_range1_low').value,
            'red_range1_high': self.get_parameter('red_range1_high').value,
            'red_range2_low': self.get_parameter('red_range2_low').value,
            'red_range2_high': self.get_parameter('red_range2_high').value,
            'yellow_range_low': self.get_parameter('yellow_range_low').value,
            'yellow_range_high': self.get_parameter('yellow_range_high').value,
            'blue_range_low': self.get_parameter('blue_range_low').value,
            'blue_range_high': self.get_parameter('blue_range_high').value,
        }

        # ── Sign → command mapping ──────────────────────
        self.sign_commands = {
            'STOP':       (0.0,                  0.0),
            'SLOW_DOWN':  (self.slow_velocity,   0.0),
            'TURN_LEFT':  (self.cruise_velocity, self.turn_heading),
            'TURN_RIGHT': (self.cruise_velocity, -self.turn_heading),
            'NO_SIGN':    (self.cruise_velocity,  0.0),
        }

        # ── Camera ──────────────────────────────────────
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.get_logger().error(f'Cannot open camera index {camera_index}')
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

        # ── Detector ────────────────────────────────────
        self.detector = SignDetector(params_dict)

        # ── Publishers ──────────────────────────────────
        self.cmd_pub = self.create_publisher(VehicleCmd, output_topic, 10)
        self.sign_pub = self.create_publisher(String, sign_topic, 10)

        # ── Timer ───────────────────────────────────────
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self._timer_callback)

        # ── State ───────────────────────────────────────
        self.prev_command = ''
        self.fps = 0.0

        self.get_logger().info('=' * 58)
        self.get_logger().info('Sign Detection Node Started')
        self.get_logger().info(f'  Publish rate   : {publish_rate} Hz')
        self.get_logger().info(f'  GUI windows    : {self.show_gui}')
        self.get_logger().info('=' * 58)

    def _timer_callback(self):
        t0 = time.time()

        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn('Camera read failed', throttle_duration_sec=5.0)
            return

        if self.flip_horizontal:
            # -1 flips both horizontally and vertically (180 degree rotation)
            frame = cv2.flip(frame, -1)

        # ── Detection ──
        cmd, conf, bbox, processed, dbg_mask, dbg_shapes, dbg_edges, timings = self.detector.detect(frame, self.show_gui)

        dt = time.time() - t0
        self.fps = 0.9 * self.fps + 0.1 / max(dt, 1e-4)

        # ── Log timings ──
        timing_str = " | ".join([f"{k}:{v*1000:.1f}ms" for k, v in timings.items()])
        self.get_logger().info(f"[Time] {timing_str} | Total:{dt*1000:.1f}ms", throttle_duration_sec=0.5)

        # ── Publish VehicleCmd ──
        vel, hdg = self.sign_commands.get(cmd, (self.cruise_velocity, 0.0))

        cmd_msg = VehicleCmd()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.header.frame_id = 'base_link'
        cmd_msg.velocity = float(vel)
        cmd_msg.heading = float(hdg)
        self.cmd_pub.publish(cmd_msg)

        # ── Publish sign label ──
        sign_msg = String()
        sign_msg.data = cmd
        self.sign_pub.publish(sign_msg)

        # ── Log command changes ──
        if cmd != self.prev_command:
            if cmd != 'NO_SIGN':
                self.get_logger().info(f'SIGN: {cmd} ({conf:.0%}) → vel={vel:.2f}, hdg={hdg:+.1f}° [{self.fps:.0f} FPS]')
            else:
                self.get_logger().info(f'SIGN: NO_SIGN → cruise vel={vel:.2f}, hdg={hdg:+.1f}°')
        self.prev_command = cmd

        # ── Desktop Debugging GUI ──
        if self.show_gui:
            self._show_debug(processed, cmd, conf, bbox, dbg_mask, dbg_shapes, dbg_edges)

    def _show_debug(self, processed, cmd, conf, bbox, dbg_mask, dbg_shapes, dbg_edges):
        COLORS = {
            'STOP':       (0,   0,   255),
            'SLOW_DOWN':  (0,   200, 255),
            'TURN_LEFT':  (255, 150, 0),
            'TURN_RIGHT': (255, 100, 0),
            'NO_SIGN':    (100, 100, 100),
        }

        display = processed.copy()
        if cmd != 'NO_SIGN':
            x, y, w, h = bbox
            col = COLORS.get(cmd, (255, 255, 255))
            cv2.rectangle(display, (x, y), (x+w, y+h), col, 2)
            lbl = f'{cmd} {conf:.0%}'
            cv2.putText(display, lbl, (x, max(15, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        fh = display.shape[0]
        cv2.rectangle(display, (0, fh-28), (FRAME_W, fh), (25, 25, 25), -1)
        cv2.putText(display, f'{self.fps:.0f} FPS | CMD: {cmd}', (8, fh-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS.get(cmd, (180, 180, 180)), 2)
        cv2.imshow('Camera Crop', display)

        if dbg_shapes is not None:
            cv2.imshow('ROI Shape Debug', dbg_shapes)
        if dbg_edges is not None:
            cv2.imshow('Canny Edges', dbg_edges)
        if dbg_mask is not None:
            r, y, b = dbg_mask
            row = np.hstack([cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) for m in (r, y, b)])
            for txt, xp, col in [('1. RED', 10, (0,0,255)), ('2. YELLOW', 650, (0,255,255)), ('3. BLUE', 1290, (255,100,0))]:
                cv2.putText(row, txt, (xp, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            cv2.imshow('Global Masks', row)
            
        cv2.waitKey(1)

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