"""
Traffic Light Detector Node
============================
A ROS2 node that detects traffic light states (RED, YELLOW, GREEN) from a
continuous camera stream using classical computer vision with OpenCV.

The node operates in two modes:
    - DETECTION: Full pipeline scans the entire ROI using HoughCircles,
      filtering, clustering, and colour classification.
    - TRACKING: Once a traffic light is found, subsequent frames only run
      circle detection + colour classification inside a small search window
      around the last known position (much cheaper).

Pipeline (DETECTION mode):
    1. Preprocessing  (BGR → grayscale / HSV, blur, ROI crop)
    2. Circle detection  (HoughCircles on full ROI)
    3. Circle filtering  (radius + weak brightness)
    4. Circle clustering  (proximity + radius similarity)
    5. Colour classification  (HSV masks per circle)
    6. Active light selection  (highest colour score → state)
    7. Temporal filtering  (rolling history for consistency)

Pipeline (TRACKING mode):
    1. Preprocessing
    2. Crop search window around tracked bbox
    3. Circle detection  (HoughCircles on small crop)
    4. Colour classification
    5. Active light selection
    6. Temporal filtering

Subscribes:
    /camera/image_raw  (sensor_msgs/Image)

Publishes:
    /traffic_light/state        (std_msgs/String)
    /traffic_light/debug_image  (sensor_msgs/Image)
"""

import csv
import os
import time
import yaml
from collections import deque

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory


# ---------------------------------------------------------------------------
# Processing modes
# ---------------------------------------------------------------------------
MODE_DETECTION = 'DETECTION'
MODE_TRACKING = 'TRACKING'


class TrafficLightDetectorNode(Node):
    """ROS2 node for real-time traffic light detection on a camera stream."""

    # ===================================================================
    # Initialisation
    # ===================================================================

    def __init__(self):
        super().__init__('traffic_light_detector_node')

        # -- Declare every configurable parameter ----------------------
        self._declare_parameters()

        # -- Subscribe to shared camera topic --------------------------
        camera_topic = self._p('camera_topic')
        self.bridge = CvBridge()
        self.latest_frame = None
        self.image_sub = self.create_subscription(
            Image, camera_topic, self._image_callback, 10
        )
        self.get_logger().info(f'Subscribing to camera on: {camera_topic}')
        
        # -- Publishers ------------------------------------------------
        self.state_pub = self.create_publisher(
            String, '/traffic_light/state', 10
        )

        # -- Processing timer (decoupled from camera FPS) --------------
        rate = self._p('processing_rate')
        self.process_timer = self.create_timer(1.0 / rate,
                                               self._process_timer_callback)

        # -- Stream state ----------------------------------------------
        self.latest_frame = None          # Most recent BGR frame

        # -- Temporal filtering ----------------------------------------
        self.state_history: deque = deque(maxlen=30)

        # -- Tracking state --------------------------------------------
        self.mode = MODE_DETECTION
        self.tracked_bbox = None          # (x1, y1, x2, y2) in ROI coords
        self.tracking_lost_count = 0      # Consecutive frames with no match

        # -- KPI logging -----------------------------------------------
        self._kpi_init()

        self.get_logger().info('Traffic light detector node started '
                               '(mode=DETECTION).')

    # ===================================================================
    # Parameter declaration
    # ===================================================================

    def _declare_parameters(self):
        """Declare all configurable ROS2 parameters.

        Values are loaded from config/traffic_light_detector.yaml
        (installed alongside the package).  If the YAML file is not
        found, hard-coded fallback defaults are used instead.

        A local override file (traffic_light_detector.local.yaml) is
        checked first — this file is gitignored so it can be edited
        freely on any machine without rebuild or git conflicts.
        """
        # Load YAML defaults
        yaml_params = self._load_yaml_params()

        def _d(name, fallback):
            """Declare a parameter using YAML value if present, else fallback."""
            self.declare_parameter(name, yaml_params.get(name, fallback))

        # Circle detection (HoughCircles)
        _d('dp', 1.2)
        _d('min_dist', 30.0)
        _d('param1', 100.0)
        _d('param2', 25.0)
        _d('min_radius', 5)
        _d('max_radius', 50)

        # ROI
        _d('roi_height_ratio', 0.6)

        # Filtering
        _d('min_brightness', 30)

        # Colour thresholds (HSV)
        _d('red_lower1', [0, 100, 80])
        _d('red_upper1', [10, 255, 255])
        _d('red_lower2', [160, 100, 80])
        _d('red_upper2', [179, 255, 255])
        _d('yellow_lower', [15, 100, 80])
        _d('yellow_upper', [35, 255, 255])
        _d('green_lower', [35, 80, 80])
        _d('green_upper', [90, 255, 255])

        # Clustering
        _d('max_circle_distance', 80.0)
        _d('radius_tolerance', 0.5)

        # Geometry validation
        _d('alignment_tolerance', 2.0)
        _d('min_cluster_size', 2)

        # Preprocessing
        _d('use_clahe', True)
        _d('clahe_clip_limit', 2.0)

        # Colour classification
        _d('min_color_ratio', 0.05)

        # Temporal
        _d('min_confirm_frames', 3)

        # Tracking
        _d('tracking_expansion', 1.5)
        _d('tracking_lost_frames', 5)

        # Stream
        _d('processing_rate', 20.0)
        _d('show_debug_display', True)
        _d('camera_topic', '/camera/image_raw')

        # KPI logging
        _d('enable_kpi_logging', True)
        _d('kpi_csv_path', os.path.expanduser('~/zooba_workspace/zooba_kpi/traffic_light_kpi.csv'))

    def _load_yaml_params(self) -> dict:
        """Load parameter values from a YAML config file.

        Resolution order:
            1. Local override:  <workspace>/src/perception/config/
               traffic_light_detector.local.yaml  (gitignored, no rebuild)
            2. Installed share: <install>/share/perception/config/
               traffic_light_detector.yaml  (requires rebuild)

        Returns:
            A flat dict of parameter_name → value, or empty dict on failure.
        """
        # --- 1. Try local override (no rebuild, gitignored) ---------------
        try:
            # Walk up from this source file to find the workspace config dir
            this_file = os.path.abspath(__file__)
            # .../src/perception/perception/nodes/traffic_light_detector_node.py
            pkg_src_dir = os.path.dirname(  # nodes/
                os.path.dirname(             # perception/
                    os.path.dirname(this_file)))  # perception/ (package root)
            local_yaml = os.path.join(
                pkg_src_dir, 'config', 'traffic_light_detector.local.yaml'
            )
            if os.path.isfile(local_yaml):
                with open(local_yaml, 'r') as f:
                    raw = yaml.safe_load(f)
                params = (
                    raw
                    .get('traffic_light_detector_node', {})
                    .get('ros__parameters', {})
                )
                self.get_logger().info(
                    f'Loaded LOCAL override config from {local_yaml}'
                )
                return params
        except Exception as e:
            self.get_logger().warn(
                f'Error reading local override config: {e}'
            )

        # --- 2. Fallback: installed share directory -----------------------
        try:
            share_dir = get_package_share_directory('perception')
            yaml_path = os.path.join(
                share_dir, 'config', 'traffic_light_detector.yaml'
            )
            with open(yaml_path, 'r') as f:
                raw = yaml.safe_load(f)

            # The YAML structure is:
            #   traffic_light_detector_node:
            #     ros__parameters:
            #       key: value
            params = (
                raw
                .get('traffic_light_detector_node', {})
                .get('ros__parameters', {})
            )
            self.get_logger().info(f'Loaded parameters from {yaml_path}')
            return params
        except Exception as e:
            self.get_logger().warn(
                f'Could not load YAML config, using defaults: {e}'
            )
            return {}

    # ===================================================================
    # Convenience: read a parameter value
    # ===================================================================

    def _p(self, name):
        """Shorthand to retrieve a parameter value."""
        return self.get_parameter(name).value

    # ===================================================================
    # Camera subscription callback
    # ===================================================================

    def _image_callback(self, msg: Image):
        """Store the latest camera frame from the shared camera topic."""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge error: {e}')

    # ===================================================================
    # KPI logging infrastructure
    # ===================================================================

    def _kpi_init(self):
        """Initialise KPI CSV logging.

        Opens the CSV in write mode so it clears on every launch.
        """
        self.kpi_total_frames = 0
        self.kpi_candidate_frames = 0
        self.kpi_confirmed_frames = 0
        self._kpi_file = None
        self._kpi_writer = None

        if not self._p('enable_kpi_logging'):
            return

        csv_path = os.path.expanduser(self._p('kpi_csv_path'))
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        self._kpi_file = open(csv_path, 'w', newline='')
        self._kpi_writer = csv.writer(self._kpi_file)
        self._kpi_writer.writerow([
            'timestamp', 'latency_ms', 'mode',
            'detection_latency_ms', 'tracking_latency_ms',
            'detected_state', 'confirmed_state',
            'num_circles', 'best_color_ratio',
            'candidates_found', 'detection_rate_pct',
        ])
        self._kpi_file.flush()
        self.get_logger().info(f'KPI CSV logging to {csv_path}')

    def _kpi_write_row(self, latency_ms, mode,
                       detection_latency_ms, tracking_latency_ms,
                       detected_state, confirmed_state,
                       num_circles, best_color_ratio):
        """Append one KPI row to the CSV."""
        if self._kpi_writer is None:
            return

        self.kpi_total_frames += 1

        # A "candidate frame" is one where circles passed the filters
        candidates_found = 1 if num_circles > 0 else 0
        if candidates_found:
            self.kpi_candidate_frames += 1

        # A "confirmed frame" is one where temporal filtering gave a real state
        if confirmed_state != 'UNKNOWN':
            self.kpi_confirmed_frames += 1

        detection_rate_pct = (
            (self.kpi_confirmed_frames / self.kpi_candidate_frames * 100.0)
            if self.kpi_candidate_frames > 0 else 0.0
        )

        self._kpi_writer.writerow([
            f'{time.time():.4f}',
            f'{latency_ms:.2f}',
            mode,
            f'{detection_latency_ms:.2f}',
            f'{tracking_latency_ms:.2f}',
            detected_state,
            confirmed_state,
            num_circles,
            f'{best_color_ratio:.4f}',
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

    # ===================================================================
    # Processing timer callback (runs the pipeline)
    # ===================================================================

    def _process_timer_callback(self):
        """
        Timer-driven processing loop.

        Uses the latest frame from the camera subscription, decides between
        DETECTION and TRACKING modes, runs the appropriate pipeline, and
        publishes results.
        """
        frame = self.latest_frame
        if frame is None:
            return

        t_start = time.time()

        # -- 1. Preprocessing ------------------------------------------
        gray, hsv, roi_frame, roi_offset = self.preprocess_image(frame)

        # -- Branch by mode (with per-mode latency) --------------------
        detection_latency_ms = 0.0
        tracking_latency_ms = 0.0

        if self.mode == MODE_TRACKING and self.tracked_bbox is not None:
            t_mode = time.time()
            detected_state, cluster_info, circles = \
                self._run_tracking_pipeline(gray, hsv)
            tracking_latency_ms = (time.time() - t_mode) * 1000.0
        else:
            t_mode = time.time()
            detected_state, cluster_info, circles = \
                self._run_detection_pipeline(gray, hsv)
            detection_latency_ms = (time.time() - t_mode) * 1000.0

        # -- 6. Temporal filtering -------------------------------------
        confirmed_state = self.select_state(detected_state)

        # -- Publish state ---------------------------------------------
        state_msg = String()
        state_msg.data = confirmed_state
        self.state_pub.publish(state_msg)

        # -- 7. Debug image --------------------------------------------
        if self._p('show_debug_display'):
            debug_img = self._draw_debug(
                frame, roi_offset, circles, cluster_info,
                confirmed_state, detected_state
            )
            cv2.imshow("Traffic Light Detector Debug", debug_img)
            cv2.waitKey(1)

        # -- Clean single-line terminal debug --------------------------
        elapsed_ms = (time.time() - t_start) * 1000.0
        radii_str = ','.join(str(r) for _, _, r in circles) if circles else '-'
        n_clusters = sum(1 for ci in cluster_info if ci)
        best_ratio = 0.0
        best_info = ''
        if cluster_info:
            top_score = 0
            top_ratio = 0.0
            top_intensity = 0
            for ci_data in cluster_info:
                for item in ci_data:
                    if item['score'] > top_score:
                        top_score = item['score']
                        top_ratio = item.get('ratio', 0)
                        top_intensity = item.get('intensity', 0)
            best_ratio = top_ratio
            best_info = (f' | best: {detected_state} '
                         f'{top_ratio:.0%} i={top_intensity}')
        self.get_logger().info(
            f'[TLD] {self.mode} | {len(circles)} circles (r={radii_str}) | '
            f'{n_clusters} cluster(s){best_info} | '
            f'-> {confirmed_state} | {elapsed_ms:.1f}ms'
        )

        # -- KPI CSV logging -------------------------------------------
        self._kpi_write_row(
            latency_ms=elapsed_ms,
            mode=self.mode,
            detection_latency_ms=detection_latency_ms,
            tracking_latency_ms=tracking_latency_ms,
            detected_state=detected_state,
            confirmed_state=confirmed_state,
            num_circles=len(circles),
            best_color_ratio=best_ratio,
        )

    # ===================================================================
    # Full DETECTION pipeline
    # ===================================================================

    def _run_detection_pipeline(self, gray, hsv):
        """
        Run the full pipeline: detect → filter → cluster → validate → classify.

        If a valid cluster is found, switch to TRACKING mode.

        Returns:
            (detected_state, cluster_info, circles)
        """
        circles = self.detect_circles(gray)
        circles = self.filter_circles(circles, gray)
        clusters = self.cluster_circles(circles)

        # -- Geometry gate: only keep clusters that look like a traffic light
        validated_clusters = [
            c for c in clusters
            if self._validate_traffic_light_geometry(c) is not None
        ]
        # Fall back to unvalidated multi-circle clusters if nothing passes
        if not validated_clusters:
            min_sz = self._p('min_cluster_size')
            validated_clusters = [c for c in clusters if len(c) >= min_sz]

        detected_state, cluster_info = self.classify_colors(
            validated_clusters, hsv
        )

        # -- Attempt to enter TRACKING mode ----------------------------
        if detected_state != 'UNKNOWN' and cluster_info:
            best_cluster = self._best_cluster(cluster_info)
            if best_cluster is not None:
                self.tracked_bbox = self._cluster_to_bbox(best_cluster)
                self.tracking_lost_count = 0
                self.mode = MODE_TRACKING
                self.get_logger().info('-> TRACKING locked.')

        return detected_state, cluster_info, circles

    # ===================================================================
    # Lightweight TRACKING pipeline
    # ===================================================================

    def _run_tracking_pipeline(self, gray, hsv):
        """
        Run the lightweight pipeline inside the tracked search window.

        Steps:
            1. Expand tracked_bbox by tracking_expansion for the search area
            2. Crop gray / hsv to the search window
            3. Detect circles in the crop
            4. Classify colours directly (skip clustering)
            5. Update tracked_bbox or increment lost counter

        Returns:
            (detected_state, cluster_info, circles_fullcoords)
        """
        expansion = self._p('tracking_expansion')
        lost_limit = self._p('tracking_lost_frames')
        h_img, w_img = gray.shape[:2]

        # -- Expand the tracked bbox to create a search window ---------
        tx1, ty1, tx2, ty2 = self.tracked_bbox
        bw = tx2 - tx1
        bh = ty2 - ty1
        cx = (tx1 + tx2) // 2
        cy = (ty1 + ty2) // 2
        half_w = int(bw * expansion / 2)
        half_h = int(bh * expansion / 2)

        sx1 = max(cx - half_w, 0)
        sy1 = max(cy - half_h, 0)
        sx2 = min(cx + half_w, w_img)
        sy2 = min(cy + half_h, h_img)

        # Validate the search window is sensible
        if sx2 - sx1 < 10 or sy2 - sy1 < 10:
            self._reset_tracking('Search window too small.')
            return 'UNKNOWN', [], []

        # -- Crop ------------------------------------------------------
        gray_crop = gray[sy1:sy2, sx1:sx2]
        hsv_crop = hsv[sy1:sy2, sx1:sx2]

        # -- Detect circles in the crop --------------------------------
        circles_local = self.detect_circles(gray_crop)
        circles_local = self.filter_circles(circles_local, gray_crop)

        if not circles_local:
            self.tracking_lost_count += 1
            if self.tracking_lost_count >= lost_limit:
                self._reset_tracking('Lost traffic light.')
            return 'UNKNOWN', [], []

        # -- Offset circles back to ROI coordinates --------------------
        circles_full = [
            (x + sx1, y + sy1, r) for (x, y, r) in circles_local
        ]

        # -- Classify colours (treat all circles as one cluster) -------
        clusters = [circles_full]
        detected_state, cluster_info = self.classify_colors(clusters, hsv)

        # -- Update tracked bbox to new circle positions ---------------
        # Hysteresis: if circles are found, always update bbox position
        # (the housing is still there even if colour is uncertain)
        if cluster_info:
            best = self._best_cluster(cluster_info)
            if best is not None:
                self.tracked_bbox = self._cluster_to_bbox(best)
            else:
                # Circles found but all UNKNOWN colour — keep position
                self.tracked_bbox = self._cluster_to_bbox(cluster_info[0])
            self.tracking_lost_count = 0
        else:
            self.tracking_lost_count += 1

        if self.tracking_lost_count >= lost_limit:
            self._reset_tracking('Lost traffic light.')

        return detected_state, cluster_info, circles_full

    # ===================================================================
    # Tracking helpers
    # ===================================================================

    def _reset_tracking(self, reason: str):
        """Reset to DETECTION mode."""
        self.mode = MODE_DETECTION
        self.tracked_bbox = None
        self.tracking_lost_count = 0
        self.get_logger().info(f'{reason} → DETECTION mode.')

    @staticmethod
    def _best_cluster(cluster_info):
        """
        Return the cluster data list with the highest individual
        circle score, or None if everything is UNKNOWN.
        """
        best_score = 0
        best = None
        for cluster_data in cluster_info:
            for item in cluster_data:
                if item['score'] > best_score:
                    best_score = item['score']
                    best = cluster_data
        return best

    @staticmethod
    def _cluster_to_bbox(cluster_data):
        """
        Compute a tight bounding box (x1, y1, x2, y2) from a cluster's
        circle list, with a small padding.
        """
        pad = 10
        xs = [d['circle'][0] for d in cluster_data]
        ys = [d['circle'][1] for d in cluster_data]
        rs = [d['circle'][2] for d in cluster_data]
        x1 = min(x - r for x, r in zip(xs, rs)) - pad
        y1 = min(y - r for y, r in zip(ys, rs)) - pad
        x2 = max(x + r for x, r in zip(xs, rs)) + pad
        y2 = max(y + r for y, r in zip(ys, rs)) + pad
        return (x1, y1, x2, y2)

    # ===================================================================
    # 1. Preprocessing
    # ===================================================================

    def preprocess_image(self, frame):
        """
        Preprocess the input BGR frame.

        Steps:
            - Crop the upper region of the image (ROI)
            - Convert ROI to grayscale and HSV
            - Apply Gaussian blur to the grayscale

        Args:
            frame: Full BGR image (numpy array).

        Returns:
            gray_roi:  Blurred grayscale of the ROI.
            hsv_roi:   HSV of the ROI.
            roi_frame: BGR ROI (for debug drawing reference).
            roi_offset: (y_start, y_end) in original frame coordinates.
        """
        h, w = frame.shape[:2]
        roi_ratio = self._p('roi_height_ratio')

        y_end = int(h * roi_ratio)
        roi_frame = frame[0:y_end, :]

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

        # CLAHE: boost local contrast for dim/unlit LEDs on low-quality cameras
        if self._p('use_clahe'):
            clahe = cv2.createCLAHE(
                clipLimit=self._p('clahe_clip_limit'),
                tileGridSize=(8, 8)
            )
            gray = clahe.apply(gray)

        # Gaussian blur to reduce noise before HoughCircles
        gray = cv2.GaussianBlur(gray, (9, 9), 2)

        return gray, hsv, roi_frame, (0, y_end)

    # ===================================================================
    # 2. Circle detection
    # ===================================================================

    def detect_circles(self, gray):
        """
        Detect circles in a grayscale image using Hough Circle Transform.

        In TRACKING mode the input is a small cropped region, so this
        runs very quickly.

        Args:
            gray: Blurred grayscale image (full ROI or crop).

        Returns:
            List of (x, y, r) tuples.  Empty list if nothing found.
        """
        dp = self._p('dp')
        min_dist = self._p('min_dist')
        p1 = self._p('param1')
        p2 = self._p('param2')
        min_r = self._p('min_radius')
        max_r = self._p('max_radius')

        detected = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=p1,
            param2=p2,
            minRadius=min_r,
            maxRadius=max_r
        )

        if detected is None:
            return []

        circles = np.round(detected[0]).astype(int)
        
        # MASSIVE CPU SAVER: Limit to top 15 strongest circles to prevent pipeline explosion
        if len(circles) > 15:
            circles = circles[:15]
            
        return [(int(c[0]), int(c[1]), int(c[2])) for c in circles]

    # ===================================================================
    # 3. Circle filtering
    # ===================================================================

    def filter_circles(self, circles, gray):
        """
        Apply weak filters to remove extreme noise while keeping dim circles.

        Criteria:
            - Radius within [min_radius, max_radius]
            - Circle centre inside image bounds
            - Mean brightness inside the circle ≥ min_brightness
              (only rejects very dark artefacts)

        Args:
            circles: List of (x, y, r).
            gray:    Grayscale image the circles were detected from.

        Returns:
            Filtered list of (x, y, r).
        """
        if not circles:
            return []

        min_r = self._p('min_radius')
        max_r = self._p('max_radius')
        min_bright = self._p('min_brightness')
        h, w = gray.shape[:2]

        filtered = []
        for (x, y, r) in circles:
            # Radius sanity
            if r < min_r or r > max_r:
                continue
            # Centre in-bounds
            if x < 0 or y < 0 or x >= w or y >= h:
                continue

            # Weak brightness filter: Use a tiny bounding box crop instead of full 640x480 mask
            x1 = max(x - r, 0)
            y1 = max(y - r, 0)
            x2 = min(x + r, w)
            y2 = min(y + r, h)
            
            if x1 >= x2 or y1 >= y2:
                continue
                
            roi = gray[y1:y2, x1:x2]
            mean_val = np.mean(roi)
            if mean_val < min_bright:
                continue

            filtered.append((x, y, r))

        return filtered

    # ===================================================================
    # 4. Clustering (DETECTION mode only)
    # ===================================================================

    def cluster_circles(self, circles):
        """
        Group circles into clusters that likely belong to the same traffic
        light housing.

        Uses complete-linkage: every circle in a cluster must be within
        max_circle_distance of every other circle, and radii must be
        similar.  Clusters are sorted top-to-bottom for vertical
        alignment.

        Clusters of 2–3 circles are preferred (typical traffic light).
        Singletons are kept as a fallback.

        Skipped entirely in TRACKING mode.

        Args:
            circles: List of (x, y, r).

        Returns:
            List of clusters, each cluster a list of (x, y, r).
        """
        if not circles:
            return []

        max_dist = self._p('max_circle_distance')
        r_tol = self._p('radius_tolerance')

        n = len(circles)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue

            cluster = [circles[i]]
            visited[i] = True

            for j in range(i + 1, n):
                if visited[j]:
                    continue

                # Check compatibility with every circle already in cluster
                compatible = True
                xj, yj, rj = circles[j]

                for (cx, cy, cr) in cluster:
                    dist = np.sqrt((cx - xj) ** 2 + (cy - yj) ** 2)
                    if dist > max_dist:
                        compatible = False
                        break
                    mean_r = (cr + rj) / 2.0
                    if mean_r > 0 and abs(cr - rj) / mean_r > r_tol:
                        compatible = False
                        break

                if compatible:
                    cluster.append(circles[j])
                    visited[j] = True

            clusters.append(cluster)

        # Sort circles within each cluster by y-coordinate (top → bottom)
        for cluster in clusters:
            cluster.sort(key=lambda c: c[1])

        # Only keep clusters with at least min_cluster_size circles.
        # Singletons from sign letters (O, S, etc.) are discarded.
        min_sz = self._p('min_cluster_size')
        valid = [c for c in clusters if len(c) >= min_sz]
        return valid

    # ===================================================================
    # 4b. Geometry validation (strict 3-circle arrangement)
    # ===================================================================

    def _validate_traffic_light_geometry(self, cluster):
        """Validate that a cluster forms a plausible traffic light.

        Supports both 2-circle and 3-circle clusters:

        For 3 circles:
            - Arranged vertically (preferred) or horizontally.
              Vertical: similar X-coords, roughly equal Y-spacing.
              Horizontal: similar Y-coords, roughly equal X-spacing.

        For 2 circles:
            - Aligned vertically or horizontally (one axis similar).
            - Spacing between centres is reasonable (1.5× – 5× avg radius).

        Args:
            cluster: List of (x, y, r) tuples.

        Returns:
            'VERTICAL' or 'HORIZONTAL' if valid, None otherwise.
        """
        if len(cluster) < 2 or len(cluster) > 3:
            return None

        align_tol = self._p('alignment_tolerance')

        xs = [c[0] for c in cluster]
        ys = [c[1] for c in cluster]
        rs = [c[2] for c in cluster]
        avg_r = sum(rs) / len(rs)
        max_axis_spread = align_tol * avg_r  # how far off-axis is OK

        if len(cluster) == 3:
            # --- 3-circle: Check VERTICAL arrangement -----------------
            sorted_by_y = sorted(cluster, key=lambda c: c[1])
            x_spread = max(c[0] for c in sorted_by_y) - min(c[0] for c in sorted_by_y)
            if x_spread <= max_axis_spread:
                gap1 = sorted_by_y[1][1] - sorted_by_y[0][1]
                gap2 = sorted_by_y[2][1] - sorted_by_y[1][1]
                if gap1 > 0 and gap2 > 0:
                    spacing_ratio = min(gap1, gap2) / max(gap1, gap2)
                    if spacing_ratio >= 0.5:
                        return 'VERTICAL'

            # --- 3-circle: Check HORIZONTAL arrangement ---------------
            sorted_by_x = sorted(cluster, key=lambda c: c[0])
            y_spread = max(c[1] for c in sorted_by_x) - min(c[1] for c in sorted_by_x)
            if y_spread <= max_axis_spread:
                gap1 = sorted_by_x[1][0] - sorted_by_x[0][0]
                gap2 = sorted_by_x[2][0] - sorted_by_x[1][0]
                if gap1 > 0 and gap2 > 0:
                    spacing_ratio = min(gap1, gap2) / max(gap1, gap2)
                    if spacing_ratio >= 0.5:
                        return 'HORIZONTAL'

        elif len(cluster) == 2:
            c0, c1 = cluster[0], cluster[1]
            dx = abs(c0[0] - c1[0])
            dy = abs(c0[1] - c1[1])
            dist = np.sqrt(dx ** 2 + dy ** 2)

            # Spacing sanity: centres should be 1.5× – 5× avg radius apart
            if dist < 1.5 * avg_r or dist > 5.0 * avg_r:
                return None

            # --- 2-circle: Check VERTICAL (X-aligned) -----------------
            if dx <= max_axis_spread and dy > dx:
                return 'VERTICAL'

            # --- 2-circle: Check HORIZONTAL (Y-aligned) ---------------
            if dy <= max_axis_spread and dx > dy:
                return 'HORIZONTAL'

        return None

    # ===================================================================
    # 5. Colour classification + active light selection
    # ===================================================================

    def classify_colors(self, clusters, hsv):
        """
        Classify each circle's colour using HSV masks and select the
        dominant (active) light.

        For each circle:
            - Extract the HSV sub-image bounded by the circle
            - Create a circular mask
            - Count pixels matching red (two ranges), yellow, green
            - Label with the colour that has the highest score

        The circle with the highest score across all clusters determines
        the overall detected state.

        Args:
            clusters: List of clusters (each a list of (x, y, r)).
            hsv:      HSV image (full ROI).

        Returns:
            detected_state: "RED" | "YELLOW" | "GREEN" | "UNKNOWN"
            cluster_info:   List of per-cluster dicts for debug drawing.
        """
        if not clusters:
            return 'UNKNOWN', []

        # Read colour thresholds
        rl1 = np.array(self._p('red_lower1'), dtype=np.uint8)
        ru1 = np.array(self._p('red_upper1'), dtype=np.uint8)
        rl2 = np.array(self._p('red_lower2'), dtype=np.uint8)
        ru2 = np.array(self._p('red_upper2'), dtype=np.uint8)
        yl = np.array(self._p('yellow_lower'), dtype=np.uint8)
        yu = np.array(self._p('yellow_upper'), dtype=np.uint8)
        gl = np.array(self._p('green_lower'), dtype=np.uint8)
        gu = np.array(self._p('green_upper'), dtype=np.uint8)

        h_img, w_img = hsv.shape[:2]

        best_state = 'UNKNOWN'
        best_score = 0
        all_cluster_info = []

        for cluster in clusters:
            cluster_data = []

            for (x, y, r) in cluster:
                # Bounding box clamped to image
                x1 = max(x - r, 0)
                y1 = max(y - r, 0)
                x2 = min(x + r, w_img)
                y2 = min(y + r, h_img)

                if x2 <= x1 or y2 <= y1:
                    cluster_data.append({
                        'circle': (x, y, r), 'label': 'UNKNOWN', 'score': 0
                    })
                    continue

                roi = hsv[y1:y2, x1:x2]

                # Circular mask within the sub-image
                roi_h, roi_w = roi.shape[:2]
                cmask = np.zeros((roi_h, roi_w), dtype=np.uint8)
                cv2.circle(cmask, (x - x1, y - y1), r, 255, -1)

                # Compute mean intensity (V channel) inside the circle
                v_channel = roi[:, :, 2]  # HSV V = brightness
                masked_v = cv2.bitwise_and(v_channel, v_channel, mask=cmask)
                pixel_count = cv2.countNonZero(cmask)
                mean_intensity = (
                    int(np.sum(masked_v) / pixel_count) if pixel_count > 0
                    else 0
                )

                # Colour masks
                red_m = cv2.bitwise_or(
                    cv2.inRange(roi, rl1, ru1),
                    cv2.inRange(roi, rl2, ru2)
                )
                yellow_m = cv2.inRange(roi, yl, yu)
                green_m = cv2.inRange(roi, gl, gu)

                # Pixel scores inside the circle (ratio-based)
                raw_scores = {
                    'RED': int(cv2.countNonZero(
                        cv2.bitwise_and(red_m, cmask))),
                    'YELLOW': int(cv2.countNonZero(
                        cv2.bitwise_and(yellow_m, cmask))),
                    'GREEN': int(cv2.countNonZero(
                        cv2.bitwise_and(green_m, cmask))),
                }

                # Ratio-based scoring: normalise by total circle pixels
                min_ratio = self._p('min_color_ratio')
                ratios = {}
                for colour_name, count in raw_scores.items():
                    ratios[colour_name] = (
                        count / pixel_count if pixel_count > 0 else 0.0
                    )

                max_label = max(ratios, key=ratios.get)
                max_ratio = ratios[max_label]
                max_score = raw_scores[max_label]

                # Require minimum ratio to accept a colour
                if max_ratio < min_ratio:
                    max_label = 'UNKNOWN'

                cluster_data.append({
                    'circle': (x, y, r),
                    'label': max_label,
                    'score': max_score,
                    'intensity': mean_intensity,
                    'ratio': max_ratio,
                })

                # Global best across all clusters
                if max_score > best_score:
                    best_score = max_score
                    best_state = max_label

            all_cluster_info.append(cluster_data)

        return best_state, all_cluster_info

    # ===================================================================
    # 6. Temporal filtering / state selection
    # ===================================================================

    def select_state(self, detected_state):
        """
        Apply temporal filtering using majority vote.

        A state is confirmed if it appears in >= 60% of the last
        min_confirm_frames entries (ignoring UNKNOWN).  This is far
        more robust than requiring all frames to agree.

        Args:
            detected_state: State detected in the current frame.

        Returns:
            Confirmed state string.
        """
        self.state_history.append(detected_state)

        min_frames = self._p('min_confirm_frames')

        if len(self.state_history) < min_frames:
            return 'UNKNOWN'

        recent = list(self.state_history)[-min_frames:]

        # Count non-UNKNOWN votes
        counts = {}
        for s in recent:
            if s != 'UNKNOWN':
                counts[s] = counts.get(s, 0) + 1

        if not counts:
            return 'UNKNOWN'

        # Majority vote: top state must appear in >= 60% of the window
        top_state = max(counts, key=counts.get)
        if counts[top_state] >= min_frames * 0.6:
            return top_state

        return 'UNKNOWN'

    # ===================================================================
    # 7. Debug visualisation
    # ===================================================================

    def _draw_debug(self, frame, roi_offset, circles, cluster_info,
                    confirmed_state, detected_state=''):
        """
        Annotate the original frame with detection results and tuning info.

        Draws:
            - ROI boundary
            - Info panel with current parameters for tuning
            - All filtered circles (grey) with radius labels
            - Clustered circles with colour, radius, and intensity
            - Cluster bounding boxes
            - Tracked search window (TRACKING mode)
            - Current mode label
            - Confirmed state text overlay

        Returns:
            Annotated BGR image (copy).
        """
        debug = frame.copy()
        h, w = debug.shape[:2]
        y_start, y_end = roi_offset

        # ROI boundary
        cv2.rectangle(debug, (0, y_start), (w - 1, y_end),
                      (255, 255, 0), 2)
        cv2.putText(debug, 'ROI', (5, y_start + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # -- Info panel (top-left, below ROI label) --------------------
        panel_lines = [
            f'r: {self._p("min_radius")}-{self._p("max_radius")}  '
            f'p1/p2: {self._p("param1"):.0f}/{self._p("param2"):.0f}',
            f'bright>={self._p("min_brightness")}  '
            f'CLAHE:{"ON" if self._p("use_clahe") else "OFF"}',
            f'Circles: {len(circles)}  Clusters: {len(cluster_info)}',
            f'Detected: {detected_state}',
        ]
        panel_y = y_start + 40
        for i, line in enumerate(panel_lines):
            cv2.putText(debug, line, (5, panel_y + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # All detected/filtered circles (grey outlines with radius)
        for (x, y, r) in circles:
            cv2.circle(debug, (x, y), r, (180, 180, 180), 1)
            cv2.putText(debug, f'r={r}', (x + r + 2, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

        # Colour map for labels
        cmap = {
            'RED': (0, 0, 255),
            'YELLOW': (0, 255, 255),
            'GREEN': (0, 255, 0),
            'UNKNOWN': (128, 128, 128),
        }

        # Clustered circles with colour, radius, intensity, ratio
        for ci, cluster_data in enumerate(cluster_info):
            for item in cluster_data:
                cx, cy, cr = item['circle']
                label = item['label']
                intensity = item.get('intensity', '?')
                ratio = item.get('ratio', 0)
                colour = cmap.get(label, (128, 128, 128))

                cv2.circle(debug, (cx, cy), cr, colour, 2)
                cv2.putText(
                    debug,
                    f'{label} r={cr} i={intensity} {ratio:.0%}',
                    (cx - cr, cy - cr - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1
                )

            # Bounding box around the cluster
            if cluster_data:
                xs = [d['circle'][0] for d in cluster_data]
                ys = [d['circle'][1] for d in cluster_data]
                rs = [d['circle'][2] for d in cluster_data]
                pad = 10
                bx1 = min(x - r for x, r in zip(xs, rs)) - pad
                by1 = min(y - r for y, r in zip(ys, rs)) - pad
                bx2 = max(x + r for x, r in zip(xs, rs)) + pad
                by2 = max(y + r for y, r in zip(ys, rs)) + pad
                cv2.rectangle(debug, (bx1, by1), (bx2, by2),
                              (255, 200, 0), 1)
                cv2.putText(debug, f'Cluster {ci}', (bx1, by1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 200, 0), 1)

        # Tracked search window (TRACKING mode)
        if self.mode == MODE_TRACKING and self.tracked_bbox is not None:
            tx1, ty1, tx2, ty2 = self.tracked_bbox
            expansion = self._p('tracking_expansion')
            tbw = tx2 - tx1
            tbh = ty2 - ty1
            tcx = (tx1 + tx2) // 2
            tcy = (ty1 + ty2) // 2
            shw = int(tbw * expansion / 2)
            shh = int(tbh * expansion / 2)
            cv2.rectangle(
                debug,
                (max(tcx - shw, 0), max(tcy - shh, 0)),
                (min(tcx + shw, w), min(tcy + shh, y_end)),
                (255, 0, 255), 2
            )
            cv2.putText(debug, 'TRACKING', (max(tcx - shw, 0),
                        max(tcy - shh, 0) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 255), 1)

        # Mode label (top-right corner)
        mode_colour = (0, 255, 0) if self.mode == MODE_TRACKING \
            else (0, 165, 255)
        cv2.putText(debug, self.mode, (w - 160, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_colour, 2)

        # Confirmed state overlay (bottom-left)
        state_colour = cmap.get(confirmed_state, (255, 255, 255))
        cv2.putText(debug, f'State: {confirmed_state}',
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    state_colour, 2)

        return debug


# ===================================================================
# Main entry point
# ===================================================================

def main(args=None):
    """Spin up the traffic light detector node."""
    rclpy.init(args=args)
    node = TrafficLightDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._kpi_close()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
