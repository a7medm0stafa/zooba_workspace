import cv2
import numpy as np
import time

FRAME_W, FRAME_H = 640, 480

RED_RANGES = [
    (np.array([0,   120, 70]),  np.array([10,  255, 255])),
    (np.array([170, 120, 70]),  np.array([179, 255, 255])),
]
YELLOW_RANGE = (
    np.array([18, 90, 90]),
    np.array([38, 255, 255])
)# Increased Min Saturation to 130 so white arrows aren't accidentally captured as blue
BLUE_RANGE   = (np.array([100, 130, 60]), np.array([130, 255, 255]))

class SignDetector:
    def __init__(self):
        self.min_area = 1200
        self.max_area = 120_000
        self.epsilon_factor = 0.02
        
        # Purity check: object must significantly fill its bounding box in the color mask
        self.min_purity = 0.50

        # Geometric Strictness
        self.stop_min_circularity  = 0.75  # Octagon is ~0.94
        self.slow_min_circularity  = 0.52  # Triangle is ~0.60
        self.turn_min_circularity  = 0.75  # Circle is ~1.0

    def detect(self, original_frame, show_dbg=False):
        """
        Architecture Pipeline
        """
        # --- Pre-processing & Crop Optimization ---
        img = cv2.resize(original_frame, (FRAME_W, FRAME_H))
        y_end = int(FRAME_H * 0.6)  # Decreased top crop to keep only 40%
        img_bgr = img[0:y_end, :]

        # Light native brightness boost for HSV
        hsv_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_raw)
        v = np.clip(cv2.add(v, 20), 0, 255).astype(np.uint8)
        hsv = cv2.merge((h, s, v))

        # Output Containers
        detections = []
        if show_dbg:
            debug_shapes = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            # Convert back to BGR so we can draw colored contours on it
            debug_shapes = cv2.cvtColor(debug_shapes, cv2.COLOR_GRAY2BGR)
            debug_edges = np.zeros_like(img_bgr)
        else:
            debug_shapes = None
            debug_edges = None

        # ------------------------------------------------------------------
        # PIECE 1: Red Signs (STOP)
        # ------------------------------------------------------------------
        # Step 1: HSV mask (color)
        mask_r = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in RED_RANGES:
            mask_r = cv2.bitwise_or(mask_r, cv2.inRange(hsv, lo, hi))
        
        # Step 2: Morphological cleaning
        clean_r = self._clean(mask_r)
        
        # Step 3, 4, 5, 6
        det_r = self._process_pipeline('STOP', clean_r, img_bgr, 6, 14, self.stop_min_circularity, debug_shapes, debug_edges, (0, 0, 255))
        detections.extend(det_r)

        # ------------------------------------------------------------------
        # PIECE 2: Yellow Signs (SLOW_DOWN)
        # ------------------------------------------------------------------
        # Step 1: HSV mask (color)
        mask_y = cv2.inRange(hsv, *YELLOW_RANGE)
        
        # Step 2: Morphological cleaning
        clean_y = self._clean(mask_y)
        
        # Step 3, 4, 5, 6
        det_y = self._process_pipeline('SLOW_DOWN', clean_y, img_bgr, 3, 6, self.slow_min_circularity, debug_shapes, debug_edges, (0, 255, 255))
        detections.extend(det_y)

        # ------------------------------------------------------------------
        # PIECE 3: Blue Signs (TURN_LEFT / RIGHT)
        # ------------------------------------------------------------------
        # Step 1 & 2
        mask_b = cv2.inRange(hsv, *BLUE_RANGE)
        clean_b = self._clean(mask_b)
        
        # Step 3, 4, 5, 6
        det_b = self._process_blue_pipeline(clean_b, img_bgr, debug_shapes, debug_edges)
        detections.extend(det_b)

        # ------------------------------------------------------------------
        
        clean_masks = (clean_r, clean_y, clean_b) if show_dbg else None

        best_cmd = 'NO_SIGN'
        best_conf = 0.0
        best_bbox = (0, 0, 0, 0)
        
        if detections:
            best_det = max(detections, key=lambda d: d[1])
            best_cmd, best_conf, best_bbox = best_det

        return best_cmd, best_conf, best_bbox, img_bgr, clean_masks, debug_shapes, debug_edges

    # =====================================================================
    #  Core Pipeline Logic Implementation
    # =====================================================================

    def _clean(self, mask):
        """Step 2: Morphological cleaning"""
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)

    def _process_pipeline(self, label, clean_mask, img_bgr, min_v, max_v, min_circ, debug_shapes, debug_edges, color):
        dets = []
        # Step 3: Find contours
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            # Step 4: Filter (area, aspect ratio, purity)
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area: 
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Aspect Ratio Filter (Signs are roughly square)
            ar = w / float(h) if h > 0 else 0
            if not (0.5 < ar < 2.0): 
                continue
                
            # Color Purity Filter: Check % of True pixels in tight bounding box
            roi_mask = clean_mask[y:y+h, x:x+w]
            purity = cv2.countNonZero(roi_mask) / float(w * h)
            if purity < self.min_purity: 
                continue

            if debug_shapes is not None:
                # Draw the ROI boundary directly
                cv2.rectangle(debug_shapes, (x, y), (x+w, y+h), (50, 200, 50), 1)

            # Step 5: Extract ROI from the original BGR image with slight padding
            pad = 10
            y1, y2 = max(0, y-pad), min(img_bgr.shape[0], y+h+pad)
            x1, x2 = max(0, x-pad), min(img_bgr.shape[1], x+w+pad)
            roi_bgr = img_bgr[y1:y2, x1:x2]

            # Step 6: Inside ROI: grayscale + edges, shape detection
            passed, circ, verts = self._roi_shape_detection(roi_bgr, min_v, max_v, min_circ, debug_shapes, debug_edges, x1, y1, color)
            
            if passed:
                conf = min(1.0, circ + 0.2 * (area / self.max_area))
                dets.append((label, round(conf, 2), (x, y, w, h)))
                
        return dets

    def _process_blue_pipeline(self, clean_mask, img_bgr, debug_shapes, debug_edges):
        """Specifically handles Blue Turn arrows which require arrow-mass calculation inside the ROI"""
        dets = []
        # Step 3
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            # Step 4: Cheap global filters
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area: continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            if not (0.6 < w / float(h) < 1.6): continue
            
            roi_mask = clean_mask[y:y+h, x:x+w]
            purity = cv2.countNonZero(roi_mask) / float(w * h)
            if purity < self.min_purity: continue

            if debug_shapes is not None:
                # Draw the ROI boundary directly
                cv2.rectangle(debug_shapes, (x, y), (x+w, y+h), (50, 200, 50), 1)

            # Step 5: Extract ROI with slight padding
            pad = 10
            y1, y2 = max(0, y-pad), min(img_bgr.shape[0], y+h+pad)
            x1, x2 = max(0, x-pad), min(img_bgr.shape[1], x+w+pad)
            roi_bgr = img_bgr[y1:y2, x1:x2]
            
            # Step 6: Grayscale geometry & arrow logic
            passed, circ, verts = self._roi_shape_detection(roi_bgr, 6, 20, self.turn_min_circularity, debug_shapes, debug_edges, x1, y1, (255, 100, 0))
            if passed:
                direction = self._arrow_direction(roi_bgr)
                if direction:
                    conf = min(1.0, circ + 0.1)
                    dets.append((f'TURN_{direction}', round(conf, 2), (x, y, w, h)))
        return dets

    def _roi_shape_detection(self, roi_bgr, min_v, max_v, min_circ, debug_shapes, debug_edges, offset_x, offset_y, color):
        """
        Step 6 Logic: Heavy mathematics explicitly restricted to tiny ROIs.
        Converts the tight ROI to Grayscale, applies Canny Edge Detection to secure the structural contour precisely, and checks geometry.
        """
        if roi_bgr.shape[0] < 10 or roi_bgr.shape[1] < 10: 
            return False, 0.0, 0
            
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)
        
        # Use Canny instead of Otsu threshold to securely define the outer boundary curve without being broken by inner white arrow symbols
        edge_map = cv2.Canny(roi_gray, 50, 150)
        edge_map = cv2.dilate(edge_map, None, iterations=1)
        
        contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if debug_edges is not None:
            edge_bgr = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
            h, w = edge_bgr.shape[:2]
            if offset_y+h <= debug_edges.shape[0] and offset_x+w <= debug_edges.shape[1]:
                debug_edges[offset_y:offset_y+h, offset_x:offset_x+w] = cv2.bitwise_or(
                    debug_edges[offset_y:offset_y+h, offset_x:offset_x+w], 
                    edge_bgr
                )
        
        if not contours: 
            return False, 0.0, 0

        # Grab the largest geometric entity inside the ROI
        cnt = max(contours, key=cv2.contourArea)
        
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if peri == 0: 
            return False, 0.0, 0
            
        circ = 4 * np.pi * area / (peri * peri)
        approx = cv2.approxPolyDP(cnt, self.epsilon_factor * peri, True)
        verts = len(approx)

        passed = (min_v <= verts <= max_v) and (circ >= min_circ)

        if debug_shapes is not None:
            # Shift the local coordinates back to global canvas space for the debug view
            approx_global = approx + np.array([offset_x, offset_y])
            cnt_global = cnt + np.array([offset_x, offset_y])
            
            # Draw the tight raw ROI contour outline
            cv2.drawContours(debug_shapes, [cnt_global], -1, (80, 80, 80), 1)
            
            if passed:
                cv2.drawContours(debug_shapes, [approx_global], -1, color, 2)
                for pt in approx_global.squeeze():
                    cv2.circle(debug_shapes, tuple(pt), 4, (255,255,255), -1)
                
            text_color = color if passed else (100, 100, 100)
            cv2.putText(debug_shapes, f'v={verts} c={circ:.2f}', (offset_x, offset_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

        return passed, circ, verts

    # ---------------------------------------------------------------------
    # Arrow Processing Logic inside the Blue ROI
    # ---------------------------------------------------------------------

    def _arrow_direction(self, roi):
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        b_mask = cv2.inRange(hsv_roi, *BLUE_RANGE)
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Use Otsu to dynamically figure out exactly how bright the arrow is compared to the rest of the dark sign
        _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        am = cv2.bitwise_and(bright, cv2.bitwise_not(b_mask))
        
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        am = cv2.morphologyEx(cv2.morphologyEx(am, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
        
        nw, total = cv2.countNonZero(am), am.shape[0]*am.shape[1]
        if nw < total * 0.05 or nw > total * 0.80: return None
        
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
        return 'LEFT' if r > 0.15 else ('RIGHT' if r < -0.15 else None)

# =========================================================================
#  Application Runner Tools
# =========================================================================

COLORS = {
    'STOP': (0,0,255), 'SLOW_DOWN': (0,200,255), 
    'TURN_LEFT': (255,150,0), 'TURN_RIGHT': (255,100,0), 'NO_SIGN': (100,100,100)
}

def draw(frame_top, cmd, conf, bbox, fps):
    x, y, w, h = bbox
    out = frame_top.copy()
    if cmd != 'NO_SIGN':
        col = COLORS.get(cmd, (255,255,255))
        cv2.rectangle(out, (x, y), (x+w, y+h), col, 2)
        lbl = f'{cmd} {conf:.0%}'
        cv2.putText(out, lbl, (x, max(15, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    fh = out.shape[0]
    cv2.rectangle(out, (0, fh-28), (FRAME_W, fh), (25, 25, 25), -1)
    cv2.putText(out, f'{fps:.0f} FPS | CMD: {cmd}', (8, fh-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS.get(cmd, (180,180,180)), 2)
    return out

def debug_masks(clean_masks):
    r, y, b = clean_masks
    row = np.hstack([cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) for m in (r, y, b)])
    for txt, xp, col in [('1. RED', 10, (0,0,255)), ('2. YELLOW', 650, (0,255,255)), ('3. BLUE', 1290, (255,100,0))]:
        cv2.putText(row, txt, (xp, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return row

def main():
    print('====================================')
    print(' Sign Detection Pipeline Architecture')
    print(' Q: Quit  |  D: Debug ROI Geometry')
    print('====================================')
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    
    detector = SignDetector()
    show_dbg, fps, prev_cmd = False, 0.0, ''
    
    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok: break
        
        cmd, conf, bbox, processed, dbg_mask, dbg_shapes, dbg_edges = detector.detect(frame, show_dbg)
        
        if cmd != prev_cmd and cmd != 'NO_SIGN':
            print(f' >>> {cmd} ({conf:.0%})')
        prev_cmd = cmd
        
        dt = time.time() - t0
        fps = 0.9 * fps + 0.1 / max(dt, 1e-4)
        
        display = draw(processed, cmd, conf, bbox, fps)
        cv2.imshow('Camera 60% Crop', display)
        
        if show_dbg and dbg_shapes is not None:
            cv2.imshow('Global Masks', debug_masks(dbg_mask))
            cv2.imshow('ROI Shape Debug (Grayscale map)', dbg_shapes)
            if dbg_edges is not None:
                cv2.imshow('Canny Edges (ROI Projection)', dbg_edges)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('d'):
            show_dbg = not show_dbg
            if not show_dbg:
                try: cv2.destroyWindow('Global Masks')
                except: pass
                try: cv2.destroyWindow('ROI Shape Debug (Grayscale map)')
                except: pass
                try: cv2.destroyWindow('Canny Edges (ROI Projection)')
                except: pass
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()