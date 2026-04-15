import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════
#  HSV Color Ranges - Adjust these if your room lighting varies
# ═══════════════════════════════════════════════════════════
RED_RANGES = [
    (np.array([0, 130, 110]), np.array([10, 255, 255])),
    (np.array([170, 120, 100]), np.array([180, 255, 255]))
]

def verify_hexagon_angles(approx):
    """Verify hexagonal/octagonal shape by checking internal angles"""
    if len(approx) < 6:
        return False
    
    angles = []
    for i in range(len(approx)):
        p1 = approx[i-1][0]
        p2 = approx[i][0]
        p3 = approx[(i+1) % len(approx)][0]
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        norm_prod = (np.linalg.norm(v1) * np.linalg.norm(v2))
        if norm_prod == 0: continue
        
        cos_angle = np.dot(v1, v2) / (norm_prod + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        angles.append(angle)
    
    if not angles: return False
    
    angle_std = np.std(angles)
    angle_mean = np.mean(angles)
    
    # Octagon angles are typically ~135°, Hexagons ~120°.
    # We use a threshold to separate these from random noise.
    return angle_std < 25 and 100 < angle_mean < 145

def find_red_sign(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create Red Mask
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in RED_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    
    # Morphological cleaning (Open then Close)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2500: continue # Adjust based on distance from camera
        
        peri = cv2.arcLength(cnt, True)
        # Using 0.02 epsilon to ensure we see actual corners
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        verts = len(approx)
        
        # --- MASS FILTERS (The Circle Killers) ---
        # 1. Solidity: Ratio of area to convex hull area
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # 2. Extent: Ratio of area to bounding rectangle area
        x, y, w, h = cv2.boundingRect(cnt)
        extent = area / (w * h) if (w * h) > 0 else 0
        
        # 3. Aspect Ratio: Road signs are roughly 1:1
        aspect_ratio = w / float(h)

        # Real-time data logging to terminal
        print(f"V:{verts} | S:{solidity:.2f} | E:{extent:.2f} | AR:{aspect_ratio:.2f}")

        # 4. FINAL DETECTION GATE
        # Octagons (STOP) have lower Solidity and higher Extent than Circles.
        if 0.8 <= aspect_ratio <= 1.2:
            if 0.80 < extent < 0.88 and solidity < 0.985:
                if 6 <= verts <= 10 and verify_hexagon_angles(approx):
                    # SUCCESS: Draw detection visual
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
                    cv2.putText(frame, "STOP DETECTED", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    print(">>> [MATCH] STOP SIGN DETECTED")

    return frame, mask

def main():
    # Initialize Laptop Camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera running. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Optional: Flip if the image is mirrored
        frame = cv2.flip(frame, 1)
        
        processed_frame, debug_mask = find_red_sign(frame)
        
        cv2.imshow("STOP Sign Detection", processed_frame)
        cv2.imshow("Red Channel Mask", debug_mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()