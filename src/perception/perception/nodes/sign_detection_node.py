import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import numpy as np

FRAME_W, FRAME_H = 320, 240   # optimized for Pi


# =====================================
# SIGN DETECTOR
# =====================================
class SignDetector:

    def __init__(self):
        self.min_area = 2000
        self.max_area = 150000
        self.min_purity = 0.4

        self.stop_min_circ = 0.75
        self.slow_min_circ = 0.30
        self.turn_min_circ = 0.65

        self.epsilon = 0.02

        self.RED = [
            (np.array([0,50,50]), np.array([15,255,255])),
            (np.array([160,50,50]), np.array([179,255,255]))
        ]
        self.YELLOW = (np.array([18,90,90]), np.array([38,255,255]))
        self.BLUE   = (np.array([100,130,60]), np.array([130,255,255]))

    # ----------------------------
    def detect(self, frame):
        img = cv2.resize(frame, (FRAME_W, FRAME_H))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self.RED:
            red_mask |= cv2.inRange(hsv, lo, hi)

        yellow_mask = cv2.inRange(hsv, *self.YELLOW)
        blue_mask   = cv2.inRange(hsv, *self.BLUE)

        red_mask = self.clean(red_mask)
        yellow_mask = self.clean(yellow_mask)
        blue_mask = self.clean(blue_mask)

        # priority: STOP > TURN > SLOW
        res = self.process_mask(img, red_mask, "STOP", 7, 10, self.stop_min_circ)
        if res: return res

        res = self.process_blue(img, blue_mask)
        if res: return res

        res = self.process_mask(img, yellow_mask, "SLOW_DOWN", 3, 5, self.slow_min_circ)
        if res: return res

        return "NO_SIGN"

    # ----------------------------
    def process_mask(self, img, mask, label, min_v, max_v, min_circ):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue

            x,y,w,h = cv2.boundingRect(c)
            ar = w / float(h)
            if not (0.8 < ar < 1.2):
                continue

            roi_mask = mask[y:y+h, x:x+w]
            purity = cv2.countNonZero(roi_mask) / (w*h)
            if purity < self.min_purity:
                continue

            cnt = max(cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)

            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue

            approx = cv2.approxPolyDP(cnt, self.epsilon * peri, True)
            verts = len(approx)

            area = cv2.contourArea(cnt)
            circ = 4*np.pi*area/(peri*peri)

            if min_v <= verts <= max_v and circ >= min_circ:
                return label

        return None

    # ----------------------------
    def process_blue(self, img, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue

            x,y,w,h = cv2.boundingRect(c)
            if not (0.8 < w/h < 1.2):
                continue

            roi_mask = mask[y:y+h, x:x+w]

            cnt = max(cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue

            area = cv2.contourArea(cnt)
            circ = 4*np.pi*area/(peri*peri)

            if circ < self.turn_min_circ:
                continue

            roi = img[y:y+h, x:x+w]
            direction = self.arrow_direction(roi)

            if direction:
                return f"TURN_{direction}"

        return None

    # ----------------------------
    def arrow_direction(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        blue_mask = cv2.inRange(hsv, *self.BLUE)
        white = cv2.inRange(hsv, (0,0,180), (180,80,255))

        arrow = cv2.bitwise_and(white, cv2.bitwise_not(blue_mask))
        arrow = self.clean(arrow)

        if cv2.countNonZero(arrow) < 80:
            return None

        h, w = arrow.shape
        left = np.sum(arrow[:, :w//2])
        right = np.sum(arrow[:, w//2:])

        # FIXED condition (no runtime error)
        if abs(left - right) < 0.1 * (left + right):
            return None

        # correct direction (no inversion)
        return "LEFT" if left > right else "RIGHT"

    # ----------------------------
    def clean(self, mask):
        k = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        return mask


# =====================================
# ROS2 NODE
# =====================================
class SignNode(Node):

    def __init__(self):
        super().__init__('sign_node')

        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(3, FRAME_W)
        self.cap.set(4, FRAME_H)

        self.detector = SignDetector()

        self.pub = self.create_publisher(String, '/sign', 10)
        self.timer = self.create_timer(0.05, self.loop)

        # ---- Anti-jitter ----
        self.prev = "NO_SIGN"
        self.stable = "NO_SIGN"
        self.counter = 0

    # ----------------------------
    def loop(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Flip horizontally AND vertically (180 deg rotation) to fix upside-down camera
        frame = cv2.flip(frame, -1)

        sign = self.detector.detect(frame)

        # ---- smoothing ----
        if sign == self.prev:
            self.counter += 1
        else:
            self.counter = 0

        if self.counter >= 2:
            self.stable = sign

        self.prev = sign
        sign = self.stable

        msg = String()
        msg.data = sign
        self.pub.publish(msg)

        cv2.putText(frame, sign, (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Detection", frame)
        cv2.waitKey(1)


# =====================================
def main():
    rclpy.init()
    node = SignNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
