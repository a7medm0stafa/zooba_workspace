import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import numpy as np
import time


class SignDetectionNode(Node):

    def __init__(self):
        super().__init__('sign_detection_node')

        # Initialize camera
        self.cap = cv2.VideoCapture(0)

        # ROS Publisher
        self.command_publisher = self.create_publisher(
            String,
            'vehicle/command',
            10
        )

        # Run pipeline ~20 FPS
        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info("Vision pipeline started")


    def timer_callback(self):

        start_time = time.time()

        ret, frame = self.cap.read()

        if not ret:
            return

        # -------------------------
        # Preprocessing Pipeline
        # -------------------------

        roi = self.crop_right_side(frame)

        scaled = self.resize_scale(roi)

        bright = self.adjust_brightness(scaled, 30)

        contrast = self.enhance_contrast(bright)

        blurred = self.apply_gaussian_blur(contrast)

        hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # -------------------------
        # Traffic Light Detection
        # -------------------------

        command, masks = self.detect_traffic_light(hsv_img)

        # Publish command
        self.command_publisher.publish(String(data=command))

        # -------------------------
        # Debug Windows
        # -------------------------

        red_mask, yellow_mask, green_mask = masks

        cv2.imshow("Original Camera", frame)
        cv2.imshow("ROI", roi)
        cv2.imshow("Processed (HSV)", hsv_img)

        cv2.imshow("Red Mask", red_mask)
        cv2.imshow("Yellow Mask", yellow_mask)
        cv2.imshow("Green Mask", green_mask)

        cv2.waitKey(1)

        # -------------------------
        # Performance log
        # -------------------------

        end_time = time.time()
        elapsed = (end_time - start_time) * 1000

        self.get_logger().info(f"Command: {command} | {elapsed:.2f} ms")


    # --------------------------------
    # Geometric Transformation 1
    # Crop Right Side
    # --------------------------------

    def crop_right_side(self, image):

        h, w, _ = image.shape

        return image[:, w//2:w]


    # --------------------------------
    # Geometric Transformation 2
    # Resize / Scaling
    # --------------------------------

    def resize_scale(self, image):

        return cv2.resize(image, (320, 240))


    # --------------------------------
    # Intensity Transformation
    # Brightness Adjustment
    # --------------------------------

    def adjust_brightness(self, image, value):

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv)

        v = cv2.add(v, value)

        v = np.clip(v, 0, 255)

        hsv = cv2.merge((h, s, v))

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


    # --------------------------------
    # Contrast Enhancement (CLAHE)
    # --------------------------------

    def enhance_contrast(self, image):

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8,8)
        )

        cl = clahe.apply(l)

        merged = cv2.merge((cl, a, b))

        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


    # --------------------------------
    # Image Smoothing
    # Gaussian Blur
    # --------------------------------

    def apply_gaussian_blur(self, image):

        return cv2.GaussianBlur(image, (5,5), 0)


    # --------------------------------
    # Traffic Light Detection
    # --------------------------------

    def detect_traffic_light(self, hsv):

        # RED ranges (two ranges in HSV)
        lower_red1 = np.array([0,120,70])
        upper_red1 = np.array([10,255,255])

        lower_red2 = np.array([170,120,70])
        upper_red2 = np.array([179,255,255])

        # GREEN
        lower_green = np.array([35,100,100])
        upper_green = np.array([85,255,255])

        # YELLOW
        lower_yellow = np.array([20,100,100])
        upper_yellow = np.array([30,255,255])


        # Create masks

        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        red_mask = red_mask1 + red_mask2

        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)


        # Count detected pixels

        red_pixels = np.sum(red_mask)

        yellow_pixels = np.sum(yellow_mask)

        green_pixels = np.sum(green_mask)


        threshold = 5000


        # Decision logic

        if red_pixels > threshold:

            command = "STOP"

        elif yellow_pixels > threshold:

            command = "SLOW"

        elif green_pixels > threshold:

            command = "GO"

        else:

            command = "NO_SIGNAL"


        return command, (red_mask, yellow_mask, green_mask)


# --------------------------------
# Main
# --------------------------------

def main(args=None):

    rclpy.init(args=args)

    node = SignDetectionNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    node.cap.release()

    cv2.destroyAllWindows()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':

    main()