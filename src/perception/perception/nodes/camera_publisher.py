"""
Simple webcam publisher for testing.

Captures frames from the laptop camera (device 0) and publishes them
on /camera/image_raw at ~30 FPS.

Usage:
    ros2 run perception camera_publisher
"""

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraPublisher(Node):

    def __init__(self):
        super().__init__('camera_publisher')

        self.declare_parameter('device_id', 0)
        self.declare_parameter('fps', 30.0)

        device = self.get_parameter('device_id').value
        fps = self.get_parameter('fps').value

        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            self.get_logger().error(f'Cannot open camera device {device}')
            return

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.timer = self.create_timer(1.0 / fps, self._timer_cb)

        self.get_logger().info(
            f'Publishing webcam (device {device}) on /camera/image_raw '
            f'at {fps} FPS'
        )

    def _timer_cb(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to read frame from camera')
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.pub.publish(msg)

    def destroy_node(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
