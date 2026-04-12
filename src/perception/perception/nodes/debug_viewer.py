"""
Debug image viewer — displays /traffic_light/debug_image using OpenCV.

Usage:
    ros2 run perception debug_viewer
"""

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class DebugViewer(Node):

    def __init__(self):
        super().__init__('debug_viewer')
        self.bridge = CvBridge()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(
            Image, '/traffic_light/debug_image', self._cb, qos
        )
        self.create_subscription(
            Image, '/camera/image_raw', self._raw_cb, 10
        )

        self.get_logger().info(
            'Viewer started — showing debug image + raw camera. '
            'Press Q to quit.'
        )

    def _cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow('Traffic Light Detection (Debug)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise SystemExit

    def _raw_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow('Raw Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = DebugViewer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
