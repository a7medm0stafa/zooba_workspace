"""
Camera Publisher Node
=====================
Single owner of the physical camera hardware.  Captures frames and
publishes them as sensor_msgs/Image on /camera/image_raw so that
multiple perception nodes can subscribe to the same stream.

Parameters:
    camera_id        (int)    – V4L2 device index (default 0)
    frame_width      (int)    – Capture width  (default 640)
    frame_height     (int)    – Capture height (default 480)
    fps              (float)  – Capture / publish rate (default 20.0)
    flip_code        (int)    – cv2.flip code (-1 = 180°, 0 = vertical,
                                1 = horizontal, -2 = no flip)
    output_topic     (str)    – Image topic (default /camera/image_raw)

Publishes:
    /camera/image_raw  (sensor_msgs/Image)
"""

import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraPublisherNode(Node):
    """Publishes camera frames as ROS Image messages."""

    def __init__(self):
        super().__init__('camera_publisher_node')

        # -- Parameters ------------------------------------------------
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('fps', 20.0)
        self.declare_parameter('flip_code', -1)       # -1 = 180° rotation
        self.declare_parameter('output_topic', '/camera/image_raw')

        cam_id = self.get_parameter('camera_id').value
        width = self.get_parameter('frame_width').value
        height = self.get_parameter('frame_height').value
        fps = self.get_parameter('fps').value
        self.flip_code = self.get_parameter('flip_code').value
        output_topic = self.get_parameter('output_topic').value

        # -- Camera hardware -------------------------------------------
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            self.get_logger().error(
                f'Cannot open camera (device {cam_id})'
            )

        # -- Publisher --------------------------------------------------
        self.image_pub = self.create_publisher(Image, output_topic, 10)

        # -- Bridge -----------------------------------------------------
        self.bridge = CvBridge()

        # -- Timer ------------------------------------------------------
        self.timer = self.create_timer(1.0 / fps, self._timer_callback)

        self.get_logger().info('=' * 50)
        self.get_logger().info('Camera Publisher Node Started')
        self.get_logger().info(f'  Device     : {cam_id}')
        self.get_logger().info(f'  Resolution : {width}x{height}')
        self.get_logger().info(f'  FPS        : {fps}')
        self.get_logger().info(f'  Flip code  : {self.flip_code}')
        self.get_logger().info(f'  Topic      : {output_topic}')
        self.get_logger().info('=' * 50)

    # ------------------------------------------------------------------
    # Timer callback
    # ------------------------------------------------------------------

    def _timer_callback(self):
        if not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warn(
                'Failed to capture frame', throttle_duration_sec=5.0
            )
            return

        # Apply flip if configured (-2 means no flip)
        if self.flip_code != -2:
            frame = cv2.flip(frame, self.flip_code)

        # Convert to ROS Image and publish
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'camera_link'
        self.image_pub.publish(img_msg)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
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
