"""
Control Merger Node
===================
Merges separate speed and lateral control outputs into a single VehicleCmd.

The speed controller publishes velocity on /teleop/speed_cmd (Float64)
The lateral controller publishes steering on /teleop/lateral_cmd (Float64)
This node combines them into /teleop/raw_cmd (VehicleCmd) for the
non-holonomic constraints node.

Subscribes:
    /teleop/speed_cmd    (std_msgs/Float64)  — velocity [m/s]
    /teleop/lateral_cmd  (std_msgs/Float64)  — steering angle [degrees]

Publishes:
    /teleop/raw_cmd  (vehicle_interfaces/VehicleCmd)
"""

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleCmd
from std_msgs.msg import Float64


class ControlMergerNode(Node):

    def __init__(self):
        super().__init__('control_merger_node')

        # ---- Parameters ----
        self.declare_parameter('speed_topic', '/teleop/speed_cmd')
        self.declare_parameter('lateral_topic', '/teleop/lateral_cmd')
        self.declare_parameter('output_topic', '/teleop/raw_cmd')
        self.declare_parameter('publish_rate', 20.0)

        speed_topic = self.get_parameter('speed_topic').value
        lateral_topic = self.get_parameter('lateral_topic').value
        output_topic = self.get_parameter('output_topic').value
        publish_rate = self.get_parameter('publish_rate').value

        # ---- State ----
        self.latest_velocity = 0.0
        self.latest_heading = 0.0

        # ---- Subscribers ----
        self.speed_sub = self.create_subscription(
            Float64, speed_topic, self._speed_callback, 10)
        self.lateral_sub = self.create_subscription(
            Float64, lateral_topic, self._lateral_callback, 10)

        # ---- Publisher ----
        self.cmd_pub = self.create_publisher(VehicleCmd, output_topic, 10)

        # ---- Timer ----
        self.timer = self.create_timer(1.0 / publish_rate, self._timer_callback)

        self.get_logger().info('=' * 50)
        self.get_logger().info('Control Merger Node Started')
        self.get_logger().info(f'  Speed input   : {speed_topic}')
        self.get_logger().info(f'  Lateral input : {lateral_topic}')
        self.get_logger().info(f'  Output        : {output_topic}')
        self.get_logger().info(f'  Publish rate  : {publish_rate:.0f} Hz')
        self.get_logger().info('=' * 50)

    def _speed_callback(self, msg: Float64):
        """Receive velocity from speed controller."""
        self.latest_velocity = msg.data

    def _lateral_callback(self, msg: Float64):
        """Receive steering angle from lateral controller."""
        self.latest_heading = msg.data

    def _timer_callback(self):
        """Publish merged VehicleCmd at fixed rate."""
        cmd = VehicleCmd()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.velocity = self.latest_velocity
        cmd.heading = self.latest_heading
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)

    node = ControlMergerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
