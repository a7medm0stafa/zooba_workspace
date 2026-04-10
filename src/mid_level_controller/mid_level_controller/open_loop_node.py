"""
Open Loop Response Node for Ackermann Vehicle
==============================================
Publishes a constant VehicleCmd at a fixed rate for open-loop testing.

All command values are configurable via ROS2 parameters so they can
be set directly from the launch file without editing code.

Parameters:
    velocity        : target velocity in m/s  (default: 0.0)
    heading         : steering angle in degrees (default: 0.0)
    publish_rate    : publishing rate in Hz (default: 10.0)
    duration        : how long to publish in seconds, 0 = forever (default: 0.0)
    output_topic    : topic to publish on (default: /teleop/raw_cmd)

Publishes:
    /teleop/raw_cmd  (vehicle_interfaces/VehicleCmd)
"""

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleCmd


class OpenLoopNode(Node):

    def __init__(self):
        super().__init__('open_loop_node')

        # ---- Parameters (all editable from launch file) ----
        self.declare_parameter('velocity', 0.0)
        self.declare_parameter('heading', 0.0)
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('duration', 0.0)          # seconds, 0 = infinite
        self.declare_parameter('output_topic', '/teleop/raw_cmd')

        self.velocity = self.get_parameter('velocity').value
        self.heading = self.get_parameter('heading').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.duration = self.get_parameter('duration').value
        output_topic = self.get_parameter('output_topic').value

        # ---- Publisher ----
        self.cmd_pub = self.create_publisher(VehicleCmd, output_topic, 10)

        # ---- Timer for periodic publishing ----
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self._timer_callback)

        # ---- Duration tracking ----
        self.start_time = self.get_clock().now()
        self.stopped = False

        # ---- Startup log ----
        self.get_logger().info('=' * 50)
        self.get_logger().info('Open Loop Response Node Started')
        self.get_logger().info(f'  Velocity     : {self.velocity:.2f} m/s')
        self.get_logger().info(f'  Heading      : {self.heading:.1f} deg')
        self.get_logger().info(f'  Publish rate : {self.publish_rate:.1f} Hz')
        self.get_logger().info(f'  Duration     : {"infinite" if self.duration <= 0 else f"{self.duration:.1f} s"}')
        self.get_logger().info(f'  Output topic : {output_topic}')
        self.get_logger().info('=' * 50)

    def _timer_callback(self):
        """Publish the constant open-loop command."""
        # Check duration limit
        if self.duration > 0.0 and not self.stopped:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
            if elapsed >= self.duration:
                self.get_logger().info('Duration elapsed — sending stop command')
                self._publish_stop()
                self.stopped = True
                return

        if self.stopped:
            # Keep publishing zero to ensure vehicle stays stopped
            self._publish_stop()
            return

        msg = VehicleCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.velocity = self.velocity
        msg.heading = self.heading
        self.cmd_pub.publish(msg)

    def _publish_stop(self):
        """Publish a zero-velocity, zero-heading command."""
        msg = VehicleCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.velocity = 0.0
        msg.heading = 0.0
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = OpenLoopNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
