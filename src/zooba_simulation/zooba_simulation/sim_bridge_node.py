"""
Simulation Bridge Node
======================
Bridges the unified VehicleCmd interface to the Gazebo Ackermann
steering vehicle's native Float64 topics.

Subscribes to:
    /vehicle/cmd  (vehicle_interfaces/VehicleCmd)
        - velocity: m/s
        - heading:  degrees from center (+right, -left)

Publishes:
    /steering_angle  (std_msgs/Float64)  — radians (+left, -right)
    /velocity        (std_msgs/Float64)  — m/s

Note: The Gazebo vehicle_controller uses the convention that positive
steering_angle = left turn and positive velocity = forward.
Our VehicleCmd uses positive heading = right, so we negate the heading
when converting to the Gazebo convention.
"""

import math

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleCmd
from std_msgs.msg import Float64


class SimBridgeNode(Node):

    def __init__(self):
        super().__init__('sim_bridge_node')

        # ---- Parameters ----
        self.declare_parameter('input_topic', '/vehicle/cmd')
        self.declare_parameter('steering_topic', '/steering_angle')
        self.declare_parameter('velocity_topic', '/velocity')

        input_topic = self.get_parameter('input_topic').value
        steering_topic = self.get_parameter('steering_topic').value
        velocity_topic = self.get_parameter('velocity_topic').value

        # ---- Subscriber ----
        self.cmd_sub = self.create_subscription(
            VehicleCmd,
            input_topic,
            self._cmd_callback,
            10
        )

        # ---- Publishers (Gazebo native topics) ----
        self.steering_pub = self.create_publisher(Float64, steering_topic, 10)
        self.velocity_pub = self.create_publisher(Float64, velocity_topic, 10)

        self.get_logger().info('=' * 50)
        self.get_logger().info('Simulation Bridge Node Started')
        self.get_logger().info(f'  Input     : {input_topic} (VehicleCmd)')
        self.get_logger().info(f'  Steering  : {steering_topic} (Float64, rad)')
        self.get_logger().info(f'  Velocity  : {velocity_topic} (Float64, m/s)')
        self.get_logger().info('=' * 50)

    def _cmd_callback(self, msg: VehicleCmd):
        """Convert VehicleCmd to Gazebo-native Float64 topics."""
        # Convert heading (degrees, +right) → steering_angle (radians, +left)
        # Negate because our convention: +heading = right, Gazebo: +angle = left
        steering_rad = -math.radians(msg.heading)

        # Velocity passes through directly (both in m/s)
        velocity_mps = msg.velocity

        # Publish steering angle
        steering_msg = Float64()
        steering_msg.data = steering_rad
        self.steering_pub.publish(steering_msg)

        # Publish velocity
        velocity_msg = Float64()
        velocity_msg.data = velocity_mps
        self.velocity_pub.publish(velocity_msg)


def main(args=None):
    rclpy.init(args=args)

    node = SimBridgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
