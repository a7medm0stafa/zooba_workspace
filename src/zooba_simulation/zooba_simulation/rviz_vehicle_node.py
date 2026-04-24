"""
RViz Vehicle Visualization Node
=================================
Subscribes to /vehicle/state and publishes visualization markers so
the physical robot can be seen moving in RViz2 in real time.

Publishes:
    /vehicle/viz_markers  (visualization_msgs/MarkerArray)
        - Body box (chassis)
        - Heading arrow
        - Trail breadcrumbs (path history)
    /vehicle/path         (nav_msgs/Path)
        - Full path history for RViz Path display

Also publishes:
    /vehicle/odom_viz     (nav_msgs/Odometry)
        - Standard ROS Odometry message for RViz's Odometry display

Parameters:
    state_topic:   input VehicleState topic  (default: /vehicle/state)
    marker_topic:  output MarkerArray topic   (default: /vehicle/viz_markers)
    path_topic:    output Path topic          (default: /vehicle/path)
    odom_topic:    output Odometry topic      (default: /vehicle/odom_viz)

    # Vehicle dimensions for the marker (meters)
    car_length:    0.30
    car_width:     0.20
    car_height:    0.10

    trail_max_points: 500   max breadcrumbs before recycling
    publish_rate:     20.0  Hz
"""

import math

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleState
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import (
    PoseStamped, Point, Quaternion, Vector3,
    Twist, TwistWithCovariance, PoseWithCovariance, Pose,
)
from std_msgs.msg import ColorRGBA, Header


class RvizVehicleNode(Node):

    def __init__(self):
        super().__init__('rviz_vehicle_node')

        # ---- Parameters ----
        self.declare_parameter('state_topic', '/vehicle/state')
        self.declare_parameter('marker_topic', '/vehicle/viz_markers')
        self.declare_parameter('path_topic', '/vehicle/path')
        self.declare_parameter('odom_topic', '/vehicle/odom_viz')
        self.declare_parameter('car_length', 0.30)
        self.declare_parameter('car_width', 0.20)
        self.declare_parameter('car_height', 0.10)
        self.declare_parameter('trail_max_points', 500)
        self.declare_parameter('publish_rate', 20.0)

        state_topic = self.get_parameter('state_topic').value
        marker_topic = self.get_parameter('marker_topic').value
        path_topic = self.get_parameter('path_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        self.car_length = self.get_parameter('car_length').value
        self.car_width = self.get_parameter('car_width').value
        self.car_height = self.get_parameter('car_height').value
        self.trail_max = self.get_parameter('trail_max_points').value
        publish_rate = self.get_parameter('publish_rate').value

        # ---- State ----
        self.latest_state = None
        self.trail_points = []

        # ---- Subscriber ----
        self.state_sub = self.create_subscription(
            VehicleState, state_topic, self._state_cb, 10)

        # ---- Publishers ----
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        self.path_pub = self.create_publisher(Path, path_topic, 10)
        self.odom_pub = self.create_publisher(Odometry, odom_topic, 10)

        # ---- Timer ----
        self.timer = self.create_timer(1.0 / publish_rate, self._publish_cb)

        self.get_logger().info('=' * 50)
        self.get_logger().info('RViz Vehicle Visualization Started')
        self.get_logger().info(f'  State topic  : {state_topic}')
        self.get_logger().info(f'  Marker topic : {marker_topic}')
        self.get_logger().info(f'  Path topic   : {path_topic}')
        self.get_logger().info(f'  Odom topic   : {odom_topic}')
        self.get_logger().info(f'  Car size     : {self.car_length}×{self.car_width}×{self.car_height} m')
        self.get_logger().info('=' * 50)

    # ---- Helpers ----

    @staticmethod
    def _yaw_to_quat(yaw: float) -> Quaternion:
        """Convert yaw angle (rad) to a Quaternion."""
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw * 0.5)
        q.w = math.cos(yaw * 0.5)
        return q

    # ---- Callbacks ----

    def _state_cb(self, msg: VehicleState):
        self.latest_state = msg

    def _publish_cb(self):
        if self.latest_state is None:
            return

        s = self.latest_state
        stamp = self.get_clock().now().to_msg()
        quat = self._yaw_to_quat(s.yaw)

        # --- Record trail ---
        self.trail_points.append((s.x, s.y))
        if len(self.trail_points) > self.trail_max:
            self.trail_points = self.trail_points[-self.trail_max:]

        # ==================== Marker Array ====================
        markers = MarkerArray()

        # 1) Chassis box
        body = Marker()
        body.header = Header(stamp=stamp, frame_id='odom')
        body.ns = 'vehicle'
        body.id = 0
        body.type = Marker.CUBE
        body.action = Marker.ADD
        body.pose.position.x = s.x
        body.pose.position.y = s.y
        body.pose.position.z = self.car_height / 2.0
        body.pose.orientation = quat
        body.scale = Vector3(x=self.car_length, y=self.car_width, z=self.car_height)
        body.color = ColorRGBA(r=0.0, g=0.6, b=1.0, a=0.85)  # blue
        body.lifetime.sec = 0
        markers.markers.append(body)

        # 2) Heading arrow
        arrow = Marker()
        arrow.header = Header(stamp=stamp, frame_id='odom')
        arrow.ns = 'vehicle'
        arrow.id = 1
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose.position.x = s.x
        arrow.pose.position.y = s.y
        arrow.pose.position.z = self.car_height + 0.02
        arrow.pose.orientation = quat
        arrow.scale = Vector3(x=self.car_length * 1.2, y=0.03, z=0.03)
        arrow.color = ColorRGBA(r=1.0, g=0.3, b=0.0, a=1.0)  # orange
        markers.markers.append(arrow)

        # 3) Velocity text
        txt = Marker()
        txt.header = Header(stamp=stamp, frame_id='odom')
        txt.ns = 'vehicle'
        txt.id = 2
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position.x = s.x
        txt.pose.position.y = s.y
        txt.pose.position.z = self.car_height + 0.15
        txt.scale.z = 0.06  # text height
        txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        txt.text = f'v={s.velocity:.2f} m/s  ψ={math.degrees(s.yaw):.1f}°'
        markers.markers.append(txt)

        # 4) Trail (line strip)
        trail = Marker()
        trail.header = Header(stamp=stamp, frame_id='odom')
        trail.ns = 'trail'
        trail.id = 0
        trail.type = Marker.LINE_STRIP
        trail.action = Marker.ADD
        trail.scale.x = 0.015  # line width
        trail.color = ColorRGBA(r=0.0, g=1.0, b=0.4, a=0.7)  # green trail
        trail.pose.orientation.w = 1.0
        for px, py in self.trail_points:
            trail.points.append(Point(x=px, y=py, z=0.005))
        markers.markers.append(trail)

        self.marker_pub.publish(markers)

        # ==================== Path ====================
        path_msg = Path()
        path_msg.header = Header(stamp=stamp, frame_id='odom')
        for px, py in self.trail_points:
            p = PoseStamped()
            p.header = path_msg.header
            p.pose.position.x = px
            p.pose.position.y = py
            p.pose.position.z = 0.0
            p.pose.orientation.w = 1.0
            path_msg.poses.append(p)
        self.path_pub.publish(path_msg)

        # ==================== Odometry ====================
        odom_msg = Odometry()
        odom_msg.header = Header(stamp=stamp, frame_id='odom')
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose.pose = Pose()
        odom_msg.pose.pose.position.x = s.x
        odom_msg.pose.pose.position.y = s.y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = quat
        odom_msg.twist.twist = Twist()
        odom_msg.twist.twist.linear.x = s.velocity
        odom_msg.twist.twist.angular.z = s.yaw_rate
        self.odom_pub.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RvizVehicleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
