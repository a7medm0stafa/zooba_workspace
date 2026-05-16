"""
Ground Truth Localization Node (Simulation Only)
==================================================
Provides the vehicle's exact world-frame pose from Gazebo's PosePublisher
plugin. This is used for controller debugging and validation — it gives
perfect state with no drift or noise.

In the real system, this node is NOT used. The odometry_node replaces it.

Subscribes:
    /model/ackermann_steering_vehicle/pose  (geometry_msgs/PoseStamped) — Gazebo model pose
    /joint_states                            (sensor_msgs/JointState)   — wheel velocities

Publishes:
    /vehicle/state_gt     (vehicle_interfaces/VehicleState) — ground truth state
"""

import math

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleState
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped


class GroundTruthNode(Node):

    def __init__(self):
        super().__init__('ground_truth_node')

        # ---- Parameters ----
        self.declare_parameter('pose_topic', '/model/ackermann_steering_vehicle/pose')
        self.declare_parameter('state_topic', '/vehicle/state')
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('wheel_radius', 0.033)
        self.declare_parameter('wheelbase', 0.265)

        pose_topic = self.get_parameter('pose_topic').value
        state_topic = self.get_parameter('state_topic').value
        publish_rate = self.get_parameter('publish_rate').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.wheelbase = self.get_parameter('wheelbase').value

        # ---- State ----
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0
        self.pose_received = False

        self.velocity = 0.0
        self.steering_angle = 0.0
        self.yaw_rate = 0.0

        # ---- Subscriber: Model Pose (ground truth from Gazebo, world frame) ----
        self.pose_sub = self.create_subscription(
            PoseStamped, pose_topic, self._pose_callback, 10)

        # ---- Subscriber: JointStates (wheel velocities + steering) ----
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_callback, 10)

        # ---- Publisher ----
        self.state_pub = self.create_publisher(VehicleState, state_topic, 10)

        # ---- Timer ----
        self.timer = self.create_timer(1.0 / publish_rate, self._publish_state)

        self.get_logger().info('=' * 55)
        self.get_logger().info('Ground Truth Node Started (Gazebo World-Frame Pose)')
        self.get_logger().info(f'  Pose topic   : {pose_topic}')
        self.get_logger().info(f'  State output : {state_topic}')
        self.get_logger().info(f'  Rate         : {publish_rate:.0f} Hz')
        self.get_logger().info('=' * 55)

    def _pose_callback(self, msg: PoseStamped):
        """Receive model pose in world frame from Gazebo."""
        t = msg.pose.position
        q = msg.pose.orientation

        self.pose_x = t.x
        self.pose_y = t.y

        # Quaternion to yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.pose_yaw = math.atan2(siny_cosp, cosy_cosp)

        if not self.pose_received:
            self.get_logger().info(
                f'[GroundTruth] First pose: '
                f'({self.pose_x:.2f}, {self.pose_y:.2f}) '
                f'yaw={math.degrees(self.pose_yaw):.1f}°')
        self.pose_received = True

    def _joint_state_callback(self, msg: JointState):
        """Extract wheel velocities and steering angle from Gazebo joint states."""
        try:
            rear_left_idx = None
            rear_right_idx = None
            front_left_steer_idx = None

            for i, name in enumerate(msg.name):
                if name == 'rear_left_wheel_joint':
                    rear_left_idx = i
                elif name == 'rear_right_wheel_joint':
                    rear_right_idx = i
                elif name == 'front_left_steering_joint':
                    front_left_steer_idx = i

            if rear_left_idx is not None and rear_right_idx is not None:
                omega_l = msg.velocity[rear_left_idx]
                omega_r = msg.velocity[rear_right_idx]
                self.velocity = ((omega_l + omega_r) / 2.0) * self.wheel_radius

            if front_left_steer_idx is not None:
                self.steering_angle = msg.position[front_left_steer_idx]

        except (IndexError, AttributeError):
            pass

    def _publish_state(self):
        """Publish ground-truth vehicle state."""
        if not self.pose_received:
            return

        now = self.get_clock().now()

        # Compute yaw rate from steering kinematics
        if abs(self.steering_angle) > 1e-4 and abs(self.velocity) > 0.01:
            turning_radius = self.wheelbase / math.tan(abs(self.steering_angle))
            self.yaw_rate = self.velocity / turning_radius
            if self.steering_angle < 0:
                self.yaw_rate = -self.yaw_rate
        else:
            self.yaw_rate = 0.0

        state = VehicleState()
        state.header.stamp = now.to_msg()
        state.header.frame_id = 'world'
        state.x = self.pose_x
        state.y = self.pose_y
        state.yaw = self.pose_yaw
        state.velocity = self.velocity
        state.yaw_rate = self.yaw_rate
        state.steering_angle = self.steering_angle
        self.state_pub.publish(state)

        # Log (throttled)
        self.get_logger().info(
            f'[GT] pos=({self.pose_x:.2f},{self.pose_y:.2f}) '
            f'yaw={math.degrees(self.pose_yaw):.1f}° '
            f'v={self.velocity:.3f} '
            f'steer={math.degrees(self.steering_angle):.1f}°',
            throttle_duration_sec=2.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
