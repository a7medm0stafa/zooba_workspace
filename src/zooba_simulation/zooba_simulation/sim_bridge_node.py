"""
Simulation Bridge Node (Extended)
==================================
Bridges the unified VehicleCmd interface to the Gazebo Ackermann
steering vehicle's native Float64 topics, AND extracts ground-truth
vehicle state from Gazebo to publish VehicleState.

Subscribes to:
    /vehicle/cmd       (vehicle_interfaces/VehicleCmd)
    /joint_states      (sensor_msgs/JointState)       — wheel velocities + steering
    /model/pose        (geometry_msgs/PoseStamped)     — world-frame model pose from Gazebo

Publishes:
    /steering_angle    (std_msgs/Float64)  — radians (+left, -right)
    /velocity          (std_msgs/Float64)  — m/s
    /vehicle/state     (vehicle_interfaces/VehicleState) — ground truth state
    /vehicle/feedback  (vehicle_interfaces/VehicleFeedback) — simulated encoder feedback

Note: The Gazebo vehicle_controller uses the convention that positive
steering_angle = left turn and positive velocity = forward.
Our VehicleCmd uses positive heading = right, so we negate the heading
when converting to the Gazebo convention.
"""

import math

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleCmd, VehicleState, VehicleFeedback
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped


class SimBridgeNode(Node):

    def __init__(self):
        super().__init__('sim_bridge_node')

        # ---- Parameters ----
        self.declare_parameter('input_topic', '/vehicle/cmd')
        self.declare_parameter('steering_topic', '/steering_angle')
        self.declare_parameter('velocity_topic', '/velocity')
        self.declare_parameter('state_topic', '/vehicle/state')
        self.declare_parameter('feedback_topic', '/vehicle/feedback')
        self.declare_parameter('pose_topic', '/model/pose')
        self.declare_parameter('wheel_radius', 0.04)
        self.declare_parameter('wheelbase', 0.22)
        self.declare_parameter('publish_rate', 20.0)

        input_topic = self.get_parameter('input_topic').value
        steering_topic = self.get_parameter('steering_topic').value
        velocity_topic = self.get_parameter('velocity_topic').value
        state_topic = self.get_parameter('state_topic').value
        feedback_topic = self.get_parameter('feedback_topic').value
        pose_topic = self.get_parameter('pose_topic').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.wheelbase = self.get_parameter('wheelbase').value
        publish_rate = self.get_parameter('publish_rate').value

        # ---- State ----
        self.velocity = 0.0
        self.steering_angle = 0.0
        self.yaw_rate = 0.0

        # Pose from Gazebo model pose (ground truth, world frame)
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0
        self.pose_received = False

        # Dead-reckoning fallback (when no pose available)
        self.dr_x = 0.0
        self.dr_y = 0.0
        self.dr_yaw = 0.0

        # Simulated encoder ticks
        self.sim_ticks = 0
        self.wheel_circumference = 2.0 * math.pi * self.wheel_radius

        # ---- Subscriber: VehicleCmd ----
        self.cmd_sub = self.create_subscription(
            VehicleCmd,
            input_topic,
            self._cmd_callback,
            10
        )

        # ---- Subscriber: JointStates (wheel velocities) ----
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )

        # ---- Subscriber: Model Pose (ground truth from Gazebo, world frame) ----
        self.pose_sub = self.create_subscription(
            PoseStamped,
            pose_topic,
            self._pose_callback,
            10
        )

        # ---- Publishers (Gazebo native topics) ----
        self.steering_pub = self.create_publisher(Float64, steering_topic, 10)
        self.velocity_pub = self.create_publisher(Float64, velocity_topic, 10)

        # ---- Publishers (state feedback) ----
        self.state_pub = self.create_publisher(VehicleState, state_topic, 10)
        self.feedback_pub = self.create_publisher(VehicleFeedback, feedback_topic, 10)

        # ---- Timer for state publishing ----
        self.last_update_time = self.get_clock().now()
        self.timer = self.create_timer(1.0 / publish_rate, self._state_timer_callback)

        self.get_logger().info('=' * 55)
        self.get_logger().info('Simulation Bridge Node Started (Extended)')
        self.get_logger().info(f'  Input       : {input_topic} (VehicleCmd)')
        self.get_logger().info(f'  Steering    : {steering_topic} (Float64, rad)')
        self.get_logger().info(f'  Velocity    : {velocity_topic} (Float64, m/s)')
        self.get_logger().info(f'  Pose input  : {pose_topic} (PoseStamped, world frame)')
        self.get_logger().info(f'  State out   : {state_topic} (VehicleState)')
        self.get_logger().info(f'  Feedback out: {feedback_topic} (VehicleFeedback)')
        self.get_logger().info(f'  Wheel radius: {self.wheel_radius:.4f} m')
        self.get_logger().info(f'  Wheelbase   : {self.wheelbase:.3f} m')
        self.get_logger().info('=' * 55)

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

                # Simulated encoder ticks
                avg_omega = (omega_l + omega_r) / 2.0
                # ticks per interval (approximate)
                dt = 0.01  # joint_states typically at 100Hz
                delta_rad = avg_omega * dt
                delta_ticks = int(delta_rad / (2.0 * math.pi) * 1968)
                self.sim_ticks += delta_ticks

            if front_left_steer_idx is not None:
                self.steering_angle = msg.position[front_left_steer_idx]

        except (IndexError, AttributeError):
            pass

    def _pose_callback(self, msg: PoseStamped):
        """Receive model pose in world frame from Gazebo (via ros_gz_bridge)."""
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
                f'[SimBridge] First world-frame pose received: '
                f'({self.pose_x:.2f}, {self.pose_y:.2f}) yaw={math.degrees(self.pose_yaw):.1f}°')
        self.pose_received = True

    def _state_timer_callback(self):
        """Publish VehicleState and VehicleFeedback at fixed rate."""
        now = self.get_clock().now()
        dt = (now - self.last_update_time).nanoseconds * 1e-9
        self.last_update_time = now

        if dt <= 0.0 or dt > 1.0:
            dt = 0.05

        # Compute yaw rate
        if abs(self.steering_angle) > 1e-4 and abs(self.velocity) > 0.01:
            turning_radius = self.wheelbase / math.tan(abs(self.steering_angle))
            self.yaw_rate = self.velocity / turning_radius
            if self.steering_angle < 0:
                self.yaw_rate = -self.yaw_rate
        else:
            self.yaw_rate = 0.0

        # Use Gazebo ground truth if available, otherwise dead-reckoning
        if self.pose_received:
            x = self.pose_x
            y = self.pose_y
            yaw = self.pose_yaw
        else:
            # Dead-reckoning fallback
            self.dr_yaw += self.yaw_rate * dt
            self.dr_x += self.velocity * math.cos(self.dr_yaw) * dt
            self.dr_y += self.velocity * math.sin(self.dr_yaw) * dt
            x = self.dr_x
            y = self.dr_y
            yaw = self.dr_yaw

        # Publish VehicleState
        state = VehicleState()
        state.header.stamp = now.to_msg()
        state.header.frame_id = 'world'
        state.x = x
        state.y = y
        state.yaw = yaw
        state.velocity = self.velocity
        state.yaw_rate = self.yaw_rate
        state.steering_angle = self.steering_angle
        self.state_pub.publish(state)

        # Publish simulated VehicleFeedback (for nodes that need it)
        fb = VehicleFeedback()
        fb.actual_velocity = self.velocity
        fb.actual_rpm = (self.velocity / self.wheel_circumference) * 60.0 if self.wheel_circumference > 0 else 0.0
        fb.encoder_ticks = self.sim_ticks
        self.feedback_pub.publish(fb)

        # Log (throttled)
        src = 'GZ' if self.pose_received else 'DR'
        self.get_logger().info(
            f'[SimBridge:{src}] pos=({x:.2f},{y:.2f}) '
            f'yaw={math.degrees(yaw):.1f}° '
            f'v={self.velocity:.3f} '
            f'steer={math.degrees(self.steering_angle):.1f}°',
            throttle_duration_sec=2.0
        )


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
