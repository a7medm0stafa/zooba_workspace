"""
Simulation Bridge Node
=======================
Bridges the unified VehicleCmd interface to the Gazebo Ackermann
steering vehicle's native Float64 topics, and publishes simulated
encoder feedback (VehicleFeedback).

Localization (VehicleState) is now handled by the localization package:
  - ground_truth_node  (sim debugging: Gazebo world-frame pose)
  - odometry_node      (realistic: IMU + encoder dead-reckoning)

Subscribes:
    /vehicle/cmd       (vehicle_interfaces/VehicleCmd)
    /joint_states      (sensor_msgs/JointState)       — wheel velocities + steering

Publishes:
    /steering_angle    (std_msgs/Float64)  — radians (+left, -right)
    /velocity          (std_msgs/Float64)  — m/s
    /vehicle/feedback  (vehicle_interfaces/VehicleFeedback) — simulated encoder feedback

Note: The Gazebo vehicle_controller uses the convention that positive
steering_angle = left turn and positive velocity = forward.
Our VehicleCmd uses positive heading = right, so we negate the heading
when converting to the Gazebo convention.
"""

import math

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleCmd, VehicleFeedback
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState


class SimBridgeNode(Node):

    def __init__(self):
        super().__init__('sim_bridge_node')

        # ---- Parameters ----
        self.declare_parameter('input_topic', '/vehicle/cmd')
        self.declare_parameter('steering_topic', '/steering_angle')
        self.declare_parameter('velocity_topic', '/velocity')
        self.declare_parameter('feedback_topic', '/vehicle/feedback')
        self.declare_parameter('wheel_radius', 0.033)
        self.declare_parameter('publish_rate', 20.0)

        input_topic = self.get_parameter('input_topic').value
        steering_topic = self.get_parameter('steering_topic').value
        velocity_topic = self.get_parameter('velocity_topic').value
        feedback_topic = self.get_parameter('feedback_topic').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        publish_rate = self.get_parameter('publish_rate').value

        # ---- State ----
        self.velocity = 0.0
        self.steering_angle = 0.0

        # Simulated encoder ticks
        self.sim_ticks = 0
        self.wheel_circumference = 2.0 * math.pi * self.wheel_radius

        # ---- Subscriber: VehicleCmd ----
        self.cmd_sub = self.create_subscription(
            VehicleCmd, input_topic, self._cmd_callback, 10)

        # ---- Subscriber: JointStates (wheel velocities) ----
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_callback, 10)

        # ---- Publishers (Gazebo native topics) ----
        self.steering_pub = self.create_publisher(Float64, steering_topic, 10)
        self.velocity_pub = self.create_publisher(Float64, velocity_topic, 10)

        # ---- Publisher (encoder feedback) ----
        self.feedback_pub = self.create_publisher(VehicleFeedback, feedback_topic, 10)

        # ---- Timer for feedback publishing ----
        self.timer = self.create_timer(1.0 / publish_rate, self._feedback_timer_callback)

        self.get_logger().info('=' * 55)
        self.get_logger().info('Simulation Bridge Node Started')
        self.get_logger().info(f'  Input       : {input_topic} (VehicleCmd)')
        self.get_logger().info(f'  Steering    : {steering_topic} (Float64, rad)')
        self.get_logger().info(f'  Velocity    : {velocity_topic} (Float64, m/s)')
        self.get_logger().info(f'  Feedback out: {feedback_topic} (VehicleFeedback)')
        self.get_logger().info(f'  Wheel radius: {self.wheel_radius:.4f} m')
        self.get_logger().info(f'  NOTE: VehicleState is published by the localization package')
        self.get_logger().info('=' * 55)

    def _cmd_callback(self, msg: VehicleCmd):
        """Convert VehicleCmd to Gazebo-native Float64 topics."""
        # Convert heading (degrees, +right) → steering_angle (radians, +left)
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
                dt = 0.01  # joint_states typically at 100Hz
                delta_rad = avg_omega * dt
                delta_ticks = int(delta_rad / (2.0 * math.pi) * 1968)
                self.sim_ticks += delta_ticks

            if front_left_steer_idx is not None:
                self.steering_angle = msg.position[front_left_steer_idx]

        except (IndexError, AttributeError):
            pass

    def _feedback_timer_callback(self):
        """Publish simulated VehicleFeedback at fixed rate."""
        fb = VehicleFeedback()
        fb.actual_velocity = self.velocity
        fb.actual_rpm = (self.velocity / self.wheel_circumference) * 60.0 if self.wheel_circumference > 0 else 0.0
        fb.encoder_ticks = self.sim_ticks
        self.feedback_pub.publish(fb)


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
