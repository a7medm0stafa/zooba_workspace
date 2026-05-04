"""
Odometry Node — Dead-Reckoning Localization
=============================================
Estimates the vehicle's pose in the world frame using dead-reckoning
from encoder (distance) and IMU (heading) feedback.

This node mirrors what the real car does: no access to a global frame,
position is estimated purely from sensor integration.

Inputs:
    - Encoder ticks (via VehicleFeedback) → distance traveled
    - IMU yaw (via ImuData)              → heading

The encoder provides incremental distance (Δd per tick), and the IMU
provides the absolute heading (yaw). Pose is integrated as:
    x += Δd * cos(yaw)
    y += Δd * sin(yaw)

For simulation, the sim_bridge publishes simulated VehicleFeedback
(encoder ticks from wheel joint velocities), and a simulated IMU yaw
can be derived from the steering kinematics or from a Gazebo IMU plugin.

Subscribes:
    /vehicle/feedback  (vehicle_interfaces/VehicleFeedback) — encoder ticks
    /imu/data          (vehicle_interfaces/ImuData)         — IMU heading

Publishes:
    /vehicle/state     (vehicle_interfaces/VehicleState)    — estimated pose
"""

import math

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleFeedback, VehicleState, ImuData


class OdometryNode(Node):

    def __init__(self):
        super().__init__('odometry_node')

        # ---- Parameters ----
        self.declare_parameter('feedback_topic', '/vehicle/feedback')
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('state_topic', '/vehicle/state')
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('wheel_radius', 0.04)
        self.declare_parameter('ticks_per_rev', 1968)
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)  # degrees

        feedback_topic = self.get_parameter('feedback_topic').value
        imu_topic = self.get_parameter('imu_topic').value
        state_topic = self.get_parameter('state_topic').value
        publish_rate = self.get_parameter('publish_rate').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.ticks_per_rev = self.get_parameter('ticks_per_rev').value
        initial_x = self.get_parameter('initial_x').value
        initial_y = self.get_parameter('initial_y').value
        initial_yaw_deg = self.get_parameter('initial_yaw').value

        # ---- Encoder constants ----
        self.wheel_circumference = 2.0 * math.pi * self.wheel_radius
        self.meters_per_tick = self.wheel_circumference / self.ticks_per_rev

        # ---- State ----
        self.x = float(initial_x)
        self.y = float(initial_y)
        self.yaw = math.radians(float(initial_yaw_deg))  # rad
        self.velocity = 0.0

        self.prev_ticks = None  # for delta computation
        self.imu_yaw = None     # latest IMU heading (rad)
        self.last_feedback_time = None

        # ---- Subscribers ----
        self.feedback_sub = self.create_subscription(
            VehicleFeedback, feedback_topic, self._feedback_callback, 10)
        self.imu_sub = self.create_subscription(
            ImuData, imu_topic, self._imu_callback, 10)

        # ---- Publisher ----
        self.state_pub = self.create_publisher(VehicleState, state_topic, 10)

        # ---- Timer ----
        self.timer = self.create_timer(1.0 / publish_rate, self._publish_state)

        self.get_logger().info('=' * 55)
        self.get_logger().info('Odometry Node Started (IMU + Encoder Dead-Reckoning)')
        self.get_logger().info(f'  Feedback topic : {feedback_topic}')
        self.get_logger().info(f'  IMU topic      : {imu_topic}')
        self.get_logger().info(f'  State output   : {state_topic}')
        self.get_logger().info(f'  Wheel radius   : {self.wheel_radius:.4f} m')
        self.get_logger().info(f'  Ticks/rev      : {self.ticks_per_rev}')
        self.get_logger().info(f'  m/tick         : {self.meters_per_tick:.6f}')
        self.get_logger().info(f'  Initial pose   : ({self.x:.2f}, {self.y:.2f}) '
                               f'yaw={initial_yaw_deg:.1f}°')
        self.get_logger().info('=' * 55)

    def _feedback_callback(self, msg: VehicleFeedback):
        """Process encoder ticks and velocity from low-level feedback."""
        now = self.get_clock().now()

        current_ticks = msg.encoder_ticks
        self.velocity = msg.actual_velocity

        if self.prev_ticks is not None:
            delta_ticks = current_ticks - self.prev_ticks
            delta_dist = delta_ticks * self.meters_per_tick

            # Use IMU yaw if available, otherwise keep last yaw
            yaw = self.imu_yaw if self.imu_yaw is not None else self.yaw

            # Integrate position
            self.x += delta_dist * math.cos(yaw)
            self.y += delta_dist * math.sin(yaw)
            self.yaw = yaw

        self.prev_ticks = current_ticks
        self.last_feedback_time = now

    def _imu_callback(self, msg: ImuData):
        """Receive IMU heading (yaw in degrees from Arduino complementary filter)."""
        # ImuData.yaw is in degrees from the Arduino
        self.imu_yaw = math.radians(msg.yaw)

    def _publish_state(self):
        """Publish the estimated vehicle state at fixed rate."""
        now = self.get_clock().now()

        state = VehicleState()
        state.header.stamp = now.to_msg()
        state.header.frame_id = 'odom'      # odometry frame (not world)
        state.x = self.x
        state.y = self.y
        state.yaw = self.yaw
        state.velocity = self.velocity
        state.yaw_rate = 0.0
        state.steering_angle = 0.0
        self.state_pub.publish(state)

        # Log (throttled)
        self.get_logger().info(
            f'[Odom] pos=({self.x:.2f},{self.y:.2f}) '
            f'yaw={math.degrees(self.yaw):.1f}° '
            f'v={self.velocity:.3f} '
            f'imu={"OK" if self.imu_yaw is not None else "WAIT"} '
            f'enc={"OK" if self.prev_ticks is not None else "WAIT"}',
            throttle_duration_sec=2.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
