"""
Odometry Node — Dead-Reckoning from Encoder + IMU
==================================================
Computes the vehicle pose (x, y, yaw) and velocity from encoder ticks
and IMU heading data, then publishes a unified VehicleState message.

Two source modes:
  - "hardware": subscribes to /vehicle/feedback (encoder) + /vehicle/imu (IMU)
  - "simulation": subscribes to /joint_states (Gazebo wheel velocities)

Publishes:
    /vehicle/state  (vehicle_interfaces/VehicleState)
"""

import math

import rclpy
from rclpy.node import Node
from vehicle_interfaces.msg import VehicleFeedback, ImuData, VehicleState
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
import tf2_ros


class OdometryNode(Node):

    def __init__(self):
        super().__init__('odometry_node')

        # ---- Parameters ----
        self.declare_parameter('wheelbase', 0.22)           # m
        self.declare_parameter('wheel_radius', 0.033)       # m
        self.declare_parameter('encoder_cpr', 5904)         # counts per output shaft revolution
        self.declare_parameter('use_imu_heading', True)     # use IMU yaw vs encoder-only
        self.declare_parameter('source', 'hardware')        # "hardware" or "simulation"
        self.declare_parameter('feedback_topic', '/vehicle/feedback')
        self.declare_parameter('imu_topic', '/vehicle/imu')
        self.declare_parameter('state_topic', '/vehicle/state')
        self.declare_parameter('publish_rate', 20.0)        # Hz

        self.wheelbase = self.get_parameter('wheelbase').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.encoder_cpr = self.get_parameter('encoder_cpr').value
        self.use_imu_heading = self.get_parameter('use_imu_heading').value
        self.source = self.get_parameter('source').value
        feedback_topic = self.get_parameter('feedback_topic').value
        imu_topic = self.get_parameter('imu_topic').value
        state_topic = self.get_parameter('state_topic').value
        publish_rate = self.get_parameter('publish_rate').value

        # ---- Precompute ----
        self.wheel_circumference = 2.0 * math.pi * self.wheel_radius
        self.meters_per_tick = self.wheel_circumference / self.encoder_cpr

        # ---- State ----
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0          # rad
        self.velocity = 0.0     # m/s
        self.yaw_rate = 0.0     # rad/s
        self.steering_angle = 0.0  # rad

        # Encoder tracking
        self.last_ticks = None
        self.last_velocity = 0.0

        # IMU heading
        self.imu_yaw_deg = 0.0
        self.imu_yaw_initialized = False
        self.imu_yaw_offset = 0.0

        # ---- Publisher & TF ----
        self.state_pub = self.create_publisher(VehicleState, state_topic, 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ---- Subscribers (based on source mode) ----
        if self.source == 'hardware':
            self.feedback_sub = self.create_subscription(
                VehicleFeedback, feedback_topic, self._feedback_callback, 10)
            self.imu_sub = self.create_subscription(
                ImuData, imu_topic, self._imu_callback, 10)
        else:
            self.joint_sub = self.create_subscription(
                JointState, '/joint_states', self._joint_state_callback, 10)

        # ---- Timer ----
        self.last_update_time = self.get_clock().now()
        self.timer = self.create_timer(1.0 / publish_rate, self._timer_callback)

        self.get_logger().info('=' * 50)
        self.get_logger().info('Odometry Node Started')
        self.get_logger().info(f'  Source         : {self.source}')
        self.get_logger().info(f'  Wheelbase      : {self.wheelbase:.3f} m')
        self.get_logger().info(f'  Wheel radius   : {self.wheel_radius:.4f} m')
        self.get_logger().info(f'  Encoder CPR    : {self.encoder_cpr}')
        self.get_logger().info(f'  Use IMU heading: {self.use_imu_heading}')
        self.get_logger().info(f'  Output topic   : {state_topic}')
        self.get_logger().info('=' * 50)

    # ==================== Hardware Callbacks ====================

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] to completely mirror Gazebo"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _feedback_callback(self, msg: VehicleFeedback):
        """Process encoder feedback for dead-reckoning."""
        self.last_velocity = msg.actual_velocity

        if self.last_ticks is None:
            self.last_ticks = msg.encoder_ticks
            return

        # Delta ticks since last reading
        delta_ticks = msg.encoder_ticks - self.last_ticks
        self.last_ticks = msg.encoder_ticks

        # Distance traveled
        distance = delta_ticks * self.meters_per_tick

        # Update velocity from feedback
        self.velocity = msg.actual_velocity

        # Update pose (dead-reckoning)
        if self.use_imu_heading and self.imu_yaw_initialized:
            # Use IMU yaw directly
            yaw_rad = math.radians(self.imu_yaw_deg - self.imu_yaw_offset)
            self.yaw = self._normalize_angle(yaw_rad)
        # else: yaw stays at previous value (will drift without IMU)

        self.x += distance * math.cos(self.yaw)
        self.y += distance * math.sin(self.yaw)

    def _imu_callback(self, msg: ImuData):
        """Process IMU data for heading estimation."""
        if not self.imu_yaw_initialized:
            # The MPU6050 gyroscope often has a power-on spike that integrates into a false initial yaw.
            # We must wait for the Arduino complementary filter to settle before capturing the zero-offset.
            # Wait 2.5 seconds after node startup before trusting the IMU.
            now = self.get_clock().now()
            # self.last_update_time was initialized at startup, we can use it to measure elapsed time.
            # Actually better to use a dedicated variable, but since we didn't declare one, we will use a static attribute approach or check time.
            if not hasattr(self, '_start_time'):
                self._start_time = now
            
            elapsed = (now - self._start_time).nanoseconds * 1e-9
            if elapsed < 2.5:
                # Still settling. Force raw yaw to match offset so result is exactly zero.
                self.imu_yaw_deg = msg.yaw
                self.imu_yaw_offset = msg.yaw
                return

            self.imu_yaw_offset = msg.yaw
            self.imu_yaw_initialized = True
            self.get_logger().info(f"IMU zero-offset captured: {self.imu_yaw_offset} degrees")

        self.imu_yaw_deg = msg.yaw
        self.yaw_rate = msg.gyro_z  # rad/s around Z axis

    # ==================== Simulation Callback ====================

    def _joint_state_callback(self, msg: JointState):
        """Extract vehicle state from Gazebo joint_states."""
        try:
            # Find rear wheel joint velocities
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
                # Average rear wheel angular velocity → linear velocity
                omega_l = msg.velocity[rear_left_idx]
                omega_r = msg.velocity[rear_right_idx]
                self.velocity = ((omega_l + omega_r) / 2.0) * self.wheel_radius

            if front_left_steer_idx is not None:
                self.steering_angle = msg.position[front_left_steer_idx]

        except (IndexError, AttributeError) as e:
            self.get_logger().warn(f'Joint state parse error: {e}', throttle_duration_sec=5.0)

    # ==================== Timer ====================

    def _timer_callback(self):
        """Publish the current vehicle state at fixed rate."""
        now = self.get_clock().now()
        dt = (now - self.last_update_time).nanoseconds * 1e-9
        self.last_update_time = now

        if dt <= 0.0 or dt > 1.0:
            dt = 0.05

        # For simulation mode, do simple dead-reckoning from velocity + steering
        if self.source == 'simulation':
            if abs(self.steering_angle) > 1e-4:
                turning_radius = self.wheelbase / math.tan(self.steering_angle)
                self.yaw_rate = self.velocity / turning_radius
            else:
                self.yaw_rate = 0.0

            self.yaw += self.yaw_rate * dt
            self.x += self.velocity * math.cos(self.yaw) * dt
            self.y += self.velocity * math.sin(self.yaw) * dt

        # Publish state
        state = VehicleState()
        state.header.stamp = now.to_msg()
        state.header.frame_id = 'odom'
        state.x = self.x
        state.y = self.y
        state.yaw = self.yaw
        state.velocity = self.velocity
        state.yaw_rate = self.yaw_rate
        state.steering_angle = self.steering_angle
        self.state_pub.publish(state)

        # Broadcast TF for Digital Twin Mirroring (RViz/Gazebo)
        t = TransformStamped()
        t.header.stamp = state.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        
        # Calculate quaternion from yaw manually to avoid external dependencies
        cy = math.cos(self.yaw * 0.5)
        sy = math.sin(self.yaw * 0.5)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = sy
        t.transform.rotation.w = cy
        
        self.tf_broadcaster.sendTransform(t)


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
