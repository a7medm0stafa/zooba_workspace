"""
Joystick Teleoperation Node for Ackermann Vehicle
==================================================
Reads sensor_msgs/Joy from /joy and publishes VehicleCmd messages on /teleop/raw_cmd.

Publishes raw (unconstrained) commands. The nonholonomic_constraints_node
downstream will enforce physical limits before forwarding to /vehicle/cmd.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from vehicle_interfaces.msg import VehicleCmd

class TeleopJoyNode(Node):
    def __init__(self):
        super().__init__('teleop_joy_node')

        # ---- Parameters ----
        # Default mappings (Standard gamepad: Left Stick for steering, Right Trigger/Left Trigger for speed)
        # For PS4/Xbox via standard Linux joy:
        # Left Stick X (L/R) = 0
        # Right Trigger (Gas) = 5
        # Left Trigger (Brake) = 2
        # Cross/A button (E-Stop) = 0
        # Circle/B button (Un-E-Stop) = 1
        
        self.declare_parameter('axis_steering', 0)
        self.declare_parameter('axis_forward', 5)  # R2 / RT
        self.declare_parameter('axis_reverse', 2)  # L2 / LT
        self.declare_parameter('button_estop', 0)  # X / A
        self.declare_parameter('button_unestop', 1) # Circle / B
        
        self.declare_parameter('max_velocity', 2.0)
        self.declare_parameter('max_heading', 35.0)
        self.declare_parameter('output_topic', '/teleop/raw_cmd')

        self.axis_steering = self.get_parameter('axis_steering').value
        self.axis_forward = self.get_parameter('axis_forward').value
        self.axis_reverse = self.get_parameter('axis_reverse').value
        self.button_estop = self.get_parameter('button_estop').value
        self.button_unestop = self.get_parameter('button_unestop').value

        self.max_velocity = self.get_parameter('max_velocity').value
        self.max_heading = self.get_parameter('max_heading').value
        output_topic = self.get_parameter('output_topic').value

        # Local state
        self.current_velocity = 0.0
        self.current_heading = 0.0
        self.estop_active = False
        
        # Monitor if triggers have been pressed at least once to avoid zero-initialization bug
        self.fwd_trigger_initialized = False
        self.rev_trigger_initialized = False

        self.cmd_pub = self.create_publisher(VehicleCmd, output_topic, 10)
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)

        self.get_logger().info('Teleop Joy Node started')
        self.get_logger().info(f'  Publishing on: {output_topic}')

    def map_trigger(self, val):
        # Triggers read 1.0 when unpressed, -1.0 when fully pressed.
        # We need 0.0 (unpressed) to 1.0 (fully pressed).
        return (1.0 - val) / 2.0

    def joy_callback(self, msg: Joy):
        # Handle buttons
        if len(msg.buttons) > self.button_estop and msg.buttons[self.button_estop] == 1:
            self.estop_active = True
        elif len(msg.buttons) > self.button_unestop and msg.buttons[self.button_unestop] == 1:
            self.estop_active = False

        if self.estop_active:
            self.publish_cmd(0.0, 0.0)
            self.get_logger().warn('E-STOP ACTIVE! Press Circle/B to reset.', throttle_duration_sec=2.0)
            return

        # Determine if triggers are initialized (workaround for Linux joystick driver 0.0 startup)
        raw_fwd = msg.axes[self.axis_forward] if len(msg.axes) > self.axis_forward else 1.0
        raw_rev = msg.axes[self.axis_reverse] if len(msg.axes) > self.axis_reverse else 1.0

        if raw_fwd != 0.0:
            self.fwd_trigger_initialized = True
        if raw_rev != 0.0:
            self.rev_trigger_initialized = True

        vel_fwd = 0.0
        vel_rev = 0.0

        if self.fwd_trigger_initialized:
            vel_fwd = self.map_trigger(raw_fwd) * self.max_velocity
            
        if self.rev_trigger_initialized:
            vel_rev = self.map_trigger(raw_rev) * self.max_velocity

        self.current_velocity = vel_fwd - vel_rev

        # Steering
        # Left on stick is +1.0, Right is -1.0
        # In this package, positive heading is left, negative is right
        if len(msg.axes) > self.axis_steering:
            self.current_heading = msg.axes[self.axis_steering] * self.max_heading

        self.publish_cmd(self.current_velocity, self.current_heading)

    def publish_cmd(self, velocity, heading):
        cmd = VehicleCmd()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.velocity = velocity
        cmd.heading = heading
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopJoyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
