"""
Traffic Light Controller Node (High-Level Controller)
=====================================================
Subscribes to traffic light state from perception and issues autonomous
driving commands to the mid-level controller.

Decision logic:
    RED     → stop       (velocity = 0.0 m/s)
    YELLOW  → slow down  (velocity = slow_velocity)
    GREEN   → cruise     (velocity = cruise_velocity)
    UNKNOWN → slow down  (velocity = slow_velocity, cautious default)

The node publishes VehicleCmd on /teleop/raw_cmd — the same topic the
teleop nodes use — so it slots in transparently as the command source
for the nonholonomic constraints node.

Subscribes:
    /traffic_light/state  (std_msgs/String)

Publishes:
    /teleop/raw_cmd       (vehicle_interfaces/VehicleCmd)
"""

import os
import yaml
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vehicle_interfaces.msg import VehicleCmd
from ament_index_python.packages import get_package_share_directory


class TrafficLightControllerNode(Node):
    """High-level controller that reacts to traffic light perception."""

    # ===================================================================
    # Initialisation
    # ===================================================================

    def __init__(self):
        super().__init__('traffic_light_controller_node')

        # -- Declare parameters ----------------------------------------
        self._declare_parameters()

        # -- Read parameters -------------------------------------------
        self.cruise_velocity = self._p('cruise_velocity')
        self.slow_velocity = self._p('slow_velocity')
        self.heading = self._p('heading')
        self.unknown_timeout = self._p('unknown_timeout')
        state_topic = self._p('state_topic')
        output_topic = self._p('output_topic')
        publish_rate = self._p('publish_rate')

        # -- State -----------------------------------------------------
        self.current_light_state = 'UNKNOWN'
        self.last_state_time = time.time()
        self.target_velocity = 0.0       # What decision logic wants
        self.current_velocity = 0.0      # Smoothed output

        # -- Subscriber: traffic light state ---------------------------
        self.state_sub = self.create_subscription(
            String,
            state_topic,
            self._state_callback,
            10
        )

        # -- Publisher: vehicle command ---------------------------------
        self.cmd_pub = self.create_publisher(VehicleCmd, output_topic, 10)

        # -- Timer for periodic command publishing ----------------------
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self._timer_callback)

        # -- Startup log -----------------------------------------------
        self.get_logger().info('=' * 58)
        self.get_logger().info('Traffic Light Controller Node Started')
        self.get_logger().info(f'  State topic     : {state_topic}')
        self.get_logger().info(f'  Output topic    : {output_topic}')
        self.get_logger().info(f'  Cruise velocity : {self.cruise_velocity:.2f} m/s')
        self.get_logger().info(f'  Slow velocity   : {self.slow_velocity:.2f} m/s')
        self.get_logger().info(f'  Heading         : {self.heading:.1f}°')
        self.get_logger().info(f'  Publish rate    : {publish_rate:.0f} Hz')
        self.get_logger().info(f'  Unknown timeout : {self.unknown_timeout:.1f} s')
        self.get_logger().info('=' * 58)

    # ===================================================================
    # Parameter declaration
    # ===================================================================

    def _declare_parameters(self):
        """Declare all configurable ROS2 parameters.

        Values are loaded from config/high_level_controller.yaml when
        available, with hard-coded fallback defaults.
        """
        yaml_params = self._load_yaml_params()

        def _d(name, fallback):
            self.declare_parameter(name, yaml_params.get(name, fallback))

        _d('cruise_velocity', 0.6)        # m/s — speed on GREEN
        _d('slow_velocity', 0.3)          # m/s — speed on YELLOW / UNKNOWN
        _d('heading', 0.0)                # degrees — fixed straight ahead
        _d('publish_rate', 20.0)          # Hz
        _d('state_topic', '/traffic_light/state')
        _d('output_topic', '/teleop/raw_cmd')
        _d('unknown_timeout', 2.0)        # seconds before treating silence as UNKNOWN

    def _load_yaml_params(self) -> dict:
        """Load parameter values from config YAML.

        Resolution order:
            1. Local override: <workspace>/src/high_level_controller/config/
               high_level_controller.local.yaml  (gitignored, no rebuild)
            2. Installed share: <install>/share/high_level_controller/config/
               high_level_controller.yaml  (requires rebuild)

        Returns:
            Flat dict of parameter_name → value, or empty dict on failure.
        """
        # --- 1. Try local override (no rebuild, gitignored) ---------------
        try:
            this_file = os.path.abspath(__file__)
            pkg_src_dir = os.path.dirname(os.path.dirname(this_file))
            local_yaml = os.path.join(
                pkg_src_dir, 'config', 'high_level_controller.local.yaml'
            )
            if os.path.isfile(local_yaml):
                with open(local_yaml, 'r') as f:
                    raw = yaml.safe_load(f)
                params = (
                    raw
                    .get('traffic_light_controller_node', {})
                    .get('ros__parameters', {})
                )
                self.get_logger().info(
                    f'Loaded LOCAL override config from {local_yaml}'
                )
                return params
        except Exception as e:
            self.get_logger().warn(
                f'Error reading local override config: {e}'
            )

        # --- 2. Fallback: installed share directory -----------------------
        try:
            share_dir = get_package_share_directory('high_level_controller')
            yaml_path = os.path.join(
                share_dir, 'config', 'high_level_controller.yaml'
            )
            with open(yaml_path, 'r') as f:
                raw = yaml.safe_load(f)
            params = (
                raw
                .get('traffic_light_controller_node', {})
                .get('ros__parameters', {})
            )
            self.get_logger().info(f'Loaded parameters from {yaml_path}')
            return params
        except Exception as e:
            self.get_logger().warn(
                f'Could not load YAML config, using defaults: {e}'
            )
            return {}

    # ===================================================================
    # Convenience
    # ===================================================================

    def _p(self, name):
        """Shorthand to retrieve a parameter value."""
        return self.get_parameter(name).value

    # ===================================================================
    # Subscriber callback
    # ===================================================================

    def _state_callback(self, msg: String):
        """Handle incoming traffic light state from perception."""
        state = msg.data.strip().upper()
        if state not in ('RED', 'YELLOW', 'GREEN', 'UNKNOWN'):
            self.get_logger().warn(f'Unexpected traffic light state: {state}')
            state = 'UNKNOWN'

        self.current_light_state = state
        self.last_state_time = time.time()

    # ===================================================================
    # Timer callback — decision + publish
    # ===================================================================

    def _timer_callback(self):
        """Decide velocity based on traffic light state and publish cmd."""
        now = time.time()

        # -- If no state received for too long, treat as UNKNOWN ----------
        if (now - self.last_state_time) > self.unknown_timeout:
            effective_state = 'UNKNOWN'
        else:
            effective_state = self.current_light_state

        # -- Decision logic -----------------------------------------------
        if effective_state == 'RED':
            self.target_velocity = 0.0
        elif effective_state == 'YELLOW':
            self.target_velocity = self.slow_velocity
        elif effective_state == 'GREEN':
            self.target_velocity = self.cruise_velocity
        else:  # UNKNOWN
            self.target_velocity = self.slow_velocity

        # -- Smooth velocity transition -----------------------------------
        # Ramp towards target at ~1.0 m/s² (configurable via timer rate)
        ramp_rate = 1.0  # m/s per second
        dt = 1.0 / self._p('publish_rate')
        max_change = ramp_rate * dt

        diff = self.target_velocity - self.current_velocity
        if abs(diff) > max_change:
            self.current_velocity += max_change if diff > 0 else -max_change
        else:
            self.current_velocity = self.target_velocity

        # -- Publish VehicleCmd -------------------------------------------
        msg = VehicleCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.velocity = self.current_velocity
        msg.heading = self.heading
        self.cmd_pub.publish(msg)

        # -- Terminal log (single-line, overwritten) ----------------------
        state_icon = {
            'RED': '🔴', 'YELLOW': '🟡', 'GREEN': '🟢', 'UNKNOWN': '⚪'
        }.get(effective_state, '?')

        self.get_logger().info(
            f'[HLC] {state_icon} {effective_state:7s} | '
            f'target={self.target_velocity:.2f} m/s | '
            f'cmd={self.current_velocity:.2f} m/s | '
            f'heading={self.heading:.1f}°'
        )

    # ===================================================================
    # Cleanup
    # ===================================================================

    def destroy_node(self):
        """Send stop command on shutdown."""
        self.get_logger().info('Shutting down — sending stop command...')
        msg = VehicleCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.velocity = 0.0
        msg.heading = 0.0
        self.cmd_pub.publish(msg)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = TrafficLightControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
