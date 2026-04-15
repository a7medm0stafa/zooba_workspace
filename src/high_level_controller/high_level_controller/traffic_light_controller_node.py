"""
High-Level Controller Node
=====================================================
Subscribes to traffic light and traffic sign states from perception 
and issues autonomous driving commands to the mid-level controller.

Velocity Decision logic (Most Restrictive Wins):
    RED or STOP                     → stop       (0.0 m/s)
    YELLOW/UNKNOWN or SLOW/TURN     → slow down  (slow_velocity)
    GREEN + NO_SIGNAL               → cruise     (cruise_velocity)

Heading Decision logic:
    TURN_LEFT                       → steer left  (+turn_heading)
    TURN_RIGHT                      → steer right (-turn_heading)
    NO_SIGNAL / Other               → straight    (default heading)
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
    """High-level controller that reacts to traffic lights and signs."""

    def __init__(self):
        super().__init__('traffic_light_controller_node')

        # -- Declare parameters ----------------------------------------
        self._declare_parameters()

        # -- Read parameters -------------------------------------------
        self.cruise_velocity = self._p('cruise_velocity')
        self.slow_velocity = self._p('slow_velocity')
        self.heading = self._p('heading')
        self.turn_heading = self._p('turn_heading')
        self.unknown_timeout = self._p('unknown_timeout')
        
        state_topic = self._p('state_topic')
        sign_topic = self._p('sign_topic')
        output_topic = self._p('output_topic')
        publish_rate = self._p('publish_rate')

        # -- State tracking --------------------------------------------
        self.current_light_state = 'UNKNOWN'
        self.last_light_time = time.time()
        
        self.current_sign_state = 'NO_SIGNAL'
        self.last_sign_time = time.time()
        
        self.target_velocity = 0.0       
        self.current_velocity = 0.0      
        self.target_heading = self.heading

        # -- Subscribers -----------------------------------------------
        self.state_sub = self.create_subscription(
            String, state_topic, self._light_callback, 10
        )
        self.sign_sub = self.create_subscription(
            String, sign_topic, self._sign_callback, 10
        )

        # -- Publisher: vehicle command ---------------------------------
        self.cmd_pub = self.create_publisher(VehicleCmd, output_topic, 10)

        # -- Timer for periodic command publishing ----------------------
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self._timer_callback)

        # -- Startup log -----------------------------------------------
        self.get_logger().info('=' * 58)
        self.get_logger().info('High-Level Controller Node Started')
        self.get_logger().info(f'  Light topic     : {state_topic}')
        self.get_logger().info(f'  Sign topic      : {sign_topic}')
        self.get_logger().info(f'  Output topic    : {output_topic}')
        self.get_logger().info(f'  Cruise velocity : {self.cruise_velocity:.2f} m/s')
        self.get_logger().info(f'  Slow velocity   : {self.slow_velocity:.2f} m/s')
        self.get_logger().info(f'  Turn heading    : +/- {self.turn_heading:.1f}°')
        self.get_logger().info('=' * 58)

    def _declare_parameters(self):
        yaml_params = self._load_yaml_params()

        def _d(name, fallback):
            self.declare_parameter(name, yaml_params.get(name, fallback))

        _d('cruise_velocity', 0.6)        
        _d('slow_velocity', 0.3)          
        _d('heading', 0.0)                
        _d('turn_heading', 30.0)          # degrees for left/right turns
        _d('publish_rate', 20.0)          
        _d('state_topic', '/traffic_light/state')
        _d('sign_topic', '/sign/command')
        _d('output_topic', '/teleop/raw_cmd')
        _d('unknown_timeout', 2.0)        

    def _load_yaml_params(self) -> dict:
        try:
            this_file = os.path.abspath(__file__)
            pkg_src_dir = os.path.dirname(os.path.dirname(this_file))
            local_yaml = os.path.join(
                pkg_src_dir, 'config', 'high_level_controller.local.yaml'
            )
            if os.path.isfile(local_yaml):
                with open(local_yaml, 'r') as f:
                    raw = yaml.safe_load(f)
                return raw.get('traffic_light_controller_node', {}).get('ros__parameters', {})
        except Exception:
            pass

        try:
            share_dir = get_package_share_directory('high_level_controller')
            yaml_path = os.path.join(share_dir, 'config', 'high_level_controller.yaml')
            with open(yaml_path, 'r') as f:
                raw = yaml.safe_load(f)
            return raw.get('traffic_light_controller_node', {}).get('ros__parameters', {})
        except Exception:
            return {}

    def _p(self, name):
        return self.get_parameter(name).value

    # ===================================================================
    # Callbacks
    # ===================================================================

    def _light_callback(self, msg: String):
        state = msg.data.strip().upper()
        if state not in ('RED', 'YELLOW', 'GREEN', 'UNKNOWN'):
            state = 'UNKNOWN'
        self.current_light_state = state
        self.last_light_time = time.time()

    def _sign_callback(self, msg: String):
        state = msg.data.strip().upper()
        valid_signs = ('STOP', 'SLOW_DOWN', 'TURN_LEFT', 'TURN_RIGHT', 'NO_SIGNAL')
        if state not in valid_signs:
            state = 'NO_SIGNAL'
        self.current_sign_state = state
        self.last_sign_time = time.time()

    # ===================================================================
    # Decision Logic
    # ===================================================================

    def _timer_callback(self):
        now = time.time()

        # 1. Handle Timeouts (if perception nodes crash/freeze)
        if (now - self.last_light_time) > self.unknown_timeout:
            eff_light = 'UNKNOWN'
        else:
            eff_light = self.current_light_state

        if (now - self.last_sign_time) > self.unknown_timeout:
            eff_sign = 'NO_SIGNAL'
        else:
            eff_sign = self.current_sign_state

        # 2. Velocity Logic (Most Restrictive Wins)
        if eff_light == 'RED' or eff_sign == 'STOP':
            self.target_velocity = 0.0
        elif eff_light in ('YELLOW', 'UNKNOWN') or eff_sign in ('SLOW_DOWN', 'TURN_LEFT', 'TURN_RIGHT'):
            # It's safer to slow down while turning
            self.target_velocity = self.slow_velocity
        elif eff_light == 'GREEN':
            self.target_velocity = self.cruise_velocity
        else:
            self.target_velocity = 0.0 # Fail-safe

        # 3. Heading Logic (ROS Standard: + is Left, - is Right)
        if eff_sign == 'TURN_LEFT':
            self.target_heading = self.turn_heading
        elif eff_sign == 'TURN_RIGHT':
            self.target_heading = -self.turn_heading
        else:
            self.target_heading = self.heading # Straight

        # 4. Smooth velocity transition (Ramp)
        ramp_rate = 1.0  # m/s per second
        dt = 1.0 / self._p('publish_rate')
        max_change = ramp_rate * dt

        diff = self.target_velocity - self.current_velocity
        if abs(diff) > max_change:
            self.current_velocity += max_change if diff > 0 else -max_change
        else:
            self.current_velocity = self.target_velocity

        # 5. Publish
        msg = VehicleCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.velocity = self.current_velocity
        msg.heading = float(self.target_heading)
        self.cmd_pub.publish(msg)

        # 6. Terminal log
        light_icon = {'RED': '🔴', 'YELLOW': '🟡', 'GREEN': '🟢', 'UNKNOWN': '⚪'}.get(eff_light, '?')
        sign_icon = {'STOP': '🛑', 'SLOW_DOWN': '⚠️', 'TURN_LEFT': '⬅️', 'TURN_RIGHT': '➡️', 'NO_SIGNAL': '➖'}.get(eff_sign, '?')

        self.get_logger().info(
            f'[HLC] {light_icon}{eff_light:7s} | {sign_icon}{eff_sign:10s} || '
            f'v={self.current_velocity:.2f}m/s | h={self.target_heading:.0f}°'
        )

    def destroy_node(self):
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