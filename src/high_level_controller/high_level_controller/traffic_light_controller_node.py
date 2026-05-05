"""
High-Level Controller Node
=====================================================
Subscribes to traffic light and traffic sign states from perception,
maintains stateful velocity and turning logic, and dynamically updates
the mid-level controllers (Stanley lateral control and PI speed control)
via ROS 2 Parameter Clients.

Velocity State Logic:
    RED or STOP                     → STOP state
    YELLOW or SLOW_DOWN             → SLOW state
    GREEN                           → FAST state
    (Maintains current state if NO_SIGNAL or UNKNOWN)

Turning State Logic:
    TURN_LEFT                       → Computes target_yaw = current_yaw + 90
    TURN_RIGHT                      → Computes target_yaw = current_yaw - 90
    (Locks turning state until abs(current_yaw - target_yaw) < 2.0 degrees)
"""

import os
import yaml
import time
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vehicle_interfaces.msg import VehicleCmd, VehicleState
from ament_index_python.packages import get_package_share_directory
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient


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
        self.unknown_timeout = self._p('unknown_timeout')
        
        state_topic = self._p('state_topic')
        sign_topic = self._p('sign_topic')
        vehicle_state_topic = self._p('vehicle_state_topic')
        output_topic = self._p('output_topic')
        publish_rate = self._p('publish_rate')

        # -- Parameter Clients for Mid-Level Controllers ---------------
        self.lat_client = AsyncParameterClient(self, 'lateral_control_node')
        self.speed_client = AsyncParameterClient(self, 'speed_control_node')

        # -- State tracking --------------------------------------------
        self.current_light_state = 'UNKNOWN'
        self.last_light_time = time.time()
        
        self.current_sign_state = 'NO_SIGNAL'
        self.last_sign_time = time.time()
        
        self.current_yaw = 0.0

        # State machine variables
        self.velocity_state = 'FAST'     # 'FAST', 'SLOW', 'STOP'
        self.turning_state = 'STRAIGHT'  # 'STRAIGHT', 'TURNING_LEFT', 'TURNING_RIGHT'
        self.target_yaw = 0.0
        
        # Legacy tracking for auto_cmd topic
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
        self.vehicle_state_sub = self.create_subscription(
            VehicleState, vehicle_state_topic, self._vehicle_state_callback, 10
        )

        # -- Publisher: vehicle command (legacy/fallback) --------------
        self.cmd_pub = self.create_publisher(VehicleCmd, output_topic, 10)

        # -- Timer for periodic command publishing ----------------------
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self._timer_callback)

        # -- Startup log -----------------------------------------------
        self.get_logger().info('=' * 58)
        self.get_logger().info('High-Level Controller Node Started (Stateful)')
        self.get_logger().info(f'  Light topic     : {state_topic}')
        self.get_logger().info(f'  Sign topic      : {sign_topic}')
        self.get_logger().info(f'  Vehicle state   : {vehicle_state_topic}')
        self.get_logger().info(f'  Output topic    : {output_topic}')
        self.get_logger().info(f'  Cruise velocity : {self.cruise_velocity:.2f} m/s')
        self.get_logger().info(f'  Slow velocity   : {self.slow_velocity:.2f} m/s')
        self.get_logger().info('=' * 58)

    def _declare_parameters(self):
        yaml_params = self._load_yaml_params()

        def _d(name, fallback):
            self.declare_parameter(name, yaml_params.get(name, fallback))

        _d('cruise_velocity', 0.6)        
        _d('slow_velocity', 0.3)          
        _d('heading', 0.0)                
        _d('turn_heading', 30.0)          # Unused now, keeping for yaml compat
        _d('publish_rate', 20.0)          
        _d('state_topic', '/traffic_light/state')
        _d('sign_topic', '/sign/command')
        _d('vehicle_state_topic', '/vehicle/state')
        _d('output_topic', '/teleop/auto_cmd')
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

    def _vehicle_state_callback(self, msg: VehicleState):
        self.current_yaw = msg.yaw

    # ===================================================================
    # Helper Math
    # ===================================================================
    
    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-π, π]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

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

        # 2. Velocity State Machine
        if eff_light == 'RED' or eff_sign == 'STOP':
            self.velocity_state = 'STOP'
        elif eff_light == 'YELLOW' or eff_sign == 'SLOW_DOWN':
            self.velocity_state = 'SLOW'
        elif eff_light == 'GREEN':
            self.velocity_state = 'FAST'
        
        # Determine target velocity based on state
        if self.velocity_state == 'STOP':
            target_speed = 0.0
        elif self.velocity_state == 'SLOW':
            target_speed = self.slow_velocity
        else: # 'FAST'
            target_speed = self.cruise_velocity

        # Asynchronously update speed controller parameter
        if self.speed_client.services_are_ready():
            self.speed_client.set_parameters_async([
                Parameter('desired_speed', Parameter.Type.DOUBLE, float(target_speed))
            ])

        # 3. Turning State Machine
        if self.turning_state == 'STRAIGHT':
            if eff_sign == 'TURN_LEFT':
                self.turning_state = 'TURNING_LEFT'
                # +90 degrees = +pi/2 (positive is left in ROS)
                self.target_yaw = self._normalize_angle(self.current_yaw + math.pi/2.0)
            elif eff_sign == 'TURN_RIGHT':
                self.turning_state = 'TURNING_RIGHT'
                # -90 degrees = -pi/2
                self.target_yaw = self._normalize_angle(self.current_yaw - math.pi/2.0)
            
            # Straight path uses base heading parameter
            target_heading_deg = self.heading
        else:
            # We are currently in a turning state. Check for completion.
            heading_error = self._normalize_angle(self.target_yaw - self.current_yaw)
            
            # If within 2 degrees of target yaw, turn is complete
            if abs(math.degrees(heading_error)) < 2.0:
                self.turning_state = 'STRAIGHT'
                target_heading_deg = self.heading
            else:
                target_heading_deg = math.degrees(self.target_yaw)

        # Asynchronously update lateral controller parameter
        if self.lat_client.services_are_ready():
            self.lat_client.set_parameters_async([
                Parameter('desired_heading', Parameter.Type.DOUBLE, float(target_heading_deg))
            ])

        # 4. Smooth velocity transition (Ramp) for auto_cmd topic
        ramp_rate = 1.0  # m/s per second
        dt = 1.0 / self._p('publish_rate')
        max_change = ramp_rate * dt

        diff = target_speed - self.current_velocity
        if abs(diff) > max_change:
            self.current_velocity += max_change if diff > 0 else -max_change
        else:
            self.current_velocity = target_speed

        self.target_heading = target_heading_deg

        # 5. Publish to auto_cmd (legacy support/arbiter fallback)
        msg = VehicleCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.velocity = self.current_velocity
        msg.heading = float(self.target_heading)
        self.cmd_pub.publish(msg)

        # 6. Terminal log
        light_icon = {'RED': '🔴', 'YELLOW': '🟡', 'GREEN': '🟢', 'UNKNOWN': '⚪'}.get(eff_light, '?')
        sign_icon = {'STOP': '🛑', 'SLOW_DOWN': '⚠️', 'TURN_LEFT': '⬅️', 'TURN_RIGHT': '➡️', 'NO_SIGNAL': '➖'}.get(eff_sign, '?')

        hdg_err_deg = math.degrees(self._normalize_angle(self.target_yaw - self.current_yaw))
        
        self.get_logger().info(
            f'[HLC] {light_icon}{eff_light:7s} | {sign_icon}{eff_sign:10s} || '
            f'Vel: {self.velocity_state} ({self.current_velocity:.2f}m/s) | '
            f'Turn: {self.turning_state} (tg={math.degrees(self.target_yaw):.0f}° err={hdg_err_deg:.0f}°) -> hdg={self.target_heading:.0f}°',
            throttle_duration_sec=0.5
        )

    def destroy_node(self):
        self.get_logger().info('Shutting down — sending stop commands...')
        
        # Stop controllers
        if self.speed_client.services_are_ready():
            self.speed_client.set_parameters_async([Parameter('desired_speed', Parameter.Type.DOUBLE, 0.0)])
        
        # Stop fallback arbiter
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