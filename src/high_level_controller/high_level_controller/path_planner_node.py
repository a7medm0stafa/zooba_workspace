"""
Path Planner Node — Cubic Spline Trajectory Generator
======================================================
Generates smooth (x, y, heading, velocity) trajectories for three tracks
using cubic spline interpolation over pre-defined waypoints.

At runtime the node:
    1. Loads waypoints for the selected track
    2. Fits parametric cubic splines  x(s), y(s)
    3. Samples a dense trajectory with heading and curvature-based velocity
    4. Tracks the vehicle along the trajectory, updating the mid-level
       controllers' desired_y, desired_heading, and desired_speed parameters

Subscribes:
    /vehicle/state   (vehicle_interfaces/VehicleState)

Publishes:
    /path_planner/target  (vehicle_interfaces/VehicleState) — current target point (debug)

Updates parameters on:
    /speed_control_node   → desired_speed
    /lateral_control_node → desired_y, desired_heading
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from vehicle_interfaces.msg import VehicleState
from std_msgs.msg import Float64MultiArray


# ======================================================================
# Cubic Spline (natural, clamped-free) — minimal standalone implementation
# ======================================================================

class CubicSpline1D:
    """Compute a 1-D natural cubic spline for data (t[i], y[i])."""

    def __init__(self, t, y):
        n = len(t)
        assert n >= 2, "Need at least 2 points"
        self.t = np.array(t, dtype=float)
        self.y = np.array(y, dtype=float)
        self.n = n

        h = np.diff(self.t)
        # Build tridiagonal system for second derivatives
        A = np.zeros((n, n))
        b = np.zeros(n)
        A[0, 0] = 1.0
        A[-1, -1] = 1.0
        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2.0 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            b[i] = 3.0 * ((self.y[i + 1] - self.y[i]) / h[i] -
                           (self.y[i] - self.y[i - 1]) / h[i - 1])

        self.c = np.linalg.solve(A, b)
        self.a = self.y[:-1].copy()
        self.b = np.zeros(n - 1)
        self.d = np.zeros(n - 1)
        for i in range(n - 1):
            self.b[i] = ((self.y[i + 1] - self.y[i]) / h[i] -
                         h[i] * (2.0 * self.c[i] + self.c[i + 1]) / 3.0)
            self.d[i] = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])
        self.c = self.c[:-1]  # trim to n-1

    def _find_segment(self, t_val):
        idx = int(np.searchsorted(self.t, t_val)) - 1
        idx = max(0, min(idx, self.n - 2))
        return idx

    def __call__(self, t_val):
        i = self._find_segment(t_val)
        dt = t_val - self.t[i]
        return self.a[i] + self.b[i] * dt + self.c[i] * dt**2 + self.d[i] * dt**3

    def derivative(self, t_val):
        """First derivative dy/dt."""
        i = self._find_segment(t_val)
        dt = t_val - self.t[i]
        return self.b[i] + 2.0 * self.c[i] * dt + 3.0 * self.d[i] * dt**2

    def second_derivative(self, t_val):
        """Second derivative d²y/dt²."""
        i = self._find_segment(t_val)
        dt = t_val - self.t[i]
        return 2.0 * self.c[i] + 6.0 * self.d[i] * dt


class CubicSpline2D:
    """Parametric 2-D cubic spline (x(s), y(s)) with arc-length parameter."""

    def __init__(self, x_pts, y_pts):
        # Cumulative chord length as parameter
        dx = np.diff(x_pts)
        dy = np.diff(y_pts)
        ds = np.sqrt(dx**2 + dy**2)
        self.s = np.concatenate([[0.0], np.cumsum(ds)])
        self.total_length = self.s[-1]

        self.sx = CubicSpline1D(self.s, x_pts)
        self.sy = CubicSpline1D(self.s, y_pts)

    def position(self, s):
        return self.sx(s), self.sy(s)

    def heading(self, s):
        dx = self.sx.derivative(s)
        dy = self.sy.derivative(s)
        return math.atan2(dy, dx)

    def curvature(self, s):
        dx = self.sx.derivative(s)
        dy = self.sy.derivative(s)
        ddx = self.sx.second_derivative(s)
        ddy = self.sy.second_derivative(s)
        denom = (dx**2 + dy**2)**1.5
        if abs(denom) < 1e-12:
            return 0.0
        return abs(dx * ddy - dy * ddx) / denom


# ======================================================================
# Track waypoint definitions
# ======================================================================

def _arc_waypoints(cx, cy, r, start_angle, end_angle, n_pts=8,
                    skip_first=True):
    """Generate waypoints along a circular arc.

    Args:
        skip_first: If True, skip the first point to avoid duplicating
                    the preceding straight segment's endpoint.
    """
    angles = np.linspace(start_angle, end_angle, n_pts)
    pts = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]
    return pts[1:] if skip_first else pts


def _deduplicate_waypoints(wps, min_dist=0.005):
    """Remove consecutive near-duplicate waypoints."""
    if not wps:
        return wps
    cleaned = [wps[0]]
    for i in range(1, len(wps)):
        dx = wps[i][0] - cleaned[-1][0]
        dy = wps[i][1] - cleaned[-1][1]
        if math.sqrt(dx*dx + dy*dy) > min_dist:
            cleaned.append(wps[i])
    return cleaned


def get_track_waypoints(track_name, start_x=0.0, start_y=0.0):
    """Return an ordered list of (x, y) waypoints for the given track.

    Track geometry is extracted directly from the .world files:

    Track 1 & 2:
        Walls at Y = ±0.375 (inner edge), track width = 0.75m
        Center divider at Y = 0 (visual only)
        Left lane center:  Y =  0.1875
        Right lane center: Y = -0.1875

    Track 3 (closed rectangular circuit):
        Outer walls:  top Y=2.0, bottom Y=-2.0, right X=2.5, left X=-2.5
        Inner walls:  top Y=1.5, bottom Y=-1.5, right X=2.0, left X=-2.0
        Lane width:   0.5m  (outer - inner)
        Lane center:  0.25m offset from inner wall
        Corner arc centers:   (±1.5, ±1.0)
        Corner inner radius:  0.5m
        Corner outer radius:  1.0m
        Corner lane radius:   0.75m
    """

    if track_name == 'track_1':
        # Straight lane keeping — start at origin, merge into left lane
        # Vehicle starts at (0,0), transitions to Y=0.1875, then drives straight
        return [
            (0.0, 0.0),            # START at origin
            (0.5, 0.0),            # settle straight
            (1.0, 0.0),            # begin merge into left lane
            (1.5, 0.0),         # in left lane
            (2.5, 0.0),
            (5.0, 0.0),
            (7.5, 0.0),
            (10.0, 0.0),
        ], False  # not closed

    elif track_name == 'track_2':
        # Two-lane track with obstacles:
        #   Obstacle 1: X=4, Y=0.1875, blocks left lane (size 0.1 x 0.375)
        #   Obstacle 2: X=8, Y=-0.1875, blocks right lane (size 0.1 x 0.375)
        #
        # Strategy: start at origin, merge into left lane, switch right
        #           before Obs1, switch left before Obs2, finish
        return [
            (0.0,  0.0),           # START at origin
            (0.5,  0.09),          # merge into left lane
            (1.0,  0.1875),        # in left lane
            (1.5,  0.1875),        # anchor
            (2.0,  0.1875),        # begin lane change to right
            (2.75, 0.0),           # mid lane change
            (3.5, -0.1875),        # in right lane before obstacle 1
            (4.0, -0.1875),        # passing obstacle 1
            (4.5, -0.1875),        # anchor right lane
            (5.0, -0.1875),        # cruising right lane
            (5.5, -0.1875),        # begin lane change to left
            (6.5,  0.0),           # mid lane change
            (7.5,  0.1875),        # in left lane before obstacle 2
            (8.0,  0.1875),        # passing obstacle 2
            (8.5,  0.1875),        # anchor left lane
            (9.0,  0.1875),        # past obstacle 2
            (10.0, 0.1875),        # finish
        ], False  # not closed

    elif track_name == 'track_3':
        # Closed rectangular track, COUNTER-CLOCKWISE direction (LEFT turns)
        # Vehicle starts at (0, 0) heading East (θ=0) on the BOTTOM straight
        # No initial_yaw needed — car faces East, IMU starts at 0°
        # Map shifted so BOTTOM straight is at Y=0 (all Y += 1.75 from original)
        # Lane center coordinates:
        #   Bottom straight: Y = 0.0   (start, between outer -0.25 and inner 0.25)
        #   Top straight:    Y = 3.5
        #   Right straight:  X = 2.25  (between inner 2.0 and outer 2.5)
        #   Left straight:   X = -2.25
        # Corner arc centers at (±1.5, 0.75) and (±1.5, 2.75), lane radius = 0.75m

        wps = []

        # Start: middle of bottom straight, heading East (+X)
        wps.append((0.0, 0.0))          # START at origin
        wps.append((1.0, 0.0))          # mid bottom straight
        wps.append((1.5, 0.0))          # end of bottom straight

        # Bottom-right corner: center (1.5, 0.75), R=0.75
        # Arc from -90° to 0° (CCW = increasing angle, turning LEFT / north)
        wps += _arc_waypoints(1.5, 0.75, 0.75,
                              math.radians(-90), math.radians(0), n_pts=8)

        # Right straight: X=2.25, going north from Y=0.75 to Y=2.75
        wps.append((2.25, 1.25))
        wps.append((2.25, 1.75))
        wps.append((2.25, 2.25))
        wps.append((2.25, 2.75))

        # Top-right corner: center (1.5, 2.75), R=0.75
        # Arc from 0° to 90° (CCW, turning LEFT / west)
        wps += _arc_waypoints(1.5, 2.75, 0.75,
                              math.radians(0), math.radians(90), n_pts=8)

        # Top straight: Y=3.5, going west from X=1.5 to X=-1.5
        wps.append((1.0, 3.5))
        wps.append((0.0, 3.5))
        wps.append((-1.0, 3.5))
        wps.append((-1.5, 3.5))

        # Top-left corner: center (-1.5, 2.75), R=0.75
        # Arc from 90° to 180° (CCW, turning LEFT / south)
        wps += _arc_waypoints(-1.5, 2.75, 0.75,
                              math.radians(90), math.radians(180), n_pts=8)

        # Left straight: X=-2.25, going south from Y=2.75 to Y=0.75
        wps.append((-2.25, 2.25))
        wps.append((-2.25, 1.75))
        wps.append((-2.25, 1.25))
        wps.append((-2.25, 0.75))

        # Bottom-left corner: center (-1.5, 0.75), R=0.75
        # Arc from 180° to 270° (CCW, turning LEFT / east)
        wps += _arc_waypoints(-1.5, 0.75, 0.75,
                              math.radians(180), math.radians(270), n_pts=8)

        # Back to bottom straight, heading East
        wps.append((-1.0, 0.0))
        wps.append((0.0, 0.0))          # BACK TO START (close the loop)

        # Remove any near-duplicate consecutive points
        wps = _deduplicate_waypoints(wps)

        return wps, True  # closed loop

    else:
        raise ValueError(f"Unknown track: {track_name}")


# ======================================================================
# Path Planner ROS 2 Node
# ======================================================================

class PathPlannerNode(Node):

    def __init__(self):
        super().__init__('path_planner_node')

        # ---- Parameters ----
        self.declare_parameter('track_name', 'track_1')
        self.declare_parameter('cruise_speed', 0.15)         # m/s
        self.declare_parameter('curve_speed', 0.10)           # m/s  (speed in tight curves)
        self.declare_parameter('lookahead_distance', 0.20)    # m
        self.declare_parameter('waypoint_tolerance', 0.15)    # m
        self.declare_parameter('trajectory_resolution', 0.02) # m
        self.declare_parameter('goal_tolerance', 0.25)        # m  (final waypoint)
        self.declare_parameter('max_lateral_accel', 0.3)      # m/s² (for curvature speed limit)
        self.declare_parameter('control_rate', 20.0)          # Hz
        self.declare_parameter('state_topic', '/vehicle/state')
        self.declare_parameter('start_delay', 5.0)            # seconds before first command

        self.track_name = self.get_parameter('track_name').value
        self.cruise_speed = self.get_parameter('cruise_speed').value
        self.curve_speed = self.get_parameter('curve_speed').value
        self.lookahead = self.get_parameter('lookahead_distance').value
        self.wp_tol = self.get_parameter('waypoint_tolerance').value
        self.traj_res = self.get_parameter('trajectory_resolution').value
        self.goal_tol = self.get_parameter('goal_tolerance').value
        self.max_lat_a = self.get_parameter('max_lateral_accel').value
        control_rate = self.get_parameter('control_rate').value
        state_topic = self.get_parameter('state_topic').value
        self.start_delay = self.get_parameter('start_delay').value

        # ---- Build trajectory ----
        self.get_logger().info(f'Building trajectory for track: {self.track_name}')
        waypoints, self.is_closed = get_track_waypoints(self.track_name)
        self.n_waypoints = len(waypoints)

        x_pts = [wp[0] for wp in waypoints]
        y_pts = [wp[1] for wp in waypoints]

        self.spline = CubicSpline2D(x_pts, y_pts)
        self._sample_trajectory()

        self.get_logger().info(
            f'Trajectory: {self.n_traj_pts} points, '
            f'length={self.spline.total_length:.2f}m, '
            f'closed={self.is_closed}'
        )

        # ---- State ----
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_velocity = 0.0
        self.state_received = False

        self.current_target_idx = 0
        self.goal_reached = False
        self.started = False
        self.start_time = None

        # ---- Subscriber ----
        self.state_sub = self.create_subscription(
            VehicleState, state_topic, self._state_callback, 10)

        # ---- Publisher (debug target point) ----
        self.target_pub = self.create_publisher(
            VehicleState, '/path_planner/target', 10)

        # Publish the full trajectory for visualization
        self.traj_pub = self.create_publisher(
            Float64MultiArray, '/path_planner/trajectory', 10)

        # ---- Service clients for parameter updates ----
        self.speed_param_client = self.create_client(
            SetParameters, '/speed_control_node/set_parameters')
        self.lateral_param_client = self.create_client(
            SetParameters, '/lateral_control_node/set_parameters')

        # ---- Timer ----
        self.timer = self.create_timer(1.0 / control_rate, self._control_callback)

        # Publish trajectory once after a short delay
        self.create_timer(2.0, self._publish_trajectory_once)
        self._trajectory_published = False

        self.get_logger().info('=' * 60)
        self.get_logger().info('Path Planner Node Started')
        self.get_logger().info(f'  Track           : {self.track_name}')
        self.get_logger().info(f'  Cruise speed    : {self.cruise_speed:.2f} m/s')
        self.get_logger().info(f'  Curve speed     : {self.curve_speed:.2f} m/s')
        self.get_logger().info(f'  Lookahead       : {self.lookahead:.2f} m')
        self.get_logger().info(f'  Goal tolerance  : {self.goal_tol:.2f} m')
        self.get_logger().info(f'  Trajectory pts  : {self.n_traj_pts}')
        self.get_logger().info(f'  Path length     : {self.spline.total_length:.2f} m')
        self.get_logger().info(f'  Closed loop     : {self.is_closed}')
        self.get_logger().info(f'  Start delay     : {self.start_delay:.1f} s')
        self.get_logger().info('=' * 60)

    # ------------------------------------------------------------------
    # Trajectory sampling
    # ------------------------------------------------------------------

    def _sample_trajectory(self):
        """Sample the spline at uniform arc-length intervals."""
        n_samples = max(int(self.spline.total_length / self.traj_res), 10)
        s_values = np.linspace(0, self.spline.total_length, n_samples)

        self.traj_x = []
        self.traj_y = []
        self.traj_heading = []
        self.traj_velocity = []
        self.traj_s = []

        for s in s_values:
            x, y = self.spline.position(s)
            heading = self.spline.heading(s)
            kappa = self.spline.curvature(s)

            # Curvature-based velocity limit:  v <= sqrt(a_lat / kappa)
            if kappa > 1e-3:
                v_curv = math.sqrt(self.max_lat_a / kappa)
                # Use curve_speed for any noticeable curvature (corners have κ ≈ 1.33)
                v = min(self.cruise_speed, v_curv,
                        self.curve_speed if kappa > 0.1 else self.cruise_speed)
            else:
                v = self.cruise_speed

            self.traj_x.append(x)
            self.traj_y.append(y)
            self.traj_heading.append(heading)
            self.traj_velocity.append(v)
            self.traj_s.append(s)

        self.n_traj_pts = len(self.traj_x)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _state_callback(self, msg: VehicleState):
        self.current_x = msg.x
        self.current_y = msg.y
        self.current_yaw = msg.yaw
        self.current_velocity = msg.velocity
        if not self.state_received:
            self.get_logger().info(
                f'[Planner] First state received: '
                f'({msg.x:.2f}, {msg.y:.2f}) yaw={math.degrees(msg.yaw):.1f}°')
            self.state_received = True

    def _publish_trajectory_once(self):
        """Publish the full trajectory array for visualization."""
        if self._trajectory_published:
            return
        msg = Float64MultiArray()
        # Flatten: [x0, y0, h0, v0,  x1, y1, h1, v1, ...]
        data = []
        for i in range(self.n_traj_pts):
            data.extend([
                self.traj_x[i], self.traj_y[i],
                self.traj_heading[i], self.traj_velocity[i]
            ])
        msg.data = data
        self.traj_pub.publish(msg)
        self._trajectory_published = True
        self.get_logger().info(f'Published trajectory ({self.n_traj_pts} pts)')

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def _control_callback(self):
        if not self.state_received:
            return

        if self.goal_reached:
            return

        # Start delay
        if self.start_time is None:
            self.start_time = self.get_clock().now()

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        if elapsed < self.start_delay:
            return

        if not self.started:
            self.started = True
            self.get_logger().info('[Planner] Starting trajectory tracking!')

        # --- Find closest trajectory point ---
        min_dist = float('inf')
        closest_idx = self.current_target_idx
        # Search in a window around current index for efficiency
        search_start = max(0, self.current_target_idx - 5)
        search_end = min(self.n_traj_pts, self.current_target_idx + 50)

        # For closed tracks, also search near the start when close to the end
        if self.is_closed and self.current_target_idx > self.n_traj_pts - 20:
            search_end = self.n_traj_pts

        for i in range(search_start, search_end):
            dx = self.traj_x[i] - self.current_x
            dy = self.traj_y[i] - self.current_y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # --- Compute lookahead index ---
        lookahead_idx = closest_idx
        accumulated_dist = 0.0
        for i in range(closest_idx, min(self.n_traj_pts - 1, closest_idx + 100)):
            dx = self.traj_x[i+1] - self.traj_x[i]
            dy = self.traj_y[i+1] - self.traj_y[i]
            accumulated_dist += math.sqrt(dx*dx + dy*dy)
            if accumulated_dist >= self.lookahead:
                lookahead_idx = i + 1
                break
        else:
            lookahead_idx = min(self.n_traj_pts - 1, closest_idx + 10)

        # Ensure forward progress (never go backwards on the path)
        if lookahead_idx < self.current_target_idx:
            lookahead_idx = self.current_target_idx

        self.current_target_idx = max(closest_idx, self.current_target_idx)

        # --- Compute lateral lookahead index (0.1m) ---
        # This provides a "virtual front axle" to prevent late steering and overshoot
        lateral_lookahead_idx = closest_idx
        accum_dist = 0.0
        for i in range(closest_idx, min(self.n_traj_pts - 1, closest_idx + 50)):
            dx = self.traj_x[i+1] - self.traj_x[i]
            dy = self.traj_y[i+1] - self.traj_y[i]
            accum_dist += math.sqrt(dx*dx + dy*dy)
            if accum_dist >= 0.10: # 10cm lateral lookahead
                lateral_lookahead_idx = i + 1
                break

        # --- Get lateral target state (Stanley target) ---
        lateral_idx = lateral_lookahead_idx
        target_x = self.traj_x[lateral_idx]
        target_y = self.traj_y[lateral_idx]
        target_heading = self.traj_heading[lateral_idx]

        # --- Get speed target state (use speed lookahead point) ---
        speed_idx = min(lookahead_idx, self.n_traj_pts - 1)
        target_velocity = self.traj_velocity[speed_idx]

        # --- Check goal reached ---
        if not self.is_closed:
            # Open track: check distance to final point
            dx_goal = self.traj_x[-1] - self.current_x
            dy_goal = self.traj_y[-1] - self.current_y
            dist_to_goal = math.sqrt(dx_goal**2 + dy_goal**2)
            if dist_to_goal < self.goal_tol:
                self.goal_reached = True
                self.get_logger().info(
                    f'🏁 GOAL REACHED! Distance to goal: {dist_to_goal:.3f}m')
                self._set_speed(0.0)
                return
        else:
            # Closed track: check if we've gone past ~90% of the path
            # AND are back near the start
            progress = self.current_target_idx / self.n_traj_pts
            if progress > 0.85:
                dx_start = self.traj_x[0] - self.current_x
                dy_start = self.traj_y[0] - self.current_y
                dist_to_start = math.sqrt(dx_start**2 + dy_start**2)
                if dist_to_start < self.goal_tol:
                    self.goal_reached = True
                    self.get_logger().info(
                        f'🏁 LAP COMPLETE! Distance to start: {dist_to_start:.3f}m')
                    self._set_speed(0.0)
                    return

        # --- Send commands to mid-level controllers ---
        target_heading_deg = math.degrees(target_heading)
        self._set_lateral(target_x, target_y, target_heading_deg)
        self._set_speed(target_velocity)

        # --- Publish target for debug ---
        target_msg = VehicleState()
        target_msg.header.stamp = self.get_clock().now().to_msg()
        target_msg.header.frame_id = 'planner'
        target_msg.x = target_x
        target_msg.y = target_y
        target_msg.yaw = float(target_heading)
        target_msg.velocity = float(target_velocity)
        self.target_pub.publish(target_msg)

        # --- Log (throttled) ---
        self.get_logger().info(
            f'[Planner] idx={lateral_idx}/{self.n_traj_pts} '
            f'tgt=({target_x:.2f},{target_y:.2f}) '
            f'h={target_heading_deg:.1f}° v={target_velocity:.2f} '
            f'pos=({self.current_x:.2f},{self.current_y:.2f}) '
            f'err={min_dist:.3f}m',
            throttle_duration_sec=1.0
        )

    # ------------------------------------------------------------------
    # Parameter update helpers (async, non-blocking)
    # ------------------------------------------------------------------

    def _set_speed(self, speed: float):
        """Update desired_speed on the speed control node."""
        if not self.speed_param_client.service_is_ready():
            return
        req = SetParameters.Request()
        p = Parameter()
        p.name = 'desired_speed'
        p.value = ParameterValue()
        p.value.type = ParameterType.PARAMETER_DOUBLE
        p.value.double_value = float(speed)
        req.parameters = [p]
        self.speed_param_client.call_async(req)

    def _set_lateral(self, desired_x: float, desired_y: float, desired_heading_deg: float):
        """Update desired_x, desired_y and desired_heading on the lateral control node."""
        if not self.lateral_param_client.service_is_ready():
            return
        req = SetParameters.Request()

        p_x = Parameter()
        p_x.name = 'desired_x'
        p_x.value = ParameterValue()
        p_x.value.type = ParameterType.PARAMETER_DOUBLE
        p_x.value.double_value = float(desired_x)

        p_y = Parameter()
        p_y.name = 'desired_y'
        p_y.value = ParameterValue()
        p_y.value.type = ParameterType.PARAMETER_DOUBLE
        p_y.value.double_value = float(desired_y)

        p_h = Parameter()
        p_h.name = 'desired_heading'
        p_h.value = ParameterValue()
        p_h.value.type = ParameterType.PARAMETER_DOUBLE
        p_h.value.double_value = float(desired_heading_deg)

        req.parameters = [p_x, p_y, p_h]
        self.lateral_param_client.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
