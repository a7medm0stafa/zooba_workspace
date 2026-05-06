# Autonomous Navigation Controller Updates: Summary of Changes

This document outlines all the architectural changes, bug fixes, and parameter tuning applied to the Ackermann vehicle's path planning and lateral control nodes to achieve stable, collision-free, and precise track navigation.

---

## 1. Path Planning Node (`path_planner_node.py`)

### A. Decoupling Lateral and Longitudinal Targets
**The Problem:** Originally, the planner computed a single "lookahead point" (e.g., 0.20m ahead of the car) and sent it to both the speed controller and the lateral controller. Because the lateral controller was aiming at a point *inside* the upcoming curve while the car was still on the straight, the car steered too early, causing it to aggressively cut corners.
**The Fix:** We decoupled the targets:
- **Speed Target:** Still uses the standard lookahead point (0.20m) so the vehicle correctly anticipates curves and slows down *before* entering them.
- **Lateral Target:** Now computes a target point extremely close to the car to trace the geometry exactly without cutting.

### B. Virtual Front Axle (Lateral Lookahead)
**The Problem:** By using the *exact* closest point on the path (0m lookahead) for lateral control, the car wouldn't start steering until it was physically deep inside the curve. Due to inertia and steering delay, this caused the car to swing wide (understeer) and overshoot the exit heading.
**The Fix:** We implemented a small **10cm lateral lookahead**. Because our localization tracks the center/rear of the vehicle, this 10cm offset acts as a "virtual front axle." It gives the Stanley controller just enough anticipation to turn into the corner smoothly without cutting it or overshooting the exit.

---

## 2. Lateral Control Node (`lateral_control_node.py`)

### A. Extended Stanley Implementation
**The Change:** We fully validated and hardened the Extended Stanley Controller. Unlike Pure Pursuit (which acts like a rubber band pulling the car toward a point), Stanley explicitly minimizes both **Heading Error** (angle to the path) and **Cross-Track Error** (perpendicular distance to the path), resulting in vastly superior lane-centering performance.

### B. Dynamic Reference Point Tracking
**The Change:** We refactored the Stanley controller to use the dynamic `desired_x` and `desired_y` coordinates sent by the path planner for every update.
**Why It Matters:** Previously, the controller might have been relying on a static or less frequent target. By calculating the Cross-Track Error (CTE) relative to the exact `(desired_x, desired_y)` point on the spline at every time step, the vehicle can track the path with millimeter precision even during complex maneuvers like obstacle avoidance.

### C. Eradicating "Derivative Kick" (Yaw Damping)
**The Problem:** The controller used a derivative term on the heading error (`d_error/dt`) to dampen oscillations. However, when the path entered a curve, the target heading (`desired_heading`) would suddenly jump. Taking the derivative of this sudden jump caused a massive artificial spike in the control signal (known as "Derivative Kick"). This caused the car to panic and steer hard into the wall.
**The Fix:** We modified the controller to damp the *process variable* rather than the *error*. Instead of differentiating the error, we now use the vehicle's actual physical `yaw_rate` (read directly from the `VehicleState` message). This completely eliminates the derivative kick, resulting in smooth, stable damping through curves.

### C. Steering Polarity Alignment
**The Fix:** We verified the Gazebo Ackermann plugin expects positive angles for turning left and negative angles for turning right. We ensured `invert_steering_output = True` was consistently applied so the Stanley controller's internal math maps perfectly to the physical joints.

---

## 3. Configuration & Launch Architecture

### A. Launch File Parameter Overrides
**The Problem:** You manually tuned the Python script parameters for strict lane keeping (`k_stanley = 5.0`, `k_heading = 3.0`), but the car was still drifting. We discovered that `closed_loop_sim_track.launch.py` was hard-coding and overriding these values with weak defaults (`1.2` and `1.0`).
**The Fix:** We updated the `DeclareLaunchArgument` defaults in the launch file to match your aggressive tuning:
- `k_stanley: 5.0` (Pulls the car tightly to the center of the lane)
- `k_heading: 3.0` (Quickly aligns the car with the path direction)
- `k_d_heading: 0.3` (Smoothly damps rotation using the new `yaw_rate` fix)

### B. Configuration Cleanup
**The Change:** Cleaned up `path_planner_config.yaml` to remove deprecated Pure Pursuit parameters (wheelbase, max steering, invert steering) since those constraints are now properly handled downstream in the mid/low-level controllers.

---

## Summary of Controller Pipeline
1. **Path Planner** creates the cubic spline and searches for a speed lookahead (to slow down) and a 10cm lateral lookahead (to steer).
2. **Lateral Controller (Stanley)** reads the lateral target, computes the exact geometric cross-track error, compares the car's yaw to the path's yaw, and damps the movement using the car's physical yaw rate.
3. **Speed Controller (PI)** reads the lookahead velocity and adjusts the throttle to maintain the target speed.
4. **Control Merger** combines these into a single command for the Ackermann Kinematics node.
