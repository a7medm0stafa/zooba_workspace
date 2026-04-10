# Zooba — Autonomous Perception and Action System (1:10 Scale Vehicle)

This repository contains the full autonomous stack for a 1:10 scale vehicle:
perception, mid-level control, low-level actuation, and Gazebo simulation.

## System Architecture

```
                     ┌─────────────────────────────────┐
                     │      Mid-Level Controller        │
                     │                                  │
  ┌──────────┐       │  teleop_keyboard_node             │
  │Perception│       │    ↓  /teleop/raw_cmd             │
  │  (sign   ├──────►│  nonholonomic_constraints_node    │
  │detection)│future │    ↓  /vehicle/cmd                │
  └──────────┘       └────┬────────────────┬────────────┘
                          │                │
               ┌──────────▼──┐     ┌───────▼─────────────┐
               │  Low-Level  │     │    Simulation        │
               │  Controller │     │                      │
               │  (serial →  │     │  sim_bridge_node     │
               │   Arduino)  │     │    ↓ /steering_angle │
               └─────────────┘     │    ↓ /velocity       │
                                   │  Gazebo Ackermann    │
                                   │  Vehicle Model       │
                                   └─────────────────────┘
```

**Data flow:**
1. `teleop_keyboard_node` reads keyboard, publishes raw commands on `/teleop/raw_cmd`
2. `nonholonomic_constraints_node` enforces Ackermann kinematics (rate limiting, steering/velocity bounds), publishes on `/vehicle/cmd`
3. Both **low-level controller** (Arduino serial) and **simulation** (Gazebo bridge) subscribe to `/vehicle/cmd`

## Repository Structure

```
zooba_workspace/
├── src/
│   ├── vehicle_interfaces/       # Custom ROS 2 messages
│   │   └── msg/
│   │       ├── VehicleCmd.msg           # velocity + heading command
│   │       ├── VehicleConstraints.msg   # constraint diagnostics
│   │       └── VehicleFeedback.msg      # encoder feedback
│   │
│   ├── mid_level_controller/     # Teleop + constraint enforcement
│   │   ├── mid_level_controller/
│   │   │   ├── teleop_keyboard_node.py
│   │   │   └── nonholonomic_constraints_node.py
│   │   ├── config/
│   │   │   └── vehicle_constraints.yaml
│   │   └── launch/
│   │       ├── teleop.launch.py
│   │       └── mid_level_controller.launch.py
│   │
│   ├── low_level_controller/     # Serial bridge to Arduino
│   │   ├── low_level_controller/
│   │   │   └── low_level_controller_node.py
│   │   └── launch/
│   │       └── low_level_controller.launch.py
│   │
│   ├── perception/               # Camera-based sign detection
│   │   └── perception/
│   │       └── nodes/
│   │           ├── sign_detection_node.py
│   │           └── vehicle_actuator_node.py
│   │
│   └── zooba_simulation/         # Gazebo simulation
│       ├── zooba_simulation/
│       │   └── sim_bridge_node.py
│       ├── external/
│       │   └── gazebo_ackermann_steering_vehicle/  (git submodule)
│       └── launch/
│           ├── simulation.launch.py
│           └── full_sim.launch.py
│
└── firmware/                     # Arduino firmware
```

## ROS 2 Topics

| Topic | Message Type | Publisher | Subscribers |
|-------|-------------|-----------|-------------|
| `/teleop/raw_cmd` | `VehicleCmd` | teleop_keyboard_node | nonholonomic_constraints_node |
| `/vehicle/cmd` | `VehicleCmd` | nonholonomic_constraints_node | low_level_controller_node, sim_bridge_node |
| `/vehicle/feedback` | `VehicleFeedback` | low_level_controller_node | — |
| `/vehicle/constraints` | `VehicleConstraints` | nonholonomic_constraints_node | — |
| `/steering_angle` | `Float64` | sim_bridge_node | vehicle_controller (Gazebo) |
| `/velocity` | `Float64` | sim_bridge_node | vehicle_controller (Gazebo) |

## Software Requirements

- Ubuntu 24.04
- ROS 2 Jazzy Jalisco
- Gazebo Harmonic (for simulation)
- Python 3.12
- OpenCV, NumPy, PySerial

Additional ROS 2 packages for simulation:
```bash
sudo apt install -y \
  ros-jazzy-ros2-controllers \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-ros-gz \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-joint-state-publisher \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-xacro \
  ros-jazzy-joy
```

## Build Instructions

```bash
cd /home/ahmed/zooba_workspace
source /opt/ros/jazzy/setup.bash

# Initialize submodules (first time only)
git submodule update --init --recursive

# Build all packages
colcon build
source install/setup.bash
```

## Run Instructions

### Full Simulation (Teleop + Gazebo)

One command to launch everything:
```bash
ros2 launch zooba_simulation full_sim.launch.py
```

This starts:
- Gazebo with the Ackermann vehicle model
- Simulation bridge node
- Keyboard teleop (opens in xterm window)
- Non-holonomic constraints enforcement

### Simulation Only (no teleop)

```bash
ros2 launch zooba_simulation simulation.launch.py
```

Then publish commands manually:
```bash
ros2 topic pub /vehicle/cmd vehicle_interfaces/msg/VehicleCmd \
  "{velocity: 1.0, heading: 10.0}"
```

### Teleop Only (for real vehicle)

Terminal 1 — Mid-level controller:
```bash
ros2 launch mid_level_controller mid_level_controller.launch.py
```

Terminal 2 — Low-level controller:
```bash
ros2 launch low_level_controller low_level_controller.launch.py
```

### Teleop Keyboard Controls

| Key | Action |
|-----|--------|
| `W` / `↑` | Increase velocity |
| `S` / `↓` | Decrease velocity |
| `A` / `←` | Steer left |
| `D` / `→` | Steer right |
| `Space` | Emergency stop |
| `Q` | Quit |

## Non-Holonomic Constraints

The `nonholonomic_constraints_node` enforces:
- **Velocity clamping**: `|v| ≤ max_velocity` (default: 2.0 m/s)
- **Steering clamping**: `|δ| ≤ max_steering_angle` (default: 35°)
- **Velocity rate limiting**: smooth acceleration/deceleration
- **Steering rate limiting**: smooth steering transitions
- **Minimum turning radius**: `R_min = wheelbase / tan(max_steering_angle)`

Parameters are configured in `config/vehicle_constraints.yaml`.

## Arduino Serial Interface

`low_level_controller_node` sends frames: `<direction>,<pwm>,<servo_angle>\n`
- Direction: `1` = forward, `0` = reverse
- PWM: `0–255`
- Servo: angle in degrees

Default serial port: `/dev/ttyACM0` at `115200` baud.

## Project Scope

End-to-end autonomous behavior on a 1:10 scale vehicle combining:
- Onboard visual perception (camera)
- Mid-level control with non-holonomic constraint enforcement
- Embedded inference/processing (Raspberry Pi 4B)
- Real-time command execution (actuation interface)
- Gazebo simulation for development and testing
