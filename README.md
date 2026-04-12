# Zooba — Autonomous 1:10 Scale Vehicle

## Project Overview

Welcome to the **Zooba Autonomous Perception and Action System**. The ultimate goal of this project is to build a fully autonomous 1:10 scale vehicle capable of making real-time, intelligent decisions such as dynamic obstacle avoidance, path planning, and environment mapping. 

The project is actively being developed. **Coming soon:** We will be integrating a complete *onboard localization system* to give the vehicle spatial awareness within its environment.

## Work Done So Far

We have currently established the foundation for both the physical hardware platform and a 1:1 digital twin in a simulated environment:
- **Simulation Environment:** A fully functional Gazebo Harmonic simulation implementing an Ackermann steering model.
- **Mid-Level Control:** A modular control system that handles raw teleoperation inputs and enforces non-holonomic constraints (limiting max speeds, angles, and acceleration rates) tailored to the vehicle's physical limits.
- **Teleoperation:** Support for both keyboard control and Bluetooth Joystick (PS4/PS5) teleoperation.
- **Low-Level Hardware Control:** A serial bridge communicating seamlessly between the Raspberry Pi 4B (running ROS 2) and an Arduino Uno.
- **Actuation:** Firmware interpreting serial commands to actuate a JGA-370 DC motor via an L298N H-Bridge and an MG995 steering servo.

---

## Hardware Setup & Connections

The current hardware stack relies on the following major components:
- **Compute:** Raspberry Pi 4B (Main ROS 2 brain) connected via USB Serial to an Arduino Uno (Low-level hardware controller).
- **Power:** 12V 5A DC Power Supply.
- **Drive Engine:** 12V JGA-370 DC Motor with encoder, driven by an L298N H-Bridge.
- **Steering:** MG995 Servo Motor.

**Wiring Notes:**
* The 12V supply powers the L298N motor driver.
* **Important:** The MG995 servo runs strictly on 6V. Ensure a step-down Buck Converter (e.g., LM2596) is used to step the 12V down to 6V for the servo. Running the servo entirely on 12V will severely damage the Arduino via backward voltage leaks.
* The L298N `ENA`, `IN1`, and `IN2` logic pins connect to the Arduino's PWM digital pins (as defined in the Arduino firmware).
* A common ground is shared between the Arduino, L298N, Servo, and the Buck Converter/Power Supply.

---

## Software Architecture

```text
                     ┌─────────────────────────────────┐
                     │      Mid-Level Controller       │
                     │                                 │
  ┌──────────┐       │  teleop_keyboard_node / joy     │
  │Perception│       │    ↓  /teleop/raw_cmd           │
  │  (sign   ├──────►│  nonholonomic_constraints_node  │
  │detection)│future │    ↓  /vehicle/cmd              │
  └──────────┘       └────┬────────────────┬───────────┘
                          │                │
               ┌──────────▼──┐     ┌───────▼─────────────┐
               │  Low-Level  │     │    Simulation       │
               │  Controller │     │                     │
               │  (serial →  │     │  sim_bridge_node    │
               │   Arduino)  │     │    ↓ /steering_angle│
               └─────────────┘     │    ↓ /velocity      │
                                   │  Gazebo Ackermann   │
                                   │  Vehicle Model      │
                                   └─────────────────────┘
```

---

## Software Setup Instructions

**Prerequisites:**
- Ubuntu 24.04
- ROS 2 Jazzy Jalisco
- Gazebo Harmonic
- Python 3.12 (with OpenCV, NumPy, PySerial)

**Install required ROS 2 dependencies:**
```bash
sudo apt update
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

**Build the Workspace:**
```bash
cd ~/zooba_workspace
source /opt/ros/jazzy/setup.bash

# Ensure external submodules are checked out
git submodule update --init --recursive

# Build and source
colcon build
source install/setup.bash
```

---

## How to Use the Vehicle

### 1. Running the Digital Twin (Simulation)

You can launch the full simulation (Gazebo + Controller stack) with one command. By default, it expects keyboard input.

**Keyboard Control:**
```bash
ros2 launch zooba_simulation full_sim.launch.py
```
*(This opens a small `xterm` window. Ensure the window is focused to capture `W`,`A`,`S`,`D` and `Space` commands).*

**Joystick Control (PS4 / PS5):**
Connect your joystick via Bluetooth to the computer and run:
```bash
ros2 launch zooba_simulation full_sim.launch.py teleop_type:=joy
```

---

### 2. Running on the Physical Hardware

When running on the real car, launch the Low-Level serial bridge alongside the Mid-Level controller.

**Terminal 1 — Mid-Level Controller (Joy/Keyboard + Constraints):**
```bash
# For Joystick
ros2 launch mid_level_controller mid_level_controller.launch.py teleop_type:=joy

# OR For Keyboard
ros2 launch mid_level_controller mid_level_controller.launch.py 
```

**Terminal 2 — Low-Level Controller (Send commands to Arduino):**
```bash
ros2 launch low_level_controller low_level_controller.launch.py
```

### Teleoperation Controls Mapping

**Keyboard:**
* `W` / `S`: Accelerate Forward / Backward
* `A` / `D`: Steer Left / Right
* `Space`: Emergency Stop
* `Q`: Quit

**Joystick (Default Layout):**
* `R2` / `RT`: Accelerate
* `L2` / `LT`: Brake / Reverse
* `Left Stick (Horizontal)`: Steering
* `X` / `A`: Emergency Stop
* `Circle` / `B`: Release Emergency Stop

---

## Configurations

If the car feels sluggish or you need to unlock higher speeds, edit the constraints parameter file: `/src/mid_level_controller/config/vehicle_constraints.yaml`. 
* **`max_velocity_rate`**: Determines how punchy the acceleration is.
* **`max_steering_rate`**: Determines how fast the servo sweeps from center to max.
* **`max_velocity`**: Top forward/reverse speed limits. (Ensure `teleop.launch.py` matches these limits).
