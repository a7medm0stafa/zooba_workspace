# Autonomous Perception and Action System (1:10 Scale Vehicle)

This repository contains the perception and decision-to-actuation pipeline for a 1:10 scale autonomous vehicle.

The project goal is to use the onboard camera to detect traffic lights and road signs, then command the vehicle to perform the correct behavior in real time.

All embedded processing is intended to run on a **Raspberry Pi 4B** mounted on the vehicle.

## Project Objective

Develop a complete onboard autonomous perception-and-action stack that can:
- detect traffic lights,
- detect traffic signs,
- interpret detected signals,
- issue appropriate control commands to the vehicle.

Target behaviors include (not limited to):
- stop,
- go,
- slow down,
- turn left,
- turn right,
- U-turn,
- and other maneuver commands based on road signs and light states.

## System Architecture

Data flow:

`Onboard Camera -> Perception Node(s) -> Decision/Command Topic(s) -> Actuation Node(s) -> Motor/Steering Controller`

Current implementation in this repository:

`USB Camera -> sign_detection_node -> /vehicle/command (std_msgs/String) -> vehicle_actuator_node -> Arduino (Serial) -> Motor Driver`

ROS 2 nodes currently in this package:
- `sign_detection_node`
  - Captures frames from camera index `0`
  - Applies preprocessing (ROI crop, resize, brightness, CLAHE, blur)
  - Performs HSV color segmentation for red/yellow/green traffic light detection
  - Publishes command strings on `vehicle/command`
- `vehicle_actuator_node`
  - Subscribes to `vehicle/command`
  - Opens serial port `/dev/ttyACM0` at `9600` baud
  - Sends one-letter control codes to Arduino (`R`, `Y`, `G`, `O`)

## Repository Structure

- `src/perception/`
  - ROS 2 package source
  - `perception/nodes/sign_detection_node.py`
  - `perception/nodes/vehicle_actuator_node.py`
- `milestone2/ms2_hardware_demo.ino`
  - Example Arduino firmware used for hardware interfacing tests
- `build/`, `install/`, `log/`
  - Colcon-generated build artifacts

## Software and Hardware Requirements

Software:
- Ubuntu Linux
- ROS 2 (tested in ament Python package workflow)
- Python 3.12 (workspace build output indicates Python 3.12)
- OpenCV (`cv2`)
- NumPy
- PySerial

ROS 2 package dependencies (from `package.xml`):
- `rclpy`
- `sensor_msgs`
- `std_msgs`
- `cv_bridge`

Embedded compute and hardware:
- Raspberry Pi 4B (on-vehicle processing target)
- 1:10 scale vehicle chassis
- USB camera
- Arduino board connected via USB serial
- Motor driver and DC motor

## Build Instructions

From workspace root:

```bash
cd /home/ahmed/autonomous_ws
colcon build --packages-select perception
source install/setup.bash
```

## Run Instructions

Use separate terminals after sourcing the workspace in each terminal.

Terminal 1:

```bash
cd /home/ahmed/autonomous_ws
source install/setup.bash
ros2 run perception sign_detection_node
```

Terminal 2:

```bash
cd /home/ahmed/autonomous_ws
source install/setup.bash
ros2 run perception vehicle_actuator_node
```

Optional monitoring terminal:

```bash
cd /home/ahmed/autonomous_ws
source install/setup.bash
ros2 topic echo /vehicle/command
```

## Arduino Serial Command Interface

`vehicle_actuator_node` writes the following bytes:
- `R` -> stop motor
- `Y` -> slow motor speed
- `G` -> normal move speed
- `O` -> no signal / idle fallback

Baud rate: `9600`

Default serial port in node code: `/dev/ttyACM0`

If your board appears on a different device path (`/dev/ttyUSB0`, etc.), update the path in:
- `src/perception/perception/nodes/vehicle_actuator_node.py`

## Integration Notes

Recommended checklist when integrating into a larger autonomous stack:
1. Keep this package under the main project `src/` directory as `perception`.
2. Verify ROS 2 environment consistency (same distro and Python environment) across modules.
3. Align topic names and message contracts with planner/controller modules.
4. Add launch files to start perception and actuator nodes together.
5. Externalize camera index, sign/light thresholds, and serial port into ROS parameters.

## Current Limitations

- Camera index and serial port are hardcoded.
- Current perception logic focuses on traffic light color detection.
- General traffic sign detection and maneuver classification are not fully implemented yet.
- HSV thresholds are fixed and may need retuning for different lighting.
- No formal launch file yet.
- No automated integration tests for camera/serial hardware-in-the-loop.

## Suggested Next Improvements

1. Add ROS 2 parameters for all runtime-tuned values.
2. Add a launch file (`perception.launch.py`) to start both nodes together.
3. Extend perception to classify road signs (stop, turn left/right, U-turn, speed/slow zones).
4. Add a command arbitration layer for sign and traffic-light priorities.
5. Replace simple pixel-count logic with region-based detection and confidence checks.
6. Add logging rate control and optional debug image publishing.

## Project Scope Summary

This project targets end-to-end autonomous behavior on a 1:10 scale vehicle by combining:
- onboard visual perception (camera),
- embedded inference/processing (Raspberry Pi 4B),
- and real-time command execution (actuation interface).
