#!/usr/bin/env python3
"""
Telemetry Plotting Script for Zooba
====================================
Subscribes to live ROS 2 topics to collect:
1. Target Trajectory (/path_planner/target)
2. Filtered EKF State (/vehicle/state)
3. Unfiltered Hardware Velocity (/vehicle/feedback)
4. Unfiltered IMU Yaw (/imu/data)

Usage:
    ros2 run ... or simply:
    python3 plot_telemetry.py

Press Ctrl+C to stop recording and generate the plots.
"""

import rclpy
from rclpy.node import Node
import matplotlib
matplotlib.use('Agg') # Use headless backend for SSH
import matplotlib.pyplot as plt
import numpy as np
import time

# Import ROS message types
from vehicle_interfaces.msg import VehicleState, VehicleFeedback, ImuData


class TelemetryPlotter(Node):
    def __init__(self):
        super().__init__('telemetry_plotter')

        # Data storage
        self.target_data = {'t': [], 'x': [], 'y': [], 'yaw': [], 'v': []}
        self.state_data = {'t': [], 'x': [], 'y': [], 'yaw': [], 'v': []}
        self.feedback_data = {'t': [], 'v': []}
        self.imu_data = {'t': [], 'yaw': []}

        self.start_time = None

        # Subscribers
        self.sub_target = self.create_subscription(
            VehicleState, '/path_planner/target', self.target_cb, 10)
        self.sub_state = self.create_subscription(
            VehicleState, '/vehicle/state', self.state_cb, 10)
        self.sub_feedback = self.create_subscription(
            VehicleFeedback, '/vehicle/feedback', self.feedback_cb, 10)
        self.sub_imu = self.create_subscription(
            ImuData, '/imu/data', self.imu_cb, 10)

        self.get_logger().info("Telemetry plotter started.")
        self.get_logger().info("Listening to /path_planner/target, /vehicle/state, /vehicle/feedback, /imu/data")
        self.get_logger().info("Press Ctrl+C to stop and display plots...")

    def get_time(self, msg_header=None):
        if self.start_time is None:
            self.start_time = time.time()
        # Fallback to local time if header has no stamp or using system time
        return time.time() - self.start_time

    def target_cb(self, msg):
        t = self.get_time(msg.header)
        self.target_data['t'].append(t)
        self.target_data['x'].append(msg.x)
        self.target_data['y'].append(msg.y)
        self.target_data['yaw'].append(msg.yaw)
        self.target_data['v'].append(msg.velocity)

    def state_cb(self, msg):
        t = self.get_time(msg.header)
        self.state_data['t'].append(t)
        self.state_data['x'].append(msg.x)
        self.state_data['y'].append(msg.y)
        self.state_data['yaw'].append(msg.yaw)
        self.state_data['v'].append(msg.velocity)

    def feedback_cb(self, msg):
        t = self.get_time(msg.header)
        self.feedback_data['t'].append(t)
        self.feedback_data['v'].append(msg.actual_velocity)

    def imu_cb(self, msg):
        t = self.get_time(msg.header)
        self.imu_data['t'].append(t)
        # Convert IMU degrees to radians for comparison
        self.imu_data['yaw'].append(np.deg2rad(msg.yaw))


def plot_results(plotter):
    print("Generating plots...")
    
    # Create a figure with 3 subplots
    fig = plt.figure(figsize=(15, 10))

    # 1. XY Trajectory Plot
    ax1 = plt.subplot(2, 2, (1, 3))
    if plotter.target_data['x']:
        ax1.plot(plotter.target_data['x'], plotter.target_data['y'], 
                 'r--', label='Desired Target', linewidth=2)
    if plotter.state_data['x']:
        ax1.plot(plotter.state_data['x'], plotter.state_data['y'], 
                 'b-', label='EKF State', linewidth=2)
    ax1.set_title('Vehicle Path: Actual vs Target')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.axis('equal')
    ax1.grid(True)
    ax1.legend()

    # 2. Velocity Comparison Plot (Filtered vs Unfiltered)
    ax2 = plt.subplot(2, 2, 2)
    if plotter.target_data['t']:
        ax2.plot(plotter.target_data['t'], plotter.target_data['v'], 
                 'r--', label='Target Velocity')
    if plotter.feedback_data['t']:
        ax2.plot(plotter.feedback_data['t'], plotter.feedback_data['v'], 
                 'g.', label='Unfiltered (Encoder)', alpha=0.5)
    if plotter.state_data['t']:
        ax2.plot(plotter.state_data['t'], plotter.state_data['v'], 
                 'b-', label='Filtered (EKF)', linewidth=2)
    ax2.set_title('Velocity: Filtered vs Unfiltered')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Velocity [m/s]')
    ax2.grid(True)
    ax2.legend()

    # 3. Yaw/Heading Comparison Plot (Filtered vs Unfiltered)
    ax3 = plt.subplot(2, 2, 4)
    if plotter.target_data['t']:
        # Target yaw could have jumps (wrapping), we unwrap it
        try:
            target_yaw = np.unwrap(plotter.target_data['yaw'])
            ax3.plot(plotter.target_data['t'], target_yaw, 'r--', label='Target Yaw')
        except:
            pass
    if plotter.imu_data['t']:
        try:
            imu_yaw = np.unwrap(plotter.imu_data['yaw'])
            ax3.plot(plotter.imu_data['t'], imu_yaw, 'g.', label='Unfiltered (IMU)', alpha=0.5)
        except:
            pass
    if plotter.state_data['t']:
        try:
            state_yaw = np.unwrap(plotter.state_data['yaw'])
            ax3.plot(plotter.state_data['t'], state_yaw, 'b-', label='Filtered (EKF)', linewidth=2)
        except:
            pass
    ax3.set_title('Heading: Filtered vs Unfiltered')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Yaw [rad]')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    # Save the figure instead of showing it (fixes SSH/headless issues)
    output_file = 'telemetry_results.png'
    plt.savefig(output_file, dpi=300)
    print(f"\n✅ Plots successfully saved to: {output_file}")
    print("You can view it via VS Code remote, SCP, or a VNC server.\n")

def main(args=None):
    rclpy.init(args=args)
    plotter = TelemetryPlotter()

    try:
        rclpy.spin(plotter)
    except KeyboardInterrupt:
        pass
    finally:
        # Plot data when node shuts down
        plot_results(plotter)
        plotter.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
