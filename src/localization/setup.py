# =====================================================================
# Localization Package — setup.py
# =====================================================================
# FILE: localization/setup.py
# STATUS: MODIFIED — added sensor_noise_node and ekf_sim_node entry points
# MODIFIED: 2026-05-16
#
# CHANGES MADE:
#   1. Added 'ekf_localization_node' to console_scripts entry_points
#   2. Added config/ directory to data_files (for ekf_localization.yaml)
#   3. Added 'sensor_noise_node' — Gaussian noise injection for simulation
#   4. Added 'ekf_sim_node'      — EKF that filters noisy simulated sensors
#   5. Bumped version to 0.3.0
#
# AVAILABLE NODES (entry_points):
#   odometry_node           — old dead-reckoning (encoder + IMU, no drift correction)
#   ground_truth_node       — simulation-only (Gazebo world-frame pose, perfect state)
#   ekf_localization_node   — EKF fusion (encoder + IMU, with drift correction)
#   sensor_noise_node       — NEW: adds Gaussian noise to ground-truth state
#   ekf_sim_node            — NEW: EKF that filters noise for simulation
#
# HOW TO BUILD:
#   cd ~/Documents/UNIVERSITY/semster\ 10/Autonoumous/project/zooba_workspace
#   colcon build --packages-select localization
#   source install/setup.bash
# =====================================================================

import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'localization'

setup(
    name=package_name,
    version='0.3.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # Include config files (NEW — for ekf_localization.yaml)
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ahmed',
    maintainer_email='ahm.mostafa03@gmail.com',
    description='Localization nodes: EKF (fused encoder + IMU), odometry (dead-reckoning), and ground-truth (Gazebo)',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'odometry_node         = localization.odometry_node:main',
            'ground_truth_node     = localization.ground_truth_node:main',
            'ekf_localization_node = localization.ekf_localization_node:main',
            'sensor_noise_node        = localization.sensor_noise_node:main',
            'ekf_sim_node             = localization.ekf_sim_node:main',
            'state_comparison_node    = localization.state_comparison_node:main',
            'state_plot_node          = localization.state_plot_node:main',
            'state_plot_noisy_vs_ekf_node = localization.state_plot_noisy_vs_ekf_node:main',
        ],
    },
)
