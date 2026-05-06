# =====================================================================
# Localization Package — setup.py
# =====================================================================
# FILE: localization/setup.py
# STATUS: MODIFIED — added EKF localization node entry point
# MODIFIED: 2026-04-24
#
# CHANGES MADE:
#   1. Added 'ekf_localization_node' to console_scripts entry_points
#   2. Added config/ directory to data_files (for ekf_localization.yaml)
#   3. Bumped version to 0.2.0
#
# AVAILABLE NODES (entry_points):
#   odometry_node           — old dead-reckoning (encoder + IMU, no drift correction)
#   ground_truth_node       — simulation-only (Gazebo world-frame pose, perfect state)
#   ekf_localization_node   — NEW: EKF fusion (encoder + IMU, with drift correction)
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
    version='0.2.0',
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
            'odometry_node = localization.odometry_node:main',
            'ground_truth_node = localization.ground_truth_node:main',
            'ekf_localization_node = localization.ekf_localization_node:main',  # NEW
        ],
    },
)
