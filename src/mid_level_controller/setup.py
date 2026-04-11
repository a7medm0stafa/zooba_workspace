import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'mid_level_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # Include config files
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ahmed',
    maintainer_email='ahm.mostafa03@gmail.com',
    description='Mid-level controller: keyboard teleop and non-holonomic constraint enforcement for Ackermann-steered vehicle',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'teleop_keyboard_node = mid_level_controller.teleop_keyboard_node:main',
            'teleop_joy_node = mid_level_controller.teleop_joy_node:main',
            'nonholonomic_constraints_node = mid_level_controller.nonholonomic_constraints_node:main',
            'open_loop_node = mid_level_controller.open_loop_node:main',
        ],
    },
)
