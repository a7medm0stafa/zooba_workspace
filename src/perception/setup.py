import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ahmed',
    maintainer_email='ahm.mostafa03@gmail.com',
    description='perception package for traffic sign detection and vehicle actuation',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sign_detection_node = perception.nodes.sign_detection_node:main',
            'vehicle_actuator_node = perception.nodes.vehicle_actuator_node:main',
            'traffic_light_detector_node = perception.nodes.traffic_light_detector_node:main',
            'camera_publisher = perception.nodes.camera_publisher:main',
            'debug_viewer = perception.nodes.debug_viewer:main',
        ],
    },
)
