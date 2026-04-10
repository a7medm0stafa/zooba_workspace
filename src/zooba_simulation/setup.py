import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'zooba_simulation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test', 'external']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ahmed',
    maintainer_email='ahm.mostafa03@gmail.com',
    description='Simulation bridge: translates VehicleCmd to Gazebo Ackermann steering vehicle topics',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sim_bridge_node = zooba_simulation.sim_bridge_node:main',
        ],
    },
)
