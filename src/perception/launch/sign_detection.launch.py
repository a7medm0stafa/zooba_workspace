import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('perception')
    config_file = os.path.join(pkg_dir, 'config', 'sign_detection_params.yaml')

    sign_detection_node = Node(
        package='perception',
        executable='sign_detection_node',
        name='sign_detection_node',
        output='screen',
        parameters=[config_file]
    )

    return LaunchDescription([
        sign_detection_node
    ])
