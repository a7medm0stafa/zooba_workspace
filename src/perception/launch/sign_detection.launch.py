import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'perception'
    
    config_file = os.path.join(
        get_package_share_directory(pkg_name),
        'config',
        'sign_detection_params.yaml'
    )
    
    show_gui_arg = DeclareLaunchArgument(
        'show_gui',
        default_value='False',
        description='Whether to show the GUI window for sign detection'
    )
    
    sign_detection_node = Node(
        package=pkg_name,
        executable='sign_detection_node',
        name='sign_detection_node',
        output='screen',
        parameters=[
            config_file,
            {'show_gui': LaunchConfiguration('show_gui')}
        ]
    )
    
    return LaunchDescription([
        show_gui_arg,
        sign_detection_node
    ])
