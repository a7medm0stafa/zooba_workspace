from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('perception')
    config_file = os.path.join(pkg_dir, 'config', 'sign_detection_params.yaml')

    # 🔥 Declare argument
    show_gui_arg = DeclareLaunchArgument(
        'show_gui',
        default_value='false'
    )

    # 🔥 Use argument
    show_gui = LaunchConfiguration('show_gui')

    sign_detection_node = Node(
        package='perception',
        executable='sign_detection_node',
        name='sign_detection_node',
        output='screen',
        parameters=[
            config_file,
            {'show_gui': show_gui}   # 👈 THIS is the missing part
        ]
    )

    return LaunchDescription([
        show_gui_arg,
        sign_detection_node
    ])