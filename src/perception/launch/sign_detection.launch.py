"""
sign_detection.launch.py
========================
Launch the GTSRB TFLite sign detection node.

Usage:
  ros2 launch perception sign_detection.launch.py
  ros2 launch perception sign_detection.launch.py show_gui:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('perception')
    config_file = os.path.join(pkg_dir, 'config', 'sign_detection_params.yaml')

    # ── Launch arguments ──────────────────────────────────────────────────────
    show_gui_arg = DeclareLaunchArgument(
        'show_gui',
        default_value='false',
        description='Enable OpenCV GUI windows for debugging'
    )

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Absolute path to .tflite model (empty = auto-detect)'
    )

    # ── Read launch arguments ─────────────────────────────────────────────────
    show_gui   = LaunchConfiguration('show_gui')
    model_path = LaunchConfiguration('model_path')

    # ── Sign detection node ───────────────────────────────────────────────────
    sign_detection_node = Node(
        package='perception',
        executable='sign_detection_node',
        name='sign_detection_node',
        namespace='perception',
        output='screen',
        parameters=[
            config_file,
            {
                'show_gui':   show_gui,
                'model_path': model_path,
            }
        ]
    )

    return LaunchDescription([
        show_gui_arg,
        model_path_arg,
        sign_detection_node,
    ])