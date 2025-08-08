from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('sam6d_wrapper')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'cad_path',
            default_value='/home/chris/franka_ros2_ws/src/sam6d_wrapper/Data/models/cube/cube_mm.ply',
            description='Path to CAD model'
        ),
        DeclareLaunchArgument(
            'sam6d_path',
            default_value='/home/chris/SAM-6D/SAM-6D',
            description='Path to SAM-6D installation'
        ),
        DeclareLaunchArgument(
            'output_frame',
            default_value='panda_link0',
            description='Output frame for poses (robot base frame)'
        ),
        DeclareLaunchArgument(
            'processing_rate',
            default_value='0.2',
            description='Processing rate in Hz'
        ),
        DeclareLaunchArgument(
            'camera_sn',
            default_value='0',
            description='ZED camera serial number (0 for first available)'
        ),
        DeclareLaunchArgument(
            'resolution',
            default_value='HD720',
            description='Camera resolution (HD720, HD1080, HD2K)'
        ),
        DeclareLaunchArgument(
            'transform_config',
            default_value=os.path.join(pkg_dir, 'config', 'transform.yaml'),
            description='Path to transform configuration file'
        ),
        
        DeclareLaunchArgument(
            'grasp_poses',
            default_value='true',
            description='Enable grasp poses generation from yaml files if available'
        ),
        
        Node(
            package='sam6d_wrapper',
            executable='sam6d_wrapper_cube',
            name='sam6d_wrapper_cube',
            parameters=[{
                'cad_path': LaunchConfiguration('cad_path'),
                'sam6d_path': LaunchConfiguration('sam6d_path'),
                'output_frame': LaunchConfiguration('output_frame'),
                'processing_rate': LaunchConfiguration('processing_rate'),
                'camera_sn': LaunchConfiguration('camera_sn'),
                'resolution': LaunchConfiguration('resolution'),
                'transform_config': LaunchConfiguration('transform_config'),
                'grasp_poses': LaunchConfiguration('grasp_poses'), 
            }],
            output='screen'
        )
    ])