from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('grasp_sam6d')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'cad_path',
            default_value='/home/chris/franka_ros2_ws/src/grasp_sam6d/Data/models/main_dataset/allergenfreejarrodophilus/allergenfreejarrodophilus.ply',
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
            default_value='HD1080',
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
        DeclareLaunchArgument(
            'instance_model',
            default_value='sam2',
            description='Instance segmentor model: sam | sam2 | fastsam'
        ),
        DeclareLaunchArgument(
            'debug_outputs',
            default_value='true',
            description='If true, save debug frames/results into the model folder; if false, use temp dir and clean up'
        ),
        DeclareLaunchArgument(
            'use_best_ism_only',
            default_value='true',
            description='Filter ISM detections to the single best before PEM'
        ),
        DeclareLaunchArgument(
            'log_benchmarks',
            default_value='true',
            description='If true, log processing benchmarks'
        ),
        DeclareLaunchArgument(
            'calib_preview',
            default_value='false',
            description='If true, open an Open3D window with point cloud in robot frame and coordinate frames'
        ),
        DeclareLaunchArgument(
            'grasps_visualization',
            default_value='false',
            description='Visualize grasps in 3D space'
        ),
        
        Node(
            package='grasp_sam6d',
            executable='grasp_sam6d',
            name='grasp_sam6d',
            parameters=[{
                'cad_path': LaunchConfiguration('cad_path'),
                'sam6d_path': LaunchConfiguration('sam6d_path'),
                'output_frame': LaunchConfiguration('output_frame'),
                'processing_rate': LaunchConfiguration('processing_rate'),
                'camera_sn': LaunchConfiguration('camera_sn'),
                'resolution': LaunchConfiguration('resolution'),
                'transform_config': LaunchConfiguration('transform_config'),
                'grasp_poses': LaunchConfiguration('grasp_poses'),
                'instance_model': LaunchConfiguration('instance_model'),
                'debug_outputs': LaunchConfiguration('debug_outputs'),
                'use_best_ism_only': LaunchConfiguration('use_best_ism_only'),
                'log_benchmarks': LaunchConfiguration('log_benchmarks'),
                'calib_preview': LaunchConfiguration('calib_preview'),
                'grasps_visualization': LaunchConfiguration('grasps_visualization'),
            }],
            output='screen'
        )
    ])