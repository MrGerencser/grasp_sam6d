from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'grasp_sam6d'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Only copy your own model files, not the entire SAM-6D repo
        ('share/' + package_name + '/models', glob('Data/models/*.ply')),
        # Include launch files
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        # Include config files
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'opencv-python',
        'numpy',
        'PyYAML',
    ],
    zip_safe=True,
    maintainer='chris',
    maintainer_email='gejan@ethz.ch',
    description='Grasp_sam6D wrapper for ROS2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'grasp_sam6d = grasp_sam6d.grasp_sam6d_node:main',
            'grasp_executor = grasp_sam6d.grasp_executor:main',
            'grasp_executor_bolt = grasp_sam6d.grasp_executor_bolt:main',
        ],
    },
)