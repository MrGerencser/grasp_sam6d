# SAM-6D ROS2 Wrapper (with ZED Camera and Grasp Synthesis)

A ROS 2 Python node that leverages the work of [SAM-6D](https://github.com/JiehongLin/SAM-6D),
integrated with a ZED camera for grasp synthesis tasks.

The original SAM-6D uses a Conda-based environment. However, since ROS 2 generally works best without multiple environments, it’s recommended to use this [modified SAM-6D](https://github.com/JiehongLin/SAM-6D) version to install the necessary Python packages globally rather than inside Conda. This ensures smoother integration with ROS 2 and avoids environment conflicts.
