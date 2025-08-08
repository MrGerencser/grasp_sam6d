# SAM-6D ROS2 Wrapper (with ZED camera and grasp synthesis)

High-level ROS 2 Python nodes that run SAM-6D instance segmentation + 6D pose estimation from a ZED camera stream, optionally generate grasp poses from YAML, and publish robot-frame poses.

- Main node: publishes best pose or best grasp.
- Cube node: publishes a PoseArray with top-N detections for debugging/evaluation.
- Optional grasp executor for Franka gripper.

## Package layout

- Nodes
  - [sam6d_wrapper/sam6d_wrapper_node.py](sam6d_wrapper/sam6d_wrapper_node.py)
  - [sam6d_wrapper/sam6d_wrapper_cube_node.py](sam6d_wrapper/sam6d_wrapper_cube_node.py)
  - [sam6d_wrapper/grasp_executor.py](sam6d_wrapper/grasp_executor.py)
- Utilities
  - [sam6d_wrapper/pose_utils.py](sam6d_wrapper/sam6d_wrapper/pose_utils.py) (pose math, parsing, template rendering, grasp helpers:
    - [sam6d_wrapper.pose_utils.GraspUtils.load_grasp_config](sam6d_wrapper/sam6d_wrapper/pose_utils.py)
    - [sam6d_wrapper.pose_utils.GraspUtils.generate_grasp_poses](sam6d_wrapper/sam6d_wrapper/pose_utils.py)
    - [sam6d_wrapper.pose_utils.GraspUtils.select_best_grasp](sam6d_wrapper/sam6d_wrapper/pose_utils.py)
  )
  - [utils/visualize_grasps.py](utils/visualize_grasps.py) (preview YAML grasps on a model)
  - Conversion helpers in [utils/](utils)
- Launch files
  - [launch/sam6d_wrapper.launch.py](launch/sam6d_wrapper.launch.py)
  - [launch/sam6d_wrapper_cube.launch.py](launch/sam6d_wrapper_cube.launch.py)
- Config/calibration
  - [config/transform.yaml](config/transform.yaml)
- Models and grasps
  - [Data/models/](Data/models) with per-model grasp YAMLs (e.g., [Data/models/mustard_bottle/mustard_bottle_grasps.yaml](Data/models/mustard_bottle/mustard_bottle_grasps.yaml))

Entry points are defined in [setup.py](setup.py).

## Requirements

- ROS 2 Humble with Python build tools (ament_python)
- ZED SDK + pyzed (import pyzed.sl)
- SAM-6D checked out locally and working:
  - The node expects its path via `sam6d_path` (default: `/home/chris/SAM-6D/SAM-6D`)
  - It uses the provided `run_inference_custom.py` in SAM-6D’s Instance_Segmentation_Model and Pose_Estimation_Model
- BlenderProc for template rendering (used automatically if no templates exist)
- Python deps are handled via [setup.py](setup.py) (opencv-python, numpy, PyYAML). Install extras if needed:
  - trimesh, open3d (for utils), scipy (for utils/visualize_grasps.py)

## Build and setup

```bash
colcon build --packages-select sam6d_wrapper
source install/setup.bash
```

## Launch

Main node (best pose + optional grasp selection):
```bash
ros2 launch sam6d_wrapper sam6d_wrapper.launch.py \
  cad_path:=/absolute/path/to/your_model.ply \
  sam6d_path:=/absolute/path/to/SAM-6D \
  output_frame:=panda_link0 \
  processing_rate:=0.2 \
  camera_sn:=0 \
  resolution:=HD720 \
  transform_config:=$(ros2 pkg prefix sam6d_wrapper)/share/sam6d_wrapper/config/transform.yaml \
  grasp_poses:=true
```

Cube/debug node (publishes PoseArray of top detections):
```bash
ros2 launch sam6d_wrapper sam6d_wrapper_cube.launch.py \
  cad_path:=/absolute/path/to/cube_mm.ply \
  sam6d_path:=/absolute/path/to/SAM-6D
```

Optional grasp executor (Franka):
```bash
ros2 run sam6d_wrapper grasp_executor
```

## Parameters

Common node parameters (see [sam6d_wrapper_node.py](sam6d_wrapper/sam6d_wrapper_node.py) and [sam6d_wrapper_cube_node.py](sam6d_wrapper/sam6d_wrapper/sam6d_wrapper_cube_node.py)):

- cad_path: Path to .ply CAD model
- sam6d_path: Path to SAM-6D repo
- output_frame: Robot base frame (default: panda_link0)
- processing_rate: Inference Hz (default: 0.5 or 0.2 in launch)
- camera_sn: ZED serial (0 = first)
- resolution: HD720 | HD1080 | HD2K
- transform_config: YAML with calibration ([config/transform.yaml](config/transform.yaml))
- instance_model: Instance segmentor to use (default: "sam")
- grasp_poses: Enable grasp generation from YAML (true/false)
- debug_outputs: If true, saves extra diagnostics

## Calibration

Provide:
- transforms.T_robot_chess
- transforms.T_chess_cam1

in [config/transform.yaml](config/transform.yaml). These are 4x4 homogeneous transforms. The node precomputes camera->robot with:
- [sam6d_wrapper.pose_utils.precompute_cam_to_robot](sam6d_wrapper/sam6d_wrapper/pose_utils.py)

## Model templates

On first run for a given CAD, the node renders templates with BlenderProc into:
```
/path/to/your_model_templates/
```
Templates are copied per-inference into a timestamped debug folder and reused across runs.

## Grasp YAMLs

Place grasp files next to the model (same folder). The loader searches common names; see [sam6d_wrapper.pose_utils.GraspUtils.load_grasp_config](sam6d_wrapper/sam6d_wrapper/pose_utils.py).

- Units: position in meters (object frame); orientation as Euler xyz in radians.
- Example:
```yaml
grasps:
  - name: "side_grasp"
    position: [0.0, 0.0, 0.0]
    orientation: [0.0, 1.5708, 0.0]
```
See samples in [Data/models/*/*_grasps.yaml](Data/models).

Preview grasps on a model point cloud:
```bash
python3 utils/visualize_grasps.py
```
Script: [utils/visualize_grasps.py](utils/visualize_grasps.py)

## Topics

Publishers:
- /sam6d/pose (geometry_msgs/PoseStamped) — best pose or best grasp (main and cube nodes)
- /sam6d/pose_array (geometry_msgs/PoseArray) — top-N poses (cube node)

The grasp executor uses:
- /cartesian_target_pose, /riemannian_motion_policy/reference_pose (PoseStamped)
- /fr3_gripper/move (action), /fr3_gripper/grasp (action)
- /fr3_gripper/joint_states (JointState)
- /franka_robot_state_broadcaster/current_pose (PoseStamped)

See [sam6d_wrapper/grasp_executor.py](sam6d_wrapper/grasp_executor.py).

## Debug outputs

Each inference saves under:
```
<model_dir>/debug_outputs/processing_<timestamp>/
  rgb.png, depth.png, camera.json, outputs/sam6d_results/...
```

## Utilities

- Mesh conversion: [utils/convert_stl_to_ply.py](utils/convert_stl_to_ply.py), [utils/convert_obj_to_ply.py](utils/convert_obj_to_ply.py), [utils/convert_ply_to_mm.py](utils/convert_ply_to_mm.py)
- Visualization: [utils/vis_cube.py](utils/vis_cube.py), [utils/visualize_grasps.py](utils/visualize_grasps.py)

## Troubleshooting

- ZED not opening: verify SDK install and permissions; check `camera_sn` and `resolution`.
- No templates: ensure BlenderProc is installed and callable (`blenderproc run ...`).
- No