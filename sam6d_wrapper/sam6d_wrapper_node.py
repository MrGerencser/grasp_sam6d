# sam6d_wrapper/sam6d_wrapper_node.py
from __future__ import annotations

import os
import json
import queue
import shutil
import subprocess
import threading
from pathlib import Path
from typing import List
import tempfile
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node, SetParametersResult
from std_msgs.msg import Header
import pyzed.sl as sl

from .pose_utils import (
    PoseWithConfidence, camera_to_robot, precompute_cam_to_robot,
    euler_to_rotation_matrix, quat_to_rot, rot_to_quat, quat_to_euler, 
    GraspUtils, Sam6DUtils, ConfigUtils
)

################################################################################
# Helper conversion -----------------------------------------------------------
################################################################################

def _dict_to_pose(obj: dict) -> PoseWithConfidence:  # camera‑frame dict → dataclass
    return PoseWithConfidence(
        position=np.array(obj['position'], dtype=float),
        orientation=np.array(obj['orientation'], dtype=float),
        confidence=float(obj.get('confidence', 1.0)),
    )

################################################################################
# Main node -------------------------------------------------------------------
################################################################################

class Sam6DWrapper(Node):
    """High‑level orchestration node."""

    def __init__(self):
        super().__init__('sam6d_wrapper')

        # ------------------------- parameters
        self.declare_parameter('cad_path',          str(Path.home() / 'model.ply'))
        self.declare_parameter('output_frame',      'panda_link0')
        self.declare_parameter('processing_rate',   0.5)
        self.declare_parameter('sam6d_path',        str(Path.home() / 'SAM-6D'))
        self.declare_parameter('camera_sn',         0)
        self.declare_parameter('resolution',        'HD2K')
        self.declare_parameter('transform_config',  str(Path.home() / 'transform.yaml'))
        self.declare_parameter('instance_model',    'sam')
        self.declare_parameter('grasp_poses',       True)
        self.declare_parameter('debug_outputs',     False)
        self.declare_parameter('calib_preview',     False)

        # ------------------------- get params
        self.cad_path         = self.get_parameter('cad_path').value
        self.output_frame     = self.get_parameter('output_frame').value
        self.processing_rate  = self.get_parameter('processing_rate').value
        self.sam6d_path       = self.get_parameter('sam6d_path').value
        self.camera_sn        = self.get_parameter('camera_sn').value
        self.resolution       = self.get_parameter('resolution').value
        self.transform_config = self.get_parameter('transform_config').value
        self.instance_model   = self.get_parameter('instance_model').value
        self.use_grasps       = self.get_parameter('grasp_poses').value
        self.debug_outputs    = self.get_parameter('debug_outputs').value
        self.calib_preview    = self.get_parameter('calib_preview').value

        # Guard for runtime model switching
        self.model_lock = threading.RLock()

        # Log the initial model
        self._last_logged_model = self.cad_path
        cad_label = Path(self.cad_path).stem
        self.get_logger().info(f'Using CAD model: {cad_label}')
        self.get_logger().info(f'Using instance model: {self.instance_model}')

        # Validate instance_model
        allowed_models = {'sam', 'sam2', 'fastsam'}
        if self.instance_model not in allowed_models:
            self.get_logger().warn(f'instance_model "{self.instance_model}" not in {allowed_models}. Falling back to "sam".')
            self.instance_model = 'sam'

        # ------------------------- load static transforms
        T_robot_chess, T_chess_cam1 = ConfigUtils.load_transforms(self.transform_config, self.get_logger())
        if T_robot_chess is None or T_chess_cam1 is None:
            raise RuntimeError('Missing calibration YAML entries.')
        self.T_cam_robot = precompute_cam_to_robot(T_robot_chess, T_chess_cam1)

        # ------------------------- validate paths
        if not os.path.isdir(self.sam6d_path):
            raise FileNotFoundError(f'SAM-6D not found at {self.sam6d_path}')

        # ------------------------- model templates
        self.model_name           = Path(self.cad_path).stem
        self.model_templates_dir  = str(Path(self.cad_path).with_suffix('')) + '_templates'
        if not self._ensure_templates_exist():
            raise RuntimeError('Template generation failed')

        # ------------------------- camera
        if not self._initialize_zed_camera():
            raise RuntimeError('ZED init failed')

        # ------------------------- threading infra
        self.processing_queue: queue.Queue = queue.Queue(maxsize=1)
        threading.Thread(target=self._capture_worker,    daemon=True).start()
        threading.Thread(target=self._processing_worker, daemon=True).start()

        # Optional: show calibration preview window (non-blocking)
        if self.calib_preview:
            threading.Thread(target=self._show_calibration_preview, daemon=True).start()

        # ------------------------- ROS interfaces
        self.pose_pub = self.create_publisher(PoseStamped, 'sam6d/pose', 10)
        self.create_timer(1.0 / self.processing_rate, self.trigger_processing)

        # Register parameter change callback for model/segmentor switching
        self.add_on_set_parameters_callback(self._on_param_change)

        self.get_logger().info('SAM‑6D wrapper up and running')

    ############################################################################
    # Template cache ----------------------------------------------------------
    ############################################################################
    def _ensure_templates_exist(self) -> bool:
        if os.path.isdir(self.model_templates_dir) and os.listdir(self.model_templates_dir):
            self.get_logger().info('Using cached templates')
            return True
        self.get_logger().info('Rendering templates via Blender')
        Path(self.model_templates_dir).mkdir(parents=True, exist_ok=True)
        return Sam6DUtils.render_templates(self.sam6d_path, self.cad_path, self.model_templates_dir, self.get_logger())

    ############################################################################
    # ZED camera --------------------------------------------------------------
    ############################################################################
    def _initialize_zed_camera(self) -> bool:
        self.zed = sl.Camera()
        init_params = sl.InitParameters(camera_resolution=getattr(sl.RESOLUTION, self.resolution, sl.RESOLUTION.HD720),
                                        camera_fps=15,
                                        depth_mode=sl.DEPTH_MODE.NEURAL,
                                        coordinate_units=sl.UNIT.MILLIMETER,
                                        depth_minimum_distance=200,
                                        depth_maximum_distance=5000)
        if self.camera_sn:
            init_params.set_from_serial_number(self.camera_sn)
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error('ZED open failed')
            return False
        cam_info = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        self.camera_intrinsics = {
            'cam_K': [cam_info.fx, 0, cam_info.cx, 0, cam_info.fy, cam_info.cy, 0, 0, 1],
            'depth_scale': 1.0,
        }
        self.image_zed, self.depth_zed = sl.Mat(), sl.Mat()
        self.runtime_params            = sl.RuntimeParameters()
        self.frame_lock                = threading.Lock()
        self.latest_rgb: np.ndarray    = None
        self.latest_depth: np.ndarray  = None
        return True

    def _capture_worker(self):
        while rclpy.ok():
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT)
                self.zed.retrieve_measure(self.depth_zed, sl.MEASURE.DEPTH)
                rgb = cv2.cvtColor(self.image_zed.get_data(), cv2.COLOR_BGRA2BGR)
                depth = np.nan_to_num(self.depth_zed.get_data(), nan=0.0).clip(0, 65535).astype(np.uint16)
                with self.frame_lock:
                    self.latest_rgb, self.latest_depth = rgb.copy(), depth.copy()
            else:
                threading.Event().wait(0.01)

    ##########################################################################
    # Pipeline entry ---------------------------------------------------------
    ##########################################################################
    def trigger_processing(self):
        if not self.processing_queue.full():
            with self.frame_lock:
                if self.latest_rgb is None:
                    return
                rgb, depth = self.latest_rgb.copy(), self.latest_depth.copy()
            frame_data = {
                'rgb': rgb,
                'depth': depth,
                'camera_info': self.camera_intrinsics.copy(),
                'timestamp': self.get_clock().now(),
            }
            try:
                self.processing_queue.put_nowait(frame_data)
            except queue.Full:
                pass

    ##########################################################################
    # Processing -------------------------------------------------------------
    ##########################################################################
    def _processing_worker(self):
        while rclpy.ok():
            frame_data = self.processing_queue.get()
            poses_raw = self._process_frame_data(frame_data) or []
            if not poses_raw:
                self.processing_queue.task_done()
                continue

            # ---------------- camera‑frame → dataclass list
            poses_cam = [_dict_to_pose(p) for p in poses_raw]
            poses_robot = [camera_to_robot(p, self.T_cam_robot) for p in poses_cam]
            best_idx = int(np.argmax([p.confidence for p in poses_robot]))
            best_obj_cam, best_obj_robot = poses_cam[best_idx], poses_robot[best_idx]

            if self.use_grasps:
                best_grasp_dict = self._process_grasp_poses(best_obj_cam)
                if best_grasp_dict:
                    self._publish_grasp(best_grasp_dict, frame_data['timestamp'])
                    self.processing_queue.task_done()
                    continue
            # Fallback: publish object pose
            self._publish_pose(best_obj_robot, frame_data['timestamp'])
            self.processing_queue.task_done()
            
    def _process_frame_data(self, frame_data):
        """Process frame data using SAM-6D subprocess"""

        # Snapshot model-related parameters to avoid races during runtime switches
        with self.model_lock:
            cad_path = self.cad_path
            templates_dir = self.model_templates_dir
            instance_model = self.instance_model
            debug_on = self.debug_outputs  # <-- snapshot

        # Choose output location
        if debug_on:
            debug_dir = os.path.join(os.path.dirname(cad_path), 'debug_outputs')
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = frame_data['timestamp'].nanoseconds
            processing_dir = os.path.join(debug_dir, f'processing_{timestamp}')
            cleanup_after = False
        else:
            processing_dir = tempfile.mkdtemp(prefix='sam6d_')  # /tmp/...
            cleanup_after = True

        os.makedirs(processing_dir, exist_ok=True)

        try:
            output_dir = os.path.join(processing_dir, 'outputs')
            os.makedirs(output_dir, exist_ok=True)
            
            # Copy pre-rendered templates
            templates_dest = os.path.join(output_dir, 'templates')
            if os.path.exists(templates_dir):
                shutil.copytree(templates_dir, templates_dest)
            else:
                self.get_logger().error(f'Templates not found at {templates_dir}')
                return None
            
            # Save frame data (always needed for SAM-6D scripts)
            rgb_path = os.path.join(processing_dir, 'rgb.png')
            depth_path = os.path.join(processing_dir, 'depth.png')
            camera_path = os.path.join(processing_dir, 'camera.json')
            
            cv2.imwrite(rgb_path, frame_data['rgb'])
            cv2.imwrite(depth_path, frame_data['depth'])
            
            with open(camera_path, 'w') as f:
                json.dump(frame_data['camera_info'], f, indent=2)
            
            self.get_logger().info(f'Processing frame: RGB {frame_data["rgb"].shape}, Depth {frame_data["depth"].shape}')
            if debug_on:
                self.get_logger().info(f'Debug files saved to: {processing_dir}')
            
            # Run SAM-6D inference
            result = self._run_sam6d_inference(output_dir, rgb_path, depth_path, camera_path, cad_path, instance_model)
            
            if result:
                if debug_on:
                    self.get_logger().info(f'SAM-6D results available at: {output_dir}')
                else:
                    self.get_logger().info('SAM-6D results computed')
            else:
                if debug_on:
                    self.get_logger().warning(f'No poses detected. Check debug files at: {processing_dir}')
                else:
                    self.get_logger().warning('No poses detected.')
            
            return result
        
        except Exception as e:
            self.get_logger().error(f'Frame processing error: {e}')
            return None
        finally:
            # Clean temp dir when debug is off
            if cleanup_after:
                try:
                    shutil.rmtree(processing_dir, ignore_errors=True)
                except Exception:
                    pass

    def _run_sam6d_inference(self, output_dir, rgb_path, depth_path, camera_path, cad_path, instance_model):
        """Run SAM-6D inference only (templates already exist)"""
        try:
            env = os.environ.copy()
            env.update({
                'OUTPUT_DIR': output_dir,
                'CAD_PATH': cad_path,
                'RGB_PATH': rgb_path,
                'DEPTH_PATH': depth_path,
                'CAMERA_PATH': camera_path,
                'SEGMENTOR_MODEL': instance_model
            })
            
            self.get_logger().info('Running SAM-6D inference...')
            
            # Run instance segmentation
            result1 = subprocess.run(
                ['python3', 'run_inference_custom.py', 
                 '--segmentor_model', instance_model,
                 '--output_dir', output_dir,
                 '--cad_path', cad_path,
                 '--rgb_path', rgb_path,
                 '--depth_path', depth_path,
                 '--cam_path', camera_path],
                cwd=os.path.join(self.sam6d_path, 'Instance_Segmentation_Model'),
                env=env,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result1.returncode != 0:
                self.get_logger().error(f'Instance segmentation failed: {result1.stderr}')
                return None
            
            # Check segmentation results
            seg_path = os.path.join(output_dir, 'sam6d_results', 'detection_ism.json')
            if not os.path.exists(seg_path):
                self.get_logger().error(f'Segmentation results not found at {seg_path}')
                return None
            
            # Run pose estimation
            result2 = subprocess.run(
                ['python3', 'run_inference_custom.py',
                 '--output_dir', output_dir,
                 '--cad_path', cad_path,
                 '--rgb_path', rgb_path,
                 '--depth_path', depth_path,
                 '--cam_path', camera_path,
                 '--seg_path', seg_path],
                cwd=os.path.join(self.sam6d_path, 'Pose_Estimation_Model'),
                env=env,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result2.returncode != 0:
                self.get_logger().error(f'Pose estimation failed: {result2.stderr}')
                return None
            
            self.get_logger().info('SAM-6D inference completed successfully')
            
            # Parse results using utils
            return Sam6DUtils.parse_results(output_dir, self.get_logger())
            
        except subprocess.TimeoutExpired:
            self.get_logger().error('SAM-6D inference timeout')
            return None
        except Exception as e:
            self.get_logger().error(f'Inference error: {e}')
            return None

    ##########################################################################
    # Publisher -----------------------------------------------------
    ##########################################################################
    def _publish_pose(self, pose_robot: PoseWithConfidence, stamp):
        self.pose_pub.publish(pose_robot.as_pose_stamped(self.output_frame, stamp))
        p = pose_robot.position
        self.get_logger().info(f'Object pose @ {p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f} (conf={pose_robot.confidence:.2f})')

    def _publish_grasp(self, grasp_pose_robot: dict, stamp):
        pose_final = PoseWithConfidence(position=np.array(grasp_pose_robot['position']), orientation=np.array(grasp_pose_robot['orientation']), confidence=grasp_pose_robot['confidence'])
        self.pose_pub.publish(pose_final.as_pose_stamped(self.output_frame, stamp))
        self.get_logger().info(f'  Grasp "{grasp_pose_robot["name"]}" position: {pose_final.position} (conf={pose_final.confidence:.2f})')
        self.get_logger().info(f'  Grasp orientation in Degrees: {np.rad2deg(quat_to_euler(pose_final.orientation))}')

    ##########################################################################
    # Grasp pipeline ---------------------------------------------------------
    ##########################################################################
    def _generate_grasp_poses(self, obj_pose_cam: PoseWithConfidence) -> List[dict]:
        # Snapshot current model info to avoid races
        with self.model_lock:
            cad_path = self.cad_path
            model_name = self.model_name
        grasps_cfg = GraspUtils.load_grasp_config(cad_path, model_name, self.get_logger())
        if not grasps_cfg:
            return []
        return GraspUtils.generate_grasp_poses({
            'position': obj_pose_cam.position.tolist(),
            'orientation': obj_pose_cam.orientation.tolist(),
            'confidence': obj_pose_cam.confidence,
        }, grasps_cfg, self.get_logger())

    def _process_grasp_poses(self, obj_pose_cam: PoseWithConfidence):
        grasps_cam_dicts = self._generate_grasp_poses(obj_pose_cam)
        if not grasps_cam_dicts:
            return None
        grasps_robot_dicts = []
        for g_cam in grasps_cam_dicts:
            g_cam_obj = _dict_to_pose(g_cam)
            g_robot_obj = camera_to_robot(g_cam_obj, self.T_cam_robot)
            R_grasp = quat_to_rot(g_robot_obj.orientation)
            R_final = R_grasp @ euler_to_rotation_matrix([np.pi, 0, 0])   # apply EE standard
            q_final = rot_to_quat(R_final)
            grasps_robot_dicts.append({
                'name': g_cam['name'],
                'position': g_robot_obj.position.tolist(),
                'orientation': q_final.tolist(),
                'confidence': g_robot_obj.confidence,
            })
        return GraspUtils.select_best_grasp(grasps_robot_dicts, logger=self.get_logger())
            
    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'zed'):
                self.zed.close()
                self.get_logger().info('ZED camera closed')
        except Exception as e:
            self.get_logger().error(f'Error during cleanup: {e}')

    def _show_calibration_preview(self):
        """Open an Open3D window showing the current RGB-D point cloud in robot frame plus coordinate frames."""
        try:
            import open3d as o3d
        except Exception as e:
            self.get_logger().warn(f'Open3D not available ({e}). Skipping calibration preview.')
            return

        # Wait for a frame
        self.get_logger().info('Calibration preview: waiting for a frame...')
        rgb, depth = None, None
        for _ in range(100):  # ~5 seconds at 50 ms
            with self.frame_lock:
                if self.latest_rgb is not None and self.latest_depth is not None:
                    rgb = self.latest_rgb.copy()
                    depth = self.latest_depth.copy()
                    break
            time.sleep(0.05)

        if rgb is None:
            self.get_logger().warn('Calibration preview: no frame captured. Skipping.')
            return

        # Build Open3D RGBD and point cloud in camera frame
        h, w = rgb.shape[:2]
        color_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth)  # uint16 in millimeters
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1000.0,  # mm -> m
            depth_trunc=5.0,
            convert_rgb_to_intensity=False
        )

        fx, fy, cx, cy = self.camera_intrinsics['cam_K'][0], self.camera_intrinsics['cam_K'][4], self.camera_intrinsics['cam_K'][2], self.camera_intrinsics['cam_K'][5]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        # Optional voxel downsample for performance
        pcd = pcd.voxel_down_sample(voxel_size=0.01)

        # Transform to robot frame
        T = np.asarray(self.T_cam_robot, dtype=float)
        pcd.transform(T)

        # Coordinate frames: robot base at origin, camera at its pose in robot frame
        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        cam_frame.transform(T)

        self.get_logger().info('Calibration preview: showing point cloud in robot frame. Close the window to continue.')
        try:
            o3d.visualization.draw_geometries([pcd, base_frame, cam_frame], window_name='SAM6D Calibration Preview')
        except Exception as e:
            self.get_logger().warn(f'Open3D visualization error: {e}')

    def _on_param_change(self, params):
        allowed_models = {'sam', 'sam2', 'fastsam'}
        for param in params:
            if param.name == 'cad_path' and isinstance(param.value, str):
                with self.model_lock:
                    self.cad_path = param.value
                    self.model_name = Path(self.cad_path).stem
                    self.model_templates_dir = str(Path(self.cad_path).with_suffix('')) + '_templates'
                    self._last_logged_model = self.cad_path
                self.get_logger().info(f'CAD model changed: {self.model_name}')
                threading.Thread(target=self._ensure_templates_exist, daemon=True).start()

            elif param.name == 'instance_model':
                new_model = str(param.value)
                if new_model not in allowed_models:
                    self.get_logger().warn(f'instance_model "{new_model}" not in {allowed_models}. Keeping "{self.instance_model}".')
                else:
                    with self.model_lock:
                        self.instance_model = new_model
                    self.get_logger().info(f'Instance model changed: {self.instance_model}')

            elif param.name == 'calib_preview':
                with self.model_lock:
                    self.calib_preview = bool(param.value)
                if self.calib_preview:
                    self.get_logger().info('Calibration preview enabled.')
                    threading.Thread(target=self._show_calibration_preview, daemon=True).start()
                else:
                    self.get_logger().info('Calibration preview disabled.')
        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    node = Sam6DWrapper()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()