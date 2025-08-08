from __future__ import annotations

import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
import yaml

from dataclasses import dataclass
from geometry_msgs.msg import PoseStamped
from rclpy.time import Time
from std_msgs.msg import Header

################################################################################
# Low-level numeric helpers
################################################################################

def _normalize_quaternion(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0.0:
        raise ValueError("Zero-norm quaternion")
    return q / n

def _homogeneous_matrix(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T

def quat_to_rot(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = _normalize_quaternion(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ])

def rot_to_quat(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        S = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / S
        x = (m21 - m12) * S
        y = (m02 - m20) * S
        z = (m10 - m01) * S
    elif (m00 > m11) and (m00 > m22):
        S = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    return _normalize_quaternion([x, y, z, w])

def quat_to_euler(q: Sequence[float]) -> np.ndarray:
    q = _normalize_quaternion(q)
    x, y, z, w = q
    roll = np.arctan2(2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(2.0 * (w * y - z * x))
    yaw = np.arctan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])

def euler_to_quat(euler_angles: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = euler_angles
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = sy * cr * cp + cy * sr * sp
    z = sy * sr * cp - cy * cr * sp

    return _normalize_quaternion([x, y, z, w])

################################################################################
# Euler/rotation conversions
################################################################################

def euler_to_rotation_matrix(euler_angles: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = euler_angles
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx  # ZYX convention

def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    sy = np.linalg.norm(R[:2, 0])
    singular = sy < 1e-6
    if not singular:
        roll  = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = np.arctan2(R[1, 0], R[0, 0])
    else:  # Gimbal lock
        roll  = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = 0.0
    return np.array([roll, pitch, yaw])

################################################################################
# User-facing dataclass
################################################################################

@dataclass
class PoseWithConfidence:
    position: np.ndarray  # shape (3,)
    orientation: np.ndarray  # quaternion (x, y, z, w)
    confidence: float = 1.0

    def as_pose_stamped(self, frame_id: str, stamp: Time) -> PoseStamped:
        msg = PoseStamped()
        msg.header = Header(frame_id=frame_id, stamp=stamp.to_msg())
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = self.position
        (msg.pose.orientation.x,
         msg.pose.orientation.y,
         msg.pose.orientation.z,
         msg.pose.orientation.w) = self.orientation
        return msg

################################################################################
# High-level helpers
################################################################################

def precompute_cam_to_robot(T_robot_chess: np.ndarray, T_chess_cam: np.ndarray) -> np.ndarray:
    """Return a single 4×4 matrix that maps camera ➜ robot coordinates."""
    return T_robot_chess @ T_chess_cam

def camera_to_robot(pose_cam: PoseWithConfidence, T_cam_robot: np.ndarray) -> PoseWithConfidence:
    """Transform ``pose_cam`` (in *camera* frame) into the *robot* frame."""
    R_cam = quat_to_rot(pose_cam.orientation)
    T_obj_cam = _homogeneous_matrix(R_cam, pose_cam.position)
    T_obj_robot = T_cam_robot @ T_obj_cam
    position_robot = T_obj_robot[:3, 3]
    R_robot = T_obj_robot[:3, :3]
    quat_robot = rot_to_quat(R_robot)
    return PoseWithConfidence(position=position_robot, orientation=quat_robot, confidence=pose_cam.confidence)

################################################################################
# Grasp and geometry utilities
################################################################################

class GraspUtils:
    @staticmethod
    def load_grasp_config(cad_path, model_name, logger=None):
        try:
            model_dir = os.path.dirname(cad_path)
            possible_grasp_files = [
                os.path.join(model_dir, f'{model_name}_grasps.yaml'),
                os.path.join(model_dir, f'{os.path.basename(model_dir)}_grasps.yaml'),
                os.path.join(model_dir, model_name, f'{model_name}_grasps.yaml'),
                os.path.join(model_dir, 'grasps.yaml'),
            ]
            grasp_file = None
            for possible_file in possible_grasp_files:
                if os.path.exists(possible_file):
                    grasp_file = possible_file
                    if logger:
                        logger.info(f'Found grasp file at: {grasp_file}')
                    break
            if grasp_file is None:
                if logger:
                    logger.debug(f'No grasp file found. Searched:')
                    for possible_file in possible_grasp_files:
                        logger.debug(f'  - {possible_file}')
                return None
            try:
                with open(grasp_file, 'r', encoding='utf-8') as f:
                    grasp_data = yaml.safe_load(f)
            except UnicodeDecodeError:
                if logger:
                    logger.warning(f'UTF-8 decode failed, trying latin-1 encoding for {grasp_file}')
                with open(grasp_file, 'r', encoding='latin-1') as f:
                    grasp_data = yaml.safe_load(f)
            grasps_config = grasp_data.get('grasps', [])
            if not grasps_config:
                if logger:
                    logger.warning(f'No grasps defined in {grasp_file}')
                return None
            return grasps_config
        except Exception as e:
            if logger:
                logger.error(f'Error loading grasp config: {e}')
            return None

    @staticmethod
    def generate_grasp_poses(object_pose, grasps_config, logger=None):
        try:
            obj_position = np.array(object_pose['position'])
            obj_orientation = object_pose['orientation']
            R_obj = quat_to_rot(obj_orientation)
            T_obj = np.eye(4)
            T_obj[:3, :3] = R_obj
            T_obj[:3, 3] = obj_position
            grasp_poses = []
            for grasp_config in grasps_config:
                try:
                    grasp_name = grasp_config.get('name', 'unnamed_grasp')
                    grasp_position = np.array(grasp_config.get('position', [0, 0, 0]))
                    grasp_orientation_euler = np.array(grasp_config.get('orientation', [0, 0, 0]))
                    R_grasp_rel = euler_to_rotation_matrix(grasp_orientation_euler)
                    T_grasp_rel = np.eye(4)
                    T_grasp_rel[:3, :3] = R_grasp_rel
                    T_grasp_rel[:3, 3] = grasp_position
                    T_grasp_world = T_obj @ T_grasp_rel
                    grasp_world_position = T_grasp_world[:3, 3]
                    grasp_world_rotation = T_grasp_world[:3, :3]
                    grasp_world_quaternion = rot_to_quat(grasp_world_rotation)
                    grasp_pose = {
                        'name': grasp_name,
                        'position': [float(grasp_world_position[0]), float(grasp_world_position[1]), float(grasp_world_position[2])],
                        'orientation': [float(grasp_world_quaternion[0]), float(grasp_world_quaternion[1]), float(grasp_world_quaternion[2]), float(grasp_world_quaternion[3])],
                        'confidence': object_pose['confidence'],
                        'object_pose': object_pose
                    }
                    grasp_poses.append(grasp_pose)
                    if logger:
                        logger.debug(f'Generated grasp "{grasp_name}": pos=[{grasp_pose["position"][0]:.3f}, {grasp_pose["position"][1]:.3f}, {grasp_pose["position"][2]:.3f}]')
                except Exception as e:
                    if logger:
                        logger.error(f'Error processing grasp "{grasp_config.get("name", "unknown")}": {e}')
                    continue
            if logger:
                logger.info(f'Generated {len(grasp_poses)} grasp poses')
            return grasp_poses
        except Exception as e:
            if logger:
                logger.error(f'Error generating grasp poses: {e}')
            return None

    @staticmethod
    def select_best_grasp(grasp_poses, workspace_bounds=None, logger=None):
        if not grasp_poses:
            return None
        if workspace_bounds is None:
            workspace_bounds = {
                'x_min': 0.2, 'x_max': 0.6,
                'y_min': -0.3, 'y_max': 0.3,
                'z_min': 0.01, 'z_max': 0.5
            }
        best_grasp = None
        best_score = -1
        robot_base = np.array([0.0, 0.0, 0.0])
        for grasp in grasp_poses:
            try:
                position = np.array(grasp['position'])
                distance_to_base = np.linalg.norm(position - robot_base)
                distance_score = 1.0 / (1.0 + distance_to_base)
                height_score = 1.0 if position[2] > 0.05 else 0.5
                workspace_score = 1.0 if (workspace_bounds['x_min'] <= position[0] <= workspace_bounds['x_max'] and
                                          workspace_bounds['y_min'] <= position[1] <= workspace_bounds['y_max'] and
                                          workspace_bounds['z_min'] <= position[2] <= workspace_bounds['z_max']) else 0.3
                orientation = grasp['orientation']
                R_grasp = quat_to_rot(orientation)
                z_axis = R_grasp[:, 2]
                downward_score = max(0.0, -z_axis[2])
                confidence_score = grasp['confidence']
                type_score = 1.0
                if 'top' in grasp['name'].lower():
                    type_score = 1.2
                elif 'side' in grasp['name'].lower():
                    type_score = 0.8
                final_score = (distance_score * 1.5 +
                               height_score * 1.0 +
                               workspace_score * 2.0 +
                               downward_score * 1.0 +
                               confidence_score * 1.0 +
                               type_score * 0.5)
                if final_score > best_score:
                    best_score = final_score
                    best_grasp = grasp
                if logger:
                    logger.debug(f'Grasp "{grasp["name"]}" scores: dist={distance_score:.2f}, height={height_score:.2f}, workspace={workspace_score:.2f}, downward={downward_score:.2f}, conf={confidence_score:.2f}, type={type_score:.2f}, final={final_score:.2f}')
            except Exception as e:
                if logger:
                    logger.warning(f'Error scoring grasp "{grasp.get("name", "unknown")}": {e}')
                continue
        if best_grasp and logger:
            logger.info(f'Selected best grasp: "{best_grasp["name"]}" with score: {best_score:.3f}')
        return best_grasp

################################################################################
# SAM-6D and config utilities
################################################################################

class Sam6DUtils:
    @staticmethod
    def render_templates(sam6d_path, cad_path, output_dir, logger=None):
        try:
            render_dir = os.path.join(sam6d_path, 'Render')
            if not os.path.exists(render_dir):
                if logger:
                    logger.error(f'Render directory not found: {render_dir}')
                return False
            render_script = os.path.join(render_dir, 'render_custom_templates.py')
            if not os.path.exists(render_script):
                if logger:
                    logger.error(f'Render script not found: {render_script}')
                return False
            with tempfile.TemporaryDirectory() as temp_output_dir:
                env = os.environ.copy()
                env.update({
                    'OUTPUT_DIR': temp_output_dir,
                    'CAD_PATH': cad_path,
                })
                if logger:
                    logger.info('Starting Blender template rendering...')
                result = subprocess.run(
                    ['blenderproc', 'run', render_script,
                     '--output_dir', temp_output_dir, '--cad_path', cad_path],
                    cwd=render_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                if result.returncode != 0:
                    if logger:
                        logger.error(f'Template rendering failed: {result.stderr}')
                    return False
                templates_found = False
                for root, dirs, files in os.walk(temp_output_dir):
                    png_files = [f for f in files if f.endswith('.png')]
                    npy_files = [f for f in files if f.endswith('.npy')]
                    if png_files or npy_files:
                        os.makedirs(output_dir, exist_ok=True)
                        for file in png_files + npy_files:
                            src = os.path.join(root, file)
                            dst = os.path.join(output_dir, file)
                            shutil.copy2(src, dst)
                        templates_found = True
                        break
                if templates_found:
                    template_count = len([f for f in os.listdir(output_dir)
                                          if f.endswith('.png') or f.endswith('.npy')])
                    if logger:
                        logger.info(f'Template rendering completed! Created {template_count} templates')
                    return True
                else:
                    if logger:
                        logger.error('Template rendering completed but no templates found')
                    return False
        except subprocess.TimeoutExpired:
            if logger:
                logger.error('Template rendering timeout (10 minutes)')
            return False
        except Exception as e:
            if logger:
                logger.error(f'Template rendering error: {e}')
            return False

    @staticmethod
    def parse_results(output_dir, logger=None):
        try:
            possible_result_files = [
                os.path.join(output_dir, 'sam6d_results', 'detection_pem.json'),
                os.path.join(output_dir, 'poses.json'),
                os.path.join(output_dir, 'sam6d_results', 'poses.json'),
                os.path.join(output_dir, 'results.json'),
                os.path.join(output_dir, 'pose_estimation_results.json'),
            ]
            pose_file = None
            for f in possible_result_files:
                if os.path.exists(f):
                    pose_file = f
                    break
            if pose_file is None:
                if logger:
                    logger.warning(f'No pose results found in {output_dir}')
                return None
            if logger:
                logger.info(f'Reading results from: {pose_file}')
            with open(pose_file, 'r') as f:
                results = json.load(f)
            poses = []
            if isinstance(results, list):
                for i, detection in enumerate(results):
                    if isinstance(detection, dict) and 'R' in detection and 't' in detection:
                        try:
                            R_data = detection['R']
                            if isinstance(R_data, list) and len(R_data) == 3:
                                if all(isinstance(row, list) and len(row) == 3 for row in R_data):
                                    R = np.array(R_data)
                                else:
                                    continue
                            elif isinstance(R_data, list) and len(R_data) == 9:
                                R = np.array(R_data).reshape(3, 3)
                            else:
                                continue
                            quaternion = rot_to_quat(R)
                            t = np.array(detection['t']) / 1000.0
                            pose = {
                                'position': [float(t[0]), float(t[1]), float(t[2])],
                                'orientation': [float(quaternion[0]), float(quaternion[1]),
                                                float(quaternion[2]), float(quaternion[3])],
                                'confidence': float(detection.get('score', 1.0))
                            }
                            poses.append(pose)
                        except Exception as e:
                            if logger:
                                logger.error(f'Error parsing detection {i}: {e}')
                            continue
            if logger:
                logger.info(f'Parsed {len(poses)} poses from results')
            return poses
        except Exception as e:
            if logger:
                logger.error(f'Result parsing error: {e}')
            return None

class ConfigUtils:
    @staticmethod
    def load_transforms(config_path, logger=None):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            transforms = config.get('transforms', {})
            T_robot_chess_data = transforms.get('T_robot_chess')
            T_chess_cam1_data = transforms.get('T_chess_cam1')
            if T_robot_chess_data is None or T_chess_cam1_data is None:
                if logger:
                    logger.error('Required transforms not found in config')
                return None, None
            T_robot_chess = np.array(T_robot_chess_data)
            T_chess_cam1 = np.array(T_chess_cam1_data)
            if logger:
                logger.info('Successfully loaded transformation matrices')
            return T_robot_chess, T_chess_cam1
        except Exception as e:
            if logger:
                logger.error(f'Failed to load transforms: {e}')
            return None, None