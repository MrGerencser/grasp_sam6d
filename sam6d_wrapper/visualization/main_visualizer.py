# superquadric_grasp_system/visualization/main_visualizer.py 
import os, sys
import threading
import time
# so that "superquadric_grasp_system" is on PYTHONPATH when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import open3d as o3d
from typing import Dict, List, Optional, Any, Tuple, Union
from scipy.spatial.transform import Rotation as R_simple
import traceback

from .geometric_primitives import Gripper

# =============================================================================
# GRASP VISUALIZATION
# =============================================================================

class PerceptionVisualizer:
    """Unified perception visualization handler"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.default_gripper = Gripper()
        
        # Visualization settings
        self.gripper_colors = [
            (0.2, 0.8, 0.2), (0.8, 0.2, 0.2), (0.2, 0.2, 0.8), 
            (0.8, 0.8, 0.2), (0.8, 0.2, 0.8), (0.2, 0.8, 0.8)
        ]
        
        self.superquadric_colors = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1], 
            [1, 1, 0], [1, 0, 1], [0, 1, 1]
        ]
        
        # Filter visualization colors
        self.filter_colors = {
            'collision': (1.0, 0.0, 0.0),      # Red
            'support': (0.0, 1.0, 0.0),        # Green
            'quality': (0.0, 0.0, 1.0),        # Blue
            'final': (1.0, 0.5, 0.0),          # Orange
            'rejected': (0.5, 0.5, 0.5)        # Gray
        }

    # =============================================================================
    # CORE GRASP VISUALIZATION
    # =============================================================================
    
    def get_gripper_meshes(self, grasp_transform: np.ndarray, gripper: Gripper = None, 
                          show_sweep_volume: bool = False, color: Tuple[float, float, float] = (0.2, 0.8, 0), 
                          finger_tip_to_origin: bool = True) -> List[o3d.geometry.TriangleMesh]:
        """Create gripper visualization meshes"""
        if gripper is None:
            gripper = self.default_gripper
        
        # Get gripper meshes from geometric primitives
        gripper_meshes = gripper.make_open3d_meshes(colour=color)
        
        # Handle different mesh configurations
        if len(gripper_meshes) == 4:
            finger_L, finger_R, connector, back_Z = gripper_meshes
            all_gripper_parts = [finger_L, finger_R, back_Z, connector]
        else:
            all_gripper_parts = gripper_meshes
        
        # Transform all meshes to world coordinates
        meshes = []
        for mesh in all_gripper_parts:
            mesh_world = mesh.transform(grasp_transform.copy())
            meshes.append(mesh_world)
        
        # Add coordinate frame at grasp pose
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        coord_frame.transform(grasp_transform)
        meshes.append(coord_frame)
        
        # Add directional arrows
        meshes.extend(self._create_direction_arrows(grasp_transform, gripper))
        
        # Add sweep volume if requested
        if show_sweep_volume:
            sweep_volume = self._create_sweep_volume(grasp_transform, gripper)
            if sweep_volume:
                meshes.append(sweep_volume)
        
        return meshes
    
    def _create_direction_arrows(self, grasp_transform: np.ndarray, gripper: Gripper) -> List[o3d.geometry.TriangleMesh]:
        """Create directional arrows for grasp visualization"""
        arrows = []
        
        # Approach direction arrow (blue)
        arrow_length = 0.04
        approach_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.002, cone_radius=0.004,
            cylinder_height=arrow_length * 0.7, cone_height=arrow_length * 0.3
        )
        
        # Arrow points along gripper's approach direction (-Z in gripper frame)
        approach_dir = grasp_transform[:3, :3] @ gripper.approach_axis
        arrow_pos = grasp_transform[:3, 3] + approach_dir * arrow_length
        
        # Orient arrow along approach direction
        z_axis = np.array([0, 0, 1])
        if not np.allclose(approach_dir, z_axis):
            if np.allclose(approach_dir, -z_axis):
                arrow_rotation = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
            else:
                v = np.cross(z_axis, approach_dir)
                s = np.linalg.norm(v)
                c = np.dot(z_axis, approach_dir)
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                arrow_rotation = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s))
        else:
            arrow_rotation = np.eye(3)
        
        arrow_transform = np.eye(4)
        arrow_transform[:3, :3] = arrow_rotation
        arrow_transform[:3, 3] = arrow_pos
        approach_arrow.transform(arrow_transform)
        approach_arrow.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
        arrows.append(approach_arrow)
        
        # Closing direction arrow (red)
        closing_dir = grasp_transform[:3, :3] @ gripper.lambda_local  # Y axis
        closing_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.002, cone_radius=0.004,
            cylinder_height=0.03, cone_height=0.008
        )
        
        # Orient closing arrow
        if not np.allclose(closing_dir, z_axis):
            if np.allclose(closing_dir, -z_axis):
                closing_rotation = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
            else:
                v = np.cross(z_axis, closing_dir)
                s = np.linalg.norm(v)
                c = np.dot(z_axis, closing_dir)
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                closing_rotation = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s))
        else:
            closing_rotation = np.eye(3)
        
        closing_arrow_transform = np.eye(4)
        closing_arrow_transform[:3, :3] = closing_rotation
        closing_arrow_transform[:3, 3] = grasp_transform[:3, 3] + closing_dir * 0.03
        closing_arrow.transform(closing_arrow_transform)
        closing_arrow.paint_uniform_color([1.0, 0.0, 0.0])  # Red
        arrows.append(closing_arrow)
        
        return arrows
    
    def _create_sweep_volume(self, grasp_transform: np.ndarray, gripper: Gripper) -> Optional[o3d.geometry.TriangleMesh]:
        """Create sweep volume visualization"""
        try:
            sweep_volume = o3d.geometry.TriangleMesh.create_box(
                width=gripper.thickness * 2,
                height=gripper.max_open,
                depth=gripper.jaw_len
            )
            
            # Center the sweep volume in gripper local coordinates
            sweep_volume.translate([
                -gripper.thickness,
                -gripper.max_open / 2,
                -gripper.jaw_len
            ])
            
            # Apply the grasp transformation
            sweep_transform = grasp_transform.copy()
            sweep_volume.transform(sweep_transform)
            
            # Make it semi-transparent blue
            sweep_volume.paint_uniform_color([0.2, 0.2, 0.8])
            
            return sweep_volume
            
        except Exception as e:
            print(f"Error creating sweep volume: {e}")
            return None

    # =============================================================================
    # MAIN VISUALIZATION METHODS
    # =============================================================================
    
    def visualize_grasps_simple(self, grasp_poses: List[dict], point_cloud_data: np.ndarray = None,
                               gripper_colors: List = None, window_name: str = "Grasp Visualization",
                               show_sweep_volume: bool = False) -> None:
        """Simple grasp visualization method"""
        geometries = []
        
        # Add point cloud if provided
        if point_cloud_data is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud_data)
            pcd.paint_uniform_color([0.7, 0.7, 0.7])
            geometries.append(pcd)
        
        # Default colors if not provided
        if gripper_colors is None:
            gripper_colors = self.gripper_colors
        
        # Add grasp visualizations
        for i, grasp_pose in enumerate(grasp_poses):
            try:
                # Parse grasp pose to transformation matrix
                transform_matrix = self.parse_grasp_pose(grasp_pose)
                
                # Get color for this gripper
                color = gripper_colors[i % len(gripper_colors)]
                
                # Create gripper visualization
                gripper_meshes = self.get_gripper_meshes(
                    transform_matrix,
                    show_sweep_volume=show_sweep_volume,
                    color=color
                )
                
                geometries.extend(gripper_meshes)
                
            except Exception as e:
                print(f"Error creating grasp visualization {i+1}: {e}")
        
        # Add origin coordinate frame
        main_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        geometries.append(main_coord_frame)
        
        # Calculate view center
        if point_cloud_data is not None:
            view_center = np.mean(point_cloud_data, axis=0)
        else:
            view_center = np.array([0, 0, 0])
        
        # Visualize
        if len(geometries) > 0:
            o3d.visualization.draw_geometries(
                geometries,
                window_name=window_name,
                front=[0, -1, 0],
                lookat=view_center,
                up=[0, 0, 1]
            )
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def load_point_cloud(self, file_path: str) -> Optional[o3d.geometry.PointCloud]:
        """Load point cloud from file"""
        try:
            if not os.path.exists(file_path):
                print(f"Point cloud file not found: {file_path}")
                return None
            
            pcd = o3d.io.read_point_cloud(file_path)
            if len(pcd.points) == 0:
                print(f"Point cloud file is empty: {file_path}")
                return None
            
            print(f"Loaded point cloud with {len(pcd.points)} points from {file_path}")
            return pcd
        
        except Exception as e:
            print(f"Error loading point cloud from {file_path}: {e}")
            return None
    
    def parse_grasp_pose(self, grasp_input: Union[np.ndarray, Dict, Tuple, List]) -> np.ndarray:
        """Parse grasp pose from various input formats"""
        try:
            if isinstance(grasp_input, np.ndarray) and grasp_input.shape == (4, 4):
                return grasp_input
            
            elif isinstance(grasp_input, dict):
                position = np.array(grasp_input['position'])
                if 'quaternion' in grasp_input:
                    quat = np.array(grasp_input['quaternion'])  # [x, y, z, w]
                    R = R_simple.from_quat(quat).as_matrix()
                elif 'rotation_matrix' in grasp_input:
                    R = np.array(grasp_input['rotation_matrix'])
                elif 'euler' in grasp_input:
                    euler = np.array(grasp_input['euler'])
                    R = R_simple.from_euler('xyz', euler).as_matrix()
                else:
                    R = np.eye(3)
                
                transform = np.eye(4)
                transform[:3, :3] = R
                transform[:3, 3] = position
                return transform
            
            elif isinstance(grasp_input, (tuple, list)) and len(grasp_input) == 2:
                position, quat = grasp_input
                position = np.array(position)
                quat = np.array(quat)  # [x, y, z, w]
                R = R_simple.from_quat(quat).as_matrix()

                transform = np.eye(4)
                transform[:3, :3] = R
                transform[:3, 3] = position
                return transform
            
            else:
                print("Invalid grasp pose format. Using identity matrix.")
                return np.eye(4)
        
        except Exception as e:
            print(f"Error parsing grasp pose: {e}")
            return np.eye(4)
        
    # =============================================================================
    # DEMO AND TESTING
    # =============================================================================
    
    def create_cubic_object(self, size: float = 0.05, center: Tuple[float, float, float] = (0, 0, 0), 
                          color: Tuple[float, float, float] = (0.8, 0.2, 0.2)) -> o3d.geometry.TriangleMesh:
        """Create a cubic object for testing/demo purposes"""
        cube = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
        
        # Center the cube at the desired position
        cube.translate([-size/2, -size/2, -size/2])  # Center at origin first
        cube.translate(center)  # Then move to desired center
        
        cube.paint_uniform_color(color)
        cube.compute_vertex_normals()
        return cube

    def demo_visualization(self) -> None:
        """Demo showing basic visualization functionality"""
        # Create cube point cloud
        cube_points = []
        for x in np.linspace(-0.03, 0.03, 10):
            for y in np.linspace(-0.03, 0.03, 10):
                for z in np.linspace(-0.03, 0.03, 10):
                    if abs(x) > 0.025 or abs(y) > 0.025 or abs(z) > 0.025:
                        cube_points.append([x, y, z])
        
        cube_points = np.array(cube_points)
        
        # Define superquadric parameters (box-like)
        sq_params = {
            'shape': np.array([0.1, 0.1]),
            'scale': np.array([0.03, 0.03, 0.03]),
            'euler': np.array([0, 0, 0]),
            'translation': np.array([0, 0, 0])
        }
        
        # Example grasp
        grasp_pose = np.array([
            [0, 0, 1, -0.08],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ])
        
        self.visualize_grasps(
            point_cloud_data=cube_points,
            superquadric_params=sq_params,
            grasp_poses=grasp_pose,
            show_sweep_volume=True,
            window_name="PerceptionVisualizer Demo"
        )

# =============================================================================
# MAIN DEMO
# =============================================================================

if __name__ == "__main__":
    # Run demo
    visualizer = PerceptionVisualizer()
    visualizer.demo_visualization()