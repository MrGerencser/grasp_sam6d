#!/usr/bin/env python3
"""
visualize_grasps.py  –  show grasp poses on top of a point cloud
"""

import yaml
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R       # Euler→matrix helper

# --------------------------------------------------------------------------- helpers
def load_grasps(yaml_path):
    """Return list of dicts with keys name, position, orientation, finger_width."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("grasps", [])

def make_frames(grasps, scale=0.03):
    """Create one Open3D coordinate-frame mesh per grasp."""
    frames = []
    for g in grasps:
        # Euler angles are in radians; order is x-y-z (roll-pitch-yaw)
        R_mat = R.from_euler("xyz", g["orientation"]).as_matrix()
        print(R_mat)
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3,  3] = g["position"]

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        frame.transform(T)
        frames.append(frame)
    return frames

# --------------------------------------------------------------------------- main
def main(pcd_path, yaml_path):
    # load point cloud --------------------------------------------------------
    pcd = o3d.io.read_point_cloud(pcd_path)          # any format Open3D recognises
    # convert from mm to meters
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) / 1000.0)  # mm → m
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    # build grasp frames ------------------------------------------------------
    grasp_frames = make_frames(load_grasps(yaml_path))

    # draw --------------------------------------------------------------------
    o3d.visualization.draw_geometries([pcd, *grasp_frames])

if __name__ == "__main__":
    pcd_path = "/home/chris/franka_ros2_ws/src/sam6d_wrapper/Data/models/mustard_bottle/mustard_bottle.ply"
    yaml_path = "/home/chris/franka_ros2_ws/src/sam6d_wrapper/Data/models/mustard_bottle/mustard_bottle_grasps.yaml"
    main(pcd_path, yaml_path)