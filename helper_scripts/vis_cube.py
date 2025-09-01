import open3d as o3d
import yaml
import os


ply = o3d.io.read_triangle_mesh('/home/chris/franka_ros2_ws/src/sam6d_wrapper/Data/models/tennisball/tennisball.ply')
if not ply.has_triangle_normals():
    ply.compute_triangle_normals()
    
coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])

# visualize the mesh
o3d.visualization.draw_geometries([ply])

# print length of the mesh
print(f"Length of the mesh: {ply.get_axis_aligned_bounding_box().get_extent()}")
