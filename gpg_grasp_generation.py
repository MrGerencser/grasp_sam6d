#!/usr/bin/env python3
import open3d as o3d, subprocess, yaml, tempfile, pathlib, re, sys
from math import atan2, asin
import numpy as np

# ---------- 1. load mesh, rescale mm→m, recenter ------------ #
mesh = o3d.io.read_triangle_mesh(
    "/home/chris/franka_ros2_ws/src/sam6d_wrapper/Data/models/banana/banana.ply")
mesh.scale(0.001, center=(0, 0, 0))               # mm → m :contentReference[oaicite:0]{index=0}

# mesh_center = mesh.get_center().copy()
# mesh.translate(-mesh_center)

pcd = mesh.sample_points_poisson_disk(8000)
pcd.estimate_normals()
pcd_f = tempfile.NamedTemporaryFile(delete=False, suffix=".pcd").name
o3d.io.write_point_cloud(pcd_f, pcd, write_ascii=True)  # save as PCD for GPG
# visualize the point cloud

# ---------- 2. run GPG and save complete log ---------------- #
gpg_bin = "/home/chris/gpg/build/generate_candidates"
cfg     = "/home/chris/gpg/cfg/params.cfg"

proc = subprocess.run([gpg_bin, cfg, pcd_f],
                      check=True,
                      stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE,
                      text=True)                                   # capture_output=True in ≥3.7  :contentReference[oaicite:1]{index=1}

log = proc.stdout + "\n" + proc.stderr
pathlib.Path("gpg_run.log").write_text(log)
print("Full console output saved to gpg_run.log")

# ---------- 3. extract numeric grasp lines ------------------ #
pat = re.compile(r"""
    ^\s*(?:\d+\s+)?                 # optional integer ID
    ([\d\.\+\-eE]+\s+){7}           # seven more floats…
    [\d\.\+\-eE]+                   # …and the 8th
    \s*$""", re.VERBOSE)

numeric_lines = [ln for ln in log.splitlines() if pat.match(ln)]
print(f"Found {len(numeric_lines)} numeric grasp lines")

if not numeric_lines:
    print("⚠ No candidates found - check gpg_run.log for warnings.")
    sys.exit(1)

# ---------- 4. convert to YAML ------------------------------ #
grasps = []
for i, ln in enumerate(numeric_lines):
    nums = list(map(float, re.split(r"\s+", ln.strip())[-8:]))  # keep last 8 floats
    x, y, z, qx, qy, qz, qw, fw = nums
    
    pos_local = np.array([x, y, z])
    pos_world = pos_local # + mesh_center

    R = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
    # R_tool_down = o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi, 0, 0])
    # R = R_tool_down @ R_from_GPG  # tool down in GPG frame
    # minimal XYZ-intrinsic RPY extraction
    roll  = atan2(R[2,1], R[2,2])
    pitch = asin(-R[2,0])
    yaw   = atan2(R[1,0], R[0,0])

    grasps.append({
        "name": f"gpg_{i}",
        "position": pos_world.tolist(),
        "orientation": [roll, pitch, yaw],
        "finger_width": fw,
    })

yaml.safe_dump({"grasps": grasps},
               open("banana_grasps.yaml", "w"),
               default_flow_style=False)
print(f"saved {len(grasps)} grasps ➜ banana_grasps.yaml")


# visualize the point cloud with grasps
from scipy.spatial.transform import Rotation as R

def make_frames(grasps, scale=0.03):
    frames = []
    for g in grasps:
        R_mat = R.from_euler("xyz", g["orientation"]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3,  3] = g["position"]
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        frame.transform(T)
        frames.append(frame)
    return frames

grasp_frames = make_frames(grasps)
o3d.visualization.draw_geometries([pcd, *grasp_frames])
