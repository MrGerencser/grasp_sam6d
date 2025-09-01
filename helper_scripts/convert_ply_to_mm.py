#!/usr/bin/env python3
import trimesh
from pathlib import Path

SCALE = 1.0

def convert(input_file: str, output_file: str) -> None:
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    mesh = trimesh.load(input_file, process=False)  # keep data as-is
    mesh.apply_scale(SCALE)                         # scale geometry
    mesh.apply_translation(-mesh.centroid)         # recenter to geometric center
    # mesh.apply_transform(mesh.principal_inertia_transform)  # align with principal axes

    # mesh.apply_transform([[0, 0, 1, 0],
    #                       [1, 0, 0, 0],
    #                       [0, 1, 0, 0],
    #                       [0, 0, 0, 1]])
    mesh.export(output_file)
    print(f"Wrote: {output_file}")

if __name__ == "__main__":
    input_file = "/home/chris/franka_ros2_ws/src/sam6d_wrapper/Data/models/cup/cup.ply"
    output_file = "/home/chris/franka_ros2_ws/src/sam6d_wrapper/Data/models/cup/cup_mm.ply"
    convert(input_file, output_file)

