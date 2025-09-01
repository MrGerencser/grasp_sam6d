# convert stl to ply

import os
import sys
import numpy as np
import trimesh

def convert_stl_to_ply(input_file, output_file, subdivide=True, smooth=False, remove_duplicates=True, target_faces=None):
    """
    Convert an STL file to a PLY file with optional mesh processing for better accuracy.
    
    Parameters:
    input_file (str): Path to the input STL file.
    output_file (str): Path to the output PLY file.
    subdivide (bool): Whether to subdivide faces for more detail
    smooth (bool): Whether to smooth the mesh
    remove_duplicates (bool): Whether to remove duplicate vertices
    target_faces (int): Target number of faces (None for no limit)
    """
    # Load the STL file
    mesh = trimesh.load(input_file, file_type='stl')
    
    print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"Mesh bounds: {mesh.bounds}")
    print(f"Mesh volume: {mesh.volume:.6f}")
    print(f"Mesh surface area: {mesh.area:.6f}")
    
    # Process mesh for better accuracy
    if remove_duplicates:
        # Use updated methods to avoid deprecation warnings
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        print(f"After cleanup: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Fix mesh issues
    mesh.fix_normals()
    mesh.fill_holes()
    
    if subdivide:
        # Subdivide faces to increase detail
        original_faces = len(mesh.faces)
        mesh = mesh.subdivide()
        print(f"After subdivision: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # If target face count is specified and we have too many faces, simplify
        if target_faces and len(mesh.faces) > target_faces:
            try:
                mesh = mesh.simplify_quadric_decimation(target_faces)
                print(f"After simplification to {target_faces}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            except Exception as e:
                print(f"Mesh simplification failed: {e}")
                print(f"Keeping original subdivision with {len(mesh.faces)} faces")
    
    if smooth:
        # Smooth the mesh while preserving volume
        mesh = mesh.smoothed()
        print(f"After smoothing: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Ensure the mesh is watertight and valid
    if mesh.is_watertight:
        print("✓ Mesh is watertight")
    else:
        print("⚠ Warning: Mesh is not watertight, this might affect SAM-6D performance")
        # Try to make it watertight
        mesh.fill_holes()
        if mesh.is_watertight:
            print("✓ Fixed: Mesh is now watertight after hole filling")
    
    # Check mesh quality (use try/except for compatibility)
    try:
        if hasattr(mesh, 'is_valid') and mesh.is_valid:
            print("✓ Mesh is valid")
        else:
            print("? Mesh validity unknown (checking basic properties)")
            # Basic validity checks
            if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                print("✓ Mesh has vertices and faces")
            else:
                print("⚠ Warning: Mesh is empty or invalid")
    except:
        print("? Unable to check mesh validity")
    
    # Export to PLY format with high precision
    mesh.export(output_file, file_type='ply', encoding='ascii')
    print(f"✓ Converted {input_file} to {output_file}")
    
    # Final statistics
    print(f"Final mesh stats:")
    print(f"  - Vertices: {len(mesh.vertices)}")
    print(f"  - Faces: {len(mesh.faces)}")
    print(f"  - Volume: {mesh.volume:.6f}")
    print(f"  - Surface area: {mesh.area:.6f}")
    
    return mesh

if __name__ == "__main__":
    input_file = '/home/chris/franka_ros2_ws/src/sam6d_wrapper/Data/models/master_chef_can/master_chef_can_mesh.stl'
    output_file = '/home/chris/franka_ros2_ws/src/sam6d_wrapper/Data/models/master_chef_can/master_chef_can_new.ply'

    # Convert with high accuracy options
    mesh = convert_stl_to_ply(
        input_file, 
        output_file,
        subdivide=True,        # Increase mesh detail
        smooth=False,          # Keep original geometry sharp
        remove_duplicates=True,
        target_faces=500000     # Limit faces if too many (adjust as needed)
    )