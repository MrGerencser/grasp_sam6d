import os
import sys
import numpy as np
import trimesh

def convert_obj_to_ply(input_file, output_file, subdivide=False, smooth=False, remove_duplicates=True):
    """
    Convert an OBJ file to a PLY file with optional mesh processing.
    """
    # Load the OBJ file
    mesh = trimesh.load(input_file, file_type='obj')
    
    # If the loaded object is a Scene, merge all geometries into one mesh
    if isinstance(mesh, trimesh.Scene):
        print("OBJ contains multiple meshes, merging into a single mesh...")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    
    print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Process mesh for better detail
    if remove_duplicates:
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        print(f"After cleanup: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    if subdivide:
        # Subdivide faces to increase detail
        mesh = mesh.subdivide()
        print(f"After subdivision: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    if smooth:
        # Smooth the mesh
        mesh = mesh.smoothed()
        print(f"After smoothing: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Ensure the mesh is watertight and valid
    if not mesh.is_watertight:
        print("Warning: Mesh is not watertight, this might affect SAM-6D performance")
    
    # Export to PLY format with high precision
    mesh.export(output_file, file_type='ply', encoding='ascii')
    print(f"Converted {input_file} to {output_file}")
    
    return mesh

def convert_obj_to_ply_highres(input_file, output_file, target_edge_length=None):
    """
    Convert OBJ to PLY with high-resolution uniform remeshing
    """
    # Load the OBJ file
    mesh = trimesh.load(input_file, file_type='obj')
    
    if isinstance(mesh, trimesh.Scene):
        print("OBJ contains multiple meshes, merging...")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    
    print(f"Original: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Calculate appropriate edge length if not provided
    if target_edge_length is None:
        # Use 1/20th of the bounding box diagonal as target edge length
        bbox_diagonal = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
        target_edge_length = bbox_diagonal / 20.0
        print(f"Auto-calculated target edge length: {target_edge_length:.4f}")
    
    # Clean up first
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    
    # Multiple subdivisions for extreme detail
    for i in range(5):  # 5 levels = 32x more faces
        mesh = mesh.subdivide()
        print(f"Subdivision {i+1}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Stop if we get enough detail
        if len(mesh.faces) > 100000:
            print("Reached 100k+ faces, stopping subdivision")
            break
    
    # Fill holes and fix normals
    mesh.fill_holes()
    mesh.fix_normals()
    
    print(f"Final high-res mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Export
    mesh.export(output_file, file_type='ply', encoding='ascii')
    print(f"Saved detailed mesh to {output_file}")
    
    return mesh

def convert_obj_to_ply_ultimate(input_file, output_file):
    """
    Ultimate detail conversion using multiple techniques
    """
    # Load the OBJ file
    mesh = trimesh.load(input_file, file_type='obj')
    
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    
    print(f"Original: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Step 1: Clean and prepare
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    mesh.fill_holes()
    
    # Step 2: Maximum subdivisions (up to 1M faces)
    for i in range(10):  # Up to 10 levels
        mesh = mesh.subdivide()
        print(f"Subdivision {i+1}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        if len(mesh.faces) > 1000000:  # Stop at 1M faces
            print("Reached 1 million faces!")
            break
    
    # Step 3: Smooth for better surface quality
    mesh = mesh.smoothed()
    
    # Step 4: Final cleanup
    mesh.fix_normals()
    mesh.update_faces(mesh.unique_faces())
    
    print(f"Ultimate detail: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Export
    mesh.export(output_file, file_type='ply', encoding='ascii')
    print(f"Ultimate detail mesh saved!")
    
    return mesh

def convert_obj_to_ply_extreme(input_file, output_file, subdivision_levels=7, target_faces=500000):
    """
    Convert OBJ to PLY with extreme detail
    """
    # Load the OBJ file
    mesh = trimesh.load(input_file, file_type='obj')
    
    if isinstance(mesh, trimesh.Scene):
        print("OBJ contains multiple meshes, merging...")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    
    print(f"Original: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Clean up first
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    mesh.fill_holes()
    
    # Extreme subdivisions
    for i in range(subdivision_levels):
        mesh = mesh.subdivide()
        print(f"Subdivision {i+1}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        if len(mesh.faces) >= target_faces:
            print(f"Reached target of {target_faces} faces")
            break

    # Additional smoothing pass for better geometry (skip for large meshes)
    if len(mesh.faces) < 100000:  # Only smooth smaller meshes
        mesh = trimesh.graph.smooth_shade(mesh)
        print(f"After smoothing: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    else:
        print(f"Skipping smoothing for large mesh ({len(mesh.faces)} faces)")
    
    # Final cleanup
    mesh.fix_normals()
    
    # Export with maximum precision
    mesh.export(output_file, file_type='ply', encoding='ascii')
    print(f"Extreme detail mesh saved: {len(mesh.faces)} faces")
    
    return mesh

if __name__ == "__main__":
    input_file = '/home/chris/franka_ros2_ws/src/sam6d_wrapper/Data/models/cup/cup.obj'
    
    # Try all three methods
    print("=== Method 1: Extreme Subdivision ===")
    mesh1 = convert_obj_to_ply_extreme(
        input_file, 
        '/home/chris/franka_ros2_ws/src/sam6d_wrapper/Data/models/cup/cup_extreme.ply',
        subdivision_levels=8,
        target_faces=500000
    )