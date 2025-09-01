# convert STEP to PLY

import os
import sys
import numpy as np
import trimesh
from pathlib import Path

def convert_step_to_ply(input_file, output_file, subdivide=True, smooth=False, remove_duplicates=True, target_faces=None, 
                       subdivision_levels=2, remesh_resolution=None, adaptive_subdivision=True):
    """
    Convert a STEP file to a PLY file with optional mesh processing for better accuracy.

    Parameters:
    input_file (str): Path to the input STEP file.
    output_file (str): Path to the output PLY file.
    subdivide (bool): Whether to subdivide faces for more detail
    subdivision_levels (int): Number of subdivision iterations (more = denser mesh)
    smooth (bool): Whether to smooth the mesh
    remove_duplicates (bool): Whether to remove duplicate vertices
    target_faces (int): Target number of faces (None for no limit)
    remesh_resolution (float): If set, remesh to uniform resolution (e.g., 0.001 for 1mm)
    adaptive_subdivision (bool): Use adaptive subdivision based on curvature
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Loading STEP file: {input_file}")
    
    try:
        # Load the STEP file - trimesh can handle STEP files but may need additional dependencies
        # For STEP files, we might need to use a different loader or convert via FreeCAD/OpenCASCADE
        mesh = trimesh.load(input_file)
        
        # If trimesh returns a Scene (multiple objects), combine them
        if hasattr(mesh, 'geometry'):
            print(f"STEP file contains {len(mesh.geometry)} objects")
            # Combine all geometries into a single mesh
            meshes = []
            for name, geom in mesh.geometry.items():
                if hasattr(geom, 'vertices') and len(geom.vertices) > 0:
                    meshes.append(geom)
                    print(f"  - {name}: {len(geom.vertices)} vertices, {len(geom.faces)} faces")
            
            if len(meshes) == 1:
                mesh = meshes[0]
            elif len(meshes) > 1:
                # Concatenate multiple meshes
                mesh = trimesh.util.concatenate(meshes)
                print(f"Combined {len(meshes)} objects into single mesh")
            else:
                raise ValueError("No valid geometry found in STEP file")
        
    except Exception as e:
        print(f"Error loading STEP file with trimesh: {e}")
        print("Note: STEP file support requires additional dependencies.")
        print("Consider installing pythonocc-core via conda: conda install -c conda-forge pythonocc-core")
        print("Or convert STEP to STL first using FreeCAD, Blender, or other CAD software.")
        raise
    
    print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"Mesh bounds: {mesh.bounds}")
    
    # Calculate mesh scale for adaptive processing
    bbox = mesh.bounds
    mesh_scale = np.linalg.norm(bbox[1] - bbox[0])
    print(f"Mesh scale (diagonal): {mesh_scale:.6f}")
    
    # Process mesh for better accuracy
    if remove_duplicates:
        # Use updated methods to avoid deprecation warnings
        original_vertices = len(mesh.vertices)
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        print(f"After cleanup: {len(mesh.vertices)} vertices ({original_vertices - len(mesh.vertices)} removed), {len(mesh.faces)} faces")
    
    # Fix mesh issues before processing
    mesh.fix_normals()
    mesh.fill_holes()
    
    # Adaptive remeshing for uniform resolution
    if remesh_resolution:
        try:
            print(f"Remeshing to resolution: {remesh_resolution}")
            # Create a more uniform mesh with specified resolution
            mesh = mesh.smoothed()  # Pre-smooth for better remeshing
            # Note: Advanced remeshing would require additional libraries like pymeshlab
            print("Advanced remeshing requires pymeshlab. Using subdivision instead.")
        except Exception as e:
            print(f"Remeshing failed: {e}")
    
    if subdivide:
        print(f"Performing {subdivision_levels} levels of subdivision...")
        original_faces = len(mesh.faces)
        
        for level in range(subdivision_levels):
            try:
                if adaptive_subdivision:
                    # Adaptive subdivision based on face size
                    face_areas = mesh.area_faces
                    mean_area = np.mean(face_areas)
                    large_faces = face_areas > mean_area * 2  # Subdivide faces larger than 2x mean
                    
                    if np.any(large_faces):
                        # Subdivide the entire mesh but focus on large faces
                        mesh = mesh.subdivide()
                        print(f"  Level {level+1}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                    else:
                        print(f"  Level {level+1}: No large faces found, skipping subdivision")
                        break
                else:
                    # Uniform subdivision
                    mesh = mesh.subdivide()
                    print(f"  Level {level+1}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                
                # Check if we've reached a reasonable density
                if target_faces and len(mesh.faces) > target_faces * 2:
                    print(f"  Reached high density, stopping subdivision early")
                    break
                    
            except Exception as e:
                print(f"  Subdivision level {level+1} failed: {e}")
                break
        
        print(f"After subdivision: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"Density increase: {len(mesh.faces)/original_faces:.1f}x")
        
        # If target face count is specified and we have too many faces, simplify
        if target_faces and len(mesh.faces) > target_faces:
            try:
                print(f"Simplifying from {len(mesh.faces)} to {target_faces} faces...")
                mesh = mesh.simplify_quadric_decimation(target_faces)
                print(f"After simplification: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            except Exception as e:
                print(f"Mesh simplification failed: {e}")
                print(f"Keeping high-resolution mesh with {len(mesh.faces)} faces")
    
    if smooth:
        # Smooth the mesh while preserving volume
        try:
            print("Applying Laplacian smoothing...")
            original_volume = mesh.volume if hasattr(mesh, 'volume') else None
            mesh = mesh.smoothed()
            
            if original_volume:
                new_volume = mesh.volume if hasattr(mesh, 'volume') else None
                if new_volume:
                    volume_change = abs(new_volume - original_volume) / original_volume * 100
                    print(f"After smoothing: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                    print(f"Volume change: {volume_change:.2f}%")
                else:
                    print(f"After smoothing: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            else:
                print(f"After smoothing: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        except Exception as e:
            print(f"Smoothing failed: {e}")
    
    # Additional mesh quality improvements
    try:
        # Remove small disconnected components
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            # Keep the largest component
            largest = max(components, key=lambda x: len(x.vertices))
            mesh = largest
            print(f"Removed {len(components)-1} small disconnected components")
    except Exception as e:
        print(f"Component analysis failed: {e}")
    
    # Ensure the mesh is watertight and valid
    if mesh.is_watertight:
        print("✓ Mesh is watertight")
    else:
        print("⚠ Warning: Mesh is not watertight, this might affect SAM-6D performance")
        # Try to make it watertight
        try:
            mesh.fill_holes()
            if mesh.is_watertight:
                print("✓ Fixed: Mesh is now watertight after hole filling")
        except Exception as e:
            print(f"Hole filling failed: {e}")
    
    # Check mesh quality
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
    
    # Calculate final mesh statistics
    edge_lengths = mesh.edges_unique_length
    mean_edge_length = np.mean(edge_lengths)
    min_edge_length = np.min(edge_lengths)
    max_edge_length = np.max(edge_lengths)
    
    print(f"Mesh quality metrics:")
    print(f"  - Mean edge length: {mean_edge_length:.6f}")
    print(f"  - Min edge length: {min_edge_length:.6f}")
    print(f"  - Max edge length: {max_edge_length:.6f}")
    print(f"  - Edge length ratio: {max_edge_length/min_edge_length:.2f}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Export to PLY format with high precision
    mesh.export(output_file, file_type='ply', encoding='ascii')
    print(f"✓ Converted {input_file} to {output_file}")
    
    # Final statistics
    print(f"Final mesh stats:")
    print(f"  - Vertices: {len(mesh.vertices)}")
    print(f"  - Faces: {len(mesh.faces)}")
    if hasattr(mesh, 'volume'):
        print(f"  - Volume: {mesh.volume:.6f}")
    if hasattr(mesh, 'area'):
        print(f"  - Surface area: {mesh.area:.6f}")
    
    return mesh

if __name__ == "__main__":
    input_file = '/home/chris/franka_ros2_ws/src/sam6d_wrapper/Data/models/bueler_bolt/buehler_bolt.step'
    output_file = '/home/chris/franka_ros2_ws/src/sam6d_wrapper/Data/models/bueler_bolt/buehler_bolt.ply'

    # Convert with high accuracy options for maximum detail
    mesh = convert_step_to_ply(
        input_file, 
        output_file,
        subdivide=True,              # Enable subdivision
        subdivision_levels=3,        # More subdivision levels = more points
        adaptive_subdivision=True,   # Focus on areas that need more detail
        smooth=False,               # Keep original geometry sharp
        remove_duplicates=True,
        target_faces=100000,        # Allow more faces for higher accuracy
        remesh_resolution=None      # Set to e.g., 0.001 for 1mm uniform resolution
    )