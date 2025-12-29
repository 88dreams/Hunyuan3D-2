#!/usr/bin/env python3
"""
SuGaR Mesh Extraction Inference Wrapper for RunPod

This script handles the server-side execution of SuGaR mesh extraction
on RunPod serverless workers. It wraps SuGaR's extraction pipeline
and manages input/output handling.

SuGaR Pipeline:
1. Load 3DGS PLY (convert from Lyra format if needed)
2. Run SuGaR regularization optimization
3. Extract mesh using Poisson reconstruction
4. Optionally refine with Gaussian+Mesh hybrid
5. Export textured mesh (GLB/OBJ)

Usage:
    python sugar_inference.py \
        --input_ply /path/to/input.ply \
        --output_dir /path/to/output \
        --output_name mesh_output \
        --regularization dn_consistency \
        --poisson_depth 10 \
        --refinement_time short \
        --export_glb
"""

import os
import sys
import argparse
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sugar-inference")

# =============================================================================
# PATHS
# =============================================================================

SUGAR_DIR = os.environ.get("SUGAR_DIR", "/workspace/SuGaR")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/workspace/checkpoints")


def validate_sugar_installation() -> bool:
    """Check if SuGaR is properly installed."""
    required_files = [
        os.path.join(SUGAR_DIR, "train_full_pipeline.py"),
        os.path.join(SUGAR_DIR, "extract_mesh.py"),
        os.path.join(SUGAR_DIR, "sugar_extractors"),
    ]
    
    for path in required_files:
        if not os.path.exists(path):
            logger.error(f"SuGaR not found: {path}")
            return False
    
    return True


def convert_lyra_to_standard_ply(
    input_ply: str,
    output_ply: str,
) -> bool:
    """
    Convert Lyra's PyTorch tensor PLY to standard 3DGS PLY format.
    
    Lyra outputs Gaussians as a PyTorch tensor saved with torch.save().
    We need to convert this to standard PLY format for SuGaR.
    """
    try:
        import torch
        import numpy as np
        from plyfile import PlyData, PlyElement
        
        logger.info(f"Converting Lyra PLY: {input_ply}")
        
        # Load Lyra's tensor format
        data = torch.load(input_ply, map_location="cpu", weights_only=False)
        
        if isinstance(data, torch.Tensor):
            gaussians = data.numpy()
        elif isinstance(data, dict):
            # Handle dictionary format
            if "gaussians" in data:
                gaussians = data["gaussians"]
                if isinstance(gaussians, torch.Tensor):
                    gaussians = gaussians.numpy()
            else:
                raise ValueError(f"Unknown Lyra format: keys={list(data.keys())}")
        else:
            raise ValueError(f"Unknown Lyra format: type={type(data)}")
        
        logger.info(f"Loaded {len(gaussians)} Gaussians, shape: {gaussians.shape}")
        
        # Lyra format: [N, 14] = xyz(3) + scale(3) + rotation(4) + opacity(1) + rgb(3)
        # Standard 3DGS: xyz, normals, f_dc, f_rest, opacity, scale, rotation
        
        num_points = len(gaussians)
        
        if gaussians.shape[1] == 14:
            # Lyra 14-feature format
            positions = gaussians[:, 0:3]
            scales = gaussians[:, 3:6]
            rotations = gaussians[:, 6:10]
            opacities = gaussians[:, 10:11]
            colors = gaussians[:, 11:14]
            
            # Convert RGB to SH DC coefficients
            # SH DC = (color - 0.5) / 0.28209479177387814
            sh_dc = (colors - 0.5) / 0.28209479177387814
        else:
            raise ValueError(f"Unexpected Lyra format with {gaussians.shape[1]} features")
        
        # Create PLY vertex data
        dtype = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ]
        
        vertices = np.zeros(num_points, dtype=dtype)
        vertices["x"] = positions[:, 0]
        vertices["y"] = positions[:, 1]
        vertices["z"] = positions[:, 2]
        vertices["nx"] = 0
        vertices["ny"] = 0
        vertices["nz"] = 1
        vertices["f_dc_0"] = sh_dc[:, 0]
        vertices["f_dc_1"] = sh_dc[:, 1]
        vertices["f_dc_2"] = sh_dc[:, 2]
        vertices["opacity"] = opacities[:, 0]
        vertices["scale_0"] = np.log(scales[:, 0] + 1e-8)  # Log scale
        vertices["scale_1"] = np.log(scales[:, 1] + 1e-8)
        vertices["scale_2"] = np.log(scales[:, 2] + 1e-8)
        vertices["rot_0"] = rotations[:, 0]
        vertices["rot_1"] = rotations[:, 1]
        vertices["rot_2"] = rotations[:, 2]
        vertices["rot_3"] = rotations[:, 3]
        
        # Write PLY
        el = PlyElement.describe(vertices, "vertex")
        PlyData([el], text=False).write(output_ply)
        
        logger.info(f"Converted PLY saved: {output_ply}")
        return True
    
    except Exception as e:
        logger.error(f"Lyra conversion failed: {e}")
        return False


def detect_input_format(input_ply: str) -> str:
    """Detect the format of the input PLY file."""
    try:
        import torch
        
        # Try loading as PyTorch tensor (Lyra format)
        try:
            data = torch.load(input_ply, map_location="cpu", weights_only=False)
            if isinstance(data, (torch.Tensor, dict)):
                return "lyra"
        except:
            pass
        
        # Check PLY header for standard 3DGS
        with open(input_ply, "rb") as f:
            header = f.read(2048).decode("utf-8", errors="ignore")
        
        if "f_dc_0" in header:
            return "standard_3dgs"
        elif "red" in header and "green" in header:
            return "point_cloud"
        
        return "unknown"
    
    except Exception as e:
        logger.warning(f"Format detection failed: {e}")
        return "unknown"


def create_colmap_structure(
    input_ply: str,
    work_dir: str,
) -> str:
    """
    Create a minimal COLMAP-like directory structure for SuGaR.
    
    SuGaR expects a COLMAP scene structure with:
    - sparse/0/cameras.bin
    - sparse/0/images.bin
    - sparse/0/points3D.bin
    - images/
    
    For PLY-only input, we create synthetic camera data.
    """
    import numpy as np
    
    # Create directory structure
    sparse_dir = os.path.join(work_dir, "sparse", "0")
    images_dir = os.path.join(work_dir, "images")
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    # Copy PLY to expected location
    gs_output_dir = os.path.join(work_dir, "output", "point_cloud", "iteration_7000")
    os.makedirs(gs_output_dir, exist_ok=True)
    shutil.copy(input_ply, os.path.join(gs_output_dir, "point_cloud.ply"))
    
    # Create minimal COLMAP binary files
    # These are empty/minimal since we're using pre-trained Gaussians
    
    # cameras.bin - single pinhole camera
    # Format: num_cameras(8) + [camera_id(8), model_id(4), width(8), height(8), params...]
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")
    with open(cameras_bin, "wb") as f:
        import struct
        # 1 camera, PINHOLE model (id=1), 1024x1024, fx=500, fy=500, cx=512, cy=512
        f.write(struct.pack("<Q", 1))  # num_cameras
        f.write(struct.pack("<I", 1))  # camera_id
        f.write(struct.pack("<i", 1))  # model_id (PINHOLE)
        f.write(struct.pack("<Q", 1024))  # width
        f.write(struct.pack("<Q", 1024))  # height
        f.write(struct.pack("<d", 500.0))  # fx
        f.write(struct.pack("<d", 500.0))  # fy
        f.write(struct.pack("<d", 512.0))  # cx
        f.write(struct.pack("<d", 512.0))  # cy
    
    # images.bin - empty (no registered images needed for PLY input)
    images_bin = os.path.join(sparse_dir, "images.bin")
    with open(images_bin, "wb") as f:
        import struct
        f.write(struct.pack("<Q", 0))  # num_images = 0
    
    # points3D.bin - empty (we use Gaussians instead)
    points_bin = os.path.join(sparse_dir, "points3D.bin")
    with open(points_bin, "wb") as f:
        import struct
        f.write(struct.pack("<Q", 0))  # num_points = 0
    
    logger.info(f"Created COLMAP structure in: {work_dir}")
    return work_dir


def load_3dgs_as_pointcloud(input_ply: str) -> tuple:
    """
    Load a 3DGS PLY file and extract positions and colors.
    
    Returns:
        (positions, colors) as numpy arrays
    """
    import numpy as np
    from plyfile import PlyData
    
    logger.info(f"Loading 3DGS PLY: {input_ply}")
    
    ply_data = PlyData.read(input_ply)
    vertex = ply_data['vertex']
    
    # Extract positions
    positions = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    
    # Extract colors from SH DC coefficients (f_dc_0, f_dc_1, f_dc_2)
    if 'f_dc_0' in vertex.data.dtype.names:
        # 3DGS format with SH coefficients
        sh_dc = np.vstack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']]).T
        # Convert SH DC to RGB: color = sh * 0.28209479177387814 + 0.5
        colors = sh_dc * 0.28209479177387814 + 0.5
        colors = np.clip(colors, 0, 1)
    elif 'red' in vertex.data.dtype.names:
        # Standard point cloud with RGB
        colors = np.vstack([
            vertex['red'] / 255.0,
            vertex['green'] / 255.0,
            vertex['blue'] / 255.0
        ]).T
    else:
        # No color info, use white
        colors = np.ones((len(positions), 3))
    
    logger.info(f"Loaded {len(positions)} points")
    return positions, colors


def run_poisson_reconstruction(
    input_ply: str,
    output_dir: str,
    output_name: str,
    poisson_depth: int = 10,
    target_vertices: int = 1_000_000,
    export_glb: bool = True,
    export_obj: bool = False,
) -> Dict[str, Any]:
    """
    Run Poisson surface reconstruction on a 3DGS point cloud.
    
    This is a direct PLY-to-mesh conversion that doesn't require training images.
    Uses Open3D's Poisson reconstruction algorithm.
    
    Returns:
        Dictionary with output paths and status
    """
    import numpy as np
    import open3d as o3d
    
    result = {
        "success": False,
        "mesh_path": None,
        "texture_path": None,
        "error": None,
    }
    
    try:
        # Load point cloud from 3DGS PLY
        input_format = detect_input_format(input_ply)
        logger.info(f"Detected input format: {input_format}")
        
        if input_format == "lyra":
            # Convert Lyra format to standard PLY first
            work_dir = tempfile.mkdtemp(prefix="lyra_convert_")
            converted_ply = os.path.join(work_dir, "converted.ply")
            if not convert_lyra_to_standard_ply(input_ply, converted_ply):
                result["error"] = "Failed to convert Lyra PLY"
                return result
            input_ply = converted_ply
        
        # Load positions and colors
        positions, colors = load_3dgs_as_pointcloud(input_ply)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        logger.info(f"Point cloud: {len(pcd.points)} points")
        
        # Estimate normals (required for Poisson)
        logger.info("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
        
        # Run Poisson reconstruction
        logger.info(f"Running Poisson reconstruction (depth={poisson_depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=poisson_depth,
            width=0,
            scale=1.1,
            linear_fit=False,
        )
        
        logger.info(f"Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
        
        # Remove low-density vertices (cleanup)
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.01)  # Remove bottom 1%
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        logger.info(f"After cleanup: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
        
        # Decimate if needed
        if len(mesh.triangles) > target_vertices * 2:
            target_faces = target_vertices * 2
            logger.info(f"Decimating to {target_faces} faces...")
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        
        # Transfer colors from point cloud to mesh vertices
        logger.info("Transferring vertex colors...")
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.zeros((len(mesh.vertices), 3))
        )
        
        # Build KD-tree for color transfer
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        mesh_vertices = np.asarray(mesh.vertices)
        mesh_colors = np.zeros((len(mesh_vertices), 3))
        
        for i, vertex in enumerate(mesh_vertices):
            [_, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 1)
            mesh_colors[i] = colors[idx[0]]
        
        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
        
        # Clean up mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        # Save output
        os.makedirs(output_dir, exist_ok=True)
        
        if export_glb:
            # Save as GLB via trimesh
            output_path = os.path.join(output_dir, f"{output_name}.glb")
            
            # Convert to trimesh for GLB export
            import trimesh
            
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            vertex_colors = np.asarray(mesh.vertex_colors)
            
            # Create trimesh with vertex colors
            # Convert colors to uint8 RGBA
            vertex_colors_uint8 = (vertex_colors * 255).astype(np.uint8)
            vertex_colors_rgba = np.hstack([
                vertex_colors_uint8,
                np.full((len(vertex_colors_uint8), 1), 255, dtype=np.uint8)
            ])
            
            tri_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors_rgba,
            )
            
            tri_mesh.export(output_path, file_type='glb')
            result["mesh_path"] = output_path
            logger.info(f"Saved GLB: {output_path}")
            
        elif export_obj:
            output_path = os.path.join(output_dir, f"{output_name}.obj")
            o3d.io.write_triangle_mesh(output_path, mesh)
            result["mesh_path"] = output_path
            logger.info(f"Saved OBJ: {output_path}")
        else:
            output_path = os.path.join(output_dir, f"{output_name}.ply")
            o3d.io.write_triangle_mesh(output_path, mesh)
            result["mesh_path"] = output_path
            logger.info(f"Saved PLY: {output_path}")
        
        result["success"] = True
        logger.info(f"Final mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
        
    except Exception as e:
        result["error"] = str(e)
        logger.exception("Poisson reconstruction error")
    
    return result


def run_sugar_pipeline(
    input_ply: str,
    output_dir: str,
    output_name: str,
    regularization: str = "dn_consistency",
    poisson_depth: int = 10,
    refinement_time: str = "short",
    target_vertices: int = 1_000_000,
    export_texture: bool = True,
    texture_resolution: int = 2048,
    export_glb: bool = True,
    export_obj: bool = False,
) -> Dict[str, Any]:
    """
    Run mesh extraction pipeline.
    
    NOTE: SuGaR's train_full_pipeline.py requires training images, which we don't have
    when converting from a standalone PLY file. Instead, we use Poisson reconstruction
    which works directly on the point cloud extracted from the 3DGS.
    
    Returns:
        Dictionary with output paths and status
    """
    logger.info("Using Poisson reconstruction (SuGaR requires training images)")
    
    return run_poisson_reconstruction(
        input_ply=input_ply,
        output_dir=output_dir,
        output_name=output_name,
        poisson_depth=poisson_depth,
        target_vertices=target_vertices,
        export_glb=export_glb,
        export_obj=export_obj,
    )


def convert_obj_to_glb(obj_path: str, glb_path: str) -> bool:
    """Convert OBJ to GLB using trimesh."""
    try:
        import trimesh
        
        # Load OBJ with materials
        mesh = trimesh.load(obj_path, force="mesh")
        
        # Export as GLB
        mesh.export(glb_path, file_type="glb")
        
        logger.info(f"Converted to GLB: {glb_path}")
        return True
    
    except Exception as e:
        logger.error(f"OBJ to GLB conversion failed: {e}")
        return False


def convert_ply_to_glb(ply_path: str, glb_path: str) -> bool:
    """Convert PLY mesh to GLB using trimesh."""
    try:
        import trimesh
        
        mesh = trimesh.load(ply_path)
        mesh.export(glb_path, file_type="glb")
        
        logger.info(f"Converted to GLB: {glb_path}")
        return True
    
    except Exception as e:
        logger.error(f"PLY to GLB conversion failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="SuGaR Mesh Extraction")
    parser.add_argument("--input_ply", required=True, help="Input 3DGS PLY file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--output_name", default="mesh", help="Output file name")
    parser.add_argument("--regularization", default="dn_consistency",
                       choices=["dn_consistency", "density", "sdf"])
    parser.add_argument("--poisson_depth", type=int, default=10)
    parser.add_argument("--refinement_time", default="short",
                       choices=["short", "medium", "long"])
    parser.add_argument("--target_vertices", type=int, default=1_000_000)
    parser.add_argument("--export_texture", action="store_true", default=True)
    parser.add_argument("--texture_resolution", type=int, default=2048)
    parser.add_argument("--export_glb", action="store_true", default=True)
    parser.add_argument("--export_obj", action="store_true")
    
    args = parser.parse_args()
    
    result = run_sugar_pipeline(
        input_ply=args.input_ply,
        output_dir=args.output_dir,
        output_name=args.output_name,
        regularization=args.regularization,
        poisson_depth=args.poisson_depth,
        refinement_time=args.refinement_time,
        target_vertices=args.target_vertices,
        export_texture=args.export_texture,
        texture_resolution=args.texture_resolution,
        export_glb=args.export_glb,
        export_obj=args.export_obj,
    )
    
    if result["success"]:
        print(f"SUCCESS: {result['mesh_path']}")
        sys.exit(0)
    else:
        print(f"FAILED: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()

