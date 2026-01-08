#!/usr/bin/env python3
"""
SV3D → 2DGS → Mesh Test Pipeline

This script runs the complete pipeline to test if everything works:
1. Generate multi-view images with SV3D
2. Estimate depth for each view
3. Train 2DGS on the views
4. Extract mesh from 2DGS

Usage:
    python test_pipeline.py --image /path/to/image.jpg
    python test_pipeline.py --image /path/to/image.jpg --quick  # Fast test (5k iterations)
"""

import os
import sys
import argparse
import time
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

class PipelineConfig:
    """Configuration for the test pipeline."""
    
    def __init__(self, quick_mode: bool = False):
        # SV3D settings
        self.sv3d_num_frames = 21
        self.sv3d_decode_chunk_size = 8
        self.sv3d_motion_bucket_id = 127
        self.sv3d_target_size = 576
        self.remove_background = False  # For interiors, keep background
        
        # View filtering
        self.front_arc_only = True
        self.arc_degrees = 180.0
        
        # 2DGS settings
        if quick_mode:
            self.two_dgs_iterations = 5000
            self.save_iterations = [1000, 3000, 5000]
        else:
            self.two_dgs_iterations = 15000
            self.save_iterations = [5000, 10000, 15000]
        
        self.depth_ratio = 1.0
        self.lambda_normal = 0.05
        
        # Mesh extraction
        self.target_triangles = 100000
        self.smooth_iterations = 2
        
        # Paths
        self.workspace = Path("/workspace/sv3d_2dgs")
        self.two_dgs_path = Path("/workspace/2d-gaussian-splatting")


# =============================================================================
# Stage 1: SV3D Multi-View Generation
# =============================================================================

def run_sv3d_generation(
    image_path: str,
    output_dir: Path,
    config: PipelineConfig,
) -> Tuple[List[Image.Image], List[np.ndarray]]:
    """
    Generate multi-view images using SV3D.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        config: Pipeline configuration
    
    Returns:
        Tuple of (list of images, list of camera poses)
    """
    print("\n" + "=" * 60)
    print("STAGE 1: SV3D Multi-View Generation")
    print("=" * 60)
    
    from diffusers import StableVideo3DPipeline
    
    # Load image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    print(f"  Original size: {original_size}")
    
    # Optionally remove background
    if config.remove_background:
        print("Removing background...")
        from rembg import remove, new_session
        session = new_session("u2net")
        image = remove(image, session=session)
        image = image.convert("RGB")
    
    # Resize to SV3D input size
    image.thumbnail((config.sv3d_target_size, config.sv3d_target_size), Image.Resampling.LANCZOS)
    
    # Create square canvas
    canvas = Image.new("RGB", (config.sv3d_target_size, config.sv3d_target_size), (255, 255, 255))
    offset = (
        (config.sv3d_target_size - image.width) // 2,
        (config.sv3d_target_size - image.height) // 2,
    )
    canvas.paste(image, offset)
    image = canvas
    print(f"  Preprocessed size: {image.size}")
    
    # Save preprocessed image
    preprocessed_path = output_dir / "preprocessed_input.png"
    image.save(preprocessed_path)
    
    # Load SV3D pipeline
    print("Loading SV3D pipeline...")
    pipe = StableVideo3DPipeline.from_pretrained(
        "stabilityai/sv3d",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    
    # Generate orbital video
    print(f"Generating {config.sv3d_num_frames} orbital views...")
    start_time = time.time()
    
    with torch.no_grad():
        result = pipe(
            image,
            num_frames=config.sv3d_num_frames,
            decode_chunk_size=config.sv3d_decode_chunk_size,
            motion_bucket_id=config.sv3d_motion_bucket_id,
        )
    
    frames = result.frames[0]
    generation_time = time.time() - start_time
    print(f"  Generated {len(frames)} frames in {generation_time:.1f}s")
    
    # Compute camera poses
    poses = compute_camera_poses(len(frames), elevation=10.0, radius=1.5)
    
    # Filter to front arc if requested
    if config.front_arc_only:
        frames, poses = filter_to_front_arc(frames, poses, config.arc_degrees)
        print(f"  Filtered to {len(frames)} front-arc views")
    
    # Save frames
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    for i, frame in enumerate(frames):
        frame.save(images_dir / f"frame_{i:04d}.png")
    
    # Save poses
    poses_array = np.stack(poses)
    np.save(output_dir / "poses.npy", poses_array)
    
    # Save intrinsics
    intrinsics = compute_intrinsics(frames[0].size)
    np.save(output_dir / "intrinsics.npy", intrinsics)
    
    # Cleanup
    del pipe
    torch.cuda.empty_cache()
    
    print(f"  Saved {len(frames)} frames to {images_dir}")
    
    return frames, poses


def compute_camera_poses(num_frames: int, elevation: float = 10.0, radius: float = 1.5) -> List[np.ndarray]:
    """Compute camera poses for SV3D orbital trajectory."""
    poses = []
    
    for i in range(num_frames):
        azimuth = (360.0 / num_frames) * i
        pose = orbital_to_camera_matrix(azimuth, elevation, radius)
        poses.append(pose)
    
    return poses


def orbital_to_camera_matrix(azimuth: float, elevation: float, radius: float) -> np.ndarray:
    """Convert orbital parameters to 4x4 camera-to-world matrix."""
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    
    x = radius * np.cos(el_rad) * np.sin(az_rad)
    y = radius * np.sin(el_rad)
    z = radius * np.cos(el_rad) * np.cos(az_rad)
    position = np.array([x, y, z])
    
    forward = -position / np.linalg.norm(position)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = position
    
    return pose


def filter_to_front_arc(
    frames: List[Image.Image],
    poses: List[np.ndarray],
    arc_degrees: float,
) -> Tuple[List[Image.Image], List[np.ndarray]]:
    """Filter frames to front-facing arc only."""
    arc_half = arc_degrees / 2
    
    filtered_frames = []
    filtered_poses = []
    
    num_frames = len(frames)
    for i, (frame, pose) in enumerate(zip(frames, poses)):
        azimuth = (360.0 / num_frames) * i
        if azimuth > 180:
            azimuth -= 360
        
        if -arc_half <= azimuth <= arc_half:
            filtered_frames.append(frame)
            filtered_poses.append(pose)
    
    return filtered_frames, filtered_poses


def compute_intrinsics(image_size: Tuple[int, int]) -> np.ndarray:
    """Compute camera intrinsic matrix."""
    W, H = image_size
    fov_deg = 46.8
    fov_rad = np.radians(fov_deg)
    focal_length = (W / 2) / np.tan(fov_rad / 2)
    
    return np.array([
        [focal_length, 0, W / 2],
        [0, focal_length, H / 2],
        [0, 0, 1],
    ])


# =============================================================================
# Stage 2: Depth Estimation
# =============================================================================

def run_depth_estimation(
    images_dir: Path,
    output_dir: Path,
    intrinsics: np.ndarray,
) -> Path:
    """
    Estimate depth for all images using Depth Anything V2.
    
    Args:
        images_dir: Directory with input images
        output_dir: Directory to save depth maps
        intrinsics: Camera intrinsic matrix
    
    Returns:
        Path to depths directory
    """
    print("\n" + "=" * 60)
    print("STAGE 2: Depth Estimation")
    print("=" * 60)
    
    from transformers import pipeline
    
    # Load depth pipeline
    print("Loading Depth Anything V2...")
    depth_pipe = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device=0,
    )
    
    # Create output directories
    depths_dir = output_dir / "depths"
    depths_dir.mkdir(exist_ok=True)
    
    # Process each image
    image_files = sorted(images_dir.glob("*.png"))
    print(f"Processing {len(image_files)} images...")
    
    for img_file in tqdm(image_files, desc="Estimating depth"):
        image = Image.open(img_file).convert("RGB")
        
        # Estimate depth
        result = depth_pipe(image)
        depth = np.array(result["depth"])
        
        # Normalize
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        # Save
        stem = img_file.stem
        np.save(depths_dir / f"{stem}_depth.npy", depth)
        
        # Save visualization
        depth_vis = (depth * 255).astype(np.uint8)
        Image.fromarray(depth_vis).save(depths_dir / f"{stem}_depth.png")
    
    # Cleanup
    del depth_pipe
    torch.cuda.empty_cache()
    
    print(f"  Saved depth maps to {depths_dir}")
    
    return depths_dir


# =============================================================================
# Stage 3: 2DGS Training
# =============================================================================

def run_2dgs_training(
    output_dir: Path,
    config: PipelineConfig,
) -> Path:
    """
    Train 2DGS on the generated multi-view images.
    
    Args:
        output_dir: Directory with images, poses, intrinsics
        config: Pipeline configuration
    
    Returns:
        Path to trained model
    """
    print("\n" + "=" * 60)
    print("STAGE 3: 2DGS Training")
    print("=" * 60)
    
    import subprocess
    
    # Prepare dataset in COLMAP format
    dataset_path = prepare_2dgs_dataset(output_dir)
    
    # Model output path
    model_path = output_dir / "model"
    model_path.mkdir(exist_ok=True)
    
    # Build training command
    train_script = config.two_dgs_path / "train.py"
    
    cmd = [
        "python", str(train_script),
        "-s", str(dataset_path),
        "-m", str(model_path),
        "--iterations", str(config.two_dgs_iterations),
        "--depth_ratio", str(config.depth_ratio),
        "--lambda_normal", str(config.lambda_normal),
        "--save_iterations", " ".join(str(i) for i in config.save_iterations),
    ]
    
    print(f"Training 2DGS for {config.two_dgs_iterations} iterations...")
    print(f"  Command: {' '.join(cmd)}")
    
    # Run training
    start_time = time.time()
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    process = subprocess.Popen(
        cmd,
        cwd=str(config.two_dgs_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    
    # Stream output
    for line in process.stdout:
        print(f"  [2DGS] {line.rstrip()}")
    
    process.wait()
    training_time = time.time() - start_time
    
    if process.returncode != 0:
        raise RuntimeError(f"2DGS training failed with code {process.returncode}")
    
    print(f"  Training complete in {training_time / 60:.1f} minutes")
    
    # Find output PLY
    ply_path = model_path / "point_cloud" / f"iteration_{config.two_dgs_iterations}" / "point_cloud.ply"
    
    if not ply_path.exists():
        # Try last saved iteration
        for iter_num in reversed(config.save_iterations):
            ply_path = model_path / "point_cloud" / f"iteration_{iter_num}" / "point_cloud.ply"
            if ply_path.exists():
                break
    
    print(f"  Output: {ply_path}")
    
    return model_path


def prepare_2dgs_dataset(output_dir: Path) -> Path:
    """Prepare dataset in COLMAP format for 2DGS."""
    from scipy.spatial.transform import Rotation
    
    dataset_path = output_dir / "dataset"
    dataset_path.mkdir(exist_ok=True)
    
    # Copy images
    images_src = output_dir / "images"
    images_dst = dataset_path / "images"
    if images_dst.exists():
        shutil.rmtree(images_dst)
    shutil.copytree(images_src, images_dst)
    
    # Load poses and intrinsics
    poses = np.load(output_dir / "poses.npy")
    intrinsics = np.load(output_dir / "intrinsics.npy")
    
    # Get image dimensions
    sample_img = Image.open(list(images_dst.glob("*.png"))[0])
    W, H = sample_img.size
    
    # Create COLMAP sparse directory
    sparse_dir = dataset_path / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    # Write cameras.txt
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    with open(sparse_dir / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")
    
    # Write images.txt
    image_files = sorted(images_dst.glob("*.png"))
    
    with open(sparse_dir / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for i, (pose, img_file) in enumerate(zip(poses, image_files)):
            # Invert pose (world-to-camera)
            R = pose[:3, :3]
            t = pose[:3, 3]
            R_inv = R.T
            t_inv = -R_inv @ t
            
            # Convert to quaternion
            rot = Rotation.from_matrix(R_inv)
            quat = rot.as_quat()  # (x, y, z, w)
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            
            f.write(f"{i+1} {qw} {qx} {qy} {qz} {t_inv[0]} {t_inv[1]} {t_inv[2]} 1 {img_file.name}\n")
            f.write("\n")
    
    # Write points3D.txt (empty)
    with open(sparse_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list (empty for 2DGS)\n")
    
    print(f"  Prepared dataset at {dataset_path}")
    
    return dataset_path


# =============================================================================
# Stage 4: Mesh Extraction
# =============================================================================

def run_mesh_extraction(
    model_path: Path,
    output_dir: Path,
    config: PipelineConfig,
) -> Path:
    """
    Extract mesh from trained 2DGS model.
    
    For this test, we use a simple point cloud to mesh approach.
    Full TSDF fusion would require rendering depth maps.
    
    Args:
        model_path: Path to trained 2DGS model
        output_dir: Directory for output mesh
        config: Pipeline configuration
    
    Returns:
        Path to output mesh
    """
    print("\n" + "=" * 60)
    print("STAGE 4: Mesh Extraction")
    print("=" * 60)
    
    import trimesh
    import open3d as o3d
    
    # Find PLY file
    ply_files = list(model_path.rglob("point_cloud.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No point_cloud.ply found in {model_path}")
    
    ply_path = sorted(ply_files)[-1]  # Use latest
    print(f"Loading point cloud: {ply_path}")
    
    # Load as Open3D point cloud
    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(f"  Points: {len(pcd.points)}")
    
    # Estimate normals if not present
    if not pcd.has_normals():
        print("  Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)
    
    # Poisson reconstruction
    print("  Running Poisson reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=9,
        width=0,
        scale=1.1,
        linear_fit=False,
    )
    
    # Remove low-density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    print(f"  Mesh vertices: {len(mesh.vertices)}")
    print(f"  Mesh triangles: {len(mesh.triangles)}")
    
    # Simplify if needed
    if len(mesh.triangles) > config.target_triangles:
        print(f"  Simplifying to {config.target_triangles} triangles...")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=config.target_triangles)
    
    # Smooth
    if config.smooth_iterations > 0:
        print(f"  Smoothing ({config.smooth_iterations} iterations)...")
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=config.smooth_iterations)
    
    # Compute normals
    mesh.compute_vertex_normals()
    
    # Convert to trimesh for GLB export
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    tmesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    
    # Export as GLB
    mesh_path = output_dir / "output_mesh.glb"
    tmesh.export(str(mesh_path), file_type="glb")
    
    # Also export as PLY for viewing
    mesh_ply_path = output_dir / "output_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_ply_path), mesh)
    
    print(f"  Exported mesh to {mesh_path}")
    print(f"  Final triangles: {len(tmesh.faces)}")
    
    return mesh_path


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(image_path: str, quick_mode: bool = False) -> Dict[str, Any]:
    """
    Run the complete SV3D → 2DGS → Mesh pipeline.
    
    Args:
        image_path: Path to input image
        quick_mode: If True, use faster settings for testing
    
    Returns:
        Dict with results and paths
    """
    print("\n" + "=" * 60)
    print("  SV3D → 2DGS → Mesh Pipeline")
    print("=" * 60)
    print(f"\nInput: {image_path}")
    print(f"Mode: {'Quick Test' if quick_mode else 'Full'}")
    
    config = PipelineConfig(quick_mode=quick_mode)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.workspace / "outputs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    results = {
        "input_image": image_path,
        "output_dir": str(output_dir),
        "quick_mode": quick_mode,
        "stages": {},
    }
    
    pipeline_start = time.time()
    
    try:
        # Stage 1: SV3D Generation
        stage_start = time.time()
        frames, poses = run_sv3d_generation(image_path, output_dir, config)
        results["stages"]["sv3d"] = {
            "num_frames": len(frames),
            "time_seconds": time.time() - stage_start,
        }
        
        # Stage 2: Depth Estimation
        stage_start = time.time()
        intrinsics = np.load(output_dir / "intrinsics.npy")
        depths_dir = run_depth_estimation(output_dir / "images", output_dir, intrinsics)
        results["stages"]["depth"] = {
            "time_seconds": time.time() - stage_start,
        }
        
        # Stage 3: 2DGS Training
        stage_start = time.time()
        model_path = run_2dgs_training(output_dir, config)
        results["stages"]["2dgs"] = {
            "iterations": config.two_dgs_iterations,
            "time_seconds": time.time() - stage_start,
        }
        
        # Stage 4: Mesh Extraction
        stage_start = time.time()
        mesh_path = run_mesh_extraction(model_path, output_dir, config)
        results["stages"]["mesh"] = {
            "mesh_path": str(mesh_path),
            "time_seconds": time.time() - stage_start,
        }
        
        results["success"] = True
        results["mesh_path"] = str(mesh_path)
        
    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
        print(f"\n❌ Pipeline failed: {e}")
    
    results["total_time_seconds"] = time.time() - pipeline_start
    
    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("  Pipeline Summary")
    print("=" * 60)
    
    if results["success"]:
        print(f"\n✅ Pipeline completed successfully!")
        print(f"\nTiming:")
        for stage, data in results["stages"].items():
            print(f"  - {stage}: {data.get('time_seconds', 0):.1f}s")
        print(f"  - Total: {results['total_time_seconds']:.1f}s ({results['total_time_seconds']/60:.1f} min)")
        print(f"\nOutputs:")
        print(f"  - Mesh: {results['mesh_path']}")
        print(f"  - Results: {results_path}")
    else:
        print(f"\n❌ Pipeline failed!")
        print(f"  Error: {results.get('error', 'Unknown')}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test SV3D → 2DGS → Mesh pipeline")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick mode (5k iterations)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    results = run_pipeline(args.image, quick_mode=args.quick)
    
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()

