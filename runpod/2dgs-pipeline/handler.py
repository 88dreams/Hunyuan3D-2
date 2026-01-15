"""
2DGS Pipeline Handler - Combined ViPE + 2DGS
Converts Gen3C/LTX-2 video(s) directly to 3D mesh in one endpoint.

Supports both single-video and multi-video modes for improved 3D reconstruction.

Input:
    {
        # Single video mode (legacy)
        "video_url": "https://...",  # OR
        "video_base64": "...",       # OR  
        "s3_input": {"bucket": "...", "key": "..."},
        
        # Multi-video mode (preferred for better reconstruction)
        "video_urls": ["https://...", "https://...", ...],  # 1-8 videos
        
        # Optional parameters
        "iterations": 5000,          # 2DGS training iterations
        "mesh_resolution": 512,      # Mesh extraction resolution
        "output_format": "glb",      # "glb", "obj", or "ply"
        "depth_threshold": 0.5       # Min depth coverage to keep frame (0.0-1.0)
    }

Output:
    {
        "status": "success",
        "mesh_url": "https://presigned-url...",
        "num_frames": 241,
        "elapsed_seconds": 600,
        "quality_stats": {           # Multi-video mode only
            "total_frames": 400,
            "valid_frames": 385,
            "skipped_frames": 15,
            "avg_depth_coverage": 0.78,
            "videos_processed": 4
        }
    }
"""

import os
import sys
import time
import json
import shutil
import tempfile
import subprocess
import base64
from pathlib import Path

import runpod
import requests
import numpy as np
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import zipfile

# Paths
VIPE_PATH = os.environ.get("VIPE_PATH", "/opt/vipe")
TDGS_PATH = os.environ.get("TDGS_PATH", "/opt/2d-gaussian-splatting")
PIPELINE_PATH = "/opt/pipeline"

sys.path.insert(0, VIPE_PATH)
sys.path.insert(0, TDGS_PATH)
sys.path.insert(0, PIPELINE_PATH)


def download_video(job_input: dict, work_dir: Path) -> Path:
    """Download video from URL, base64, or S3."""
    video_path = work_dir / "input.mp4"
    
    if "video_url" in job_input:
        url = job_input["video_url"]
        print(f"Downloading video from URL: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(video_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
    elif "video_base64" in job_input:
        print("Decoding base64 video...")
        video_data = base64.b64decode(job_input["video_base64"])
        with open(video_path, "wb") as f:
            f.write(video_data)
            
    elif "s3_input" in job_input:
        import boto3
        s3_config = job_input["s3_input"]
        print(f"Downloading from S3: {s3_config['bucket']}/{s3_config['key']}")
        s3 = boto3.client("s3")
        s3.download_file(s3_config["bucket"], s3_config["key"], str(video_path))
    else:
        raise ValueError("No video input provided. Use 'video_url', 'video_base64', or 's3_input'")
    
    print(f"Video saved to {video_path} ({video_path.stat().st_size / 1e6:.1f} MB)")
    return video_path


def download_videos(video_urls: List[str], work_dir: Path) -> List[Path]:
    """
    Download multiple videos from URLs.
    Returns list of paths to downloaded video files.
    """
    print(f"\n[Multi-Video] Downloading {len(video_urls)} videos...")
    video_paths = []
    
    def download_one(idx_url: Tuple[int, str]) -> Path:
        idx, url = idx_url
        video_path = work_dir / f"input_{idx:02d}.mp4"
        print(f"  [{idx+1}/{len(video_urls)}] Downloading: {url[:80]}...")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        with open(video_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        size_mb = video_path.stat().st_size / 1e6
        print(f"  [{idx+1}/{len(video_urls)}] Downloaded: {video_path.name} ({size_mb:.1f} MB)")
        return video_path
    
    # Download videos in parallel (max 4 concurrent)
    with ThreadPoolExecutor(max_workers=4) as executor:
        video_paths = list(executor.map(download_one, enumerate(video_urls)))
    
    print(f"[Multi-Video] All {len(video_paths)} videos downloaded")
    return video_paths


def filter_frames_by_depth_coverage(
    depth_dir: Path, 
    threshold: float = 0.5
) -> Tuple[List[int], float, int, int]:
    """
    Filter frames based on depth coverage.
    
    Args:
        depth_dir: Directory containing depth .npy files
        threshold: Minimum fraction of valid (non-zero) depth pixels
        
    Returns:
        Tuple of (valid_frame_indices, avg_coverage, total_frames, skipped_frames)
    """
    valid_frames = []
    total_coverage = 0.0
    skipped = 0
    
    # Find all depth files
    depth_files = sorted(depth_dir.glob("*.npy"))
    
    if not depth_files:
        # Try loading from zip
        zip_path = depth_dir.parent / "depth" / "input.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as z:
                depth_files = sorted([n for n in z.namelist() if n.endswith('.npy')])
                for i, name in enumerate(depth_files):
                    with z.open(name) as f:
                        depth = np.load(f)
                        valid_pixels = np.sum(depth > 0)
                        total_pixels = depth.size
                        coverage = valid_pixels / total_pixels
                        total_coverage += coverage
                        
                        if coverage >= threshold:
                            valid_frames.append(i)
                        else:
                            skipped += 1
                            print(f"    [Quality] Skipping frame {i}: depth coverage {coverage:.1%} < {threshold:.0%}")
        else:
            print(f"    [Quality] Warning: No depth files found in {depth_dir}")
            return list(range(100)), 0.0, 0, 0  # Fallback - keep all frames
    else:
        for i, depth_file in enumerate(depth_files):
            depth = np.load(depth_file)
            valid_pixels = np.sum(depth > 0)
            total_pixels = depth.size
            coverage = valid_pixels / total_pixels
            total_coverage += coverage
            
            if coverage >= threshold:
                valid_frames.append(i)
            else:
                skipped += 1
                print(f"    [Quality] Skipping frame {i}: depth coverage {coverage:.1%} < {threshold:.0%}")
    
    total_frames = len(depth_files) if depth_files else len(valid_frames) + skipped
    avg_coverage = total_coverage / total_frames if total_frames > 0 else 0.0
    
    print(f"    [Quality] Kept {len(valid_frames)}/{total_frames} frames (avg coverage: {avg_coverage:.1%})")
    return valid_frames, avg_coverage, total_frames, skipped


def run_vipe_multi(
    video_paths: List[Path], 
    work_dir: Path,
    depth_threshold: float = 0.5
) -> Tuple[List[Dict], Dict]:
    """
    Run ViPE on multiple videos and filter frames by depth coverage.
    
    Returns:
        Tuple of (list of vipe_outputs dicts, quality_stats dict)
    """
    print(f"\n[Multi-Video] Running ViPE on {len(video_paths)} videos...")
    
    all_vipe_outputs = []
    total_frames = 0
    total_valid = 0
    total_skipped = 0
    total_coverage = 0.0
    
    for idx, video_path in enumerate(video_paths):
        print(f"\n--- Video {idx+1}/{len(video_paths)}: {video_path.name} ---")
        
        vipe_output_dir = work_dir / f"vipe_{idx:02d}"
        vipe_outputs = run_vipe(video_path, vipe_output_dir.parent)
        
        # The run_vipe function creates "vipe" subdir, so adjust path
        actual_vipe_dir = vipe_output_dir.parent / "vipe"
        if actual_vipe_dir.exists():
            # Rename to indexed directory
            actual_vipe_dir.rename(vipe_output_dir)
            # Update paths in outputs dict
            vipe_outputs = {
                "poses": vipe_output_dir / "pose" / "input.npz",
                "intrinsics": vipe_output_dir / "intrinsics" / "input.npz", 
                "depth": vipe_output_dir / "depth" / "input.zip",
                "rgb": vipe_output_dir / "rgb" / "input.mp4",
                "video_idx": idx
            }
        
        # Filter frames by depth coverage
        depth_dir = vipe_output_dir / "depth"
        valid_frames, avg_cov, n_total, n_skipped = filter_frames_by_depth_coverage(
            depth_dir, threshold=depth_threshold
        )
        
        vipe_outputs["valid_frames"] = valid_frames
        vipe_outputs["depth_coverage"] = avg_cov
        all_vipe_outputs.append(vipe_outputs)
        
        total_frames += n_total
        total_valid += len(valid_frames)
        total_skipped += n_skipped
        total_coverage += avg_cov
    
    quality_stats = {
        "total_frames": total_frames,
        "valid_frames": total_valid,
        "skipped_frames": total_skipped,
        "avg_depth_coverage": round(total_coverage / len(video_paths), 3) if video_paths else 0,
        "videos_processed": len(video_paths)
    }
    
    print(f"\n[Multi-Video] ViPE complete: {total_valid}/{total_frames} frames kept")
    return all_vipe_outputs, quality_stats


def align_poses_frame0(poses_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    Align all pose sequences using frame 0 as the anchor point.
    
    Since all videos start from the same input image, frame 0 poses should
    represent the same camera position. We align all subsequent videos to
    video 1's frame 0 pose.
    
    Args:
        poses_list: List of (N, 4, 4) pose arrays, one per video
        
    Returns:
        List of aligned pose arrays (with frame 0 removed from videos 2+)
    """
    if len(poses_list) <= 1:
        return poses_list
    
    print(f"\n[Pose Alignment] Aligning {len(poses_list)} video pose sequences...")
    
    # Reference pose is frame 0 of first video
    reference_pose = poses_list[0][0]  # (4, 4)
    
    aligned_poses = [poses_list[0]]  # Keep all frames from video 1
    
    for idx, poses in enumerate(poses_list[1:], start=2):
        # Compute transform to align this video's frame 0 to reference
        source_pose = poses[0]  # Frame 0 of this video
        
        # Transform = reference * inverse(source)
        # This gives us the transform that maps source coordinate system to reference
        transform = reference_pose @ np.linalg.inv(source_pose)
        
        # Apply transform to all poses in this video
        aligned = np.array([transform @ p for p in poses])
        
        # Skip frame 0 (it's a duplicate of the reference)
        aligned_poses.append(aligned[1:])
        
        print(f"  Video {idx}: Aligned {len(poses)-1} poses (skipped duplicate frame 0)")
    
    return aligned_poses


def merge_vipe_outputs(
    vipe_outputs_list: List[Dict],
    output_dir: Path,
    depth_threshold: float = 0.5
) -> Tuple[Path, int]:
    """
    Merge multiple ViPE outputs into a single COLMAP-format dataset.
    
    Args:
        vipe_outputs_list: List of vipe output dicts from run_vipe_multi
        output_dir: Directory for merged output
        depth_threshold: Minimum depth coverage to keep frame
        
    Returns:
        Tuple of (path to merged 2DGS data directory, total frame count)
    """
    print(f"\n[Merge] Combining {len(vipe_outputs_list)} video outputs...")
    
    merged_dir = output_dir / "merged_vipe"
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and align poses
    all_poses = []
    all_intrinsics = []
    all_frames = []  # (video_idx, frame_idx) tuples
    
    for vipe_out in vipe_outputs_list:
        poses_data = np.load(vipe_out["poses"])
        poses = poses_data["data"]  # (N, 4, 4)
        all_poses.append(poses)
        
        intrinsics_data = np.load(vipe_out["intrinsics"])
        intrinsics = intrinsics_data["data"]  # (N, 3, 3) or (3, 3)
        all_intrinsics.append(intrinsics)
        
        # Track valid frames
        valid_frames = vipe_out.get("valid_frames", list(range(len(poses))))
        video_idx = vipe_out.get("video_idx", 0)
        for frame_idx in valid_frames:
            all_frames.append((video_idx, frame_idx))
    
    # Align poses using frame 0 anchor
    aligned_poses = align_poses_frame0(all_poses)
    
    # Flatten aligned poses and corresponding intrinsics
    merged_poses = []
    merged_intrinsics = []
    frame_mapping = []  # Maps merged index to (video_idx, original_frame_idx)
    
    for video_idx, (poses, intrinsics) in enumerate(zip(aligned_poses, all_intrinsics)):
        valid_frames = vipe_outputs_list[video_idx].get("valid_frames", list(range(len(poses))))
        
        # For video 0, use all valid frames
        # For video 1+, skip frame 0 (already done in align_poses_frame0)
        start_idx = 0 if video_idx == 0 else 0  # aligned_poses already skips frame 0
        
        for i, frame_idx in enumerate(valid_frames):
            # Skip frame 0 for videos after the first
            if video_idx > 0 and frame_idx == 0:
                continue
            
            # Adjust index for aligned poses (which skip frame 0)
            adjusted_idx = frame_idx if video_idx == 0 else frame_idx - 1
            
            if adjusted_idx < len(poses):
                merged_poses.append(poses[adjusted_idx])
                
                # Handle intrinsics (may be per-frame or shared)
                if len(intrinsics.shape) == 3:
                    merged_intrinsics.append(intrinsics[frame_idx] if frame_idx < len(intrinsics) else intrinsics[0])
                else:
                    merged_intrinsics.append(intrinsics)
                
                frame_mapping.append((video_idx, frame_idx))
    
    # Save merged poses and intrinsics
    merged_poses = np.array(merged_poses)
    merged_intrinsics = np.array(merged_intrinsics)
    
    pose_dir = merged_dir / "pose"
    pose_dir.mkdir(parents=True, exist_ok=True)
    np.savez(pose_dir / "input.npz", data=merged_poses)
    
    intrinsics_dir = merged_dir / "intrinsics"
    intrinsics_dir.mkdir(parents=True, exist_ok=True)
    np.savez(intrinsics_dir / "input.npz", data=merged_intrinsics)
    
    # Create merged depth directory with symlinks/copies
    merged_depth_dir = merged_dir / "depth"
    merged_depth_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract and merge depth maps
    for merged_idx, (video_idx, frame_idx) in enumerate(frame_mapping):
        src_zip = vipe_outputs_list[video_idx]["depth"]
        
        # Extract specific frame from source zip
        with zipfile.ZipFile(src_zip, 'r') as z:
            # ViPE names depth files as 000000.npy, 000001.npy, etc.
            src_name = f"{frame_idx:06d}.npy"
            if src_name in z.namelist():
                depth_data = z.read(src_name)
                dst_path = merged_depth_dir / f"{merged_idx:06d}.npy"
                with open(dst_path, 'wb') as f:
                    f.write(depth_data)
    
    # Create merged RGB directory
    merged_rgb_dir = merged_dir / "rgb"
    merged_rgb_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frames from each video
    import subprocess
    for video_idx, vipe_out in enumerate(vipe_outputs_list):
        video_path = vipe_out["rgb"]
        valid_frames = vipe_out.get("valid_frames", [])
        
        # Extract all frames from this video to temp dir
        temp_frames = merged_dir / f"temp_frames_{video_idx}"
        temp_frames.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-q:v", "2",
            str(temp_frames / "%06d.jpg")
        ]
        subprocess.run(cmd, capture_output=True)
    
    # Copy valid frames to merged rgb directory
    merged_idx = 0
    for video_idx, frame_idx in frame_mapping:
        temp_frames = merged_dir / f"temp_frames_{video_idx}"
        src_frame = temp_frames / f"{frame_idx+1:06d}.jpg"  # ffmpeg is 1-indexed
        
        if src_frame.exists():
            dst_frame = merged_rgb_dir / f"{merged_idx:06d}.jpg"
            shutil.copy(src_frame, dst_frame)
            merged_idx += 1
    
    # Cleanup temp directories
    for video_idx in range(len(vipe_outputs_list)):
        temp_frames = merged_dir / f"temp_frames_{video_idx}"
        if temp_frames.exists():
            shutil.rmtree(temp_frames)
    
    print(f"[Merge] Created merged dataset with {len(merged_poses)} frames")
    print(f"  Poses: {merged_dir / 'pose' / 'input.npz'}")
    print(f"  Intrinsics: {merged_dir / 'intrinsics' / 'input.npz'}")
    print(f"  Depth: {merged_depth_dir} ({len(list(merged_depth_dir.glob('*.npy')))} files)")
    print(f"  RGB: {merged_rgb_dir} ({len(list(merged_rgb_dir.glob('*.jpg')))} files)")
    
    return merged_dir, len(merged_poses)


def run_vipe(video_path: Path, output_dir: Path) -> dict:
    """
    Run ViPE to extract camera poses, intrinsics, and depth maps.
    Returns paths to output files.
    """
    print("\n" + "="*50)
    print("STEP 1: Running ViPE (pose + depth extraction)")
    print("="*50)
    
    vipe_output = output_dir / "vipe"
    vipe_output.mkdir(parents=True, exist_ok=True)
    
    # ViPE CLI is defined in pyproject.toml as: vipe = "vipe.cli.main:main"
    # Call it directly via Python to avoid PATH issues
    cmd = [
        sys.executable, "-c",
        f"from vipe.cli.main import main; main(['infer', '{video_path}', '--output', '{vipe_output}'])"
    ]
    
    print(f"Running ViPE infer on {video_path}")
    print(f"Output: {vipe_output}")
    
    # Set environment for ViPE
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True, 
        cwd=VIPE_PATH,
        env=env
    )
    
    if result.returncode != 0:
        print(f"ViPE STDERR: {result.stderr}")
        raise RuntimeError(f"ViPE failed: {result.stderr}")
    
    print(f"ViPE STDOUT: {result.stdout}")
    
    # List what ViPE actually created
    print(f"\nViPE output directory contents:")
    for root, dirs, files in os.walk(vipe_output):
        level = root.replace(str(vipe_output), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')
    
    # ViPE saves to subdirectories: pose/input.npz, intrinsics/input.npz, etc.
    outputs = {
        "poses": vipe_output / "pose" / "input.npz",
        "intrinsics": vipe_output / "intrinsics" / "input.npz", 
        "depth": vipe_output / "depth" / "input.zip",
        "rgb": vipe_output / "rgb" / "input.mp4"
    }
    
    for name, path in outputs.items():
        if not path.exists():
            raise FileNotFoundError(f"ViPE output not found: {path}")
        print(f"  ✓ {name}: {path}")
    
    return outputs


def convert_to_2dgs_format(vipe_dir: Path, output_dir: Path) -> Path:
    """
    Convert ViPE outputs to COLMAP format for 2DGS.
    Returns the 2DGS-ready data directory.
    """
    print("\n" + "="*50)
    print("STEP 2: Converting to 2DGS format")
    print("="*50)
    
    # Import the converter
    from vipe_to_2dgs import convert_vipe_to_2dgs
    
    data_dir = output_dir / "2dgs_data"
    
    # Converter expects vipe_dir containing pose/, intrinsics/, depth/, rgb/ subdirs
    convert_vipe_to_2dgs(
        vipe_dir=str(vipe_dir),
        output_dir=str(data_dir),
        video_name="input"  # ViPE saves files as input.npz, input.zip, etc.
    )
    
    print(f"  ✓ 2DGS data prepared at: {data_dir}")
    return data_dir


def init_points_from_depth(data_dir: Path) -> None:
    """
    Generate initial 3D point cloud from depth maps.
    Required for 2DGS training initialization.
    """
    print("\n" + "="*50)
    print("STEP 3: Generating initial point cloud")
    print("="*50)
    
    from init_points_from_depth import generate_points_from_depth
    
    generate_points_from_depth(str(data_dir))
    
    points_file = data_dir / "sparse" / "0" / "points3D.ply"
    if points_file.exists():
        print(f"  ✓ Point cloud generated: {points_file}")
    else:
        raise FileNotFoundError("Failed to generate point cloud")


def train_2dgs(data_dir: Path, model_dir: Path, iterations: int = 5000) -> Path:
    """
    Train 2D Gaussian Splatting model.
    Returns path to trained model.
    """
    print("\n" + "="*50)
    print(f"STEP 4: Training 2DGS ({iterations} iterations)")
    print("="*50)
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "train.py",
        "-s", str(data_dir),
        "-m", str(model_dir),
        "--iterations", str(iterations),
        "--save_iterations", str(iterations)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True, 
        cwd=TDGS_PATH,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    )
    
    if result.returncode != 0:
        print(f"2DGS STDERR: {result.stderr}")
        raise RuntimeError(f"2DGS training failed: {result.stderr}")
    
    print(f"  ✓ Model trained at: {model_dir}")
    return model_dir


def extract_mesh(
    data_dir: Path, 
    model_dir: Path, 
    mesh_resolution: int = 512,
    mesh_quality: str = "balanced",
    depth_trunc: float = None,
    voxel_size: float = None,
    num_cluster: int = None,
) -> Path:
    """
    Extract mesh from trained 2DGS model.
    
    Quality presets:
    - "fast": Quick extraction, lower quality (mesh_res=512, num_cluster=1)
    - "balanced": Good balance (mesh_res=512, num_cluster=50) [default]
    - "high": High quality (mesh_res=1024, voxel_size=0.004, num_cluster=100)
    - "ultra": Maximum quality (mesh_res=2048, voxel_size=0.002, num_cluster=200)
    
    Returns path to extracted mesh.
    """
    print("\n" + "="*50)
    print(f"STEP 5: Extracting mesh (quality: {mesh_quality}, resolution: {mesh_resolution})")
    print("="*50)
    
    # Quality presets - 2DGS native extraction parameters
    quality_presets = {
        "fast": {
            "mesh_res": 512,
            "depth_trunc": 6.0,
            "voxel_size": 0.01,
            "num_cluster": 1,
        },
        "balanced": {
            "mesh_res": 512,
            "depth_trunc": 4.0,
            "voxel_size": 0.006,
            "num_cluster": 50,
        },
        "high": {
            "mesh_res": 1024,
            "depth_trunc": 3.0,
            "voxel_size": 0.004,
            "num_cluster": 100,
        },
        "ultra": {
            "mesh_res": 2048,
            "depth_trunc": 2.0,
            "voxel_size": 0.002,
            "num_cluster": 200,
        },
    }
    
    # Get preset values, allow overrides
    preset = quality_presets.get(mesh_quality, quality_presets["balanced"])
    final_mesh_res = mesh_resolution if mesh_resolution != 512 else preset["mesh_res"]
    final_depth_trunc = depth_trunc if depth_trunc is not None else preset["depth_trunc"]
    final_voxel_size = voxel_size if voxel_size is not None else preset["voxel_size"]
    final_num_cluster = num_cluster if num_cluster is not None else preset["num_cluster"]
    
    print(f"  Mesh resolution: {final_mesh_res}")
    print(f"  Depth truncation: {final_depth_trunc}")
    print(f"  Voxel size: {final_voxel_size}")
    print(f"  Num clusters: {final_num_cluster}")
    
    # Find the iteration checkpoint
    ckpt_dirs = list((model_dir / "point_cloud").glob("iteration_*"))
    if not ckpt_dirs:
        raise FileNotFoundError("No checkpoint found in model directory")
    
    latest_iter = max(int(d.name.split("_")[1]) for d in ckpt_dirs)
    
    # Build command with high-quality 2DGS native extraction parameters
    cmd = [
        sys.executable, "render.py",
        "-s", str(data_dir),
        "-m", str(model_dir),
        "--iteration", str(latest_iter),
        "--skip_train",
        "--skip_test",
        "--unbounded",
        "--mesh_res", str(final_mesh_res),
        "--depth_trunc", str(final_depth_trunc),
        "--voxel_size", str(final_voxel_size),
        "--num_cluster", str(final_num_cluster),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=TDGS_PATH,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    )
    
    # Log output regardless of return code
    if result.stdout:
        print(f"Mesh extraction STDOUT: {result.stdout[-2000:]}")  # Last 2000 chars
    if result.stderr:
        print(f"Mesh extraction STDERR: {result.stderr}")
    
    # Find the mesh file - 2DGS with --unbounded outputs to train/ours_{iteration}/
    # This matches the working handler_2dgs.py behavior
    
    # Priority 1: Look in train/ours_{iteration}/ for unbounded mesh (this is where render.py outputs)
    train_dir = model_dir / "train" / f"ours_{latest_iter}"
    mesh_files = []
    
    if train_dir.exists():
        # Look for the post-processed mesh first, then raw
        for pattern in ["fuse_unbounded_post.ply", "fuse_unbounded.ply", "fuse_post.ply", "fuse.ply"]:
            candidates = list(train_dir.glob(pattern))
            if candidates:
                mesh_files = candidates
                break
    
    # Priority 2: Fallback to mesh/ directory
    if not mesh_files:
        mesh_dir = model_dir / "mesh"
        if mesh_dir.exists():
            mesh_files = list(mesh_dir.glob("fuse*.ply"))
    
    # Priority 3: Search recursively
    if not mesh_files:
        mesh_files = list(model_dir.glob("**/fuse*.ply"))
    
    # Check if we found a real mesh (should be > 1MB for a proper mesh)
    valid_mesh_files = []
    for f in mesh_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  Found: {f.name} ({size_mb:.2f} MB)")
        if size_mb > 0.5:  # Real meshes are usually > 500KB
            valid_mesh_files.append(f)
    
    if not valid_mesh_files:
        # Mesh extraction failed - log what we found
        print(f"  ⚠️ No valid mesh found. Model dir contents:")
        for item in model_dir.rglob("*.ply"):
            size_mb = item.stat().st_size / 1024 / 1024
            print(f"    - {item.relative_to(model_dir)}: {size_mb:.2f} MB")
        
        if result.returncode != 0:
            raise RuntimeError(f"Mesh extraction failed (exit code {result.returncode}). "
                             f"The 2DGS marching cubes step failed. Check logs for details.")
        else:
            raise FileNotFoundError("No valid mesh file found. The marching cubes extraction may have failed.")
    
    mesh_path = valid_mesh_files[0]
    print(f"  ✓ Mesh extracted: {mesh_path} ({mesh_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return mesh_path


def convert_mesh_format(mesh_path: Path, output_format: str) -> Path:
    """Convert mesh to requested format (glb, obj, ply)."""
    import trimesh
    
    if output_format == "ply":
        return mesh_path
    
    mesh = trimesh.load(mesh_path)
    output_path = mesh_path.with_suffix(f".{output_format}")
    
    mesh.export(output_path)
    print(f"  ✓ Converted to {output_format}: {output_path}")
    return output_path


def upload_result(mesh_path: Path, job_input: dict) -> str:
    """
    Upload mesh to S3 and return presigned URL.
    Falls back to base64 if no S3 config provided.
    """
    if "output_s3" in job_input:
        import boto3
        from botocore.config import Config
        
        s3_config = job_input["output_s3"]
        bucket = s3_config["bucket"]
        prefix = s3_config.get("prefix", "2dgs-pipeline/")
        region = s3_config.get("region", "us-west-1")  # Default to us-west-1
        
        key = f"{prefix}{mesh_path.name}"
        
        # Create S3 client with correct region for presigned URLs
        s3 = boto3.client(
            "s3", 
            region_name=region,
            config=Config(signature_version="s3v4")
        )
        s3.upload_file(str(mesh_path), bucket, key)
        
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=86400  # 24 hours
        )
        return url
    else:
        # Return base64 encoded mesh
        with open(mesh_path, "rb") as f:
            return f"data:application/octet-stream;base64,{base64.b64encode(f.read()).decode()}"


def handler(job):
    """
    Main RunPod handler - runs the full 2DGS pipeline.
    Supports both single-video (legacy) and multi-video modes.
    
    Video(s) → ViPE → (Optional: Merge + Align) → 2DGS → Mesh
    """
    job_input = job["input"]
    start_time = time.time()
    
    # Detect multi-video mode
    video_urls = job_input.get("video_urls", [])
    is_multi_video = len(video_urls) > 1
    
    print("\n" + "="*60)
    if is_multi_video:
        print(f"2DGS PIPELINE: Multi-Video ({len(video_urls)} videos) → 3D Mesh")
    else:
        print("2DGS PIPELINE: Video → 3D Mesh")
    print("="*60)
    
    # Parameters
    iterations = job_input.get("iterations", 5000)
    mesh_resolution = job_input.get("mesh_resolution", 512)
    mesh_quality = job_input.get("mesh_quality", "high")  # Default to high quality
    output_format = job_input.get("output_format", "glb").lower()
    depth_threshold = job_input.get("depth_threshold", 0.5)  # Min depth coverage
    
    # Advanced mesh extraction parameters (optional overrides)
    depth_trunc = job_input.get("depth_trunc", None)
    voxel_size = job_input.get("voxel_size", None)
    num_cluster = job_input.get("num_cluster", None)

    print(f"Parameters:")
    print(f"  - Training iterations: {iterations}")
    print(f"  - Mesh quality: {mesh_quality}")
    print(f"  - Mesh resolution: {mesh_resolution}")
    print(f"  - Output format: {output_format}")
    print(f"  - Depth coverage threshold: {depth_threshold}")
    if is_multi_video:
        print(f"  - Videos: {len(video_urls)}")
    
    # Create working directory
    work_dir = Path(tempfile.mkdtemp(prefix="2dgs_pipeline_"))
    print(f"Working directory: {work_dir}")
    
    quality_stats = None
    
    try:
        if is_multi_video:
            # ============================================
            # MULTI-VIDEO MODE
            # ============================================
            
            # Step 1: Download all videos
            video_paths = download_videos(video_urls, work_dir)
            
            # Step 2: Run ViPE on all videos with depth filtering
            vipe_outputs_list, quality_stats = run_vipe_multi(
                video_paths, work_dir, depth_threshold=depth_threshold
            )
            
            # Step 3: Merge ViPE outputs with frame 0 alignment
            vipe_dir, num_frames = merge_vipe_outputs(
                vipe_outputs_list, work_dir, depth_threshold=depth_threshold
            )
            
            print(f"\n[Multi-Video] Merged {num_frames} frames from {len(video_urls)} videos")
            
        else:
            # ============================================
            # SINGLE-VIDEO MODE (legacy compatibility)
            # ============================================
            
            # Handle single video_url in array OR legacy single video input
            if len(video_urls) == 1:
                job_input["video_url"] = video_urls[0]
            
            # Step 1: Download video
            video_path = download_video(job_input, work_dir)
            
            # Step 2: Run ViPE
            vipe_outputs = run_vipe(video_path, work_dir)
            
            # Count frames from poses
            poses_data = np.load(vipe_outputs["poses"])
            num_frames = len(poses_data["data"])
            
            vipe_dir = work_dir / "vipe"
        
        print(f"Processed {num_frames} frames")
        
        # Step 3: Convert to 2DGS format
        data_dir = convert_to_2dgs_format(vipe_dir, work_dir)
        
        # Step 4: Initialize point cloud
        init_points_from_depth(data_dir)
        
        # Step 5: Train 2DGS
        model_dir = work_dir / "model"
        train_2dgs(data_dir, model_dir, iterations)
        
        # Step 6: Extract mesh with high-quality 2DGS native extraction
        mesh_path = extract_mesh(
            data_dir, 
            model_dir, 
            mesh_resolution=mesh_resolution,
            mesh_quality=mesh_quality,
            depth_trunc=depth_trunc,
            voxel_size=voxel_size,
            num_cluster=num_cluster,
        )
        
        # Step 7: Convert format if needed
        if output_format != "ply":
            mesh_path = convert_mesh_format(mesh_path, output_format)
        
        # Step 8: Upload result
        mesh_url = upload_result(mesh_path, job_input)
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print(f"PIPELINE COMPLETE in {elapsed:.1f}s")
        print("="*60)
        
        result = {
            "status": "success",
            "mesh_url": mesh_url,
            "num_frames": num_frames,
            "iterations": iterations,
            "elapsed_seconds": round(elapsed, 1)
        }
        
        # Include quality stats for multi-video mode
        if quality_stats:
            result["quality_stats"] = quality_stats
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\nPIPELINE ERROR: {error_msg}")
        traceback.print_exc()
        
        result = {
            "status": "error",
            "error": error_msg,
            "elapsed_seconds": round(time.time() - start_time, 1)
        }
        
        if quality_stats:
            result["quality_stats"] = quality_stats
        
        return result
        
    finally:
        # Cleanup
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
            print(f"Cleaned up: {work_dir}")


if __name__ == "__main__":
    print("Starting 2DGS Pipeline serverless worker...")
    runpod.serverless.start({"handler": handler})
