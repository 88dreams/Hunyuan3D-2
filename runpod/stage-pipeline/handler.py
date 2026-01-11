"""
Stage Pipeline Handler - Combined ViPE + 2DGS
Converts Gen3C video directly to 3D mesh in one endpoint.

Input:
    {
        "video_url": "https://...",  # OR
        "video_base64": "...",       # OR  
        "s3_input": {"bucket": "...", "key": "..."},
        
        # Optional parameters
        "iterations": 5000,          # 2DGS training iterations
        "mesh_resolution": 512,      # Mesh extraction resolution
        "output_format": "glb"       # "glb", "obj", or "ply"
    }

Output:
    {
        "status": "success",
        "mesh_url": "https://presigned-url...",
        "num_frames": 241,
        "elapsed_seconds": 600
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


def extract_mesh(data_dir: Path, model_dir: Path, mesh_resolution: int = 512) -> Path:
    """
    Extract mesh from trained 2DGS model using TSDF fusion.
    Returns path to extracted mesh.
    """
    print("\n" + "="*50)
    print(f"STEP 5: Extracting mesh (resolution: {mesh_resolution})")
    print("="*50)
    
    # Find the iteration checkpoint
    ckpt_dirs = list((model_dir / "point_cloud").glob("iteration_*"))
    if not ckpt_dirs:
        raise FileNotFoundError("No checkpoint found in model directory")
    
    latest_iter = max(int(d.name.split("_")[1]) for d in ckpt_dirs)
    
    cmd = [
        sys.executable, "render.py",
        "-s", str(data_dir),
        "-m", str(model_dir),
        "--iteration", str(latest_iter),
        "--skip_train",
        "--skip_test",
        "--unbounded",
        "--mesh_res", str(mesh_resolution)
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
    
    # Find the mesh file - check if it exists before failing on return code
    mesh_dir = model_dir / "mesh"
    mesh_files = list(mesh_dir.glob("*.ply")) if mesh_dir.exists() else []
    
    if not mesh_files:
        # Try alternative locations
        mesh_files = list(model_dir.glob("**/fuse*.ply"))
    
    if not mesh_files:
        mesh_files = list(model_dir.glob("**/*.ply"))
    
    # Only fail if no mesh was created AND return code was non-zero
    if not mesh_files:
        if result.returncode != 0:
            raise RuntimeError(f"Mesh extraction failed (exit code {result.returncode}): {result.stderr}")
        else:
            raise FileNotFoundError("No mesh file found after extraction")
    
    mesh_path = mesh_files[0]
    print(f"  ✓ Mesh extracted: {mesh_path}")
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
        prefix = s3_config.get("prefix", "stage-pipeline/")
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
    Main RunPod handler - runs the full stage pipeline.
    Video → ViPE → 2DGS → Mesh
    """
    job_input = job["input"]
    start_time = time.time()
    
    print("\n" + "="*60)
    print("STAGE PIPELINE: Gen3C Video → 3D Mesh")
    print("="*60)
    
    # Parameters
    iterations = job_input.get("iterations", 5000)
    mesh_resolution = job_input.get("mesh_resolution", 512)
    output_format = job_input.get("output_format", "glb").lower()
    
    print(f"Parameters:")
    print(f"  - Training iterations: {iterations}")
    print(f"  - Mesh resolution: {mesh_resolution}")
    print(f"  - Output format: {output_format}")
    
    # Create working directory
    work_dir = Path(tempfile.mkdtemp(prefix="stage_pipeline_"))
    print(f"Working directory: {work_dir}")
    
    try:
        # Step 1: Download video
        video_path = download_video(job_input, work_dir)
        
        # Step 2: Run ViPE
        vipe_outputs = run_vipe(video_path, work_dir)
        
        # Count frames from poses
        poses_data = np.load(vipe_outputs["poses"])
        num_frames = len(poses_data["data"])
        print(f"Processed {num_frames} frames")
        
        # Step 3: Convert to 2DGS format
        vipe_dir = work_dir / "vipe"
        data_dir = convert_to_2dgs_format(vipe_dir, work_dir)
        
        # Step 4: Initialize point cloud
        init_points_from_depth(data_dir)
        
        # Step 5: Train 2DGS
        model_dir = work_dir / "model"
        train_2dgs(data_dir, model_dir, iterations)
        
        # Step 6: Extract mesh
        mesh_path = extract_mesh(data_dir, model_dir, mesh_resolution)
        
        # Step 7: Convert format if needed
        if output_format != "ply":
            mesh_path = convert_mesh_format(mesh_path, output_format)
        
        # Step 8: Upload result
        mesh_url = upload_result(mesh_path, job_input)
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print(f"PIPELINE COMPLETE in {elapsed:.1f}s")
        print("="*60)
        
        return {
            "status": "success",
            "mesh_url": mesh_url,
            "num_frames": num_frames,
            "iterations": iterations,
            "elapsed_seconds": round(elapsed, 1)
        }
        
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\nPIPELINE ERROR: {error_msg}")
        traceback.print_exc()
        
        return {
            "status": "error",
            "error": error_msg,
            "elapsed_seconds": round(time.time() - start_time, 1)
        }
        
    finally:
        # Cleanup
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
            print(f"Cleaned up: {work_dir}")


if __name__ == "__main__":
    print("Starting Stage Pipeline serverless worker...")
    runpod.serverless.start({"handler": handler})
