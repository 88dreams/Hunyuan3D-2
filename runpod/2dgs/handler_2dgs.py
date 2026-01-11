#!/usr/bin/env python3
"""
2DGS RunPod Serverless Handler

Trains 2D Gaussian Splatting model and extracts mesh from ViPE outputs.

Input (JSON):
{
    "input": {
        "vipe_results_url": "https://...",    # URL to ViPE results zip OR
        "vipe_s3": {                          # S3 location of ViPE results
            "bucket": "bucket-name",
            "poses_key": "path/to/poses.npz",
            "intrinsics_key": "path/to/intrinsics.npz",
            "depth_key": "path/to/depth.zip",
            "rgb_key": "path/to/rgb.mp4"
        },
        "iterations": 5000,                   # Training iterations (default: 5000)
        "mesh_resolution": 512,               # Mesh extraction resolution (default: 512)
        "output_format": "glb",               # Output format: glb, obj, ply (default: glb)
        "output_s3": {                        # S3 output (optional)
            "bucket": "bucket-name",
            "prefix": "2dgs_results/"
        }
    }
}

Output (JSON):
{
    "mesh_url": "...",            # URL to download mesh
    "stats": {
        "num_frames": 241,
        "final_loss": 0.029,
        "num_points": 156032,
        "training_seconds": 150,
        "mesh_vertices": 830682,
        "mesh_faces": 1661624
    },
    "status": "success"
}
"""

import os
import sys
import json
import time
import base64
import tempfile
import zipfile
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import runpod
import requests
import numpy as np

# Optional S3 support
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


def download_file(url: str, output_path: str) -> str:
    """Download file from URL."""
    print(f"Downloading from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return output_path


def download_from_s3(bucket: str, key: str, output_path: str) -> str:
    """Download file from S3."""
    if not HAS_BOTO3:
        raise RuntimeError("boto3 not installed for S3 support")
    
    print(f"Downloading from s3://{bucket}/{key}")
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, output_path)
    return output_path


def upload_to_s3(local_path: str, bucket: str, key: str) -> str:
    """Upload file to S3 and return presigned URL."""
    if not HAS_BOTO3:
        raise RuntimeError("boto3 not installed for S3 support")
    
    print(f"Uploading {local_path} to s3://{bucket}/{key}")
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket, key)
    
    url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': key},
        ExpiresIn=604800
    )
    return url


def convert_vipe_to_2dgs(vipe_dir: str, output_dir: str) -> int:
    """Convert ViPE output to 2DGS format using our conversion script."""
    
    cmd = [
        "python", "/workspace/vipe_to_2dgs.py",
        "--vipe_dir", vipe_dir,
        "--output_dir", output_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Conversion stderr: {result.stderr}")
        raise RuntimeError(f"ViPE to 2DGS conversion failed: {result.stderr}")
    
    print(result.stdout)
    
    # Count images
    images_dir = os.path.join(output_dir, "images")
    if os.path.exists(images_dir):
        return len(list(Path(images_dir).glob("*")))
    return 0


def init_points_from_depth(data_dir: str) -> int:
    """Initialize 3D points from depth maps."""
    
    cmd = [
        "python", "/workspace/init_points_from_depth.py",
        "--data_dir", data_dir,
        "--sample_rate", "8",
        "--num_frames", "20"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Init points stderr: {result.stderr}")
        raise RuntimeError(f"Point initialization failed: {result.stderr}")
    
    print(result.stdout)
    
    # Return point count from output
    for line in result.stdout.split('\n'):
        if 'Total points:' in line:
            return int(line.split(':')[1].strip())
    return 0


def train_2dgs(data_dir: str, model_dir: str, iterations: int = 5000) -> Dict[str, Any]:
    """Train 2DGS model."""
    
    cmd = [
        "python", "/workspace/2d-gaussian-splatting/train.py",
        "-s", data_dir,
        "-m", model_dir,
        "--iterations", str(iterations),
        "--save_iterations", str(iterations)
    ]
    
    print(f"Training 2DGS for {iterations} iterations...")
    start_time = time.time()
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="/workspace/2d-gaussian-splatting"
    )
    
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"Training stderr: {result.stderr}")
        raise RuntimeError(f"2DGS training failed: {result.stderr}")
    
    # Parse final stats from output
    stats = {
        "training_seconds": elapsed,
        "iterations": iterations,
    }
    
    # Try to extract loss and point count from output
    for line in result.stdout.split('\n'):
        if 'Loss=' in line:
            try:
                loss_str = line.split('Loss=')[1].split(',')[0]
                stats["final_loss"] = float(loss_str)
            except:
                pass
        if 'Points=' in line:
            try:
                points_str = line.split('Points=')[1].split(']')[0]
                stats["num_points"] = int(points_str)
            except:
                pass
    
    print(f"Training completed in {elapsed:.1f}s")
    return stats


def extract_mesh(model_dir: str, data_dir: str, iteration: int, mesh_resolution: int = 512) -> str:
    """Extract mesh from trained 2DGS model."""
    
    cmd = [
        "python", "/workspace/2d-gaussian-splatting/render.py",
        "-s", data_dir,
        "-m", model_dir,
        "--iteration", str(iteration),
        "--skip_train",
        "--skip_test",
        "--unbounded",
        "--mesh_res", str(mesh_resolution)
    ]
    
    print(f"Extracting mesh at resolution {mesh_resolution}...")
    start_time = time.time()
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="/workspace/2d-gaussian-splatting"
    )
    
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"Mesh extraction stderr: {result.stderr}")
        raise RuntimeError(f"Mesh extraction failed: {result.stderr}")
    
    print(f"Mesh extraction completed in {elapsed:.1f}s")
    
    # Find the mesh file
    mesh_path = os.path.join(model_dir, "train", f"ours_{iteration}", "fuse_unbounded_post.ply")
    if not os.path.exists(mesh_path):
        mesh_path = os.path.join(model_dir, "train", f"ours_{iteration}", "fuse_unbounded.ply")
    
    if not os.path.exists(mesh_path):
        raise RuntimeError(f"Mesh file not found at expected location")
    
    return mesh_path


def convert_mesh(input_path: str, output_format: str, output_dir: str) -> str:
    """Convert mesh to desired format."""
    import trimesh
    
    mesh = trimesh.load(input_path)
    
    output_name = f"stage_mesh.{output_format.lower()}"
    output_path = os.path.join(output_dir, output_name)
    
    mesh.export(output_path)
    
    return output_path, {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces)
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for 2DGS training and mesh extraction.
    """
    job_input = job.get("input", {})
    
    # Parameters
    iterations = job_input.get("iterations", 5000)
    mesh_resolution = job_input.get("mesh_resolution", 512)
    output_format = job_input.get("output_format", "glb")
    
    # Create temp directories
    work_dir = tempfile.mkdtemp(prefix="2dgs_")
    vipe_dir = os.path.join(work_dir, "vipe_results")
    data_dir = os.path.join(work_dir, "2dgs_data")
    model_dir = os.path.join(work_dir, "model")
    output_dir = os.path.join(work_dir, "output")
    
    os.makedirs(vipe_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {}
    
    try:
        # Download ViPE results
        if "vipe_results_url" in job_input:
            zip_path = os.path.join(work_dir, "vipe_results.zip")
            download_file(job_input["vipe_results_url"], zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(vipe_dir)
        
        elif "vipe_s3" in job_input:
            s3_config = job_input["vipe_s3"]
            bucket = s3_config["bucket"]
            
            # Download each ViPE output
            os.makedirs(os.path.join(vipe_dir, "pose"), exist_ok=True)
            os.makedirs(os.path.join(vipe_dir, "intrinsics"), exist_ok=True)
            os.makedirs(os.path.join(vipe_dir, "depth"), exist_ok=True)
            os.makedirs(os.path.join(vipe_dir, "rgb"), exist_ok=True)
            
            download_from_s3(bucket, s3_config["poses_key"], 
                           os.path.join(vipe_dir, "pose", "video.npz"))
            download_from_s3(bucket, s3_config["intrinsics_key"],
                           os.path.join(vipe_dir, "intrinsics", "video.npz"))
            download_from_s3(bucket, s3_config["depth_key"],
                           os.path.join(vipe_dir, "depth", "video.zip"))
            download_from_s3(bucket, s3_config["rgb_key"],
                           os.path.join(vipe_dir, "rgb", "video.mp4"))
        
        else:
            return {"error": "No ViPE results provided. Use vipe_results_url or vipe_s3"}
        
        # Step 1: Convert ViPE to 2DGS format
        print("Step 1: Converting ViPE output to 2DGS format...")
        num_frames = convert_vipe_to_2dgs(vipe_dir, data_dir)
        stats["num_frames"] = num_frames
        
        # Step 2: Initialize points from depth
        print("Step 2: Initializing points from depth maps...")
        num_points = init_points_from_depth(data_dir)
        stats["initial_points"] = num_points
        
        # Step 3: Train 2DGS
        print("Step 3: Training 2DGS model...")
        train_stats = train_2dgs(data_dir, model_dir, iterations)
        stats.update(train_stats)
        
        # Step 4: Extract mesh
        print("Step 4: Extracting mesh...")
        mesh_path = extract_mesh(model_dir, data_dir, iterations, mesh_resolution)
        
        # Step 5: Convert mesh format
        print(f"Step 5: Converting to {output_format}...")
        final_mesh_path, mesh_stats = convert_mesh(mesh_path, output_format, output_dir)
        stats.update(mesh_stats)
        
        # Prepare result
        result = {
            "status": "success",
            "stats": stats
        }
        
        # Upload to S3 if configured
        if "output_s3" in job_input:
            s3_config = job_input["output_s3"]
            bucket = s3_config["bucket"]
            prefix = s3_config.get("prefix", "2dgs/")
            
            mesh_key = f"{prefix}stage_mesh.{output_format.lower()}"
            result["mesh_url"] = upload_to_s3(final_mesh_path, bucket, mesh_key)
        else:
            # Return base64 encoded mesh
            with open(final_mesh_path, 'rb') as f:
                result["mesh_base64"] = base64.b64encode(f.read()).decode('utf-8')
        
        return result
    
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
    
    finally:
        # Cleanup
        shutil.rmtree(work_dir, ignore_errors=True)


# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
