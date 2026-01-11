#!/usr/bin/env python3
"""
ViPE RunPod Serverless Handler

Extracts camera poses and depth maps from video using NVIDIA's ViPE.

Input (JSON):
{
    "input": {
        "video_url": "https://...",           # URL to download video OR
        "video_base64": "...",                # Base64 encoded video
        "s3_input": {                         # S3 input (optional)
            "bucket": "bucket-name",
            "key": "path/to/video.mp4"
        },
        "output_s3": {                        # S3 output (optional)
            "bucket": "bucket-name",
            "prefix": "vipe_results/"
        }
    }
}

Output (JSON):
{
    "poses_url": "...",           # URL to download poses.npz
    "intrinsics_url": "...",      # URL to download intrinsics.npz
    "depth_url": "...",           # URL to download depth.zip
    "num_frames": 241,
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


def download_video(video_url: str, output_path: str) -> str:
    """Download video from URL."""
    print(f"Downloading video from {video_url}")
    response = requests.get(video_url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded to {output_path}")
    return output_path


def download_from_s3(bucket: str, key: str, output_path: str) -> str:
    """Download file from S3."""
    if not HAS_BOTO3:
        raise RuntimeError("boto3 not installed for S3 support")
    
    print(f"Downloading from s3://{bucket}/{key}")
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, output_path)
    print(f"Downloaded to {output_path}")
    return output_path


def upload_to_s3(local_path: str, bucket: str, key: str) -> str:
    """Upload file to S3 and return URL."""
    if not HAS_BOTO3:
        raise RuntimeError("boto3 not installed for S3 support")
    
    print(f"Uploading {local_path} to s3://{bucket}/{key}")
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket, key)
    
    # Generate presigned URL (valid for 7 days)
    url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': key},
        ExpiresIn=604800  # 7 days
    )
    return url


def run_vipe(video_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Run ViPE inference on video.
    
    Returns dict with paths to output files.
    """
    import subprocess
    
    print(f"Running ViPE on {video_path}")
    print(f"Output directory: {output_dir}")
    
    # Run ViPE CLI
    cmd = [
        "vipe", "infer",
        video_path,
        "--output", output_dir
    ]
    
    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="/workspace/vipe"
    )
    
    elapsed = time.time() - start_time
    print(f"ViPE completed in {elapsed:.1f}s")
    
    if result.returncode != 0:
        print(f"ViPE stderr: {result.stderr}")
        raise RuntimeError(f"ViPE failed: {result.stderr}")
    
    # Find output files
    video_name = Path(video_path).stem
    
    outputs = {
        "poses_path": os.path.join(output_dir, "pose", f"{video_name}.npz"),
        "intrinsics_path": os.path.join(output_dir, "intrinsics", f"{video_name}.npz"),
        "depth_path": os.path.join(output_dir, "depth", f"{video_name}.zip"),
        "rgb_path": os.path.join(output_dir, "rgb", f"{video_name}.mp4"),
        "elapsed_seconds": elapsed,
    }
    
    # Verify outputs exist
    for key, path in outputs.items():
        if key.endswith("_path") and not os.path.exists(path):
            print(f"Warning: Expected output not found: {path}")
    
    # Get frame count from poses
    if os.path.exists(outputs["poses_path"]):
        poses = np.load(outputs["poses_path"])
        outputs["num_frames"] = len(poses["data"])
    
    return outputs


def package_results(vipe_outputs: Dict, output_dir: str) -> str:
    """Package all ViPE results into a single zip file."""
    zip_path = os.path.join(output_dir, "vipe_results.zip")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for key, path in vipe_outputs.items():
            if key.endswith("_path") and os.path.exists(path):
                arcname = os.path.basename(path)
                zf.write(path, arcname)
    
    return zip_path


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for ViPE.
    """
    job_input = job.get("input", {})
    
    # Create temp directories
    work_dir = tempfile.mkdtemp(prefix="vipe_")
    video_dir = os.path.join(work_dir, "input")
    output_dir = os.path.join(work_dir, "output")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Get video input
        video_path = None
        
        if "video_url" in job_input:
            video_path = os.path.join(video_dir, "input.mp4")
            download_video(job_input["video_url"], video_path)
        
        elif "video_base64" in job_input:
            video_path = os.path.join(video_dir, "input.mp4")
            video_data = base64.b64decode(job_input["video_base64"])
            with open(video_path, 'wb') as f:
                f.write(video_data)
        
        elif "s3_input" in job_input:
            s3_config = job_input["s3_input"]
            video_path = os.path.join(video_dir, "input.mp4")
            download_from_s3(s3_config["bucket"], s3_config["key"], video_path)
        
        else:
            return {"error": "No video input provided. Use video_url, video_base64, or s3_input"}
        
        # Run ViPE
        vipe_outputs = run_vipe(video_path, output_dir)
        
        # Handle output
        result = {
            "status": "success",
            "num_frames": vipe_outputs.get("num_frames", 0),
            "elapsed_seconds": vipe_outputs.get("elapsed_seconds", 0),
        }
        
        # Upload to S3 if configured
        if "output_s3" in job_input:
            s3_config = job_input["output_s3"]
            bucket = s3_config["bucket"]
            prefix = s3_config.get("prefix", "vipe/")
            
            # Upload individual files
            if os.path.exists(vipe_outputs["poses_path"]):
                result["poses_url"] = upload_to_s3(
                    vipe_outputs["poses_path"],
                    bucket,
                    f"{prefix}poses.npz"
                )
            
            if os.path.exists(vipe_outputs["intrinsics_path"]):
                result["intrinsics_url"] = upload_to_s3(
                    vipe_outputs["intrinsics_path"],
                    bucket,
                    f"{prefix}intrinsics.npz"
                )
            
            if os.path.exists(vipe_outputs["depth_path"]):
                result["depth_url"] = upload_to_s3(
                    vipe_outputs["depth_path"],
                    bucket,
                    f"{prefix}depth.zip"
                )
            
            if os.path.exists(vipe_outputs["rgb_path"]):
                result["rgb_url"] = upload_to_s3(
                    vipe_outputs["rgb_path"],
                    bucket,
                    f"{prefix}rgb.mp4"
                )
        else:
            # Package results and return base64
            zip_path = package_results(vipe_outputs, output_dir)
            
            with open(zip_path, 'rb') as f:
                result["results_base64"] = base64.b64encode(f.read()).decode('utf-8')
        
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
