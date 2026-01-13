#!/usr/bin/env python3
"""
SEVA (Stable Virtual Camera) RunPod Serverless Handler

Generates novel view videos from single images with precise camera control.

Model: Stability AI Stable Virtual Camera (1.3B parameters)
Input: Single image + camera trajectory
Output: Video (MP4) with camera movement

Supported trajectories:
    - orbit: 360° rotation around subject
    - pan: Horizontal camera movement  
    - tilt: Vertical camera angle change
    - spiral: Spiral path around subject
    - zoom-out: Camera moves backward
    - dolly-zoom-out: Vertigo/Hitchcock effect
    - arc: Curved path
    - crane: Vertical + horizontal movement
    - custom: User-defined camera poses (C2W matrices)
"""

import os
import sys
import base64
import tempfile
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import time

import runpod

# =============================================================================
# CONFIGURATION
# =============================================================================

SEVA_DIR = os.environ.get("SEVA_DIR", "/workspace/seva")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/workspace/checkpoints")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/runpod-volume/outputs/seva")

# S3 Configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_REGION = os.environ.get("S3_REGION", "us-west-1")
S3_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")
S3_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
S3_PREFIX = os.environ.get("S3_PREFIX", "MediaContent/outputs/seva")
S3_ENABLED = bool(S3_BUCKET and S3_ACCESS_KEY and S3_SECRET_KEY)

# Maximum file size for base64 encoding (8MB)
MAX_BASE64_SIZE = int(os.environ.get("MAX_BASE64_SIZE", 8 * 1024 * 1024))

# Ensure directories exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seva-handler")

# =============================================================================
# VALID OPTIONS
# =============================================================================

VALID_TRAJECTORIES = [
    "orbit",
    "pan", 
    "tilt",
    "spiral",
    "zoom-out",
    "dolly-zoom-out",
    "arc",
    "crane",
    "left",
    "right",
    "up",
    "down",
    "custom"
]

# =============================================================================
# ENVIRONMENT CHECK
# =============================================================================

def check_environment():
    """Log environment info at startup."""
    logger.info("=" * 60)
    logger.info("SEVA ENVIRONMENT CHECK")
    logger.info("=" * 60)
    
    logger.info(f"Python: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    
    # Check PyTorch
    try:
        import torch
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError as e:
        logger.error(f"PyTorch import failed: {e}")
    
    # Check SEVA installation
    seva_path = Path(SEVA_DIR)
    if seva_path.exists():
        logger.info(f"SEVA directory: {SEVA_DIR} (exists)")
    else:
        logger.warning(f"SEVA directory: {SEVA_DIR} (NOT FOUND)")
    
    # Check S3 configuration
    logger.info(f"S3 enabled: {S3_ENABLED}")
    if S3_ENABLED:
        logger.info(f"S3 bucket: {S3_BUCKET}")
    
    logger.info("=" * 60)

check_environment()

# =============================================================================
# S3 UTILITIES
# =============================================================================

_s3_client = None

def get_s3_client():
    """Get or create S3 client."""
    global _s3_client
    if _s3_client is None and S3_ENABLED:
        try:
            import boto3
            _s3_client = boto3.client(
                's3',
                region_name=S3_REGION,
                aws_access_key_id=S3_ACCESS_KEY,
                aws_secret_access_key=S3_SECRET_KEY,
            )
            logger.info(f"S3 client initialized for bucket: {S3_BUCKET}")
        except Exception as e:
            logger.warning(f"Failed to initialize S3 client: {e}")
    return _s3_client


def upload_to_s3(local_path: str, s3_key: str) -> Optional[str]:
    """Upload file to S3 and return URL."""
    client = get_s3_client()
    if client is None:
        return None
    
    try:
        client.upload_file(local_path, S3_BUCKET, s3_key)
        url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
        logger.info(f"Uploaded to S3: {url}")
        return url
    except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        return None


def download_from_s3(s3_url: str, local_path: str) -> bool:
    """Download file from S3 URL."""
    client = get_s3_client()
    if client is None:
        return False
    
    try:
        # Parse S3 URL
        if s3_url.startswith("s3://"):
            parts = s3_url[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        elif "s3.amazonaws.com" in s3_url or "s3." in s3_url:
            # HTTPS URL format
            from urllib.parse import urlparse
            parsed = urlparse(s3_url)
            if ".s3." in parsed.netloc:
                bucket = parsed.netloc.split(".s3.")[0]
            else:
                bucket = S3_BUCKET
            key = parsed.path.lstrip("/")
        else:
            return False
        
        client.download_file(bucket, key, local_path)
        logger.info(f"Downloaded from S3: {s3_url}")
        return True
    except Exception as e:
        logger.error(f"S3 download failed: {e}")
        return False


def download_from_url(url: str, local_path: str) -> bool:
    """Download file from HTTP(S) URL."""
    try:
        import requests
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded from URL: {url}")
        return True
    except Exception as e:
        logger.error(f"URL download failed: {e}")
        return False


# =============================================================================
# SEVA MODEL
# =============================================================================

_seva_model = None

def load_seva_model():
    """Load SEVA model (lazy initialization)."""
    global _seva_model
    
    if _seva_model is not None:
        return _seva_model
    
    logger.info("Loading SEVA model...")
    start_time = time.time()
    
    try:
        # Add SEVA to path
        sys.path.insert(0, SEVA_DIR)
        
        # Import SEVA - actual import path may vary based on repo structure
        # This is a placeholder - will need adjustment based on actual API
        from seva.pipeline import SEVAPipeline
        
        _seva_model = SEVAPipeline.from_pretrained(
            "stabilityai/stable-virtual-camera",
            torch_dtype="auto",
            device_map="auto"
        )
        
        load_time = time.time() - start_time
        logger.info(f"SEVA model loaded in {load_time:.1f}s")
        
        return _seva_model
        
    except ImportError as e:
        logger.error(f"Failed to import SEVA: {e}")
        logger.info("Attempting alternative import...")
        
        # Alternative: use CLI-based approach
        return "cli_mode"
        
    except Exception as e:
        logger.error(f"Failed to load SEVA model: {e}")
        raise


def run_seva_cli(
    input_image: str,
    output_path: str,
    trajectory: str = "orbit",
    duration: float = 5.0,
    fps: int = 24,
    custom_poses: Optional[List] = None
) -> str:
    """Run SEVA using CLI interface."""
    import subprocess
    
    cmd = [
        "python", f"{SEVA_DIR}/demo.py",
        "--input_image", input_image,
        "--output", output_path,
        "--trajectory", trajectory,
        "--duration", str(duration),
        "--fps", str(fps)
    ]
    
    logger.info(f"Running SEVA CLI: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        cwd=SEVA_DIR,
        capture_output=True,
        text=True,
        timeout=600  # 10 minute timeout
    )
    
    if result.returncode != 0:
        logger.error(f"SEVA CLI failed: {result.stderr}")
        raise RuntimeError(f"SEVA CLI error: {result.stderr}")
    
    logger.info(f"SEVA CLI output: {result.stdout}")
    return output_path


def run_seva_pipeline(
    input_image: str,
    output_path: str,
    trajectory: str = "orbit",
    duration: float = 5.0,
    fps: int = 24,
    num_frames: Optional[int] = None,
    custom_poses: Optional[List] = None,
    seed: Optional[int] = None
) -> str:
    """
    Run SEVA to generate novel view video.
    
    Args:
        input_image: Path to input image
        output_path: Path for output video
        trajectory: Camera trajectory type
        duration: Video duration in seconds
        fps: Frames per second
        num_frames: Override frame count (if set, ignores duration)
        custom_poses: List of 4x4 C2W matrices for custom trajectory
        seed: Random seed for reproducibility
        
    Returns:
        Path to generated video
    """
    model = load_seva_model()
    
    if model == "cli_mode":
        return run_seva_cli(input_image, output_path, trajectory, duration, fps, custom_poses)
    
    # Calculate frame count
    if num_frames is None:
        num_frames = int(duration * fps)
    
    logger.info(f"Generating {num_frames} frames at {fps} fps ({duration}s)")
    
    # Load input image
    from PIL import Image
    image = Image.open(input_image).convert("RGB")
    
    # Set seed if provided
    if seed is not None:
        import torch
        torch.manual_seed(seed)
    
    # Generate video
    # Note: Actual API may differ - this is based on expected interface
    if trajectory == "custom" and custom_poses:
        import numpy as np
        poses = [np.array(p) for p in custom_poses]
        video_frames = model.generate(
            image=image,
            camera_poses=poses,
            num_frames=num_frames
        )
    else:
        video_frames = model.generate(
            image=image,
            trajectory=trajectory,
            num_frames=num_frames
        )
    
    # Save video
    import imageio
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in video_frames:
            writer.append_data(frame)
    
    logger.info(f"Video saved to: {output_path}")
    return output_path


# =============================================================================
# INPUT HANDLING
# =============================================================================

def save_input_image(job_input: Dict) -> str:
    """Save input image from base64 or URL to temp file."""
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, "input.png")
    
    # Check for image data
    image_data = job_input.get("image") or job_input.get("image_data") or job_input.get("image_base64")
    image_url = job_input.get("image_url") or job_input.get("input_url")
    
    if image_data:
        # Base64 encoded image
        if isinstance(image_data, str):
            if image_data.startswith("data:"):
                # Data URL format
                image_data = image_data.split(",", 1)[1]
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
            
        with open(input_path, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Saved base64 image to: {input_path}")
        
    elif image_url:
        # Download from URL
        if "s3.amazonaws.com" in image_url or image_url.startswith("s3://"):
            success = download_from_s3(image_url, input_path)
        else:
            success = download_from_url(image_url, input_path)
            
        if not success:
            raise ValueError(f"Failed to download image from: {image_url}")
        logger.info(f"Downloaded image to: {input_path}")
        
    else:
        raise ValueError("No image provided. Use 'image' (base64) or 'image_url'")
    
    return input_path


# =============================================================================
# MAIN HANDLER
# =============================================================================

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main handler for SEVA serverless jobs.
    
    Input parameters:
        image: Base64 encoded image (or image_data, image_base64)
        image_url: URL to download image from
        trajectory: Camera trajectory (orbit, pan, tilt, etc.)
        duration: Video duration in seconds (default: 5.0)
        fps: Frames per second (default: 24)
        num_frames: Override frame count
        custom_poses: List of 4x4 C2W matrices for custom trajectory
        seed: Random seed
        output_name: Name for output file (without extension)
        return_base64: Return video as base64 (default: False for large files)
        
    Output:
        status: "success" or "error"
        video_url: S3 URL to video (if S3 enabled)
        video_path: Local path on network volume
        video_base64: Base64 encoded video (if small and return_base64=True)
        duration: Actual video duration
        frame_count: Number of frames generated
    """
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})
    
    logger.info(f"Processing SEVA job: {job_id}")
    logger.info(f"Input parameters: {list(job_input.keys())}")
    
    try:
        # Parse parameters
        trajectory = job_input.get("trajectory", "orbit")
        duration = float(job_input.get("duration", 5.0))
        fps = int(job_input.get("fps", 24))
        num_frames = job_input.get("num_frames")
        custom_poses = job_input.get("custom_poses")
        seed = job_input.get("seed")
        output_name = job_input.get("output_name", f"seva_{job_id}")
        return_base64 = job_input.get("return_base64", False)
        
        # Validate trajectory
        if trajectory not in VALID_TRAJECTORIES:
            return {
                "status": "error",
                "message": f"Invalid trajectory '{trajectory}'. Valid options: {VALID_TRAJECTORIES}"
            }
        
        # Validate duration
        duration = max(1.0, min(30.0, duration))
        
        # Validate fps
        fps = max(12, min(60, fps))
        
        # Save input image
        input_path = save_input_image(job_input)
        
        # Generate output path
        output_filename = f"{output_name}.mp4"
        temp_output = os.path.join(tempfile.gettempdir(), output_filename)
        
        # Run SEVA
        logger.info(f"Running SEVA: trajectory={trajectory}, duration={duration}s, fps={fps}")
        
        run_seva_pipeline(
            input_image=input_path,
            output_path=temp_output,
            trajectory=trajectory,
            duration=duration,
            fps=fps,
            num_frames=num_frames,
            custom_poses=custom_poses,
            seed=seed
        )
        
        # Copy to network volume
        final_output = os.path.join(OUTPUT_DIR, output_filename)
        shutil.copy2(temp_output, final_output)
        logger.info(f"Video saved to network volume: {final_output}")
        
        # Get file info
        file_size = os.path.getsize(final_output)
        actual_frames = num_frames if num_frames else int(duration * fps)
        
        # Build response
        response = {
            "status": "success",
            "message": "Video generated successfully",
            "model": "seva",
            "video_path": final_output,
            "video_name": output_filename,
            "file_size": file_size,
            "duration": duration,
            "fps": fps,
            "frame_count": actual_frames,
            "trajectory": trajectory
        }
        
        # Upload to S3
        if S3_ENABLED:
            s3_key = f"{S3_PREFIX}/{output_filename}"
            s3_url = upload_to_s3(final_output, s3_key)
            if s3_url:
                response["video_url"] = s3_url
        
        # Return base64 if requested and file is small enough
        if return_base64 and file_size <= MAX_BASE64_SIZE:
            with open(final_output, "rb") as f:
                response["video_base64"] = base64.b64encode(f.read()).decode("utf-8")
        
        # Cleanup temp files
        if os.path.exists(input_path):
            os.remove(input_path)
            parent = os.path.dirname(input_path)
            if os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
        if os.path.exists(temp_output) and temp_output != final_output:
            os.remove(temp_output)
        
        logger.info(f"SEVA job {job_id} completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"SEVA job {job_id} failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }


# =============================================================================
# HEALTH CHECK
# =============================================================================

def health_check(_job: Dict) -> Dict:
    """Health check endpoint."""
    import torch
    
    return {
        "status": "healthy",
        "model": "seva",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "s3_enabled": S3_ENABLED,
        "output_dir": OUTPUT_DIR,
        "valid_trajectories": VALID_TRAJECTORIES
    }


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting SEVA serverless handler...")
    
    runpod.serverless.start({
        "handler": handler,
        "health_check": health_check
    })
