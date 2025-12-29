#!/usr/bin/env python3
"""
TRELLIS.2 RunPod Serverless Handler

Dedicated handler for TRELLIS.2 3D generation.
Runs on its own endpoint, separate from the unified GEN3C/SHARP endpoint.

Supports:
- Image to 3D GLB/PLY generation
- Multiple resolutions (512, 768, 1024, 1536)
- PBR material output
- S3 upload for large files
"""

import os
import sys
import time
import base64
import tempfile
import subprocess
import logging
from typing import Dict, Any, Optional, Tuple

import runpod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(filename)s :%(lineno)d %(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

TRELLIS_DIR = "/workspace/TRELLIS2"
CHECKPOINT_DIR = os.environ.get("TRELLIS_CHECKPOINT_DIR", "/runpod-volume/checkpoints/trellis")
OUTPUT_DIR = "/runpod-volume/outputs/trellis"
INFERENCE_SCRIPT = "/workspace/trellis_inference.py"
PYTHON_PATH = "/root/miniforge3/envs/cosmos-predict1/bin/python"

# S3 Configuration
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_REGION = os.environ.get("S3_REGION", "us-west-1")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
S3_PREFIX = os.environ.get("S3_PREFIX", "")

# File size limits
MAX_BASE64_SIZE = 8 * 1024 * 1024  # 8MB

# ============================================================================
# S3 UTILITIES
# ============================================================================

_s3_client = None

def get_s3_client():
    """Initialize and return S3 client."""
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    
    if not S3_BUCKET:
        logger.info("S3_BUCKET not configured")
        return None
    
    try:
        import boto3
        _s3_client = boto3.client(
            "s3",
            region_name=S3_REGION,
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
        logger.info(f"S3 client initialized for bucket: {S3_BUCKET}")
        return _s3_client
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {e}")
        return None


def upload_to_s3(local_path: str, s3_key: str) -> Optional[str]:
    """Upload file to S3 and return URL."""
    client = get_s3_client()
    if client is None:
        return None
    
    try:
        file_size = os.path.getsize(local_path) / 1024 / 1024
        logger.info(f"Uploading to S3: {local_path} -> s3://{S3_BUCKET}/{s3_key} ({file_size:.1f}MB)")
        
        client.upload_file(local_path, S3_BUCKET, s3_key)
        
        s3_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
        logger.info(f"S3 upload complete: {s3_url}")
        return s3_url
    except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        return None


def upload_file_to_s3_if_large(file_path: str, filename: str) -> Dict[str, Any]:
    """Upload to S3 if file is too large for base64."""
    file_size = os.path.getsize(file_path)
    if file_size > MAX_BASE64_SIZE:
        s3_key = f"{S3_PREFIX}/trellis/{filename}" if S3_PREFIX else f"trellis/{filename}"
        s3_url = upload_to_s3(file_path, s3_key)
        return {"uploaded": s3_url is not None, "s3_url": s3_url}
    return {"uploaded": False, "s3_url": None}


def encode_file_if_small(file_path: str) -> Optional[str]:
    """Encode file to base64 if small enough."""
    file_size = os.path.getsize(file_path)
    if file_size > MAX_BASE64_SIZE:
        logger.info(f"File too large for base64 ({file_size / 1024 / 1024:.1f}MB > {MAX_BASE64_SIZE / 1024 / 1024:.1f}MB): {file_path}")
        return None
    
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================================================
# TRELLIS.2 INFERENCE
# ============================================================================

def validate_trellis() -> bool:
    """Check if TRELLIS.2 is properly installed."""
    # Check for TRELLIS.2 directory
    if not os.path.isdir(TRELLIS_DIR):
        logger.warning(f"TRELLIS.2 directory not found: {TRELLIS_DIR}")
        return False
    
    # Check for inference script
    if not os.path.exists(INFERENCE_SCRIPT):
        logger.warning(f"Inference script not found: {INFERENCE_SCRIPT}")
        return False
    
    # Check for Python environment
    if not os.path.exists(PYTHON_PATH):
        logger.warning(f"Python not found: {PYTHON_PATH}")
        return False
    
    return True


def run_trellis(
    input_image_path: str,
    output_dir: str,
    output_name: str = "trellis_output",
    resolution: int = 1024,
    guidance_scale: float = 7.5,
    output_glb: bool = True,
    output_ply: bool = False,
    seed: Optional[int] = None,
    timeout: int = 1800,  # 30 minutes for high resolution
) -> Dict[str, str]:
    """
    Run TRELLIS.2 inference.
    
    Returns dict with paths to generated files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        PYTHON_PATH,
        INFERENCE_SCRIPT,
        "--input_image", input_image_path,
        "--output_dir", output_dir,
        "--output_name", output_name,
        "--resolution", str(resolution),
        "--guidance_scale", str(guidance_scale),
        "--checkpoint_dir", CHECKPOINT_DIR,
    ]
    
    if output_glb:
        cmd.append("--output_glb")
    if output_ply:
        cmd.append("--output_ply")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    
    logger.info(f"Running TRELLIS.2: {' '.join(cmd)}")
    
    # Set environment
    env = os.environ.copy()
    env["PYTHONPATH"] = TRELLIS_DIR
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Run inference
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    
    if result.returncode != 0:
        logger.error(f"TRELLIS.2 failed. stderr: {result.stderr}")
        raise RuntimeError(f"TRELLIS.2 failed: {result.stderr[-2000:]}")
    
    # Parse output paths from stdout
    outputs = {}
    
    # Check for GLB
    glb_path = os.path.join(output_dir, f"{output_name}.glb")
    if os.path.exists(glb_path):
        outputs["glb_path"] = glb_path
        logger.info(f"GLB generated: {glb_path}")
    
    # Check for PLY
    ply_path = os.path.join(output_dir, f"{output_name}.ply")
    if os.path.exists(ply_path):
        outputs["ply_path"] = ply_path
        logger.info(f"PLY generated: {ply_path}")
    
    if not outputs:
        raise RuntimeError("TRELLIS.2 completed but no output files found")
    
    return outputs


# ============================================================================
# HANDLER
# ============================================================================

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless Handler for TRELLIS.2.
    
    Expected input:
    {
        "input": {
            "image_base64": "...",  # Base64 encoded input image
            "output_name": "my_model",  # Optional output name
            "resolution": 1024,  # 512, 768, 1024, or 1536
            "guidance_scale": 7.5,  # Guidance scale
            "output_glb": true,  # Export GLB
            "output_ply": false,  # Export PLY
            "seed": 42,  # Optional seed
            "return_base64": false,  # Return base64 or save to S3
        }
    }
    """
    job_input = job.get("input", {})
    
    # Validate TRELLIS.2 installation
    if not validate_trellis():
        return {
            "status": "error",
            "message": "TRELLIS.2 is not properly installed on this worker",
            "model": "trellis"
        }
    
    # Extract parameters
    image_base64 = job_input.get("image_base64")
    if not image_base64:
        return {
            "status": "error",
            "message": "Missing required parameter: image_base64",
            "model": "trellis"
        }
    
    output_name = job_input.get("output_name", "trellis_output")
    resolution = job_input.get("resolution", 1024)
    guidance_scale = job_input.get("guidance_scale", 7.5)
    output_glb = job_input.get("output_glb", True)
    output_ply = job_input.get("output_ply", False)
    seed = job_input.get("seed")
    return_base64 = job_input.get("return_base64", False)
    
    # Validate resolution
    valid_resolutions = [512, 768, 1024, 1536]
    if resolution not in valid_resolutions:
        return {
            "status": "error",
            "message": f"Invalid resolution {resolution}. Must be one of: {valid_resolutions}",
            "model": "trellis"
        }
    
    logger.info(f"TRELLIS.2 job: output_name={output_name}, resolution={resolution}, guidance={guidance_scale}")
    
    try:
        # Save input image to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(base64.b64decode(image_base64))
            input_path = f.name
        
        # Create temp output directory
        with tempfile.TemporaryDirectory() as temp_output_dir:
            # Run inference
            start_time = time.time()
            results = run_trellis(
                input_image_path=input_path,
                output_dir=temp_output_dir,
                output_name=output_name,
                resolution=resolution,
                guidance_scale=guidance_scale,
                output_glb=output_glb,
                output_ply=output_ply,
                seed=seed,
            )
            duration = time.time() - start_time
            
            logger.info(f"TRELLIS.2 inference completed in {duration:.1f}s")
            
            # Prepare response
            response = {
                "status": "success",
                "model": "trellis",
                "duration_seconds": duration,
                "resolution": resolution,
            }
            
            # Process GLB output
            if "glb_path" in results:
                glb_path = results["glb_path"]
                glb_size = os.path.getsize(glb_path)
                
                # Copy to persistent storage
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                persistent_glb_path = os.path.join(OUTPUT_DIR, f"{output_name}.glb")
                import shutil
                shutil.copy2(glb_path, persistent_glb_path)
                logger.info(f"TRELLIS.2 GLB saved: {persistent_glb_path}")
                
                response["glb_path"] = persistent_glb_path
                response["glb_size"] = glb_size
                
                # Try S3 upload
                s3_result = upload_file_to_s3_if_large(persistent_glb_path, f"{output_name}.glb")
                if s3_result["uploaded"]:
                    response["glb_s3_url"] = s3_result["s3_url"]
                    response["message"] = f"GLB uploaded to S3 ({glb_size / 1024 / 1024:.1f}MB)"
                elif return_base64:
                    encoded = encode_file_if_small(persistent_glb_path)
                    if encoded:
                        response["glb_base64"] = encoded
                    else:
                        response["download_required"] = True
                else:
                    response["download_required"] = True
            
            # Process PLY output
            if "ply_path" in results:
                ply_path = results["ply_path"]
                ply_size = os.path.getsize(ply_path)
                
                # Copy to persistent storage
                persistent_ply_path = os.path.join(OUTPUT_DIR, f"{output_name}.ply")
                import shutil
                shutil.copy2(ply_path, persistent_ply_path)
                logger.info(f"TRELLIS.2 PLY saved: {persistent_ply_path}")
                
                response["ply_path"] = persistent_ply_path
                response["ply_size"] = ply_size
                
                # Try S3 upload
                s3_result = upload_file_to_s3_if_large(persistent_ply_path, f"{output_name}.ply")
                if s3_result["uploaded"]:
                    response["ply_s3_url"] = s3_result["s3_url"]
                elif return_base64:
                    encoded = encode_file_if_small(persistent_ply_path)
                    if encoded:
                        response["ply_base64"] = encoded
            
            return response
            
    except subprocess.TimeoutExpired:
        logger.error("TRELLIS.2 inference timed out")
        return {
            "status": "error",
            "message": "TRELLIS.2 inference timed out (30 minute limit)",
            "model": "trellis"
        }
    except Exception as e:
        logger.error(f"Handler error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "model": "trellis"
        }
    finally:
        # Cleanup temp input file
        if 'input_path' in locals() and os.path.exists(input_path):
            os.remove(input_path)


# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting TRELLIS.2 RunPod Serverless Handler")
    logger.info(f"TRELLIS.2 Directory: {TRELLIS_DIR}")
    logger.info(f"Checkpoint Directory: {CHECKPOINT_DIR}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info(f"TRELLIS.2 available: {validate_trellis()}")
    
    runpod.serverless.start({"handler": handler})

