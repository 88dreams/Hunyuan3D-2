#!/usr/bin/env python3
"""
Hunyuan3D RunPod Serverless Handler

Dedicated handler for Hunyuan3D mesh generation.
Supports Mini (faster) and Full (higher quality) models.

Input:
{
    "input": {
        "image_base64": "...",
        "model": "mini" or "full",
        "guidance_scale": 9.0,
        "steps": 40,
        "octree_resolution": 380,
        "seed": 42,
        "remove_background": true,
        "output_name": "my_model",
        "return_base64": false
    }
}

Output:
{
    "status": "success",
    "model": "hunyuan",
    "glb_path": "/runpod-volume/outputs/hunyuan/my_model.glb",
    "glb_s3_url": "https://...",
    "duration_seconds": 120.5
}
"""

import os
import sys
import time
import base64
import tempfile
import gc
import logging
from typing import Dict, Any, Optional

import torch
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

HUNYUAN_DIR = "/workspace/Hunyuan3D-2"
CHECKPOINT_DIR = os.environ.get("HUNYUAN_CHECKPOINT_DIR", "/runpod-volume/checkpoints/hunyuan")
OUTPUT_DIR = "/runpod-volume/outputs/hunyuan"

# S3 Configuration
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_REGION = os.environ.get("S3_REGION", "us-west-1")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
S3_PREFIX = os.environ.get("S3_PREFIX", "")

# File size limits
MAX_BASE64_SIZE = 8 * 1024 * 1024  # 8MB

# Model paths
MODEL_PATHS = {
    "mini": {
        "repo": "tencent/Hunyuan3D-2mini",
        "subfolder": "hunyuan3d-dit-v2-mini-turbo",
    },
    "full": {
        "repo": "tencent/Hunyuan3D-2",
        "subfolder": "hunyuan3d-dit-v2-0",
    }
}

# Global pipeline cache
_pipeline = None
_current_model = None
_rembg = None

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
        s3_key = f"{S3_PREFIX}/hunyuan/{filename}" if S3_PREFIX else f"hunyuan/{filename}"
        s3_url = upload_to_s3(file_path, s3_key)
        return {"uploaded": s3_url is not None, "s3_url": s3_url}
    return {"uploaded": False, "s3_url": None}


def encode_file_if_small(file_path: str) -> Optional[str]:
    """Encode file to base64 if small enough."""
    file_size = os.path.getsize(file_path)
    if file_size > MAX_BASE64_SIZE:
        logger.info(f"File too large for base64 ({file_size / 1024 / 1024:.1f}MB): {file_path}")
        return None
    
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================================================
# HUNYUAN3D INFERENCE
# ============================================================================

def validate_hunyuan() -> bool:
    """Check if Hunyuan3D is properly installed."""
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        return True
    except ImportError as e:
        logger.error(f"Hunyuan3D not available: {e}")
        return False


def get_pipeline(model_type: str = "mini"):
    """Load and cache the Hunyuan3D pipeline."""
    global _pipeline, _current_model
    
    if _pipeline is not None and _current_model == model_type:
        return _pipeline
    
    # Clear existing pipeline
    if _pipeline is not None:
        del _pipeline
        _pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info(f"Loading Hunyuan3D pipeline: {model_type}")
    
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    
    model_config = MODEL_PATHS.get(model_type, MODEL_PATHS["mini"])
    
    # Always load from HuggingFace with cache_dir for persistent storage
    # This ensures correct model structure and allows caching to network volume
    logger.info(f"Loading model: {model_config['repo']} (subfolder: {model_config['subfolder']})")
    logger.info(f"Cache directory: {CHECKPOINT_DIR}")
    
    _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_config["repo"],
        subfolder=model_config["subfolder"],
        cache_dir=CHECKPOINT_DIR,
        torch_dtype=torch.float16,
    )
    
    _current_model = model_type
    logger.info(f"Pipeline loaded: {model_type}")
    
    return _pipeline


def get_rembg():
    """Load and cache the background remover."""
    global _rembg
    
    if _rembg is None:
        logger.info("Loading background remover...")
        from hy3dgen.rembg import BackgroundRemover
        _rembg = BackgroundRemover()
        logger.info("Background remover loaded")
    
    return _rembg


def run_hunyuan(
    image_path: str,
    output_dir: str,
    output_name: str = "hunyuan_output",
    model_type: str = "mini",
    guidance_scale: float = 9.0,
    steps: int = 40,
    octree_resolution: int = 380,
    seed: Optional[int] = None,
    remove_background: bool = True,
) -> Dict[str, str]:
    """
    Run Hunyuan3D inference.
    
    Returns dict with path to generated GLB.
    """
    from PIL import Image
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    logger.info(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGBA")
    
    # Remove background if requested
    if remove_background:
        logger.info("Removing background...")
        rembg = get_rembg()
        image = rembg(image.convert("RGB")).convert("RGBA")
        logger.info("Background removed")
    
    # Set seed
    if seed is not None:
        torch.manual_seed(seed)
        logger.info(f"Set seed: {seed}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load pipeline
    pipeline = get_pipeline(model_type)
    
    # Adaptive settings based on model
    # Allow user-specified resolution up to 512, with sensible defaults
    if model_type == "mini":
        # Mini model can handle higher resolutions
        octree_resolution = min(octree_resolution, 512)
        num_chunks = 6000
    else:
        # Full model - cap at 450 for stability
        octree_resolution = min(octree_resolution, 450)
        num_chunks = 5000
    
    logger.info(f"Running inference: steps={steps}, guidance={guidance_scale}, resolution={octree_resolution}")
    
    # Run inference
    result = pipeline(
        image=image,
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(steps),
        octree_resolution=octree_resolution,
        num_chunks=num_chunks,
    )
    
    mesh = result[0] if isinstance(result, (list, tuple)) else result
    
    # Export GLB
    glb_path = os.path.join(output_dir, f"{output_name}.glb")
    logger.info(f"Exporting GLB: {glb_path}")
    mesh.export(glb_path)
    
    # Cleanup
    del mesh
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if os.path.exists(glb_path):
        file_size = os.path.getsize(glb_path) / 1024 / 1024
        logger.info(f"GLB exported: {glb_path} ({file_size:.1f}MB)")
        return {"glb_path": glb_path}
    else:
        raise RuntimeError("GLB export failed - file not created")


# ============================================================================
# VERSION QUERY
# ============================================================================

def get_git_version(repo_dir: str) -> Dict[str, str]:
    """Get git commit information for a repository."""
    import subprocess
    
    result = {"commit_sha": "unknown", "commit_date": "unknown", "branch": "unknown"}
    
    if not os.path.exists(repo_dir):
        result["error"] = f"Directory not found: {repo_dir}"
        return result
    
    try:
        sha_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, timeout=5
        )
        if sha_result.returncode == 0:
            result["commit_sha"] = sha_result.stdout.strip()
        
        date_result = subprocess.run(
            ["git", "log", "-1", "--format=%ci"],
            cwd=repo_dir, capture_output=True, text=True, timeout=5
        )
        if date_result.returncode == 0:
            result["commit_date"] = date_result.stdout.strip()[:10]
        
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, timeout=5
        )
        if branch_result.returncode == 0:
            result["branch"] = branch_result.stdout.strip()
    except Exception as e:
        result["error"] = str(e)
    
    if result.get("commit_sha") != "unknown" and result.get("commit_date") != "unknown":
        result["display"] = f"{result['commit_sha']} ({result['commit_date']})"
    else:
        result["display"] = result.get("error", "Not installed")
    
    return result


# ============================================================================
# HANDLER
# ============================================================================

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless Handler for Hunyuan3D.
    
    Supports action: "version" to query installed version.
    """
    job_input = job.get("input", {})
    
    # Check for version query action
    action = job_input.get("action", "").lower()
    if action == "version":
        version_info = get_git_version(HUNYUAN_DIR)
        return {
            "status": "success",
            "action": "version",
            "model": "hunyuan",
            "versions": {"hunyuan": version_info},
            "message": "Version information retrieved successfully"
        }
    
    # Validate installation
    if not validate_hunyuan():
        return {
            "status": "error",
            "message": "Hunyuan3D is not properly installed on this worker",
            "model": "hunyuan"
        }
    
    # Extract parameters
    image_base64 = job_input.get("image_base64")
    if not image_base64:
        return {
            "status": "error",
            "message": "Missing required parameter: image_base64",
            "model": "hunyuan"
        }
    
    # Check model_variant first (from client), then fall back to model
    model_type = job_input.get("model_variant") or job_input.get("model", "mini")
    # Normalize model type - handle various formats
    model_type = model_type.lower() if model_type else "mini"
    if "full" in model_type:
        model_type = "full"
    else:
        model_type = "mini"  # Default to mini for any other value including "turbo"
    
    guidance_scale = job_input.get("guidance_scale", 9.0)
    steps = job_input.get("steps", 40)
    octree_resolution = job_input.get("octree_resolution", 380)
    seed = job_input.get("seed")
    remove_background = job_input.get("remove_background", True)
    output_name = job_input.get("output_name", "hunyuan_output")
    return_base64 = job_input.get("return_base64", False)
    
    logger.info(f"Hunyuan3D job: model={model_type}, output_name={output_name}, steps={steps}")
    
    try:
        # Save input image to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(base64.b64decode(image_base64))
            input_path = f.name
        
        # Run inference
        start_time = time.time()
        results = run_hunyuan(
            image_path=input_path,
            output_dir=OUTPUT_DIR,
            output_name=output_name,
            model_type=model_type,
            guidance_scale=guidance_scale,
            steps=steps,
            octree_resolution=octree_resolution,
            seed=seed,
            remove_background=remove_background,
        )
        duration = time.time() - start_time
        
        logger.info(f"Hunyuan3D inference completed in {duration:.1f}s")
        
        # Prepare response
        response = {
            "status": "success",
            "model": "hunyuan",
            "model_variant": model_type,
            "duration_seconds": duration,
        }
        
        # Process GLB output
        glb_path = results["glb_path"]
        glb_size = os.path.getsize(glb_path)
        
        response["glb_path"] = glb_path
        response["glb_size"] = glb_size
        
        # Try S3 upload for large files
        s3_result = upload_file_to_s3_if_large(glb_path, f"{output_name}.glb")
        if s3_result["uploaded"]:
            response["glb_s3_url"] = s3_result["s3_url"]
            response["message"] = f"GLB uploaded to S3 ({glb_size / 1024 / 1024:.1f}MB)"
        elif return_base64:
            encoded = encode_file_if_small(glb_path)
            if encoded:
                response["glb_base64"] = encoded
            else:
                response["download_required"] = True
        else:
            response["download_required"] = True
        
        return response
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "model": "hunyuan"
        }
    finally:
        # Cleanup temp input file
        if 'input_path' in locals() and os.path.exists(input_path):
            os.remove(input_path)


# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Hunyuan3D RunPod Serverless Handler")
    logger.info(f"Hunyuan3D Directory: {HUNYUAN_DIR}")
    logger.info(f"Checkpoint Directory: {CHECKPOINT_DIR}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info(f"Hunyuan3D available: {validate_hunyuan()}")
    
    runpod.serverless.start({"handler": handler})

