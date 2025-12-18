#!/usr/bin/env python3
"""
GEN3C RunPod Serverless Handler

This handler is used for RunPod Serverless Endpoints.
It processes jobs submitted via the RunPod API.

Usage:
    RunPod automatically calls this handler when a job is submitted
    to the serverless endpoint.
"""

import os
import sys
import base64
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import runpod

# =============================================================================
# CONFIGURATION
# =============================================================================

GEN3C_DIR = os.environ.get("GEN3C_DIR", "/workspace/GEN3C")
CHECKPOINT_DIR = os.environ.get("GEN3C_CHECKPOINT_DIR", "/workspace/checkpoints")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/outputs")

# Ensure directories exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gen3c-handler")

# =============================================================================
# MODEL LOADING (Cached)
# =============================================================================

# Global model reference (loaded once, reused across requests)
_model_loaded = False

def ensure_model_loaded():
    """
    Ensure the model environment is ready.
    For GEN3C, we don't pre-load the model since it handles loading internally.
    This function just validates the environment.
    """
    global _model_loaded
    
    if _model_loaded:
        return
    
    # Validate checkpoint directory
    if not os.path.exists(CHECKPOINT_DIR):
        raise RuntimeError(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
    
    # Validate GEN3C installation
    script_path = Path(GEN3C_DIR) / "cosmos_predict1" / "diffusion" / "inference" / "gen3c_single_image.py"
    if not script_path.exists():
        raise RuntimeError(f"GEN3C script not found: {script_path}")
    
    logger.info("GEN3C environment validated")
    _model_loaded = True

# =============================================================================
# INFERENCE
# =============================================================================

def run_gen3c(
    input_image_path: str,
    video_name: str,
    guidance: float = 1.0,
    num_frames: int = 121,
    trajectory: str = "left",
    foreground_masking: bool = True,
    seed: Optional[int] = None
) -> str:
    """
    Run GEN3C inference.
    
    Returns:
        Path to the generated video file
    """
    script_path = Path(GEN3C_DIR) / "cosmos_predict1" / "diffusion" / "inference" / "gen3c_single_image.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--checkpoint_dir", CHECKPOINT_DIR,
        "--input_image_path", input_image_path,
        "--video_save_name", video_name,
        "--guidance", str(guidance),
        "--num_video_frames", str(num_frames),
        "--trajectory", trajectory,
        "--offload_diffusion_transformer",
        "--offload_tokenizer",
        "--offload_text_encoder_model",
        "--disable_prompt_upsampler",
        "--disable_guardrail",
    ]
    
    if foreground_masking:
        cmd.append("--foreground_masking")
    
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{GEN3C_DIR}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        env=env,
        cwd=GEN3C_DIR,
        capture_output=True,
        text=True,
        timeout=7200  # 2 hour timeout
    )
    
    if result.returncode != 0:
        logger.error(f"GEN3C stderr: {result.stderr}")
        raise RuntimeError(f"GEN3C failed: {result.stderr[-2000:]}")  # Last 2000 chars
    
    # Find output - check both possible locations
    output_video_1 = Path(GEN3C_DIR) / "videos" / f"{video_name}.mp4"
    output_video_2 = Path(GEN3C_DIR) / "outputs" / f"{video_name}.mp4"
    
    if output_video_1.exists():
        return str(output_video_1)
    elif output_video_2.exists():
        return str(output_video_2)
    else:
        raise RuntimeError(f"Output video not found at {output_video_1} or {output_video_2}")

# =============================================================================
# HANDLER
# =============================================================================

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function.
    
    Input (job["input"]):
        - image_base64: Base64 encoded input image
        - video_name: Output video name (optional)
        - guidance: Guidance scale (optional, default 1.0)
        - num_frames: Number of frames (optional, default 121)
        - trajectory: Camera trajectory (optional, default "left")
        - foreground_masking: Enable masking (optional, default True)
        - seed: Random seed (optional)
        - return_base64: Return video as base64 (optional, default True)
    
    Output:
        - video_base64: Base64 encoded video (if return_base64=True)
        - video_path: Path to video file (always)
        - status: "success" or "error"
        - message: Status message
    """
    try:
        # Validate environment
        ensure_model_loaded()
        
        # Parse input
        job_input = job.get("input", {})
        
        if "image_base64" not in job_input:
            return {"status": "error", "message": "Missing required field: image_base64"}
        
        # Extract parameters
        image_base64 = job_input["image_base64"]
        video_name = job_input.get("video_name", f"gen3c_{job.get('id', 'output')}")
        guidance = float(job_input.get("guidance", 1.0))
        num_frames = int(job_input.get("num_frames", 121))
        trajectory = job_input.get("trajectory", "left")
        foreground_masking = job_input.get("foreground_masking", True)
        seed = job_input.get("seed")
        return_base64 = job_input.get("return_base64", True)
        
        # Validate frames
        valid_frames = [121, 241, 361, 481]
        if num_frames not in valid_frames:
            return {"status": "error", "message": f"num_frames must be one of {valid_frames}"}
        
        # Decode image to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(base64.b64decode(image_base64))
            input_path = f.name
        
        logger.info(f"Processing job: video_name={video_name}, frames={num_frames}, trajectory={trajectory}")
        
        # Run inference
        output_path = run_gen3c(
            input_image_path=input_path,
            video_name=video_name,
            guidance=guidance,
            num_frames=num_frames,
            trajectory=trajectory,
            foreground_masking=foreground_masking,
            seed=seed
        )
        
        # Prepare response
        response = {
            "status": "success",
            "message": "Video generated successfully",
            "video_path": output_path,
            "video_name": f"{video_name}.mp4"
        }
        
        # Optionally include base64 encoded video
        if return_base64:
            with open(output_path, "rb") as f:
                response["video_base64"] = base64.b64encode(f.read()).decode("utf-8")
        
        # Cleanup temp file
        Path(input_path).unlink(missing_ok=True)
        
        return response
        
    except Exception as e:
        logger.exception("Handler error")
        return {
            "status": "error",
            "message": str(e)
        }

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting GEN3C RunPod Serverless Handler")
    logger.info(f"GEN3C Directory: {GEN3C_DIR}")
    logger.info(f"Checkpoint Directory: {CHECKPOINT_DIR}")
    
    # Start the serverless worker
    runpod.serverless.start({"handler": handler})

