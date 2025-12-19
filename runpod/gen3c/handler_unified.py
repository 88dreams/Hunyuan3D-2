#!/usr/bin/env python3
"""
Unified 3D Generation RunPod Serverless Handler

This handler supports multiple models:
- GEN3C: Image to video generation (NVIDIA Cosmos)
- SHARP: Image to 3D Gaussian Splatting (Apple)
- Lyra: Image/Video to 3D/4D Gaussian Splatting (NVIDIA) [Coming Soon]
- TRELLIS.2: Image to 3D with O-Voxel (Microsoft) [Coming Soon]

Environment:
    Python 3.10 + NumPy 1.26.4 + PyTorch 2.6.0 (NVIDIA stack)

Usage:
    RunPod automatically calls this handler when a job is submitted
    to the serverless endpoint.
"""

import os
import sys
import base64
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import runpod

# =============================================================================
# CONFIGURATION
# =============================================================================

GEN3C_DIR = os.environ.get("GEN3C_DIR", "/workspace/GEN3C")
SHARP_DIR = os.environ.get("SHARP_DIR", "/workspace/ml-sharp")
CHECKPOINT_DIR = os.environ.get("GEN3C_CHECKPOINT_DIR", "/workspace/checkpoints")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/outputs")

# Ensure directories exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("3dgen-handler")

# =============================================================================
# MODEL VALIDATION
# =============================================================================

_gen3c_validated = False
_sharp_validated = False


def validate_gen3c():
    """Validate GEN3C environment."""
    global _gen3c_validated
    if _gen3c_validated:
        return True
    
    script_path = Path(GEN3C_DIR) / "cosmos_predict1" / "diffusion" / "inference" / "gen3c_single_image.py"
    if not script_path.exists():
        logger.warning(f"GEN3C script not found: {script_path}")
        return False
    
    if not os.path.exists(CHECKPOINT_DIR):
        logger.warning(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
        return False
    
    _gen3c_validated = True
    logger.info("GEN3C environment validated")
    return True


def validate_sharp():
    """Validate SHARP environment."""
    global _sharp_validated
    if _sharp_validated:
        return True
    
    # Check if 'sharp' CLI is available
    sharp_path = shutil.which("sharp")
    if not sharp_path:
        logger.warning("SHARP CLI not found in PATH")
        return False
    
    _sharp_validated = True
    logger.info(f"SHARP environment validated (CLI: {sharp_path})")
    return True


# =============================================================================
# GEN3C INFERENCE
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
    
    logger.info(f"Running GEN3C: {' '.join(cmd)}")
    
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
        raise RuntimeError(f"GEN3C failed: {result.stderr[-2000:]}")
    
    # Find output
    output_video_1 = Path(GEN3C_DIR) / "videos" / f"{video_name}.mp4"
    output_video_2 = Path(GEN3C_DIR) / "outputs" / f"{video_name}.mp4"
    
    if output_video_1.exists():
        return str(output_video_1)
    elif output_video_2.exists():
        return str(output_video_2)
    else:
        raise RuntimeError(f"Output video not found at {output_video_1} or {output_video_2}")


# =============================================================================
# SHARP INFERENCE
# =============================================================================

def run_sharp(
    input_image_path: str,
    output_name: str,
    render_video: bool = False,
) -> Dict[str, str]:
    """
    Run SHARP inference.
    
    Args:
        input_image_path: Path to input image
        output_name: Base name for output files
        render_video: Whether to render video trajectory
    
    Returns:
        Dict with paths to generated files (ply_path, video_path if applicable)
    """
    # Create temp directories
    temp_input_dir = tempfile.mkdtemp(prefix="sharp_input_")
    temp_output_dir = tempfile.mkdtemp(prefix="sharp_output_")
    
    try:
        # Copy input image to temp directory (SHARP expects a directory)
        input_ext = os.path.splitext(input_image_path)[1]
        temp_input_path = os.path.join(temp_input_dir, f"input{input_ext}")
        shutil.copy2(input_image_path, temp_input_path)
        
        # Build command
        cmd = [
            "sharp", "predict",
            "-i", temp_input_dir,
            "-o", temp_output_dir,
        ]
        
        if render_video:
            cmd.append("--render")
        
        logger.info(f"Running SHARP: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        
        if result.returncode != 0:
            logger.error(f"SHARP stderr: {result.stderr}")
            raise RuntimeError(f"SHARP failed: {result.stderr[-2000:]}")
        
        # Find output files
        output_files = os.listdir(temp_output_dir)
        ply_files = [f for f in output_files if f.endswith('.ply')]
        mp4_files = [f for f in output_files if f.endswith('.mp4')]
        
        results = {}
        
        # Copy PLY to output directory
        if ply_files:
            src_ply = os.path.join(temp_output_dir, ply_files[0])
            dst_ply = os.path.join(OUTPUT_DIR, f"{output_name}.ply")
            shutil.copy2(src_ply, dst_ply)
            results["ply_path"] = dst_ply
            logger.info(f"SHARP PLY saved: {dst_ply}")
        
        # Copy video if rendered
        if mp4_files:
            src_video = os.path.join(temp_output_dir, mp4_files[0])
            dst_video = os.path.join(OUTPUT_DIR, f"{output_name}.mp4")
            shutil.copy2(src_video, dst_video)
            results["video_path"] = dst_video
            logger.info(f"SHARP video saved: {dst_video}")
        
        if not results:
            raise RuntimeError("No output files generated by SHARP")
        
        return results
        
    finally:
        # Cleanup
        shutil.rmtree(temp_input_dir, ignore_errors=True)
        shutil.rmtree(temp_output_dir, ignore_errors=True)


# =============================================================================
# UNIFIED HANDLER
# =============================================================================

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function.
    
    Input (job["input"]):
        Common:
        - model: "gen3c" or "sharp" (required)
        - image_base64: Base64 encoded input image (required)
        - output_name: Output file name (optional)
        - return_base64: Return output as base64 (optional, default True)
        
        GEN3C specific:
        - guidance: Guidance scale (default 1.0)
        - num_frames: Number of frames (default 121)
        - trajectory: Camera trajectory (default "left")
        - foreground_masking: Enable masking (default True)
        - seed: Random seed (optional)
        
        SHARP specific:
        - render_video: Render video trajectory (default False)
    
    Output:
        - status: "success" or "error"
        - message: Status message
        - Model-specific outputs (video_base64, ply_base64, etc.)
    """
    try:
        job_input = job.get("input", {})
        
        # Determine model type
        model = job_input.get("model", "gen3c").lower()
        
        if "image_base64" not in job_input:
            return {"status": "error", "message": "Missing required field: image_base64"}
        
        # Decode image to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(base64.b64decode(job_input["image_base64"]))
            input_path = f.name
        
        return_base64 = job_input.get("return_base64", True)
        
        try:
            if model == "gen3c":
                return handle_gen3c(job, job_input, input_path, return_base64)
            elif model == "sharp":
                return handle_sharp(job, job_input, input_path, return_base64)
            else:
                return {"status": "error", "message": f"Unknown model: {model}. Supported: gen3c, sharp"}
        finally:
            # Cleanup temp input
            Path(input_path).unlink(missing_ok=True)
            
    except Exception as e:
        logger.exception("Handler error")
        return {"status": "error", "message": str(e)}


def handle_gen3c(job: Dict, job_input: Dict, input_path: str, return_base64: bool) -> Dict:
    """Handle GEN3C job."""
    if not validate_gen3c():
        return {"status": "error", "message": "GEN3C environment not available"}
    
    video_name = job_input.get("video_name", job_input.get("output_name", f"gen3c_{job.get('id', 'output')}"))
    guidance = float(job_input.get("guidance", 1.0))
    num_frames = int(job_input.get("num_frames", 121))
    trajectory = job_input.get("trajectory", "left")
    foreground_masking = job_input.get("foreground_masking", True)
    seed = job_input.get("seed")
    
    # Validate frames
    valid_frames = [121, 241, 361, 481]
    if num_frames not in valid_frames:
        return {"status": "error", "message": f"num_frames must be one of {valid_frames}"}
    
    logger.info(f"GEN3C job: video_name={video_name}, frames={num_frames}, trajectory={trajectory}")
    
    output_path = run_gen3c(
        input_image_path=input_path,
        video_name=video_name,
        guidance=guidance,
        num_frames=num_frames,
        trajectory=trajectory,
        foreground_masking=foreground_masking,
        seed=seed
    )
    
    response = {
        "status": "success",
        "message": "Video generated successfully",
        "model": "gen3c",
        "video_path": output_path,
        "video_name": f"{video_name}.mp4"
    }
    
    if return_base64:
        with open(output_path, "rb") as f:
            response["video_base64"] = base64.b64encode(f.read()).decode("utf-8")
    
    return response


def handle_sharp(job: Dict, job_input: Dict, input_path: str, return_base64: bool) -> Dict:
    """Handle SHARP job."""
    if not validate_sharp():
        return {"status": "error", "message": "SHARP environment not available"}
    
    output_name = job_input.get("output_name", f"sharp_{job.get('id', 'output')}")
    render_video = job_input.get("render_video", False)
    
    logger.info(f"SHARP job: output_name={output_name}, render_video={render_video}")
    
    results = run_sharp(
        input_image_path=input_path,
        output_name=output_name,
        render_video=render_video,
    )
    
    response = {
        "status": "success",
        "message": "PLY generated successfully",
        "model": "sharp",
        "ply_name": f"{output_name}.ply",
    }
    
    if "ply_path" in results:
        response["ply_path"] = results["ply_path"]
        if return_base64:
            with open(results["ply_path"], "rb") as f:
                response["ply_base64"] = base64.b64encode(f.read()).decode("utf-8")
    
    if "video_path" in results:
        response["video_path"] = results["video_path"]
        response["video_name"] = f"{output_name}.mp4"
        if return_base64:
            with open(results["video_path"], "rb") as f:
                response["video_base64"] = base64.b64encode(f.read()).decode("utf-8")
    
    return response


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting Unified 3D Generation RunPod Serverless Handler")
    logger.info(f"GEN3C Directory: {GEN3C_DIR}")
    logger.info(f"SHARP Directory: {SHARP_DIR}")
    logger.info(f"Checkpoint Directory: {CHECKPOINT_DIR}")
    
    # Validate environments at startup
    gen3c_ok = validate_gen3c()
    sharp_ok = validate_sharp()
    
    logger.info(f"GEN3C available: {gen3c_ok}")
    logger.info(f"SHARP available: {sharp_ok}")
    
    # Start the serverless worker
    runpod.serverless.start({"handler": handler})

