#!/usr/bin/env python3
"""
Unified 3D Generation API Server for RunPod Pods

This server supports multiple models via REST API:
- GEN3C: Image to video generation
- SHARP: Image to 3D Gaussian Splatting (PLY)

Endpoints:
    POST /generate/gen3c  - Generate video from image
    POST /generate/sharp  - Generate PLY from image
    GET  /health          - Health check
    GET  /models          - List available models
"""

import os
import sys
import base64
import tempfile
import subprocess
import shutil
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# =============================================================================
# CONFIGURATION
# =============================================================================

GEN3C_DIR = os.environ.get("GEN3C_DIR", "/workspace/GEN3C")
SHARP_DIR = os.environ.get("SHARP_DIR", "/workspace/ml-sharp")
CHECKPOINT_DIR = os.environ.get("GEN3C_CHECKPOINT_DIR", "/workspace/checkpoints")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/outputs")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("3dgen-server")

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="3D Generation API",
    description="Unified API for GEN3C and SHARP models",
    version="1.0.0"
)

# =============================================================================
# REQUEST MODELS
# =============================================================================

class GEN3CRequest(BaseModel):
    image_base64: str
    video_name: Optional[str] = None
    guidance: float = 1.0
    num_frames: int = 121
    trajectory: str = "left"
    foreground_masking: bool = True
    seed: Optional[int] = None
    return_base64: bool = False


class SHARPRequest(BaseModel):
    image_base64: str
    output_name: Optional[str] = None
    render_video: bool = False
    return_base64: bool = False


# =============================================================================
# JOB TRACKING
# =============================================================================

jobs = {}


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# MODEL VALIDATION
# =============================================================================

def check_gen3c_available() -> bool:
    """Check if GEN3C is available."""
    script_path = Path(GEN3C_DIR) / "cosmos_predict1" / "diffusion" / "inference" / "gen3c_single_image.py"
    return script_path.exists() and os.path.exists(CHECKPOINT_DIR)


def check_sharp_available() -> bool:
    """Check if SHARP is available."""
    return shutil.which("sharp") is not None


def check_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


# =============================================================================
# GEN3C INFERENCE
# =============================================================================

def run_gen3c_inference(
    input_path: str,
    video_name: str,
    guidance: float,
    num_frames: int,
    trajectory: str,
    foreground_masking: bool,
    seed: Optional[int]
) -> str:
    """Run GEN3C inference and return output path."""
    script_path = Path(GEN3C_DIR) / "cosmos_predict1" / "diffusion" / "inference" / "gen3c_single_image.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--checkpoint_dir", CHECKPOINT_DIR,
        "--input_image_path", input_path,
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
    
    result = subprocess.run(cmd, env=env, cwd=GEN3C_DIR, capture_output=True, text=True, timeout=7200)
    
    if result.returncode != 0:
        raise RuntimeError(f"GEN3C failed: {result.stderr[-2000:]}")
    
    # Find output
    for output_dir in [Path(GEN3C_DIR) / "videos", Path(GEN3C_DIR) / "outputs"]:
        output_path = output_dir / f"{video_name}.mp4"
        if output_path.exists():
            return str(output_path)
    
    raise RuntimeError("Output video not found")


# =============================================================================
# SHARP INFERENCE
# =============================================================================

def run_sharp_inference(
    input_path: str,
    output_name: str,
    render_video: bool
) -> dict:
    """Run SHARP inference and return output paths."""
    temp_input_dir = tempfile.mkdtemp(prefix="sharp_input_")
    temp_output_dir = tempfile.mkdtemp(prefix="sharp_output_")
    
    try:
        # Copy input
        input_ext = os.path.splitext(input_path)[1]
        temp_input_path = os.path.join(temp_input_dir, f"input{input_ext}")
        shutil.copy2(input_path, temp_input_path)
        
        cmd = ["sharp", "predict", "-i", temp_input_dir, "-o", temp_output_dir]
        if render_video:
            cmd.append("--render")
        
        logger.info(f"Running SHARP: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            raise RuntimeError(f"SHARP failed: {result.stderr[-2000:]}")
        
        # Collect outputs
        results = {}
        output_files = os.listdir(temp_output_dir)
        
        for f in output_files:
            if f.endswith('.ply'):
                dst = os.path.join(OUTPUT_DIR, f"{output_name}.ply")
                shutil.copy2(os.path.join(temp_output_dir, f), dst)
                results["ply_path"] = dst
            elif f.endswith('.mp4'):
                dst = os.path.join(OUTPUT_DIR, f"{output_name}.mp4")
                shutil.copy2(os.path.join(temp_output_dir, f), dst)
                results["video_path"] = dst
        
        if not results:
            raise RuntimeError("No output files generated")
        
        return results
        
    finally:
        shutil.rmtree(temp_input_dir, ignore_errors=True)
        shutil.rmtree(temp_output_dir, ignore_errors=True)


# =============================================================================
# BACKGROUND JOB PROCESSING
# =============================================================================

def process_gen3c_job(job_id: str, input_path: str, request: GEN3CRequest):
    """Process GEN3C job in background."""
    try:
        jobs[job_id]["status"] = JobStatus.RUNNING
        jobs[job_id]["started_at"] = datetime.now().isoformat()
        
        video_name = request.video_name or f"gen3c_{job_id}"
        output_path = run_gen3c_inference(
            input_path, video_name, request.guidance, request.num_frames,
            request.trajectory, request.foreground_masking, request.seed
        )
        
        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["output_path"] = output_path
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"] = str(e)
        logger.exception(f"Job {job_id} failed")
    finally:
        Path(input_path).unlink(missing_ok=True)


def process_sharp_job(job_id: str, input_path: str, request: SHARPRequest):
    """Process SHARP job in background."""
    try:
        jobs[job_id]["status"] = JobStatus.RUNNING
        jobs[job_id]["started_at"] = datetime.now().isoformat()
        
        output_name = request.output_name or f"sharp_{job_id}"
        results = run_sharp_inference(input_path, output_name, request.render_video)
        
        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["outputs"] = results
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"] = str(e)
        logger.exception(f"Job {job_id} failed")
    finally:
        Path(input_path).unlink(missing_ok=True)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    import torch
    
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gen3c_available": check_gen3c_available(),
        "sharp_available": check_sharp_available(),
    }


@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {
                "name": "gen3c",
                "description": "Image to video generation (NVIDIA Cosmos)",
                "available": check_gen3c_available(),
            },
            {
                "name": "sharp",
                "description": "Image to 3D Gaussian Splatting (Apple)",
                "available": check_sharp_available(),
            },
        ]
    }


@app.post("/generate/gen3c")
async def generate_gen3c(request: GEN3CRequest, background_tasks: BackgroundTasks):
    """Generate video from image using GEN3C."""
    if not check_gen3c_available():
        raise HTTPException(status_code=503, detail="GEN3C not available")
    
    # Validate frames
    valid_frames = [121, 241, 361, 481]
    if request.num_frames not in valid_frames:
        raise HTTPException(status_code=400, detail=f"num_frames must be one of {valid_frames}")
    
    # Decode image
    job_id = str(uuid.uuid4())[:8]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(base64.b64decode(request.image_base64))
        input_path = f.name
    
    # Create job
    jobs[job_id] = {
        "id": job_id,
        "model": "gen3c",
        "status": JobStatus.PENDING,
        "created_at": datetime.now().isoformat(),
    }
    
    # Start background processing
    background_tasks.add_task(process_gen3c_job, job_id, input_path, request)
    
    return {"job_id": job_id, "status": "pending", "message": "Job submitted"}


@app.post("/generate/sharp")
async def generate_sharp(request: SHARPRequest, background_tasks: BackgroundTasks):
    """Generate PLY from image using SHARP."""
    if not check_sharp_available():
        raise HTTPException(status_code=503, detail="SHARP not available")
    
    # Decode image
    job_id = str(uuid.uuid4())[:8]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(base64.b64decode(request.image_base64))
        input_path = f.name
    
    # Create job
    jobs[job_id] = {
        "id": job_id,
        "model": "sharp",
        "status": JobStatus.PENDING,
        "created_at": datetime.now().isoformat(),
    }
    
    # Start background processing
    background_tasks.add_task(process_sharp_job, job_id, input_path, request)
    
    return {"job_id": job_id, "status": "pending", "message": "Job submitted"}


@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/job/{job_id}/download")
async def download_output(job_id: str, file_type: str = "primary"):
    """Download job output file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Job not completed: {job['status']}")
    
    # Determine output path
    if job["model"] == "gen3c":
        output_path = job.get("output_path")
    elif job["model"] == "sharp":
        outputs = job.get("outputs", {})
        if file_type == "video" and "video_path" in outputs:
            output_path = outputs["video_path"]
        else:
            output_path = outputs.get("ply_path")
    else:
        output_path = None
    
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(output_path, filename=os.path.basename(output_path))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting Unified 3D Generation API Server")
    logger.info(f"GEN3C available: {check_gen3c_available()}")
    logger.info(f"SHARP available: {check_sharp_available()}")
    logger.info(f"CUDA available: {check_cuda_available()}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

