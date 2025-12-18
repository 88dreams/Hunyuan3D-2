#!/usr/bin/env python3
"""
GEN3C API Server for RunPod GPU Pods

This server provides a REST API for GEN3C video generation.
It's designed to run on RunPod GPU pods with an A100 80GB GPU.

Endpoints:
    POST /generate - Generate video from image
    GET /health - Health check
    GET /status/{job_id} - Check job status
    GET /download/{job_id} - Download generated video
    GET /jobs - List all jobs
"""

import os
import sys
import uuid
import base64
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import logging
import threading

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths - FIXED: Use parent directory, GEN3C appends model name internally
GEN3C_DIR = os.environ.get("GEN3C_DIR", "/workspace/GEN3C")
CHECKPOINT_DIR = os.environ.get("GEN3C_CHECKPOINT_DIR", "/workspace/checkpoints")  # FIXED: removed /Gen3C-Cosmos-7B
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/outputs")
TEMP_DIR = os.environ.get("TEMP_DIR", "/tmp/gen3c")

# GEN3C saves videos here (not /videos as originally assumed)
GEN3C_OUTPUT_DIR = Path(GEN3C_DIR) / "outputs"

# Ensure directories exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gen3c-server")

# =============================================================================
# DATA MODELS
# =============================================================================

class GenerateRequest(BaseModel):
    """Request model for video generation."""
    image_base64: str = Field(..., description="Base64 encoded input image (PNG/JPG)")
    video_name: str = Field(default="gen3c_output", description="Output video filename (without extension)")
    guidance: float = Field(default=1.0, ge=0.5, le=3.0, description="Guidance scale")
    num_frames: int = Field(default=121, description="Number of frames (121, 241, 361, 481)")
    trajectory: str = Field(default="left", description="Camera trajectory (left, right, zoom_in, zoom_out, orbit)")
    foreground_masking: bool = Field(default=True, description="Enable foreground masking")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

class GenerateResponse(BaseModel):
    """Response model for video generation."""
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    """Job status model."""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: Optional[float] = None
    message: Optional[str] = None
    video_url: Optional[str] = None
    video_base64: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None

# =============================================================================
# JOB TRACKING
# =============================================================================

# In-memory job storage (for single-pod deployment)
# Using a lock for thread safety
jobs: Dict[str, JobStatus] = {}
jobs_lock = threading.Lock()

# =============================================================================
# GEN3C INFERENCE
# =============================================================================

def run_gen3c_inference(
    input_image_path: str,
    video_name: str,
    guidance: float = 1.0,
    num_frames: int = 121,
    trajectory: str = "left",
    foreground_masking: bool = True,
    seed: Optional[int] = None
) -> str:
    """
    Run GEN3C inference and return the output video path.
    
    Returns:
        Path to the generated video file
    """
    # Build command
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
    
    # Set environment
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{GEN3C_DIR}:{env.get('PYTHONPATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    logger.info(f"Running GEN3C command: {' '.join(cmd)}")
    
    # Run inference
    result = subprocess.run(
        cmd,
        env=env,
        cwd=GEN3C_DIR,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"GEN3C failed: {result.stderr}")
        raise RuntimeError(f"GEN3C inference failed: {result.stderr}")
    
    logger.info(f"GEN3C stdout: {result.stdout}")
    
    # FIXED: GEN3C saves to /workspace/GEN3C/outputs/, not /videos/
    output_video = GEN3C_OUTPUT_DIR / f"{video_name}.mp4"
    
    # Also check alternate locations
    if not output_video.exists():
        alt_path = Path(GEN3C_DIR) / "videos" / f"{video_name}.mp4"
        if alt_path.exists():
            output_video = alt_path
    
    if not output_video.exists():
        # Search for the video
        for search_dir in [GEN3C_OUTPUT_DIR, Path(GEN3C_DIR) / "videos", Path(GEN3C_DIR)]:
            if search_dir.exists():
                matches = list(search_dir.glob(f"*{video_name}*.mp4"))
                if matches:
                    output_video = matches[0]
                    break
    
    if not output_video.exists():
        raise RuntimeError(f"Output video not found. Searched in {GEN3C_OUTPUT_DIR} and {GEN3C_DIR}/videos/")
    
    # Copy to output directory
    final_path = Path(OUTPUT_DIR) / f"{video_name}.mp4"
    shutil.copy(output_video, final_path)
    
    return str(final_path)

def process_job_sync(job_id: str, request: GenerateRequest):
    """Synchronous job processing (runs in a thread)."""
    with jobs_lock:
        jobs[job_id].status = "running"
        jobs[job_id].started_at = datetime.now().isoformat()
    
    try:
        # Decode image
        image_data = base64.b64decode(request.image_base64)
        input_path = Path(TEMP_DIR) / f"{job_id}_input.png"
        with open(input_path, "wb") as f:
            f.write(image_data)
        
        logger.info(f"Job {job_id}: Starting GEN3C inference")
        
        # Run inference
        output_path = run_gen3c_inference(
            input_image_path=str(input_path),
            video_name=f"{job_id}_{request.video_name}",
            guidance=request.guidance,
            num_frames=request.num_frames,
            trajectory=request.trajectory,
            foreground_masking=request.foreground_masking,
            seed=request.seed
        )
        
        # Read video and encode as base64
        with open(output_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        with jobs_lock:
            jobs[job_id].status = "completed"
            jobs[job_id].completed_at = datetime.now().isoformat()
            jobs[job_id].video_url = output_path
            jobs[job_id].video_base64 = video_base64
            jobs[job_id].message = "Video generation completed successfully"
        
        logger.info(f"Job {job_id}: Completed successfully")
        
        # Cleanup temp file
        input_path.unlink(missing_ok=True)
        
    except Exception as e:
        logger.error(f"Job {job_id}: Failed with error: {e}")
        with jobs_lock:
            jobs[job_id].status = "failed"
            jobs[job_id].completed_at = datetime.now().isoformat()
            jobs[job_id].error = str(e)

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="GEN3C API Server",
    description="REST API for GEN3C video generation on RunPod",
    version="1.1.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    import torch
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "not detected"
    
    # Check checkpoint structure
    checkpoint_model = Path(CHECKPOINT_DIR) / "Gen3C-Cosmos-7B" / "model.pt"
    tokenizer_path = Path(CHECKPOINT_DIR) / "Cosmos-Tokenize1-CV8x8x8-720p" / "mean_std.pt"
    
    return {
        "status": "healthy",
        "gpu": gpu_name if gpu_available else "not detected",
        "gpu_available": gpu_available,
        "checkpoint_dir": CHECKPOINT_DIR,
        "model_exists": checkpoint_model.exists(),
        "tokenizer_exists": tokenizer_path.exists(),
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_video(request: GenerateRequest):
    """
    Submit a video generation job.
    
    Returns immediately with a job_id. Use /status/{job_id} to check progress.
    """
    # Validate frames
    valid_frames = [121, 241, 361, 481]
    if request.num_frames not in valid_frames:
        raise HTTPException(
            status_code=400,
            detail=f"num_frames must be one of {valid_frames}"
        )
    
    # Create job
    job_id = str(uuid.uuid4())[:8]
    with jobs_lock:
        jobs[job_id] = JobStatus(
            job_id=job_id,
            status="pending",
            message="Job queued for processing"
        )
    
    # Start processing in a separate thread (non-blocking)
    thread = threading.Thread(target=process_job_sync, args=(job_id, request))
    thread.start()
    
    return GenerateResponse(
        job_id=job_id,
        status="pending",
        message="Job submitted successfully. Use /status/{job_id} to check progress."
    )

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a generation job."""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return jobs[job_id]

@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """Download the generated video file."""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        job = jobs[job_id]
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job.status}")
    
    if not job.video_url or not os.path.exists(job.video_url):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        job.video_url,
        media_type="video/mp4",
        filename=os.path.basename(job.video_url)
    )

@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    with jobs_lock:
        return {"jobs": list(jobs.values())}

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting GEN3C API Server...")
    logger.info(f"GEN3C Directory: {GEN3C_DIR}")
    logger.info(f"Checkpoint Directory: {CHECKPOINT_DIR}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
