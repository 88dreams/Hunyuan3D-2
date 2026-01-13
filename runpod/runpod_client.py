#!/usr/bin/env python3
"""
RunPod GEN3C Client

This module provides a client for interacting with the GEN3C API running on RunPod.
It can be used standalone or integrated into the Gradio UI.
"""

import base64
import requests
import time
import os
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from enum import Enum

# SSH download support
try:
    from runpod.ssh_download import (
        download_from_runpod,
        load_ssh_config,
        SSHConfig,
    )
    _ssh_available = True
except ImportError:
    _ssh_available = False
    download_from_runpod = None
    load_ssh_config = None
    SSHConfig = None

# S3 download support
try:
    import boto3
    from botocore.exceptions import ClientError
    _s3_available = True
except ImportError:
    _s3_available = False
    boto3 = None
    ClientError = Exception


def download_from_s3(s3_url: str, local_dir: str) -> tuple[Optional[str], str]:
    """
    Download a file from S3 to local directory.
    
    Args:
        s3_url: Full S3 URL (https://bucket.s3.region.amazonaws.com/key)
        local_dir: Local directory to save the file
        
    Returns:
        Tuple of (local_path, message)
    """
    if not _s3_available:
        return None, "❌ boto3 not installed. Run: pip install boto3"
    
    try:
        # Parse S3 URL
        # Format: https://bucket.s3.region.amazonaws.com/key
        # or: https://bucket.s3.amazonaws.com/key (us-east-1)
        import re
        
        # Try regional format first
        match = re.match(r'https://([^.]+)\.s3\.([^.]+)\.amazonaws\.com/(.+)', s3_url)
        if match:
            bucket = match.group(1)
            region = match.group(2)
            key = match.group(3)
        else:
            # Try us-east-1 format
            match = re.match(r'https://([^.]+)\.s3\.amazonaws\.com/(.+)', s3_url)
            if match:
                bucket = match.group(1)
                region = "us-east-1"
                key = match.group(2)
            else:
                return None, f"❌ Could not parse S3 URL: {s3_url}"
        
        # Create local path
        filename = os.path.basename(key)
        local_path = os.path.join(local_dir, filename)
        os.makedirs(local_dir, exist_ok=True)
        
        # Download using requests (public bucket) or boto3 (private)
        # Try public download first (simpler, no credentials needed)
        import requests
        response = requests.get(s3_url, stream=True, timeout=300)
        
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = os.path.getsize(local_path)
            return local_path, f"✅ Downloaded {filename} ({file_size / 1024 / 1024:.1f}MB) from S3"
        elif response.status_code == 403:
            return None, f"❌ Access denied to S3 object. Check bucket policy."
        else:
            return None, f"❌ S3 download failed: HTTP {response.status_code}"
            
    except requests.exceptions.Timeout:
        return None, "❌ S3 download timed out"
    except Exception as e:
        return None, f"❌ S3 download error: {e}"


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RunPodJobResult:
    """Result from a RunPod job (GEN3C, SHARP, Lyra, or TRELLIS)."""
    success: bool
    job_id: str
    status: JobStatus
    model: str = "gen3c"  # "gen3c", "sharp", "lyra", or "trellis"
    output_path: Optional[str] = None
    video_base64: Optional[str] = None
    ply_base64: Optional[str] = None
    glb_base64: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    logs: str = ""
    # For large files that couldn't be returned via API
    download_required: bool = False
    remote_ply_path: Optional[str] = None
    remote_video_path: Optional[str] = None
    remote_glb_path: Optional[str] = None


class RunPodGEN3CClient:
    """Client for RunPod GEN3C API."""
    
    # Valid options for GEN3C
    VALID_TRAJECTORIES = [
        "left", "right", "up", "down", 
        "zoom_in", "zoom_out", 
        "clockwise", "counterclockwise", 
        "none"
    ]
    VALID_FRAME_COUNTS = [121, 241, 361, 481]
    
    def __init__(self, api_url: str, timeout: int = 30):
        """
        Initialize the RunPod GEN3C client.
        
        Args:
            api_url: Full URL to the RunPod API (e.g., https://xxx-8000.proxy.runpod.net)
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the RunPod API is healthy and ready."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": str(e),
                "gpu_available": False,
                "model_exists": False,
                "tokenizer_exists": False
            }
    
    def is_ready(self) -> bool:
        """Check if the RunPod instance is ready for inference."""
        health = self.health_check()
        return (
            health.get("status") == "healthy" and
            health.get("gpu_available", False) and
            health.get("model_exists", False) and
            health.get("tokenizer_exists", False)
        )
    
    def submit_job(
        self,
        image_path: str,
        video_name: str = "gen3c_output",
        num_frames: int = 121,
        trajectory: str = "left",
        guidance: float = 1.0,
        foreground_masking: bool = True,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Submit a GEN3C generation job.
        
        Args:
            image_path: Path to the input image
            video_name: Name for the output video
            num_frames: Number of frames (121, 241, 361, 481)
            trajectory: Camera trajectory
            guidance: Guidance scale (0.5-3.0)
            foreground_masking: Enable foreground masking
            seed: Random seed for reproducibility
            
        Returns:
            Dict with job_id and status
        """
        # Validate inputs
        if num_frames not in self.VALID_FRAME_COUNTS:
            raise ValueError(f"num_frames must be one of {self.VALID_FRAME_COUNTS}")
        if trajectory not in self.VALID_TRAJECTORIES:
            raise ValueError(f"trajectory must be one of {self.VALID_TRAJECTORIES}")
        
        # Read and encode image
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()
        
        # Build request
        payload = {
            "image_base64": image_base64,
            "video_name": video_name,
            "num_frames": num_frames,
            "trajectory": trajectory,
            "guidance": guidance,
            "foreground_masking": foreground_masking
        }
        if seed is not None:
            payload["seed"] = seed
        
        response = requests.post(
            f"{self.api_url}/generate",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a job."""
        response = requests.get(
            f"{self.api_url}/status/{job_id}",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a running job on the Pod.
        
        Note: This requires the server to implement a /cancel endpoint.
        """
        try:
            response = requests.post(
                f"{self.api_url}/cancel/{job_id}",
                timeout=self.timeout
            )
            response.raise_for_status()
            return {"success": True, "message": f"Job {job_id} cancelled"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 30,
        max_wait: int = 3600,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Wait for a job to complete.
        
        Args:
            job_id: The job ID to wait for
            poll_interval: Seconds between status checks
            max_wait: Maximum seconds to wait
            progress_callback: Optional callback(status, elapsed_seconds)
            
        Returns:
            Final job status dict
        """
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > max_wait:
                return {
                    "job_id": job_id,
                    "status": "failed",
                    "error": f"Timeout after {max_wait} seconds"
                }
            
            try:
                status = self.get_status(job_id)
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Status check failed: {e}", elapsed)
                time.sleep(poll_interval)
                continue
            
            if progress_callback:
                progress_callback(status.get("status", "unknown"), elapsed)
            
            if status.get("status") in ["completed", "failed"]:
                return status
            
            time.sleep(poll_interval)
    
    def generate_sync(
        self,
        image_path: str,
        output_dir: str,
        video_name: str = "gen3c_output",
        num_frames: int = 121,
        trajectory: str = "left",
        guidance: float = 1.0,
        foreground_masking: bool = True,
        seed: Optional[int] = None,
        poll_interval: int = 30,
        max_wait: int = 3600,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> RunPodJobResult:
        """
        Submit a job and wait for completion synchronously.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save output video
            video_name: Name for output video
            num_frames: Number of frames
            trajectory: Camera trajectory
            guidance: Guidance scale
            foreground_masking: Enable foreground masking
            seed: Random seed
            poll_interval: Seconds between status checks
            max_wait: Maximum wait time
            progress_callback: Optional progress callback
            
        Returns:
            RunPodJobResult with success/failure info
        """
        start_time = time.time()
        logs = []
        
        # Check health first
        health = self.health_check()
        if health.get("status") != "healthy":
            return RunPodJobResult(
                success=False,
                job_id="",
                status=JobStatus.FAILED,
                error=f"RunPod not ready: {health.get('error', 'Unknown error')}",
                logs="Health check failed"
            )
        
        logs.append(f"RunPod ready: GPU={health.get('gpu', 'unknown')}")
        
        # Submit job
        try:
            submit_result = self.submit_job(
                image_path=image_path,
                video_name=video_name,
                num_frames=num_frames,
                trajectory=trajectory,
                guidance=guidance,
                foreground_masking=foreground_masking,
                seed=seed
            )
        except Exception as e:
            return RunPodJobResult(
                success=False,
                job_id="",
                status=JobStatus.FAILED,
                error=f"Failed to submit job: {e}",
                logs="\n".join(logs)
            )
        
        job_id = submit_result.get("job_id", "")
        logs.append(f"Job submitted: {job_id}")
        
        if progress_callback:
            progress_callback("pending", 0)
        
        # Wait for completion
        final_status = self.wait_for_completion(
            job_id=job_id,
            poll_interval=poll_interval,
            max_wait=max_wait,
            progress_callback=progress_callback
        )
        
        duration = time.time() - start_time
        
        if final_status.get("status") == "completed":
            # Save video
            output_path = None
            if final_status.get("video_base64"):
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{video_name}.mp4")
                video_data = base64.b64decode(final_status["video_base64"])
                with open(output_path, "wb") as f:
                    f.write(video_data)
                logs.append(f"Video saved: {output_path}")
            
            return RunPodJobResult(
                success=True,
                job_id=job_id,
                status=JobStatus.COMPLETED,
                output_path=output_path,
                video_base64=final_status.get("video_base64"),
                duration_seconds=duration,
                logs="\n".join(logs)
            )
        else:
            return RunPodJobResult(
                success=False,
                job_id=job_id,
                status=JobStatus.FAILED,
                error=final_status.get("error", "Unknown error"),
                duration_seconds=duration,
                logs="\n".join(logs)
            )


# =============================================================================
# SERVERLESS CLIENT
# =============================================================================

class RunPodServerlessClient:
    """
    Client for RunPod Serverless Endpoints.
    
    This client uses RunPod's managed API for serverless inference.
    Workers spin up on-demand and shut down when idle.
    """
    
    RUNPOD_API_BASE = "https://api.runpod.ai/v2"
    VALID_TRAJECTORIES = RunPodGEN3CClient.VALID_TRAJECTORIES
    VALID_FRAME_COUNTS = RunPodGEN3CClient.VALID_FRAME_COUNTS
    
    def __init__(self, endpoint_id: str, api_key: str, timeout: int = 30):
        """
        Initialize the RunPod Serverless client.
        
        Args:
            endpoint_id: Your serverless endpoint ID (from RunPod console)
            api_key: Your RunPod API key (starts with 'rp_')
            timeout: Request timeout in seconds
        """
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = f"{self.RUNPOD_API_BASE}/{endpoint_id}"
    
    def _headers(self) -> Dict[str, str]:
        """Get headers for RunPod API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check endpoint health/status.
        
        Note: Serverless endpoints don't have a traditional health check.
        This returns endpoint info from RunPod's API.
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self._headers(),
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                # RunPod returns workers info
                return {
                    "status": "healthy",
                    "workers": data.get("workers", {}),
                    "jobs": data.get("jobs", {}),
                    "endpoint_id": self.endpoint_id
                }
            else:
                return {
                    "status": "error",
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def submit_job(
        self,
        image_path: str,
        video_name: str = "gen3c_output",
        num_frames: int = 121,
        trajectory: str = "left",
        guidance: float = 1.0,
        foreground_masking: bool = True,
        seed: Optional[int] = None,
        movement_distance: float = 0.3,
        camera_rotation: str = "center_facing"
    ) -> Dict[str, Any]:
        """
        Submit a job to the serverless endpoint.
        
        Args:
            image_path: Path to the input image
            video_name: Name for the output video
            num_frames: Number of frames (121, 241, 361, 481)
            trajectory: Camera trajectory
            guidance: Guidance scale (0.5-3.0)
            foreground_masking: Enable foreground masking
            seed: Random seed for reproducibility
            movement_distance: How far the camera moves (0.1-1.0, default 0.3)
            camera_rotation: How camera rotates (center_facing, no_rotation, trajectory_aligned)
            
        Returns:
            Dict with id (job_id) and status
        """
        # Validate inputs
        if num_frames not in self.VALID_FRAME_COUNTS:
            raise ValueError(f"num_frames must be one of {self.VALID_FRAME_COUNTS}")
        if trajectory not in self.VALID_TRAJECTORIES:
            raise ValueError(f"trajectory must be one of {self.VALID_TRAJECTORIES}")
        
        # Read and encode image
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()
        
        # Build RunPod serverless request format
        payload = {
            "input": {
                "image_base64": image_base64,
                "video_name": video_name,
                "num_frames": num_frames,
                "trajectory": trajectory,
                "guidance": guidance,
                "foreground_masking": foreground_masking,
                "movement_distance": movement_distance,
                "camera_rotation": camera_rotation,
                "return_base64": True
            }
        }
        if seed is not None:
            payload["input"]["seed"] = seed
        
        response = requests.post(
            f"{self.base_url}/run",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        
        # RunPod returns {"id": "...", "status": "IN_QUEUE"}
        return {
            "job_id": result.get("id", ""),
            "status": result.get("status", "unknown").lower()
        }
    
    def get_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a serverless job.
        
        RunPod status values: IN_QUEUE, IN_PROGRESS, COMPLETED, FAILED, CANCELLED
        """
        response = requests.get(
            f"{self.base_url}/status/{job_id}",
            headers=self._headers(),
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        
        # Normalize status to match our JobStatus enum
        runpod_status = result.get("status", "unknown").upper()
        status_map = {
            "IN_QUEUE": "pending",
            "IN_PROGRESS": "running",
            "COMPLETED": "completed",
            "FAILED": "failed",
            "CANCELLED": "failed"
        }
        
        normalized = {
            "job_id": job_id,
            "status": status_map.get(runpod_status, "unknown"),
            "runpod_status": runpod_status
        }
        
        # Extract logs if available (RunPod streams these during execution)
        if "logs" in result:
            normalized["logs"] = result["logs"]
        
        # Extract output if completed
        if runpod_status == "COMPLETED" and "output" in result:
            output = result["output"]
            if isinstance(output, dict):
                normalized["video_base64"] = output.get("video_base64")
                normalized["video_path"] = output.get("video_path")
                normalized["message"] = output.get("message")
                if output.get("status") == "error":
                    normalized["status"] = "failed"
                    normalized["error"] = output.get("message", "Unknown error")
        
        # Extract error if failed
        if runpod_status == "FAILED":
            normalized["error"] = result.get("error", "Unknown error")
        
        return normalized
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a running serverless job.
        
        Args:
            job_id: The job ID to cancel
            
        Returns:
            Dict with cancellation status
        """
        try:
            response = requests.post(
                f"{self.base_url}/cancel/{job_id}",
                headers=self._headers(),
                timeout=self.timeout
            )
            response.raise_for_status()
            return {"success": True, "message": f"Job {job_id} cancelled"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 30,
        max_wait: int = 3600,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Wait for a serverless job to complete.
        
        Args:
            job_id: The job ID to wait for
            poll_interval: Seconds between status checks
            max_wait: Maximum seconds to wait
            progress_callback: Optional callback(status, elapsed_seconds)
            
        Returns:
            Final job status dict
        """
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > max_wait:
                return {
                    "job_id": job_id,
                    "status": "failed",
                    "error": f"Timeout after {max_wait} seconds"
                }
            
            try:
                status = self.get_status(job_id)
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Status check failed: {e}", elapsed)
                time.sleep(poll_interval)
                continue
            
            if progress_callback:
                progress_callback(status.get("status", "unknown"), elapsed)
            
            if status.get("status") in ["completed", "failed"]:
                return status
            
            time.sleep(poll_interval)
    
    def generate_sync(
        self,
        image_path: str,
        output_dir: str,
        video_name: str = "gen3c_output",
        num_frames: int = 121,
        trajectory: str = "left",
        guidance: float = 1.0,
        foreground_masking: bool = True,
        seed: Optional[int] = None,
        movement_distance: float = 0.3,
        camera_rotation: str = "center_facing",
        poll_interval: int = 30,
        max_wait: int = 3600,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> RunPodJobResult:
        """
        Submit a serverless job and wait for completion synchronously.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save output video
            video_name: Name for output video
            num_frames: Number of frames (121, 241, 361, 481)
            trajectory: Camera trajectory
            guidance: Guidance scale
            foreground_masking: Enable foreground masking
            seed: Random seed
            movement_distance: How far the camera moves (0.1-1.0)
            camera_rotation: How camera rotates (center_facing, no_rotation, trajectory_aligned)
            poll_interval: Seconds between status checks
            max_wait: Maximum wait time
            progress_callback: Optional progress callback
            
        Returns:
            RunPodJobResult with success/failure info
        """
        start_time = time.time()
        logs = []
        
        logs.append(f"Submitting to serverless endpoint: {self.endpoint_id}")
        
        # Submit job
        try:
            submit_result = self.submit_job(
                image_path=image_path,
                video_name=video_name,
                num_frames=num_frames,
                trajectory=trajectory,
                guidance=guidance,
                foreground_masking=foreground_masking,
                seed=seed,
                movement_distance=movement_distance,
                camera_rotation=camera_rotation
            )
        except Exception as e:
            return RunPodJobResult(
                success=False,
                job_id="",
                status=JobStatus.FAILED,
                error=f"Failed to submit job: {e}",
                logs="\n".join(logs)
            )
        
        job_id = submit_result.get("job_id", "")
        logs.append(f"Job submitted: {job_id}")
        logs.append(f"Initial status: {submit_result.get('status', 'unknown')}")
        
        if progress_callback:
            progress_callback("pending", 0)
        
        # Wait for completion
        final_status = self.wait_for_completion(
            job_id=job_id,
            poll_interval=poll_interval,
            max_wait=max_wait,
            progress_callback=progress_callback
        )
        
        duration = time.time() - start_time
        
        if final_status.get("status") == "completed":
            # Save video
            output_path = None
            if final_status.get("video_base64"):
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{video_name}.mp4")
                video_data = base64.b64decode(final_status["video_base64"])
                with open(output_path, "wb") as f:
                    f.write(video_data)
                logs.append(f"Video saved: {output_path}")
            
            return RunPodJobResult(
                success=True,
                job_id=job_id,
                status=JobStatus.COMPLETED,
                output_path=output_path,
                video_base64=final_status.get("video_base64"),
                duration_seconds=duration,
                logs="\n".join(logs)
            )
        else:
            logs.append(f"Job failed: {final_status.get('error', 'Unknown error')}")
            return RunPodJobResult(
                success=False,
                job_id=job_id,
                status=JobStatus.FAILED,
                error=final_status.get("error", "Unknown error"),
                duration_seconds=duration,
                logs="\n".join(logs)
            )


# =============================================================================
# DEFAULT CLIENTS
# =============================================================================

# Default client instances (can be configured)
_default_client: Optional[RunPodGEN3CClient] = None
_default_serverless_client: Optional[RunPodServerlessClient] = None


def get_runpod_client(api_url: Optional[str] = None) -> Optional[RunPodGEN3CClient]:
    """Get or create the default RunPod Pod client."""
    global _default_client
    
    if api_url:
        _default_client = RunPodGEN3CClient(api_url)
    
    return _default_client


def set_runpod_url(api_url: str) -> RunPodGEN3CClient:
    """Set the RunPod Pod API URL and return the client."""
    global _default_client
    _default_client = RunPodGEN3CClient(api_url)
    return _default_client


def get_serverless_client(
    endpoint_id: Optional[str] = None, 
    api_key: Optional[str] = None
) -> Optional[RunPodServerlessClient]:
    """Get or create the default RunPod Serverless client."""
    global _default_serverless_client
    
    if endpoint_id and api_key:
        _default_serverless_client = RunPodServerlessClient(endpoint_id, api_key)
    
    return _default_serverless_client


def set_serverless_config(endpoint_id: str, api_key: str) -> RunPodServerlessClient:
    """Set the RunPod Serverless config and return the client."""
    global _default_serverless_client
    _default_serverless_client = RunPodServerlessClient(endpoint_id, api_key)
    return _default_serverless_client


# =============================================================================
# UNIFIED SERVERLESS CLIENT (Multi-Model Support)
# =============================================================================

class UnifiedServerlessClient:
    """
    Unified client for RunPod Serverless supporting multiple models.
    
    Supports:
    - GEN3C: Image to video generation
    - SHARP: Image to 3D Gaussian Splatting
    """
    
    RUNPOD_API_BASE = "https://api.runpod.ai/v2"
    
    VALID_TRAJECTORIES = [
        "left", "right", "up", "down", 
        "zoom_in", "zoom_out", 
        "clockwise", "counterclockwise", 
        "none"
    ]
    VALID_FRAME_COUNTS = [121, 241, 361, 481]
    
    def __init__(self, endpoint_id: str, api_key: str, timeout: int = 30):
        """
        Initialize the unified serverless client.
        
        Args:
            endpoint_id: Your serverless endpoint ID
            api_key: Your RunPod API key
            timeout: Request timeout in seconds
        """
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = f"{self.RUNPOD_API_BASE}/{endpoint_id}"
    
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check endpoint health."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self._headers(),
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "workers": data.get("workers", {}),
                    "jobs": data.get("jobs", {}),
                    "endpoint_id": self.endpoint_id
                }
            else:
                return {"status": "error", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def submit_gen3c_job(
        self,
        image_path: str,
        video_name: str = "gen3c_output",
        num_frames: int = 121,
        trajectory: str = "left",
        guidance: float = 1.0,
        foreground_masking: bool = True,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Submit a GEN3C job."""
        if num_frames not in self.VALID_FRAME_COUNTS:
            raise ValueError(f"num_frames must be one of {self.VALID_FRAME_COUNTS}")
        if trajectory not in self.VALID_TRAJECTORIES:
            raise ValueError(f"trajectory must be one of {self.VALID_TRAJECTORIES}")
        
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()
        
        payload = {
            "input": {
                "model": "gen3c",
                "image_base64": image_base64,
                "output_name": video_name,
                "num_frames": num_frames,
                "trajectory": trajectory,
                "guidance": guidance,
                "foreground_masking": foreground_masking,
                "return_base64": True
            }
        }
        if seed is not None:
            payload["input"]["seed"] = seed
        
        response = requests.post(
            f"{self.base_url}/run",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        
        return {
            "job_id": result.get("id", ""),
            "status": result.get("status", "unknown").lower(),
            "model": "gen3c"
        }
    
    def submit_sharp_job(
        self,
        image_path: str,
        output_name: str = "sharp_output",
        render_video: bool = False,
        trajectory_type: str = "rotate_forward",
        num_steps: int = 60,
        num_repeats: int = 1,
        max_disparity: float = 0.08,
        max_zoom: float = 0.15,
        lookat_mode: str = "point",
    ) -> Dict[str, Any]:
        """Submit a SHARP job with optional video rendering and trajectory parameters."""
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()
        
        payload = {
            "input": {
                "model": "sharp",
                "image_base64": image_base64,
                "output_name": output_name,
                "render_video": render_video,
                "trajectory_type": trajectory_type,
                "num_steps": num_steps,
                "num_repeats": num_repeats,
                "max_disparity": max_disparity,
                "max_zoom": max_zoom,
                "lookat_mode": lookat_mode,
                "return_base64": True
            }
        }
        
        response = requests.post(
            f"{self.base_url}/run",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        
        return {
            "job_id": result.get("id", ""),
            "status": result.get("status", "unknown").lower(),
            "model": "sharp"
        }
    
    def submit_generic_job(self, job_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a generic job with custom input payload.
        
        Args:
            job_input: Dictionary containing job parameters (must include 'model' key)
        
        Returns:
            Dictionary with job_id and status
        """
        model = job_input.get("model", "unknown")
        
        response = requests.post(
            f"{self.base_url}/run",
            headers=self._headers(),
            json={"input": job_input},
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        
        return {
            "job_id": result.get("id", ""),
            "status": result.get("status", "unknown").lower(),
            "model": model
        }
    
    def generate_lyra_sync(
        self,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        output_dir: str = "/tmp",
        output_name: str = "lyra_output",
        generation_mode: str = "static",
        num_views: int = 8,
        camera_motion_scale: float = 1.0,
        multi_trajectory: bool = True,
        foreground_masking: bool = True,
        max_gaussians: int = 100000,
        seed: Optional[int] = None,
        poll_interval: int = 30,
        max_wait: int = 7200,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> RunPodJobResult:
        """
        Submit Lyra job and wait for completion.
        
        Args:
            image_path: Path to input image (for static mode)
            video_path: Path to input video (for dynamic mode)
            output_dir: Directory to save output
            output_name: Base name for output files
            generation_mode: "static" or "dynamic"
            num_views: Number of multi-view positions
            camera_motion_scale: Camera motion scale
            multi_trajectory: Enable multi-trajectory
            foreground_masking: Enable foreground masking
            max_gaussians: Maximum Gaussian splats
            seed: Random seed
            poll_interval: Seconds between status checks
            max_wait: Maximum wait time
            progress_callback: Optional progress callback
            
        Returns:
            RunPodJobResult with PLY/video output
        """
        start_time = time.time()
        logs = []
        
        # Determine input type and path
        is_static = generation_mode == "static"
        if is_static:
            if not image_path:
                return RunPodJobResult(
                    success=False, job_id="", status=JobStatus.FAILED,
                    model="lyra", error="No image path provided for static mode"
                )
            input_path = image_path
            input_key = "image_base64"
        else:
            if not video_path:
                return RunPodJobResult(
                    success=False, job_id="", status=JobStatus.FAILED,
                    model="lyra", error="No video path provided for dynamic mode"
                )
            input_path = video_path
            input_key = "video_base64"
        
        logs.append(f"Submitting Lyra job to endpoint: {self.endpoint_id}")
        logs.append(f"Mode: {generation_mode}, Views: {num_views}")
        
        try:
            # Encode input
            with open(input_path, "rb") as f:
                input_base64 = base64.b64encode(f.read()).decode()
            
            # Build job payload
            job_input = {
                "model": "lyra",
                input_key: input_base64,
                "output_name": output_name,
                "generation_mode": generation_mode,
                "num_views": num_views,
                "camera_motion_scale": camera_motion_scale,
                "multi_trajectory": multi_trajectory,
                "foreground_masking": foreground_masking,
                "max_gaussians": max_gaussians,
                "return_base64": False,  # Use S3 for large files
            }
            if seed is not None:
                job_input["seed"] = seed
            
            submit_result = self.submit_generic_job(job_input)
        except Exception as e:
            return RunPodJobResult(
                success=False, job_id="", status=JobStatus.FAILED,
                model="lyra", error=f"Failed to submit job: {e}",
                logs="\n".join(logs)
            )
        
        job_id = submit_result.get("job_id", "")
        logs.append(f"Job submitted: {job_id}")
        
        if progress_callback:
            progress_callback("pending", 0)
        
        # Wait for completion
        final_status = self.wait_for_completion(
            job_id=job_id,
            poll_interval=poll_interval,
            max_wait=max_wait,
            progress_callback=progress_callback
        )
        
        duration = time.time() - start_time
        
        if final_status.get("status") == "completed":
            os.makedirs(output_dir, exist_ok=True)
            output_path = None
            download_required = final_status.get("download_required", False)
            
            # Handle PLY output
            if final_status.get("ply_base64"):
                output_path = os.path.join(output_dir, f"{output_name}.ply")
                ply_data = base64.b64decode(final_status["ply_base64"])
                with open(output_path, "wb") as f:
                    f.write(ply_data)
                logs.append(f"PLY saved: {output_path}")
            elif final_status.get("ply_s3_url"):
                s3_url = final_status["ply_s3_url"]
                logs.append(f"PLY available at S3: {s3_url}")
                logs.append("Downloading from S3...")
                local_path, s3_msg = download_from_s3(s3_url, output_dir)
                logs.append(s3_msg)
                if local_path:
                    output_path = local_path
                    download_required = False
            elif final_status.get("ply_path"):
                remote_path = final_status["ply_path"]
                logs.append(f"PLY on RunPod volume: {remote_path}")
                download_required = True
            
            # Handle video output
            video_path_out = None
            if final_status.get("video_base64"):
                video_path_out = os.path.join(output_dir, f"{output_name}.mp4")
                video_data = base64.b64decode(final_status["video_base64"])
                with open(video_path_out, "wb") as f:
                    f.write(video_data)
                logs.append(f"Video saved: {video_path_out}")
            elif final_status.get("video_s3_url"):
                s3_url = final_status["video_s3_url"]
                logs.append(f"Video available at S3: {s3_url}")
                logs.append("Downloading video from S3...")
                local_path, s3_msg = download_from_s3(s3_url, output_dir)
                logs.append(s3_msg)
                if local_path:
                    video_path_out = local_path
            
            return RunPodJobResult(
                success=True,
                job_id=job_id,
                status=JobStatus.COMPLETED,
                model="lyra",
                output_path=output_path,
                duration_seconds=duration,
                logs="\n".join(logs),
                download_required=download_required,
            )
        else:
            error = final_status.get("error", "Unknown error")
            logs.append(f"Job failed: {error}")
            return RunPodJobResult(
                success=False,
                job_id=job_id,
                status=JobStatus.FAILED,
                model="lyra",
                error=error,
                duration_seconds=duration,
                logs="\n".join(logs),
            )
    
    def generate_trellis_sync(
        self,
        image_path: str,
        output_dir: str = "/tmp",
        output_name: str = "trellis_output",
        resolution: int = 1024,
        guidance_scale: float = 7.5,
        output_glb: bool = True,
        seed: Optional[int] = None,
        poll_interval: int = 30,
        max_wait: int = 1800,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> RunPodJobResult:
        """
        Submit TRELLIS.2 job and wait for completion.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save output
            output_name: Base name for output files
            resolution: Voxel resolution (512, 1024, or 1536)
            guidance_scale: CFG scale
            output_glb: Export GLB format (True) or PLY (False)
            seed: Random seed
            poll_interval: Seconds between status checks
            max_wait: Maximum wait time
            progress_callback: Optional progress callback
            
        Returns:
            RunPodJobResult with GLB/PLY output
        """
        start_time = time.time()
        logs = []
        
        logs.append(f"Submitting TRELLIS.2 job to endpoint: {self.endpoint_id}")
        logs.append(f"Resolution: {resolution}³, Guidance: {guidance_scale}")
        
        try:
            # Encode input image
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode()
            
            # Build job payload
            job_input = {
                "model": "trellis",
                "image_base64": image_base64,
                "output_name": output_name,
                "resolution": resolution,
                "guidance_scale": guidance_scale,
                "output_glb": output_glb,
                "output_ply": not output_glb,
                "return_base64": False,  # Use S3 for large files
            }
            if seed is not None:
                job_input["seed"] = seed
            
            submit_result = self.submit_generic_job(job_input)
        except Exception as e:
            return RunPodJobResult(
                success=False, job_id="", status=JobStatus.FAILED,
                model="trellis", error=f"Failed to submit job: {e}",
                logs="\n".join(logs)
            )
        
        job_id = submit_result.get("job_id", "")
        logs.append(f"Job submitted: {job_id}")
        
        if progress_callback:
            progress_callback("pending", 0)
        
        # Wait for completion
        final_status = self.wait_for_completion(
            job_id=job_id,
            poll_interval=poll_interval,
            max_wait=max_wait,
            progress_callback=progress_callback
        )
        
        duration = time.time() - start_time
        
        if final_status.get("status") == "completed":
            os.makedirs(output_dir, exist_ok=True)
            output_path = None
            download_required = final_status.get("download_required", False)
            
            # Handle GLB output
            if final_status.get("glb_base64"):
                output_path = os.path.join(output_dir, f"{output_name}.glb")
                glb_data = base64.b64decode(final_status["glb_base64"])
                with open(output_path, "wb") as f:
                    f.write(glb_data)
                logs.append(f"GLB saved: {output_path}")
            elif final_status.get("glb_s3_url"):
                s3_url = final_status["glb_s3_url"]
                logs.append(f"GLB available at S3: {s3_url}")
                logs.append("Downloading from S3...")
                local_path, s3_msg = download_from_s3(s3_url, output_dir)
                logs.append(s3_msg)
                if local_path:
                    output_path = local_path
                    download_required = False
            elif final_status.get("glb_path"):
                remote_path = final_status["glb_path"]
                logs.append(f"GLB on RunPod volume: {remote_path}")
                download_required = True
            
            # Handle PLY output
            if not output_path:
                if final_status.get("ply_base64"):
                    output_path = os.path.join(output_dir, f"{output_name}.ply")
                    ply_data = base64.b64decode(final_status["ply_base64"])
                    with open(output_path, "wb") as f:
                        f.write(ply_data)
                    logs.append(f"PLY saved: {output_path}")
                elif final_status.get("ply_s3_url"):
                    s3_url = final_status["ply_s3_url"]
                    logs.append(f"PLY available at S3: {s3_url}")
                    logs.append("Downloading from S3...")
                    local_path, s3_msg = download_from_s3(s3_url, output_dir)
                    logs.append(s3_msg)
                    if local_path:
                        output_path = local_path
                        download_required = False
                elif final_status.get("ply_path"):
                    remote_path = final_status["ply_path"]
                    logs.append(f"PLY on RunPod volume: {remote_path}")
                    download_required = True
            
            return RunPodJobResult(
                success=True,
                job_id=job_id,
                status=JobStatus.COMPLETED,
                model="trellis",
                output_path=output_path,
                duration_seconds=duration,
                logs="\n".join(logs),
                download_required=download_required,
            )
        else:
            error = final_status.get("error", "Unknown error")
            logs.append(f"Job failed: {error}")
            return RunPodJobResult(
                success=False,
                job_id=job_id,
                status=JobStatus.FAILED,
                model="trellis",
                error=error,
                duration_seconds=duration,
                logs="\n".join(logs),
            )
    
    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status."""
        response = requests.get(
            f"{self.base_url}/status/{job_id}",
            headers=self._headers(),
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        
        runpod_status = result.get("status", "unknown").upper()
        status_map = {
            "IN_QUEUE": "pending",
            "IN_PROGRESS": "running",
            "COMPLETED": "completed",
            "FAILED": "failed",
            "CANCELLED": "failed"
        }
        
        normalized = {
            "job_id": job_id,
            "status": status_map.get(runpod_status, "unknown"),
            "runpod_status": runpod_status
        }
        
        if "logs" in result:
            normalized["logs"] = result["logs"]
        
        if runpod_status == "COMPLETED" and "output" in result:
            output = result["output"]
            if isinstance(output, dict):
                normalized["model"] = output.get("model", "unknown")
                # GEN3C outputs
                if output.get("video_base64"):
                    normalized["video_base64"] = output["video_base64"]
                if output.get("video_path"):
                    normalized["video_path"] = output["video_path"]
                if output.get("video_s3_url"):
                    normalized["video_s3_url"] = output["video_s3_url"]
                # SHARP/Lyra outputs
                if output.get("ply_base64"):
                    normalized["ply_base64"] = output["ply_base64"]
                if output.get("ply_path"):
                    normalized["ply_path"] = output["ply_path"]
                if output.get("ply_s3_url"):
                    normalized["ply_s3_url"] = output["ply_s3_url"]
                if output.get("ply_size"):
                    normalized["ply_size"] = output["ply_size"]
                # TRELLIS outputs
                if output.get("glb_base64"):
                    normalized["glb_base64"] = output["glb_base64"]
                if output.get("glb_path"):
                    normalized["glb_path"] = output["glb_path"]
                if output.get("glb_s3_url"):
                    normalized["glb_s3_url"] = output["glb_s3_url"]
                # SuGaR/Mesh extraction outputs
                if output.get("mesh_base64"):
                    normalized["mesh_base64"] = output["mesh_base64"]
                if output.get("mesh_path"):
                    normalized["mesh_path"] = output["mesh_path"]
                if output.get("mesh_s3_url"):
                    normalized["mesh_s3_url"] = output["mesh_s3_url"]
                if output.get("mesh_size"):
                    normalized["mesh_size"] = output["mesh_size"]
                # Download required flag
                if output.get("download_required"):
                    normalized["download_required"] = output["download_required"]
                # Status
                if output.get("status") == "error":
                    normalized["status"] = "failed"
                    normalized["error"] = output.get("message", "Unknown error")
        
        if runpod_status == "FAILED":
            normalized["error"] = result.get("error", "Unknown error")
        
        return normalized
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a job."""
        try:
            response = requests.post(
                f"{self.base_url}/cancel/{job_id}",
                headers=self._headers(),
                timeout=self.timeout
            )
            response.raise_for_status()
            return {"success": True, "message": f"Job {job_id} cancelled"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 30,
        max_wait: int = 3600,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """Wait for job completion."""
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > max_wait:
                return {"job_id": job_id, "status": "failed", "error": f"Timeout after {max_wait}s"}
            
            try:
                status = self.get_status(job_id)
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Status check failed: {e}", elapsed)
                time.sleep(poll_interval)
                continue
            
            if progress_callback:
                progress_callback(status.get("status", "unknown"), elapsed)
            
            if status.get("status") in ["completed", "failed"]:
                return status
            
            time.sleep(poll_interval)
    
    def generate_sharp_sync(
        self,
        image_path: str,
        output_dir: str,
        output_name: str = "sharp_output",
        render_video: bool = False,
        trajectory_type: str = "rotate_forward",
        num_steps: int = 60,
        num_repeats: int = 1,
        max_disparity: float = 0.08,
        max_zoom: float = 0.15,
        lookat_mode: str = "point",
        poll_interval: int = 10,
        max_wait: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> RunPodJobResult:
        """
        Submit SHARP job and wait for completion.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save output
            output_name: Base name for output files
            render_video: Whether to render video trajectory
            trajectory_type: Camera trajectory type (rotate_forward, rotate, swipe, shake)
            num_steps: Number of frames in video
            num_repeats: Number of trajectory loops
            max_disparity: Maximum lateral camera offset
            max_zoom: Maximum forward camera movement
            lookat_mode: Camera focus mode (point, ahead)
            poll_interval: Seconds between status checks
            max_wait: Maximum wait time (default: 600s for PLY only, 1800s with video)
            progress_callback: Optional progress callback
            
        Returns:
            RunPodJobResult with PLY (and optionally video) output
        """
        # Use longer timeout when video rendering is requested
        # Video rendering can take 15+ minutes on cold start (gsplat kernel compilation)
        if max_wait is None:
            max_wait = 1800 if render_video else 600  # 30 min with video, 10 min without
        
        start_time = time.time()
        logs = []
        
        logs.append(f"Submitting SHARP job to endpoint: {self.endpoint_id}")
        
        try:
            submit_result = self.submit_sharp_job(
                image_path=image_path,
                output_name=output_name,
                render_video=render_video,
                trajectory_type=trajectory_type,
                num_steps=num_steps,
                num_repeats=num_repeats,
                max_disparity=max_disparity,
                max_zoom=max_zoom,
                lookat_mode=lookat_mode,
            )
        except Exception as e:
            return RunPodJobResult(
                success=False,
                job_id="",
                status=JobStatus.FAILED,
                model="sharp",
                error=f"Failed to submit job: {e}",
                logs="\n".join(logs)
            )
        
        job_id = submit_result.get("job_id", "")
        logs.append(f"Job submitted: {job_id}")
        
        if progress_callback:
            progress_callback("pending", 0)
        
        final_status = self.wait_for_completion(
            job_id=job_id,
            poll_interval=poll_interval,
            max_wait=max_wait,
            progress_callback=progress_callback
        )
        
        duration = time.time() - start_time
        
        if final_status.get("status") == "completed":
            os.makedirs(output_dir, exist_ok=True)
            output_path = None
            download_required = final_status.get("download_required", False)
            
            # Save PLY (if base64 available)
            if final_status.get("ply_base64"):
                output_path = os.path.join(output_dir, f"{output_name}.ply")
                ply_data = base64.b64decode(final_status["ply_base64"])
                with open(output_path, "wb") as f:
                    f.write(ply_data)
                logs.append(f"PLY saved: {output_path}")
            elif final_status.get("ply_s3_url"):
                # Download from S3
                s3_url = final_status["ply_s3_url"]
                logs.append(f"PLY available at S3: {s3_url}")
                logs.append("Downloading from S3...")
                local_path, s3_msg = download_from_s3(s3_url, output_dir)
                logs.append(s3_msg)
                if local_path:
                    output_path = local_path
                    download_required = False
                else:
                    logs.append(f"Manual download: {s3_url}")
            elif final_status.get("ply_path"):
                # File too large - try SSH download
                remote_path = final_status["ply_path"]
                ply_size = final_status.get("ply_size", 0)
                logs.append(f"PLY on RunPod volume: {remote_path} ({ply_size / 1024 / 1024:.1f}MB)")
                
                if _ssh_available:
                    ssh_config = load_ssh_config()
                    if ssh_config.is_configured():
                        logs.append("Attempting SSH download...")
                        local_path, ssh_msg = download_from_runpod(remote_path, output_dir)
                        logs.append(ssh_msg)
                        if local_path:
                            output_path = local_path
                            download_required = False  # Successfully downloaded
                    else:
                        logs.append("⚠️ SSH not configured. Run: python -m runpod.ssh_download configure --host <IP> --key <path>")
                else:
                    logs.append("⚠️ SSH module not available. Manual download required.")
            
            # Save video if rendered (if base64 available)
            video_path = None
            if final_status.get("video_base64"):
                video_path = os.path.join(output_dir, f"{output_name}.mp4")
                video_data = base64.b64decode(final_status["video_base64"])
                with open(video_path, "wb") as f:
                    f.write(video_data)
                logs.append(f"Video saved: {video_path}")
            elif final_status.get("video_s3_url"):
                # Download from S3
                s3_url = final_status["video_s3_url"]
                logs.append(f"Video available at S3: {s3_url}")
                logs.append("Downloading video from S3...")
                local_path, s3_msg = download_from_s3(s3_url, output_dir)
                logs.append(s3_msg)
                if local_path:
                    video_path = local_path
            elif final_status.get("video_path"):
                # File too large - try SSH download
                remote_path = final_status["video_path"]
                video_size = final_status.get("video_size", 0)
                logs.append(f"Video on RunPod volume: {remote_path} ({video_size / 1024 / 1024:.1f}MB)")
                
                if _ssh_available:
                    ssh_config = load_ssh_config()
                    if ssh_config.is_configured():
                        logs.append("Attempting SSH download for video...")
                        local_path, ssh_msg = download_from_runpod(remote_path, output_dir)
                        logs.append(ssh_msg)
                        if local_path:
                            video_path = local_path
                else:
                    logs.append("⚠️ SSH module not available for video download.")
            
            # Build result message
            if download_required:
                message = final_status.get("message", "Files generated. Download from RunPod volume required.")
                logs.append(f"⚠️ {message}")
            
            return RunPodJobResult(
                success=True,
                job_id=job_id,
                status=JobStatus.COMPLETED,
                model="sharp",
                output_path=output_path,
                ply_base64=final_status.get("ply_base64"),
                video_base64=final_status.get("video_base64"),
                duration_seconds=duration,
                logs="\n".join(logs),
                download_required=download_required,
                remote_ply_path=final_status.get("ply_path"),
                remote_video_path=final_status.get("video_path"),
            )
        else:
            logs.append(f"Job failed: {final_status.get('error', 'Unknown error')}")
            return RunPodJobResult(
                success=False,
                job_id=job_id,
                status=JobStatus.FAILED,
                model="sharp",
                error=final_status.get("error", "Unknown error"),
                duration_seconds=duration,
                logs="\n".join(logs)
            )


# =============================================================================
# 2DGS PIPELINE CLIENT
# =============================================================================

class TwoDGSPipelineClient:
    """
    Client for the 2DGS Pipeline serverless endpoint.
    
    Converts Gen3C videos to 3D meshes using ViPE + 2DGS.
    Endpoint ID: s9txp6edtf2vg4
    """
    
    RUNPOD_API_BASE = "https://api.runpod.ai/v2"
    DEFAULT_ENDPOINT_ID = "s9txp6edtf2vg4"
    
    def __init__(
        self, 
        endpoint_id: str = DEFAULT_ENDPOINT_ID, 
        api_key: str = "",
        timeout: int = 30
    ):
        """
        Initialize the 2DGS Pipeline client.
        
        Args:
            endpoint_id: RunPod serverless endpoint ID (default: s9txp6edtf2vg4)
            api_key: Your RunPod API key
            timeout: Request timeout in seconds
        """
        self.endpoint_id = endpoint_id or self.DEFAULT_ENDPOINT_ID
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = f"{self.RUNPOD_API_BASE}/{self.endpoint_id}"
    
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check endpoint health."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self._headers(),
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "workers": data.get("workers", {}),
                    "jobs": data.get("jobs", {}),
                    "endpoint_id": self.endpoint_id
                }
            else:
                return {"status": "error", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def submit_job(
        self,
        video_url: str,
        iterations: int = 5000,
        mesh_quality: str = "high",
        output_format: str = "glb",
        s3_bucket: str = "arkrunr",
        s3_region: str = "us-west-1",
        s3_prefix: str = "MediaContent/2dgs-pipeline/outputs/"
    ) -> Dict[str, Any]:
        """
        Submit a 2DGS pipeline job.
        
        Args:
            video_url: URL to the video (S3 presigned, HTTP, etc.)
            iterations: 2DGS training iterations
            mesh_quality: Mesh quality preset (fast, balanced, high, ultra)
            output_format: Output format (glb, obj, ply)
            s3_bucket: S3 bucket for output
            s3_region: S3 region
            s3_prefix: S3 key prefix for output
            
        Returns:
            Dict with job_id and status
        """
        payload = {
            "input": {
                "video_url": video_url,
                "iterations": iterations,
                "mesh_quality": mesh_quality,
                "output_format": output_format,
                "output_s3": {
                    "bucket": s3_bucket,
                    "region": s3_region,
                    "prefix": s3_prefix
                }
            }
        }
        
        response = requests.post(
            f"{self.base_url}/run",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        
        return {
            "job_id": result.get("id", ""),
            "status": result.get("status", "unknown").lower(),
            "model": "2dgs"
        }
    
    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status."""
        response = requests.get(
            f"{self.base_url}/status/{job_id}",
            headers=self._headers(),
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        
        runpod_status = result.get("status", "unknown").upper()
        status_map = {
            "IN_QUEUE": "pending",
            "IN_PROGRESS": "running",
            "COMPLETED": "completed",
            "FAILED": "failed",
            "CANCELLED": "failed"
        }
        
        normalized = {
            "job_id": job_id,
            "status": status_map.get(runpod_status, "unknown"),
            "runpod_status": runpod_status
        }
        
        if "logs" in result:
            normalized["logs"] = result["logs"]
        
        if runpod_status == "COMPLETED" and "output" in result:
            output = result["output"]
            if isinstance(output, dict):
                normalized["mesh_url"] = output.get("mesh_url")
                normalized["num_frames"] = output.get("num_frames")
                normalized["iterations"] = output.get("iterations")
                normalized["elapsed_seconds"] = output.get("elapsed_seconds")
                if output.get("status") == "error":
                    normalized["status"] = "failed"
                    normalized["error"] = output.get("error", "Unknown error")
        
        if runpod_status == "FAILED":
            normalized["error"] = result.get("error", "Unknown error")
        
        return normalized
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a job."""
        try:
            response = requests.post(
                f"{self.base_url}/cancel/{job_id}",
                headers=self._headers(),
                timeout=self.timeout
            )
            response.raise_for_status()
            return {"success": True, "message": f"Job {job_id} cancelled"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _rotate_mesh_180_x(self, mesh_path: str, logs: list) -> str:
        """
        Rotate mesh 180 degrees around X axis to correct orientation.
        
        The 2DGS pipeline outputs meshes rotated 180° on X axis compared
        to the original video frames. This corrects that orientation.
        
        Args:
            mesh_path: Path to the mesh file
            logs: List to append log messages to
            
        Returns:
            Path to the rotated mesh (same path, overwritten)
        """
        import trimesh
        import numpy as np
        
        logs.append("Applying 180° X-axis rotation to correct orientation...")
        
        # Load the mesh
        mesh = trimesh.load(mesh_path)
        
        # Create 180-degree rotation matrix around X axis
        # cos(180°) = -1, sin(180°) = 0
        # This flips Y and Z while keeping X the same
        rotation_matrix = np.array([
            [ 1,  0,  0,  0],
            [ 0, -1,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0,  0,  1]
        ])
        
        # Apply rotation
        mesh.apply_transform(rotation_matrix)
        
        # Export back to same file
        mesh.export(mesh_path)
        
        logs.append(f"✅ Mesh rotated 180° on X-axis: {mesh_path}")
        return mesh_path
    
    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 15,
        max_wait: int = 1200,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """Wait for job completion."""
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > max_wait:
                return {"job_id": job_id, "status": "failed", "error": f"Timeout after {max_wait}s"}
            
            try:
                status = self.get_status(job_id)
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Status check failed: {e}", elapsed)
                time.sleep(poll_interval)
                continue
            
            if progress_callback:
                progress_callback(status.get("status", "unknown"), elapsed)
            
            if status.get("status") in ["completed", "failed"]:
                return status
            
            time.sleep(poll_interval)
    
    def generate_sync(
        self,
        video_url: str,
        output_dir: str = "./outputs/mesh_2dgs",
        iterations: int = 5000,
        mesh_quality: str = "high",
        output_format: str = "glb",
        s3_bucket: str = "arkrunr",
        s3_region: str = "us-west-1",
        poll_interval: int = 15,
        max_wait: int = 1200,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> RunPodJobResult:
        """
        Submit 2DGS pipeline job and wait for completion.
        
        Args:
            video_url: URL to the Gen3C video
            output_dir: Directory to save output mesh
            iterations: 2DGS training iterations (1000-10000, default 5000)
            mesh_quality: Mesh quality preset (fast, balanced, high, ultra)
            output_format: Output format (glb, obj, ply)
            s3_bucket: S3 bucket for output
            s3_region: S3 region
            poll_interval: Seconds between status checks
            max_wait: Maximum wait time (default 20 minutes)
            progress_callback: Optional callback(status, elapsed)
            
        Returns:
            RunPodJobResult with mesh output
        """
        start_time = time.time()
        logs = []
        
        logs.append(f"Submitting 2DGS pipeline job to endpoint: {self.endpoint_id}")
        logs.append(f"Video: {video_url[:80]}...")
        logs.append(f"Iterations: {iterations}, Quality: {mesh_quality}")
        
        try:
            submit_result = self.submit_job(
                video_url=video_url,
                iterations=iterations,
                mesh_quality=mesh_quality,
                output_format=output_format,
                s3_bucket=s3_bucket,
                s3_region=s3_region,
            )
        except Exception as e:
            return RunPodJobResult(
                success=False,
                job_id="",
                status=JobStatus.FAILED,
                model="2dgs",
                error=f"Failed to submit job: {e}",
                logs="\n".join(logs)
            )
        
        job_id = submit_result.get("job_id", "")
        logs.append(f"Job submitted: {job_id}")
        
        if progress_callback:
            progress_callback("pending", 0)
        
        # Wait for completion
        final_status = self.wait_for_completion(
            job_id=job_id,
            poll_interval=poll_interval,
            max_wait=max_wait,
            progress_callback=progress_callback
        )
        
        duration = time.time() - start_time
        
        if final_status.get("status") == "completed":
            os.makedirs(output_dir, exist_ok=True)
            output_path = None
            
            mesh_url = final_status.get("mesh_url")
            if mesh_url:
                logs.append(f"Mesh available at: {mesh_url[:80]}...")
                logs.append("Downloading mesh...")
                
                # Download from presigned URL
                try:
                    response = requests.get(mesh_url, timeout=300)
                    response.raise_for_status()
                    
                    # Generate filename
                    filename = f"2dgs_mesh_{job_id[:8]}.{output_format}"
                    output_path = os.path.join(output_dir, filename)
                    
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    
                    file_size = os.path.getsize(output_path)
                    logs.append(f"✅ Downloaded: {output_path} ({file_size / 1024 / 1024:.2f} MB)")
                    
                    # Apply 180° X-axis rotation to correct orientation from 2DGS pipeline
                    try:
                        output_path = self._rotate_mesh_180_x(output_path, logs)
                    except Exception as e:
                        logs.append(f"⚠️ Rotation skipped: {e}")
                except Exception as e:
                    logs.append(f"❌ Download failed: {e}")
                    logs.append(f"Manual download: {mesh_url}")
            
            # Build stats
            stats = {
                "num_frames": final_status.get("num_frames"),
                "iterations": final_status.get("iterations"),
                "elapsed_seconds": final_status.get("elapsed_seconds"),
                "mesh_url": mesh_url,
            }
            
            return RunPodJobResult(
                success=True,
                job_id=job_id,
                status=JobStatus.COMPLETED,
                model="2dgs",
                output_path=output_path,
                duration_seconds=duration,
                logs="\n".join(logs),
            )
        else:
            error = final_status.get("error", "Unknown error")
            logs.append(f"Job failed: {error}")
            return RunPodJobResult(
                success=False,
                job_id=job_id,
                status=JobStatus.FAILED,
                model="2dgs",
                error=error,
                duration_seconds=duration,
                logs="\n".join(logs),
            )


# =============================================================================
# SEVA (Stable Virtual Camera) CLIENT
# =============================================================================

class SEVAServerlessClient:
    """
    Client for the SEVA (Stable Virtual Camera) serverless endpoint.
    
    Generates novel view videos from single images with precise camera control.
    
    Supported trajectories:
        - orbit: 360° rotation around subject
        - pan: Horizontal camera movement  
        - tilt: Vertical camera angle change
        - spiral: Spiral path around subject
        - zoom-out: Camera moves backward
        - dolly-zoom-out: Vertigo/Hitchcock effect
        - arc: Curved path
        - crane: Vertical + horizontal movement
        - left, right, up, down: Simple directional movements
        - custom: User-defined camera poses (C2W matrices)
    """
    
    RUNPOD_API_BASE = "https://api.runpod.ai/v2"
    DEFAULT_ENDPOINT_ID = ""  # Set when endpoint is created
    
    VALID_TRAJECTORIES = [
        "orbit", "pan", "tilt", "spiral", 
        "zoom-out", "dolly-zoom-out", "arc", "crane",
        "left", "right", "up", "down",
        "custom"
    ]
    
    def __init__(
        self, 
        endpoint_id: str = "", 
        api_key: str = "",
        timeout: int = 30
    ):
        """
        Initialize the SEVA client.
        
        Args:
            endpoint_id: RunPod serverless endpoint ID
            api_key: Your RunPod API key
            timeout: Request timeout in seconds
        """
        self.endpoint_id = endpoint_id or self.DEFAULT_ENDPOINT_ID
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = f"{self.RUNPOD_API_BASE}/{self.endpoint_id}"
    
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check endpoint health."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self._headers(),
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "workers": data.get("workers", {}),
                    "jobs": data.get("jobs", {}),
                    "endpoint_id": self.endpoint_id
                }
            else:
                return {"status": "error", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def submit_job(
        self,
        image_path: str,
        trajectory: str = "orbit",
        duration: float = 5.0,
        fps: int = 24,
        num_frames: Optional[int] = None,
        custom_poses: Optional[List] = None,
        seed: Optional[int] = None,
        output_name: str = "seva_output",
        s3_bucket: str = "arkrunr",
        s3_region: str = "us-west-1"
    ) -> Dict[str, Any]:
        """
        Submit a SEVA job.
        
        Args:
            image_path: Path to input image (will be uploaded to S3)
            trajectory: Camera trajectory type
            duration: Video duration in seconds (1-30)
            fps: Frames per second (12-60)
            num_frames: Override frame count (ignores duration if set)
            custom_poses: List of 4x4 C2W matrices for custom trajectory
            seed: Random seed for reproducibility
            output_name: Name for output file (without extension)
            s3_bucket: S3 bucket for input/output
            s3_region: S3 region
            
        Returns:
            Dict with job_id and status
        """
        # Validate trajectory
        if trajectory not in self.VALID_TRAJECTORIES:
            return {
                "status": "error", 
                "error": f"Invalid trajectory '{trajectory}'. Valid: {self.VALID_TRAJECTORIES}"
            }
        
        # Validate parameters
        duration = max(1.0, min(30.0, duration))
        fps = max(12, min(60, fps))
        
        # Upload image to S3
        image_url = self._upload_to_s3(image_path, s3_bucket, s3_region)
        if not image_url:
            return {"status": "error", "error": "Failed to upload image to S3"}
        
        # Build payload
        payload = {
            "input": {
                "image_url": image_url,
                "trajectory": trajectory,
                "duration": duration,
                "fps": fps,
                "output_name": output_name,
                "return_base64": False  # We'll download from S3
            }
        }
        
        if num_frames is not None:
            payload["input"]["num_frames"] = num_frames
        if custom_poses is not None:
            payload["input"]["custom_poses"] = custom_poses
        if seed is not None:
            payload["input"]["seed"] = seed
        
        # Submit job
        response = requests.post(
            f"{self.base_url}/run",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        
        return {
            "job_id": result.get("id", ""),
            "status": result.get("status", "unknown").lower(),
            "model": "seva"
        }
    
    def _upload_to_s3(
        self, 
        file_path: str, 
        bucket: str, 
        region: str
    ) -> Optional[str]:
        """Upload file to S3 and return URL."""
        try:
            import boto3
            from pathlib import Path
            import os
            from dotenv import load_dotenv
            
            load_dotenv()
            
            s3_client = boto3.client(
                's3',
                region_name=region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
            
            # Generate S3 key
            filename = Path(file_path).name
            timestamp = int(time.time())
            s3_key = f"MediaContent/inputs/seva/{timestamp}_{filename}"
            
            # Upload
            s3_client.upload_file(file_path, bucket, s3_key)
            
            # Return URL
            url = f"https://{bucket}.s3.{region}.amazonaws.com/{s3_key}"
            return url
            
        except Exception as e:
            print(f"S3 upload error: {e}")
            return None
    
    def _download_from_s3(
        self,
        s3_url: str,
        local_path: str,
        region: str = "us-west-1"
    ) -> bool:
        """Download file from S3 URL."""
        try:
            import boto3
            from urllib.parse import urlparse
            import os
            from dotenv import load_dotenv
            
            load_dotenv()
            
            # Parse URL
            parsed = urlparse(s3_url)
            if ".s3." in parsed.netloc:
                bucket = parsed.netloc.split(".s3.")[0]
            else:
                bucket = "arkrunr"
            key = parsed.path.lstrip("/")
            
            s3_client = boto3.client(
                's3',
                region_name=region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download
            s3_client.download_file(bucket, key, local_path)
            return True
            
        except Exception as e:
            print(f"S3 download error: {e}")
            return False
    
    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status."""
        response = requests.get(
            f"{self.base_url}/status/{job_id}",
            headers=self._headers(),
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        
        runpod_status = result.get("status", "unknown").upper()
        status_map = {
            "IN_QUEUE": "pending",
            "IN_PROGRESS": "running",
            "COMPLETED": "completed",
            "FAILED": "failed",
            "CANCELLED": "failed"
        }
        
        normalized = {
            "job_id": job_id,
            "status": status_map.get(runpod_status, "unknown"),
            "runpod_status": runpod_status
        }
        
        if "logs" in result:
            normalized["logs"] = result["logs"]
        
        if runpod_status == "COMPLETED" and "output" in result:
            output = result["output"]
            if isinstance(output, dict):
                normalized["video_url"] = output.get("video_url")
                normalized["video_path"] = output.get("video_path")
                normalized["duration"] = output.get("duration")
                normalized["fps"] = output.get("fps")
                normalized["frame_count"] = output.get("frame_count")
                normalized["trajectory"] = output.get("trajectory")
                if output.get("status") == "error":
                    normalized["status"] = "failed"
                    normalized["error"] = output.get("message")
        
        if runpod_status == "FAILED":
            normalized["error"] = result.get("error", "Unknown error")
        
        return normalized
    
    def wait_for_completion(
        self, 
        job_id: str,
        poll_interval: int = 10,
        max_wait: int = 600,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Wait for job to complete.
        
        Args:
            job_id: Job ID to poll
            poll_interval: Seconds between polls
            max_wait: Maximum wait time in seconds
            progress_callback: Optional callback for progress updates
            
        Returns:
            Final job status
        """
        start_time = time.time()
        last_status = None
        
        while time.time() - start_time < max_wait:
            status = self.get_status(job_id)
            
            if status.get("status") != last_status:
                last_status = status.get("status")
                if progress_callback:
                    elapsed = time.time() - start_time
                    progress_callback(status, elapsed)
            
            if status.get("status") in ["completed", "failed"]:
                return status
            
            time.sleep(poll_interval)
        
        return {
            "job_id": job_id,
            "status": "timeout",
            "error": f"Job did not complete within {max_wait} seconds"
        }
    
    def generate_sync(
        self,
        image_path: str,
        output_dir: str = "./outputs/seva",
        output_name: str = "seva_output",
        trajectory: str = "orbit",
        duration: float = 5.0,
        fps: int = 24,
        num_frames: Optional[int] = None,
        custom_poses: Optional[List] = None,
        seed: Optional[int] = None,
        s3_bucket: str = "arkrunr",
        s3_region: str = "us-west-1",
        poll_interval: int = 10,
        max_wait: int = 600,
        progress_callback: Optional[callable] = None
    ) -> "SEVAResult":
        """
        Generate novel view video synchronously (submit, wait, download).
        
        Args:
            image_path: Path to input image
            output_dir: Local directory for output
            output_name: Name for output file (without extension)
            trajectory: Camera trajectory type
            duration: Video duration in seconds
            fps: Frames per second
            num_frames: Override frame count
            custom_poses: List of 4x4 C2W matrices for custom trajectory
            seed: Random seed
            s3_bucket: S3 bucket
            s3_region: S3 region  
            poll_interval: Seconds between status polls
            max_wait: Maximum wait time in seconds
            progress_callback: Optional callback for progress updates
            
        Returns:
            SEVAResult with video path and metadata
        """
        start_time = time.time()
        
        # Submit job
        submit_result = self.submit_job(
            image_path=image_path,
            trajectory=trajectory,
            duration=duration,
            fps=fps,
            num_frames=num_frames,
            custom_poses=custom_poses,
            seed=seed,
            output_name=output_name,
            s3_bucket=s3_bucket,
            s3_region=s3_region
        )
        
        if submit_result.get("status") == "error":
            return SEVAResult(
                success=False,
                error=submit_result.get("error", "Unknown error"),
                duration_seconds=time.time() - start_time
            )
        
        job_id = submit_result.get("job_id")
        if not job_id:
            return SEVAResult(
                success=False,
                error="No job ID returned",
                duration_seconds=time.time() - start_time
            )
        
        # Wait for completion
        final_status = self.wait_for_completion(
            job_id=job_id,
            poll_interval=poll_interval,
            max_wait=max_wait,
            progress_callback=progress_callback
        )
        
        elapsed = time.time() - start_time
        logs = final_status.get("logs", "")
        
        if final_status.get("status") != "completed":
            return SEVAResult(
                success=False,
                error=final_status.get("error", f"Job status: {final_status.get('status')}"),
                duration_seconds=elapsed,
                logs=logs
            )
        
        # Download video from S3
        video_url = final_status.get("video_url")
        if not video_url:
            return SEVAResult(
                success=False,
                error="No video URL in result",
                duration_seconds=elapsed,
                logs=logs
            )
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        local_path = os.path.join(output_dir, f"{output_name}.mp4")
        
        # Download
        if not self._download_from_s3(video_url, local_path, s3_region):
            return SEVAResult(
                success=False,
                error="Failed to download video from S3",
                video_url=video_url,
                duration_seconds=elapsed,
                logs=logs
            )
        
        return SEVAResult(
            success=True,
            video_path=local_path,
            video_url=video_url,
            video_duration=final_status.get("duration", duration),
            fps=final_status.get("fps", fps),
            frame_count=final_status.get("frame_count"),
            trajectory=trajectory,
            duration_seconds=elapsed,
            logs=logs
        )


class SEVAResult:
    """Result from SEVA video generation."""
    
    def __init__(
        self,
        success: bool,
        video_path: Optional[str] = None,
        video_url: Optional[str] = None,
        video_duration: Optional[float] = None,
        fps: Optional[int] = None,
        frame_count: Optional[int] = None,
        trajectory: Optional[str] = None,
        error: Optional[str] = None,
        duration_seconds: float = 0,
        logs: str = ""
    ):
        self.success = success
        self.video_path = video_path
        self.video_url = video_url
        self.video_duration = video_duration
        self.fps = fps
        self.frame_count = frame_count
        self.trajectory = trajectory
        self.error = error
        self.duration_seconds = duration_seconds
        self.logs = logs
    
    def __repr__(self) -> str:
        if self.success:
            return f"SEVAResult(success=True, video_path='{self.video_path}', duration={self.video_duration}s)"
        else:
            return f"SEVAResult(success=False, error='{self.error}')"


# Default SEVA client
_seva_client: Optional[SEVAServerlessClient] = None


def get_seva_client(
    endpoint_id: Optional[str] = None,
    api_key: Optional[str] = None
) -> Optional[SEVAServerlessClient]:
    """Get or create the SEVA client."""
    global _seva_client
    
    if endpoint_id and api_key:
        _seva_client = SEVAServerlessClient(
            endpoint_id=endpoint_id,
            api_key=api_key
        )
    
    return _seva_client


# =============================================================================
# LTX-2 CLIENT (Lightricks Video Generation)
# =============================================================================

class LTX2Result:
    """Result from LTX-2 video generation."""

    def __init__(
        self,
        success: bool,
        video_path: Optional[str] = None,
        video_url: Optional[str] = None,
        num_frames: Optional[int] = None,
        duration: Optional[float] = None,
        fps: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        camera_motion: Optional[str] = None,
        seed: Optional[int] = None,
        error: Optional[str] = None,
        duration_seconds: float = 0,
        logs: str = ""
    ):
        self.success = success
        self.video_path = video_path
        self.video_url = video_url
        self.num_frames = num_frames
        self.duration = duration
        self.fps = fps
        self.width = width
        self.height = height
        self.camera_motion = camera_motion
        self.seed = seed
        self.error = error
        self.duration_seconds = duration_seconds
        self.logs = logs

    def __repr__(self) -> str:
        if self.success:
            return f"LTX2Result(success=True, video_path='{self.video_path}', duration={self.duration}s)"
        else:
            return f"LTX2Result(success=False, error='{self.error}')"


class LTX2ServerlessClient:
    """
    Client for the LTX-2 serverless endpoint.
    
    Generates high-quality videos from images using Lightricks' LTX-2 model.
    
    Camera Motion Options:
        - dolly_left: Camera moves laterally left
        - dolly_right: Camera moves laterally right
        - dolly_in: Camera pushes toward subject
        - dolly_out: Camera pulls away (best for 3D reconstruction)
        - jib_up: Camera rises vertically
        - static: No camera movement
        - none: No camera LoRA applied
    """
    
    RUNPOD_API_BASE = "https://api.runpod.ai/v2"
    DEFAULT_ENDPOINT_ID = ""  # Set when endpoint is created
    
    VALID_CAMERA_MOTIONS = [
        "dolly_left", "dolly_right",
        "dolly_in", "dolly_out",
        "jib_up", "static", "none"
    ]
    
    def __init__(
        self,
        endpoint_id: str = "",
        api_key: str = "",
        timeout: int = 30
    ):
        """
        Initialize the LTX-2 client.
        
        Args:
            endpoint_id: RunPod serverless endpoint ID
            api_key: Your RunPod API key
            timeout: Request timeout in seconds
        """
        self.endpoint_id = endpoint_id or self.DEFAULT_ENDPOINT_ID
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = f"{self.RUNPOD_API_BASE}/{self.endpoint_id}"
    
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check endpoint health."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self._headers(),
                timeout=self.timeout
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "workers": data.get("workers", {}),
                    "jobs": data.get("jobs", {}),
                    "endpoint_id": self.endpoint_id
                }
            else:
                return {"status": "error", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def submit_job(
        self,
        image_path: str,
        prompt: str = "",
        negative_prompt: str = "",
        camera_motion: str = "dolly_out",
        model_variant: str = "19b-dev-fp8",
        num_frames: int = 97,
        width: int = 768,
        height: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        fps: int = 24,
        seed: Optional[int] = None,
        output_name: str = "ltx2_output",
        s3_bucket: str = "arkrunr",
        s3_region: str = "us-west-1"
    ) -> Dict[str, Any]:
        """
        Submit an LTX-2 job.
        
        Args:
            image_path: Path to input image (will be uploaded to S3)
            prompt: Text prompt for video generation
            negative_prompt: What to avoid
            camera_motion: Camera LoRA to use
            num_frames: Number of frames (must be 8n+1)
            width: Output width (divisible by 32)
            height: Output height (divisible by 32)
            num_inference_steps: Diffusion steps
            guidance_scale: CFG scale
            fps: Frames per second
            seed: Random seed
            output_name: Name for output file
            s3_bucket: S3 bucket for input/output
            s3_region: S3 region
            
        Returns:
            Dict with job_id or error
        """
        import os
        
        # Upload input image to S3
        try:
            s3_key = f"MediaContent/inputs/ltx2/{os.path.basename(image_path)}"
            image_url = upload_file_to_s3(
                image_path, s3_bucket, s3_key, s3_region
            )
            if not image_url:
                return {"error": "Failed to upload image to S3"}
        except Exception as e:
            return {"error": f"S3 upload failed: {e}"}
        
        # Build job input
        job_input = {
            "image_url": image_url,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "camera_motion": camera_motion,
            "model_variant": model_variant,
            "num_frames": num_frames,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "fps": fps,
            "output_name": output_name
        }
        
        if seed is not None:
            job_input["seed"] = seed
        
        # Submit job
        try:
            response = requests.post(
                f"{self.base_url}/run",
                headers=self._headers(),
                json={"input": job_input},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return {"job_id": data.get("id"), "status": data.get("status")}
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a job."""
        try:
            response = requests.get(
                f"{self.base_url}/status/{job_id}",
                headers=self._headers(),
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 10,
        max_wait: int = 600,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Wait for a job to complete.
        
        Args:
            job_id: The job ID to wait for
            poll_interval: Seconds between status checks
            max_wait: Maximum time to wait in seconds
            progress_callback: Optional callback(status_dict, elapsed_seconds)
            
        Returns:
            Final job status dict
        """
        import time
        
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > max_wait:
                return {"error": f"Timeout after {max_wait}s", "status": "TIMEOUT"}
            
            status = self.get_job_status(job_id)
            
            if progress_callback:
                progress_callback(status, elapsed)
            
            job_status = status.get("status", "").upper()
            
            if job_status == "COMPLETED":
                return status
            elif job_status in ["FAILED", "CANCELLED", "ERROR"]:
                return status
            
            time.sleep(poll_interval)
    
    def generate_sync(
        self,
        image_path: str,
        output_dir: str,
        output_name: str = "ltx2_output",
        prompt: str = "",
        negative_prompt: str = "",
        camera_motion: str = "dolly_out",
        model_variant: str = "19b-dev-fp8",
        num_frames: int = 97,
        width: int = 768,
        height: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        fps: int = 24,
        seed: Optional[int] = None,
        s3_bucket: str = "arkrunr",
        s3_region: str = "us-west-1",
        poll_interval: int = 10,
        max_wait: int = 600,
        progress_callback: Optional[Callable] = None
    ) -> LTX2Result:
        """
        Generate video synchronously (submit and wait).
        
        Args:
            image_path: Path to input image
            output_dir: Local directory for output video
            output_name: Name for output file
            prompt: Text prompt
            negative_prompt: Negative prompt
            camera_motion: Camera LoRA to use
            num_frames: Number of frames
            width: Output width
            height: Output height
            num_inference_steps: Diffusion steps
            guidance_scale: CFG scale
            fps: Frames per second
            seed: Random seed
            s3_bucket: S3 bucket
            s3_region: S3 region
            poll_interval: Poll interval in seconds
            max_wait: Max wait time
            progress_callback: Progress callback
            
        Returns:
            LTX2Result with video path and metadata
        """
        import os
        import time
        
        start_time = time.time()
        logs = []
        
        # Validate camera motion
        if camera_motion not in self.VALID_CAMERA_MOTIONS:
            return LTX2Result(
                success=False,
                error=f"Invalid camera_motion '{camera_motion}'. Valid: {self.VALID_CAMERA_MOTIONS}"
            )
        if model_variant not in ["19b-dev", "19b-dev-fp8", "19b-dev-fp4", "19b-distilled"]:
            return LTX2Result(
                success=False,
                error="model_variant must be one of ['19b-dev', '19b-dev-fp8', '19b-dev-fp4', '19b-distilled']"
            )
        
        # Submit job
        logs.append(f"Submitting LTX-2 job...")
        submit_result = self.submit_job(
            image_path=image_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            camera_motion=camera_motion,
            model_variant=model_variant,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            fps=fps,
            seed=seed,
            output_name=output_name,
            s3_bucket=s3_bucket,
            s3_region=s3_region
        )
        
        if "error" in submit_result:
            return LTX2Result(
                success=False,
                error=submit_result["error"],
                logs="\n".join(logs)
            )
        
        job_id = submit_result.get("job_id")
        logs.append(f"Job submitted: {job_id}")
        
        # Wait for completion
        logs.append("Waiting for completion...")
        result = self.wait_for_completion(
            job_id=job_id,
            poll_interval=poll_interval,
            max_wait=max_wait,
            progress_callback=progress_callback
        )
        
        elapsed = time.time() - start_time
        
        if result.get("status", "").upper() != "COMPLETED":
            error = result.get("error") or result.get("status", "Unknown error")
            return LTX2Result(
                success=False,
                error=error,
                duration_seconds=elapsed,
                logs="\n".join(logs)
            )
        
        # Extract output
        output = result.get("output", {})
        video_url = output.get("video_url")
        
        if not video_url:
            return LTX2Result(
                success=False,
                error="No video URL in response",
                duration_seconds=elapsed,
                logs="\n".join(logs)
            )
        
        # Download video
        os.makedirs(output_dir, exist_ok=True)
        local_path, msg = download_from_s3(video_url, output_dir)
        logs.append(msg)
        
        if not local_path:
            return LTX2Result(
                success=False,
                video_url=video_url,
                error=f"Download failed: {msg}",
                duration_seconds=elapsed,
                logs="\n".join(logs)
            )
        
        logs.append(f"Video saved to: {local_path}")
        
        return LTX2Result(
            success=True,
            video_path=local_path,
            video_url=video_url,
            num_frames=output.get("num_frames", num_frames),
            duration=output.get("duration", num_frames / fps),
            fps=output.get("fps", fps),
            width=output.get("width", width),
            height=output.get("height", height),
            camera_motion=output.get("camera_motion", camera_motion),
            seed=output.get("seed", seed),
            duration_seconds=elapsed,
            logs="\n".join(logs)
        )


# Default LTX-2 client
_ltx2_client: Optional[LTX2ServerlessClient] = None


def get_ltx2_client(
    endpoint_id: Optional[str] = None,
    api_key: Optional[str] = None
) -> Optional[LTX2ServerlessClient]:
    """Get or create the LTX-2 client."""
    global _ltx2_client
    
    if endpoint_id and api_key:
        _ltx2_client = LTX2ServerlessClient(
            endpoint_id=endpoint_id,
            api_key=api_key
        )
    
    return _ltx2_client


# =============================================================================
# 2DGS PIPELINE CLIENT (existing)
# =============================================================================

# Default 2DGS client
_2dgs_client: Optional[TwoDGSPipelineClient] = None


def get_2dgs_client(
    endpoint_id: Optional[str] = None,
    api_key: Optional[str] = None
) -> Optional[TwoDGSPipelineClient]:
    """Get or create the 2DGS pipeline client."""
    global _2dgs_client
    
    if api_key:
        _2dgs_client = TwoDGSPipelineClient(
            endpoint_id=endpoint_id or TwoDGSPipelineClient.DEFAULT_ENDPOINT_ID,
            api_key=api_key
        )
    
    return _2dgs_client


# Default unified client
_unified_client: Optional[UnifiedServerlessClient] = None


def get_unified_client(
    endpoint_id: Optional[str] = None,
    api_key: Optional[str] = None
) -> Optional[UnifiedServerlessClient]:
    """Get or create the unified serverless client."""
    global _unified_client
    
    if endpoint_id and api_key:
        _unified_client = UnifiedServerlessClient(endpoint_id, api_key)
    
    return _unified_client

