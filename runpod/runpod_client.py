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
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RunPodJobResult:
    """Result from a RunPod GEN3C job."""
    success: bool
    job_id: str
    status: JobStatus
    output_path: Optional[str] = None
    video_base64: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    logs: str = ""


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
        seed: Optional[int] = None
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

