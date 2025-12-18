#!/usr/bin/env python3
"""
RunPod Client for GEN3C Integration

This module provides a client for interacting with GEN3C running on RunPod,
either as a GPU Pod (via REST API) or as a Serverless Endpoint.

Usage:
    from runpod_client import RunPodClient
    
    # For GPU Pod
    client = RunPodClient(mode="pod", pod_url="https://xxx-8000.proxy.runpod.net")
    
    # For Serverless
    client = RunPodClient(mode="serverless", api_key="xxx", endpoint_id="xxx")
    
    # Generate video
    result = client.generate_video(
        image_path="/path/to/image.png",
        guidance=1.0,
        num_frames=121,
        trajectory="left"
    )
"""

import os
import base64
import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RunPodMode(Enum):
    """RunPod deployment mode."""
    POD = "pod"           # GPU Pod with REST API
    SERVERLESS = "serverless"  # Serverless Endpoint


@dataclass
class JobResult:
    """Result of a RunPod job."""
    success: bool
    job_id: str
    video_path: Optional[str] = None
    video_base64: Optional[str] = None
    error: Optional[str] = None
    elapsed_time: Optional[float] = None


class RunPodClient:
    """
    Client for RunPod GEN3C inference.
    
    Supports both GPU Pod (direct API) and Serverless Endpoint modes.
    """
    
    def __init__(
        self,
        mode: str = "pod",
        pod_url: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        timeout: int = 7200,  # 2 hours default timeout
    ):
        """
        Initialize RunPod client.
        
        Args:
            mode: "pod" or "serverless"
            pod_url: URL of the GPU Pod API (for pod mode)
            api_key: RunPod API key (for serverless mode)
            endpoint_id: Serverless endpoint ID (for serverless mode)
            timeout: Maximum time to wait for job completion (seconds)
        """
        self.mode = RunPodMode(mode)
        self.pod_url = pod_url.rstrip("/") if pod_url else None
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self.endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID")
        self.timeout = timeout
        
        # Validate configuration
        if self.mode == RunPodMode.POD and not self.pod_url:
            raise ValueError("pod_url required for pod mode")
        if self.mode == RunPodMode.SERVERLESS and (not self.api_key or not self.endpoint_id):
            raise ValueError("api_key and endpoint_id required for serverless mode")
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image file to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _decode_video(self, video_base64: str, output_path: str) -> str:
        """Decode base64 video and save to file."""
        video_data = base64.b64decode(video_base64)
        with open(output_path, "wb") as f:
            f.write(video_data)
        return output_path
    
    # =========================================================================
    # GPU POD MODE
    # =========================================================================
    
    def _pod_health_check(self) -> bool:
        """Check if the pod is healthy."""
        try:
            response = requests.get(f"{self.pod_url}/health", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Pod health check failed: {e}")
            return False
    
    def _pod_submit_job(
        self,
        image_base64: str,
        video_name: str,
        guidance: float,
        num_frames: int,
        trajectory: str,
        foreground_masking: bool,
        seed: Optional[int]
    ) -> str:
        """Submit job to GPU Pod API. Returns job_id."""
        payload = {
            "image_base64": image_base64,
            "video_name": video_name,
            "guidance": guidance,
            "num_frames": num_frames,
            "trajectory": trajectory,
            "foreground_masking": foreground_masking,
        }
        if seed is not None:
            payload["seed"] = seed
        
        response = requests.post(
            f"{self.pod_url}/generate",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["job_id"]
    
    def _pod_get_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status from GPU Pod API."""
        response = requests.get(f"{self.pod_url}/status/{job_id}", timeout=10)
        response.raise_for_status()
        return response.json()
    
    def _pod_wait_for_completion(
        self,
        job_id: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """Wait for pod job to complete."""
        start_time = time.time()
        poll_interval = 10  # seconds
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                raise TimeoutError(f"Job {job_id} timed out after {self.timeout}s")
            
            status = self._pod_get_status(job_id)
            
            if progress_callback:
                progress = status.get("progress", 0)
                progress_callback(status["status"], progress)
            
            if status["status"] == "completed":
                return status
            elif status["status"] == "failed":
                raise RuntimeError(f"Job failed: {status.get('error', 'Unknown error')}")
            
            time.sleep(poll_interval)
    
    # =========================================================================
    # SERVERLESS MODE
    # =========================================================================
    
    def _serverless_submit_job(
        self,
        image_base64: str,
        video_name: str,
        guidance: float,
        num_frames: int,
        trajectory: str,
        foreground_masking: bool,
        seed: Optional[int]
    ) -> str:
        """Submit job to RunPod Serverless Endpoint. Returns job_id."""
        payload = {
            "input": {
                "image_base64": image_base64,
                "video_name": video_name,
                "guidance": guidance,
                "num_frames": num_frames,
                "trajectory": trajectory,
                "foreground_masking": foreground_masking,
                "return_base64": True,
            }
        }
        if seed is not None:
            payload["input"]["seed"] = seed
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"https://api.runpod.ai/v2/{self.endpoint_id}/run",
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["id"]
    
    def _serverless_get_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status from RunPod Serverless API."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        response = requests.get(
            f"https://api.runpod.ai/v2/{self.endpoint_id}/status/{job_id}",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    
    def _serverless_wait_for_completion(
        self,
        job_id: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """Wait for serverless job to complete."""
        start_time = time.time()
        poll_interval = 15  # seconds (serverless has longer cold starts)
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                raise TimeoutError(f"Job {job_id} timed out after {self.timeout}s")
            
            status = self._serverless_get_status(job_id)
            status_str = status.get("status", "UNKNOWN")
            
            if progress_callback:
                # Serverless doesn't provide progress %, estimate based on status
                progress_map = {
                    "IN_QUEUE": 0.1,
                    "IN_PROGRESS": 0.5,
                    "COMPLETED": 1.0,
                    "FAILED": 0.0
                }
                progress_callback(status_str, progress_map.get(status_str, 0))
            
            if status_str == "COMPLETED":
                return status
            elif status_str == "FAILED":
                error = status.get("error", "Unknown error")
                raise RuntimeError(f"Job failed: {error}")
            
            time.sleep(poll_interval)
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def health_check(self) -> bool:
        """
        Check if RunPod backend is available.
        
        Returns:
            True if healthy, False otherwise
        """
        if self.mode == RunPodMode.POD:
            return self._pod_health_check()
        else:
            # Serverless is always "available" if credentials are set
            return bool(self.api_key and self.endpoint_id)
    
    def generate_video(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        video_name: str = "gen3c_output",
        guidance: float = 1.0,
        num_frames: int = 121,
        trajectory: str = "left",
        foreground_masking: bool = True,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> JobResult:
        """
        Generate video from image using GEN3C on RunPod.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output video (optional)
            video_name: Name for the output video
            guidance: Guidance scale (0.5-3.0)
            num_frames: Number of frames (121, 241, 361, 481)
            trajectory: Camera trajectory
            foreground_masking: Enable foreground masking
            seed: Random seed for reproducibility
            progress_callback: Callback function(status, progress) for updates
        
        Returns:
            JobResult with video_path or video_base64
        """
        start_time = time.time()
        
        try:
            # Encode image
            logger.info(f"Encoding image: {image_path}")
            image_base64 = self._encode_image(image_path)
            
            # Submit job
            logger.info(f"Submitting job to RunPod ({self.mode.value} mode)")
            if self.mode == RunPodMode.POD:
                job_id = self._pod_submit_job(
                    image_base64, video_name, guidance, num_frames,
                    trajectory, foreground_masking, seed
                )
            else:
                job_id = self._serverless_submit_job(
                    image_base64, video_name, guidance, num_frames,
                    trajectory, foreground_masking, seed
                )
            
            logger.info(f"Job submitted: {job_id}")
            
            # Wait for completion
            if self.mode == RunPodMode.POD:
                result = self._pod_wait_for_completion(job_id, progress_callback)
                video_base64 = result.get("video_base64")
            else:
                result = self._serverless_wait_for_completion(job_id, progress_callback)
                output = result.get("output", {})
                video_base64 = output.get("video_base64")
            
            elapsed = time.time() - start_time
            logger.info(f"Job {job_id} completed in {elapsed:.1f}s")
            
            # Save video if output path provided
            video_path = None
            if output_path and video_base64:
                video_path = self._decode_video(video_base64, output_path)
                logger.info(f"Video saved to: {video_path}")
            
            return JobResult(
                success=True,
                job_id=job_id,
                video_path=video_path,
                video_base64=video_base64,
                elapsed_time=elapsed
            )
            
        except Exception as e:
            logger.exception("RunPod job failed")
            return JobResult(
                success=False,
                job_id="",
                error=str(e),
                elapsed_time=time.time() - start_time
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_pod_client(pod_url: str, timeout: int = 7200) -> RunPodClient:
    """Create a client for GPU Pod mode."""
    return RunPodClient(mode="pod", pod_url=pod_url, timeout=timeout)


def create_serverless_client(
    api_key: Optional[str] = None,
    endpoint_id: Optional[str] = None,
    timeout: int = 7200
) -> RunPodClient:
    """Create a client for Serverless mode."""
    return RunPodClient(
        mode="serverless",
        api_key=api_key,
        endpoint_id=endpoint_id,
        timeout=timeout
    )


# =============================================================================
# CLI TESTING
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RunPod GEN3C client")
    parser.add_argument("--mode", choices=["pod", "serverless"], default="pod")
    parser.add_argument("--pod-url", help="GPU Pod URL")
    parser.add_argument("--api-key", help="RunPod API key")
    parser.add_argument("--endpoint-id", help="Serverless endpoint ID")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument("--frames", type=int, default=121, help="Number of frames")
    parser.add_argument("--trajectory", default="left", help="Camera trajectory")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.mode == "pod":
        client = create_pod_client(args.pod_url)
    else:
        client = create_serverless_client(args.api_key, args.endpoint_id)
    
    def progress_callback(status: str, progress: float):
        print(f"Status: {status}, Progress: {progress*100:.0f}%")
    
    result = client.generate_video(
        image_path=args.image,
        output_path=args.output,
        num_frames=args.frames,
        trajectory=args.trajectory,
        progress_callback=progress_callback
    )
    
    if result.success:
        print(f"Success! Video saved to: {result.video_path}")
        print(f"Elapsed time: {result.elapsed_time:.1f}s")
    else:
        print(f"Failed: {result.error}")

