#!/usr/bin/env python3
"""
Unified Job Manager for Hunyuan3D and GEN3C generation.

This module provides a high-level interface for submitting and tracking
generation jobs. It automatically chooses between:
- Ray distributed execution (when cluster is available)
- Local single-system execution (fallback)

The JobManager handles:
- Job submission (async or sync)
- Status tracking
- Result retrieval
- Automatic mode selection based on cluster availability
"""

import os
import sys
import uuid
import time
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

# Handle both package import and direct script execution
if __name__ == "__main__" or __package__ is None:
    # Running as script - use absolute import
    import ray_jobs
    is_ray_available = ray_jobs.is_ray_available
    is_ray_initialized = ray_jobs.is_ray_initialized
    init_ray = ray_jobs.init_ray
    get_cluster_status = ray_jobs.get_cluster_status
    local_run_hunyuan = ray_jobs.local_run_hunyuan
    local_run_gen3c = ray_jobs.local_run_gen3c
else:
    # Running as package - use relative import
    from .ray_jobs import (
        is_ray_available,
        is_ray_initialized,
        init_ray,
        get_cluster_status,
        local_run_hunyuan,
        local_run_gen3c,
    )

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(Enum):
    """Type of generation job."""
    HUNYUAN = "hunyuan"
    GEN3C = "gen3c"


@dataclass
class JobResult:
    """Result of a completed job."""
    success: bool
    output_path: Optional[str] = None
    logs: str = ""
    error: Optional[str] = None
    node: str = "local"
    duration_seconds: float = 0.0


@dataclass
class Job:
    """Represents a generation job."""
    job_id: str
    job_type: JobType
    status: JobStatus
    params: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[JobResult] = None
    ray_ref: Any = None  # Ray ObjectRef when using distributed mode
    
    @property
    def duration(self) -> Optional[float]:
        """Get job duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()


class JobManager:
    """
    Manages generation jobs with automatic Ray/local mode selection.
    
    Usage:
        manager = JobManager()
        
        # Submit a job (returns immediately)
        job_id = manager.submit_hunyuan(image_path="/path/to/image.png", ...)
        
        # Check status
        status = manager.get_status(job_id)
        
        # Get result (blocks if not complete)
        result = manager.get_result(job_id)
        
        # Or run synchronously
        result = manager.run_hunyuan_sync(image_path="/path/to/image.png", ...)
    """
    
    def __init__(self, use_ray: Optional[bool] = None, ray_address: str = "auto"):
        """
        Initialize the JobManager.
        
        Args:
            use_ray: Force Ray mode (True), local mode (False), or auto-detect (None)
            ray_address: Ray cluster address for auto-connect
        """
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._ray_address = ray_address
        
        # Determine execution mode
        if use_ray is None:
            # Auto-detect: use Ray if available and cluster is running
            self._use_ray = self._try_connect_ray()
        elif use_ray:
            # Forced Ray mode
            self._use_ray = self._try_connect_ray()
            if not self._use_ray:
                logger.warning("Ray requested but not available, falling back to local mode")
        else:
            # Forced local mode
            self._use_ray = False
        
        mode = "distributed (Ray)" if self._use_ray else "single-system (local)"
        logger.info(f"JobManager initialized in {mode} mode")
    
    def _try_connect_ray(self) -> bool:
        """Try to connect to Ray cluster."""
        if not is_ray_available():
            return False
        
        if is_ray_initialized():
            return True
        
        return init_ray(self._ray_address)
    
    @property
    def mode(self) -> str:
        """Get current execution mode."""
        return "distributed" if self._use_ray else "local"
    
    @property
    def cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status."""
        return get_cluster_status()
    
    def _generate_job_id(self) -> str:
        """Generate unique job ID."""
        return f"job_{uuid.uuid4().hex[:12]}"
    
    def _create_job(self, job_type: JobType, params: Dict[str, Any]) -> Job:
        """Create and register a new job."""
        job = Job(
            job_id=self._generate_job_id(),
            job_type=job_type,
            status=JobStatus.PENDING,
            params=params,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job
    
    def _update_job(self, job_id: str, **updates):
        """Update job attributes."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                for key, value in updates.items():
                    setattr(job, key, value)
    
    # =========================================================================
    # HUNYUAN JOBS
    # =========================================================================
    
    def submit_hunyuan(
        self,
        image_path: str,
        guidance_scale: float = 9.0,
        steps: int = 40,
        seed: Optional[int] = None,
        model_choice: str = "Mini Model (Faster)",
        use_fp16: bool = True,
        attention_slicing: bool = True,
        cpu_offload: bool = True,
        remove_background: bool = False,
        output_name: str = "output_model",
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Submit a Hunyuan3D generation job.
        
        Returns:
            Job ID for tracking
        """
        if output_dir is None:
            try:
                from config import get_path
                output_dir = get_path('hunyuan_outputs', create=True)
            except ImportError:
                output_dir = "/srv/searidge_share/outputs/hunyuan"
        
        params = {
            "image_path": image_path,
            "guidance_scale": guidance_scale,
            "steps": steps,
            "seed": seed,
            "model_choice": model_choice,
            "use_fp16": use_fp16,
            "attention_slicing": attention_slicing,
            "cpu_offload": cpu_offload,
            "remove_background": remove_background,
            "output_name": output_name,
            "output_dir": output_dir,
        }
        
        job = self._create_job(JobType.HUNYUAN, params)
        
        # Start execution
        if self._use_ray:
            self._submit_ray_hunyuan(job)
        else:
            self._submit_local_hunyuan(job)
        
        return job.job_id
    
    def _submit_ray_hunyuan(self, job: Job):
        """Submit Hunyuan job to Ray cluster."""
        if __name__ == "__main__" or __package__ is None:
            from ray_jobs import ray_run_hunyuan
        else:
            from .ray_jobs import ray_run_hunyuan
        
        self._update_job(job.job_id, status=JobStatus.RUNNING, started_at=datetime.now())
        
        ref = ray_run_hunyuan.remote(
            image_path=job.params["image_path"],
            guidance_scale=job.params["guidance_scale"],
            steps=job.params["steps"],
            seed=job.params["seed"],
            model_choice=job.params["model_choice"],
            use_fp16=job.params["use_fp16"],
            attention_slicing=job.params["attention_slicing"],
            cpu_offload=job.params["cpu_offload"],
            remove_background=job.params["remove_background"],
            output_name=job.params["output_name"],
            output_dir=job.params["output_dir"],
        )
        
        self._update_job(job.job_id, ray_ref=ref)
        
        # Start background thread to wait for result
        thread = threading.Thread(
            target=self._wait_for_ray_result,
            args=(job.job_id, ref),
            daemon=True,
        )
        thread.start()
    
    def _submit_local_hunyuan(self, job: Job):
        """Submit Hunyuan job for local execution."""
        self._update_job(job.job_id, status=JobStatus.RUNNING, started_at=datetime.now())
        
        # Run in background thread
        thread = threading.Thread(
            target=self._run_local_hunyuan,
            args=(job.job_id,),
            daemon=True,
        )
        thread.start()
    
    def _run_local_hunyuan(self, job_id: str):
        """Execute Hunyuan job locally."""
        job = self._jobs.get(job_id)
        if not job:
            return
        
        start_time = time.time()
        
        try:
            result_dict = local_run_hunyuan(
                image_path=job.params["image_path"],
                guidance_scale=job.params["guidance_scale"],
                steps=job.params["steps"],
                seed=job.params["seed"],
                model_choice=job.params["model_choice"],
                use_fp16=job.params["use_fp16"],
                attention_slicing=job.params["attention_slicing"],
                cpu_offload=job.params["cpu_offload"],
                remove_background=job.params["remove_background"],
                output_name=job.params["output_name"],
                output_dir=job.params["output_dir"],
            )
            
            duration = time.time() - start_time
            
            result = JobResult(
                success=result_dict["success"],
                output_path=result_dict.get("output_path"),
                logs=result_dict.get("logs", ""),
                error=result_dict.get("error"),
                node=result_dict.get("node", "local"),
                duration_seconds=duration,
            )
            
            status = JobStatus.COMPLETED if result.success else JobStatus.FAILED
            self._update_job(
                job_id,
                status=status,
                completed_at=datetime.now(),
                result=result,
            )
            
        except Exception as e:
            duration = time.time() - start_time
            result = JobResult(
                success=False,
                error=str(e),
                duration_seconds=duration,
            )
            self._update_job(
                job_id,
                status=JobStatus.FAILED,
                completed_at=datetime.now(),
                result=result,
            )
    
    # =========================================================================
    # GEN3C JOBS
    # =========================================================================
    
    def submit_gen3c(
        self,
        image_path: str,
        guidance: float = 1.0,
        frames: int = 121,
        video_name: str = "gen3c_video",
        checkpoint_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        extra_args: str = "",
    ) -> str:
        """
        Submit a GEN3C video generation job.
        
        Returns:
            Job ID for tracking
        """
        if checkpoint_dir is None:
            try:
                from config import get_path
                checkpoint_dir = get_path('gen3c_checkpoints')
            except ImportError:
                checkpoint_dir = "/srv/searidge_share/checkpoints/gen3c"
        
        if output_dir is None:
            try:
                from config import get_path
                output_dir = get_path('gen3c_outputs', create=True)
            except ImportError:
                output_dir = "/srv/searidge_share/outputs/gen3c"
        
        params = {
            "image_path": image_path,
            "guidance": guidance,
            "frames": frames,
            "video_name": video_name,
            "checkpoint_dir": checkpoint_dir,
            "output_dir": output_dir,
            "extra_args": extra_args,
        }
        
        job = self._create_job(JobType.GEN3C, params)
        
        if self._use_ray:
            self._submit_ray_gen3c(job)
        else:
            self._submit_local_gen3c(job)
        
        return job.job_id
    
    def _submit_ray_gen3c(self, job: Job):
        """Submit GEN3C job to Ray cluster."""
        if __name__ == "__main__" or __package__ is None:
            from ray_jobs import ray_run_gen3c
        else:
            from .ray_jobs import ray_run_gen3c
        
        self._update_job(job.job_id, status=JobStatus.RUNNING, started_at=datetime.now())
        
        ref = ray_run_gen3c.remote(
            image_path=job.params["image_path"],
            guidance=job.params["guidance"],
            frames=job.params["frames"],
            video_name=job.params["video_name"],
            checkpoint_dir=job.params["checkpoint_dir"],
            output_dir=job.params["output_dir"],
            extra_args=job.params["extra_args"],
        )
        
        self._update_job(job.job_id, ray_ref=ref)
        
        thread = threading.Thread(
            target=self._wait_for_ray_result,
            args=(job.job_id, ref),
            daemon=True,
        )
        thread.start()
    
    def _submit_local_gen3c(self, job: Job):
        """Submit GEN3C job for local execution."""
        self._update_job(job.job_id, status=JobStatus.RUNNING, started_at=datetime.now())
        
        thread = threading.Thread(
            target=self._run_local_gen3c,
            args=(job.job_id,),
            daemon=True,
        )
        thread.start()
    
    def _run_local_gen3c(self, job_id: str):
        """Execute GEN3C job locally."""
        job = self._jobs.get(job_id)
        if not job:
            return
        
        start_time = time.time()
        
        try:
            result_dict = local_run_gen3c(
                image_path=job.params["image_path"],
                guidance=job.params["guidance"],
                frames=job.params["frames"],
                video_name=job.params["video_name"],
                checkpoint_dir=job.params["checkpoint_dir"],
                output_dir=job.params["output_dir"],
                extra_args=job.params["extra_args"],
            )
            
            duration = time.time() - start_time
            
            result = JobResult(
                success=result_dict["success"],
                output_path=result_dict.get("output_path"),
                logs=result_dict.get("logs", ""),
                error=result_dict.get("error"),
                node=result_dict.get("node", "local"),
                duration_seconds=duration,
            )
            
            status = JobStatus.COMPLETED if result.success else JobStatus.FAILED
            self._update_job(
                job_id,
                status=status,
                completed_at=datetime.now(),
                result=result,
            )
            
        except Exception as e:
            duration = time.time() - start_time
            result = JobResult(
                success=False,
                error=str(e),
                duration_seconds=duration,
            )
            self._update_job(
                job_id,
                status=JobStatus.FAILED,
                completed_at=datetime.now(),
                result=result,
            )
    
    # =========================================================================
    # RAY RESULT HANDLING
    # =========================================================================
    
    def _wait_for_ray_result(self, job_id: str, ray_ref):
        """Wait for Ray job to complete and update job status."""
        import ray
        
        start_time = time.time()
        
        try:
            result_dict = ray.get(ray_ref)
            duration = time.time() - start_time
            
            result = JobResult(
                success=result_dict["success"],
                output_path=result_dict.get("output_path"),
                logs=result_dict.get("logs", ""),
                error=result_dict.get("error"),
                node=result_dict.get("node", "unknown"),
                duration_seconds=duration,
            )
            
            status = JobStatus.COMPLETED if result.success else JobStatus.FAILED
            self._update_job(
                job_id,
                status=status,
                completed_at=datetime.now(),
                result=result,
            )
            
        except Exception as e:
            duration = time.time() - start_time
            result = JobResult(
                success=False,
                error=str(e),
                duration_seconds=duration,
            )
            self._update_job(
                job_id,
                status=JobStatus.FAILED,
                completed_at=datetime.now(),
                result=result,
            )
    
    # =========================================================================
    # SYNCHRONOUS EXECUTION
    # =========================================================================
    
    def run_hunyuan_sync(self, **kwargs) -> JobResult:
        """
        Run Hunyuan generation synchronously (blocking).
        
        Returns:
            JobResult with generation outcome
        """
        job_id = self.submit_hunyuan(**kwargs)
        return self.wait_for_result(job_id)
    
    def run_gen3c_sync(self, **kwargs) -> JobResult:
        """
        Run GEN3C generation synchronously (blocking).
        
        Returns:
            JobResult with generation outcome
        """
        job_id = self.submit_gen3c(**kwargs)
        return self.wait_for_result(job_id)
    
    # =========================================================================
    # JOB STATUS AND RESULTS
    # =========================================================================
    
    def get_status(self, job_id: str) -> Optional[JobStatus]:
        """Get current status of a job."""
        with self._lock:
            job = self._jobs.get(job_id)
            return job.status if job else None
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get full job information."""
        with self._lock:
            return self._jobs.get(job_id)
    
    def get_result(self, job_id: str) -> Optional[JobResult]:
        """Get result of a completed job (non-blocking)."""
        with self._lock:
            job = self._jobs.get(job_id)
            return job.result if job else None
    
    def wait_for_result(self, job_id: str, timeout: Optional[float] = None) -> JobResult:
        """
        Wait for job to complete and return result.
        
        Args:
            job_id: Job ID to wait for
            timeout: Maximum seconds to wait (None = forever)
        
        Returns:
            JobResult
        
        Raises:
            TimeoutError if timeout exceeded
            KeyError if job not found
        """
        start_time = time.time()
        
        while True:
            job = self.get_job(job_id)
            if job is None:
                raise KeyError(f"Job not found: {job_id}")
            
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return job.result or JobResult(success=False, error="No result available")
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")
            
            time.sleep(0.5)
    
    def is_complete(self, job_id: str) -> bool:
        """Check if job has completed (success or failure)."""
        status = self.get_status(job_id)
        return status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
    
    def list_jobs(self, status: Optional[JobStatus] = None) -> List[Job]:
        """List all jobs, optionally filtered by status."""
        with self._lock:
            jobs = list(self._jobs.values())
            if status:
                jobs = [j for j in jobs if j.status == status]
            return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending or running job.
        
        Returns:
            True if cancelled, False if job not found or already complete
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return False
            
            # For Ray jobs, try to cancel
            if job.ray_ref and self._use_ray:
                try:
                    import ray
                    ray.cancel(job.ray_ref, force=True)
                except Exception:
                    pass
            
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            return True
    
    def cleanup_old_jobs(self, max_age_hours: float = 24):
        """Remove completed jobs older than max_age_hours."""
        cutoff = datetime.now()
        with self._lock:
            to_remove = []
            for job_id, job in self._jobs.items():
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                    if job.completed_at:
                        age_hours = (cutoff - job.completed_at).total_seconds() / 3600
                        if age_hours > max_age_hours:
                            to_remove.append(job_id)
            
            for job_id in to_remove:
                del self._jobs[job_id]
            
            return len(to_remove)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_default_manager: Optional[JobManager] = None


def get_job_manager(use_ray: Optional[bool] = None) -> JobManager:
    """
    Get or create the default JobManager instance.
    
    Args:
        use_ray: Force Ray mode (True), local mode (False), or auto-detect (None)
    
    Returns:
        JobManager instance
    """
    global _default_manager
    
    if _default_manager is None:
        _default_manager = JobManager(use_ray=use_ray)
    
    return _default_manager


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Job Manager Test")
    print("=" * 50)
    
    manager = JobManager()
    print(f"Mode: {manager.mode}")
    print(f"Cluster status: {manager.cluster_status}")
    
    print("\nReady to submit jobs!")

