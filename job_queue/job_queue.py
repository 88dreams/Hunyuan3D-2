"""
Job Queue for 3D Generation Studio

This module provides job queue management for parallel execution including:
- Job status tracking
- Queue operations (add, update, cancel, remove)
- Statistics and display formatting
"""

import os
import uuid
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any


class JobStatus(Enum):
    """Status of a job in the queue."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueuedJob:
    """Represents a job in the queue."""
    id: str
    model: str  # "hunyuan", "gen3c", "lyra", "sharp", "trellis"
    input_path: str
    input_type: str  # "image" or "video"
    status: JobStatus
    settings: Dict[str, Any]
    runpod_job_id: Optional[str] = None
    progress: float = 0.0
    output_path: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    def to_display_row(self) -> List[str]:
        """Convert to display row for Dataframe."""
        status_icons = {
            JobStatus.QUEUED: "🕐",
            JobStatus.RUNNING: "⏳",
            JobStatus.COMPLETED: "✅",
            JobStatus.FAILED: "❌",
            JobStatus.CANCELLED: "🚫",
        }
        icon = status_icons.get(self.status, "❓")
        
        if self.status == JobStatus.RUNNING:
            progress_str = f"{icon} {int(self.progress * 100)}%"
        elif self.status == JobStatus.COMPLETED:
            progress_str = f"{icon} {self.duration_seconds:.1f}s"
        elif self.status == JobStatus.FAILED:
            progress_str = f"{icon} Error"
        else:
            progress_str = f"{icon} {self.status.value}"
        
        return [
            self.id[:8],  # Short ID
            self.model.upper(),
            progress_str,
            os.path.basename(self.input_path) if self.input_path else "N/A",
        ]


class JobQueue:
    """Manages the job queue for parallel execution."""
    
    def __init__(self):
        self.jobs: Dict[str, QueuedJob] = {}
        self._lock = threading.Lock()
    
    def add_job(
        self, 
        model: str, 
        input_path: str, 
        input_type: str, 
        settings: Dict[str, Any]
    ) -> QueuedJob:
        """Add a new job to the queue."""
        job_id = str(uuid.uuid4())
        job = QueuedJob(
            id=job_id,
            model=model,
            input_path=input_path,
            input_type=input_type,
            status=JobStatus.QUEUED,
            settings=settings,
        )
        with self._lock:
            self.jobs[job_id] = job
        return job
    
    def update_job(self, job_id: str, **kwargs) -> Optional[QueuedJob]:
        """Update job properties."""
        with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                for key, value in kwargs.items():
                    if hasattr(job, key):
                        setattr(job, key, value)
                return job
        return None
    
    def get_job(self, job_id: str) -> Optional[QueuedJob]:
        """Get a job by ID."""
        return self.jobs.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                    job.status = JobStatus.CANCELLED
                    return True
        return False
    
    def remove_job(self, job_id: str) -> bool:
        """Remove a job from the queue."""
        with self._lock:
            if job_id in self.jobs:
                del self.jobs[job_id]
                return True
        return False
    
    def clear_completed(self) -> int:
        """Remove all completed/failed/cancelled jobs."""
        count = 0
        with self._lock:
            to_remove = [
                job_id for job_id, job in self.jobs.items()
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
            ]
            for job_id in to_remove:
                del self.jobs[job_id]
                count += 1
        return count
    
    def get_all_jobs(self) -> List[QueuedJob]:
        """Get all jobs sorted by creation time."""
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        stats = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        for job in self.jobs.values():
            if job.status == JobStatus.QUEUED:
                stats["pending"] += 1
            elif job.status == JobStatus.RUNNING:
                stats["running"] += 1
            elif job.status == JobStatus.COMPLETED:
                stats["completed"] += 1
            elif job.status in (JobStatus.FAILED, JobStatus.CANCELLED):
                stats["failed"] += 1
        return stats
    
    def to_dataframe_data(self) -> List[List[str]]:
        """Convert queue to dataframe format."""
        jobs = self.get_all_jobs()
        return [job.to_display_row() for job in jobs]


# Global job queue instance
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Get the global job queue instance."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
    return _job_queue


def get_queue_display() -> List[List[str]]:
    """Get queue data for display in dataframe."""
    return get_job_queue().to_dataframe_data()


def get_queue_stats_display() -> str:
    """Get queue statistics as formatted string."""
    stats = get_job_queue().get_queue_stats()
    return f"Job Queue ({stats['pending']} pending, {stats['running']} running)"


def clear_completed_jobs() -> tuple:
    """Clear completed jobs and return updated display."""
    count = get_job_queue().clear_completed()
    return get_queue_display(), f"Cleared {count} completed jobs"

