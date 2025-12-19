"""
System Metrics for 3D Generation Studio

This module provides system monitoring utilities including:
- CPU/Memory usage tracking
- GPU monitoring (ROCm/AMD)
- Background metrics collection
"""

import time
import threading
import subprocess
from typing import Dict

import psutil  # type: ignore


# Global metrics storage
_system_metrics: Dict[str, float] = {
    "cpu_percent": 0.0,
    "memory_percent": 0.0,
    "gpu_percent": 0.0,
    "gpu_memory_percent": 0.0,
    "last_update": 0.0
}

# Background monitoring thread
_monitor_thread = None
_monitoring_active = False


def _update_system_metrics_loop():
    """Background loop to update system metrics every 2 seconds."""
    global _monitoring_active
    
    while _monitoring_active:
        try:
            _system_metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
            _system_metrics["memory_percent"] = psutil.virtual_memory().percent
            _system_metrics["gpu_percent"] = 0.0
            _system_metrics["gpu_memory_percent"] = 0.0

            # Try ROCm/AMD GPU monitoring
            try:
                result = subprocess.run(
                    ['rocm-smi', '--showuse'], 
                    capture_output=True, 
                    text=True, 
                    timeout=2
                )
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'GPU use' in line:
                            parts = line.split()
                            if len(parts) >= 3:
                                try:
                                    _system_metrics["gpu_percent"] = float(parts[-1].rstrip('%'))
                                except (ValueError, IndexError):
                                    pass
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            _system_metrics["last_update"] = time.time()
        except Exception:
            pass

        time.sleep(2.0)


def start_monitoring():
    """Start the background monitoring thread."""
    global _monitor_thread, _monitoring_active
    
    if _monitor_thread is not None and _monitor_thread.is_alive():
        return  # Already running
    
    _monitoring_active = True
    _monitor_thread = threading.Thread(target=_update_system_metrics_loop, daemon=True)
    _monitor_thread.start()


def stop_monitoring():
    """Stop the background monitoring thread."""
    global _monitoring_active
    _monitoring_active = False


def get_system_metrics() -> Dict[str, float]:
    """Get current system metrics."""
    return _system_metrics.copy()


def format_system_metrics(metrics: Dict[str, float]) -> str:
    """Format system metrics for display in UI."""
    cpu = metrics.get("cpu_percent", 0)
    mem = metrics.get("memory_percent", 0)
    gpu = metrics.get("gpu_percent", 0)
    gpu_mem = metrics.get("gpu_memory_percent", 0)

    def color_code(value: float, thresholds: tuple = (50, 80)) -> str:
        if value >= thresholds[1]:
            return f"🔴 {value:.1f}%"
        elif value >= thresholds[0]:
            return f"🟡 {value:.1f}%"
        else:
            return f"🟢 {value:.1f}%"

    lines = [
        f"CPU: {color_code(cpu)} | Memory: {color_code(mem)}",
        f"GPU: {color_code(gpu)} | GPU Memory: {color_code(gpu_mem)}"
    ]
    return "\n".join(lines)

