"""
RunPod Integration Package

Provides clients and utilities for running 3D generation models on RunPod.
"""

from runpod.runpod_client import (
    RunPodGEN3CClient,
    UnifiedServerlessClient,
    RunPodJobResult,
    JobStatus,
    get_unified_client,
    download_from_s3,
)

# SSH download utilities
try:
    from runpod.ssh_download import (
        configure_ssh,
        test_ssh_connection,
        download_from_runpod,
        download_sharp_output,
        download_gen3c_output,
        download_lyra_output,
        download_trellis_output,
        list_remote_files,
        load_ssh_config,
        save_ssh_config,
        SSHConfig,
    )
    SSH_AVAILABLE = True
except ImportError:
    SSH_AVAILABLE = False

__all__ = [
    # Clients
    "RunPodGEN3CClient",
    "UnifiedServerlessClient",
    "RunPodJobResult",
    "JobStatus",
    "get_unified_client",
    # S3 download
    "download_from_s3",
    # SSH
    "SSH_AVAILABLE",
    "configure_ssh",
    "test_ssh_connection",
    "download_from_runpod",
    "download_sharp_output",
    "download_gen3c_output",
    "download_lyra_output",
    "download_trellis_output",
    "list_remote_files",
    "load_ssh_config",
    "save_ssh_config",
    "SSHConfig",
]

