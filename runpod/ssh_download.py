"""
RunPod SSH Download Utility

Downloads files from RunPod network volume to local machine via SSH/SCP.

Configuration:
    SSH credentials are stored in ~/.runpod_ssh_config.json or can be
    passed directly to the download functions.

Usage:
    from runpod.ssh_download import download_from_runpod, configure_ssh
    
    # Configure once
    configure_ssh(host="your-pod-ip", user="root", key_path="~/.ssh/runpod_key")
    
    # Download files
    local_path = download_from_runpod("/runpod-volume/outputs/sharp/file.ply", "/local/path/")
"""

import os
import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

SSH_CONFIG_FILE = os.path.expanduser("~/.runpod_ssh_config.json")

@dataclass
class SSHConfig:
    """SSH configuration for RunPod connection."""
    host: str = ""
    user: str = "root"
    port: int = 22
    key_path: str = ""
    # RunPod storage pod specific
    storage_pod_id: str = ""
    
    def is_configured(self) -> bool:
        """Check if SSH is properly configured."""
        return bool(self.host and self.key_path and os.path.exists(os.path.expanduser(self.key_path)))
    
    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "user": self.user,
            "port": self.port,
            "key_path": self.key_path,
            "storage_pod_id": self.storage_pod_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SSHConfig":
        return cls(
            host=data.get("host", ""),
            user=data.get("user", "root"),
            port=data.get("port", 22),
            key_path=data.get("key_path", ""),
            storage_pod_id=data.get("storage_pod_id", ""),
        )


def load_ssh_config() -> SSHConfig:
    """Load SSH configuration from file."""
    if os.path.exists(SSH_CONFIG_FILE):
        try:
            with open(SSH_CONFIG_FILE, "r") as f:
                return SSHConfig.from_dict(json.load(f))
        except Exception as e:
            print(f"[SSH] Warning: Could not load SSH config: {e}")
    return SSHConfig()


def save_ssh_config(config: SSHConfig) -> None:
    """Save SSH configuration to file."""
    try:
        with open(SSH_CONFIG_FILE, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        # Secure the config file
        os.chmod(SSH_CONFIG_FILE, 0o600)
    except Exception as e:
        print(f"[SSH] Warning: Could not save SSH config: {e}")


def configure_ssh(
    host: str,
    user: str = "root",
    port: int = 22,
    key_path: str = "",
    storage_pod_id: str = "",
) -> SSHConfig:
    """
    Configure SSH connection to RunPod.
    
    Args:
        host: RunPod pod IP or hostname (e.g., "123.456.789.012")
        user: SSH username (default: "root")
        port: SSH port (default: 22, RunPod often uses custom ports)
        key_path: Path to SSH private key (e.g., "~/.ssh/runpod_key")
        storage_pod_id: Optional RunPod storage pod ID
    
    Returns:
        SSHConfig object
    """
    config = SSHConfig(
        host=host,
        user=user,
        port=port,
        key_path=os.path.expanduser(key_path),
        storage_pod_id=storage_pod_id,
    )
    save_ssh_config(config)
    return config


# =============================================================================
# SSH OPERATIONS
# =============================================================================

def test_ssh_connection(config: Optional[SSHConfig] = None) -> Tuple[bool, str]:
    """
    Test SSH connection to RunPod.
    
    Returns:
        Tuple of (success, message)
    """
    if config is None:
        config = load_ssh_config()
    
    if not config.is_configured():
        return False, "SSH not configured. Call configure_ssh() first."
    
    try:
        cmd = [
            "ssh",
            "-i", config.key_path,
            "-p", str(config.port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            f"{config.user}@{config.host}",
            "echo 'SSH connection successful'"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode == 0:
            return True, "✅ SSH connection successful"
        else:
            return False, f"❌ SSH connection failed: {result.stderr}"
    
    except subprocess.TimeoutExpired:
        return False, "❌ SSH connection timed out"
    except Exception as e:
        return False, f"❌ SSH error: {e}"


def download_from_runpod(
    remote_path: str,
    local_dir: str,
    config: Optional[SSHConfig] = None,
    filename: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """
    Download a file from RunPod network volume via SCP.
    
    Args:
        remote_path: Path on RunPod (e.g., "/runpod-volume/outputs/sharp/file.ply")
        local_dir: Local directory to save the file
        config: Optional SSH config (loads from file if not provided)
        filename: Optional local filename (uses remote filename if not provided)
    
    Returns:
        Tuple of (local_file_path or None, message)
    """
    if config is None:
        config = load_ssh_config()
    
    if not config.is_configured():
        return None, "❌ SSH not configured. Use configure_ssh() or set up ~/.runpod_ssh_config.json"
    
    # Ensure local directory exists
    local_dir = os.path.expanduser(local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    # Determine local filename
    if filename is None:
        filename = os.path.basename(remote_path)
    local_path = os.path.join(local_dir, filename)
    
    try:
        # Build SCP command
        cmd = [
            "scp",
            "-i", config.key_path,
            "-P", str(config.port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=30",
            f"{config.user}@{config.host}:{remote_path}",
            local_path
        ]
        
        print(f"[SSH] Downloading: {remote_path} -> {local_path}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for large files
        )
        
        if result.returncode == 0 and os.path.exists(local_path):
            file_size = os.path.getsize(local_path)
            return local_path, f"✅ Downloaded {filename} ({file_size / 1024 / 1024:.1f}MB)"
        else:
            error = result.stderr.strip() if result.stderr else "Unknown error"
            return None, f"❌ SCP failed: {error}"
    
    except subprocess.TimeoutExpired:
        return None, "❌ Download timed out (file may be very large)"
    except Exception as e:
        return None, f"❌ Download error: {e}"


def list_remote_files(
    remote_dir: str,
    config: Optional[SSHConfig] = None,
) -> Tuple[list, str]:
    """
    List files in a remote directory on RunPod.
    
    Args:
        remote_dir: Remote directory path
        config: Optional SSH config
    
    Returns:
        Tuple of (file_list, message)
    """
    if config is None:
        config = load_ssh_config()
    
    if not config.is_configured():
        return [], "❌ SSH not configured"
    
    try:
        cmd = [
            "ssh",
            "-i", config.key_path,
            "-p", str(config.port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            f"{config.user}@{config.host}",
            f"ls -la {remote_dir} 2>/dev/null || echo 'Directory not found'"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            files = [line for line in lines if line and not line.startswith("total")]
            return files, f"✅ Found {len(files)} items"
        else:
            return [], f"❌ Could not list directory: {result.stderr}"
    
    except Exception as e:
        return [], f"❌ Error: {e}"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def download_sharp_output(
    filename: str,
    local_dir: Optional[str] = None,
    config: Optional[SSHConfig] = None,
) -> Tuple[Optional[str], str]:
    """
    Download a SHARP output file from RunPod.
    
    Args:
        filename: Filename (e.g., "output.ply")
        local_dir: Local directory (defaults to SHARP output dir)
        config: Optional SSH config
    
    Returns:
        Tuple of (local_path, message)
    """
    if local_dir is None:
        local_dir = "/srv/searidge_share/outputs/sharp"
    
    remote_path = f"/runpod-volume/outputs/sharp/{filename}"
    return download_from_runpod(remote_path, local_dir, config)


def download_gen3c_output(
    filename: str,
    local_dir: Optional[str] = None,
    config: Optional[SSHConfig] = None,
) -> Tuple[Optional[str], str]:
    """
    Download a GEN3C output file from RunPod.
    """
    if local_dir is None:
        local_dir = "/srv/searidge_share/outputs/gen3c"
    
    remote_path = f"/runpod-volume/outputs/gen3c/{filename}"
    return download_from_runpod(remote_path, local_dir, config)


def download_lyra_output(
    filename: str,
    local_dir: Optional[str] = None,
    config: Optional[SSHConfig] = None,
) -> Tuple[Optional[str], str]:
    """
    Download a Lyra output file from RunPod.
    """
    if local_dir is None:
        local_dir = "/srv/searidge_share/outputs/lyra"
    
    remote_path = f"/runpod-volume/outputs/lyra/{filename}"
    return download_from_runpod(remote_path, local_dir, config)


def download_trellis_output(
    filename: str,
    local_dir: Optional[str] = None,
    config: Optional[SSHConfig] = None,
) -> Tuple[Optional[str], str]:
    """
    Download a TRELLIS output file from RunPod.
    """
    if local_dir is None:
        local_dir = "/srv/searidge_share/outputs/trellis"
    
    remote_path = f"/runpod-volume/outputs/trellis/{filename}"
    return download_from_runpod(remote_path, local_dir, config)


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RunPod SSH Download Utility")
    subparsers = parser.add_subparsers(dest="command")
    
    # Configure command
    config_parser = subparsers.add_parser("configure", help="Configure SSH connection")
    config_parser.add_argument("--host", required=True, help="RunPod pod IP/hostname")
    config_parser.add_argument("--user", default="root", help="SSH username")
    config_parser.add_argument("--port", type=int, default=22, help="SSH port")
    config_parser.add_argument("--key", required=True, help="Path to SSH private key")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test SSH connection")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List remote files")
    list_parser.add_argument("path", help="Remote directory path")
    
    # Download command
    dl_parser = subparsers.add_parser("download", help="Download a file")
    dl_parser.add_argument("remote_path", help="Remote file path")
    dl_parser.add_argument("local_dir", help="Local directory")
    
    args = parser.parse_args()
    
    if args.command == "configure":
        config = configure_ssh(
            host=args.host,
            user=args.user,
            port=args.port,
            key_path=args.key,
        )
        print(f"✅ SSH configured: {config.user}@{config.host}:{config.port}")
    
    elif args.command == "test":
        success, msg = test_ssh_connection()
        print(msg)
    
    elif args.command == "list":
        files, msg = list_remote_files(args.path)
        print(msg)
        for f in files:
            print(f"  {f}")
    
    elif args.command == "download":
        path, msg = download_from_runpod(args.remote_path, args.local_dir)
        print(msg)
        if path:
            print(f"  Saved to: {path}")
    
    else:
        parser.print_help()

