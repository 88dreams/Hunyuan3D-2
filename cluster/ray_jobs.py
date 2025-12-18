#!/usr/bin/env python3
"""
Ray-based distributed job execution for Hunyuan3D and GEN3C.

This module provides Ray remote functions that can execute generation jobs
on any node in the cluster. It automatically handles:
- GPU resource allocation
- Environment setup (ROCm, HIP)
- Model loading and caching
- Result transfer back to the head node

When Ray is not available, provides fallback stubs for single-system mode.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# RAY AVAILABILITY CHECK
# =============================================================================

_ray_available = False
_ray_initialized = False
ray = None

def _try_import_ray():
    """Attempt to import Ray and check if cluster is accessible."""
    global _ray_available, ray
    try:
        import ray as ray_module
        ray = ray_module
        _ray_available = True
        return True
    except ImportError:
        logger.warning("Ray not installed. Running in single-system mode.")
        return False

# Try to import Ray at module load
_try_import_ray()


def is_ray_available() -> bool:
    """Check if Ray is installed and importable."""
    return _ray_available


def is_ray_initialized() -> bool:
    """Check if Ray has been initialized and connected to a cluster."""
    global _ray_initialized
    if not _ray_available:
        return False
    try:
        return ray.is_initialized()
    except Exception:
        return False


def init_ray(address: str = "auto") -> bool:
    """
    Initialize Ray connection to the cluster.
    
    Args:
        address: Ray cluster address. "auto" to auto-detect, or specific address like "searidge02:6380"
    
    Returns:
        True if successfully connected, False otherwise
    """
    global _ray_initialized
    
    if not _ray_available:
        logger.warning("Ray not available, cannot initialize")
        return False
    
    if ray.is_initialized():
        logger.info("Ray already initialized")
        _ray_initialized = True
        return True
    
    try:
        ray.init(address=address, ignore_reinit_error=True)
        _ray_initialized = True
        logger.info(f"Ray initialized, connected to cluster at {address}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}")
        _ray_initialized = False
        return False


def shutdown_ray():
    """Shutdown Ray connection."""
    global _ray_initialized
    if _ray_available and ray.is_initialized():
        ray.shutdown()
        _ray_initialized = False


def get_cluster_status() -> Dict[str, Any]:
    """
    Get current Ray cluster status.
    
    Returns:
        Dictionary with cluster information including:
        - available: bool - whether cluster is accessible
        - nodes: list of node info
        - resources: total cluster resources
        - gpus: number of GPUs available
    """
    if not is_ray_initialized():
        return {
            "available": False,
            "mode": "single-system",
            "nodes": [],
            "resources": {},
            "gpus": 0,
            "message": "Ray not initialized. Running in single-system mode."
        }
    
    try:
        resources = ray.cluster_resources()
        nodes = ray.nodes()
        
        # Count alive nodes and GPUs
        alive_nodes = [n for n in nodes if n.get("Alive", False)]
        total_gpus = int(resources.get("GPU", 0))
        
        return {
            "available": True,
            "mode": "distributed",
            "nodes": [
                {
                    "node_id": n.get("NodeID", "unknown")[:8],
                    "hostname": n.get("NodeManagerHostname", "unknown"),
                    "alive": n.get("Alive", False),
                    "resources": n.get("Resources", {}),
                }
                for n in alive_nodes
            ],
            "resources": dict(resources),
            "gpus": total_gpus,
            "message": f"Cluster ready: {len(alive_nodes)} nodes, {total_gpus} GPUs"
        }
    except Exception as e:
        return {
            "available": False,
            "mode": "error",
            "nodes": [],
            "resources": {},
            "gpus": 0,
            "message": f"Error getting cluster status: {e}"
        }


# =============================================================================
# ROCM ENVIRONMENT SETUP
# =============================================================================

def _get_rocm_env() -> Dict[str, str]:
    """Get ROCm environment variables for AMD GPU execution."""
    try:
        from config import get_rocm_env
        return get_rocm_env()
    except ImportError:
        # Fallback defaults for RX 6900 XT
        return {
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "0"),
            "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
            "PYTORCH_ALLOC_CONF": "max_split_size_mb:512",
            "ROCM_HOME": "/opt/rocm",
            "CUDA_HOME": "/opt/rocm",
        }


def _setup_execution_env() -> Dict[str, str]:
    """Setup full execution environment for remote jobs."""
    env = os.environ.copy()
    env.update(_get_rocm_env())
    
    # Ensure config module is importable
    try:
        from config import get_path
        project_root = get_path('hunyuan_dir')
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"
    except ImportError:
        pass
    
    return env


# =============================================================================
# RAY REMOTE FUNCTIONS (only defined if Ray is available)
# =============================================================================

if _ray_available:
    
    @ray.remote(num_gpus=1)
    def ray_run_hunyuan(
        image_path: str,
        guidance_scale: float,
        steps: int,
        seed: Optional[int],
        model_choice: str,
        use_fp16: bool,
        attention_slicing: bool,
        cpu_offload: bool,
        remove_background: bool,
        output_name: str,
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Ray remote function for Hunyuan3D shape generation.
        
        Runs on a worker node with 1 GPU allocated.
        Returns dict with 'success', 'output_path', 'logs', 'error'.
        """
        import socket
        hostname = socket.gethostname()
        
        logs = [f"[Ray] Running on node: {hostname}"]
        
        try:
            # Setup environment
            env = _setup_execution_env()
            for k, v in env.items():
                if k.startswith(("HIP_", "HSA_", "PYTORCH_", "ROCM_", "CUDA_")):
                    os.environ[k] = v
            
            # Import generation code
            # We import here to ensure it happens on the worker node
            import torch
            import gc
            from PIL import Image
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            
            try:
                from config import get_path
                cache_dir = get_path('hf_cache_dir')
            except ImportError:
                cache_dir = "/srv/searidge_share/checkpoints/huggingface"
            
            logs.append(f"[Ray] Using cache: {cache_dir}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
            
            # Determine model and subfolder
            selected_model = "tencent/Hunyuan3D-2" if "Full Model" in model_choice else "tencent/Hunyuan3D-2mini"
            subfolder = "hunyuan3d-dit-v2-0" if "Full Model" in model_choice else "hunyuan3d-dit-v2-mini-turbo"
            
            logs.append(f"[Ray] Loading model: {selected_model}/{subfolder}")
            
            # Load pipeline
            dtype = torch.float16 if use_fp16 else torch.float32
            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                selected_model,
                subfolder=subfolder,
                cache_dir=cache_dir,
                torch_dtype=dtype,
            )
            
            # Apply memory optimizations
            if attention_slicing and hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
            if cpu_offload and hasattr(pipeline, "enable_sequential_cpu_offload"):
                pipeline.enable_sequential_cpu_offload()
            
            # Set seed
            if seed is not None:
                torch.manual_seed(int(seed))
            
            # Load and process image
            img = Image.open(image_path).convert("RGB")
            
            # Background removal if requested
            if remove_background:
                logs.append("[Ray] Removing background...")
                from hy3dgen.rembg import BackgroundRemover
                rembg = BackgroundRemover()
                img = rembg(img).convert("RGBA")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Run generation
            logs.append("[Ray] Running shape generation...")
            
            octree_res = 380 if "Mini Model" in model_choice else 360
            chunks = 6000 if "Mini Model" in model_choice else 5000
            
            result = pipeline(
                image=img,
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(steps),
                octree_resolution=octree_res,
                num_chunks=chunks,
            )
            
            mesh = result[0] if isinstance(result, (list, tuple)) else result
            
            # Save output
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(output_name)[0]
            output_path = os.path.join(output_dir, f"{base_name}_shape.glb")
            mesh.export(output_path)
            
            logs.append(f"[Ray] Saved: {output_path}")
            
            # Cleanup
            del pipeline, mesh
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                "success": True,
                "output_path": output_path,
                "logs": "\n".join(logs),
                "error": None,
                "node": hostname,
            }
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logs.append(f"[Ray] Error: {error_msg}")
            return {
                "success": False,
                "output_path": None,
                "logs": "\n".join(logs),
                "error": error_msg,
                "node": hostname,
            }
    
    
    @ray.remote(num_gpus=1)
    def ray_run_gen3c(
        image_path: str,
        guidance: float,
        frames: int,
        video_name: str,
        checkpoint_dir: str,
        output_dir: str,
        extra_args: str,
    ) -> Dict[str, Any]:
        """
        Ray remote function for GEN3C video generation.
        
        Runs on a worker node with 1 GPU allocated.
        Executes the run_gen3c.sh script as a subprocess.
        Returns dict with 'success', 'output_path', 'logs', 'error'.
        """
        import socket
        hostname = socket.gethostname()
        
        logs = [f"[Ray] Running GEN3C on node: {hostname}"]
        
        try:
            # Get paths from config
            try:
                from config import get_path
                project_root = get_path('hunyuan_dir')
            except ImportError:
                project_root = "/srv/searidge_share/projects/Hunyuan3D-2-Fork"
            
            gen3c_script = os.path.join(project_root, "scripts", "run_gen3c.sh")
            
            if not os.path.isfile(gen3c_script):
                raise FileNotFoundError(f"GEN3C script not found: {gen3c_script}")
            
            # Build command
            cmd = [
                "bash", gen3c_script,
                "--input", image_path,
                "--video-name", video_name,
                "--guidance", str(guidance),
                "--checkpoint-dir", checkpoint_dir,
                "--output-dir", output_dir,
            ]
            
            if frames:
                cmd.extend(["--frames", str(frames)])
            
            if extra_args and extra_args.strip():
                cmd.extend(["--extra", extra_args.strip()])
            
            logs.append(f"[Ray] Command: {' '.join(cmd)}")
            
            # Setup environment
            env = _setup_execution_env()
            
            # Run subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=project_root,
            )
            
            if result.stdout:
                logs.append(f"[Ray] STDOUT:\n{result.stdout}")
            if result.stderr:
                logs.append(f"[Ray] STDERR:\n{result.stderr}")
            
            if result.returncode != 0:
                raise RuntimeError(f"GEN3C exited with code {result.returncode}")
            
            # Check for output
            video_path = os.path.join(output_dir, f"{video_name}.mp4")
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")
            
            logs.append(f"[Ray] Generated: {video_path}")
            
            return {
                "success": True,
                "output_path": video_path,
                "logs": "\n".join(logs),
                "error": None,
                "node": hostname,
            }
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logs.append(f"[Ray] Error: {error_msg}")
            return {
                "success": False,
                "output_path": None,
                "logs": "\n".join(logs),
                "error": error_msg,
                "node": hostname,
            }


# =============================================================================
# LOCAL FALLBACK FUNCTIONS (when Ray is not available)
# =============================================================================

def local_run_hunyuan(
    image_path: str,
    guidance_scale: float,
    steps: int,
    seed: Optional[int],
    model_choice: str,
    use_fp16: bool,
    attention_slicing: bool,
    cpu_offload: bool,
    remove_background: bool,
    output_name: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Local execution of Hunyuan3D generation (single-system mode).
    
    This is the fallback when Ray is not available.
    """
    import socket
    hostname = socket.gethostname()
    logs = [f"[Local] Running on: {hostname}"]
    
    try:
        import torch
        import gc
        from PIL import Image
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        
        try:
            from config import get_path
            cache_dir = get_path('hf_cache_dir')
        except ImportError:
            cache_dir = "/srv/searidge_share/checkpoints/huggingface"
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
        
        # Determine model
        selected_model = "tencent/Hunyuan3D-2" if "Full Model" in model_choice else "tencent/Hunyuan3D-2mini"
        subfolder = "hunyuan3d-dit-v2-0" if "Full Model" in model_choice else "hunyuan3d-dit-v2-mini-turbo"
        
        logs.append(f"[Local] Loading: {selected_model}/{subfolder}")
        
        dtype = torch.float16 if use_fp16 else torch.float32
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            selected_model,
            subfolder=subfolder,
            cache_dir=cache_dir,
            torch_dtype=dtype,
        )
        
        if attention_slicing and hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()
        if cpu_offload and hasattr(pipeline, "enable_sequential_cpu_offload"):
            pipeline.enable_sequential_cpu_offload()
        
        if seed is not None:
            torch.manual_seed(int(seed))
        
        img = Image.open(image_path).convert("RGB")
        
        if remove_background:
            logs.append("[Local] Removing background...")
            from hy3dgen.rembg import BackgroundRemover
            rembg = BackgroundRemover()
            img = rembg(img).convert("RGBA")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logs.append("[Local] Generating shape...")
        
        octree_res = 380 if "Mini Model" in model_choice else 360
        chunks = 6000 if "Mini Model" in model_choice else 5000
        
        result = pipeline(
            image=img,
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(steps),
            octree_resolution=octree_res,
            num_chunks=chunks,
        )
        
        mesh = result[0] if isinstance(result, (list, tuple)) else result
        
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(output_name)[0]
        output_path = os.path.join(output_dir, f"{base_name}_shape.glb")
        mesh.export(output_path)
        
        logs.append(f"[Local] Saved: {output_path}")
        
        del pipeline, mesh
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "success": True,
            "output_path": output_path,
            "logs": "\n".join(logs),
            "error": None,
            "node": hostname,
        }
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logs.append(f"[Local] Error: {error_msg}")
        return {
            "success": False,
            "output_path": None,
            "logs": "\n".join(logs),
            "error": error_msg,
            "node": hostname,
        }


def local_run_gen3c(
    image_path: str,
    guidance: float,
    frames: int,
    video_name: str,
    checkpoint_dir: str,
    output_dir: str,
    extra_args: str,
) -> Dict[str, Any]:
    """
    Local execution of GEN3C generation (single-system mode).
    """
    import socket
    hostname = socket.gethostname()
    logs = [f"[Local] Running GEN3C on: {hostname}"]
    
    try:
        try:
            from config import get_path
            project_root = get_path('hunyuan_dir')
        except ImportError:
            project_root = "/srv/searidge_share/projects/Hunyuan3D-2-Fork"
        
        gen3c_script = os.path.join(project_root, "scripts", "run_gen3c.sh")
        
        if not os.path.isfile(gen3c_script):
            raise FileNotFoundError(f"GEN3C script not found: {gen3c_script}")
        
        cmd = [
            "bash", gen3c_script,
            "--input", image_path,
            "--video-name", video_name,
            "--guidance", str(guidance),
            "--checkpoint-dir", checkpoint_dir,
            "--output-dir", output_dir,
        ]
        
        if frames:
            cmd.extend(["--frames", str(frames)])
        
        if extra_args and extra_args.strip():
            cmd.extend(["--extra", extra_args.strip()])
        
        logs.append(f"[Local] Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        
        if result.stdout:
            logs.append(f"[Local] STDOUT:\n{result.stdout}")
        if result.stderr:
            logs.append(f"[Local] STDERR:\n{result.stderr}")
        
        if result.returncode != 0:
            raise RuntimeError(f"GEN3C exited with code {result.returncode}")
        
        video_path = os.path.join(output_dir, f"{video_name}.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        logs.append(f"[Local] Generated: {video_path}")
        
        return {
            "success": True,
            "output_path": video_path,
            "logs": "\n".join(logs),
            "error": None,
            "node": hostname,
        }
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logs.append(f"[Local] Error: {error_msg}")
        return {
            "success": False,
            "output_path": None,
            "logs": "\n".join(logs),
            "error": error_msg,
            "node": hostname,
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Ray Jobs Module Test")
    print("=" * 50)
    print(f"Ray available: {is_ray_available()}")
    
    if is_ray_available():
        print("\nAttempting to connect to Ray cluster...")
        if init_ray():
            status = get_cluster_status()
            print(f"\nCluster Status:")
            print(f"  Mode: {status['mode']}")
            print(f"  GPUs: {status['gpus']}")
            print(f"  Nodes: {len(status['nodes'])}")
            for node in status['nodes']:
                print(f"    - {node['hostname']} (alive={node['alive']})")
            shutdown_ray()
        else:
            print("Failed to connect to Ray cluster")
    else:
        print("\nRunning in single-system mode (Ray not installed)")

