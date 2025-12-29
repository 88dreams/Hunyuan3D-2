#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
3D Generation Studio - Multi-Model Gradio Interface

This module provides a unified web interface for multiple 3D generation models:
- Hunyuan3D: Image to GLB mesh generation
- GEN3C: Image to video generation
- Lyra: Image/Video to 3D Gaussian Splatting (coming soon)
- SHARP: Fast Gaussian splatting (coming soon)
- TRELLIS.2: High-quality 3D generation (coming soon)

Multi-System Configuration:
    All paths are loaded from config/multi_system.yaml to support distributed
    deployment across the searidge cluster (searidge01, searidge02, searidge03).

Execution Modes:
    - RunPod Serverless: Jobs run on RunPod cloud GPUs (recommended for GEN3C)
    - RunPod Pod: Jobs run on dedicated RunPod pod
    - Local CUDA: Jobs run on local GPU
    - Local Cluster (Ray): Jobs distributed across local cluster
"""
print("DEBUG: STARTING 3D GENERATION STUDIO v2.1 (Port 5683) - Refactored")

import os
import json
import time
from typing import Optional, Tuple, Union
from pathlib import Path

# Load AWS credentials from config file if not already set
def _load_aws_credentials():
    """Load AWS credentials from config file."""
    aws_creds_file = Path.home() / ".config" / "3d_studio" / "aws_credentials.env"
    if aws_creds_file.exists():
        with open(aws_creds_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Only set if not already in environment
                    if key not in os.environ or not os.environ[key]:
                        os.environ[key] = value
        print(f"[CONFIG] Loaded AWS credentials from {aws_creds_file}")
    else:
        print(f"[CONFIG] AWS credentials file not found: {aws_creds_file}")

_load_aws_credentials()

import torch  # type: ignore
import gradio as gr  # type: ignore

# =============================================================================
# LOCAL IMPORTS
# =============================================================================

from ui.styles import CUSTOM_CSS
from ui.tabs import (
    create_hunyuan_tab,
    create_gen3c_tab,
    create_lyra_tab,
    create_sharp_tab,
    create_trellis_tab,
    create_mesh_extraction_tab,
)

from generators import (
    run_hunyuan,
    run_hunyuan_runpod,
    check_hunyuan_runpod_status,
    run_gen3c_runpod,
    run_gen3c_serverless,
    run_gen3c_local,
    check_runpod_status,
    check_serverless_status,
    cancel_serverless_job,
    GEN3C_DEFAULT_OUTPUT_DIR,
    # SHARP
    run_sharp_local,
    run_sharp_runpod,
    check_sharp_installation,
    SHARP_DEFAULT_OUTPUT_DIR,
    # Lyra
    run_lyra_runpod,
    check_lyra_status,
    LYRA_DEFAULT_OUTPUT_DIR,
    # TRELLIS.2
    run_trellis_runpod,
    check_trellis_status,
    TRELLIS_DEFAULT_OUTPUT_DIR,
    # SuGaR (Mesh Extraction)
    run_sugar_extraction,
    run_tsdf_extraction,
    check_sugar_status,
    detect_ply_format,
    MESH_DEFAULT_OUTPUT_DIR,
)

from job_queue import (
    get_queue_display,
    clear_completed_jobs,
)

from utils import (
    clamp_scale_value,
    update_image_info_display,
    maybe_downscale_image,
    start_monitoring,
    get_system_metrics,
    format_system_metrics,
)


# =============================================================================
# CLUSTER / JOB MANAGER
# =============================================================================

_job_manager = None
_cluster_available = False

try:
    from cluster import JobManager, get_cluster_status
    _cluster_available = True
    print("[CLUSTER] Cluster module loaded successfully")
except ImportError as e:
    print(f"[CLUSTER] Warning: Cluster module not available ({e})")
    get_cluster_status = lambda: {"available": False, "mode": "local", "message": "Cluster module not installed"}


def _init_job_manager(use_ray: Optional[bool] = None) -> Optional['JobManager']:
    """Initialize the job manager with optional Ray support."""
    global _job_manager
    
    if not _cluster_available:
        print("[CLUSTER] Job manager not available (cluster module not installed)")
        return None
    
    try:
        if use_ray is None:
            try:
                from config import is_ray_enabled, should_auto_connect_ray
                use_ray = is_ray_enabled() and should_auto_connect_ray()
            except ImportError:
                use_ray = True
        
        _job_manager = JobManager(use_ray=use_ray)
        print(f"[CLUSTER] Job manager initialized in {_job_manager.mode} mode")
        return _job_manager
    except Exception as e:
        print(f"[CLUSTER] Failed to initialize job manager: {e}")
        return None


def format_cluster_status() -> str:
    """Format cluster status for display in UI."""
    if not _cluster_available:
        return "🔴 Cluster module not installed - Local mode only"
    
    if _job_manager is None:
        return "🟡 Job manager not initialized"
    
    status = _job_manager.cluster_status
    
    if status.get("available"):
        mode = status.get("mode", "unknown")
        gpus = status.get("gpus", 0)
        nodes = len(status.get("nodes", []))
        return f"🟢 {mode.title()} Mode | {nodes} nodes | {gpus} GPUs available"
    else:
        message = status.get("message", "Not connected")
        return f"🟡 Local Mode | {message}"


# =============================================================================
# PERSISTENT RUNPOD CONFIG
# =============================================================================

RUNPOD_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".runpod_config.json")


def _load_runpod_config() -> dict:
    """Load RunPod config from local file."""
    if os.path.exists(RUNPOD_CONFIG_FILE):
        try:
            with open(RUNPOD_CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_runpod_config(config: dict) -> None:
    """Save RunPod config to local file."""
    try:
        with open(RUNPOD_CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"[CONFIG] Warning: Could not save RunPod config: {e}")


def save_serverless_credentials(endpoint_id: str, api_key: str, model: str = "gen3c") -> Tuple[str, dict]:
    """Save serverless credentials to config file for a specific model."""
    config = _load_runpod_config()
    # Save per-model credentials
    config[f"{model}_endpoint_id"] = endpoint_id.strip() if endpoint_id else ""
    config[f"{model}_api_key"] = api_key.strip() if api_key else ""
    # Also save as default for backwards compatibility
    if model == "gen3c":
        config["serverless_endpoint_id"] = endpoint_id.strip() if endpoint_id else ""
        config["serverless_api_key"] = api_key.strip() if api_key else ""
    _save_runpod_config(config)
    return "✅ Credentials saved", gr.update(visible=False)


def save_pod_url(pod_url: str) -> str:
    """Save pod URL to config file."""
    config = _load_runpod_config()
    config["pod_url"] = pod_url.strip() if pod_url else ""
    _save_runpod_config(config)
    return "✅ Pod URL saved"


# Load saved config
_runpod_config = _load_runpod_config()
DEFAULT_RUNPOD_URL = _runpod_config.get("pod_url", "https://iv94zuokozefxc-8000.proxy.runpod.net")
# Legacy/default credentials (used by GEN3C/SHARP/Lyra)
DEFAULT_SERVERLESS_ENDPOINT = _runpod_config.get("serverless_endpoint_id", "")
DEFAULT_SERVERLESS_API_KEY = _runpod_config.get("serverless_api_key", "")
# Per-model credentials
DEFAULT_GEN3C_ENDPOINT = _runpod_config.get("gen3c_endpoint_id", DEFAULT_SERVERLESS_ENDPOINT)
DEFAULT_GEN3C_API_KEY = _runpod_config.get("gen3c_api_key", DEFAULT_SERVERLESS_API_KEY)
DEFAULT_SHARP_ENDPOINT = _runpod_config.get("sharp_endpoint_id", DEFAULT_SERVERLESS_ENDPOINT)
DEFAULT_SHARP_API_KEY = _runpod_config.get("sharp_api_key", DEFAULT_SERVERLESS_API_KEY)
DEFAULT_LYRA_ENDPOINT = _runpod_config.get("lyra_endpoint_id", DEFAULT_SERVERLESS_ENDPOINT)
DEFAULT_LYRA_API_KEY = _runpod_config.get("lyra_api_key", DEFAULT_SERVERLESS_API_KEY)
DEFAULT_TRELLIS_ENDPOINT = _runpod_config.get("trellis_endpoint_id", "")
DEFAULT_TRELLIS_API_KEY = _runpod_config.get("trellis_api_key", DEFAULT_SERVERLESS_API_KEY)
DEFAULT_HUNYUAN_ENDPOINT = _runpod_config.get("hunyuan_endpoint_id", "")
DEFAULT_HUNYUAN_API_KEY = _runpod_config.get("hunyuan_api_key", DEFAULT_SERVERLESS_API_KEY)
# Mesh extraction uses the same endpoint as GEN3C/SHARP/Lyra by default
DEFAULT_MESH_ENDPOINT = _runpod_config.get("mesh_extraction_endpoint_id", DEFAULT_GEN3C_ENDPOINT)
DEFAULT_MESH_API_KEY = _runpod_config.get("mesh_extraction_api_key", DEFAULT_SERVERLESS_API_KEY)

print(f"[CONFIG] Loaded RunPod credentials:")
print(f"  - GEN3C/SHARP/Lyra: endpoint={'set' if DEFAULT_GEN3C_ENDPOINT else 'not set'}")
print(f"  - TRELLIS: endpoint={'set' if DEFAULT_TRELLIS_ENDPOINT else 'not set'}")
print(f"  - Hunyuan: endpoint={'set' if DEFAULT_HUNYUAN_ENDPOINT else 'not set'}")
print(f"  - Mesh Extraction: endpoint={'set' if DEFAULT_MESH_ENDPOINT else 'not set'}")


# =============================================================================
# UNIFIED GENERATION HANDLERS
# =============================================================================

def handle_hunyuan_generation(
    image_path: Union[str, None],
    image_scale: Union[float, int],
    exec_mode: str,
    endpoint_id: str,
    api_key: str,
    guidance_scale: float,
    steps: int,
    seed: Optional[int],
    octree_resolution: int,
    model_choice: str,
    use_fp16: bool,
    attention_slicing: bool,
    cpu_offload: bool,
    remove_background: bool,
    output_name: str,
    save_location: str,
) -> Tuple[Optional[str], str, str]:
    """Handle Hunyuan3D generation (Local or RunPod)."""
    scale_value = clamp_scale_value(image_scale)
    scaled_path, temp_scaled = maybe_downscale_image(image_path, scale_value)
    effective_image_path = scaled_path or image_path

    try:
        if exec_mode == "RunPod Serverless":
            # RunPod execution
            return run_hunyuan_runpod(
                image_path=effective_image_path,
                endpoint_id=endpoint_id,
                api_key=api_key,
                model_choice=model_choice,
                guidance_scale=guidance_scale,
                steps=steps,
                octree_resolution=int(octree_resolution) if octree_resolution else 380,
                seed=int(seed) if seed else None,
                remove_background=remove_background,
                output_name=output_name,
                output_dir=save_location,
            )
        else:
            # Local execution
            return run_hunyuan(
                image_path=effective_image_path,
                guidance_scale=guidance_scale,
                steps=steps,
                seed=seed,
                model_choice=model_choice,
                use_fp16=use_fp16,
                attention_slicing=attention_slicing,
                cpu_offload=cpu_offload,
                remove_background=remove_background,
                output_name=output_name,
                save_location=save_location,
            )
    finally:
        if temp_scaled and os.path.exists(temp_scaled):
            try:
                os.remove(temp_scaled)
            except Exception:
                pass


def handle_gen3c_generation(
    image_path: Union[str, None],
    image_scale: Union[float, int],
    exec_mode: str,
    runpod_url: str,
    serverless_endpoint_id: str,
    serverless_api_key: str,
    guidance: float,
    frames: Union[str, int],
    trajectory: str,
    foreground_mask: bool,
    video_name: str,
    seed: Optional[int],
    checkpoint_dir: str,
    output_dir: str,
    extra_args: str,
) -> Tuple[Optional[str], str, str]:
    """Handle GEN3C generation based on execution mode."""
    scale_value = clamp_scale_value(image_scale)
    scaled_path, temp_scaled = maybe_downscale_image(image_path, scale_value)
    effective_image_path = scaled_path or image_path
    
    resolved_output_dir = output_dir.strip() if output_dir else GEN3C_DEFAULT_OUTPUT_DIR
    
    try:
        if exec_mode == "RunPod Pod":
            return run_gen3c_runpod(
                image_path=effective_image_path,
                runpod_url=runpod_url,
                guidance=guidance,
                frames=frames,
                trajectory=trajectory,
                foreground_masking=foreground_mask,
                video_name=video_name,
                seed=seed,
                output_dir=resolved_output_dir,
            )
        elif exec_mode == "RunPod Serverless":
            return run_gen3c_serverless(
                image_path=effective_image_path,
                endpoint_id=serverless_endpoint_id,
                api_key=serverless_api_key,
                guidance=guidance,
                frames=frames,
                trajectory=trajectory,
                foreground_masking=foreground_mask,
                video_name=video_name,
                seed=seed,
                output_dir=resolved_output_dir,
            )
        else:
            # Local execution
            extra_args_parts = []
            if trajectory:
                extra_args_parts.append(f"--trajectory {trajectory}")
            if foreground_mask:
                extra_args_parts.append("--foreground_masking")
            if extra_args and extra_args.strip():
                extra_args_parts.append(extra_args.strip())
            
            combined_extra_args = " ".join(extra_args_parts)
            
            return run_gen3c_local(
                image_path=effective_image_path,
                guidance=guidance,
                frames=frames,
                video_name=video_name,
                checkpoint_dir=checkpoint_dir,
                output_dir=resolved_output_dir,
                extra_args=combined_extra_args,
            )
    finally:
        if temp_scaled and os.path.exists(temp_scaled):
            try:
                os.remove(temp_scaled)
            except Exception:
                pass


def handle_sharp_generation(
    image_path: Union[str, None],
    image_scale: Union[float, int],
    exec_mode: str,
    endpoint_id: str,
    api_key: str,
    render_video: bool,
    output_name: str,
    output_dir: str,
) -> Tuple[Optional[str], str, str]:
    """Handle SHARP generation based on execution mode."""
    scale_value = clamp_scale_value(image_scale)
    scaled_path, temp_scaled = maybe_downscale_image(image_path, scale_value)
    effective_image_path = scaled_path or image_path
    
    try:
        if "RunPod" in exec_mode:
            return run_sharp_runpod(
                image_path=effective_image_path,
                endpoint_id=endpoint_id,
                api_key=api_key,
                output_name=output_name,
                output_dir=output_dir,
                render_video=render_video,
            )
        else:
            # Local mode - video rendering not supported without CUDA
            if render_video:
                return None, "⚠️ Video rendering requires RunPod (CUDA GPU). Generating PLY only.", "⚠️ Video rendering not available locally"
            return run_sharp_local(
                image_path=effective_image_path,
                output_name=output_name,
                output_dir=output_dir,
                render_video=False,
            )
    finally:
        if temp_scaled and os.path.exists(temp_scaled):
            try:
                os.remove(temp_scaled)
            except Exception:
                pass


def handle_lyra_generation(
    image_path: Union[str, None],
    video_path: Union[str, None],
    image_scale: Union[float, int],
    exec_mode: str,
    endpoint_id: str,
    api_key: str,
    generation_mode: str,
    num_views: int,
    camera_motion: float,
    multi_trajectory: bool,
    foreground_masking: bool,
    num_gaussians: int,
    seed: Optional[int],
    output_name: str,
    output_dir: str,
    output_ply: bool,
    output_video: bool,
) -> Tuple[Optional[str], str, str, str]:
    """Handle Lyra generation (RunPod only).
    
    Returns:
        Tuple of (output_path, logs, progress, ply_path_for_conversion)
    """
    scale_value = clamp_scale_value(image_scale)
    
    # Lyra can use image or video depending on mode
    is_static = "Static" in generation_mode
    
    if is_static:
        scaled_path, temp_scaled = maybe_downscale_image(image_path, scale_value)
        effective_image_path = scaled_path or image_path
        effective_video_path = None
    else:
        effective_image_path = None
        effective_video_path = video_path
        temp_scaled = None
    
    try:
        result = run_lyra_runpod(
            image_path=effective_image_path,
            video_path=effective_video_path,
            endpoint_id=endpoint_id,
            api_key=api_key,
            generation_mode=generation_mode,
            num_views=num_views,
            camera_motion=camera_motion,
            multi_trajectory=multi_trajectory,
            foreground_masking=foreground_masking,
            num_gaussians=num_gaussians,
            seed=int(seed) if seed else None,
            output_name=output_name,
            output_dir=output_dir,
            output_ply=output_ply,
            output_video=output_video,
        )
        
        # result is (output_path, logs, progress)
        output_path, logs, progress = result
        
        # Extract PLY path for conversion field (if it's a PLY file)
        ply_path = ""
        if output_path and output_path.endswith(".ply"):
            ply_path = output_path
        
        return output_path, logs, progress, ply_path
        
    finally:
        if temp_scaled and os.path.exists(temp_scaled):
            try:
                os.remove(temp_scaled)
            except Exception:
                pass


def handle_trellis_generation(
    image_path: Union[str, None],
    image_scale: Union[float, int],
    exec_mode: str,
    endpoint_id: str,
    api_key: str,
    resolution: str,
    guidance_scale: float,
    seed: Optional[int],
    output_name: str,
    output_dir: str,
    output_format: str,
) -> Tuple[Optional[str], str, str]:
    """Handle TRELLIS.2 generation (RunPod only)."""
    scale_value = clamp_scale_value(image_scale)
    scaled_path, temp_scaled = maybe_downscale_image(image_path, scale_value)
    effective_image_path = scaled_path or image_path
    
    try:
        return run_trellis_runpod(
            image_path=effective_image_path,
            endpoint_id=endpoint_id,
            api_key=api_key,
            resolution=resolution,
            guidance_scale=guidance_scale,
            seed=int(seed) if seed else None,
            output_name=output_name,
            output_dir=output_dir,
            output_format=output_format,
        )
    finally:
        if temp_scaled and os.path.exists(temp_scaled):
            try:
                os.remove(temp_scaled)
            except Exception:
                pass


def generate_blender_script(
    ply_path: str,
    display_mode: str,
    point_size: float,
    max_points: int,
) -> str:
    """
    Generate a customized Blender import script for the given PLY file.
    
    Args:
        ply_path: Path to the point cloud PLY file
        display_mode: "points", "spheres", or "cubes"
        point_size: Size of spheres/cubes
        max_points: Maximum points to import (0 = all)
    
    Returns:
        Status message with path to generated script
    """
    from pathlib import Path
    
    if not ply_path or not ply_path.strip():
        return "❌ Error: No PLY path provided"
    
    ply_path = ply_path.strip()
    if not os.path.exists(ply_path):
        return f"❌ Error: PLY file not found: {ply_path}"
    
    # Read the template script
    template_path = os.path.join(os.path.dirname(__file__), "scripts", "blender_import_pointcloud.py")
    if not os.path.exists(template_path):
        return f"❌ Error: Template script not found: {template_path}"
    
    with open(template_path, 'r') as f:
        script_content = f.read()
    
    # Customize the script with user settings
    max_points_value = "None" if max_points == 0 else str(int(max_points))
    
    # Replace configuration values
    script_content = script_content.replace(
        'PLY_PATH = "/srv/searidge_share/outputs/lyra/bright23_h200_lyra_converted.ply"',
        f'PLY_PATH = "{ply_path}"'
    )
    script_content = script_content.replace(
        'DISPLAY_MODE = "points"',
        f'DISPLAY_MODE = "{display_mode}"'
    )
    script_content = script_content.replace(
        'POINT_SIZE = 0.01',
        f'POINT_SIZE = {point_size}'
    )
    script_content = script_content.replace(
        'MAX_POINTS = None',
        f'MAX_POINTS = {max_points_value}'
    )
    
    # Write customized script
    ply_file = Path(ply_path)
    output_script = ply_file.parent / f"{ply_file.stem}_blender_import.py"
    
    with open(output_script, 'w') as f:
        f.write(script_content)
    
    return f"✅ Script generated: {output_script}\n\nTo use:\n1. Open Blender\n2. Go to Scripting workspace\n3. Open this script\n4. Press Alt+P to run"


def open_in_blender(
    ply_path: str,
    display_mode: str,
    point_size: float,
    max_points: int,
) -> str:
    """
    Generate script and attempt to open Blender with it.
    
    Returns:
        Status message
    """
    import subprocess
    from pathlib import Path
    
    # First generate the script
    result = generate_blender_script(ply_path, display_mode, point_size, max_points)
    
    if result.startswith("❌"):
        return result
    
    # Extract script path from result
    script_path = result.split(": ")[1].split("\n")[0]
    
    # Try to find and run Blender
    blender_paths = [
        "/usr/bin/blender",
        "/snap/bin/blender",
        "/usr/local/bin/blender",
        os.path.expanduser("~/blender/blender"),
    ]
    
    blender_exe = None
    for path in blender_paths:
        if os.path.exists(path):
            blender_exe = path
            break
    
    if not blender_exe:
        # Try to find via 'which'
        try:
            result_which = subprocess.run(["which", "blender"], capture_output=True, text=True)
            if result_which.returncode == 0:
                blender_exe = result_which.stdout.strip()
        except Exception:
            pass
    
    if not blender_exe:
        return f"✅ Script generated: {script_path}\n\n⚠️ Blender not found in PATH. Please open Blender manually and run the script."
    
    # Launch Blender with the script
    try:
        subprocess.Popen([blender_exe, "--python", script_path])
        return f"✅ Blender launched with script: {script_path}"
    except Exception as e:
        return f"✅ Script generated: {script_path}\n\n⚠️ Failed to launch Blender: {e}"


def handle_lyra_ply_conversion(
    input_path: str,
    convert_pointcloud: bool,
    convert_3dgs: bool,
    max_points: float,
    min_opacity: float,
    preset: str,
) -> Tuple[str, str]:
    """
    Convert Lyra's PyTorch PLY format to standard formats.
    
    Args:
        input_path: Path to Lyra's raw .ply output (PyTorch tensor format)
        convert_pointcloud: If True, convert to simple point cloud (MeshLab/Blender)
        convert_3dgs: If True, convert to full 3DGS format (SuperSplat)
        max_points: Maximum number of points (0 = no limit)
        min_opacity: Minimum opacity threshold (0 = no filter)
        preset: Preset name for quick settings
    
    Returns:
        Tuple of (status_message, pointcloud_path_for_blender)
    """
    import subprocess
    from pathlib import Path
    
    if not input_path or not input_path.strip():
        return "❌ Error: No input PLY path provided", ""
    
    input_path = input_path.strip()
    if not os.path.exists(input_path):
        return f"❌ Error: Input file not found: {input_path}", ""
    
    if not convert_pointcloud and not convert_3dgs:
        return "❌ Error: Select at least one output format", ""
    
    # Get the conversion script path
    script_path = os.path.join(os.path.dirname(__file__), "scripts", "convert_lyra_ply.py")
    if not os.path.exists(script_path):
        return f"❌ Error: Conversion script not found: {script_path}", ""
    
    # Apply preset settings (override sliders if not "Custom")
    if preset == "Full Quality":
        max_points = 0
        min_opacity = 0
    elif preset == "Web Viewer (500k)":
        max_points = 500000
        min_opacity = 0.01
    elif preset == "Quick Preview (100k)":
        max_points = 100000
        min_opacity = 0.05
    # "Custom" uses the slider values directly
    
    input_file = Path(input_path)
    output_dir = input_file.parent
    base_name = input_file.stem
    
    # Build suffix based on settings
    suffix = ""
    if max_points > 0:
        suffix += f"_{int(max_points)//1000}k"
    if min_opacity > 0:
        suffix += f"_op{int(min_opacity*100)}"
    
    results = []
    pointcloud_path = ""  # For Blender import
    
    # Build common arguments
    common_args = []
    if max_points > 0:
        common_args.extend(["--max-points", str(int(max_points))])
    if min_opacity > 0:
        common_args.extend(["--min-opacity", str(min_opacity)])
    
    # Convert to 3DGS format (default)
    if convert_3dgs:
        output_3dgs = output_dir / f"{base_name}_3dgs{suffix}.ply"
        try:
            cmd = ["python", script_path, str(input_path), str(output_3dgs)] + common_args
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # Increased timeout for large files
            )
            if result.returncode == 0:
                # Extract point count from output
                output_lines = result.stdout.strip().split('\n')
                point_info = [l for l in output_lines if 'Gaussians' in l or 'points' in l]
                point_summary = point_info[-1] if point_info else ""
                results.append(f"✅ 3DGS: {output_3dgs.name}")
                if point_summary:
                    results.append(f"   {point_summary}")
            else:
                results.append(f"❌ 3DGS failed: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            results.append("❌ 3DGS conversion timed out (>5 min)")
        except Exception as e:
            results.append(f"❌ 3DGS error: {e}")
    
    # Convert to simple point cloud
    if convert_pointcloud:
        output_pc = output_dir / f"{base_name}_pointcloud{suffix}.ply"
        try:
            cmd = ["python", script_path, str(input_path), str(output_pc), "--simple"] + common_args
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                output_lines = result.stdout.strip().split('\n')
                point_info = [l for l in output_lines if 'points' in l]
                point_summary = point_info[-1] if point_info else ""
                results.append(f"✅ Point cloud: {output_pc.name}")
                if point_summary:
                    results.append(f"   {point_summary}")
                pointcloud_path = str(output_pc)  # Set for Blender import
            else:
                results.append(f"❌ Point cloud failed: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            results.append("❌ Point cloud conversion timed out (>5 min)")
        except Exception as e:
            results.append(f"❌ Point cloud error: {e}")
    
    status = "\n".join(results) if results else "No conversions performed"
    return status, pointcloud_path


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

with gr.Blocks(title="3D Generation Studio") as demo:
    
    # Header
    gr.Markdown("""
    # 3D Generation Studio
    **Multi-model 3D generation from images and videos**
    """)

    with gr.Row():
        # =================================================================
        # LEFT COLUMN - Input, Output, Queue, System
        # =================================================================
        with gr.Column(scale=1):
            
            # --- Input Upload Area ---
            with gr.Group():
                gr.Markdown("### Input")
                input_image = gr.Image(
                    type="filepath",
                    label="Input Image",
                    height=250,
                )
                input_video = gr.Video(
                    label="Input Video",
                    visible=False,
                    height=250,
                )
            
            # --- Input Parameters ---
            with gr.Group():
                gr.Markdown("### Input Parameters")
                with gr.Row():
                    input_type_selector = gr.Dropdown(
                        choices=["Image"],
                        value="Image",
                        label="Input Type",
                        scale=1,
                        interactive=False,
                    )
                    image_scale_slider = gr.Slider(
                        minimum=0.25,
                        maximum=1.0,
                        value=1.0,
                        step=0.05,
                        label="Scale",
                        scale=2,
                    )
                image_resolution_display = gr.Textbox(
                    label="Resolution Info",
                    value="No image loaded.",
                    interactive=False,
                    lines=1,
                )
            
            # --- Output Viewer ---
            with gr.Group():
                gr.Markdown("### Output")
                # Note: Model3D viewer removed - PLY files not supported
                # Files are saved to output directory and can be viewed in external tools
                output_model_viewer = gr.Textbox(
                    label="Output File",
                    value="Output file path will appear here",
                    interactive=False,
                    lines=2,
                )
                output_video_player = gr.Video(
                    label="Video Output",
                    height=300,
                    visible=False,
                )
            
            # --- Job Queue ---
            with gr.Group():
                with gr.Row():
                    gr.Markdown("### Job Queue")
                    clear_queue_btn = gr.Button("Clear", size="sm", scale=0)
                
                job_queue_display = gr.Dataframe(
                    headers=["ID", "Model", "Status", "Input"],
                    datatype=["str", "str", "str", "str"],
                    value=[],
                    row_count=(3, "dynamic"),
                    interactive=False,
                )
            
            # --- System Resources ---
            with gr.Group():
                gr.Markdown("### System Resources")
                system_metrics_display = gr.Textbox(
                    value="Loading system metrics...",
                    interactive=False,
                    lines=2,
                    show_label=False,
                )

        # =================================================================
        # RIGHT COLUMN - Model Tabs and Settings
        # =================================================================
        with gr.Column(scale=1):
            
            with gr.Tabs() as model_tabs:
                
                # TAB 1: HUNYUAN 3D
                with gr.TabItem("Hunyuan 3D", id="hunyuan"):
                    hunyuan = create_hunyuan_tab(
                        default_endpoint_id=DEFAULT_HUNYUAN_ENDPOINT,
                        default_api_key=DEFAULT_HUNYUAN_API_KEY,
                    )
                
                # TAB 2: GEN3C VIDEO
                with gr.TabItem("GEN3C Video", id="gen3c"):
                    gen3c = create_gen3c_tab(
                        default_runpod_url=DEFAULT_RUNPOD_URL,
                        default_endpoint_id=DEFAULT_GEN3C_ENDPOINT,
                        default_api_key=DEFAULT_GEN3C_API_KEY,
                        format_cluster_status_fn=format_cluster_status,
                    )
                
                # TAB 3: LYRA 3DGS
                with gr.TabItem("Lyra 3DGS", id="lyra"):
                    lyra = create_lyra_tab(
                        default_endpoint_id=DEFAULT_LYRA_ENDPOINT,
                        default_api_key=DEFAULT_LYRA_API_KEY,
                    )
                
                # TAB 4: SHARP
                with gr.TabItem("SHARP", id="sharp"):
                    sharp = create_sharp_tab(
                        default_endpoint_id=DEFAULT_SHARP_ENDPOINT,
                        default_api_key=DEFAULT_SHARP_API_KEY,
                    )
                
                # TAB 5: TRELLIS.2
                with gr.TabItem("TRELLIS.2", id="trellis"):
                    trellis = create_trellis_tab(
                        default_endpoint_id=DEFAULT_TRELLIS_ENDPOINT,
                        default_api_key=DEFAULT_TRELLIS_API_KEY,
                    )
                
                # TAB 6: MESH EXTRACTION (SuGaR)
                mesh_extraction = create_mesh_extraction_tab(
                    default_endpoint_id=DEFAULT_MESH_ENDPOINT,
                    default_api_key=DEFAULT_MESH_API_KEY,
                )
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    # --- Image Preview Updates ---
    def update_resolution_on_image_change(image_path, scale):
        return update_image_info_display(image_path, scale)
    
    input_image.change(
        fn=update_resolution_on_image_change,
        inputs=[input_image, image_scale_slider],
        outputs=[image_resolution_display],
    )
    
    image_scale_slider.change(
        fn=update_resolution_on_image_change,
        inputs=[input_image, image_scale_slider],
        outputs=[image_resolution_display],
    )
    
    # --- GEN3C Mode Toggle ---
    def gen3c_mode_change(mode: str):
        """Toggle visibility of GEN3C execution mode settings."""
        show_pod = mode == "RunPod Pod"
        show_serverless = mode == "RunPod Serverless"
        show_local = mode == "Local Cluster"
        
        if show_pod:
            status = "Enter Pod URL and click Check Status"
        elif show_serverless:
            if DEFAULT_SERVERLESS_ENDPOINT and DEFAULT_SERVERLESS_API_KEY:
                status = "✅ Credentials loaded - Click Check Status"
            else:
                status = "⚠️ Enter credentials below"
        else:
            status = format_cluster_status()
        
        return (
            gr.update(visible=show_pod),
            gr.update(visible=show_serverless),
            gr.update(visible=show_local),
            status,
        )
    
    gen3c["exec_mode"].change(
        fn=gen3c_mode_change,
        inputs=[gen3c["exec_mode"]],
        outputs=[gen3c["pod_settings"], gen3c["serverless_settings"], gen3c["local_settings"], gen3c["status"]],
    )
    
    # --- GEN3C Status Checks ---
    gen3c["check_pod_btn"].click(
        fn=check_runpod_status,
        inputs=[gen3c["runpod_url"]],
        outputs=[gen3c["status"]],
    )
    
    gen3c["save_pod_btn"].click(
        fn=save_pod_url,
        inputs=[gen3c["runpod_url"]],
        outputs=[gen3c["status"]],
    )
    
    gen3c["check_serverless_btn"].click(
        fn=check_serverless_status,
        inputs=[gen3c["endpoint_id"], gen3c["api_key"]],
        outputs=[gen3c["status"]],
    )
    
    gen3c["save_creds_btn"].click(
        fn=lambda eid, key: save_serverless_credentials(eid, key, "gen3c"),
        inputs=[gen3c["endpoint_id"], gen3c["api_key"]],
        outputs=[gen3c["status"], gen3c["creds_group"]],
    )
    
    gen3c["cancel_btn"].click(
        fn=cancel_serverless_job,
        inputs=[gen3c["endpoint_id"], gen3c["api_key"], gen3c["current_job_id"]],
        outputs=[gen3c["status"], gen3c["current_job_id"]],
    )
    
    def toggle_creds_visibility():
        return gr.update(visible=True)
    
    gen3c["edit_creds_btn"].click(
        fn=toggle_creds_visibility,
        outputs=[gen3c["creds_group"]],
    )
    
    gen3c["check_cluster_btn"].click(
        fn=format_cluster_status,
        outputs=[gen3c["status"]],
    )
    
    # --- Hunyuan Generation ---
    hunyuan["generate_btn"].click(
        fn=handle_hunyuan_generation,
        inputs=[
            input_image, image_scale_slider, hunyuan["exec_mode"],
            hunyuan["endpoint_id"], hunyuan["api_key"],
            hunyuan["guidance"], hunyuan["steps"], hunyuan["seed"], hunyuan["octree_resolution"],
            hunyuan["model_choice"],
            hunyuan["fp16"], hunyuan["attention_slicing"], hunyuan["cpu_offload"], hunyuan["remove_bg"],
            hunyuan["output_name"], hunyuan["save_location"],
        ],
        outputs=[output_model_viewer, hunyuan["logs_box"], hunyuan["progress_display"]],
    )
    
    # --- Hunyuan Check Status ---
    def check_hunyuan_status(endpoint_id, api_key):
        return check_hunyuan_runpod_status(endpoint_id, api_key)
    
    hunyuan["check_serverless_btn"].click(
        fn=check_hunyuan_status,
        inputs=[hunyuan["endpoint_id"], hunyuan["api_key"]],
        outputs=[hunyuan["status"]],
    )
    
    # --- Hunyuan Edit Credentials Toggle ---
    hunyuan["edit_creds_btn"].click(
        fn=lambda: gr.update(visible=True),
        outputs=[hunyuan["creds_group"]],
    )
    
    hunyuan["save_creds_btn"].click(
        fn=lambda eid, key: save_serverless_credentials(eid, key, "hunyuan"),
        inputs=[hunyuan["endpoint_id"], hunyuan["api_key"]],
        outputs=[hunyuan["status"], hunyuan["creds_group"]],
    )
    
    # --- GEN3C Generation ---
    gen3c["generate_btn"].click(
        fn=handle_gen3c_generation,
        inputs=[
            input_image, image_scale_slider, gen3c["exec_mode"],
            gen3c["runpod_url"], gen3c["endpoint_id"], gen3c["api_key"],
            gen3c["guidance"], gen3c["frames"], gen3c["trajectory"], gen3c["foreground_mask"],
            gen3c["video_name"], gen3c["seed"], gen3c["checkpoint_dir"], gen3c["output_dir"], gen3c["extra_args"],
        ],
        outputs=[output_video_player, gen3c["logs_box"], gen3c["progress_display"]],
    )
    
    # --- SHARP Mode Toggle ---
    def sharp_mode_change(mode: str):
        """Toggle visibility of SHARP execution mode settings."""
        show_runpod = "RunPod" in mode
        if show_runpod:
            if DEFAULT_SERVERLESS_ENDPOINT and DEFAULT_SERVERLESS_API_KEY:
                status = "✅ Credentials loaded - Click Check Connection"
            else:
                status = "⚠️ Enter RunPod credentials below"
        else:
            status = check_sharp_installation()
        return gr.update(visible=show_runpod), status
    
    sharp["exec_mode"].change(
        fn=sharp_mode_change,
        inputs=[sharp["exec_mode"]],
        outputs=[sharp["creds_group"], sharp["status"]],
    )
    
    # --- SHARP Status Checks ---
    sharp["check_local_btn"].click(
        fn=check_sharp_installation,
        outputs=[sharp["local_status"]],
    )
    
    sharp["check_serverless_btn"].click(
        fn=check_serverless_status,
        inputs=[sharp["endpoint_id"], sharp["api_key"]],
        outputs=[sharp["status"]],
    )
    
    sharp["save_creds_btn"].click(
        fn=lambda eid, key: save_serverless_credentials(eid, key, "sharp"),
        inputs=[sharp["endpoint_id"], sharp["api_key"]],
        outputs=[sharp["status"], sharp["creds_group"]],
    )
    
    sharp["edit_creds_btn"].click(
        fn=lambda: gr.update(visible=True),
        outputs=[sharp["creds_group"]],
    )
    
    # --- SHARP Generation ---
    sharp["generate_btn"].click(
        fn=handle_sharp_generation,
        inputs=[
            input_image, image_scale_slider, sharp["exec_mode"],
            sharp["endpoint_id"], sharp["api_key"],
            sharp["render_video"], sharp["output_name"], sharp["output_dir"],
        ],
        outputs=[output_model_viewer, sharp["logs_box"], sharp["progress_display"]],
    )
    
    # --- Lyra Status Checks ---
    lyra["check_serverless_btn"].click(
        fn=lambda ep, key: check_lyra_status(ep, key),
        inputs=[lyra["endpoint_id"], lyra["api_key"]],
        outputs=[lyra["status"]],
    )
    
    lyra["save_creds_btn"].click(
        fn=lambda eid, key: save_serverless_credentials(eid, key, "lyra"),
        inputs=[lyra["endpoint_id"], lyra["api_key"]],
        outputs=[lyra["status"], lyra["creds_group"]],
    )
    
    lyra["edit_creds_btn"].click(
        fn=lambda: gr.update(visible=True),
        outputs=[lyra["creds_group"]],
    )
    
    # --- Lyra Generation ---
    lyra["generate_btn"].click(
        fn=handle_lyra_generation,
        inputs=[
            input_image, input_video, image_scale_slider, lyra["exec_mode"],
            lyra["endpoint_id"], lyra["api_key"],
            lyra["generation_mode"], lyra["num_views"], lyra["camera_motion"],
            lyra["multi_trajectory"], lyra["foreground_masking"],
            lyra["num_gaussians"], lyra["seed"],
            lyra["output_name"], lyra["output_dir"],
            lyra["output_ply"], lyra["output_video"],
        ],
        outputs=[output_model_viewer, lyra["logs_box"], lyra["progress_display"], lyra["ply_input_path"]],
    )
    
    # --- Lyra PLY Conversion ---
    lyra["convert_btn"].click(
        fn=handle_lyra_ply_conversion,
        inputs=[
            lyra["ply_input_path"],
            lyra["convert_pointcloud"],
            lyra["convert_3dgs"],
            lyra["downsample_max_points"],
            lyra["downsample_min_opacity"],
            lyra["downsample_presets"],
        ],
        outputs=[lyra["convert_status"], lyra["blender_ply_path"]],
    )
    
    # --- Lyra Preset Updates (sync sliders with preset selection) ---
    def update_downsample_from_preset(preset):
        """Update slider values when preset changes."""
        if preset == "Full Quality":
            return 0, 0.0
        elif preset == "Web Viewer (500k)":
            return 500000, 0.01
        elif preset == "Quick Preview (100k)":
            return 100000, 0.05
        else:  # Custom
            return gr.update(), gr.update()  # Keep current values
    
    lyra["downsample_presets"].change(
        fn=update_downsample_from_preset,
        inputs=[lyra["downsample_presets"]],
        outputs=[lyra["downsample_max_points"], lyra["downsample_min_opacity"]],
    )
    
    # --- Lyra Blender Import ---
    lyra["blender_generate_btn"].click(
        fn=generate_blender_script,
        inputs=[
            lyra["blender_ply_path"],
            lyra["blender_display_mode"],
            lyra["blender_point_size"],
            lyra["blender_max_points"],
        ],
        outputs=[lyra["blender_script_output"]],
    )
    
    lyra["blender_open_btn"].click(
        fn=open_in_blender,
        inputs=[
            lyra["blender_ply_path"],
            lyra["blender_display_mode"],
            lyra["blender_point_size"],
            lyra["blender_max_points"],
        ],
        outputs=[lyra["blender_script_output"]],
    )
    
    # --- Lyra PLY Dropdown (Post-Process tab) ---
    from ui.tabs.lyra_tab import scan_for_lyra_ply_files
    
    def refresh_lyra_ply_dropdown():
        """Refresh the list of available PLY files for Lyra post-processing."""
        files = scan_for_lyra_ply_files()
        return gr.update(choices=files)
    
    def on_lyra_ply_selected(selection: str):
        """Extract path from dropdown selection (removes size info)."""
        if selection and " (" in selection:
            # Remove the " (X.X MB)" suffix
            return selection.rsplit(" (", 1)[0]
        return selection or ""
    
    lyra["refresh_ply_btn"].click(
        fn=refresh_lyra_ply_dropdown,
        outputs=[lyra["ply_dropdown"]],
    )
    
    lyra["ply_dropdown"].change(
        fn=on_lyra_ply_selected,
        inputs=[lyra["ply_dropdown"]],
        outputs=[lyra["ply_input_path"]],
    )
    
    # --- TRELLIS.2 Status Checks ---
    trellis["check_serverless_btn"].click(
        fn=lambda ep, key: check_trellis_status(ep, key),
        inputs=[trellis["endpoint_id"], trellis["api_key"]],
        outputs=[trellis["status"]],
    )
    
    trellis["save_creds_btn"].click(
        fn=lambda eid, key: save_serverless_credentials(eid, key, "trellis"),
        inputs=[trellis["endpoint_id"], trellis["api_key"]],
        outputs=[trellis["status"], trellis["creds_group"]],
    )
    
    trellis["edit_creds_btn"].click(
        fn=lambda: gr.update(visible=True),
        outputs=[trellis["creds_group"]],
    )
    
    # --- TRELLIS.2 Generation ---
    trellis["generate_btn"].click(
        fn=handle_trellis_generation,
        inputs=[
            input_image, image_scale_slider, trellis["exec_mode"],
            trellis["endpoint_id"], trellis["api_key"],
            trellis["resolution"], trellis["guidance_scale"], trellis["seed"],
            trellis["output_name"], trellis["output_dir"], trellis["output_format"],
        ],
        outputs=[output_model_viewer, trellis["logs_box"], trellis["progress_display"]],
    )
    
    # =========================================================================
    # MESH EXTRACTION (SuGaR) EVENT HANDLERS
    # =========================================================================
    
    def handle_mesh_extraction(
        input_ply: str,
        input_format: str,
        method: str,
        regularization: str,
        quality_preset: str,
        poisson_depth: int,
        decimate_faces: int,
        export_texture: bool,
        texture_resolution: str,
        refinement_time: str,
        tsdf_voxel_size: float,
        tsdf_num_views: int,
        output_name: str,
        output_format: str,
        output_dir: str,
        endpoint_id: str,
        api_key: str,
    ):
        """Handle mesh extraction request."""
        if not input_ply or not input_ply.strip():
            return "", "Error: No input PLY file specified", "❌ Missing input"
        
        if "SuGaR" in method:
            result = run_sugar_extraction(
                input_ply=input_ply,
                input_format=input_format,
                regularization=regularization,
                quality_preset=quality_preset,
                poisson_depth=poisson_depth,
                decimate_faces=decimate_faces,
                export_texture=export_texture,
                texture_resolution=int(texture_resolution),
                refinement_time=refinement_time,
                output_name=output_name,
                output_format=output_format,
                output_dir=output_dir,
                endpoint_id=endpoint_id,
                api_key=api_key,
            )
        else:
            # TSDF method
            result = run_tsdf_extraction(
                input_ply=input_ply,
                input_format=input_format,
                voxel_size=tsdf_voxel_size,
                num_views=tsdf_num_views,
                output_name=output_name,
                output_dir=output_dir,
                endpoint_id=endpoint_id,
                api_key=api_key,
            )
        
        output_path, logs, status = result
        return output_path or "", logs, status
    
    # Mesh extraction button
    mesh_extraction["extract_btn"].click(
        fn=handle_mesh_extraction,
        inputs=[
            mesh_extraction["input_ply"],
            mesh_extraction["input_format"],
            mesh_extraction["method"],
            mesh_extraction["regularization"],
            mesh_extraction["quality_preset"],
            mesh_extraction["poisson_depth"],
            mesh_extraction["decimate_faces"],
            mesh_extraction["export_texture"],
            mesh_extraction["texture_resolution"],
            mesh_extraction["refinement_time"],
            mesh_extraction["tsdf_voxel_size"],
            mesh_extraction["tsdf_num_views"],
            mesh_extraction["output_name"],
            mesh_extraction["output_format"],
            mesh_extraction["output_dir"],
            mesh_extraction["endpoint_id"],
            mesh_extraction["api_key"],
        ],
        outputs=[
            mesh_extraction["output_path"],
            mesh_extraction["logs"],
            mesh_extraction["status"],
        ],
    )
    
    # Credential save/test buttons
    mesh_extraction["save_creds_btn"].click(
        fn=lambda eid, key: save_serverless_credentials(eid, key, "mesh_extraction"),
        inputs=[mesh_extraction["endpoint_id"], mesh_extraction["api_key"]],
        outputs=[mesh_extraction["creds_status"], mesh_extraction["creds_group"]],
    )
    
    mesh_extraction["test_creds_btn"].click(
        fn=check_sugar_status,
        inputs=[mesh_extraction["endpoint_id"], mesh_extraction["api_key"]],
        outputs=[mesh_extraction["creds_status"]],
    )
    
    mesh_extraction["edit_creds_btn"].click(
        fn=lambda: gr.update(visible=True),
        inputs=[],
        outputs=[mesh_extraction["creds_group"]],
    )
    
    # Method toggle - show/hide settings accordions
    def toggle_method_settings(method: str):
        show_sugar = "SuGaR" in method
        show_tsdf = "TSDF" in method
        return gr.update(open=show_sugar), gr.update(open=show_tsdf)
    
    mesh_extraction["method"].change(
        fn=toggle_method_settings,
        inputs=[mesh_extraction["method"]],
        outputs=[mesh_extraction["sugar_settings"], mesh_extraction["tsdf_settings"]],
    )
    
    # Auto-fill Blender path from output
    mesh_extraction["output_path"].change(
        fn=lambda x: x,
        inputs=[mesh_extraction["output_path"]],
        outputs=[mesh_extraction["blender_mesh_path"]],
    )
    
    # PLY file dropdown - refresh and selection handlers
    from ui.tabs.mesh_extraction_tab import scan_for_ply_files
    
    def refresh_ply_dropdown():
        """Refresh the list of available PLY files."""
        files = scan_for_ply_files()
        return gr.update(choices=files)
    
    def on_ply_selected(selection: str):
        """Extract path from dropdown selection (removes size info)."""
        if selection and " (" in selection:
            # Remove the " (X.X MB)" suffix
            return selection.rsplit(" (", 1)[0]
        return selection or ""
    
    mesh_extraction["refresh_btn"].click(
        fn=refresh_ply_dropdown,
        inputs=[],
        outputs=[mesh_extraction["ply_dropdown"]],
    )
    
    mesh_extraction["ply_dropdown"].change(
        fn=on_ply_selected,
        inputs=[mesh_extraction["ply_dropdown"]],
        outputs=[mesh_extraction["input_ply"]],
    )
    
    # --- Clear Queue ---
    def clear_queue_and_status():
        data, msg = clear_completed_jobs()
        return data
    
    clear_queue_btn.click(
        fn=clear_queue_and_status,
        outputs=[job_queue_display],
    )
    
    # --- System Metrics Timer ---
    def update_metrics():
        return format_system_metrics(get_system_metrics())

    timer = gr.Timer(2.0)
    timer.tick(fn=update_metrics, outputs=[system_metrics_display])


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Initialize job manager for distributed execution
    print("[STARTUP] Initializing job manager...")
    _init_job_manager()
    
    # Start monitoring thread
    start_monitoring()
    
    # Give monitoring thread time to start
    time.sleep(0.5)
    
    # Print startup status
    print(f"[STARTUP] Cluster status: {format_cluster_status()}")
    print("[STARTUP] 3D Generation Studio v2.1 ready!")

    demo.launch(
        server_port=5683,
        share=False,
        css=CUSTOM_CSS,
        allowed_paths=[
            "/srv/searidge_share/outputs",  # All model outputs
            "/tmp",
        ]
    )
