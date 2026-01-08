#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
3D Generation Studio - Sidebar UI Version

Modern sidebar-based interface for multi-model 3D generation.
Organized by workflow: Input → Create → Refine → Monitor

Usage:
    python app_sidebar.py
    
Runs on port 5683 by default.
"""
print("DEBUG: STARTING 3D GENERATION STUDIO v2.2 (Sidebar UI) - Port 5684")

import os
import json
import time
from datetime import datetime
from typing import Optional, Tuple, Union, Dict, Any
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
                    if key not in os.environ or not os.environ[key]:
                        os.environ[key] = value
        print(f"[CONFIG] Loaded AWS credentials from {aws_creds_file}")

_load_aws_credentials()

import gradio as gr

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
    scan_for_ply_files,
    scan_for_lyra_ply_files,
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
    run_sharp_local,
    run_sharp_runpod,
    check_sharp_installation,
    SHARP_DEFAULT_OUTPUT_DIR,
    run_lyra_runpod,
    check_lyra_status,
    LYRA_DEFAULT_OUTPUT_DIR,
    run_trellis_runpod,
    check_trellis_status,
    TRELLIS_DEFAULT_OUTPUT_DIR,
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

from utils.version_tracker import (
    check_all_versions,
    format_version_check_status,
    get_local_version,
    query_runpod_versions,
    update_local_versions_from_runpod,
)


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
    config[f"{model}_endpoint_id"] = endpoint_id.strip() if endpoint_id else ""
    config[f"{model}_api_key"] = api_key.strip() if api_key else ""
    if model == "gen3c":
        config["serverless_endpoint_id"] = endpoint_id.strip() if endpoint_id else ""
        config["serverless_api_key"] = api_key.strip() if api_key else ""
    _save_runpod_config(config)
    return "✅ Credentials saved", gr.update(visible=False)


# Load saved config
_runpod_config = _load_runpod_config()
DEFAULT_RUNPOD_URL = _runpod_config.get("pod_url", "")
DEFAULT_GEN3C_ENDPOINT = _runpod_config.get("gen3c_endpoint_id", _runpod_config.get("serverless_endpoint_id", ""))
DEFAULT_GEN3C_API_KEY = _runpod_config.get("gen3c_api_key", _runpod_config.get("serverless_api_key", ""))
DEFAULT_SHARP_ENDPOINT = _runpod_config.get("sharp_endpoint_id", DEFAULT_GEN3C_ENDPOINT)
DEFAULT_SHARP_API_KEY = _runpod_config.get("sharp_api_key", DEFAULT_GEN3C_API_KEY)
DEFAULT_LYRA_ENDPOINT = _runpod_config.get("lyra_endpoint_id", DEFAULT_GEN3C_ENDPOINT)
DEFAULT_LYRA_API_KEY = _runpod_config.get("lyra_api_key", DEFAULT_GEN3C_API_KEY)
DEFAULT_TRELLIS_ENDPOINT = _runpod_config.get("trellis_endpoint_id", "")
DEFAULT_TRELLIS_API_KEY = _runpod_config.get("trellis_api_key", DEFAULT_GEN3C_API_KEY)
DEFAULT_HUNYUAN_ENDPOINT = _runpod_config.get("hunyuan_endpoint_id", "")
DEFAULT_HUNYUAN_API_KEY = _runpod_config.get("hunyuan_api_key", DEFAULT_GEN3C_API_KEY)
DEFAULT_MESH_ENDPOINT = _runpod_config.get("mesh_extraction_endpoint_id", DEFAULT_GEN3C_ENDPOINT)
DEFAULT_MESH_API_KEY = _runpod_config.get("mesh_extraction_api_key", DEFAULT_GEN3C_API_KEY)


# =============================================================================
# GENERATION HANDLERS (same as original)
# =============================================================================

def handle_sharp_generation(
    image_path: Union[str, None],
    image_scale: Union[float, int],
    exec_mode: str,
    endpoint_id: str,
    api_key: str,
    render_video: bool,
    output_name: str,
    output_dir: str,
    trajectory_type: str = "rotate_forward",
    num_steps: int = 60,
    num_repeats: int = 1,
    max_disparity: float = 0.08,
    max_zoom: float = 0.15,
    lookat_mode: str = "point",
    log_params: bool = True,
    encode_params: bool = False,
) -> Tuple[Optional[str], str, str]:
    """Handle SHARP generation with optional video rendering."""
    scale_value = clamp_scale_value(image_scale)
    scaled_path, temp_scaled = maybe_downscale_image(image_path, scale_value)
    effective_image_path = scaled_path or image_path
    
    # Generate parameter-encoded filename if requested
    # Note: SHARP doesn't expose many params in UI, so encoding is minimal
    effective_output_name = output_name
    if encode_params:
        try:
            from scripts.experiment_logger import sharp_param_filename
            encoded_name = sharp_param_filename(
                base_name=output_name,
                guidance=7.5,  # Default - not exposed in UI
                steps=50,  # Default - not exposed in UI
                ext=""  # No extension
            )
            effective_output_name = encoded_name.rstrip(".")
        except Exception as e:
            print(f"[SHARP] Warning: Could not encode params in filename: {e}")
    
    try:
        if "RunPod" in exec_mode:
            result = run_sharp_runpod(
                image_path=effective_image_path,
                endpoint_id=endpoint_id,
                api_key=api_key,
                output_name=effective_output_name,
                output_dir=output_dir,
                render_video=render_video,
                trajectory_type=trajectory_type,
                num_steps=int(num_steps),
                num_repeats=int(num_repeats),
                max_disparity=float(max_disparity),
                max_zoom=float(max_zoom),
                lookat_mode=lookat_mode,
            )
        else:
            result = run_sharp_local(
                image_path=effective_image_path,
                output_name=effective_output_name,
                output_dir=output_dir,
                render_video=render_video,
                trajectory_type=trajectory_type,
                num_steps=int(num_steps),
                num_repeats=int(num_repeats),
                max_disparity=float(max_disparity),
                max_zoom=float(max_zoom),
                lookat_mode=lookat_mode,
            )
        
        # Log experiment if enabled and successful
        output_path, logs, progress = result
        if log_params and output_path and os.path.exists(output_path):
            try:
                from scripts.experiment_logger import log_sharp_experiment
                log_files = log_sharp_experiment(
                    output_path=output_path,
                    input_image=effective_image_path or "",
                    seed=0,  # SHARP doesn't expose seed in current UI
                    guidance_scale=7.5,  # Default
                    inference_steps=50,  # Default
                    output_format=os.path.splitext(output_path)[1],
                    results={"success": True, "exec_mode": exec_mode},
                    save_json=True,
                )
                logs += f"\n📋 Logged to: {log_files.get('csv', 'N/A')}"
            except Exception as e:
                logs += f"\n⚠️ Logging failed: {e}"
        
        return result
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
    endpoint_id: str,
    api_key: str,
    guidance: float,
    frames: Union[str, int],
    trajectory: str,
    movement_distance: float,
    camera_rotation: str,
    foreground_mask: bool,
    video_name: str,
    seed: Optional[int],
    output_dir: str,
    log_params: bool = True,
    encode_params: bool = False,
) -> Tuple[Optional[str], str, str]:
    """Handle GEN3C generation."""
    scale_value = clamp_scale_value(image_scale)
    scaled_path, temp_scaled = maybe_downscale_image(image_path, scale_value)
    effective_image_path = scaled_path or image_path
    resolved_output_dir = output_dir.strip() if output_dir else GEN3C_DEFAULT_OUTPUT_DIR
    
    # Generate parameter-encoded filename if requested
    effective_video_name = video_name
    if encode_params:
        try:
            from scripts.experiment_logger import gen3c_param_filename
            encoded_name = gen3c_param_filename(
                base_name=video_name,
                frames=int(frames) if frames else 121,
                trajectory=trajectory,
                ext=""  # No extension
            )
            effective_video_name = encoded_name.rstrip(".")
        except Exception as e:
            print(f"[Gen3C] Warning: Could not encode params in filename: {e}")
    
    try:
        result = run_gen3c_serverless(
            image_path=effective_image_path,
            endpoint_id=endpoint_id,
            api_key=api_key,
            guidance=guidance,
            frames=frames,
            trajectory=trajectory,
            movement_distance=movement_distance,
            camera_rotation=camera_rotation,
            foreground_masking=foreground_mask,
            video_name=effective_video_name,
            seed=seed,
            output_dir=resolved_output_dir,
        )
        
        # Log experiment if enabled and successful
        output_path, logs, progress = result
        if log_params and output_path and os.path.exists(output_path):
            try:
                from scripts.experiment_logger import log_gen3c_experiment
                # Parse camera rotation
                try:
                    cam_rot = float(camera_rotation) if camera_rotation else 0.0
                except (ValueError, TypeError):
                    cam_rot = 0.0
                
                log_files = log_gen3c_experiment(
                    output_path=output_path,
                    input_image=effective_image_path or "",
                    seed=int(seed) if seed else 0,
                    num_frames=int(frames) if frames else 121,
                    guidance_scale=guidance,
                    trajectory=trajectory,
                    camera_rotation=cam_rot,
                    movement_distance=movement_distance,
                    results={"success": True, "foreground_mask": foreground_mask},
                    save_json=True,
                )
                logs += f"\n📋 Logged to: {log_files.get('csv', 'N/A')}"
            except Exception as e:
                logs += f"\n⚠️ Logging failed: {e}"
        
        return output_path, logs, progress
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
    log_params: bool = True,
    encode_params: bool = False,
) -> Tuple[Optional[str], str, str, str]:
    """Handle Lyra generation."""
    scale_value = clamp_scale_value(image_scale)
    is_static = "Static" in generation_mode
    
    if is_static:
        scaled_path, temp_scaled = maybe_downscale_image(image_path, scale_value)
        effective_image_path = scaled_path or image_path
        effective_video_path = None
    else:
        effective_image_path = None
        effective_video_path = video_path
        temp_scaled = None
    
    # Generate parameter-encoded filename if requested
    effective_output_name = output_name
    if encode_params:
        try:
            from scripts.experiment_logger import lyra_param_filename
            encoded_name = lyra_param_filename(
                base_name=output_name,
                guidance=7.5,  # Default - not exposed in UI
                sdg_steps=250,  # Default - not exposed in UI
                ext=""  # No extension
            )
            effective_output_name = encoded_name.rstrip(".")
        except Exception as e:
            print(f"[Lyra] Warning: Could not encode params in filename: {e}")
    
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
            output_name=effective_output_name,
            output_dir=output_dir,
            output_ply=output_ply,
            output_video=output_video,
        )
        
        output_path, logs, progress = result
        ply_path = output_path if output_path and output_path.endswith(".ply") else ""
        
        # Log experiment if enabled and successful
        if log_params and output_path and os.path.exists(output_path):
            try:
                from scripts.experiment_logger import log_lyra_experiment
                log_files = log_lyra_experiment(
                    output_path=output_path,
                    input_image=effective_image_path or effective_video_path or "",
                    seed=int(seed) if seed else 0,
                    guidance_scale=7.5,  # Default - Lyra doesn't expose this
                    inference_steps=50,  # Default
                    sdg_steps=250,  # Default
                    resolution=512,  # Default
                    mode="3dgs" if is_static else "4dgs",
                    results={
                        "success": True,
                        "generation_mode": generation_mode,
                        "num_views": num_views,
                        "num_gaussians": num_gaussians,
                    },
                    save_json=True,
                )
                logs += f"\n📋 Logged to: {log_files.get('csv', 'N/A')}"
            except Exception as e:
                logs += f"\n⚠️ Logging failed: {e}"
        
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
    endpoint_id: str,
    api_key: str,
    resolution: str,
    guidance_scale: float,
    seed: Optional[int],
    output_name: str,
    output_dir: str,
    output_format: str,
    log_params: bool = True,
    encode_params: bool = False,
) -> Tuple[Optional[str], str, str]:
    """Handle TRELLIS.2 generation."""
    scale_value = clamp_scale_value(image_scale)
    scaled_path, temp_scaled = maybe_downscale_image(image_path, scale_value)
    effective_image_path = scaled_path or image_path
    
    # Generate parameter-encoded filename if requested
    effective_output_name = output_name
    if encode_params:
        try:
            from scripts.experiment_logger import trellis_param_filename
            # Parse resolution
            try:
                res_int = int(resolution) if resolution else 1024
            except (ValueError, TypeError):
                res_int = 1024
            
            encoded_name = trellis_param_filename(
                base_name=output_name,
                resolution=res_int,
                guidance=guidance_scale,
                ext=""  # No extension
            )
            effective_output_name = encoded_name.rstrip(".")
        except Exception as e:
            print(f"[Trellis] Warning: Could not encode params in filename: {e}")
    
    try:
        result = run_trellis_runpod(
            image_path=effective_image_path,
            endpoint_id=endpoint_id,
            api_key=api_key,
            resolution=resolution,
            guidance_scale=guidance_scale,
            seed=int(seed) if seed else None,
            output_name=effective_output_name,
            output_dir=output_dir,
            output_format=output_format,
        )
        
        # Log experiment if enabled and successful
        output_path, logs, progress = result
        if log_params and output_path and os.path.exists(output_path):
            try:
                from scripts.experiment_logger import log_trellis_experiment
                # Parse resolution
                try:
                    res_int = int(resolution) if resolution else 1024
                except (ValueError, TypeError):
                    res_int = 1024
                
                log_files = log_trellis_experiment(
                    output_path=output_path,
                    input_image=effective_image_path or "",
                    seed=int(seed) if seed else 0,
                    resolution=res_int,
                    guidance_scale=guidance_scale,
                    output_format=output_format,
                    results={"success": True},
                    save_json=True,
                )
                logs += f"\n📋 Logged to: {log_files.get('csv', 'N/A')}"
            except Exception as e:
                logs += f"\n⚠️ Logging failed: {e}"
        
        return output_path, logs, progress
    finally:
        if temp_scaled and os.path.exists(temp_scaled):
            try:
                os.remove(temp_scaled)
            except Exception:
                pass


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
    log_params: bool = True,
    encode_params: bool = False,
) -> Tuple[Optional[str], str, str]:
    """Handle Hunyuan3D generation."""
    scale_value = clamp_scale_value(image_scale)
    scaled_path, temp_scaled = maybe_downscale_image(image_path, scale_value)
    effective_image_path = scaled_path or image_path

    # Generate parameter-encoded filename if requested
    effective_output_name = output_name
    if encode_params:
        try:
            from scripts.experiment_logger import hunyuan_param_filename
            # Generate encoded filename (without extension - the generator adds .glb)
            encoded_name = hunyuan_param_filename(
                base_name=output_name,
                guidance=guidance_scale,
                octree=int(octree_resolution) if octree_resolution else 380,
                steps=int(steps) if steps else 40,
                ext=""  # No extension, generator adds it
            )
            # Remove any trailing dots from the name
            effective_output_name = encoded_name.rstrip(".")
        except Exception as e:
            print(f"[Hunyuan] Warning: Could not encode params in filename: {e}")
            effective_output_name = output_name

    try:
        if exec_mode == "RunPod Serverless":
            result = run_hunyuan_runpod(
                image_path=effective_image_path,
                endpoint_id=endpoint_id,
                api_key=api_key,
                model_choice=model_choice,
                guidance_scale=guidance_scale,
                steps=steps,
                octree_resolution=int(octree_resolution) if octree_resolution else 380,
                seed=int(seed) if seed else None,
                remove_background=remove_background,
                output_name=effective_output_name,
                output_dir=save_location,
            )
        else:
            result = run_hunyuan(
                image_path=effective_image_path,
                guidance_scale=guidance_scale,
                steps=steps,
                seed=seed,
                model_choice=model_choice,
                use_fp16=use_fp16,
                attention_slicing=attention_slicing,
                cpu_offload=cpu_offload,
                remove_background=remove_background,
                output_name=effective_output_name,
                save_location=save_location,
            )
        
        # Log experiment if enabled and successful
        output_path, logs, progress = result
        if log_params and output_path and os.path.exists(output_path):
            try:
                from scripts.experiment_logger import log_hunyuan_experiment
                log_files = log_hunyuan_experiment(
                    output_path=output_path,
                    input_image=effective_image_path or "",
                    seed=int(seed) if seed else 0,
                    guidance_scale=guidance_scale,
                    inference_steps=steps,
                    octree_depth=int(octree_resolution) if octree_resolution else 380,
                    model_type=model_choice,
                    remove_bg=remove_background,
                    results={"success": True, "exec_mode": exec_mode},
                    save_json=True,
                )
                logs += f"\n📋 Logged to: {log_files.get('csv', 'N/A')}"
            except Exception as e:
                logs += f"\n⚠️ Logging failed: {e}"
        
        return output_path, logs, progress
    finally:
        if temp_scaled and os.path.exists(temp_scaled):
            try:
                os.remove(temp_scaled)
            except Exception:
                pass


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
    
    # Clean up dropdown value - remove format info in parentheses if present
    if " (" in input_ply:
        input_ply = input_ply.rsplit(" (", 1)[0]
    
    if "SuGaR" in method or "Poisson" in method:
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


def handle_mesh_analyze(
    file_input,
    path_input: str,
) -> Tuple[str, str]:
    """Analyze a mesh file and return statistics."""
    # Determine input path
    if file_input is not None:
        input_path = file_input.name if hasattr(file_input, 'name') else str(file_input)
    elif path_input and path_input.strip():
        input_path = path_input.strip()
    else:
        return "No input file specified", "❌ Missing input"
    
    if not os.path.exists(input_path):
        return f"File not found: {input_path}", "❌ File not found"
    
    try:
        # Import the cleanup script functions
        from scripts.cleanup_mesh import analyze_mesh, TRIMESH_AVAILABLE
        
        if not TRIMESH_AVAILABLE:
            return "trimesh not installed. Install with: pip install trimesh", "❌ Missing dependency"
        
        stats = analyze_mesh(input_path)
        
        # Format analysis results
        result_lines = [
            f"File: {os.path.basename(input_path)}",
            f"",
            f"Geometry:",
            f"  Vertices: {stats['vertices']:,}",
            f"  Triangles: {stats['triangles']:,}",
            f"  Components: {stats['components']}",
            f"",
            f"Size:",
            f"  X: {stats['size'][0]:.3f}",
            f"  Y: {stats['size'][1]:.3f}",
            f"  Z: {stats['size'][2]:.3f}",
            f"",
            f"Quality:",
            f"  Watertight: {'✅' if stats['is_watertight'] else '❌'}",
            f"  Consistent Winding: {'✅' if stats['is_winding_consistent'] else '❌'}",
        ]
        
        if stats.get('issues'):
            result_lines.append("")
            result_lines.append("Issues Found:")
            for issue in stats['issues']:
                result_lines.append(f"  ⚠️ {issue}")
        else:
            result_lines.append("")
            result_lines.append("✅ No issues detected")
        
        return "\n".join(result_lines), "✅ Analysis complete"
        
    except Exception as e:
        return f"Error analyzing mesh: {str(e)}", f"❌ {str(e)}"


def handle_mesh_cleanup(
    file_input,
    path_input: str,
    target_triangles: int,
    smooth_iterations: int,
    preserve_detail: bool,
    post_decimate_smooth: int,
    remove_components: bool,
    fix_normals: bool,
    fill_holes: bool,
    aggressive: bool,
    min_component_ratio: float,
    output_name: str,
    output_dir: str,
    output_format: str,
    log_params: bool,
    encode_params: bool,
) -> Tuple[str, str, str]:
    """Clean up a mesh file."""
    # Determine input path
    if file_input is not None:
        input_path = file_input.name if hasattr(file_input, 'name') else str(file_input)
    elif path_input and path_input.strip():
        input_path = path_input.strip()
    else:
        return "", "No input file specified", "❌ Missing input"
    
    if not os.path.exists(input_path):
        return "", f"File not found: {input_path}", "❌ File not found"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Build output path
    ext = output_format.lower()
    output_path = os.path.join(output_dir, f"{output_name}.{ext}")
    
    try:
        # Import the cleanup script functions
        from scripts.cleanup_mesh import cleanup_mesh, TRIMESH_AVAILABLE
        
        if not TRIMESH_AVAILABLE:
            return "", "trimesh not installed. Install with: pip install trimesh", "❌ Missing dependency"
        
        stats = cleanup_mesh(
            input_path=input_path,
            output_path=output_path,
            target_triangles=int(target_triangles),
            remove_small_components=remove_components,
            min_component_ratio=min_component_ratio,
            fix_normals=fix_normals,
            fill_holes=fill_holes,
            smooth_iterations=int(smooth_iterations),
            aggressive=aggressive,
            preserve_detail=preserve_detail,
            post_decimate_smooth=int(post_decimate_smooth),
            log_params=log_params,
            encode_params_in_filename=encode_params,
        )
        
        # Get actual output path (may have been modified if encode_params)
        actual_output = stats.get('output_file', output_path)
        
        # Format log output
        log_lines = [
            f"Input: {stats['input_triangles']:,} triangles",
            f"Output: {stats['output_triangles']:,} triangles",
            f"Reduction: {100 * (1 - stats['output_triangles'] / stats['input_triangles']):.1f}%",
            f"",
            f"Operations performed:",
        ]
        for op in stats.get('operations', []):
            log_lines.append(f"  - {op}")
        
        log_lines.append("")
        log_lines.append(f"Components removed: {stats.get('components_removed', 0)}")
        log_lines.append(f"Watertight: {'✅' if stats.get('is_watertight') else '❌'}")
        log_lines.append(f"Consistent winding: {'✅' if stats.get('is_winding_consistent') else '❌'}")
        
        # Add parameter log info
        if log_params and 'param_log' in stats:
            log_lines.append("")
            log_lines.append(f"📋 Parameter log: {stats['param_log']}")
            log_lines.append(f"📊 Experiment CSV: {stats.get('csv_log', '/srv/searidge_share/outputs/logs/mesh_cleanup.csv')}")
        log_lines.append("")
        log_lines.append(f"Saved to: {actual_output}")
        
        return actual_output, "\n".join(log_lines), "✅ Cleanup complete!"
        
    except Exception as e:
        import traceback
        return "", f"Error: {str(e)}\n\n{traceback.format_exc()}", f"❌ {str(e)}"


# =============================================================================
# SIDEBAR NAVIGATION STRUCTURE
# =============================================================================

NAV_ITEMS = [
    {"id": "sharp", "icon": "⚡", "label": "SHARP", "category": "create", "description": "Fast 3DGS (~60s)"},
    {"id": "gen3c", "icon": "🎬", "label": "GEN3C", "category": "create", "description": "Video generation (~10min)"},
    {"id": "lyra", "icon": "🌀", "label": "Lyra", "category": "create", "description": "3DGS from video (~15min)"},
    {"id": "trellis", "icon": "🔷", "label": "TRELLIS.2", "category": "create", "description": "High-quality 3D"},
    {"id": "hunyuan", "icon": "🏔️", "label": "Hunyuan3D", "category": "create", "description": "Image to GLB mesh"},
    {"id": "mesh", "icon": "🔶", "label": "Mesh Extract", "category": "refine", "description": "3DGS → GLB mesh"},
    {"id": "settings", "icon": "⚙️", "label": "Settings", "category": "monitor", "description": "Credentials & config"},
]


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

with gr.Blocks(title="3D Generation Studio") as demo:
    
    # State for current page
    current_page = gr.State(value="sharp")
    
    with gr.Row(elem_classes=["main-layout"]):
        
        # =====================================================================
        # SIDEBAR
        # =====================================================================
        with gr.Column(scale=0, min_width=280, elem_classes=["sidebar"]):
            
            # Header
            gr.HTML("""
                <div class="sidebar-header">
                    <h2>ARKRUNR WORLDS</h2>
                </div>
            """)
            
            # Input Section (always visible)
            gr.HTML('<div class="sidebar-category">INPUT</div>')
            
            input_image = gr.Image(
                type="filepath",
                label="Drop Image Here",
                height=200,
                show_label=True,
                sources=["upload", "clipboard"],
            )
            
            with gr.Group(elem_classes=["scale-info-box"]):
                image_scale = gr.Slider(
                    minimum=0.25, maximum=1.0, value=1.0, step=0.05,
                    label="Scale",
                )
                image_info = gr.Textbox(
                    value="No image loaded.",
                    show_label=False,
                    interactive=False,
                    max_lines=1,
                    elem_classes=["image-info-text"],
                )
            
            # Logging controls (applies to all models)
            with gr.Accordion("📋 Experiment Logging", open=False, elem_classes=["logging-accordion"]):
                global_log_params = gr.Checkbox(
                    value=True, 
                    label="Log Parameters",
                    info="Save JSON + CSV logs"
                )
                global_encode_params = gr.Checkbox(
                    value=False,
                    label="Encode in Filename",
                    info="Add params to output name"
                )
                gr.Markdown(
                    "📁 CSV logs: `/srv/searidge_share/outputs/logs/`",
                    elem_classes=["log-path-info"]
                )
            
            input_video = gr.Video(
                label="Source Video",
                visible=False,
                height=150,
            )
            
            # Create Section
            gr.HTML('<div class="sidebar-category">CREATE</div>')
            
            # Button grid - 2 columns
            with gr.Row(elem_classes=["button-grid"]):
                nav_sharp = gr.Button("SHARP", elem_classes=["sidebar-nav"], elem_id="nav-sharp", scale=1)
                nav_gen3c = gr.Button("GEN3C", elem_classes=["sidebar-nav"], elem_id="nav-gen3c", scale=1)
            with gr.Row(elem_classes=["button-grid"]):
                nav_lyra = gr.Button("Lyra", elem_classes=["sidebar-nav"], elem_id="nav-lyra", scale=1)
                nav_trellis = gr.Button("TRELLIS", elem_classes=["sidebar-nav"], elem_id="nav-trellis", scale=1)
            with gr.Row(elem_classes=["button-grid"]):
                nav_hunyuan = gr.Button("Hunyuan", elem_classes=["sidebar-nav"], elem_id="nav-hunyuan", scale=1)
                nav_mesh = gr.Button("Mesh", elem_classes=["sidebar-nav"], elem_id="nav-mesh", scale=1)
            
            # Refine/Monitor Section combined
            gr.HTML('<div class="sidebar-category">TOOLS</div>')
            
            with gr.Row(elem_classes=["button-grid"]):
                nav_settings = gr.Button("Settings", elem_classes=["sidebar-nav"], elem_id="nav-settings", scale=1)
                nav_update = gr.Button("Update", elem_classes=["sidebar-nav"], elem_id="nav-update", scale=1)
            with gr.Row(elem_classes=["button-grid"]):
                nav_help = gr.Button("Help", elem_classes=["sidebar-nav"], elem_id="nav-help", scale=1)
            
            # System metrics at bottom
            gr.HTML('<div class="sidebar-category">SYSTEM</div>')
            system_metrics = gr.Textbox(
                value="Loading...",
                show_label=False,
                interactive=False,
                max_lines=4,
                lines=3,
            )
            
            # Footer
            gr.HTML("""
                <div class="sidebar-footer">
                    v2.2 • Sidebar UI
                </div>
            """)
        
        # =====================================================================
        # MAIN CONTENT AREA
        # =====================================================================
        with gr.Column(scale=1, elem_classes=["main-content"]):
            
            # Output display (shared across all pages)
            with gr.Group():
                output_display = gr.Textbox(
                    label="Output",
                    value="Output file path will appear here",
                    interactive=False,
                )
            
            # Video preview - separate from output, hidden by default
            output_video = gr.Video(
                label="Video Preview",
                visible=False,
                height=200,
            )
            
            # ─────────────────────────────────────────────────────────────────
            # PAGES CONTAINER - Using Tabs with hidden tab bar
            # ─────────────────────────────────────────────────────────────────
            with gr.Tabs(elem_classes=["hidden-tabs"]) as page_tabs:
                
                # PAGE: SHARP
                with gr.TabItem("SHARP", id="sharp"):
                    gr.HTML("""
                        <div class="page-header">
                            <h1>SHARP</h1>
                            <p>Apple's fast 3D Gaussian Splatting from a single image. Generates PLY in ~60 seconds, optional video rendering.</p>
                        </div>
                    """)
                    
                    with gr.Tabs():
                        # TAB: Generate PLY
                        with gr.Tab("Generate"):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    with gr.Group():
                                        gr.Markdown("### Output Settings")
                                        sharp_output_name = gr.Textbox(value="sharp_output", label="Output Name")
                                        sharp_output_dir = gr.Textbox(value=SHARP_DEFAULT_OUTPUT_DIR, label="Output Directory")
                                    
                                    sharp_generate_btn = gr.Button("Generate PLY", variant="primary", size="lg")
                                    sharp_progress = gr.Textbox(value="Ready", label="Status", interactive=False)
                                
                                with gr.Column(scale=1):
                                    sharp_3d_viewer = gr.Model3D(
                                        label="3D Preview",
                                        height=400,
                                        clear_color=[0.1, 0.1, 0.1, 1.0],
                                    )
                            
                            with gr.Accordion("📋 Logs", open=False):
                                sharp_logs = gr.Textbox(label="Generation Logs", lines=8, interactive=False)
                        
                        # TAB: Render Video
                        with gr.Tab("Render Video"):
                            gr.Markdown("""
                            **Video Rendering** creates a camera trajectory video from a SHARP PLY file.
                            Requires CUDA GPU (RunPod recommended). Can be done during generation or from existing PLY.
                            """)
                            
                            with gr.Row():
                                with gr.Column(scale=1):
                                    with gr.Group():
                                        gr.Markdown("### Input")
                                        sharp_video_mode = gr.Radio(
                                            choices=["Generate PLY + Video", "Render from existing PLY"],
                                            value="Generate PLY + Video",
                                            label="Mode",
                                        )
                                        sharp_video_ply_path = gr.Textbox(
                                            label="PLY Path (for existing PLY mode)",
                                            placeholder="Path to .ply file",
                                            visible=False,
                                        )
                                    
                                    with gr.Group():
                                        gr.Markdown("### Trajectory Settings")
                                        sharp_trajectory_type = gr.Dropdown(
                                            choices=["rotate_forward", "rotate", "swipe", "shake"],
                                            value="rotate_forward",
                                            label="Trajectory Type",
                                            info="Camera movement pattern"
                                        )
                                        with gr.Row():
                                            sharp_num_steps = gr.Slider(
                                                minimum=30, maximum=180, value=60, step=10,
                                                label="Frames",
                                                info="Number of frames in video"
                                            )
                                            sharp_num_repeats = gr.Slider(
                                                minimum=1, maximum=4, value=1, step=1,
                                                label="Repeats",
                                                info="Number of trajectory loops"
                                            )
                                        with gr.Row():
                                            sharp_max_disparity = gr.Slider(
                                                minimum=0.02, maximum=0.20, value=0.08, step=0.01,
                                                label="Lateral Offset",
                                                info="Max horizontal/vertical movement"
                                            )
                                            sharp_max_zoom = gr.Slider(
                                                minimum=0.05, maximum=0.40, value=0.15, step=0.05,
                                                label="Zoom/Forward",
                                                info="Max forward movement"
                                            )
                                        sharp_lookat_mode = gr.Dropdown(
                                            choices=["point", "ahead"],
                                            value="point",
                                            label="Look-At Mode",
                                            info="Camera focus behavior"
                                        )
                                    
                                    with gr.Group():
                                        gr.Markdown("### Output")
                                        sharp_video_output_name = gr.Textbox(value="sharp_video", label="Video Name")
                                        sharp_video_output_dir = gr.Textbox(value=SHARP_DEFAULT_OUTPUT_DIR, label="Output Directory")
                                    
                                    sharp_render_video_btn = gr.Button("Render Video", variant="primary", size="lg")
                                    sharp_video_progress = gr.Textbox(value="Ready", label="Status", interactive=False)
                                
                                with gr.Column(scale=1):
                                    sharp_video_preview = gr.Video(
                                        label="Video Preview",
                                        height=400,
                                    )
                            
                            with gr.Accordion("📋 Trajectory Types Explained", open=False):
                                gr.Markdown("""
                                | Type | Description |
                                |------|-------------|
                                | **rotate_forward** | Circular rotation with forward zoom (default, best for most scenes) |
                                | **rotate** | Pure circular rotation around the scene center |
                                | **swipe** | Left-to-right horizontal pan |
                                | **shake** | Horizontal shake followed by vertical shake |
                                
                                **Parameters:**
                                - **Frames**: Total frames in video (60 = ~2s at 30fps)
                                - **Repeats**: How many times to loop the trajectory
                                - **Lateral Offset**: How far camera moves sideways (higher = more dramatic)
                                - **Zoom/Forward**: How far camera moves forward (higher = more zoom effect)
                                - **Look-At Mode**: "point" keeps camera focused on scene center, "ahead" looks straight ahead
                                """)
                            
                            with gr.Accordion("📋 Logs", open=False):
                                sharp_video_logs = gr.Textbox(label="Render Logs", lines=8, interactive=False)
                
                # PAGE: GEN3C
                with gr.TabItem("GEN3C", id="gen3c"):
                    gr.HTML("""
                        <div class="page-header">
                            <h1>GEN3C</h1>
                            <p>NVIDIA's 3D-consistent video generation from a single image. Creates camera-controlled videos with 3D consistency.</p>
                        </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Group():
                                gr.Markdown("### Video Settings")
                                with gr.Row():
                                    gen3c_trajectory = gr.Dropdown(
                                        choices=["left", "right", "up", "down", "zoom_in", "zoom_out", "clockwise", "counterclockwise"],
                                        value="left",
                                        label="Camera Trajectory",
                                    )
                                    gen3c_frames = gr.Dropdown(
                                        choices=["121", "241", "361"],
                                        value="121",
                                        label="Frames (121*N - 1)",
                                    )
                                with gr.Row():
                                    gen3c_movement_distance = gr.Slider(
                                        minimum=0.1, maximum=1.0, value=0.3, step=0.05,
                                        label="Movement Distance",
                                        info="How far camera moves (0.1=subtle, 1.0=dramatic)"
                                    )
                                    gen3c_camera_rotation = gr.Dropdown(
                                        choices=["center_facing", "no_rotation", "trajectory_aligned"],
                                        value="center_facing",
                                        label="Camera Rotation",
                                        info="How camera rotates during movement"
                                    )
                                with gr.Row():
                                    gen3c_guidance = gr.Slider(minimum=0.5, maximum=5.0, value=1.0, label="Guidance")
                                    gen3c_seed = gr.Number(value=None, label="Seed", precision=0)
                                gen3c_foreground = gr.Checkbox(value=True, label="Foreground Masking")
                            
                            with gr.Group():
                                gr.Markdown("### Output")
                                gen3c_video_name = gr.Textbox(value="gen3c_video", label="Video Name")
                                gen3c_output_dir = gr.Textbox(value=GEN3C_DEFAULT_OUTPUT_DIR, label="Output Directory")
                            
                            gen3c_generate_btn = gr.Button("Generate Video", variant="primary", size="lg")
                            gen3c_progress = gr.Textbox(value="Ready", label="Status", interactive=False)
                    
                    with gr.Accordion("📋 Logs", open=False):
                        gen3c_logs = gr.Textbox(label="Generation Logs", lines=8, interactive=False)
                
                # PAGE: LYRA
                with gr.TabItem("Lyra", id="lyra"):
                    gr.HTML("""
                        <div class="page-header">
                            <h1>Lyra</h1>
                            <p>NVIDIA's Image/Video to 3D Gaussian Splatting. High-quality 3DGS with multi-view consistency.</p>
                        </div>
                    """)
                    
                    with gr.Tabs():
                        with gr.Tab("Generate"):
                            with gr.Row():
                                with gr.Column(scale=2):
                                    lyra_mode = gr.Radio(
                                        choices=["Static (Image → 3DGS)", "Dynamic (Video → 4DGS)"],
                                        value="Static (Image → 3DGS)",
                                        label="Mode",
                                    )
                                    
                                    with gr.Accordion("⚙️ Advanced", open=False):
                                        with gr.Row():
                                            lyra_views = gr.Slider(4, 16, 8, step=1, label="Views")
                                            lyra_motion = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="Camera Motion")
                                        with gr.Row():
                                            lyra_multi_traj = gr.Checkbox(True, label="Multi-Trajectory")
                                            lyra_fg_mask = gr.Checkbox(True, label="Foreground Mask")
                                        with gr.Row():
                                            lyra_gaussians = gr.Slider(10000, 500000, 100000, step=10000, label="Max Gaussians")
                                            lyra_seed = gr.Number(None, label="Seed", precision=0)
                                        with gr.Row():
                                            lyra_out_ply = gr.Checkbox(True, label="Export PLY")
                                            lyra_out_video = gr.Checkbox(True, label="Export Video")
                                    
                                    with gr.Group():
                                        lyra_output_name = gr.Textbox(value="lyra_output", label="Output Name")
                                        lyra_output_dir = gr.Textbox(value=LYRA_DEFAULT_OUTPUT_DIR, label="Output Dir")
                                    
                                    lyra_generate_btn = gr.Button("Generate 3DGS", variant="primary", size="lg")
                                    lyra_progress = gr.Textbox(value="Ready", label="Status", interactive=False)
                            
                            with gr.Accordion("📋 Logs", open=False):
                                lyra_logs = gr.Textbox(lines=8, interactive=False)
                        
                        with gr.Tab("Post-Process"):
                            gr.Markdown("### Convert Lyra PLY Output")
                            
                            initial_ply = scan_for_lyra_ply_files()
                            with gr.Row():
                                lyra_ply_dropdown = gr.Dropdown(choices=initial_ply, label="Select PLY", scale=3, allow_custom_value=True)
                                lyra_refresh_btn = gr.Button("🔄", scale=0)
                            lyra_ply_path = gr.Textbox(label="Or enter path", placeholder="/path/to/file.ply")
                            
                            with gr.Row():
                                lyra_conv_3dgs = gr.Checkbox(True, label="3DGS Format")
                                lyra_conv_pc = gr.Checkbox(False, label="Point Cloud")
                            
                            lyra_convert_btn = gr.Button("🔄 Convert", variant="secondary")
                            lyra_convert_status = gr.Textbox(label="Result", interactive=False)
                
                # PAGE: TRELLIS.2
                with gr.TabItem("TRELLIS.2", id="trellis"):
                    gr.HTML("""
                        <div class="page-header">
                            <h1>TRELLIS.2</h1>
                            <p>Microsoft's high-quality 3D generation. Creates detailed GLB meshes from images.</p>
                        </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Row():
                                trellis_resolution = gr.Dropdown(["512", "1024"], value="1024", label="Resolution")
                                trellis_guidance = gr.Slider(1.0, 15.0, 7.5, label="Guidance")
                            trellis_seed = gr.Number(None, label="Seed", precision=0)
                            trellis_format = gr.Dropdown(["GLB", "OBJ", "PLY"], value="GLB", label="Format")
                            
                            with gr.Group():
                                trellis_output_name = gr.Textbox(value="trellis_output", label="Output Name")
                                trellis_output_dir = gr.Textbox(value=TRELLIS_DEFAULT_OUTPUT_DIR, label="Output Dir")
                            
                            trellis_generate_btn = gr.Button("Generate 3D", variant="primary", size="lg")
                            trellis_progress = gr.Textbox(value="Ready", label="Status", interactive=False)
                    
                    with gr.Accordion("📋 Logs", open=False):
                        trellis_logs = gr.Textbox(lines=8, interactive=False)
                
                # PAGE: HUNYUAN3D
                with gr.TabItem("Hunyuan3D", id="hunyuan"):
                    gr.HTML("""
                        <div class="page-header">
                            <h1>Hunyuan3D</h1>
                            <p>Tencent's image-to-3D mesh generation. Creates GLB meshes directly from images.</p>
                        </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            hunyuan_model = gr.Dropdown(
                                ["mini", "full"], 
                                value="mini", 
                                label="Model",
                                info="mini: faster (2-5 min), full: higher quality (5-15 min)"
                            )
                            
                            with gr.Accordion("⚙️ Advanced", open=False):
                                with gr.Row():
                                    hunyuan_guidance = gr.Slider(1.0, 10.0, 5.0, label="Guidance")
                                    hunyuan_steps = gr.Slider(10, 100, 40, step=5, label="Steps")
                                with gr.Row():
                                    hunyuan_octree = gr.Slider(256, 512, 380, step=1, label="Octree Resolution")
                                    hunyuan_seed = gr.Number(None, label="Seed", precision=0)
                                with gr.Row():
                                    hunyuan_fp16 = gr.Checkbox(False, label="FP16")
                                    hunyuan_attn_slice = gr.Checkbox(False, label="Attention Slicing")
                                    hunyuan_cpu_offload = gr.Checkbox(False, label="CPU Offload")
                                hunyuan_remove_bg = gr.Checkbox(True, label="Remove Background")
                            
                            with gr.Group():
                                hunyuan_output_name = gr.Textbox(value="hunyuan_output", label="Output Name")
                                hunyuan_output_dir = gr.Textbox(value="/srv/searidge_share/outputs/hunyuan", label="Output Dir")
                            
                            hunyuan_exec_mode = gr.Radio(["RunPod Serverless", "Local"], value="RunPod Serverless", label="Mode", visible=False)
                            hunyuan_generate_btn = gr.Button("Generate Mesh", variant="primary", size="lg")
                            hunyuan_progress = gr.Textbox(value="Ready", label="Status", interactive=False)
                        
                        with gr.Column(scale=1):
                            hunyuan_3d_viewer = gr.Model3D(
                                label="3D Preview",
                                height=400,
                                clear_color=[0.1, 0.1, 0.1, 1.0],
                            )
                    
                    with gr.Accordion("📋 Logs", open=False):
                        hunyuan_logs = gr.Textbox(lines=8, interactive=False)
                
                # PAGE: MESH EXTRACTION
                with gr.TabItem("Mesh", id="mesh"):
                    gr.HTML("""
                        <div class="page-header">
                            <h1>Mesh Tools</h1>
                            <p>Extract meshes from 3DGS and clean up generated meshes for production use.</p>
                        </div>
                    """)
                    
                    with gr.Tabs():
                        with gr.Tab("Extract"):
                            with gr.Row():
                                with gr.Column(scale=2):
                                    with gr.Group():
                                        gr.Markdown("### Input PLY")
                                        mesh_ply_files = scan_for_ply_files()
                                        with gr.Row(elem_classes=["ply-input-row"]):
                                            mesh_ply_dropdown = gr.Dropdown(
                                                choices=mesh_ply_files, 
                                                label="Select or enter PLY path", 
                                                scale=5, 
                                                allow_custom_value=True,
                                                elem_classes=["ply-dropdown"],
                                            )
                                            mesh_refresh_btn = gr.Button("🔄", scale=0, elem_classes=["refresh-btn-inline"], min_width=40)
                                        mesh_format = gr.Radio(["Auto-detect", "Lyra", "SHARP", "Standard 3DGS"], value="Auto-detect", label="Format")
                                    
                                    with gr.Group():
                                        gr.Markdown("### Reconstruction Settings")
                                        with gr.Row():
                                            mesh_quality = gr.Dropdown(
                                                ["High Poly (1M)", "Low Poly (200k)", "Custom"],
                                                value="High Poly (1M)",
                                                label="Quality",
                                            )
                                            mesh_poisson_depth = gr.Slider(6, 12, 10, step=1, label="Poisson Depth")
                                        mesh_decimate = gr.Number(0, label="Decimate to (faces, 0=none)")
                                    
                                    with gr.Group():
                                        gr.Markdown("### Output Settings")
                                        mesh_output_name = gr.Textbox(value="mesh_output", label="Output Name")
                                        mesh_output_dir = gr.Textbox(value=MESH_DEFAULT_OUTPUT_DIR, label="Output Directory")
                                        mesh_output_format = gr.Dropdown(["GLB", "OBJ", "PLY"], value="GLB", label="Output Format")
                                    
                                    mesh_extract_btn = gr.Button("Extract Mesh", variant="primary", size="lg")
                                    mesh_progress = gr.Textbox(value="Ready", label="Status", interactive=False)
                            
                            with gr.Accordion("📋 Logs", open=False):
                                mesh_logs = gr.Textbox(lines=8, interactive=False)
                        
                        with gr.Tab("Cleanup"):
                            gr.Markdown("### Clean Up Generated Meshes")
                            gr.Markdown("Remove artifacts, fix normals, decimate, and prepare meshes for Unity/Blender.")
                            
                            with gr.Row():
                                with gr.Column(scale=1):
                                    with gr.Group():
                                        gr.Markdown("### Input Mesh")
                                        cleanup_input = gr.File(
                                            label="Drop mesh file (GLB, OBJ, PLY, STL)",
                                            file_types=[".glb", ".gltf", ".obj", ".ply", ".stl"],
                                        )
                                        cleanup_input_path = gr.Textbox(
                                            label="Or enter path",
                                            placeholder="/path/to/mesh.glb",
                                        )
                                    
                                    with gr.Group():
                                        gr.Markdown("### Cleanup Options")
                                        with gr.Row():
                                            cleanup_target_tris = gr.Slider(
                                                minimum=0, maximum=500000, value=150000, step=10000,
                                                label="Target Triangles (0=no decimation)",
                                                info="Set to 0 to skip decimation entirely",
                                            )
                                            cleanup_smooth = gr.Slider(
                                                minimum=0, maximum=5, value=0, step=1,
                                                label="Pre-Decimation Smooth",
                                                info="Smoothing BEFORE decimation (0 recommended)",
                                            )
                                        with gr.Row():
                                            cleanup_preserve_detail = gr.Checkbox(True, label="Preserve Detail", 
                                                info="Use higher quality decimation (slower but better edges)")
                                            cleanup_post_smooth = gr.Slider(
                                                minimum=0, maximum=5, value=2, step=1,
                                                label="Post-Decimation Smooth",
                                                info="Smoothing AFTER decimation (softens hard edges)",
                                            )
                                        with gr.Row():
                                            cleanup_remove_components = gr.Checkbox(True, label="Remove Small Components")
                                            cleanup_fix_normals = gr.Checkbox(True, label="Fix Normals")
                                        with gr.Row():
                                            cleanup_fill_holes = gr.Checkbox(False, label="Fill Holes")
                                            cleanup_aggressive = gr.Checkbox(False, label="Aggressive Mode")
                                        cleanup_min_ratio = gr.Slider(
                                            minimum=0.001, maximum=0.1, value=0.01, step=0.001,
                                            label="Min Component Ratio (keep components > this % of total)",
                                        )
                                    
                                    with gr.Group():
                                        gr.Markdown("### Output")
                                        cleanup_output_name = gr.Textbox(value="cleaned_mesh", label="Output Name")
                                        cleanup_output_dir = gr.Textbox(value=MESH_DEFAULT_OUTPUT_DIR, label="Output Directory")
                                        cleanup_output_format = gr.Dropdown(["GLB", "OBJ", "PLY", "STL"], value="GLB", label="Format")
                                        with gr.Row():
                                            cleanup_log_params = gr.Checkbox(True, label="Log Parameters", 
                                                info="Save JSON sidecar + CSV experiment log")
                                            cleanup_encode_params = gr.Checkbox(False, label="Encode in Filename",
                                                info="Add params to filename (e.g., mesh_t300k_pd1_ps2.glb)")
                                    
                                    with gr.Row():
                                        cleanup_analyze_btn = gr.Button("Analyze", variant="secondary", size="lg")
                                        cleanup_run_btn = gr.Button("Clean Up", variant="primary", size="lg")
                                    cleanup_status = gr.Textbox(value="Ready", label="Status", interactive=False)
                                
                                with gr.Column(scale=1):
                                    gr.Markdown("### 3D Preview")
                                    cleanup_before_viewer = gr.Model3D(
                                        label="Before (Input)",
                                        height=350,
                                        clear_color=[0.1, 0.1, 0.1, 1.0],
                                    )
                                    cleanup_after_viewer = gr.Model3D(
                                        label="After (Cleaned)",
                                        height=350,
                                        clear_color=[0.1, 0.1, 0.1, 1.0],
                                    )
                                    
                                    gr.Markdown("### Analysis Results")
                                    cleanup_analysis = gr.Textbox(
                                        label="Mesh Info",
                                        lines=6,
                                        interactive=False,
                                        placeholder="Click 'Analyze' to inspect mesh...",
                                    )
                            
                            with gr.Accordion("📋 Cleanup Log", open=False):
                                cleanup_logs = gr.Textbox(lines=8, interactive=False)
                
                # PAGE: SETTINGS
                with gr.TabItem("Settings", id="settings"):
                    gr.HTML("""
                        <div class="page-header">
                            <h1>Settings</h1>
                            <p>Manage RunPod credentials and system configuration.</p>
                        </div>
                    """)
                    
                    gr.Markdown("### RunPod Endpoints")
                    gr.Markdown("Configure API credentials for each model. Credentials are saved locally.")
                    
                    # Row 1: Gen3C/Sharp/Lyra and TRELLIS
                    with gr.Row(elem_classes=["settings-row"]):
                        with gr.Column(scale=1, min_width=300):
                            with gr.Group(elem_classes=["settings-card"]):
                                gr.Markdown("**GEN3C / SHARP / Lyra**")
                                settings_gen3c_endpoint = gr.Textbox(value=DEFAULT_GEN3C_ENDPOINT, label="Endpoint ID")
                                settings_gen3c_key = gr.Textbox(value=DEFAULT_GEN3C_API_KEY, label="API Key", type="password")
                                with gr.Row():
                                    settings_gen3c_test = gr.Button("Test", size="sm")
                                    settings_gen3c_save = gr.Button("Save", size="sm", variant="primary")
                                settings_gen3c_status = gr.Textbox(value="", interactive=False, max_lines=1, show_label=False)
                        
                        with gr.Column(scale=1, min_width=300):
                            with gr.Group(elem_classes=["settings-card"]):
                                gr.Markdown("**TRELLIS.2**")
                                settings_trellis_endpoint = gr.Textbox(value=DEFAULT_TRELLIS_ENDPOINT, label="Endpoint ID")
                                settings_trellis_key = gr.Textbox(value=DEFAULT_TRELLIS_API_KEY, label="API Key", type="password")
                                with gr.Row():
                                    settings_trellis_test = gr.Button("Test", size="sm")
                                    settings_trellis_save = gr.Button("Save", size="sm", variant="primary")
                                settings_trellis_status = gr.Textbox(value="", interactive=False, max_lines=1, show_label=False)
                    
                    # Row 2: Hunyuan and AWS
                    with gr.Row(elem_classes=["settings-row"]):
                        with gr.Column(scale=1, min_width=300):
                            with gr.Group(elem_classes=["settings-card"]):
                                gr.Markdown("**Hunyuan3D**")
                                settings_hunyuan_endpoint = gr.Textbox(value=DEFAULT_HUNYUAN_ENDPOINT, label="Endpoint ID")
                                settings_hunyuan_key = gr.Textbox(value=DEFAULT_HUNYUAN_API_KEY, label="API Key", type="password")
                                with gr.Row():
                                    settings_hunyuan_test = gr.Button("Test", size="sm")
                                    settings_hunyuan_save = gr.Button("Save", size="sm", variant="primary")
                                settings_hunyuan_status = gr.Textbox(value="", interactive=False, max_lines=1, show_label=False)
                        
                        with gr.Column(scale=1, min_width=300):
                            with gr.Group(elem_classes=["settings-card"]):
                                gr.Markdown("**AWS S3**")
                                aws_configured = "Configured" if os.environ.get("AWS_ACCESS_KEY_ID") else "Not configured"
                                gr.Textbox(value=aws_configured, label="Status", interactive=False)
                                gr.Markdown("S3 for files >30MB. Configure in:")
                                gr.Markdown("`~/.config/3d_studio/aws_credentials.env`")
                
                # PAGE: UPDATE
                with gr.TabItem("Update", id="update"):
                    gr.HTML("""
                        <div class="page-header">
                            <h1>Update Tracker</h1>
                            <p>Monitor upstream repository versions and manage updates for all models.</p>
                        </div>
                    """)
                    
                    with gr.Row():
                        update_check_btn = gr.Button("Check Upstream", variant="primary", size="lg")
                        query_deployed_btn = gr.Button("Query Deployed", variant="secondary", size="lg")
                        update_status = gr.Textbox(value="Click 'Check Upstream' to scan GitHub, or 'Query Deployed' to check RunPod", 
                                                   label="Status", interactive=False, scale=2)
                    
                    gr.Markdown("### Version Comparison")
                    
                    # Version tracking table
                    with gr.Row(elem_classes=["settings-row"]):
                        with gr.Column(scale=1, min_width=400):
                            with gr.Group(elem_classes=["settings-card"]):
                                gr.Markdown("**SHARP** (Apple)")
                                gr.Markdown("*Repository:* [apple/ml-sharp](https://github.com/apple/ml-sharp)")
                                with gr.Row():
                                    sharp_upstream = gr.Textbox(value="Not checked", label="Upstream", interactive=False)
                                    sharp_local = gr.Textbox(value=get_local_version("sharp"), label="ARKRUNR Version", interactive=False)
                                sharp_update_info = gr.Markdown("""
**Update Method:** Docker rebuild | **Difficulty:** 🟢 Low  
**Notes:** Pure inference model. Update by pulling latest repo and rebuilding Docker image.
                                """)
                        
                        with gr.Column(scale=1, min_width=400):
                            with gr.Group(elem_classes=["settings-card"]):
                                gr.Markdown("**GEN3C** (NVIDIA)")
                                gr.Markdown("*Repository:* [nv-tlabs/GEN3C](https://github.com/nv-tlabs/GEN3C)")
                                with gr.Row():
                                    gen3c_upstream = gr.Textbox(value="Not checked", label="Upstream", interactive=False)
                                    gen3c_local = gr.Textbox(value=get_local_version("gen3c"), label="ARKRUNR Version", interactive=False)
                                gen3c_update_info = gr.Markdown("""
**Update Method:** Docker rebuild | **Difficulty:** 🟡 Medium  
**Notes:** Complex dependencies. Test thoroughly after updates. May require checkpoint re-download.
                                """)
                    
                    with gr.Row(elem_classes=["settings-row"]):
                        with gr.Column(scale=1, min_width=400):
                            with gr.Group(elem_classes=["settings-card"]):
                                gr.Markdown("**Lyra** (NVIDIA)")
                                gr.Markdown("*Repository:* [nv-tlabs/LYRA](https://github.com/nv-tlabs/LYRA)")
                                with gr.Row():
                                    lyra_upstream = gr.Textbox(value="Not checked", label="Upstream", interactive=False)
                                    lyra_local = gr.Textbox(value=get_local_version("lyra"), label="ARKRUNR Version", interactive=False)
                                lyra_update_info = gr.Markdown("""
**Update Method:** Docker rebuild + checkpoint sync | **Difficulty:** 🔴 High  
**Notes:** Large checkpoints (~15GB). Hardcoded paths may need symlink updates. Test SDG step carefully.
                                """)
                        
                        with gr.Column(scale=1, min_width=400):
                            with gr.Group(elem_classes=["settings-card"]):
                                gr.Markdown("**TRELLIS.2** (Microsoft)")
                                gr.Markdown("*Repository:* [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS)")
                                with gr.Row():
                                    trellis_upstream = gr.Textbox(value="Not checked", label="Upstream", interactive=False)
                                    trellis_local = gr.Textbox(value=get_local_version("trellis"), label="ARKRUNR Version", interactive=False)
                                trellis_update_info = gr.Markdown("""
**Update Method:** Docker rebuild | **Difficulty:** 🟡 Medium  
**Notes:** Separate endpoint. Check for API changes in handler interface.
                                """)
                    
                    with gr.Row(elem_classes=["settings-row"]):
                        with gr.Column(scale=1, min_width=400):
                            with gr.Group(elem_classes=["settings-card"]):
                                gr.Markdown("**Hunyuan3D** (Tencent)")
                                gr.Markdown("*Repository:* [Tencent/Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2)")
                                with gr.Row():
                                    hunyuan_upstream = gr.Textbox(value="Not checked", label="Upstream", interactive=False)
                                    hunyuan_local = gr.Textbox(value=get_local_version("hunyuan"), label="ARKRUNR Version", interactive=False)
                                hunyuan_update_info = gr.Markdown("""
**Update Method:** Docker rebuild OR local update | **Difficulty:** 🟢 Low  
**Notes:** This fork repo. Can run locally or on RunPod. Pull upstream changes carefully.
                                """)
                        
                        with gr.Column(scale=1, min_width=400):
                            with gr.Group(elem_classes=["settings-card"]):
                                gr.Markdown("**Update Guide**")
                                gr.Markdown("""
**General Process:**
1. Check upstream for breaking changes
2. Update Docker image on RunPod
3. Test with simple inference
4. Update version tracking below

**RunPod Update Commands:**
```bash
# SSH to storage pod
ssh runpod@<pod-ip>

# Pull latest code
cd /workspace/<model>
git pull origin main

# Rebuild if needed
docker build -t <image> .
```
                                """)
                
                # =============================================================
                # HELP TAB
                # =============================================================
                with gr.TabItem("Help", id="help"):
                    gr.HTML("""
                        <div class="page-header">
                            <h1>Model Documentation</h1>
                            <p>Comprehensive guide to each 3D generation model and optimal settings for architectural interiors.</p>
                        </div>
                    """)
                    
                    # SHARP Documentation
                    with gr.Accordion("SHARP - Single-Image 3D Gaussian Splatting", open=False):
                        gr.Markdown("""
## SHARP (Apple)

**What it does:** SHARP generates 3D Gaussian Splatting (3DGS) representations from a single image in under 1 second. The output is a PLY file containing Gaussian splats that can be rendered in real-time using 3DGS viewers. SHARP also supports optional video rendering to visualize the 3D reconstruction with camera movement.

**Key features:**
- Fastest single-image to 3DGS (sub-second inference)
- Metric scale output (real-world units)
- Optional video trajectory rendering (CUDA GPU required)

### Output Format

SHARP outputs standard 3DGS PLY files compatible with various Gaussian Splatting viewers:
- Contains: positions, spherical harmonics (colors), scales, rotations, opacities
- Coordinate system: OpenCV (x right, y down, z forward)
- Color space: sRGB (converted from internal linearRGB for compatibility)

### Video Rendering Options

| Setting | Description | Range | Default |
|---------|-------------|-------|---------|
| **Trajectory Type** | Camera movement pattern | rotate_forward, rotate, swipe, shake | rotate_forward |
| **Frames** | Number of video frames | 30-180 | 60 |
| **Repeats** | Trajectory loop count | 1-4 | 1 |
| **Lateral Offset** | Max horizontal/vertical movement | 0.02-0.20 | 0.08 |
| **Zoom/Forward** | Max forward camera movement | 0.05-0.40 | 0.15 |
| **Look-At Mode** | Camera focus behavior | point, ahead | point |

### Trajectory Types Explained

| Type | Description | Best For |
|------|-------------|----------|
| **rotate_forward** | Circular rotation + forward zoom | Most scenes (default) |
| **rotate** | Pure circular rotation | Objects, centered subjects |
| **swipe** | Left-to-right horizontal pan | Wide scenes, panoramas |
| **shake** | Horizontal then vertical shake | Dynamic preview |

### Architectural Interior Settings

```
Trajectory: rotate_forward (shows depth well)
Frames: 90-120 (smooth, longer preview)
Lateral Offset: 0.06-0.10 (moderate movement)
Zoom/Forward: 0.10-0.20 (subtle zoom effect)
Look-At: point (keeps focus on room center)
```

**Tips for Architectural Interiors:**
- SHARP excels at capturing room geometry and furniture
- Use high-resolution input images for best detail
- Video rendering requires CUDA GPU (use RunPod)
- The 3DGS output can be converted to mesh using the MESH tab
                        """)
                    
                    # Gen3C Documentation
                    with gr.Accordion("GEN3C - Image-to-Video with Camera Control", open=False):
                        gr.Markdown("""
## GEN3C (NVIDIA)

**What it does:** Gen3C generates **videos from single images** with precise camera control and 3D consistency. It uses a 3D cache (point clouds from depth prediction) to maintain spatial coherence as the camera moves through the scene. The model excels at creating smooth, realistic camera movements while keeping the scene consistent.

**Key capability:** Unlike other video generators, Gen3C maintains 3D consistency by using depth-based point clouds to guide generation. This means objects stay in place as the camera moves, rather than morphing or popping in/out.

### Key Settings

| Setting | Description | Range | Default |
|---------|-------------|-------|---------|
| **Seed** | Random seed for reproducibility | 0-999999 | 42 |
| **Num Frames** | Output video length (121*N - 1 pattern) | 121-361+ | 121 |
| **Guidance Scale** | Controls generation fidelity | 1.0-15.0 | 1.0 |
| **Trajectory** | Camera movement pattern | left/right/up/down/zoom_in/zoom_out/clockwise/counterclockwise | left |
| **Camera Rotation** | Rotation angle in degrees | 0-360 | varies |
| **Movement Distance** | How far camera moves | 0.1-2.0 | varies |

### Architectural Interior Settings

For architectural interiors with Gen3C:

```
Seed: Fixed for consistency
Num Frames: 121 (or 241 for longer tours)
Guidance Scale: 1.0 (default works well)
Trajectory: clockwise or counterclockwise (for room tours)
           zoom_out (to reveal full space)
           left/right (for corridor walkthroughs)
```

**Tips for Architectural Interiors:**
- Use high-quality input images (1024x1024+) with good depth cues
- Clockwise/counterclockwise trajectories create room tour effect
- Zoom_out reveals the full space from a detail shot
- Works best with images that have clear foreground/background separation
- Enable foreground_masking for better depth handling
- Ideal for: virtual tours, real estate walkthroughs, design visualization
- Output is VIDEO (mp4), not 3D model - use Lyra for 3DGS output
                        """)
                    
                    # Lyra Documentation
                    with gr.Accordion("LYRA - Image/Video to 3DGS/4DGS", open=False):
                        gr.Markdown("""
## LYRA (NVIDIA)

**What it does:** Lyra generates high-quality 3D Gaussian Splats (3DGS) from single images or 4D Gaussian Splats (4DGS) from videos. It uses a sophisticated diffusion-based approach to create detailed, renderable 3D scenes with realistic lighting and materials.

### Key Settings

| Setting | Description | Range | Default |
|---------|-------------|-------|---------|
| **Seed** | Random seed for reproducibility | 0-999999 | 42 |
| **Guidance Scale** | Controls adherence to input | 1.0-20.0 | 7.5 |
| **Inference Steps** | Diffusion steps | 20-100 | 50 |
| **SDG Steps** | Scene Diffusion Generation steps | 100-500 | 250 |
| **Resolution** | Output resolution | 256-1024 | 512 |
| **Mode** | 3DGS (image) or 4DGS (video) | 3dgs/4dgs | 3dgs |

### Architectural Interior Settings

For architectural interiors with Lyra:

```
Seed: Fixed for reproducibility
Guidance Scale: 9.0-12.0 (higher for detailed interiors)
Inference Steps: 75-100
SDG Steps: 350-500 (maximize for complex scenes)
Resolution: 512-1024 (higher for large spaces)
Mode: 3DGS for still images
```

**Tips for Architectural Interiors:**
- High-quality input images are critical
- Works exceptionally well for detailed furniture and fixtures
- SDG step is compute-intensive but crucial for quality
- Output PLY can be converted to mesh via MESH tab
- Ideal for: bedrooms, offices, detailed room corners
- May require longer processing for very detailed scenes
                        """)
                    
                    # TRELLIS Documentation
                    with gr.Accordion("TRELLIS.2 - Structured 3D Generation", open=False):
                        gr.Markdown("""
## TRELLIS.2 (Microsoft)

**What it does:** TRELLIS.2 generates structured 3D assets using a two-stage latent diffusion approach with O-Voxel representation. It produces clean, well-organized meshes with consistent topology and PBR materials, making outputs ideal for further editing in 3D software and game engines.

### Key Settings

| Setting | Description | Range | Default |
|---------|-------------|-------|---------|
| **Resolution** | Voxel resolution for generation | 512, 1024 | 1024 |
| **Guidance Scale** | Controls generation fidelity | 1.0-15.0 | 7.5 |
| **Seed** | Random seed for reproducibility | 0-999999 | Random |
| **Output Format** | GLB (with PBR), OBJ, or PLY | GLB/OBJ/PLY | GLB |

### Architectural Interior Settings

For architectural interiors with TRELLIS.2:

```
Resolution: 1024 (maximize detail)
Guidance Scale: 8.0-10.0
Output Format: GLB (preserves PBR materials)
```

**Tips for Architectural Interiors:**
- Produces cleaner meshes than diffusion-only methods
- Excellent for furniture and architectural elements
- Good topology makes outputs suitable for game engines
- Works well with: chairs, tables, cabinets, fixtures
- PBR materials include Base Color, Roughness, Metallic, Opacity
- Less suited for entire room reconstructions
- Best for individual objects within interiors
                        """)
                    
                    # Hunyuan3D Documentation
                    with gr.Accordion("HUNYUAN3D - Text/Image to 3D", open=False):
                        gr.Markdown("""
## HUNYUAN3D 2.1 (Tencent)

**What it does:** Hunyuan3D 2.1 generates high-fidelity 3D models from images using a scalable diffusion-based pipeline. It features production-ready **Physically-Based Rendering (PBR)** materials with realistic light interactions (metallic reflections, subsurface scattering). Based on [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1).

**Models Available:**
- **Mini Model (Faster):** 3.3B parameters, ~10 GB VRAM, 2-5 minutes
- **Full Model (Higher Quality):** Requires ~29 GB VRAM, 5-15 minutes

### Key Settings

| Setting | Description | Range | Default |
|---------|-------------|-------|---------|
| **Seed** | Random seed for reproducibility | 0-999999 | 42 |
| **Guidance Scale** | Controls adherence to input image | 1.0-15.0 | 9.0 |
| **Inference Steps** | Diffusion sampling steps | 10-100 | 40 |
| **Octree Resolution** | Mesh detail level (higher = more detail, slower) | 128-512 | 380 |
| **Remove Background** | Auto background removal before processing | true/false | true |

### Memory Optimization Options (Local Mode Only)

These options help run Hunyuan3D on GPUs with limited VRAM. They are only relevant for **Local** execution mode - RunPod Serverless handles memory management automatically.

| Setting | Description | VRAM Savings | Trade-off |
|---------|-------------|--------------|-----------|
| **FP16** | Half-precision floating point | ~50% | Minimal quality loss |
| **Attention Slicing** | Process attention in chunks | ~25-40% | Slower generation |
| **CPU Offload** | Move unused layers to RAM | ~60-70% | Much slower |

---

#### FP16 (Half Precision)

**What it does:** Converts model weights and computations from 32-bit (FP32) to 16-bit (FP16) floating point numbers.

**Technical details:**
- Uses `torch.float16` dtype instead of `torch.float32`
- Each number uses 2 bytes instead of 4 bytes
- Supported natively by modern NVIDIA GPUs (Tensor Cores)

**Benefits:**
- Reduces VRAM usage by approximately **50%**
- Often **faster** on GPUs with Tensor Cores (RTX 20/30/40 series)
- Minimal impact on output quality for most use cases

**When to use:**
- ✅ **Always enable** unless you have 24GB+ VRAM and notice quality issues
- ✅ Safe for all architectural interior work
- ⚠️ Very rare edge cases may show minor artifacts in fine details

---

#### Attention Slicing

**What it does:** Splits the attention computation into smaller sequential chunks instead of computing it all at once.

**Technical details:**
- The attention mechanism in diffusion models requires storing large intermediate matrices
- Attention slicing processes these in smaller "slices" sequentially
- Calls `pipeline.enable_attention_slicing()` on the diffusers pipeline

**Benefits:**
- Reduces **peak** VRAM usage by ~25-40%
- Allows running on GPUs that would otherwise run out of memory

**Trade-offs:**
- Generation takes **longer** (10-30% slower)
- Sequential processing can't be parallelized

**When to use:**
- ✅ Enable if you get OOM (Out of Memory) errors with just FP16
- ✅ Good for 8-12 GB GPUs running Mini Model
- ❌ Disable if you have plenty of VRAM and want faster generation

---

#### CPU Offload

**What it does:** Moves model layers to system RAM when not actively being used, then moves them back to GPU when needed.

**Technical details:**
- Uses `pipeline.enable_sequential_cpu_offload()` from diffusers
- Only the currently active layer stays on GPU
- Other layers wait in system RAM

**Benefits:**
- **Dramatically** reduces VRAM requirements (can run on 6-8 GB GPUs)
- Makes it possible to run large models on consumer GPUs

**Trade-offs:**
- **Significantly slower** (2-5x longer generation time)
- Requires sufficient system RAM (16GB+ recommended)
- Heavy CPU-GPU data transfer overhead

**When to use:**
- ✅ Only as a **last resort** for very limited VRAM (6-8 GB GPUs)
- ✅ When you need to run Full Model but only have 12 GB VRAM
- ❌ Avoid if possible - use RunPod Serverless instead for faster results

---

### VRAM Requirements

| Configuration | Approximate VRAM | Generation Time |
|--------------|------------------|-----------------|
| Full Model (no optimization) | 29 GB | 5-10 min |
| Full Model + FP16 | ~15 GB | 5-10 min |
| Full Model + FP16 + Attention Slicing | ~10-12 GB | 7-15 min |
| Mini Model (no optimization) | 21 GB | 2-5 min |
| Mini Model + FP16 | ~10 GB | 2-5 min |
| Mini Model + FP16 + Attention Slicing | ~6-8 GB | 3-7 min |
| Any + CPU Offload | ~6-8 GB | 15-30 min |

### Recommended Settings by GPU

| GPU VRAM | Recommended Configuration |
|----------|--------------------------|
| 24GB+ (RTX 4090, A100) | FP16 ON, others OFF |
| 16GB (RTX 4080, A4000) | FP16 ON, Attention Slicing ON |
| 12GB (RTX 3080, 4070) | FP16 ON, Attention Slicing ON, Mini Model |
| 8GB (RTX 3070, 4060) | All ON, Mini Model only |
| <8GB | Use RunPod Serverless instead |

### Architectural Interior Settings

For architectural interiors with Hunyuan3D:

```
Model: Mini Model (Faster) for iteration, Full Model for final
Seed: Fixed for reproducibility
Guidance Scale: 10.0-12.0 (higher for detailed objects)
Inference Steps: 40-60 (higher for complex shapes)
Octree Resolution: 380-450 (balance detail vs speed)
Remove Background: true (for object isolation)

Memory (if running locally):
FP16: ON (always recommended)
Attention Slicing: ON if needed
CPU Offload: OFF unless necessary
```

**Tips for Architectural Interiors:**
- Excellent for generating furniture from text descriptions
- "Modern minimalist sofa, white leather, chrome legs"
- "Art deco floor lamp, brass finish, geometric shade"
- Works well for: furniture, decor, lighting fixtures
- Can generate from reference images of real furniture
- Combine with other models for complete room scenes
- Use RunPod Serverless to avoid local VRAM limitations
                        """)
                    
                    # MESH Extraction Documentation
                    with gr.Accordion("MESH - 3DGS to Mesh Conversion", open=False):
                        gr.Markdown("""
## MESH Extraction

**What it does:** Converts 3D Gaussian Splat (3DGS) files to traditional mesh formats (GLB/OBJ) using Poisson surface reconstruction. This allows 3DGS outputs from Lyra, SHARP, or other sources to be used in standard 3D software.

### Key Settings

| Setting | Description | Range | Default |
|---------|-------------|-------|---------|
| **Input PLY** | Source 3DGS PLY file | file path | - |
| **Poisson Depth** | Reconstruction detail level | 6-12 | 9 |
| **Point Weight** | Influence of input points | 0.0-10.0 | 4.0 |
| **Scale** | Output mesh scale | 0.1-10.0 | 1.0 |
| **Output Format** | GLB or OBJ | glb/obj | glb |

### Architectural Interior Settings

For architectural interiors:

```
Poisson Depth: 10-11 (higher for detailed interiors)
Point Weight: 3.0-5.0 (balance detail vs smoothness)
Scale: 1.0 (maintain original scale)
Output Format: GLB (preserves vertex colors as texture)
```

**Tips for Architectural Interiors:**
- Higher Poisson depth = more detail but longer processing
- Lower point weight = smoother surfaces (good for walls)
- Higher point weight = more detail preservation (good for furniture)
- Process individual objects separately for best results
- Large room scans may need to be segmented first
                        """)
                    
                    # MESH Cleanup Documentation
                    with gr.Accordion("MESH CLEANUP - Prepare Meshes for Production", open=False):
                        gr.Markdown("""
## Mesh Cleanup

**What it does:** Cleans up meshes generated by AI models (Hunyuan3D, SHARP, TRELLIS.2, etc.) for use in Unity, Blender, or other production environments. Addresses common issues like floating artifacts, excessive polygon counts, and inconsistent normals.

### Key Settings

| Setting | Description | Range | Default |
|---------|-------------|-------|---------|
| **Target Triangles** | Reduce mesh to this triangle count (0 = no decimation) | 0-500,000 | 150,000 |
| **Pre-Decimation Smooth** | Smoothing BEFORE decimation | 0-5 | 0 |
| **Preserve Detail** | Use higher quality decimation algorithm | on/off | on |
| **Post-Decimation Smooth** | Smoothing AFTER decimation (softens hard edges) | 0-5 | 2 |
| **Remove Small Components** | Delete disconnected floating artifacts | on/off | on |
| **Fix Normals** | Recompute and fix inconsistent normals | on/off | on |
| **Fill Holes** | Attempt to close holes in mesh | on/off | off |
| **Aggressive Mode** | Enable all cleanup options with stronger settings | on/off | off |
| **Min Component Ratio** | Keep components with at least this % of total faces | 0.1%-10% | 1% |

### Understanding Each Option

**Target Triangles:**
- 500,000: High detail, large file size
- 300,000: Recommended for detailed architectural interiors
- 150,000: Good balance for most uses
- 50,000: Low poly, fast rendering, mobile-friendly
- 0: No decimation, keep original triangle count

**Preserve Detail (NEW - IMPORTANT):**
When ON, uses quadric decimation which better preserves important edges and detail. Slower but produces much better results. **Always keep ON for architectural interiors.**

When OFF, uses fast_simplification which is faster but creates more faceted/sharp edges.

**Pre-Decimation Smooth:**
Smoothing applied BEFORE decimation. Generally keep at 0 - smoothing before decimation can blur important details.

**Post-Decimation Smooth (NEW - IMPORTANT):**
Smoothing applied AFTER decimation. This is the key to softening the hard/faceted edges that decimation creates.
- 0: No post-smoothing (keep sharp decimated edges)
- 1-2: Light smoothing (recommended for architectural)
- 3-4: Moderate smoothing (good for organic shapes)
- 5: Heavy smoothing (may over-smooth)

**Remove Small Components:**
AI-generated meshes often have floating artifacts - small disconnected pieces that appear as noise. This option removes components smaller than the Min Component Ratio threshold.

**Fix Normals:**
Ensures all face normals point outward consistently. Essential for proper lighting in game engines.

**Fill Holes:**
Attempts to close gaps in the mesh. Use with caution - can create unwanted geometry on intentionally open surfaces.

**Aggressive Mode:**
Enables: 100k triangle target, 2+ smoothing passes, hole filling, 2% component threshold. Use for heavily problematic meshes.

### Architectural Interior Settings

For architectural interior meshes (RECOMMENDED):

```
Target Triangles: 250,000-350,000 (preserve detail)
Pre-Decimation Smooth: 0 (don't blur before decimation)
Preserve Detail: ON (critical for quality)
Post-Decimation Smooth: 2 (soften decimation artifacts)
Remove Small Components: ON (clean up artifacts)
Fix Normals: ON (essential for lighting)
Fill Holes: OFF (preserve intentional openings like windows)
Aggressive Mode: OFF
Min Component Ratio: 1% (default)
```

**Why these settings work:**
- Higher triangle target (300k vs 150k) preserves more detail
- Preserve Detail ON uses better decimation algorithm
- Post-decimation smoothing softens the hard edges created by decimation
- No pre-decimation smoothing keeps original detail intact

**Tips for Architectural Interiors:**
- **Problem: Faceted/sharp edges after cleanup** → Increase Post-Decimation Smooth to 2-3
- **Problem: Lost too much detail** → Increase Target Triangles to 300k-400k
- **Problem: Edges still too sharp** → Enable Preserve Detail
- Higher triangle counts for detailed furniture, lower for walls
- Always fix normals for proper lighting in Unity/Unreal
- Use Analyze first to check mesh quality before cleanup
- For very large meshes, consider splitting into separate objects first

### Analyze vs Clean Up

**Analyze Button:**
- Inspects mesh without modifying
- Shows: vertices, triangles, components, size, watertight status
- Identifies issues: floating components, holes, high poly count
- Use this first to understand what cleanup is needed

**Clean Up Button:**
- Applies selected cleanup operations
- Saves cleaned mesh to output location
- Shows reduction statistics and operations performed
                        """)
                    
                    # General Tips Section
                    with gr.Accordion("General Tips for Architectural Interiors", open=True):
                        gr.Markdown("""
## Workflow Recommendations

### Best Model for Each Task

| Task | Recommended Model | Why |
|------|-------------------|-----|
| **Single room photo → 3D mesh** | SHARP | Fast, high detail, direct GLB output |
| **Single room photo → 3DGS** | Lyra | Best 3D Gaussian Splat quality |
| **Virtual tour video from photo** | Gen3C | Camera-controlled video with 3D consistency |
| **Furniture generation** | Hunyuan3D | Text-to-3D for custom pieces |
| **Clean mesh output** | TRELLIS.2 | Best topology for editing |
| **3DGS to mesh** | MESH tab | Poisson reconstruction |

### Input Image Guidelines

1. **Resolution:** Minimum 1024x1024, ideally 2048x2048
2. **Lighting:** Even, diffuse lighting without harsh shadows
3. **Angle:** 3/4 view captures more depth information
4. **Focus:** Sharp focus throughout, avoid depth-of-field blur
5. **Content:** Single room or object, avoid mirrors/glass

### Recommended Workflow for Complete Rooms

1. **Capture:** Take high-quality photos from key angles
2. **Generate 3D:** Use SHARP or Lyra for 3D reconstruction
3. **Create Video:** Use Gen3C to create walkthrough videos from photos
4. **Convert:** Use MESH tab to convert 3DGS to editable mesh
5. **Enhance:** Add furniture with Hunyuan3D or TRELLIS.2
6. **Composite:** Combine in Blender or Unity

### Output Format Guide

| Format | Best For | Software Compatibility |
|--------|----------|----------------------|
| **GLB** | Textured meshes | Blender, Unity, Unreal, Web |
| **OBJ** | Mesh editing | All 3D software |
| **PLY** | Point clouds, 3DGS | Blender, specialized viewers |
| **3DGS** | Real-time rendering | Gaussian splat viewers |

### Performance Tips

- Start with lower settings to test, then increase for final output
- Use fixed seeds for reproducibility when iterating
- Process during off-peak hours for faster RunPod response
- Save intermediate outputs (PLY) before mesh conversion
                        """)
   
    # =========================================================================
    # NAVIGATION EVENT HANDLERS
    # =========================================================================
    
    # JavaScript to highlight active nav button
    highlight_js = """
    () => {
        const navButtons = ['nav-sharp', 'nav-gen3c', 'nav-lyra', 'nav-trellis', 'nav-hunyuan', 'nav-mesh', 'nav-settings', 'nav-update', 'nav-help'];
        navButtons.forEach(id => {
            const btn = document.getElementById(id);
            if (btn) btn.classList.remove('nav-active');
        });
        const activeBtn = document.getElementById('%s');
        if (activeBtn) activeBtn.classList.add('nav-active');
        return [];
    }
    """
    
    # Navigation buttons - use gr.Tabs.select() to switch tabs
    nav_sharp.click(fn=lambda: gr.Tabs(selected="sharp"), outputs=[page_tabs], js=highlight_js % 'nav-sharp')
    nav_gen3c.click(fn=lambda: gr.Tabs(selected="gen3c"), outputs=[page_tabs], js=highlight_js % 'nav-gen3c')
    nav_lyra.click(fn=lambda: gr.Tabs(selected="lyra"), outputs=[page_tabs], js=highlight_js % 'nav-lyra')
    nav_trellis.click(fn=lambda: gr.Tabs(selected="trellis"), outputs=[page_tabs], js=highlight_js % 'nav-trellis')
    nav_hunyuan.click(fn=lambda: gr.Tabs(selected="hunyuan"), outputs=[page_tabs], js=highlight_js % 'nav-hunyuan')
    nav_mesh.click(fn=lambda: gr.Tabs(selected="mesh"), outputs=[page_tabs], js=highlight_js % 'nav-mesh')
    nav_settings.click(fn=lambda: gr.Tabs(selected="settings"), outputs=[page_tabs], js=highlight_js % 'nav-settings')
    nav_update.click(fn=lambda: gr.Tabs(selected="update"), outputs=[page_tabs], js=highlight_js % 'nav-update')
    nav_help.click(fn=lambda: gr.Tabs(selected="help"), outputs=[page_tabs], js=highlight_js % 'nav-help')
    
    # =========================================================================
    # IMAGE INPUT HANDLERS
    # =========================================================================
    
    input_image.change(
        fn=lambda img, scale: update_image_info_display(img, scale),
        inputs=[input_image, image_scale],
        outputs=[image_info],
    )
    
    image_scale.change(
        fn=lambda img, scale: update_image_info_display(img, scale),
        inputs=[input_image, image_scale],
        outputs=[image_info],
    )
    
    # =========================================================================
    # SHARP EVENT HANDLERS
    # =========================================================================
    
    def convert_3dgs_to_preview_glb(input_ply: str) -> str:
        """Convert 3DGS PLY to GLB point cloud mesh for preview.
        
        Extracts positions and colors from 3DGS format and creates a 
        viewable point cloud in GLB format.
        """
        if not input_ply or not os.path.exists(input_ply):
            return None
        
        try:
            import numpy as np
            from plyfile import PlyData
            import trimesh
            
            # Read the 3DGS PLY
            plydata = PlyData.read(input_ply)
            vertex = plydata['vertex']
            
            # Extract XYZ positions
            x = np.array(vertex['x'])
            y = np.array(vertex['y'])
            z = np.array(vertex['z'])
            
            # Try to get colors from spherical harmonics (f_dc_0, f_dc_1, f_dc_2)
            # 3DGS stores colors as SH coefficients where DC component = color * C0
            # The formula is: color = sigmoid(sh_dc) for some implementations
            # or color = sh_dc * C0 + 0.5 for others
            if 'f_dc_0' in vertex.data.dtype.names:
                # Get raw SH DC values
                sh_r = np.array(vertex['f_dc_0'])
                sh_g = np.array(vertex['f_dc_1'])
                sh_b = np.array(vertex['f_dc_2'])
                
                # Method 1: Direct sigmoid conversion (common in 3DGS)
                def sigmoid(x):
                    return 1 / (1 + np.exp(-x))
                
                r = np.clip(sigmoid(sh_r) * 255, 0, 255).astype(np.uint8)
                g = np.clip(sigmoid(sh_g) * 255, 0, 255).astype(np.uint8)
                b = np.clip(sigmoid(sh_b) * 255, 0, 255).astype(np.uint8)
                
                print(f"[SHARP] SH color range: R({sh_r.min():.2f}-{sh_r.max():.2f}), G({sh_g.min():.2f}-{sh_g.max():.2f}), B({sh_b.min():.2f}-{sh_b.max():.2f})")
                print(f"[SHARP] RGB range: R({r.min()}-{r.max()}), G({g.min()}-{g.max()}), B({b.min()}-{b.max()})")
                
            elif 'red' in vertex.data.dtype.names:
                r = np.array(vertex['red']).astype(np.uint8)
                g = np.array(vertex['green']).astype(np.uint8)
                b = np.array(vertex['blue']).astype(np.uint8)
            else:
                # Fallback: use height-based coloring for visualization
                z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
                r = np.clip(z_norm * 255, 0, 255).astype(np.uint8)
                g = np.clip((1 - z_norm) * 200 + 55, 0, 255).astype(np.uint8)
                b = np.full_like(x, 150, dtype=np.uint8)
            
            # Downsample for preview (100k points max for performance)
            max_points = 100000
            if len(x) > max_points:
                indices = np.random.choice(len(x), max_points, replace=False)
                x, y, z = x[indices], y[indices], z[indices]
                r, g, b = r[indices], g[indices], b[indices]
            
            # Stack into vertices array
            vertices = np.column_stack([x, y, z])
            colors = np.column_stack([r, g, b, np.full(len(r), 255, dtype=np.uint8)])  # RGBA
            
            # Create a point cloud and convert to GLB
            cloud = trimesh.PointCloud(vertices=vertices, colors=colors)
            
            # Save as GLB
            preview_path = input_ply.replace('.ply', '_preview.glb')
            cloud.export(preview_path)
            
            print(f"[SHARP] Preview GLB created: {preview_path} ({len(vertices)} points)")
            return preview_path
            
        except Exception as e:
            print(f"[SHARP] Preview conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def sharp_generate_with_preview(*args):
        """Wrapper that returns a preview-compatible GLB for 3D viewer."""
        result = handle_sharp_generation(*args)
        output_path, logs, progress = result
        
        print(f"[SHARP DEBUG] output_path: {output_path}")
        print(f"[SHARP DEBUG] output_path exists: {os.path.exists(output_path) if output_path else 'N/A'}")
        
        # Convert to GLB point cloud for preview (gr.Model3D works better with GLB)
        preview_path = convert_3dgs_to_preview_glb(output_path) if output_path else None
        
        print(f"[SHARP DEBUG] preview_path: {preview_path}")
        print(f"[SHARP DEBUG] preview_path exists: {os.path.exists(preview_path) if preview_path else 'N/A'}")
        
        return output_path, logs, progress, preview_path
    
    sharp_generate_btn.click(
        fn=sharp_generate_with_preview,
        inputs=[
            input_image, image_scale, gr.State("RunPod Serverless"),
            settings_gen3c_endpoint, settings_gen3c_key, gr.State(False),
            sharp_output_name, sharp_output_dir,
            gr.State("rotate_forward"), gr.State(60), gr.State(1),
            gr.State(0.08), gr.State(0.15), gr.State("point"),
            global_log_params, global_encode_params,
        ],
        outputs=[output_display, sharp_logs, sharp_progress, sharp_3d_viewer],
    )
    
    # SHARP Video Mode Toggle - show/hide PLY path input
    def toggle_sharp_video_mode(mode):
        """Toggle visibility of PLY path input based on mode."""
        return gr.update(visible=mode == "Render from existing PLY")
    
    sharp_video_mode.change(
        fn=toggle_sharp_video_mode,
        inputs=[sharp_video_mode],
        outputs=[sharp_video_ply_path],
    )
    
    # SHARP Video Render Handler
    def handle_sharp_video_render(
        mode, ply_path, image_path, image_scale,
        endpoint_id, api_key,
        trajectory_type, num_steps, num_repeats,
        max_disparity, max_zoom, lookat_mode,
        output_name, output_dir,
        log_params, encode_params,
    ):
        """Handle SHARP video rendering - either from new generation or existing PLY."""
        if mode == "Render from existing PLY":
            # Render from existing PLY (not yet supported via CLI)
            return None, "⚠️ Rendering from existing PLY is not yet supported via SHARP CLI.\nPlease use 'Generate PLY + Video' mode.", "Not supported"
        else:
            # Generate PLY + Video
            result = handle_sharp_generation(
                image_path=image_path,
                image_scale=image_scale,
                exec_mode="RunPod Serverless",
                endpoint_id=endpoint_id,
                api_key=api_key,
                render_video=True,
                output_name=output_name,
                output_dir=output_dir,
                trajectory_type=trajectory_type,
                num_steps=int(num_steps),
                num_repeats=int(num_repeats),
                max_disparity=float(max_disparity),
                max_zoom=float(max_zoom),
                lookat_mode=lookat_mode,
                log_params=log_params,
                encode_params=encode_params,
            )
            output_path, logs, progress = result
            
            # Check if we got a video file
            video_path = None
            if output_path and output_path.endswith('.mp4'):
                video_path = output_path
            elif output_path:
                # Check if video was generated alongside PLY
                potential_video = output_path.replace('.ply', '.mp4')
                if os.path.exists(potential_video):
                    video_path = potential_video
            
            return video_path, logs, progress
    
    sharp_render_video_btn.click(
        fn=handle_sharp_video_render,
        inputs=[
            sharp_video_mode, sharp_video_ply_path,
            input_image, image_scale,
            settings_gen3c_endpoint, settings_gen3c_key,
            sharp_trajectory_type, sharp_num_steps, sharp_num_repeats,
            sharp_max_disparity, sharp_max_zoom, sharp_lookat_mode,
            sharp_video_output_name, sharp_video_output_dir,
            global_log_params, global_encode_params,
        ],
        outputs=[sharp_video_preview, sharp_video_logs, sharp_video_progress],
    )
    
    # =========================================================================
    # GEN3C EVENT HANDLERS
    # =========================================================================
    
    gen3c_generate_btn.click(
        fn=handle_gen3c_generation,
        inputs=[
            input_image, image_scale, gr.State("RunPod Serverless"),
            settings_gen3c_endpoint, settings_gen3c_key,
            gen3c_guidance, gen3c_frames, gen3c_trajectory,
            gen3c_movement_distance, gen3c_camera_rotation,
            gen3c_foreground,
            gen3c_video_name, gen3c_seed, gen3c_output_dir,
            global_log_params, global_encode_params,
        ],
        outputs=[output_video, gen3c_logs, gen3c_progress],
    )
    
    # =========================================================================
    # LYRA EVENT HANDLERS
    # =========================================================================
    
    lyra_generate_btn.click(
        fn=handle_lyra_generation,
        inputs=[
            input_image, input_video, image_scale,
            settings_gen3c_endpoint, settings_gen3c_key,
            lyra_mode, lyra_views, lyra_motion,
            lyra_multi_traj, lyra_fg_mask,
            lyra_gaussians, lyra_seed,
            lyra_output_name, lyra_output_dir,
            lyra_out_ply, lyra_out_video,
            global_log_params, global_encode_params,
        ],
        outputs=[output_display, lyra_logs, lyra_progress, lyra_ply_path],
    )
    
    # Lyra PLY dropdown
    def refresh_lyra_ply():
        return gr.update(choices=scan_for_lyra_ply_files())
    
    lyra_refresh_btn.click(fn=refresh_lyra_ply, outputs=[lyra_ply_dropdown])
    
    lyra_ply_dropdown.change(
        fn=lambda x: x.rsplit(" (", 1)[0] if x and " (" in x else (x or ""),
        inputs=[lyra_ply_dropdown],
        outputs=[lyra_ply_path],
    )
    
    # =========================================================================
    # TRELLIS EVENT HANDLERS
    # =========================================================================
    
    trellis_generate_btn.click(
        fn=handle_trellis_generation,
        inputs=[
            input_image, image_scale,
            settings_trellis_endpoint, settings_trellis_key,
            trellis_resolution, trellis_guidance, trellis_seed,
            trellis_output_name, trellis_output_dir, trellis_format,
            global_log_params, global_encode_params,
        ],
        outputs=[output_display, trellis_logs, trellis_progress],
    )
    
    # =========================================================================
    # HUNYUAN EVENT HANDLERS
    # =========================================================================
    
    def hunyuan_generate_with_preview(*args):
        """Wrapper that returns the GLB path for 3D viewer."""
        result = handle_hunyuan_generation(*args)
        output_path, logs, progress = result
        # Return path for 3D viewer (gr.Model3D accepts file path directly)
        return output_path, logs, progress, output_path
    
    hunyuan_generate_btn.click(
        fn=hunyuan_generate_with_preview,
        inputs=[
            input_image, image_scale, hunyuan_exec_mode,
            settings_hunyuan_endpoint, settings_hunyuan_key,
            hunyuan_guidance, hunyuan_steps, hunyuan_seed, hunyuan_octree,
            hunyuan_model,
            hunyuan_fp16, hunyuan_attn_slice, hunyuan_cpu_offload, hunyuan_remove_bg,
            hunyuan_output_name, hunyuan_output_dir,
            global_log_params, global_encode_params,
        ],
        outputs=[output_display, hunyuan_logs, hunyuan_progress, hunyuan_3d_viewer],
    )
    
    # =========================================================================
    # MESH EXTRACTION EVENT HANDLERS
    # =========================================================================
    
    mesh_extract_btn.click(
        fn=handle_mesh_extraction,
        inputs=[
            mesh_ply_dropdown, mesh_format,
            gr.State("SuGaR"), gr.State("dn_consistency"),
            mesh_quality, mesh_poisson_depth, mesh_decimate,
            gr.State(False), gr.State("2048"), gr.State("short"),
            gr.State(0.01), gr.State(32),
            mesh_output_name, mesh_output_format, mesh_output_dir,
            settings_gen3c_endpoint, settings_gen3c_key,
        ],
        outputs=[output_display, mesh_logs, mesh_progress],
    )
    
    # Mesh PLY dropdown
    def refresh_mesh_ply():
        return gr.update(choices=scan_for_ply_files())
    
    mesh_refresh_btn.click(fn=refresh_mesh_ply, outputs=[mesh_ply_dropdown])
    
    # =========================================================================
    # MESH CLEANUP EVENT HANDLERS
    # =========================================================================
    
    def get_cleanup_output_name(file_input, path_input):
        """Generate default output name from input file."""
        import os
        if file_input is not None:
            input_path = file_input.name if hasattr(file_input, 'name') else str(file_input)
        elif path_input and path_input.strip():
            input_path = path_input.strip()
        else:
            return "cleaned_mesh"
        
        # Get base name without extension
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        return f"{base_name}-CLEAN"
    
    def get_input_path_for_viewer(file_input, path_input):
        """Get the input path for the 3D viewer."""
        if file_input is not None:
            return file_input.name if hasattr(file_input, 'name') else str(file_input)
        elif path_input and path_input.strip():
            return path_input.strip()
        return None
    
    cleanup_input.change(
        fn=lambda f: (get_cleanup_output_name(f, None), get_input_path_for_viewer(f, None)),
        inputs=[cleanup_input],
        outputs=[cleanup_output_name, cleanup_before_viewer],
    )
    
    cleanup_input_path.change(
        fn=lambda p: (get_cleanup_output_name(None, p), p.strip() if p else None),
        inputs=[cleanup_input_path],
        outputs=[cleanup_output_name, cleanup_before_viewer],
    )
    
    cleanup_analyze_btn.click(
        fn=handle_mesh_analyze,
        inputs=[cleanup_input, cleanup_input_path],
        outputs=[cleanup_analysis, cleanup_status],
    )
    
    def cleanup_with_preview(*args):
        """Wrapper that returns output path for after viewer."""
        result = handle_mesh_cleanup(*args)
        output_path, logs, status = result
        return output_path, logs, status, output_path
    
    cleanup_run_btn.click(
        fn=cleanup_with_preview,
        inputs=[
            cleanup_input, cleanup_input_path,
            cleanup_target_tris, cleanup_smooth,
            cleanup_preserve_detail, cleanup_post_smooth,
            cleanup_remove_components, cleanup_fix_normals,
            cleanup_fill_holes, cleanup_aggressive,
            cleanup_min_ratio,
            cleanup_output_name, cleanup_output_dir, cleanup_output_format,
            cleanup_log_params, cleanup_encode_params,
        ],
        outputs=[output_display, cleanup_logs, cleanup_status, cleanup_after_viewer],
    )
    
    # =========================================================================
    # SETTINGS EVENT HANDLERS
    # =========================================================================
    
    settings_gen3c_test.click(
        fn=check_serverless_status,
        inputs=[settings_gen3c_endpoint, settings_gen3c_key],
        outputs=[settings_gen3c_status],
    )
    
    settings_gen3c_save.click(
        fn=lambda e, k: save_serverless_credentials(e, k, "gen3c")[0],
        inputs=[settings_gen3c_endpoint, settings_gen3c_key],
        outputs=[settings_gen3c_status],
    )
    
    settings_trellis_test.click(
        fn=check_trellis_status,
        inputs=[settings_trellis_endpoint, settings_trellis_key],
        outputs=[settings_trellis_status],
    )
    
    settings_trellis_save.click(
        fn=lambda e, k: save_serverless_credentials(e, k, "trellis")[0],
        inputs=[settings_trellis_endpoint, settings_trellis_key],
        outputs=[settings_trellis_status],
    )
    
    settings_hunyuan_test.click(
        fn=check_hunyuan_runpod_status,
        inputs=[settings_hunyuan_endpoint, settings_hunyuan_key],
        outputs=[settings_hunyuan_status],
    )
    
    settings_hunyuan_save.click(
        fn=lambda e, k: save_serverless_credentials(e, k, "hunyuan")[0],
        inputs=[settings_hunyuan_endpoint, settings_hunyuan_key],
        outputs=[settings_hunyuan_status],
    )
    
    # =========================================================================
    # UPDATE PAGE HANDLERS
    # =========================================================================
    
    def check_versions_handler():
        """Check all upstream versions (doesn't overwrite local versions)."""
        results = check_all_versions()
        status = format_version_check_status(results)
        return (
            results.get("sharp", {}).get("upstream", "Error"),
            results.get("gen3c", {}).get("upstream", "Error"),
            results.get("lyra", {}).get("upstream", "Error"),
            results.get("trellis", {}).get("upstream", "Error"),
            results.get("hunyuan", {}).get("upstream", "Error"),
            f"Checked at {datetime.now().strftime('%H:%M:%S')} - {status}",
        )
    
    update_check_btn.click(
        fn=check_versions_handler,
        outputs=[
            sharp_upstream,
            gen3c_upstream,
            lyra_upstream,
            trellis_upstream,
            hunyuan_upstream,
            update_status,
        ],
    )
    
    def query_deployed_handler():
        """Query RunPod endpoints for deployed versions."""
        # Get endpoint credentials from config
        config = _load_runpod_config()
        
        unified_endpoint = config.get("gen3c_endpoint_id", "")
        unified_api_key = config.get("gen3c_api_key", "")
        trellis_endpoint = config.get("trellis_endpoint_id", "")
        trellis_api_key = config.get("trellis_api_key", "")
        hunyuan_endpoint = config.get("hunyuan_endpoint_id", "")
        hunyuan_api_key = config.get("hunyuan_api_key", "")
        
        if not unified_endpoint or not unified_api_key:
            return (
                "Not configured",
                "Not configured",
                "Not configured",
                "Not configured",
                "Not configured",
                "Error: Configure RunPod credentials in Settings first",
            )
        
        # Query endpoints
        results = query_runpod_versions(
            unified_endpoint=unified_endpoint,
            unified_api_key=unified_api_key,
            trellis_endpoint=trellis_endpoint,
            trellis_api_key=trellis_api_key,
            hunyuan_endpoint=hunyuan_endpoint,
            hunyuan_api_key=hunyuan_api_key,
        )
        
        # Update local version tracking
        update_local_versions_from_runpod(results)
        
        # Format results
        def get_display(model):
            info = results.get(model, {})
            if info.get("error"):
                return f"Error: {info['error']}"
            return info.get("display", "Not available")
        
        status_parts = []
        for model in ["sharp", "gen3c", "lyra", "trellis", "hunyuan"]:
            info = results.get(model, {})
            if info.get("error"):
                status_parts.append(f"✗ {model.upper()}")
            elif info.get("display") not in ["Not queried", "unknown"]:
                status_parts.append(f"✓ {model.upper()}")
            else:
                status_parts.append(f"? {model.upper()}")
        
        return (
            get_display("sharp"),
            get_display("gen3c"),
            get_display("lyra"),
            get_display("trellis"),
            get_display("hunyuan"),
            f"Queried at {datetime.now().strftime('%H:%M:%S')} - {' | '.join(status_parts)}",
        )
    
    query_deployed_btn.click(
        fn=query_deployed_handler,
        outputs=[
            sharp_local,
            gen3c_local,
            lyra_local,
            trellis_local,
            hunyuan_local,
            update_status,
        ],
    )
    
    # =========================================================================
    # SYSTEM METRICS TIMER
    # =========================================================================
    
    timer = gr.Timer(2.0)
    timer.tick(
        fn=lambda: format_system_metrics(get_system_metrics()),
        outputs=[system_metrics],
    )
    
    # =========================================================================
    # INITIAL PAGE LOAD - Highlight SHARP button
    # =========================================================================
    
    demo.load(
        fn=lambda: None,
        outputs=None,
        js="""
        () => {
            setTimeout(() => {
                const sharpBtn = document.getElementById('nav-sharp');
                if (sharpBtn) sharpBtn.classList.add('nav-active');
            }, 500);
            return [];
        }
        """
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("[STARTUP] Initializing 3D Generation Studio v2.2 (Sidebar UI)...")
    
    start_monitoring()
    time.sleep(0.5)
    
    print("[STARTUP] Ready!")
    
    demo.launch(
        server_port=5684,  # Different port from 2d3d.py (5683)
        share=False,
        allowed_paths=[
            "/srv/searidge_share/outputs",
            "/tmp",
        ],
        css=CUSTOM_CSS,
    )

