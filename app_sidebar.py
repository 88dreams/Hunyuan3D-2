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
) -> Tuple[Optional[str], str, str]:
    """Handle SHARP generation."""
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


def handle_gen3c_generation(
    image_path: Union[str, None],
    image_scale: Union[float, int],
    exec_mode: str,
    endpoint_id: str,
    api_key: str,
    guidance: float,
    frames: Union[str, int],
    trajectory: str,
    foreground_mask: bool,
    video_name: str,
    seed: Optional[int],
    output_dir: str,
) -> Tuple[Optional[str], str, str]:
    """Handle GEN3C generation."""
    scale_value = clamp_scale_value(image_scale)
    scaled_path, temp_scaled = maybe_downscale_image(image_path, scale_value)
    effective_image_path = scaled_path or image_path
    resolved_output_dir = output_dir.strip() if output_dir else GEN3C_DEFAULT_OUTPUT_DIR
    
    try:
        return run_gen3c_serverless(
            image_path=effective_image_path,
            endpoint_id=endpoint_id,
            api_key=api_key,
            guidance=guidance,
            frames=frames,
            trajectory=trajectory,
            foreground_masking=foreground_mask,
            video_name=video_name,
            seed=seed,
            output_dir=resolved_output_dir,
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
        
        output_path, logs, progress = result
        ply_path = output_path if output_path and output_path.endswith(".ply") else ""
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
) -> Tuple[Optional[str], str, str]:
    """Handle TRELLIS.2 generation."""
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
    """Handle Hunyuan3D generation."""
    scale_value = clamp_scale_value(image_scale)
    scaled_path, temp_scaled = maybe_downscale_image(image_path, scale_value)
    effective_image_path = scaled_path or image_path

    try:
        if exec_mode == "RunPod Serverless":
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
                            <p>Apple's fast 3D Gaussian Splatting from a single image. Generates PLY in ~60 seconds.</p>
                        </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Group():
                                gr.Markdown("### Output Settings")
                                sharp_output_name = gr.Textbox(value="sharp_output", label="Output Name")
                                sharp_output_dir = gr.Textbox(value=SHARP_DEFAULT_OUTPUT_DIR, label="Output Directory")
                            
                            sharp_generate_btn = gr.Button("Generate PLY", variant="primary", size="lg")
                            sharp_progress = gr.Textbox(value="Ready", label="Status", interactive=False)
                        
                        with gr.Column(scale=1):
                            with gr.Accordion("🔌 RunPod Connection", open=False):
                                sharp_endpoint = gr.Textbox(value=DEFAULT_SHARP_ENDPOINT, label="Endpoint ID")
                                sharp_api_key = gr.Textbox(value=DEFAULT_SHARP_API_KEY, label="API Key", type="password")
                                with gr.Row():
                                    sharp_check_btn = gr.Button("Check", size="sm")
                                    sharp_save_btn = gr.Button("Save", size="sm")
                                sharp_status = gr.Textbox(value="", label="Connection", interactive=False, max_lines=1)
                    
                    with gr.Accordion("📋 Logs", open=False):
                        sharp_logs = gr.Textbox(label="Generation Logs", lines=8, interactive=False)
                
                # PAGE: GEN3C
                with gr.TabItem("GEN3C", id="gen3c"):
                    gr.HTML("""
                        <div class="page-header">
                            <h1>GEN3C</h1>
                            <p>NVIDIA's 3D-consistent video generation from a single image. Creates orbital camera videos.</p>
                        </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Group():
                                gr.Markdown("### Video Settings")
                                with gr.Row():
                                    gen3c_trajectory = gr.Dropdown(
                                        choices=["orbit", "left", "right", "up", "down", "zoom_in", "zoom_out"],
                                        value="orbit",
                                        label="Camera Trajectory",
                                    )
                                    gen3c_frames = gr.Dropdown(
                                        choices=["61", "121", "241"],
                                        value="121",
                                        label="Frames",
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
                        
                        with gr.Column(scale=1):
                            with gr.Accordion("🔌 RunPod Connection", open=False):
                                gen3c_endpoint = gr.Textbox(value=DEFAULT_GEN3C_ENDPOINT, label="Endpoint ID")
                                gen3c_api_key = gr.Textbox(value=DEFAULT_GEN3C_API_KEY, label="API Key", type="password")
                                with gr.Row():
                                    gen3c_check_btn = gr.Button("Check", size="sm")
                                    gen3c_save_btn = gr.Button("Save", size="sm")
                                gen3c_status = gr.Textbox(value="", label="Connection", interactive=False, max_lines=1)
                    
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
                                
                                with gr.Column(scale=1):
                                    with gr.Accordion("🔌 RunPod", open=False):
                                        lyra_endpoint = gr.Textbox(value=DEFAULT_LYRA_ENDPOINT, label="Endpoint")
                                        lyra_api_key = gr.Textbox(value=DEFAULT_LYRA_API_KEY, label="API Key", type="password")
                                        with gr.Row():
                                            lyra_check_btn = gr.Button("Check", size="sm")
                                            lyra_save_btn = gr.Button("Save", size="sm")
                                        lyra_status = gr.Textbox(value="", interactive=False, max_lines=1)
                            
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
                        
                        with gr.Column(scale=1):
                            with gr.Accordion("🔌 RunPod", open=False):
                                trellis_endpoint = gr.Textbox(value=DEFAULT_TRELLIS_ENDPOINT, label="Endpoint")
                                trellis_api_key = gr.Textbox(value=DEFAULT_TRELLIS_API_KEY, label="API Key", type="password")
                                with gr.Row():
                                    trellis_check_btn = gr.Button("Check", size="sm")
                                    trellis_save_btn = gr.Button("Save", size="sm")
                                trellis_status = gr.Textbox(value="", interactive=False, max_lines=1)
                    
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
                        with gr.Column(scale=2):
                            hunyuan_model = gr.Dropdown(["mini", "turbo", "full"], value="mini", label="Model")
                            
                            with gr.Accordion("⚙️ Advanced", open=False):
                                with gr.Row():
                                    hunyuan_guidance = gr.Slider(1.0, 10.0, 5.0, label="Guidance")
                                    hunyuan_steps = gr.Slider(10, 100, 40, step=5, label="Steps")
                                with gr.Row():
                                    hunyuan_octree = gr.Slider(256, 512, 380, step=1, label="Octree Resolution")
                                    hunyuan_seed = gr.Number(None, label="Seed", precision=0)
                                with gr.Row():
                                    hunyuan_fp16 = gr.Checkbox(True, label="FP16")
                                    hunyuan_attn_slice = gr.Checkbox(True, label="Attention Slicing")
                                    hunyuan_cpu_offload = gr.Checkbox(True, label="CPU Offload")
                                hunyuan_remove_bg = gr.Checkbox(True, label="Remove Background")
                            
                            with gr.Group():
                                hunyuan_output_name = gr.Textbox(value="hunyuan_output", label="Output Name")
                                hunyuan_output_dir = gr.Textbox(value="/srv/searidge_share/outputs/hunyuan", label="Output Dir")
                            
                            hunyuan_exec_mode = gr.Radio(["RunPod Serverless", "Local"], value="RunPod Serverless", label="Mode", visible=False)
                            hunyuan_generate_btn = gr.Button("Generate Mesh", variant="primary", size="lg")
                            hunyuan_progress = gr.Textbox(value="Ready", label="Status", interactive=False)
                        
                        with gr.Column(scale=1):
                            with gr.Accordion("🔌 RunPod", open=False):
                                hunyuan_endpoint = gr.Textbox(value=DEFAULT_HUNYUAN_ENDPOINT, label="Endpoint")
                                hunyuan_api_key = gr.Textbox(value=DEFAULT_HUNYUAN_API_KEY, label="API Key", type="password")
                                with gr.Row():
                                    hunyuan_check_btn = gr.Button("Check", size="sm")
                                    hunyuan_save_btn = gr.Button("Save", size="sm")
                                hunyuan_status = gr.Textbox(value="", interactive=False, max_lines=1)
                    
                    with gr.Accordion("📋 Logs", open=False):
                        hunyuan_logs = gr.Textbox(lines=8, interactive=False)
                
                # PAGE: MESH EXTRACTION
                with gr.TabItem("Mesh", id="mesh"):
                    gr.HTML("""
                        <div class="page-header">
                            <h1>Mesh Extraction</h1>
                            <p>Convert 3D Gaussian Splatting to GLB mesh using Poisson surface reconstruction.</p>
                        </div>
                    """)
                    
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
                        
                        with gr.Column(scale=1):
                            with gr.Accordion("🔌 RunPod", open=False):
                                mesh_endpoint = gr.Textbox(value=DEFAULT_MESH_ENDPOINT, label="Endpoint")
                                mesh_api_key = gr.Textbox(value=DEFAULT_MESH_API_KEY, label="API Key", type="password")
                                with gr.Row():
                                    mesh_check_btn = gr.Button("Check", size="sm")
                                    mesh_save_btn = gr.Button("Save", size="sm")
                                mesh_status = gr.Textbox(value="", interactive=False, max_lines=1)
                    
                    with gr.Accordion("📋 Logs", open=False):
                        mesh_logs = gr.Textbox(lines=8, interactive=False)
                
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
                    with gr.Accordion("SHARP - Single-Image 3D Reconstruction", open=False):
                        gr.Markdown("""
## SHARP (Apple)

**What it does:** SHARP reconstructs detailed 3D meshes from a single image using a feed-forward neural network. It excels at capturing fine geometric details and textures, producing high-quality meshes suitable for rendering and further editing.

### Key Settings

| Setting | Description | Range | Default |
|---------|-------------|-------|---------|
| **Seed** | Random seed for reproducibility | 0-999999 | 42 |
| **Guidance Scale** | Controls adherence to input image | 1.0-20.0 | 7.5 |
| **Inference Steps** | Diffusion steps (more = higher quality, slower) | 10-100 | 50 |
| **Output Format** | GLB (textured mesh) or OBJ | glb/obj | glb |

### Architectural Interior Settings

For architectural interiors, SHARP works best with:

```
Seed: Any (for reproducibility, use fixed seed)
Guidance Scale: 10.0-12.0 (higher for more faithful reconstruction)
Inference Steps: 75-100 (maximize detail for complex scenes)
Output Format: GLB (preserves textures)
```

**Tips for Architectural Interiors:**
- Use high-resolution input images (1024x1024 minimum)
- Ensure good lighting in source photo - avoid harsh shadows
- Works best with single-room views, not panoramas
- Ideal for furniture, fixtures, and room corners
- May struggle with very large open spaces or complex reflections
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

**What it does:** TRELLIS.2 generates structured 3D assets using a latent diffusion approach. It produces clean, well-organized meshes with consistent topology, making outputs ideal for further editing in 3D software.

### Key Settings

| Setting | Description | Range | Default |
|---------|-------------|-------|---------|
| **Seed** | Random seed for reproducibility | 0-999999 | 42 |
| **Guidance Scale** | Controls generation fidelity | 1.0-15.0 | 7.5 |
| **Inference Steps** | Diffusion steps | 20-100 | 50 |
| **Sparse Steps** | Sparse structure generation | 10-50 | 20 |
| **SLAT Steps** | SLAT refinement steps | 10-50 | 20 |
| **Output Format** | GLB, OBJ, or 3DGS | glb/obj/3dgs | glb |

### Architectural Interior Settings

For architectural interiors with TRELLIS.2:

```
Seed: Fixed for consistency
Guidance Scale: 8.0-10.0
Inference Steps: 50-75
Sparse Steps: 25-30 (more for complex geometry)
SLAT Steps: 25-30 (more for refined surfaces)
Output Format: GLB (for textured meshes)
```

**Tips for Architectural Interiors:**
- Produces cleaner meshes than diffusion-only methods
- Excellent for furniture and architectural elements
- Good topology makes outputs suitable for game engines
- Works well with: chairs, tables, cabinets, fixtures
- Less suited for entire room reconstructions
- Best for individual objects within interiors
                        """)
                    
                    # Hunyuan3D Documentation
                    with gr.Accordion("HUNYUAN3D - Text/Image to 3D", open=False):
                        gr.Markdown("""
## HUNYUAN3D (Tencent)

**What it does:** Hunyuan3D generates 3D models from text prompts or images using a multi-stage pipeline. It can create both meshes and Gaussian splats, offering flexibility in output format and quality.

### Key Settings

| Setting | Description | Range | Default |
|---------|-------------|-------|---------|
| **Seed** | Random seed for reproducibility | 0-999999 | 42 |
| **Guidance Scale** | Controls prompt adherence | 1.0-20.0 | 7.5 |
| **Inference Steps** | Diffusion steps | 20-100 | 50 |
| **Octree Depth** | Mesh resolution (higher = more detail) | 6-10 | 8 |
| **Remove Background** | Auto background removal | true/false | true |
| **Output Format** | GLB, OBJ, or PLY | glb/obj/ply | glb |

### Architectural Interior Settings

For architectural interiors with Hunyuan3D:

```
Seed: Fixed for reproducibility
Guidance Scale: 10.0-12.0 (higher for detailed objects)
Inference Steps: 75-100
Octree Depth: 9-10 (maximize for architectural detail)
Remove Background: true (for object isolation)
Output Format: GLB (for textured meshes)
```

**Tips for Architectural Interiors:**
- Excellent for generating furniture from text descriptions
- "Modern minimalist sofa, white leather, chrome legs"
- "Art deco floor lamp, brass finish, geometric shade"
- Works well for: furniture, decor, lighting fixtures
- Can generate from reference images of real furniture
- Combine with other models for complete room scenes
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
    
    sharp_generate_btn.click(
        fn=handle_sharp_generation,
        inputs=[
            input_image, image_scale, gr.State("RunPod Serverless"),
            sharp_endpoint, sharp_api_key, gr.State(False),
            sharp_output_name, sharp_output_dir,
        ],
        outputs=[output_display, sharp_logs, sharp_progress],
    )
    
    sharp_check_btn.click(
        fn=check_serverless_status,
        inputs=[sharp_endpoint, sharp_api_key],
        outputs=[sharp_status],
    )
    
    sharp_save_btn.click(
        fn=lambda e, k: save_serverless_credentials(e, k, "sharp")[0],
        inputs=[sharp_endpoint, sharp_api_key],
        outputs=[sharp_status],
    )
    
    # =========================================================================
    # GEN3C EVENT HANDLERS
    # =========================================================================
    
    gen3c_generate_btn.click(
        fn=handle_gen3c_generation,
        inputs=[
            input_image, image_scale, gr.State("RunPod Serverless"),
            gen3c_endpoint, gen3c_api_key,
            gen3c_guidance, gen3c_frames, gen3c_trajectory, gen3c_foreground,
            gen3c_video_name, gen3c_seed, gen3c_output_dir,
        ],
        outputs=[output_video, gen3c_logs, gen3c_progress],
    )
    
    gen3c_check_btn.click(
        fn=check_serverless_status,
        inputs=[gen3c_endpoint, gen3c_api_key],
        outputs=[gen3c_status],
    )
    
    gen3c_save_btn.click(
        fn=lambda e, k: save_serverless_credentials(e, k, "gen3c")[0],
        inputs=[gen3c_endpoint, gen3c_api_key],
        outputs=[gen3c_status],
    )
    
    # =========================================================================
    # LYRA EVENT HANDLERS
    # =========================================================================
    
    lyra_generate_btn.click(
        fn=handle_lyra_generation,
        inputs=[
            input_image, input_video, image_scale,
            lyra_endpoint, lyra_api_key,
            lyra_mode, lyra_views, lyra_motion,
            lyra_multi_traj, lyra_fg_mask,
            lyra_gaussians, lyra_seed,
            lyra_output_name, lyra_output_dir,
            lyra_out_ply, lyra_out_video,
        ],
        outputs=[output_display, lyra_logs, lyra_progress, lyra_ply_path],
    )
    
    lyra_check_btn.click(
        fn=check_lyra_status,
        inputs=[lyra_endpoint, lyra_api_key],
        outputs=[lyra_status],
    )
    
    lyra_save_btn.click(
        fn=lambda e, k: save_serverless_credentials(e, k, "lyra")[0],
        inputs=[lyra_endpoint, lyra_api_key],
        outputs=[lyra_status],
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
            trellis_endpoint, trellis_api_key,
            trellis_resolution, trellis_guidance, trellis_seed,
            trellis_output_name, trellis_output_dir, trellis_format,
        ],
        outputs=[output_display, trellis_logs, trellis_progress],
    )
    
    trellis_check_btn.click(
        fn=check_trellis_status,
        inputs=[trellis_endpoint, trellis_api_key],
        outputs=[trellis_status],
    )
    
    trellis_save_btn.click(
        fn=lambda e, k: save_serverless_credentials(e, k, "trellis")[0],
        inputs=[trellis_endpoint, trellis_api_key],
        outputs=[trellis_status],
    )
    
    # =========================================================================
    # HUNYUAN EVENT HANDLERS
    # =========================================================================
    
    hunyuan_generate_btn.click(
        fn=handle_hunyuan_generation,
        inputs=[
            input_image, image_scale, hunyuan_exec_mode,
            hunyuan_endpoint, hunyuan_api_key,
            hunyuan_guidance, hunyuan_steps, hunyuan_seed, hunyuan_octree,
            hunyuan_model,
            hunyuan_fp16, hunyuan_attn_slice, hunyuan_cpu_offload, hunyuan_remove_bg,
            hunyuan_output_name, hunyuan_output_dir,
        ],
        outputs=[output_display, hunyuan_logs, hunyuan_progress],
    )
    
    hunyuan_check_btn.click(
        fn=check_hunyuan_runpod_status,
        inputs=[hunyuan_endpoint, hunyuan_api_key],
        outputs=[hunyuan_status],
    )
    
    hunyuan_save_btn.click(
        fn=lambda e, k: save_serverless_credentials(e, k, "hunyuan")[0],
        inputs=[hunyuan_endpoint, hunyuan_api_key],
        outputs=[hunyuan_status],
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
            mesh_endpoint, mesh_api_key,
        ],
        outputs=[output_display, mesh_logs, mesh_progress],
    )
    
    mesh_check_btn.click(
        fn=check_sugar_status,
        inputs=[mesh_endpoint, mesh_api_key],
        outputs=[mesh_status],
    )
    
    mesh_save_btn.click(
        fn=lambda e, k: save_serverless_credentials(e, k, "mesh_extraction")[0],
        inputs=[mesh_endpoint, mesh_api_key],
        outputs=[mesh_status],
    )
    
    # Mesh PLY dropdown
    def refresh_mesh_ply():
        return gr.update(choices=scan_for_ply_files())
    
    mesh_refresh_btn.click(fn=refresh_mesh_ply, outputs=[mesh_ply_dropdown])
    
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

