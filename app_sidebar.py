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

from ui.styles import CUSTOM_CSS, VIDEO_PREVIEW_JS
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
    check_hunyuan_runpod_status,
    check_runpod_status,
    check_serverless_status,
    cancel_serverless_job,
    check_sharp_installation,
    check_lyra_status,
    check_trellis_status,
    check_sugar_status,
    detect_ply_format,
    SHARP_DEFAULT_OUTPUT_DIR,
    LYRA_DEFAULT_OUTPUT_DIR,
    TRELLIS_DEFAULT_OUTPUT_DIR,
    MESH_DEFAULT_OUTPUT_DIR,
    GEN3C_DEFAULT_OUTPUT_DIR,
)

# Generation handlers - extracted to handlers/ module
from handlers import (
    handle_sharp_generation,
    handle_gen3c_generation,
    handle_lyra_generation,
    handle_trellis_generation,
    handle_hunyuan_generation,
    handle_mesh_extraction,
    handle_mesh_analyze,
    handle_mesh_cleanup,
)

# Help documentation - extracted to help/ module
from help import (
    SHARP_HELP,
    GEN3C_HELP,
    LTX2_HELP,
    LYRA_HELP,
    TRELLIS_HELP,
    HUNYUAN_HELP,
    MESH_HELP,
    MESH_CLEANUP_HELP,
    GENERAL_TIPS_HELP,
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
# TRELLIS.2 now uses unified endpoint (gen3c) - no separate settings needed
DEFAULT_HUNYUAN_ENDPOINT = _runpod_config.get("hunyuan_endpoint_id", "")
DEFAULT_HUNYUAN_API_KEY = _runpod_config.get("hunyuan_api_key", DEFAULT_GEN3C_API_KEY)
DEFAULT_MESH_ENDPOINT = _runpod_config.get("mesh_extraction_endpoint_id", DEFAULT_GEN3C_ENDPOINT)
DEFAULT_MESH_API_KEY = _runpod_config.get("mesh_extraction_api_key", DEFAULT_GEN3C_API_KEY)
DEFAULT_2DGS_ENDPOINT = _runpod_config.get("2dgs_endpoint_id", "s9txp6edtf2vg4")
DEFAULT_2DGS_API_KEY = _runpod_config.get("2dgs_api_key", DEFAULT_GEN3C_API_KEY)
# LTX API (Official Lightricks API - not RunPod)
DEFAULT_LTX_API_KEY = _runpod_config.get("ltx_api_key", "")


# =============================================================================
# SIDEBAR NAVIGATION STRUCTURE
# =============================================================================

NAV_ITEMS = [
    {"id": "sharp", "icon": "", "label": "SHARP", "category": "create", "description": "Fast 3DGS (~60s)"},
    {"id": "gen3c", "icon": "", "label": "GEN3C", "category": "create", "description": "Video generation (~10min)"},
    {"id": "ltx2", "icon": "", "label": "LTX-2", "category": "create", "description": "Video with camera control"},
    {"id": "lyra", "icon": "", "label": "Lyra", "category": "create", "description": "3DGS from video (~15min)"},
    {"id": "trellis", "icon": "", "label": "TRELLIS.2", "category": "create", "description": "High-quality 3D"},
    {"id": "hunyuan", "icon": "", "label": "Hunyuan3D", "category": "create", "description": "Image to GLB mesh"},
    {"id": "mesh", "icon": "", "label": "Mesh Extract", "category": "refine", "description": "3DGS to GLB mesh"},
    {"id": "settings", "icon": "", "label": "Settings", "category": "monitor", "description": "Credentials & config"},
]


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

with gr.Blocks(title="3D Generation Studio") as demo:

    # Inject JavaScript for video preview interactions
    gr.HTML(VIDEO_PREVIEW_JS)

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
            with gr.Accordion("Experiment Logging", open=False, elem_classes=["logging-accordion"]):
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
            
            # Button grid - 2 columns with section indicators
            # VIDEO: Generate video from image
            gr.Markdown("<small style='color:#666;margin:4px 0 2px 4px;'>VIDEO</small>", elem_classes=["section-label"])
            with gr.Row(elem_classes=["button-grid"]):
                nav_gen3c = gr.Button("GEN3C", elem_classes=["sidebar-nav"], elem_id="nav-gen3c", scale=1)
                nav_ltx2 = gr.Button("LTX-2", elem_classes=["sidebar-nav"], elem_id="nav-ltx2", scale=1)
            
            # SPLAT: Generate 3D Gaussian Splat (PLY)
            gr.Markdown("<small style='color:#666;margin:4px 0 2px 4px;'>SPLAT</small>", elem_classes=["section-label"])
            with gr.Row(elem_classes=["button-grid"]):
                nav_sharp = gr.Button("SHARP", elem_classes=["sidebar-nav"], elem_id="nav-sharp", scale=1)
                nav_lyra = gr.Button("Lyra", elem_classes=["sidebar-nav"], elem_id="nav-lyra", scale=1)
            
            # MESH: Generate mesh from image
            gr.Markdown("<small style='color:#666;margin:4px 0 2px 4px;'>MESH</small>", elem_classes=["section-label"])
            with gr.Row(elem_classes=["button-grid"]):
                nav_hunyuan = gr.Button("Hunyuan", elem_classes=["sidebar-nav"], elem_id="nav-hunyuan", scale=1)
                nav_trellis = gr.Button("TRELLIS", elem_classes=["sidebar-nav"], elem_id="nav-trellis", scale=1)
            
            # CONVERT: Convert/refine to mesh
            gr.Markdown("<small style='color:#666;margin:4px 0 2px 4px;'>CONVERT</small>", elem_classes=["section-label"])
            with gr.Row(elem_classes=["button-grid"]):
                nav_create = gr.Button("2DGS", elem_classes=["sidebar-nav"], elem_id="nav-create", scale=1)
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
                                    
                                    with gr.Accordion("Logs", open=False):
                                        sharp_logs = gr.Textbox(label="Generation Logs", lines=8, interactive=False)
                                
                                with gr.Column(scale=1):
                                    sharp_3d_viewer = gr.Model3D(
                                        label="3D Preview",
                                        height=400,
                                        clear_color=[0.1, 0.1, 0.1, 1.0],
                                    )
                        
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
                                    
                                    with gr.Accordion("Trajectory Types Explained", open=False):
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
                                    
                                    with gr.Accordion("Logs", open=False):
                                        sharp_video_logs = gr.Textbox(label="Render Logs", lines=8, interactive=False)
                                
                                with gr.Column(scale=1):
                                    sharp_video_preview = gr.Video(
                                        label="Video Preview",
                                        height=400,
                                    )
                
                # PAGE: GEN3C
                with gr.TabItem("GEN3C", id="gen3c"):
                    gr.HTML("""
                        <div class="page-header">
                            <h1>GEN3C</h1>
                            <p>NVIDIA's 3D-consistent video generation from a single image. Creates camera-controlled videos with 3D consistency.</p>
                        </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Group():
                                gr.Markdown("### Video Settings")
                                gr.Markdown("**Camera Trajectories** (select one or more)")
                                gen3c_trajectories = gr.CheckboxGroup(
                                    choices=["left", "right", "up", "down", "zoom_in", "zoom_out", "clockwise", "counterclockwise"],
                                    value=["left"],
                                    label="Camera Trajectories",
                                    info="Select multiple for multi-view generation"
                                )
                                gen3c_camera_rotation = gr.Dropdown(
                                    choices=["center_facing", "no_rotation", "trajectory_aligned"],
                                    value="center_facing",
                                    label="Camera Rotation",
                                    elem_id="gen3c-camera-rotation",
                                )
                                with gr.Row():
                                    gen3c_movement_distance = gr.Slider(
                                        minimum=0.1, maximum=1.0, value=0.3, step=0.05,
                                        label="Movement Distance",
                                        elem_id="gen3c-movement-distance",
                                    )
                                    gen3c_frames = gr.Dropdown(
                                        choices=["121", "241", "361"],
                                        value="121",
                                        label="Frames (121*N - 1)",
                                    )
                                with gr.Row():
                                    gen3c_guidance = gr.Slider(minimum=0.5, maximum=5.0, value=1.0, label="Guidance")
                                    gen3c_seed = gr.Number(value=None, label="Seed", precision=0)
                                gen3c_foreground = gr.Checkbox(value=True, label="Foreground Masking")
                            
                            with gr.Group():
                                gr.Markdown("### Output")
                                gen3c_video_name = gr.Textbox(value="gen3c_video", label="Video Name (base)")
                                gen3c_output_dir = gr.Textbox(value=GEN3C_DEFAULT_OUTPUT_DIR, label="Output Directory")
                            
                            gen3c_generate_btn = gr.Button("Generate Videos", variant="primary", size="lg")
                            gen3c_progress = gr.Textbox(value="Ready", label="Status", interactive=False)
                            gen3c_generated_videos = gr.State(value=[])  # Track generated video paths
                            
                            with gr.Accordion("Logs", open=False):
                                gen3c_logs = gr.Textbox(label="Generation Logs", lines=8, interactive=False)
                        
                        with gr.Column(scale=1):
                            gen3c_video_viewer = gr.Video(
                                label="Video Preview",
                                height=400,
                                autoplay=True,
                                loop=True,
                            )
                
                # PAGE: LTX-2
                with gr.TabItem("LTX-2", id="ltx2"):
                    gr.HTML("""
                        <div class="page-header">
                            <h1>LTX-2</h1>
                            <p>Lightricks' official API for high-quality video generation. Up to 4K @ 50fps with AI audio.</p>
                        </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Group():
                                gr.Markdown("### Model & Quality")
                                ltx2_model = gr.Radio(
                                    choices=["ltx-2-pro", "ltx-2-fast"],
                                    value="ltx-2-pro",
                                    label="Model",
                                    info="Pro: Best quality | Fast: Quicker generation"
                                )
                                with gr.Row():
                                    ltx2_resolution = gr.Dropdown(
                                        choices=["1080p", "1440p", "4K"],
                                        value="1080p",
                                        label="Resolution"
                                    )
                                    ltx2_duration = gr.Dropdown(
                                        choices=["6", "8", "10"],
                                        value="6",
                                        label="Duration (sec)"
                                    )
                                    ltx2_fps = gr.Dropdown(
                                        choices=["25", "50"],
                                        value="25",
                                        label="FPS"
                                    )
                            
                            with gr.Group():
                                gr.Markdown("### Camera Motions")
                                gr.Markdown("*Select multiple for multi-view 3D reconstruction*", elem_classes=["model-note"])
                                ltx2_camera_motions = gr.CheckboxGroup(
                                    choices=[
                                        "dolly_out",
                                        "dolly_in",
                                        "dolly_left",
                                        "dolly_right",
                                        "orbit",
                                        "jib_up",
                                        "static"
                                    ],
                                    value=["dolly_out"],
                                    label="Camera Motions",
                                    info="Each generates a separate video"
                                )
                                ltx2_custom_prompt = gr.Textbox(
                                    value="",
                                    label="Custom Prompt Addition (optional)",
                                    lines=2,
                                    info="Added to camera motion prompt, e.g. 'detailed textures, cinematic lighting'"
                                )
                            
                            with gr.Accordion("Options", open=False):
                                ltx2_generate_audio = gr.Checkbox(
                                    value=False,
                                    label="Generate AI Audio",
                                    info="Add AI-generated audio matching the scene"
                                )
                                ltx2_use_s3 = gr.Checkbox(
                                    value=True,
                                    label="Use S3 for image upload",
                                    info="Faster for larger images; disable to use base64"
                                )
                            
                            with gr.Group():
                                gr.Markdown("### Output")
                                ltx2_output_name = gr.Textbox(value="ltx2_video", label="Video Name")
                                ltx2_output_dir = gr.Textbox(value="/srv/searidge_share/outputs/ltx2", label="Output Directory")
                            
                            ltx2_generate_btn = gr.Button("Generate Videos", variant="primary", size="lg")
                            ltx2_progress = gr.Textbox(value="Ready", label="Status", interactive=False)
                            ltx2_generated_videos = gr.State(value=[])  # Track generated video paths
                            
                            with gr.Accordion("Logs", open=False):
                                ltx2_logs = gr.Textbox(label="Generation Logs", lines=8, interactive=False)
                        
                        with gr.Column(scale=1):
                            # Video Previews header with Play All button
                            with gr.Row():
                                gr.Markdown("### Video Previews")
                                ltx2_play_all_btn = gr.Button("▶ Play All", size="sm", scale=0, min_width=100, elem_id="ltx2-play-all-btn")
                            ltx2_videos_status = gr.Markdown("*No videos generated*")
                            
                            # Video preview grid (2x2) with labels
                            with gr.Row():
                                with gr.Column(scale=1, min_width=150):
                                    ltx2_video_label_1 = gr.Markdown("**1.** —", elem_classes=["video-slot-label"])
                                    ltx2_video_preview_1 = gr.Video(
                                        label="",
                                        height=170,
                                        visible=True,
                                        interactive=False,
                                        elem_id="ltx2-video-1",
                                    )
                                with gr.Column(scale=1, min_width=150):
                                    ltx2_video_label_2 = gr.Markdown("**2.** —", elem_classes=["video-slot-label"])
                                    ltx2_video_preview_2 = gr.Video(
                                        label="",
                                        height=170,
                                        visible=True,
                                        interactive=False,
                                        elem_id="ltx2-video-2",
                                    )
                            with gr.Row():
                                with gr.Column(scale=1, min_width=150):
                                    ltx2_video_label_3 = gr.Markdown("**3.** —", elem_classes=["video-slot-label"])
                                    ltx2_video_preview_3 = gr.Video(
                                        label="",
                                        height=170,
                                        visible=True,
                                        interactive=False,
                                        elem_id="ltx2-video-3",
                                    )
                                with gr.Column(scale=1, min_width=150):
                                    ltx2_video_label_4 = gr.Markdown("**4.** —", elem_classes=["video-slot-label"])
                                    ltx2_video_preview_4 = gr.Video(
                                        label="",
                                        height=170,
                                        visible=True,
                                        interactive=False,
                                        elem_id="ltx2-video-4",
                                    )

                            # State for tracking generated videos
                            ltx2_last_video_path = gr.State(value=None)

                            gr.Markdown("""
                            **Camera Motions for 3D**:
                            - `dolly_out` - Best for 3D (reveals full object)
                            - `dolly_left/right` - Side views for multi-angle
                            - `orbit` - 360° rotation (excellent for 3D)
                            - `jib_up` - Top-down perspective
                            """)
                
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
                                with gr.Column(scale=1):
                                    lyra_mode = gr.Radio(
                                        choices=["Static (Image → 3DGS)", "Dynamic (Video → 4DGS)"],
                                        value="Static (Image → 3DGS)",
                                        label="Mode",
                                    )
                                    
                                    with gr.Accordion("Advanced", open=False):
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
                                    
                                    with gr.Accordion("Logs", open=False):
                                        lyra_logs = gr.Textbox(lines=8, interactive=False)
                                
                                with gr.Column(scale=1):
                                    lyra_3d_viewer = gr.Model3D(
                                        label="3D Preview",
                                        height=400,
                                        clear_color=[0.1, 0.1, 0.1, 1.0],
                                    )
                        
                        with gr.Tab("Post-Process"):
                            gr.Markdown("### Convert Lyra PLY Output")
                            
                            initial_ply = scan_for_lyra_ply_files()
                            with gr.Row():
                                lyra_ply_dropdown = gr.Dropdown(choices=initial_ply, label="Select PLY", scale=3, allow_custom_value=True)
                                lyra_refresh_btn = gr.Button("Refresh", scale=0)
                            lyra_ply_path = gr.Textbox(label="Or enter path", placeholder="/path/to/file.ply")
                            
                            with gr.Row():
                                lyra_conv_3dgs = gr.Checkbox(True, label="3DGS Format")
                                lyra_conv_pc = gr.Checkbox(False, label="Point Cloud")
                            
                            lyra_convert_btn = gr.Button("Convert", variant="secondary")
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
                        with gr.Column(scale=1):
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
                            
                            with gr.Accordion("Logs", open=False):
                                trellis_logs = gr.Textbox(lines=8, interactive=False)
                        
                        with gr.Column(scale=1):
                            trellis_3d_viewer = gr.Model3D(
                                label="3D Preview",
                                height=400,
                                clear_color=[0.1, 0.1, 0.1, 1.0],
                            )
                
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
                                value="full", 
                                label="Model",
                                info="mini: faster (2-5 min), full: higher quality (5-15 min)"
                            )
                            
                            with gr.Accordion("Advanced", open=False):
                                with gr.Row():
                                    hunyuan_guidance = gr.Slider(1.0, 10.0, 5.0, label="Guidance")
                                    hunyuan_steps = gr.Slider(10, 100, 40, step=5, label="Steps")
                                with gr.Row():
                                    hunyuan_octree = gr.Slider(256, 512, 380, step=1, label="Octree Resolution")
                                    hunyuan_seed = gr.Number(None, label="Seed", precision=0)
                                with gr.Row():
                                    hunyuan_fp16 = gr.Checkbox(False, label="FP16", visible=False)
                                    hunyuan_attn_slice = gr.Checkbox(False, label="Attention Slicing")
                                    hunyuan_cpu_offload = gr.Checkbox(False, label="CPU Offload")
                                hunyuan_remove_bg = gr.Checkbox(True, label="Remove Background")
                            
                            with gr.Group():
                                hunyuan_output_name = gr.Textbox(value="hunyuan_output", label="Output Name")
                                hunyuan_output_dir = gr.Textbox(value="/srv/searidge_share/outputs/hunyuan", label="Output Dir")
                            
                            hunyuan_exec_mode = gr.Radio(["RunPod Serverless", "Local"], value="RunPod Serverless", label="Mode", visible=False)
                            hunyuan_generate_btn = gr.Button("Generate Mesh", variant="primary", size="lg")
                            hunyuan_progress = gr.Textbox(value="Ready", label="Status", interactive=False)
                            
                            with gr.Accordion("Logs", open=False):
                                hunyuan_logs = gr.Textbox(lines=8, interactive=False)
                        
                        with gr.Column(scale=1):
                            hunyuan_3d_viewer = gr.Model3D(
                                label="3D Preview",
                                height=400,
                                clear_color=[0.1, 0.1, 0.1, 1.0],
                            )
                
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
                                with gr.Column(scale=1):
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
                                            mesh_refresh_btn = gr.Button("Refresh", scale=0, elem_classes=["refresh-btn-inline"], min_width=40)
                                        mesh_format = gr.Radio(["Auto-detect", "Lyra", "SHARP", "Standard 3DGS"], value="Auto-detect", label="Format")
                                    
                                    with gr.Group():
                                        gr.Markdown("### Reconstruction Settings")
                                        mesh_method = gr.Radio(
                                            choices=["Poisson (High Quality)", "TSDF (Fast)"],
                                            value="Poisson (High Quality)",
                                            label="Method",
                                            info="Poisson: watertight mesh, better quality. TSDF: faster, good for previews.",
                                        )
                                        with gr.Row():
                                            mesh_quality = gr.Dropdown(
                                                ["High Poly (1M)", "Low Poly (200k)", "Custom"],
                                                value="High Poly (1M)",
                                                label="Quality",
                                            )
                                            mesh_poisson_depth = gr.Slider(6, 12, 10, step=1, label="Poisson Depth")
                                        with gr.Row(visible=False) as mesh_tsdf_row:
                                            mesh_tsdf_voxel_size = gr.Slider(0.002, 0.02, 0.008, step=0.002, label="TSDF Voxel Size")
                                            mesh_tsdf_num_views = gr.Slider(8, 64, 32, step=8, label="TSDF Views")
                                        mesh_decimate = gr.Number(0, label="Decimate to (faces, 0=none)")
                                    
                                    with gr.Group():
                                        gr.Markdown("### Output Settings")
                                        mesh_output_name = gr.Textbox(value="mesh_output", label="Output Name")
                                        mesh_output_dir = gr.Textbox(value=MESH_DEFAULT_OUTPUT_DIR, label="Output Directory")
                                        mesh_output_format = gr.Dropdown(["GLB", "OBJ", "PLY"], value="GLB", label="Output Format")
                                    
                                    mesh_extract_btn = gr.Button("Extract Mesh", variant="primary", size="lg")
                                    mesh_progress = gr.Textbox(value="Ready", label="Status", interactive=False)
                                    
                                    with gr.Accordion("Logs", open=False):
                                        mesh_logs = gr.Textbox(lines=8, interactive=False)
                                
                                with gr.Column(scale=1):
                                    mesh_3d_viewer = gr.Model3D(
                                        label="3D Preview",
                                        height=400,
                                        clear_color=[0.1, 0.1, 0.1, 1.0],
                                    )
                        
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
                                    
                                    with gr.Accordion("Cleanup Log", open=False):
                                        cleanup_logs = gr.Textbox(lines=8, interactive=False)
                                
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
                
                # PAGE: 2DGS (Video → Mesh Pipeline)
                with gr.TabItem("2DGS", id="create"):
                    gr.HTML("""
                        <div class="page-header">
                            <h1>2DGS - Video to Mesh</h1>
                            <p>Convert Gen3C videos to 3D meshes using ViPE pose extraction and 2D Gaussian Splatting.</p>
                        </div>
                    """)
                    
                    from ui.tabs.create_tab import create_2dgs_tab
                    twodgs_components = create_2dgs_tab(
                        default_endpoint_id="s9txp6edtf2vg4",
                        default_api_key=_runpod_config.get("gen3c_api_key", ""),
                    )
                
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
                    
                    # All endpoints in one row with narrower columns
                    with gr.Row(elem_classes=["settings-row"]):
                        with gr.Column(scale=1, min_width=200):
                            with gr.Group(elem_classes=["settings-card"]):
                                gr.Markdown("**GEN3C / SHARP / Lyra / TRELLIS.2**")
                                settings_gen3c_endpoint = gr.Textbox(value=DEFAULT_GEN3C_ENDPOINT, label="Endpoint ID")
                                settings_gen3c_key = gr.Textbox(value=DEFAULT_GEN3C_API_KEY, label="API Key", type="password")
                                settings_gen3c_test = gr.Button("Test", size="sm")
                                settings_gen3c_save = gr.Button("Save", size="sm", variant="primary")
                                settings_gen3c_status = gr.Textbox(value="", interactive=False, max_lines=1, show_label=False)
                        
                        with gr.Column(scale=1, min_width=200):
                            with gr.Group(elem_classes=["settings-card"]):
                                gr.Markdown("**Hunyuan3D**")
                                settings_hunyuan_endpoint = gr.Textbox(value=DEFAULT_HUNYUAN_ENDPOINT, label="Endpoint ID")
                                settings_hunyuan_key = gr.Textbox(value=DEFAULT_HUNYUAN_API_KEY, label="API Key", type="password")
                                settings_hunyuan_test = gr.Button("Test", size="sm")
                                settings_hunyuan_save = gr.Button("Save", size="sm", variant="primary")
                                settings_hunyuan_status = gr.Textbox(value="", interactive=False, max_lines=1, show_label=False)
                        
                        with gr.Column(scale=1, min_width=200):
                            with gr.Group(elem_classes=["settings-card"]):
                                gr.Markdown("**2DGS Pipeline**")
                                settings_2dgs_endpoint = gr.Textbox(value=DEFAULT_2DGS_ENDPOINT, label="Endpoint ID")
                                settings_2dgs_key = gr.Textbox(value=DEFAULT_2DGS_API_KEY, label="API Key", type="password")
                                settings_2dgs_test = gr.Button("Test", size="sm")
                                settings_2dgs_save = gr.Button("Save", size="sm", variant="primary")
                                settings_2dgs_status = gr.Textbox(value="", interactive=False, max_lines=1, show_label=False)
                    
                    gr.Markdown("### External APIs")
                    gr.Markdown("Third-party API credentials (not RunPod).")
                    
                    with gr.Row(elem_classes=["settings-row"]):
                        with gr.Column(scale=1, min_width=200):
                            with gr.Group(elem_classes=["settings-card"]):
                                gr.Markdown("**LTX-2 (Lightricks API)**")
                                gr.Markdown("*Get API key at [ltx.video](https://ltx.video)*", elem_classes=["model-note"])
                                settings_ltx_api_key = gr.Textbox(value=DEFAULT_LTX_API_KEY, label="API Key", type="password")
                                settings_ltx_save = gr.Button("Save", size="sm", variant="primary")
                                settings_ltx_status = gr.Textbox(value="", interactive=False, max_lines=1, show_label=False)
                        
                        # Empty columns to match 3-column layout of RunPod settings
                        with gr.Column(scale=1, min_width=200):
                            pass  # Placeholder for future external APIs
                        with gr.Column(scale=1, min_width=200):
                            pass  # Placeholder for future external APIs
                
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
                    
                    # Gen3C Documentation (VIDEO - 1)
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
- Output is VIDEO (mp4), not 3D model - use 2DGS or Lyra for 3D output
                        """)
                    
                    # LTX-2 Documentation (VIDEO - 2)
                    with gr.Accordion("LTX-2 - Image-to-Video with Camera Control", open=False):
                        gr.Markdown(LTX2_HELP)
                    
                    # SHARP Documentation (SPLAT - 3)
                    with gr.Accordion("SHARP - Single-Image 3D Gaussian Splatting", open=False):
                        gr.Markdown(SHARP_HELP)
                    
                    # Lyra Documentation (SPLAT - 4)
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
                    
                    # Hunyuan3D Documentation (MESH - 5)
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
                    
                    # TRELLIS Documentation (MESH - 6)
                    with gr.Accordion("TRELLIS.2 - Structured 3D Generation", open=False):
                        gr.Markdown(TRELLIS_HELP)
                    
                    # 2DGS Documentation (CONVERT - 7)
                    with gr.Accordion("2DGS - Video to Mesh Pipeline", open=False):
                        from help.documentation import TWODGS_HELP
                        gr.Markdown(TWODGS_HELP)
                    
                    # MESH Extraction Documentation (CONVERT - 8)
                    with gr.Accordion("MESH - 3DGS to Mesh Conversion", open=False):
                        gr.Markdown("""
## MESH Extraction

**What it does:** Converts 3D Gaussian Splat (3DGS) files to traditional mesh formats (GLB/OBJ). Works with PLY files from Lyra, SHARP, or other 3DGS sources.

### Extraction Methods

| Method | Quality | Speed | Best For |
|--------|---------|-------|----------|
| **Poisson** | ⭐⭐⭐⭐ | Slower | Final outputs, watertight meshes, editing |
| **TSDF** | ⭐⭐⭐ | Fast | Quick previews, iterating on models |

**Poisson (Recommended for quality):**
- Creates watertight meshes suitable for 3D printing/editing
- Better surface smoothness
- Preserves detail with adjustable depth parameter

**TSDF (Recommended for speed):**
- Fast depth-based volumetric fusion
- Good enough for previews and iteration
- Adjustable voxel size controls quality/speed tradeoff

### Key Settings

| Setting | Description | Range | Default |
|---------|-------------|-------|---------|
| **Method** | Extraction algorithm | Poisson/TSDF | Poisson |
| **Poisson Depth** | Detail level (Poisson only) | 6-12 | 10 |
| **TSDF Voxel Size** | Resolution (TSDF only) | 0.002-0.02 | 0.008 |
| **TSDF Views** | Viewpoints (TSDF only) | 8-64 | 32 |
| **Decimate** | Reduce faces (0=none) | 0-500k | 0 |

### Recommended Settings

**For final quality output (Poisson):**
```
Method: Poisson (High Quality)
Quality: High Poly (1M)
Poisson Depth: 10-11
Output Format: GLB
```

**For quick previews (TSDF):**
```
Method: TSDF (Fast)
TSDF Voxel Size: 0.01 (larger = faster)
TSDF Views: 16-32
Output Format: GLB
```
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
        const navButtons = ['nav-sharp', 'nav-gen3c', 'nav-ltx2', 'nav-lyra', 'nav-trellis', 'nav-hunyuan', 'nav-mesh', 'nav-create', 'nav-settings', 'nav-update', 'nav-help'];
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
    nav_ltx2.click(fn=lambda: gr.Tabs(selected="ltx2"), outputs=[page_tabs], js=highlight_js % 'nav-ltx2')
    nav_lyra.click(fn=lambda: gr.Tabs(selected="lyra"), outputs=[page_tabs], js=highlight_js % 'nav-lyra')
    nav_trellis.click(fn=lambda: gr.Tabs(selected="trellis"), outputs=[page_tabs], js=highlight_js % 'nav-trellis')
    nav_hunyuan.click(fn=lambda: gr.Tabs(selected="hunyuan"), outputs=[page_tabs], js=highlight_js % 'nav-hunyuan')
    nav_mesh.click(fn=lambda: gr.Tabs(selected="mesh"), outputs=[page_tabs], js=highlight_js % 'nav-mesh')
    nav_create.click(fn=lambda: gr.Tabs(selected="create"), outputs=[page_tabs], js=highlight_js % 'nav-create')
    nav_settings.click(fn=lambda: gr.Tabs(selected="settings"), outputs=[page_tabs], js=highlight_js % 'nav-settings')
    nav_update.click(fn=lambda: gr.Tabs(selected="update"), outputs=[page_tabs], js=highlight_js % 'nav-update')
    nav_help.click(fn=lambda: gr.Tabs(selected="help"), outputs=[page_tabs], js=highlight_js % 'nav-help')
    
    # =========================================================================
    # IMAGE INPUT HANDLERS
    # =========================================================================
    
    def get_output_name_from_input(image_path: str, model_suffix: str) -> str:
        """Generate default output name from input filename + model suffix.
        
        Example: 'CBGB1.jpg' + 'gen3c' -> 'CBGB1-gen3c'
        """
        if not image_path:
            return f"output-{model_suffix}"
        
        from pathlib import Path
        # Get filename without extension
        basename = Path(image_path).stem
        # Clean up any special characters that might cause issues
        basename = basename.replace(" ", "_")
        return f"{basename}-{model_suffix}"
    
    def update_all_output_names(image_path: str):
        """Update all model output names when input image changes."""
        return (
            update_image_info_display(image_path, 1.0),  # image_info (scale not available here)
            get_output_name_from_input(image_path, "gen3c"),      # gen3c_video_name
            get_output_name_from_input(image_path, "sharp"),      # sharp_output_name
            get_output_name_from_input(image_path, "sharp"),      # sharp_video_output_name
            get_output_name_from_input(image_path, "ltx2"),       # ltx2_output_name
            get_output_name_from_input(image_path, "lyra"),       # lyra_output_name
            get_output_name_from_input(image_path, "trellis"),    # trellis_output_name
            get_output_name_from_input(image_path, "hunyuan"),    # hunyuan_output_name
        )
    
    input_image.change(
        fn=update_all_output_names,
        inputs=[input_image],
        outputs=[
            image_info,
            gen3c_video_name,
            sharp_output_name,
            sharp_video_output_name,
            ltx2_output_name,
            lyra_output_name,
            trellis_output_name,
            hunyuan_output_name,
        ],
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

    def gen3c_multi_generate(
        input_img, img_scale, exec_mode,
        endpoint_id, api_key,
        guidance, frames, trajectories,
        movement_distance, camera_rotation,
        foreground_mask,
        video_name, seed, output_dir,
        log_params, encode_params
    ):
        """Handle GEN3C generation for multiple trajectories."""
        logs = []
        generated_videos = []
        last_video_path = None
        
        if not trajectories:
            return None, "Error: Select at least one trajectory", "❌ No trajectory selected", []
        
        total = len(trajectories)
        logs.append(f"[GEN3C] Generating {total} video(s)...")
        
        for idx, traj in enumerate(trajectories, 1):
            # Create unique name for each trajectory
            unique_name = f"{video_name}_{traj}"
            
            logs.append(f"\n[GEN3C] ({idx}/{total}) Generating {traj}...")
            
            result = handle_gen3c_generation(
                image_path=input_img,
                image_scale=img_scale,
                exec_mode=exec_mode,
                endpoint_id=endpoint_id,
                api_key=api_key,
                guidance=guidance,
                frames=frames,
                trajectory=traj,
                movement_distance=movement_distance,
                camera_rotation=camera_rotation,
                foreground_mask=foreground_mask,
                video_name=unique_name,
                seed=seed,
                output_dir=output_dir,
                log_params=log_params,
                encode_params=encode_params,
            )
            
            output_path, gen_logs, status = result
            logs.append(gen_logs)
            
            if output_path:
                logs.append(f"[GEN3C] ✅ {traj}: {output_path}")
                generated_videos.append(output_path)
                last_video_path = output_path
            else:
                logs.append(f"[GEN3C] ❌ {traj} failed")
        
        if generated_videos:
            status = f"✅ Generated {len(generated_videos)}/{total} videos"
        else:
            status = "❌ All generations failed"
        
        return last_video_path, "\n".join(logs), status, generated_videos

    def gen3c_generate_with_output(*args):
        """Wrapper that returns output paths for display."""
        result = gen3c_multi_generate(*args)
        last_video, logs, progress, video_list = result
        return last_video, last_video, logs, progress, video_list

    gen3c_generate_btn.click(
        fn=gen3c_generate_with_output,
        inputs=[
            input_image, image_scale, gr.State("RunPod Serverless"),
            settings_gen3c_endpoint, settings_gen3c_key,
            gen3c_guidance, gen3c_frames, gen3c_trajectories,
            gen3c_movement_distance, gen3c_camera_rotation,
            gen3c_foreground,
            gen3c_video_name, gen3c_seed, gen3c_output_dir,
            global_log_params, global_encode_params,
        ],
        outputs=[output_display, gen3c_video_viewer, gen3c_logs, gen3c_progress, gen3c_generated_videos],
    )

    # =========================================================================
    # LTX-2 EVENT HANDLERS (Official Lightricks API)
    # =========================================================================
    
    # Camera motion to prompt mapping
    LTX2_CAMERA_MOTION_PROMPTS = {
        "dolly_out": "Camera slowly pulls back from the subject, revealing the full scene.",
        "dolly_in": "Camera pushes forward toward the subject, focusing on intricate details.",
        "dolly_left": "Camera moves laterally to the left, revealing the side of the subject.",
        "dolly_right": "Camera moves laterally to the right, showing another angle.",
        "orbit": "Camera orbits around the subject in a smooth circular motion, revealing all sides.",
        "jib_up": "Camera rises vertically, showing the subject from above.",
        "jib_down": "Camera lowers, revealing the subject from a lower angle.",
        "static": "Camera remains stationary, subject may animate in place."
    }

    def handle_ltx2_multi_generation(
        input_img, img_scale,
        ltx_api_key,
        model, resolution, duration, fps,
        camera_motions, custom_prompt,
        generate_audio, use_s3,
        output_name, output_dir,
        log_params, encode_params
    ):
        """Handle LTX-2 video generation for multiple camera motions via official API."""
        import os
        from generators.ltx2 import run_ltx2_api

        logs = []
        generated_videos = []
        last_video_path = None

        # Validate inputs
        if input_img is None:
            return None, "Error: No input image", "❌ No input image", []

        if not ltx_api_key:
            return None, "Error: LTX API key required (configure in Settings → External APIs)", "❌ Missing LTX API key", []

        if not camera_motions:
            return None, "Error: Select at least one camera motion", "❌ No motion selected", []

        # Save input image temporarily
        import tempfile
        import shutil
        from PIL import Image

        temp_dir = tempfile.mkdtemp()
        try:
            # Handle Gradio image input
            if isinstance(input_img, str):
                input_path = input_img
            else:
                input_path = os.path.join(temp_dir, "input.png")
                if hasattr(input_img, 'save'):
                    input_img.save(input_path)
                else:
                    Image.fromarray(input_img).save(input_path)

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            total = len(camera_motions)
            logs.append(f"[LTX-2 API] Generating {total} video(s)...")
            logs.append(f"[LTX-2 API] Model: {model}, Resolution: {resolution}, Duration: {duration}s")

            for idx, motion in enumerate(camera_motions, 1):
                # Build prompt for this camera motion
                base_prompt = LTX2_CAMERA_MOTION_PROMPTS.get(motion, "Smooth camera movement.")
                full_prompt = f"{base_prompt} Smooth continuous motion, sharp focus, clear lighting, detailed textures."
                if custom_prompt:
                    full_prompt = f"{full_prompt} {custom_prompt}"
                
                # Create unique name for each motion
                if encode_params:
                    try:
                        from scripts.experiment_logger import ltx2_param_filename
                        video_name = ltx2_param_filename(
                            base_name=output_name,
                            model=model,
                            duration=float(duration),
                            resolution=resolution,
                            camera_motion=motion,
                            ext=""  # No extension - generator adds it
                        ).rstrip(".")
                    except Exception as e:
                        print(f"[LTX-2] Warning: Could not encode params in filename: {e}")
                        video_name = f"{output_name}_{motion}"
                else:
                    video_name = f"{output_name}_{motion}"

                logs.append(f"\n[LTX-2 API] ({idx}/{total}) Generating {motion}...")
                logs.append(f"[LTX-2 API] Prompt: {base_prompt[:60]}...")

                result = run_ltx2_api(
                    image_path=input_path,
                    output_dir=output_dir,
                    output_name=video_name,
                    prompt=full_prompt,
                    model=model,
                    resolution=resolution,
                    duration=int(duration),
                    fps=int(fps),
                    generate_audio=generate_audio,
                    api_key=ltx_api_key,
                    use_s3=use_s3,
                )

                if result.logs:
                    logs.append(result.logs)
                
                if result.success:
                    logs.append(f"[LTX-2 API] ✅ {motion}: {result.video_path}")
                    generated_videos.append(result.video_path)
                    last_video_path = result.video_path
                else:
                    logs.append(f"[LTX-2 API] ❌ {motion} failed: {result.error}")
            
            if generated_videos:
                status = f"✅ Generated {len(generated_videos)}/{total} videos"
            else:
                status = "❌ All generations failed"
            
            return last_video_path, "\n".join(logs), status, generated_videos
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            logs.append(f"[LTX-2 API] ❌ Exception: {str(e)}")
            return None, "\n".join(logs), f"❌ {str(e)}", []
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def ltx2_generate_with_output(*args):
        """Wrapper that returns output paths for 4 video previews."""
        from pathlib import Path
        
        result = handle_ltx2_multi_generation(*args)
        last_video, logs, progress, video_list = result
        
        # Prepare 4 video slots
        video_paths = list(video_list) if video_list else []
        video_labels = []
        
        for i in range(4):
            if i < len(video_paths) and video_paths[i]:
                # Get filename without extension
                filename = Path(video_paths[i]).stem
                video_labels.append(f"**{i+1}.** {filename}")
            else:
                video_labels.append(f"**{i+1}.** —")
        
        # Pad video paths to 4
        while len(video_paths) < 4:
            video_paths.append(None)
        
        # Status text
        if video_list:
            status = f"*{len(video_list)} video(s) generated*"
        else:
            status = "*No videos generated*"
        
        return (
            last_video,           # output_display
            logs,                 # ltx2_logs
            progress,             # ltx2_progress
            last_video,           # ltx2_last_video_path
            video_list,           # ltx2_generated_videos
            status,               # ltx2_videos_status
            video_labels[0],      # ltx2_video_label_1
            video_labels[1],      # ltx2_video_label_2
            video_labels[2],      # ltx2_video_label_3
            video_labels[3],      # ltx2_video_label_4
            video_paths[0],       # ltx2_video_preview_1
            video_paths[1],       # ltx2_video_preview_2
            video_paths[2],       # ltx2_video_preview_3
            video_paths[3],       # ltx2_video_preview_4
        )

    ltx2_generate_btn.click(
        fn=ltx2_generate_with_output,
        inputs=[
            input_image, image_scale,
            settings_ltx_api_key,  # LTX API key (not RunPod)
            ltx2_model, ltx2_resolution, ltx2_duration, ltx2_fps,
            ltx2_camera_motions, ltx2_custom_prompt,
            ltx2_generate_audio, ltx2_use_s3,
            ltx2_output_name, ltx2_output_dir,
            global_log_params, global_encode_params,
        ],
        outputs=[
            output_display, ltx2_logs, ltx2_progress, 
            ltx2_last_video_path, ltx2_generated_videos,
            ltx2_videos_status,
            ltx2_video_label_1, ltx2_video_label_2, ltx2_video_label_3, ltx2_video_label_4,
            ltx2_video_preview_1, ltx2_video_preview_2, ltx2_video_preview_3, ltx2_video_preview_4,
        ],
    )

    # LTX-2 Play All button - detect video state directly via JavaScript
    def ltx2_get_play_label():
        return gr.update()
    
    ltx2_play_all_btn.click(
        fn=ltx2_get_play_label,
        inputs=[],
        outputs=[ltx2_play_all_btn],
        js="""() => {
            console.log('LTX-2 Play All button clicked');
            const videos = [];
            for (let i = 1; i <= 4; i++) {
                const container = document.getElementById('ltx2-video-' + i);
                if (container) {
                    const video = container.querySelector('video');
                    if (video && video.src) {
                        videos.push(video);
                    }
                }
            }
            console.log('Found', videos.length, 'LTX-2 videos');
            
            if (videos.length === 0) {
                console.log('No videos found');
                return '▶ Play All';
            }
            
            const anyPlaying = videos.some(v => !v.paused && !v.ended);
            console.log('Any video playing?', anyPlaying);
            
            if (anyPlaying) {
                videos.forEach(v => { 
                    try { v.pause(); } catch(e) { console.log('Pause error:', e); } 
                });
                console.log('PAUSED all videos');
                return '▶ Play All';
            } else {
                videos.forEach(v => { 
                    try { 
                        v.play().catch(e => console.log('Play error:', e)); 
                    } catch(e) { console.log('Error:', e); } 
                });
                console.log('PLAYING all videos');
                return '⏸ Pause All';
            }
        }"""
    )

    # Note: "Send to 2DGS" button removed - users select videos directly from 2DGS tab

    # =========================================================================
    # LYRA EVENT HANDLERS
    # =========================================================================
    
    def lyra_generate_with_preview(*args):
        """Wrapper that converts PLY to GLB for 3D viewer."""
        result = handle_lyra_generation(*args)
        output_path, logs, progress, ply_path = result
        
        # Convert PLY to GLB for preview if PLY exists
        preview_path = None
        if ply_path and os.path.exists(ply_path):
            try:
                preview_path = convert_3dgs_to_preview_glb(ply_path)
            except Exception as e:
                print(f"[LYRA] Preview conversion failed: {e}")
                preview_path = None
        
        return output_path, logs, progress, ply_path, preview_path
    
    lyra_generate_btn.click(
        fn=lyra_generate_with_preview,
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
        outputs=[output_display, lyra_logs, lyra_progress, lyra_ply_path, lyra_3d_viewer],
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
    
    def trellis_generate_with_preview(*args):
        """Wrapper that returns GLB path for 3D viewer."""
        result = handle_trellis_generation(*args)
        output_path, logs, progress = result
        
        # Trellis outputs GLB directly, perfect for Model3D
        # If format is OBJ or PLY, try to find GLB version or return None
        preview_path = None
        if output_path:
            if output_path.endswith('.glb'):
                preview_path = output_path
            elif output_path.endswith(('.obj', '.ply')):
                # Try to find GLB version
                glb_path = output_path.rsplit('.', 1)[0] + '.glb'
                if os.path.exists(glb_path):
                    preview_path = glb_path
        
        return output_path, logs, progress, preview_path
    
    trellis_generate_btn.click(
        fn=trellis_generate_with_preview,
        inputs=[
            input_image, image_scale,
            settings_gen3c_endpoint, settings_gen3c_key,  # Uses unified endpoint
            trellis_resolution, trellis_guidance, trellis_seed,
            trellis_output_name, trellis_output_dir, trellis_format,
            global_log_params, global_encode_params,
        ],
        outputs=[output_display, trellis_logs, trellis_progress, trellis_3d_viewer],
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
    
    # Toggle visibility of method-specific settings
    def toggle_mesh_method_settings(method):
        """Show/hide settings based on extraction method."""
        is_poisson = "Poisson" in method
        return (
            gr.update(visible=is_poisson),   # poisson_depth row
            gr.update(visible=not is_poisson),  # tsdf row
        )
    
    mesh_method.change(
        fn=toggle_mesh_method_settings,
        inputs=[mesh_method],
        outputs=[mesh_poisson_depth, mesh_tsdf_row],
    )
    
    def mesh_extract_with_preview(
        ply_dropdown, input_format, method,
        quality, poisson_depth, decimate,
        tsdf_voxel_size, tsdf_num_views,
        output_name, output_format, output_dir,
        endpoint_id, api_key,
        encode_params
    ):
        """Wrapper that returns GLB path for 3D viewer."""
        # Map method selection to handler parameters
        if "Poisson" in method:
            handler_method = "SuGaR"
            voxel_size = 0.01  # not used for Poisson
            num_views = 32  # not used for Poisson
        else:
            handler_method = "TSDF"
            voxel_size = tsdf_voxel_size
            num_views = int(tsdf_num_views)
            poisson_depth = 10  # not used for TSDF
        
        # Generate encoded filename if requested
        effective_output_name = output_name
        if encode_params:
            try:
                from scripts.experiment_logger import mesh_extract_param_filename
                # Get the input filename from ply_dropdown (e.g., "lyra_bedroom_g75.ply")
                input_filename = ply_dropdown if ply_dropdown else "mesh"
                encoded_name = mesh_extract_param_filename(
                    input_filename=input_filename,
                    ext=""  # No extension - handler adds it
                ).rstrip(".")
                effective_output_name = encoded_name
            except Exception as e:
                print(f"[MESH] Warning: Could not encode params in filename: {e}")
        
        result = handle_mesh_extraction(
            ply_dropdown, input_format,
            handler_method, "dn_consistency",
            quality, poisson_depth, decimate,
            False, "2048", "short",
            voxel_size, num_views,
            effective_output_name, output_format, output_dir,
            endpoint_id, api_key,
        )
        output_path, logs, progress = result
        
        # Mesh extraction outputs GLB/OBJ/PLY
        # Model3D works best with GLB
        preview_path = None
        if output_path:
            if output_path.endswith('.glb'):
                preview_path = output_path
            elif output_path.endswith('.obj'):
                # Try to find GLB version
                glb_path = output_path.rsplit('.', 1)[0] + '.glb'
                if os.path.exists(glb_path):
                    preview_path = glb_path
            elif output_path.endswith('.ply'):
                # Convert PLY to GLB for preview
                try:
                    preview_path = convert_3dgs_to_preview_glb(output_path)
                except Exception as e:
                    print(f"[MESH] Preview conversion failed: {e}")
                    preview_path = None
        
        return output_path, logs, progress, preview_path
    
    mesh_extract_btn.click(
        fn=mesh_extract_with_preview,
        inputs=[
            mesh_ply_dropdown, mesh_format, mesh_method,
            mesh_quality, mesh_poisson_depth, mesh_decimate,
            mesh_tsdf_voxel_size, mesh_tsdf_num_views,
            mesh_output_name, mesh_output_format, mesh_output_dir,
            settings_gen3c_endpoint, settings_gen3c_key,
            global_encode_params,
        ],
        outputs=[output_display, mesh_logs, mesh_progress, mesh_3d_viewer],
    )
    
    # Mesh PLY dropdown
    def refresh_mesh_ply():
        return gr.update(choices=scan_for_ply_files())

    mesh_refresh_btn.click(fn=refresh_mesh_ply, outputs=[mesh_ply_dropdown])
    
    def update_mesh_output_name(ply_selection):
        """Update mesh output name based on selected PLY file."""
        from pathlib import Path
        if not ply_selection:
            return "mesh_output"
        
        # Handle display format: "filename.ply (Model)"
        if " (" in ply_selection:
            ply_path = ply_selection.rsplit(" (", 1)[0]
        else:
            ply_path = ply_selection
        
        basename = Path(ply_path).stem
        return f"{basename}-mesh"
    
    mesh_ply_dropdown.change(
        fn=update_mesh_output_name,
        inputs=[mesh_ply_dropdown],
        outputs=[mesh_output_name],
    )
    
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
    # 2DGS PIPELINE EVENT HANDLERS
    # =========================================================================

    from handlers.generation_handlers import handle_2dgs_pipeline, list_gen3c_videos, list_all_videos, get_video_info

    # Store video paths mapping (display name -> full path)
    _video_paths_cache = {}

    # Refresh video list (Gen3C + LTX-2) with sort and filter
    def refresh_video_list(sort_by="Date (newest)", limit="10"):
        """Refresh video checkboxes with videos from Gen3C and LTX-2 directories."""
        global _video_paths_cache
        gen3c_dir = "/srv/searidge_share/outputs/gen3c"
        ltx2_dir = "/srv/searidge_share/outputs/ltx2"
        videos = list_all_videos(gen3c_dir, ltx2_dir, sort_by=sort_by, limit=limit)

        # Update cache and create choices
        _video_paths_cache = {v[0]: v[1] for v in videos}
        choices = [v[0] for v in videos]

        return gr.update(choices=choices, value=[])

    # Wire refresh button
    twodgs_components["refresh_videos_btn"].click(
        fn=refresh_video_list,
        inputs=[twodgs_components["video_sort_by"], twodgs_components["video_limit"]],
        outputs=[twodgs_components["video_checkboxes"]],
    )
    
    # Auto-refresh when sort or limit changes
    twodgs_components["video_sort_by"].change(
        fn=refresh_video_list,
        inputs=[twodgs_components["video_sort_by"], twodgs_components["video_limit"]],
        outputs=[twodgs_components["video_checkboxes"]],
    )
    twodgs_components["video_limit"].change(
        fn=refresh_video_list,
        inputs=[twodgs_components["video_sort_by"], twodgs_components["video_limit"]],
        outputs=[twodgs_components["video_checkboxes"]],
    )

    # Helper to get full path from display name
    def get_video_full_path(display_name):
        """Get full path from display name."""
        global _video_paths_cache
        if display_name in _video_paths_cache:
            return _video_paths_cache[display_name]
        
        # Fallback: parse the display name
        from pathlib import Path
        if display_name.startswith("[Gen3C] "):
            filename = display_name.replace("[Gen3C] ", "")
            return str(Path("/srv/searidge_share/outputs/gen3c") / filename)
        elif display_name.startswith("[LTX-2] "):
            filename = display_name.replace("[LTX-2] ", "")
            return str(Path("/srv/searidge_share/outputs/ltx2") / filename)
        return display_name

    # Update video previews when selection changes
    def update_video_previews(selected_videos):
        """Update video preview panels and output name based on selection."""
        from pathlib import Path
        
        # Limit to 4 videos
        selected = selected_videos[:4] if selected_videos else []
        
        # Get full paths and display names
        video_paths = [get_video_full_path(v) for v in selected]
        
        # Generate output name from first video
        output_name = "mesh_output"
        if video_paths and video_paths[0]:
            video_basename = Path(video_paths[0]).stem
            for suffix in ["_dolly_out", "_dolly_in", "_dolly_left", "_dolly_right", "_orbit", "_jib_up", "_static"]:
                if video_basename.endswith(suffix):
                    video_basename = video_basename[:-len(suffix)]
                    break
            output_name = f"{video_basename}-2dgs"
        
        # Create individual labels for each slot (filename without extension)
        video_labels = []
        display_names = list(selected)
        for i, name in enumerate(display_names):
            if name:
                # Remove model prefix and extension
                short_name = name.replace("[Gen3C] ", "").replace("[LTX-2] ", "")
                # Remove extension
                short_name = Path(short_name).stem if short_name else ""
                video_labels.append(f"**{i+1}.** {short_name}")
            else:
                video_labels.append(f"**{i+1}.** —")
        
        # Pad to 4 elements
        while len(video_labels) < 4:
            video_labels.append(f"**{len(video_labels)+1}.** —")
        while len(display_names) < 4:
            display_names.append("")
        while len(video_paths) < 4:
            video_paths.append(None)
        
        count_text = f"**Selected: {len(selected)}/4**"
        
        # Simple status text
        if selected:
            label_text = f"*{len(selected)} video(s) loaded*"
        else:
            label_text = "*No videos selected*"
        
        # Hidden span for JavaScript slot mapping
        js_update = f'<span style="display:none" id="slot-map-data" data-slot1="{display_names[0]}" data-slot2="{display_names[1]}" data-slot3="{display_names[2]}" data-slot4="{display_names[3]}"></span>'
        
        return (
            count_text,
            label_text + js_update,
            video_labels[0],
            video_labels[1],
            video_labels[2],
            video_labels[3],
            video_paths[0],
            video_paths[1],
            video_paths[2],
            video_paths[3],
            output_name,
        )

    twodgs_components["video_checkboxes"].change(
        fn=update_video_previews,
        inputs=[twodgs_components["video_checkboxes"]],
        outputs=[
            twodgs_components["selected_count"],
            twodgs_components["selected_videos_label"],
            twodgs_components["video_label_1"],
            twodgs_components["video_label_2"],
            twodgs_components["video_label_3"],
            twodgs_components["video_label_4"],
            twodgs_components["video_preview_1"],
            twodgs_components["video_preview_2"],
            twodgs_components["video_preview_3"],
            twodgs_components["video_preview_4"],
            twodgs_components["output_name"],
        ],
    )
    
    # Play All button - detect video state directly via JavaScript
    def get_play_label():
        # This just returns a dummy - JS handles the actual logic
        return gr.update()
    
    twodgs_components["play_all_btn"].click(
        fn=get_play_label,
        inputs=[],
        outputs=[twodgs_components["play_all_btn"]],
        js="""() => {
            console.log('Play All button clicked');
            const videos = [];
            for (let i = 1; i <= 4; i++) {
                const container = document.getElementById('twodgs-video-' + i);
                if (container) {
                    const video = container.querySelector('video');
                    if (video && video.src) {
                        videos.push(video);
                    }
                }
            }
            console.log('Found', videos.length, 'videos');
            
            if (videos.length === 0) {
                console.log('No videos found');
                return '▶ Play All';
            }
            
            // Check if ANY video is currently playing
            const anyPlaying = videos.some(v => !v.paused && !v.ended);
            console.log('Any video playing?', anyPlaying);
            
            if (anyPlaying) {
                // Pause all
                videos.forEach(v => { 
                    try { v.pause(); } catch(e) { console.log('Pause error:', e); } 
                });
                console.log('PAUSED all videos');
                return '▶ Play All';
            } else {
                // Resume playing from current position
                videos.forEach(v => { 
                    try { 
                        v.play().catch(e => console.log('Play error:', e)); 
                    } catch(e) { console.log('Error:', e); } 
                });
                console.log('PLAYING all videos (resumed)');
                return '⏸ Pause All';
            }
        }"""
    )

    # Main generate button for multi-video
    def run_2dgs_multi_video(
        video_source, selected_videos, video_uploads,
        iterations, mesh_quality, output_format, output_dir, output_name,
        max_videos, depth_threshold,
        endpoint_id, api_key, s3_bucket, s3_region,
        encode_params
    ):
        """Run 2DGS pipeline with multiple videos."""
        from pathlib import Path
        from runpod.runpod_client import TwoDGSPipelineClient
        import time
        import re
        
        # Convert max_videos to int
        max_videos_limit = int(max_videos) if max_videos else 4
        
        # Determine video paths based on source
        video_paths = []
        
        if video_source == "Select Videos":
            if not selected_videos:
                return None, {}, "Error: No videos selected", None
            
            # Limit to configured max
            selected = selected_videos[:max_videos_limit]
            video_paths = [get_video_full_path(v) for v in selected]
            
        elif video_source == "Upload Video":
            if not video_uploads:
                return None, {}, "Error: No videos uploaded", None
            
            # Handle uploaded files
            uploads = video_uploads[:max_videos_limit] if isinstance(video_uploads, list) else [video_uploads]
            video_paths = [f.name if hasattr(f, 'name') else str(f) for f in uploads]
        
        # Validate paths exist
        valid_paths = []
        for p in video_paths:
            if p and Path(p).exists():
                valid_paths.append(p)
            else:
                print(f"[2DGS] Warning: Video not found: {p}")
        
        if not valid_paths:
            return None, {}, "Error: No valid video files found", None
        
        # Single video - use existing handler
        if len(valid_paths) == 1:
            output_path, stats, status = handle_2dgs_pipeline(
                video_source="Video Path",
                video_path=valid_paths[0],
                video_upload=None,
                video_dropdown="",
                gen3c_output_dir="/srv/searidge_share/outputs/gen3c",
                iterations=int(iterations),
                mesh_quality=mesh_quality,
                output_format=output_format,
                output_dir=output_dir,
                endpoint_id=endpoint_id,
                api_key=api_key,
                s3_bucket=s3_bucket,
                s3_region=s3_region,
            )
        else:
            # Multi-video mode - use new submit_multi_video_job
            print(f"[2DGS Multi] Processing {len(valid_paths)} videos with depth threshold {depth_threshold}")
            
            try:
                client = TwoDGSPipelineClient(
                    endpoint_id=endpoint_id or "s9txp6edtf2vg4",
                    api_key=api_key,
                )
                
                # Submit multi-video job
                job_result = client.submit_multi_video_job(
                    video_paths=valid_paths,
                    iterations=int(iterations),
                    mesh_quality=mesh_quality,
                    output_format=output_format,
                    depth_threshold=float(depth_threshold) if depth_threshold else 0.5,
                    s3_bucket=s3_bucket,
                    s3_region=s3_region,
                )
                
                if job_result.get("status") == "error":
                    return None, {}, f"Error: {job_result.get('error')}", None
                
                job_id = job_result.get("job_id")
                print(f"[2DGS Multi] Job submitted: {job_id}")
                
                # Wait for completion
                final_status = client.wait_for_completion(
                    job_id=job_id,
                    poll_interval=15,
                    max_wait=1800,  # 30 minutes for multi-video
                )
                
                if final_status.get("status") == "completed":
                    mesh_url = final_status.get("mesh_url")
                    quality_stats = final_status.get("quality_stats", {})
                    
                    # Download mesh
                    output_path = None
                    if mesh_url:
                        timestamp = int(time.time())
                        
                        # Generate encoded filename if requested
                        if encode_params:
                            try:
                                from scripts.experiment_logger import twodgs_param_filename
                                # Extract base names from video paths (strip motion suffixes)
                                motion_pattern = re.compile(r'_?(dolly_out|dolly_in|pan_left|pan_right|tilt_up|tilt_down|zoom_in|zoom_out|orbit_left|orbit_right|rotate_cw|rotate_ccw|static)$', re.IGNORECASE)
                                input_basenames = []
                                for vp in valid_paths:
                                    stem = Path(vp).stem
                                    # Remove motion suffix to get original image name
                                    clean_name = motion_pattern.sub('', stem)
                                    if clean_name and clean_name not in input_basenames:
                                        input_basenames.append(clean_name)
                                
                                local_filename = twodgs_param_filename(
                                    input_basenames=input_basenames,
                                    ext=f".{output_format}"
                                )
                            except Exception as e:
                                print(f"[2DGS] Warning: Could not encode params in filename: {e}")
                                local_filename = f"{output_name}_{timestamp}.{output_format}"
                        else:
                            local_filename = f"{output_name}_{timestamp}.{output_format}"
                        
                        local_path = os.path.join(output_dir, local_filename)
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Download from URL
                        import requests
                        response = requests.get(mesh_url, stream=True)
                        if response.status_code == 200:
                            with open(local_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            output_path = local_path
                            print(f"[2DGS Multi] Downloaded mesh to {output_path}")
                    
                    stats = {
                        "job_id": job_id,
                        "videos_processed": len(valid_paths),
                        "num_frames": final_status.get("num_frames"),
                        "iterations": final_status.get("iterations"),
                        "elapsed_seconds": final_status.get("elapsed_seconds"),
                        **quality_stats
                    }
                    
                    status = f"✅ Multi-video processing complete!\n"
                    status += f"Videos: {len(valid_paths)}, "
                    status += f"Frames: {stats.get('valid_frames', 'N/A')}/{stats.get('total_frames', 'N/A')}, "
                    status += f"Time: {stats.get('elapsed_seconds', 'N/A')}s"
                    
                else:
                    output_path = None
                    stats = {"job_id": job_id, "error": final_status.get("error")}
                    status = f"❌ Multi-video processing failed: {final_status.get('error', 'Unknown error')}"
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, {"error": str(e)}, f"Error: {str(e)}", None
        
        # Determine preview path
        preview_path = None
        if output_path:
            if output_path.endswith('.glb'):
                preview_path = output_path
            elif output_path.endswith('.obj'):
                glb_path = output_path.rsplit('.', 1)[0] + '.glb'
                if os.path.exists(glb_path):
                    preview_path = glb_path
        
        return output_path, stats, status, preview_path
    
    twodgs_components["generate_btn"].click(
        fn=run_2dgs_multi_video,
        inputs=[
            twodgs_components["video_source"],
            twodgs_components["video_checkboxes"],
            twodgs_components["video_uploads"],
            twodgs_components["iterations"],
            twodgs_components["mesh_quality"],
            twodgs_components["output_format"],
            twodgs_components["output_dir"],
            twodgs_components["output_name"],
            twodgs_components["max_videos"],
            twodgs_components["depth_threshold"],
            settings_2dgs_endpoint,
            settings_2dgs_key,
            gr.State("arkrunr"),  # S3 bucket
            gr.State("us-west-1"),  # S3 region
            global_encode_params,
        ],
        outputs=[
            twodgs_components["output_file"],
            twodgs_components["output_stats"],
            twodgs_components["status_text"],
            twodgs_components["twodgs_3d_viewer"],
        ],
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
    
    # TRELLIS.2 now uses unified endpoint (gen3c) - no separate settings needed

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
    
    settings_2dgs_test.click(
        fn=check_serverless_status,
        inputs=[settings_2dgs_endpoint, settings_2dgs_key],
        outputs=[settings_2dgs_status],
    )
    
    settings_2dgs_save.click(
        fn=lambda e, k: save_serverless_credentials(e, k, "2dgs")[0],
        inputs=[settings_2dgs_endpoint, settings_2dgs_key],
        outputs=[settings_2dgs_status],
    )
    
    # LTX API key save handler
    def save_ltx_api_key(api_key: str) -> str:
        """Save LTX API key to config file."""
        config = _load_runpod_config()
        config["ltx_api_key"] = api_key.strip() if api_key else ""
        _save_runpod_config(config)
        if api_key:
            return "✅ LTX API key saved"
        return "⚠️ No API key provided"
    
    settings_ltx_save.click(
        fn=save_ltx_api_key,
        inputs=[settings_ltx_api_key],
        outputs=[settings_ltx_status],
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
                // Highlight SHARP button on load
                const sharpBtn = document.getElementById('nav-sharp');
                if (sharpBtn) sharpBtn.classList.add('nav-active');
                
                // Add native tooltips to Gen3C controls
                const movementDist = document.getElementById('gen3c-movement-distance');
                if (movementDist) {
                    const label = movementDist.querySelector('label, span');
                    if (label) label.title = 'How far camera moves (0.1=subtle, 1.0=dramatic)';
                }
                const cameraRot = document.getElementById('gen3c-camera-rotation');
                if (cameraRot) {
                    const label = cameraRot.querySelector('label, span');
                    if (label) label.title = 'How camera rotates during movement';
                }
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

