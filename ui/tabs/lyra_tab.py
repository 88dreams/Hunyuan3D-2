"""
Lyra Tab for 3D Generation Studio

NVIDIA's Lyra: Image/Video → 3D/4D Gaussian Splatting
Built on GEN3C video diffusion + 3DGS reconstruction decoder.

Supports:
- Static 3DGS from single image (Image → multi-view video → 3DGS)
- Dynamic 4DGS from video input (Video → multi-view video → 4DGS)

UI Structure:
- Generate sub-tab: All generation settings + Generate button
- Post-Process sub-tab: PLY conversion, downsampling, Blender export
"""

import gradio as gr  # type: ignore
from pathlib import Path
from typing import List


# Default output directory (will be set from config)
LYRA_DEFAULT_OUTPUT_DIR = "/srv/searidge_share/outputs/lyra"


def scan_for_lyra_ply_files() -> List[str]:
    """
    Scan output directories for Lyra PLY files.
    Returns list of paths sorted by modification time (newest first).
    """
    ply_files = []
    
    # Directories to scan (Lyra-specific first, then general)
    search_dirs = [
        Path("/srv/searidge_share/outputs/lyra"),
        Path("/srv/searidge_share/outputs/sharp"),
        Path("/srv/searidge_share/outputs"),
    ]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            # Find all .ply files recursively
            for ply_file in search_dir.rglob("*.ply"):
                try:
                    # Get file info
                    stat = ply_file.stat()
                    size_mb = stat.st_size / (1024 * 1024)
                    
                    # Skip very small files (likely not valid 3DGS)
                    if size_mb > 1:
                        ply_files.append((str(ply_file), stat.st_mtime, size_mb))
                except Exception:
                    continue
    
    # Sort by modification time (newest first) and deduplicate
    seen = set()
    unique_files = []
    for path, mtime, size in sorted(ply_files, key=lambda x: -x[1]):
        if path not in seen:
            seen.add(path)
            unique_files.append((path, size))
    
    # Return just the paths, with size info in display
    return [f"{path} ({size:.1f} MB)" for path, size in unique_files[:20]]  # Limit to 20 most recent


def create_lyra_tab(
    default_endpoint_id: str = "",
    default_api_key: str = "",
):
    """
    Create the Lyra 3DGS/4DGS tab UI components.
    
    Args:
        default_endpoint_id: Default RunPod endpoint ID (from saved config)
        default_api_key: Default RunPod API key (from saved config)
    
    Returns:
        Dictionary of all UI components for event handler binding
    """
    
    # Use sub-tabs to separate Generation from Post-Processing
    with gr.Tabs():
        
        # =====================================================================
        # GENERATE TAB - All settings needed to create a 3DGS
        # =====================================================================
        with gr.Tab("🎬 Generate", id="lyra_generate"):
            
            gr.Markdown("### Image/Video → 3D Gaussian Splatting")
            
            # -----------------------------------------------------------------
            # ROW 1: Mode Selection + Generate Button (most important!)
            # -----------------------------------------------------------------
            with gr.Row():
                with gr.Column(scale=2):
                    generation_mode = gr.Radio(
                        choices=["Static (Image → 3DGS)", "Dynamic (Video → 4DGS)"],
                        value="Static (Image → 3DGS)",
                        label="Generation Mode",
                        info="Static: Single image to 3D. Dynamic: Video to animated 4D.",
                    )
                with gr.Column(scale=1):
                    generate_btn = gr.Button(
                        "🚀 Generate 3DGS",
                        variant="primary",
                        size="lg",
                    )
                    queue_btn = gr.Button(
                        "+ Add to Queue",
                        variant="secondary",
                        size="sm",
                    )
            
            # -----------------------------------------------------------------
            # ROW 2: Output Settings (name + directory)
            # -----------------------------------------------------------------
            with gr.Row():
                output_name = gr.Textbox(
                    value="lyra_output",
                    label="Output Name",
                    scale=1,
                )
                output_dir = gr.Textbox(
                    value=LYRA_DEFAULT_OUTPUT_DIR,
                    label="Output Directory",
                    scale=2,
                )
            
            # -----------------------------------------------------------------
            # Progress Display (always visible)
            # -----------------------------------------------------------------
            progress_display = gr.Textbox(
                value="Ready to generate...",
                label="Status",
                interactive=False,
                lines=2,
                max_lines=3,
            )
            
            # -----------------------------------------------------------------
            # Advanced Settings (collapsed by default)
            # -----------------------------------------------------------------
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                
                gr.Markdown("##### Diffusion (Video Generation Phase)")
                with gr.Row():
                    num_views = gr.Slider(
                        minimum=4, maximum=16, value=8, step=1,
                        label="Multi-view Count",
                        info="Number of camera views to generate",
                    )
                    camera_motion = gr.Slider(
                        minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                        label="Camera Motion Scale",
                        info="1.0=normal, 2.0=more motion",
                    )
                with gr.Row():
                    multi_trajectory = gr.Checkbox(
                        value=True,
                        label="Multi-Trajectory",
                        info="Multiple camera paths for better coverage",
                    )
                    foreground_masking = gr.Checkbox(
                        value=True,
                        label="Foreground Masking",
                        info="Mask background for object-centric scenes",
                    )
                
                gr.Markdown("##### Reconstruction (3DGS Phase)")
                with gr.Row():
                    num_gaussians = gr.Slider(
                        minimum=10000, maximum=500000, value=100000, step=10000,
                        label="Max Gaussians",
                        info="Maximum number of Gaussian splats",
                    )
                    seed = gr.Number(
                        value=None,
                        label="Seed",
                        precision=0,
                        info="Random seed (empty = random)",
                    )
                
                gr.Markdown("##### Output Options")
                with gr.Row():
                    output_ply = gr.Checkbox(
                        value=True,
                        label="Export PLY",
                        info="3D Gaussian Splat file",
                    )
                    output_video = gr.Checkbox(
                        value=True,
                        label="Export Render Video",
                        info="Rendered video of the result",
                    )
            
            # -----------------------------------------------------------------
            # RunPod Connection (collapsed)
            # -----------------------------------------------------------------
            with gr.Accordion("🔌 RunPod Connection", open=False):
                exec_mode = gr.Radio(
                    choices=["RunPod Serverless"],
                    value="RunPod Serverless",
                    label="Execution Mode",
                    info="Lyra requires H100/A100 GPUs",
                    visible=False,  # Only one option, hide it
                )
                status = gr.Textbox(
                    value="Click 'Check Status' to verify connection",
                    label="Connection Status",
                    interactive=False,
                )
                
                _has_creds = bool(default_endpoint_id and default_api_key)
                with gr.Row():
                    check_serverless_btn = gr.Button("Check Status", size="sm")
                    cancel_btn = gr.Button("Cancel Job", variant="stop", size="sm")
                    edit_creds_btn = gr.Button("Edit Credentials", size="sm")
                
                with gr.Group(visible=not _has_creds) as creds_group:
                    endpoint_id = gr.Textbox(
                        value=default_endpoint_id,
                        label="Endpoint ID",
                    )
                    api_key = gr.Textbox(
                        value=default_api_key,
                        label="API Key",
                        type="password",
                    )
                    save_creds_btn = gr.Button("Save Credentials", variant="primary")
                
                serverless_settings = gr.Group(visible=True)  # Placeholder for compatibility
                current_job_id = gr.State(value=None)
            
            # -----------------------------------------------------------------
            # Logs (collapsed)
            # -----------------------------------------------------------------
            with gr.Accordion("📋 Generation Logs", open=False):
                logs_box = gr.Textbox(
                    label="Logs",
                    lines=10,
                    max_lines=20,
                    interactive=False,
                    autoscroll=True,
                )
        
        # =====================================================================
        # POST-PROCESS TAB - Convert and export the generated PLY
        # =====================================================================
        with gr.Tab("🔧 Post-Process", id="lyra_postprocess"):
            
            gr.Markdown("""
            ### Convert & Export Lyra Output
            
            Lyra outputs PyTorch tensors in a `.ply` file. Use these tools to convert 
            to standard formats for viewing and editing.
            """)
            
            # -----------------------------------------------------------------
            # Input Selection with Dropdown + Refresh
            # -----------------------------------------------------------------
            with gr.Group():
                gr.Markdown("##### 1️⃣ Select Input")
                
                # Scan for available PLY files
                initial_ply_files = scan_for_lyra_ply_files()
                
                with gr.Row():
                    ply_dropdown = gr.Dropdown(
                        choices=initial_ply_files,
                        label="Select PLY File",
                        info="Recent PLY files from Lyra and SHARP outputs",
                        scale=3,
                        allow_custom_value=True,
                    )
                    refresh_ply_btn = gr.Button("🔄 Refresh", scale=1)
                
                ply_input_path = gr.Textbox(
                    value="",
                    label="Or Enter Path Manually",
                    placeholder="Auto-filled from dropdown, or enter path manually",
                    info="Path to Lyra's raw .ply output",
                )
            
            # -----------------------------------------------------------------
            # Conversion Options
            # -----------------------------------------------------------------
            with gr.Group():
                gr.Markdown("##### 2️⃣ Choose Output Format")
                with gr.Row():
                    convert_3dgs = gr.Checkbox(
                        value=True,
                        label="3DGS Format (.ply)",
                        info="For SuperSplat, gsplat viewers",
                    )
                    convert_pointcloud = gr.Checkbox(
                        value=False,
                        label="Point Cloud (.ply)",
                        info="For MeshLab, Blender, CloudCompare",
                    )
            
            # -----------------------------------------------------------------
            # Downsampling (important for large files)
            # -----------------------------------------------------------------
            with gr.Group():
                gr.Markdown("##### 3️⃣ Downsampling (for large files)")
                
                downsample_presets = gr.Radio(
                    choices=["Full Quality", "Web Viewer (500k)", "Quick Preview (100k)", "Custom"],
                    value="Web Viewer (500k)",
                    label="Preset",
                    info="Lyra can output 2M+ Gaussians. Downsample for faster viewing.",
                )
                
                with gr.Row():
                    downsample_max_points = gr.Slider(
                        minimum=0, maximum=2000000, value=500000, step=50000,
                        label="Max Points",
                        info="0 = keep all",
                    )
                    downsample_min_opacity = gr.Slider(
                        minimum=0.0, maximum=0.5, value=0.01, step=0.01,
                        label="Min Opacity Filter",
                        info="Remove near-invisible Gaussians",
                    )
            
            # -----------------------------------------------------------------
            # Convert Button + Status
            # -----------------------------------------------------------------
            with gr.Row():
                convert_btn = gr.Button(
                    "🔄 Convert PLY",
                    variant="primary",
                    size="lg",
                )
            
            convert_status = gr.Textbox(
                value="",
                label="Conversion Result",
                interactive=False,
                lines=3,
            )
            
            # -----------------------------------------------------------------
            # Blender Export (separate section)
            # -----------------------------------------------------------------
            with gr.Accordion("🎨 Blender Import Helper", open=False):
                gr.Markdown("""
                Generate a Python script to import the point cloud into Blender.
                Run the script in Blender's Scripting workspace.
                """)
                
                blender_ply_path = gr.Textbox(
                    value="",
                    label="PLY Path for Blender",
                    placeholder="Use a converted *_pointcloud.ply file",
                    info="Point cloud format works best in Blender",
                )
                
                with gr.Row():
                    blender_display_mode = gr.Radio(
                        choices=["points", "spheres", "cubes"],
                        value="points",
                        label="Display Mode",
                    )
                    blender_point_size = gr.Slider(
                        minimum=0.001, maximum=0.1, value=0.01, step=0.001,
                        label="Point Size",
                        info="For spheres/cubes mode",
                    )
                    blender_max_points = gr.Number(
                        value=0,
                        label="Max Points",
                        precision=0,
                        info="0 = all",
                    )
                
                with gr.Row():
                    blender_generate_btn = gr.Button("Generate Script", variant="secondary")
                    blender_open_btn = gr.Button("Open in Blender", variant="primary")
                
                blender_script_output = gr.Textbox(
                    value="",
                    label="Script Path",
                    interactive=False,
                    lines=2,
                )
    
    # =========================================================================
    # Return all components for event binding
    # =========================================================================
    return {
        # Generation controls
        "exec_mode": exec_mode,
        "status": status,
        "serverless_settings": serverless_settings,
        "check_serverless_btn": check_serverless_btn,
        "cancel_btn": cancel_btn,
        "edit_creds_btn": edit_creds_btn,
        "creds_group": creds_group,
        "endpoint_id": endpoint_id,
        "api_key": api_key,
        "save_creds_btn": save_creds_btn,
        "current_job_id": current_job_id,
        "generation_mode": generation_mode,
        "num_views": num_views,
        "camera_motion": camera_motion,
        "multi_trajectory": multi_trajectory,
        "foreground_masking": foreground_masking,
        "num_gaussians": num_gaussians,
        "seed": seed,
        "output_name": output_name,
        "output_dir": output_dir,
        "output_ply": output_ply,
        "output_video": output_video,
        "generate_btn": generate_btn,
        "queue_btn": queue_btn,
        "progress_display": progress_display,
        "logs_box": logs_box,
        # Post-processing controls
        "ply_dropdown": ply_dropdown,
        "refresh_ply_btn": refresh_ply_btn,
        "convert_pointcloud": convert_pointcloud,
        "convert_3dgs": convert_3dgs,
        "ply_input_path": ply_input_path,
        "downsample_max_points": downsample_max_points,
        "downsample_min_opacity": downsample_min_opacity,
        "downsample_presets": downsample_presets,
        "convert_btn": convert_btn,
        "convert_status": convert_status,
        # Blender components
        "blender_ply_path": blender_ply_path,
        "blender_display_mode": blender_display_mode,
        "blender_point_size": blender_point_size,
        "blender_max_points": blender_max_points,
        "blender_generate_btn": blender_generate_btn,
        "blender_open_btn": blender_open_btn,
        "blender_script_output": blender_script_output,
    }
