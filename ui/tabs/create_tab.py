"""
CREATE Tab - Interior Stage Reconstruction Pipeline

This tab implements the full Gen3C → ViPE → 2DGS → Mesh pipeline
for reconstructing 3D stages from interior photographs.

Workflow:
1. Upload Gen3C video (or provide path)
2. Run ViPE to extract camera poses + depth
3. Convert to 2DGS format
4. Train 2DGS model
5. Extract mesh (GLB/OBJ)
"""

import gradio as gr
from pathlib import Path


# Default paths
VIPE_DEFAULT_OUTPUT_DIR = "./outputs/vipe"
MESH_DEFAULT_OUTPUT_DIR = "./outputs/mesh_vipe"


def create_create_tab(
    default_runpod_url: str = "",
    default_endpoint_id: str = "",
    default_api_key: str = "",
):
    """
    Create the CREATE tab for stage reconstruction pipeline.
    
    This is a multi-step workflow:
    1. Video Input → 2. ViPE Pose Extraction → 3. 2DGS Training → 4. Mesh Export
    
    Args:
        default_runpod_url: Default RunPod pod URL
        default_endpoint_id: Default serverless endpoint ID  
        default_api_key: Default serverless API key
    
    Returns:
        Dictionary of all UI components for event handler binding
    """
    
    gr.Markdown("""
    ### Stage Reconstruction Pipeline
    **Video → Poses → 3D Training → Mesh**
    
    Create 3D architectural meshes from Gen3C videos using ViPE pose extraction and 2D Gaussian Splatting.
    """)
    
    # ===========================================
    # STEP 1: VIDEO INPUT
    # ===========================================
    with gr.Group():
        gr.Markdown("#### Step 1: Video Input")
        
        with gr.Row():
            video_source = gr.Radio(
                choices=["Upload Video", "Video Path", "Use Gen3C Output"],
                value="Video Path",
                label="Source",
            )
        
        # Upload option
        with gr.Group(visible=False) as upload_group:
            video_upload = gr.Video(
                label="Upload Gen3C Video",
                format="mp4",
            )
        
        # Path option  
        with gr.Group(visible=True) as path_group:
            video_path = gr.Textbox(
                value="",
                label="Video Path",
                placeholder="/path/to/gen3c_video.mp4",
            )
            browse_video_btn = gr.Button("Browse...", size="sm")
        
        # Gen3C output option
        with gr.Group(visible=False) as gen3c_group:
            gen3c_output_dir = gr.Textbox(
                value="./outputs/gen3c",
                label="Gen3C Output Directory",
            )
            video_dropdown = gr.Dropdown(
                choices=[],
                label="Select Video",
            )
            refresh_videos_btn = gr.Button("Refresh", size="sm")
        
        # Video preview
        video_preview = gr.Video(
            label="Preview",
            visible=False,
        )
        video_info = gr.Textbox(
            value="No video selected",
            label="Video Info",
            interactive=False,
            lines=2,
        )
    
    # ===========================================
    # STEP 2: VIPE POSE EXTRACTION
    # ===========================================
    with gr.Group():
        gr.Markdown("#### Step 2: ViPE Pose Extraction")
        gr.Markdown("*Extract camera poses and depth maps from video*")
        
        with gr.Row():
            vipe_exec_mode = gr.Radio(
                choices=["RunPod Serverless", "RunPod Pod"],
                value="RunPod Serverless",
                label="Execution Mode",
            )
        
        # Serverless credentials
        with gr.Group() as vipe_creds_group:
            with gr.Row():
                vipe_endpoint_id = gr.Textbox(
                    value=default_endpoint_id,
                    label="ViPE Endpoint ID",
                    placeholder="vipe-endpoint-id",
                )
                vipe_api_key = gr.Textbox(
                    value=default_api_key,
                    label="API Key",
                    type="password",
                )
            save_vipe_creds_btn = gr.Button("Save Credentials", size="sm")
        
        # ViPE output settings
        with gr.Row():
            vipe_output_dir = gr.Textbox(
                value=VIPE_DEFAULT_OUTPUT_DIR,
                label="Output Directory",
            )
        
        with gr.Row():
            run_vipe_btn = gr.Button(
                "Run ViPE",
                variant="primary",
            )
            vipe_status = gr.Textbox(
                value="Ready",
                label="Status",
                interactive=False,
            )
        
        # ViPE results preview
        with gr.Accordion("ViPE Results", open=False):
            vipe_results_info = gr.Textbox(
                value="No results yet",
                label="Extracted Data",
                interactive=False,
                lines=4,
            )
    
    # ===========================================
    # STEP 3: 2DGS TRAINING
    # ===========================================
    with gr.Group():
        gr.Markdown("#### Step 3: 2DGS Training")
        gr.Markdown("*Train 2D Gaussian Splatting model from poses*")
        
        with gr.Row():
            gs_exec_mode = gr.Radio(
                choices=["RunPod Pod", "Local"],
                value="RunPod Pod",
                label="Execution Mode",
            )
        
        # Training parameters
        with gr.Row():
            gs_iterations = gr.Slider(
                minimum=1000,
                maximum=30000,
                value=5000,
                step=1000,
                label="Training Iterations",
            )
            gs_resolution = gr.Slider(
                minimum=256,
                maximum=1024,
                value=512,
                step=128,
                label="Mesh Resolution",
            )
        
        with gr.Row():
            gs_output_name = gr.Textbox(
                value="stage_model",
                label="Model Name",
            )
        
        with gr.Row():
            run_gs_btn = gr.Button(
                "Train 2DGS",
                variant="primary",
            )
            gs_status = gr.Textbox(
                value="Waiting for ViPE results",
                label="Status",
                interactive=False,
            )
        
        # Training progress
        with gr.Accordion("Training Progress", open=False):
            gs_progress = gr.Textbox(
                value="",
                label="Progress",
                interactive=False,
                lines=6,
            )
    
    # ===========================================
    # STEP 4: MESH EXTRACTION
    # ===========================================
    with gr.Group():
        gr.Markdown("#### Step 4: Mesh Extraction")
        gr.Markdown("*Extract and export final mesh*")
        
        with gr.Row():
            mesh_format = gr.Radio(
                choices=["GLB", "OBJ", "PLY"],
                value="GLB",
                label="Export Format",
            )
            mesh_cleanup = gr.Checkbox(
                value=True,
                label="Post-process Mesh",
                info="Remove floaters and clean geometry",
            )
        
        with gr.Row():
            mesh_output_dir = gr.Textbox(
                value=MESH_DEFAULT_OUTPUT_DIR,
                label="Output Directory",
            )
        
        with gr.Row():
            extract_mesh_btn = gr.Button(
                "Extract Mesh",
                variant="primary",
            )
            mesh_status = gr.Textbox(
                value="Waiting for training",
                label="Status",
                interactive=False,
            )
    
    # ===========================================
    # FINAL OUTPUT
    # ===========================================
    with gr.Group():
        gr.Markdown("#### Output")
        
        with gr.Row():
            output_mesh_file = gr.File(
                label="Downloaded Mesh",
                file_types=[".glb", ".obj", ".ply"],
            )
            download_mesh_btn = gr.Button(
                "Download Mesh",
                variant="secondary",
            )
        
        output_stats = gr.JSON(
            label="Pipeline Statistics",
            value={},
        )
    
    # ===========================================
    # ONE-CLICK PIPELINE
    # ===========================================
    with gr.Group():
        gr.Markdown("---")
        gr.Markdown("#### One-Click Pipeline")
        gr.Markdown("*Run all steps automatically*")
        
        run_full_pipeline_btn = gr.Button(
            "Run Full Pipeline (Video → Mesh)",
            variant="primary",
            size="lg",
        )
        
        pipeline_progress = gr.Textbox(
            value="Ready to start...",
            label="Pipeline Progress",
            interactive=False,
            lines=3,
        )
    
    # ===========================================
    # VISIBILITY HANDLERS
    # ===========================================
    def update_video_source_visibility(source):
        return (
            gr.update(visible=source == "Upload Video"),
            gr.update(visible=source == "Video Path"),
            gr.update(visible=source == "Use Gen3C Output"),
        )
    
    video_source.change(
        fn=update_video_source_visibility,
        inputs=[video_source],
        outputs=[upload_group, path_group, gen3c_group],
    )
    
    # Return all components for external event binding
    return {
        # Video input
        "video_source": video_source,
        "upload_group": upload_group,
        "video_upload": video_upload,
        "path_group": path_group,
        "video_path": video_path,
        "browse_video_btn": browse_video_btn,
        "gen3c_group": gen3c_group,
        "gen3c_output_dir": gen3c_output_dir,
        "video_dropdown": video_dropdown,
        "refresh_videos_btn": refresh_videos_btn,
        "video_preview": video_preview,
        "video_info": video_info,
        
        # ViPE
        "vipe_exec_mode": vipe_exec_mode,
        "vipe_creds_group": vipe_creds_group,
        "vipe_endpoint_id": vipe_endpoint_id,
        "vipe_api_key": vipe_api_key,
        "save_vipe_creds_btn": save_vipe_creds_btn,
        "vipe_output_dir": vipe_output_dir,
        "run_vipe_btn": run_vipe_btn,
        "vipe_status": vipe_status,
        "vipe_results_info": vipe_results_info,
        
        # 2DGS Training
        "gs_exec_mode": gs_exec_mode,
        "gs_iterations": gs_iterations,
        "gs_resolution": gs_resolution,
        "gs_output_name": gs_output_name,
        "run_gs_btn": run_gs_btn,
        "gs_status": gs_status,
        "gs_progress": gs_progress,
        
        # Mesh extraction
        "mesh_format": mesh_format,
        "mesh_cleanup": mesh_cleanup,
        "mesh_output_dir": mesh_output_dir,
        "extract_mesh_btn": extract_mesh_btn,
        "mesh_status": mesh_status,
        
        # Output
        "output_mesh_file": output_mesh_file,
        "download_mesh_btn": download_mesh_btn,
        "output_stats": output_stats,
        
        # Full pipeline
        "run_full_pipeline_btn": run_full_pipeline_btn,
        "pipeline_progress": pipeline_progress,
    }
