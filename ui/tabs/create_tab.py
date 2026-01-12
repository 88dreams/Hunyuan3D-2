"""
2DGS Tab - Video to 3D Mesh Pipeline

This tab provides the UI for the 2DGS Pipeline (ViPE + 2DGS)
which converts Gen3C videos into 3D meshes.

The pipeline is a ONE-SHOT serverless endpoint that handles:
1. ViPE pose extraction
2. Format conversion to COLMAP
3. Point cloud initialization
4. 2DGS training
5. Mesh extraction

Endpoint ID: s9txp6edtf2vg4
"""

import gradio as gr
from pathlib import Path


# Default paths
MESH_DEFAULT_OUTPUT_DIR = "/srv/searidge_share/outputs/mesh_2dgs"
GEN3C_DEFAULT_OUTPUT_DIR = "/srv/searidge_share/outputs/gen3c"

# Default endpoint
DEFAULT_2DGS_ENDPOINT_ID = "s9txp6edtf2vg4"


def create_2dgs_tab(
    default_endpoint_id: str = DEFAULT_2DGS_ENDPOINT_ID,
    default_api_key: str = "",
):
    """
    Create the 2DGS tab for video-to-mesh pipeline.
    
    This is a simplified one-shot workflow:
    1. Video Input → 2. Parameters → 3. Generate → 4. Download Mesh
    
    Args:
        default_endpoint_id: Default RunPod endpoint ID
        default_api_key: Default serverless API key
    
    Returns:
        Dictionary of all UI components for event handler binding
    """
    
    # ===========================================
    # TWO-COLUMN LAYOUT (matching Sharp/other tabs)
    # ===========================================
    with gr.Row():
        # LEFT COLUMN: All controls
        with gr.Column(scale=1):
            gr.Markdown("""
            **ViPE + 2DGS**: Extract camera poses and reconstruct 3D geometry from Gen3C videos.
            """)
            
            # VIDEO INPUT
            with gr.Group():
                gr.Markdown("### Video Input")
                
                video_source = gr.Radio(
                    choices=["Upload Video", "Video Path", "Use Gen3C Output"],
                    value="Use Gen3C Output",
                    label="Source",
                )
                
                # Upload option
                video_upload = gr.Video(
                    label="Upload Gen3C Video",
                    format="mp4",
                    visible=False,
                )
                upload_group = video_upload  # For compatibility
                
                # Path option  
                video_path = gr.Textbox(
                    value="",
                    label="Video Path",
                    placeholder="/path/to/gen3c_video.mp4 or https://...",
                    visible=False,
                )
                path_group = video_path  # For compatibility
                
                # Gen3C output option (default visible)
                with gr.Column(visible=True) as gen3c_group:
                    gen3c_output_dir = gr.Textbox(
                        value=GEN3C_DEFAULT_OUTPUT_DIR,
                        label="Gen3C Output Directory",
                    )
                    video_dropdown = gr.Dropdown(
                        choices=[],
                        label="Select Video",
                    )
                    refresh_videos_btn = gr.Button("🔄 Refresh", size="sm")
                
                # Video info
                video_info = gr.Textbox(
                    value="No video selected",
                    label="Video Info",
                    interactive=False,
                    lines=2,
                )
            
            # PIPELINE PARAMETERS
            with gr.Group():
                gr.Markdown("### Pipeline Parameters")
                
                with gr.Row():
                    iterations = gr.Slider(
                        minimum=1000,
                        maximum=10000,
                        value=5000,
                        step=500,
                        label="Training Iterations",
                    )
                    mesh_quality = gr.Radio(
                        choices=["fast", "balanced", "high", "ultra"],
                        value="high",
                        label="Mesh Quality",
                        info="Higher = better geometry, slower extraction",
                    )
                
                with gr.Row():
                    output_format = gr.Radio(
                        choices=["glb", "obj", "ply"],
                        value="glb",
                        label="Output Format",
                    )
                    output_dir = gr.Textbox(
                        value=MESH_DEFAULT_OUTPUT_DIR,
                        label="Output Directory",
                    )
            
            # EXECUTION
            with gr.Group():
                gr.Markdown("### Output")
                output_stats = gr.JSON(
                    label="Pipeline Statistics",
                    value={},
                )
            
            generate_btn = gr.Button(
                "🚀 Generate 3D Mesh",
                variant="primary",
                size="lg",
            )
            cancel_btn = gr.Button(
                "⏹️ Cancel",
                variant="stop",
                size="lg",
                visible=False,
            )
            status_text = gr.Textbox(
                value="Ready to start...",
                label="Status",
                interactive=False,
                lines=2,
            )
        
        # RIGHT COLUMN: Preview outputs
        with gr.Column(scale=1):
            output_file = gr.File(
                label="Downloaded Mesh",
                file_types=[".glb", ".obj", ".ply"],
            )
            twodgs_3d_viewer = gr.Model3D(
                label="3D Preview",
                height=400,
                clear_color=[0.1, 0.1, 0.1, 1.0],
            )
    
    # ===========================================
    # ENDPOINT CONFIGURATION (Accordion)
    # ===========================================
    with gr.Accordion("⚙️ Endpoint Settings", open=False):
        with gr.Row():
            endpoint_id = gr.Textbox(
                value=default_endpoint_id,
                label="2DGS Endpoint ID",
                placeholder="s9txp6edtf2vg4",
                info="RunPod serverless endpoint ID"
            )
            api_key = gr.Textbox(
                value=default_api_key,
                label="API Key",
                type="password",
                placeholder="rp_...",
                info="Your RunPod API key"
            )
        
        with gr.Row():
            s3_bucket = gr.Textbox(
                value="arkrunr",
                label="S3 Bucket",
                info="For video upload and mesh download"
            )
            s3_region = gr.Textbox(
                value="us-west-1",
                label="S3 Region",
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
        "gen3c_group": gen3c_group,
        "gen3c_output_dir": gen3c_output_dir,
        "video_dropdown": video_dropdown,
        "refresh_videos_btn": refresh_videos_btn,
        "video_info": video_info,
        
        # Parameters
        "iterations": iterations,
        "mesh_quality": mesh_quality,
        "output_format": output_format,
        "output_dir": output_dir,
        
        # Execution
        "generate_btn": generate_btn,
        "cancel_btn": cancel_btn,
        "status_text": status_text,
        
        # Output
        "output_file": output_file,
        "output_stats": output_stats,
        "twodgs_3d_viewer": twodgs_3d_viewer,
        
        # Settings
        "endpoint_id": endpoint_id,
        "api_key": api_key,
        "s3_bucket": s3_bucket,
        "s3_region": s3_region,
    }


# Legacy alias for backwards compatibility
def create_create_tab(*args, **kwargs):
    """Alias for create_2dgs_tab (backwards compatibility)."""
    return create_2dgs_tab(*args, **kwargs)
