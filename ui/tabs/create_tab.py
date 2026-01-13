"""
2DGS Tab - Video to 3D Mesh Pipeline (Multi-Video Support)

This tab provides the UI for the 2DGS Pipeline (ViPE + 2DGS)
which converts videos into 3D meshes.

Features:
- Multi-video selection (up to 4 videos)
- Support for Gen3C and LTX-2 videos
- Video preview before processing
- Pose alignment and merging

Endpoint ID: s9txp6edtf2vg4
"""

import gradio as gr
from pathlib import Path


# Default paths
MESH_DEFAULT_OUTPUT_DIR = "/srv/searidge_share/outputs/mesh_2dgs"
GEN3C_DEFAULT_OUTPUT_DIR = "/srv/searidge_share/outputs/gen3c"
LTX2_DEFAULT_OUTPUT_DIR = "/srv/searidge_share/outputs/ltx2"

# Default endpoint
DEFAULT_2DGS_ENDPOINT_ID = "s9txp6edtf2vg4"

# Maximum videos for multi-video mode
MAX_VIDEOS = 4


def create_2dgs_tab(
    default_endpoint_id: str = DEFAULT_2DGS_ENDPOINT_ID,
    default_api_key: str = "",
):
    """
    Create the 2DGS tab for video-to-mesh pipeline with multi-video support.
    
    Args:
        default_endpoint_id: Default RunPod endpoint ID
        default_api_key: Default serverless API key
    
    Returns:
        Dictionary of all UI components for event handler binding
    """
    
    # ===========================================
    # TWO-COLUMN LAYOUT
    # ===========================================
    with gr.Row():
        # LEFT COLUMN: Controls
        with gr.Column(scale=1):
            gr.Markdown("""
            **ViPE + 2DGS**: Extract camera poses and reconstruct 3D geometry from videos.
            Select up to 4 videos for better reconstruction quality.
            """)
            
            # VIDEO SELECTION
            with gr.Group():
                gr.Markdown("### Video Selection")
                
                video_source = gr.Radio(
                    choices=["Select Videos", "Upload Video"],
                    value="Select Videos",
                    label="Source",
                )
                
                # Select from directories option (default)
                with gr.Column(visible=True) as select_group:
                    gr.Markdown("**Available Videos** (Gen3C + LTX-2)")
                    video_checkboxes = gr.CheckboxGroup(
                        choices=[],
                        value=[],
                        label="Select up to 4 videos",
                        info="Videos from Gen3C and LTX-2 directories",
                    )
                    with gr.Row():
                        refresh_videos_btn = gr.Button("Refresh", size="sm")
                        selected_count = gr.Markdown("**Selected: 0/4**")
                
                # Upload option
                with gr.Column(visible=False) as upload_group:
                    video_uploads = gr.File(
                        label="Upload Videos (up to 4)",
                        file_count="multiple",
                        file_types=[".mp4", ".avi", ".mov"],
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
                        info="Higher = better geometry, slower",
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
                
                output_name = gr.Textbox(
                    value="mesh_output",
                    label="Output Name",
                )
            
            # GENERATE BUTTON
            generate_btn = gr.Button(
                "Generate 3D Mesh",
                variant="primary",
                size="lg",
            )
            status_text = gr.Textbox(
                value="Ready - Select videos to begin",
                label="Status",
                interactive=False,
                lines=2,
            )
            
            # STATS
            with gr.Accordion("Pipeline Statistics", open=False):
                output_stats = gr.JSON(
                    label="Statistics",
                    value={},
                )
        
        # RIGHT COLUMN: Previews
        with gr.Column(scale=1):
            gr.Markdown("### Selected Videos (0/4)")
            selected_videos_label = gr.Markdown("*No videos selected*")
            
            # Video preview grid (2x2)
            with gr.Row():
                video_preview_1 = gr.Video(
                    label="Video 1",
                    height=180,
                    visible=True,
                    interactive=False,
                )
                video_preview_2 = gr.Video(
                    label="Video 2", 
                    height=180,
                    visible=True,
                    interactive=False,
                )
            with gr.Row():
                video_preview_3 = gr.Video(
                    label="Video 3",
                    height=180,
                    visible=True,
                    interactive=False,
                )
                video_preview_4 = gr.Video(
                    label="Video 4",
                    height=180,
                    visible=True,
                    interactive=False,
                )
            
            # 3D Output Preview
            gr.Markdown("### 3D Mesh Output")
            twodgs_3d_viewer = gr.Model3D(
                label="3D Preview",
                height=350,
                clear_color=[0.1, 0.1, 0.1, 1.0],
            )
            output_file = gr.File(
                label="Download Mesh",
                file_types=[".glb", ".obj", ".ply"],
            )
    
    # Note: Endpoint settings moved to Settings page
    # These are placeholder State components that get populated from settings
    
    # ===========================================
    # VISIBILITY HANDLERS
    # ===========================================
    def update_source_visibility(source):
        return (
            gr.update(visible=source == "Select Videos"),
            gr.update(visible=source == "Upload Video"),
        )
    
    video_source.change(
        fn=update_source_visibility,
        inputs=[video_source],
        outputs=[select_group, upload_group],
    )
    
    # Return all components for external event binding
    return {
        # Video selection
        "video_source": video_source,
        "select_group": select_group,
        "video_checkboxes": video_checkboxes,
        "refresh_videos_btn": refresh_videos_btn,
        "selected_count": selected_count,
        "upload_group": upload_group,
        "video_uploads": video_uploads,
        
        # Video previews
        "selected_videos_label": selected_videos_label,
        "video_preview_1": video_preview_1,
        "video_preview_2": video_preview_2,
        "video_preview_3": video_preview_3,
        "video_preview_4": video_preview_4,
        
        # Parameters
        "iterations": iterations,
        "mesh_quality": mesh_quality,
        "output_format": output_format,
        "output_dir": output_dir,
        "output_name": output_name,
        
        # Execution
        "generate_btn": generate_btn,
        "status_text": status_text,
        
        # Output
        "output_file": output_file,
        "output_stats": output_stats,
        "twodgs_3d_viewer": twodgs_3d_viewer,
        
        # Settings (populated from Settings page - these are just placeholders)
        "endpoint_id": gr.State(default_endpoint_id),
        "api_key": gr.State(default_api_key),
        "s3_bucket": gr.State("arkrunr"),
        "s3_region": gr.State("us-west-1"),
        
        # Legacy compatibility (for old handlers)
        "video_path": gr.State(""),
        "video_upload": video_uploads,
        "video_dropdown": video_checkboxes,
        "gen3c_output_dir": gr.State(GEN3C_DEFAULT_OUTPUT_DIR),
        "video_info": gr.State(""),
    }


# Legacy alias for backwards compatibility
def create_create_tab(*args, **kwargs):
    """Alias for create_2dgs_tab (backwards compatibility)."""
    return create_2dgs_tab(*args, **kwargs)
