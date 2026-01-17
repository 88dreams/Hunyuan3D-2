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
                    
                    # Sort and filter controls
                    with gr.Row():
                        video_sort_by = gr.Dropdown(
                            choices=["Date (newest)", "Date (oldest)", "Filename (A-Z)", "Filename (Z-A)", "Model (Gen3C first)", "Model (LTX-2 first)", "Tagged first"],
                            value="Date (newest)",
                            label="Sort by",
                            scale=2,
                        )
                        video_limit = gr.Dropdown(
                            choices=["10", "20", "30", "50", "All"],
                            value="10",
                            label="Show",
                            scale=1,
                        )
                        refresh_videos_btn = gr.Button("🔄", size="sm", scale=0, min_width=40)
                    
                    # Tag filter
                    try:
                        from utils.tag_manager import get_tag_manager
                        _tag_choices = get_tag_manager().get_all_tags_flat()
                    except ImportError:
                        _tag_choices = []
                    
                    video_tag_filter = gr.Dropdown(
                        choices=_tag_choices,
                        value=[],
                        label="Filter by Tags",
                        multiselect=True,
                        info="Show only videos with selected tags",
                    )
                    
                    # Single-column video list with custom CSS
                    video_checkboxes = gr.CheckboxGroup(
                        choices=[],
                        value=[],
                        label="",
                        info="",
                        elem_classes=["video-list-single-column"],
                    )
                    selected_count = gr.Markdown("**Selected: 0**")
                
                # Upload option
                with gr.Column(visible=False) as upload_group:
                    video_uploads = gr.File(
                        label="Upload Videos (max configurable in Advanced Options)",
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
            
            # ADVANCED OPTIONS
            with gr.Accordion("Advanced Options", open=False):
                with gr.Row():
                    max_videos = gr.Slider(
                        minimum=4,
                        maximum=8,
                        value=4,
                        step=1,
                        label="Max Videos",
                        info="More videos = better quality, longer processing"
                    )
                    depth_threshold = gr.Slider(
                        minimum=0.3,
                        maximum=0.7,
                        value=0.5,
                        step=0.1,
                        label="Depth Coverage Threshold",
                        info="Frames below this coverage % are skipped"
                    )
                
                with gr.Row():
                    save_vipe_checkpoint = gr.Checkbox(
                        label="Save ViPE Checkpoint",
                        value=True,
                        info="Save ViPE outputs to S3 for resume capability"
                    )
                    resume_from_checkpoint = gr.Textbox(
                        label="Resume from Checkpoint",
                        placeholder="s3://arkrunr/MediaContent/2dgs-pipeline/vipe-checkpoints/...",
                        value="",
                        info="Paste checkpoint URL to skip ViPE step"
                    )
                
                gr.Markdown("""
                **Advanced Settings:**
                - **Max Videos**: More camera angles improve 3D reconstruction quality
                - **Depth Threshold**: Higher values filter out more low-quality frames
                - **Save ViPE Checkpoint**: If enabled, saves ViPE outputs to S3. If job fails after ViPE, you can resume using the checkpoint URL
                - **Resume from Checkpoint**: Paste a checkpoint URL to skip the ~15min ViPE step
                """)
            
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
            with gr.Row():
                gr.Markdown("### Video Previews")
                play_all_btn = gr.Button("▶ Play All", size="sm", scale=0, min_width=100, elem_id="play-all-btn")
            selected_videos_label = gr.Markdown("*No videos selected*")
            
            # Video preview grid (2x2) with filename labels above each
            with gr.Row():
                with gr.Column(scale=1, min_width=150):
                    video_label_1 = gr.Markdown("**1.** —", elem_classes=["video-slot-label"])
                    video_preview_1 = gr.Video(
                        label="",
                        height=170,
                        visible=True,
                        interactive=False,
                        elem_id="twodgs-video-1",
                    )
                with gr.Column(scale=1, min_width=150):
                    video_label_2 = gr.Markdown("**2.** —", elem_classes=["video-slot-label"])
                    video_preview_2 = gr.Video(
                        label="", 
                        height=170,
                        visible=True,
                        interactive=False,
                        elem_id="twodgs-video-2",
                    )
            with gr.Row():
                with gr.Column(scale=1, min_width=150):
                    video_label_3 = gr.Markdown("**3.** —", elem_classes=["video-slot-label"])
                    video_preview_3 = gr.Video(
                        label="",
                        height=170,
                        visible=True,
                        interactive=False,
                        elem_id="twodgs-video-3",
                    )
                with gr.Column(scale=1, min_width=150):
                    video_label_4 = gr.Markdown("**4.** —", elem_classes=["video-slot-label"])
                    video_preview_4 = gr.Video(
                        label="",
                        height=170,
                        visible=True,
                        interactive=False,
                        elem_id="twodgs-video-4",
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
        "video_sort_by": video_sort_by,
        "video_limit": video_limit,
        "video_tag_filter": video_tag_filter,
        "refresh_videos_btn": refresh_videos_btn,
        "selected_count": selected_count,
        "upload_group": upload_group,
        "video_uploads": video_uploads,
        
        # Video previews
        "selected_videos_label": selected_videos_label,
        "video_label_1": video_label_1,
        "video_label_2": video_label_2,
        "video_label_3": video_label_3,
        "video_label_4": video_label_4,
        "video_preview_1": video_preview_1,
        "video_preview_2": video_preview_2,
        "video_preview_3": video_preview_3,
        "video_preview_4": video_preview_4,
        "play_all_btn": play_all_btn,
        
        # Parameters
        "iterations": iterations,
        "mesh_quality": mesh_quality,
        "output_format": output_format,
        "output_dir": output_dir,
        "output_name": output_name,
        
        # Advanced options
        "max_videos": max_videos,
        "depth_threshold": depth_threshold,
        "save_vipe_checkpoint": save_vipe_checkpoint,
        "resume_from_checkpoint": resume_from_checkpoint,
        
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
