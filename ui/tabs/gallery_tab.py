"""
Gallery Tab - Browse and tag render outputs from all models.

Provides a unified interface to:
- Browse outputs from Gen3C, LTX-2, Hunyuan, Trellis, 2DGS
- Filter by model and tags
- View and edit tags for any output
- Auto-save tags on selection
- Bulk clear tags
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import gradio as gr

from utils.tag_manager import TagManager, get_tag_manager

# Output directories by model
MODEL_OUTPUT_DIRS = {
    "Gen3C": "/srv/searidge_share/outputs/gen3c",
    "LTX-2": "/srv/searidge_share/outputs/ltx2",
    "Hunyuan": "/srv/searidge_share/outputs/hunyuan",
    "Trellis": "/srv/searidge_share/outputs/trellis",
    "2DGS": "/srv/searidge_share/outputs/2dgs",
}

# File extensions by model type
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi"}
MESH_EXTENSIONS = {".glb", ".obj", ".ply", ".stl"}

# Tag layout - Column 1 (positive), Column 2 (action/status)
TAG_COLUMN_1 = ["approved", "favorite", "best-take"]
TAG_COLUMN_2 = ["review", "bad", "delete"]


def scan_output_files(
    model_filter: Optional[str] = None,
    tag_filter: Optional[List[str]] = None,
    limit: int = 50,
    include_deleted: bool = False,
) -> List[Dict[str, Any]]:
    """
    Scan output directories for files.
    
    Args:
        model_filter: Optional model to filter by (Gen3C, LTX-2, etc.)
        tag_filter: Optional list of tags to filter by
        limit: Maximum number of files to return
        include_deleted: If False, exclude files with 'delete' tag
        
    Returns:
        List of file info dicts with path, name, model, mtime, tags
    """
    tag_manager = get_tag_manager()
    files = []
    
    # Determine which directories to scan
    if model_filter and model_filter != "All":
        dirs_to_scan = {model_filter: MODEL_OUTPUT_DIRS.get(model_filter, "")}
    else:
        dirs_to_scan = MODEL_OUTPUT_DIRS
    
    for model, output_dir in dirs_to_scan.items():
        if not output_dir or not os.path.exists(output_dir):
            continue
        
        # Determine extensions based on model
        if model in ["Gen3C", "LTX-2"]:
            extensions = VIDEO_EXTENSIONS
        else:
            extensions = MESH_EXTENSIONS | VIDEO_EXTENSIONS
        
        # Scan directory
        for path in Path(output_dir).iterdir():
            if path.is_file() and path.suffix.lower() in extensions:
                try:
                    mtime = path.stat().st_mtime
                    file_tags = tag_manager.get_tags(str(path))
                    
                    # Exclude files with 'delete' tag unless explicitly requested
                    if not include_deleted and "delete" in file_tags:
                        continue
                    
                    # Apply tag filter
                    if tag_filter:
                        if not any(t in file_tags for t in tag_filter):
                            continue
                    
                    files.append({
                        "path": str(path),
                        "name": path.name,
                        "model": model,
                        "mtime": mtime,
                        "mtime_str": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
                        "tags": file_tags,
                        "is_video": path.suffix.lower() in VIDEO_EXTENSIONS,
                    })
                except OSError:
                    pass
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: x["mtime"], reverse=True)
    
    return files[:limit]


def format_file_list(files: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Format file list for Gradio dropdown/radio.
    
    Returns:
        List of (display_name, path) tuples
    """
    result = []
    for f in files:
        tags_str = f" [{', '.join(f['tags'])}]" if f['tags'] else ""
        display = f"[{f['model']}] {f['name']}{tags_str} ({f['mtime_str']})"
        result.append((display, f["path"]))
    return result


# CSS for scrollable file list with more height and custom tag row styling
GALLERY_CSS = """
<style>
.gallery-file-list {
    max-height: 450px;
    overflow-y: auto;
}
.gallery-file-list .wrap {
    max-height: 430px;
    overflow-y: auto;
}
/* Remove separator between Custom checkbox and text input */
.custom-tag-row {
    gap: 8px !important;
    border: none !important;
}
.custom-tag-row > div {
    border: none !important;
    border-left: none !important;
    border-right: none !important;
}
.custom-tag-row input[type="text"] {
    margin-left: 0 !important;
}
</style>
"""


def create_gallery_tab() -> Dict[str, Any]:
    """
    Create the Gallery tab UI.
    
    Returns:
        Dictionary of UI components for event binding
    """
    tag_manager = get_tag_manager()
    
    # Inject CSS for scrollable file list
    gr.HTML(GALLERY_CSS)
    
    with gr.Row():
        # Left panel - Filters and file list
        with gr.Column(scale=1):
            # Combined filter box
            with gr.Group():
                gr.Markdown("### Filters")
                model_filter = gr.Dropdown(
                    choices=["All", "Gen3C", "LTX-2", "Hunyuan", "Trellis", "2DGS"],
                    value="All",
                    label="Model",
                )
                tag_filter = gr.Dropdown(
                    choices=tag_manager.get_all_tags_flat(),
                    value=[],
                    label="Tags",
                    multiselect=True,
                )
                limit_slider = gr.Slider(
                    minimum=10,
                    maximum=200,
                    value=50,
                    step=10,
                    label="Max Results",
                )
            
            # Refresh button between filter and file list
            refresh_btn = gr.Button("Refresh", variant="primary")
            
            # File list with scroll - taller to show more files
            gr.Markdown("### Files")
            file_list = gr.Radio(
                choices=[],
                label="",
                value=None,
                elem_classes=["gallery-file-list"],
            )
            
            file_count = gr.Markdown("*0 files*")
        
        # Right panel - Preview and tagging (aligned like Gen3C, Sharp, Lyra)
        with gr.Column(scale=1):
            # Fixed-height preview area - no header, aligned with Filters
            video_preview = gr.Video(
                label="Video Preview",
                height=400,
                visible=False,
                autoplay=True,
                loop=True,
            )
            model_preview = gr.Model3D(
                label="3D Preview",
                height=400,
                visible=False,
                clear_color=[0.1, 0.1, 0.1, 1.0],
            )
            # Placeholder with same height when no file selected
            no_preview = gr.HTML(
                value='<div style="height:400px;display:flex;align-items:center;justify-content:center;background:#1a1a1a;border-radius:8px;color:#666;"><span>Select a file to preview</span></div>',
                visible=True,
            )
            
            # File info
            file_info = gr.Markdown("**File:** —")
            
            # Tagging section - All in one Group box
            with gr.Group():
                # Row 1: Two columns of preset tags
                with gr.Row():
                    # Column 1: Positive tags
                    with gr.Column(scale=1, min_width=120):
                        tag_approved = gr.Checkbox(label="Approved", value=False)
                        tag_favorite = gr.Checkbox(label="Favorite", value=False)
                        tag_best_take = gr.Checkbox(label="Best-take", value=False)
                    
                    # Column 2: Review/status tags
                    with gr.Column(scale=1, min_width=120):
                        tag_review = gr.Checkbox(label="Review", value=False)
                        tag_bad = gr.Checkbox(label="Bad", value=False)
                        tag_delete = gr.Checkbox(label="Delete", value=False)
                
                # Row 2: Custom tag spanning full width - no separator between elements
                with gr.Row(elem_classes=["custom-tag-row"]):
                    tag_custom_checkbox = gr.Checkbox(label="Custom", value=False, scale=0, min_width=90)
                    custom_tag_input = gr.Textbox(
                        placeholder="Type tag, Enter to save",
                        show_label=False,
                        scale=1,
                        container=False,
                    )
            
            # Current custom tags display
            custom_tags_display = gr.Markdown("*Custom tags: —*")
            
            # Clear buttons stacked
            clear_tags_btn = gr.Button("Clear Tags", variant="primary", size="sm")
            clear_all_tags_btn = gr.Button("Clear Tags (ALL displayed)", variant="primary", size="sm")
            
            tag_status = gr.Markdown("*Select a file to edit tags*")
            
            # Delete confirmation dialog (hidden by default)
            with gr.Row(visible=False) as delete_confirm_row:
                gr.Markdown("**Delete this file?**")
                delete_yes_btn = gr.Button("Yes, Delete", size="sm", variant="stop")
                delete_no_btn = gr.Button("Cancel", size="sm")
            
            # Clear ALL confirmation dialog (hidden by default)
            with gr.Row(visible=False) as clear_all_confirm_row:
                gr.Markdown("**Clear tags from ALL displayed files?**")
                clear_all_yes_btn = gr.Button("Yes, Clear All", size="sm", variant="stop")
                clear_all_no_btn = gr.Button("Cancel", size="sm")
            
            # Current file state
            current_file_path = gr.State(value=None)
            current_file_model = gr.State(value=None)
            current_custom_tags = gr.State(value=[])  # Track custom tags separately
            displayed_file_paths = gr.State(value=[])  # Track displayed files for bulk clear
    
    # =========================================================================
    # Event Handlers
    # =========================================================================
    
    def refresh_file_list(model: str, tags: List[str], limit: int):
        """Refresh the file list based on filters."""
        tag_list = tags if tags else None
        files = scan_output_files(
            model_filter=model if model != "All" else None,
            tag_filter=tag_list,
            limit=int(limit),
            include_deleted=False,  # Never show deleted files
        )
        
        choices = format_file_list(files)
        count_text = f"*{len(files)} file(s)*"
        file_paths = [f["path"] for f in files]
        
        return gr.update(choices=choices, value=None), count_text, file_paths
    
    def select_file(file_path: str):
        """Handle file selection - show preview and load tags."""
        if not file_path:
            return (
                gr.update(visible=False),  # video
                gr.update(visible=False),  # model3d
                gr.update(visible=True),   # no_preview
                "**File:** —",
                False, False, False, False, False, False, False,  # tag checkboxes: approved, favorite, best-take, review, bad, delete, custom
                [],  # custom tags
                "*Custom tags: —*",
                "*Select a file to edit tags*",
                None,  # file path state
                None,  # model state
                gr.update(visible=False),  # delete confirm
            )
        
        path = Path(file_path)
        if not path.exists():
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                "**File:** —",
                False, False, False, False, False, False, False,
                [],
                "*Custom tags: —*",
                "*File not found*",
                None,
                None,
                gr.update(visible=False),
            )
        
        # Determine file type and model
        is_video = path.suffix.lower() in VIDEO_EXTENSIONS
        
        # Try to determine model from path
        model = None
        for m, dir_path in MODEL_OUTPUT_DIRS.items():
            if dir_path and file_path.startswith(dir_path):
                model = m
                break
        
        # Get existing tags
        tags = tag_manager.get_tags(file_path)
        
        # Separate predefined and custom tags
        predefined = set(TAG_COLUMN_1 + TAG_COLUMN_2)
        custom_tags = [t for t in tags if t not in predefined]
        has_custom = len(custom_tags) > 0
        
        # File info
        mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        size_mb = path.stat().st_size / (1024 * 1024)
        info = f"**File:** {path.name}  \n**Model:** {model or 'Unknown'} | **Modified:** {mtime} | **Size:** {size_mb:.1f} MB"
        
        custom_display = f"*Custom tags: {', '.join(custom_tags)}*" if custom_tags else "*Custom tags: —*"
        
        if is_video:
            return (
                gr.update(visible=True, value=file_path),  # video
                gr.update(visible=False),                   # model3d
                gr.update(visible=False),                   # no_preview
                info,
                "approved" in tags,
                "favorite" in tags,
                "best-take" in tags,
                "review" in tags,
                "bad" in tags,
                "delete" in tags,
                has_custom,  # custom checkbox
                custom_tags,
                custom_display,
                f"*Loaded {len(tags)} tag(s)*",
                file_path,
                model,
                gr.update(visible=False),  # hide delete confirm
            )
        else:
            return (
                gr.update(visible=False),                   # video
                gr.update(visible=True, value=file_path),   # model3d
                gr.update(visible=False),                   # no_preview
                info,
                "approved" in tags,
                "favorite" in tags,
                "best-take" in tags,
                "review" in tags,
                "bad" in tags,
                "delete" in tags,
                has_custom,  # custom checkbox
                custom_tags,
                custom_display,
                f"*Loaded {len(tags)} tag(s)*",
                file_path,
                model,
                gr.update(visible=False),  # hide delete confirm
            )
    
    def auto_save_tags(approved, favorite, best_take, review, bad, delete, custom_checkbox, custom_tags, file_path, model):
        """Auto-save tags when any checkbox changes (except delete which needs confirmation)."""
        if not file_path or not os.path.exists(file_path):
            return "*No file selected*", gr.update(visible=False)
        
        # If delete is checked, show confirmation instead of saving
        if delete:
            return "*Confirm deletion below*", gr.update(visible=True)
        
        # Build tag list (without delete)
        tags = []
        if approved:
            tags.append("approved")
        if favorite:
            tags.append("favorite")
        if best_take:
            tags.append("best-take")
        if review:
            tags.append("review")
        if bad:
            tags.append("bad")
        
        # Add custom tags if custom checkbox is checked
        if custom_checkbox and custom_tags:
            tags.extend(custom_tags)
        
        # Save
        tag_manager.set_tags(file_path, tags, model=model)
        
        if tags:
            return f"*Saved: {', '.join(tags)}*", gr.update(visible=False)
        else:
            return "*No tags*", gr.update(visible=False)
    
    def on_custom_input_change(text, current_checkbox):
        """Auto-check the custom checkbox when user starts typing."""
        if text and text.strip():
            return True
        return current_checkbox
    
    def confirm_delete(file_path, model, displayed_paths, model_f, tag_f, limit):
        """Actually delete the file after confirmation."""
        if not file_path:
            return ("*No file selected*", gr.update(visible=False), gr.update(), "*0 files*", 
                    displayed_paths, False, False, False, False, False, False, False, [], "*Custom tags: —*",
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=True))
        
        deleted = False
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted = True
            tag_manager.clear_tags(file_path)
        except OSError as e:
            return (f"*Delete failed: {e}*", gr.update(visible=False), gr.update(), 
                    f"*{len(displayed_paths)} file(s)*", displayed_paths,
                    False, False, False, False, False, False, False, [], "*Custom tags: —*",
                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=True))
        
        # Refresh the file list
        tag_list = tag_f if tag_f else None
        files = scan_output_files(
            model_filter=model_f if model_f != "All" else None,
            tag_filter=tag_list,
            limit=int(limit),
            include_deleted=False,
        )
        choices = format_file_list(files)
        new_paths = [f["path"] for f in files]
        
        status = "*File deleted*" if deleted else "*File removed from list*"
        
        return (status, gr.update(visible=False), gr.update(choices=choices, value=None), 
                f"*{len(files)} file(s)*", new_paths,
                False, False, False, False, False, False, False, [], "*Custom tags: —*",
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=True))
    
    def cancel_delete():
        """Cancel the delete operation."""
        return gr.update(visible=False), False, "*Delete cancelled*"
    
    def show_clear_all_confirm():
        """Show the clear all confirmation dialog."""
        return gr.update(visible=True)
    
    def cancel_clear_all():
        """Cancel the clear all operation."""
        return gr.update(visible=False), "*Clear all cancelled*"
    
    def confirm_clear_all(file_paths: List[str], model: str, tags: List[str], limit: int):
        """Clear tags from all currently displayed files after confirmation."""
        if not file_paths:
            return "*No files to clear*", gr.update(), "*0 files*", [], gr.update(visible=False)
        
        cleared = tag_manager.clear_tags_for_files(file_paths)
        
        # Refresh the list
        tag_list = tags if tags else None
        files = scan_output_files(
            model_filter=model if model != "All" else None,
            tag_filter=tag_list,
            limit=int(limit),
            include_deleted=False,
        )
        choices = format_file_list(files)
        new_paths = [f["path"] for f in files]
        
        return f"*Cleared tags from {cleared} file(s)*", gr.update(choices=choices, value=None), f"*{len(files)} file(s)*", new_paths, gr.update(visible=False)
    
    def add_custom_tag(custom_input, current_custom, file_path, model, 
                       approved, favorite, best_take, review, bad, delete):
        """Add a custom tag and save."""
        if not custom_input or not custom_input.strip():
            return current_custom, "", True if current_custom else False, f"*Custom tags: {', '.join(current_custom)}*" if current_custom else "*Custom tags: —*", "*No tag entered*"
        
        if not file_path or not os.path.exists(file_path):
            return current_custom, "", True if current_custom else False, f"*Custom tags: {', '.join(current_custom)}*" if current_custom else "*Custom tags: —*", "*No file selected*"
        
        tag = custom_input.strip().lower().replace(" ", "-")
        new_custom = list(current_custom)
        if tag not in new_custom:
            new_custom.append(tag)
            tag_manager.add_custom_tag(tag)  # Add to autocomplete list
        
        # Build full tag list and save (don't include delete)
        tags = []
        if approved:
            tags.append("approved")
        if favorite:
            tags.append("favorite")
        if best_take:
            tags.append("best-take")
        if review:
            tags.append("review")
        if bad:
            tags.append("bad")
        tags.extend(new_custom)
        
        tag_manager.set_tags(file_path, tags, model=model)
        
        custom_display = f"*Custom tags: {', '.join(new_custom)}*" if new_custom else "*Custom tags: —*"
        return new_custom, "", True, custom_display, f"*Added: {tag}*"
    
    def clear_tags(file_path: str):
        """Clear all tags from the current file."""
        if not file_path:
            return (False, False, False, False, False, False, False, [], "*Custom tags: —*", "*No file selected*")
        
        tag_manager.clear_tags(file_path)
        return (False, False, False, False, False, False, False, [], "*Custom tags: —*", "*Tags cleared*")
    
    # Wire up events
    refresh_btn.click(
        fn=refresh_file_list,
        inputs=[model_filter, tag_filter, limit_slider],
        outputs=[file_list, file_count, displayed_file_paths],
    )
    
    # Also refresh on filter changes
    model_filter.change(
        fn=refresh_file_list,
        inputs=[model_filter, tag_filter, limit_slider],
        outputs=[file_list, file_count, displayed_file_paths],
    )
    
    tag_filter.change(
        fn=refresh_file_list,
        inputs=[model_filter, tag_filter, limit_slider],
        outputs=[file_list, file_count, displayed_file_paths],
    )
    
    limit_slider.release(
        fn=refresh_file_list,
        inputs=[model_filter, tag_filter, limit_slider],
        outputs=[file_list, file_count, displayed_file_paths],
    )
    
    # File selection
    file_list.change(
        fn=select_file,
        inputs=[file_list],
        outputs=[
            video_preview, model_preview, no_preview,
            file_info,
            tag_approved, tag_favorite, tag_best_take, tag_review, tag_bad, tag_delete, tag_custom_checkbox,
            current_custom_tags, custom_tags_display,
            tag_status,
            current_file_path, current_file_model,
            delete_confirm_row,
        ],
    )
    
    # Auto-save on tag checkbox change (shows confirm for delete)
    for checkbox in [tag_approved, tag_favorite, tag_best_take, tag_review, tag_bad, tag_delete, tag_custom_checkbox]:
        checkbox.change(
            fn=auto_save_tags,
            inputs=[tag_approved, tag_favorite, tag_best_take, tag_review, tag_bad, tag_delete, tag_custom_checkbox,
                    current_custom_tags, current_file_path, current_file_model],
            outputs=[tag_status, delete_confirm_row],
        )
    
    # Auto-check custom checkbox when typing
    custom_tag_input.change(
        fn=on_custom_input_change,
        inputs=[custom_tag_input, tag_custom_checkbox],
        outputs=[tag_custom_checkbox],
    )
    
    # Delete confirmation
    delete_yes_btn.click(
        fn=confirm_delete,
        inputs=[current_file_path, current_file_model, displayed_file_paths, 
                model_filter, tag_filter, limit_slider],
        outputs=[tag_status, delete_confirm_row, file_list, file_count, displayed_file_paths,
                 tag_approved, tag_favorite, tag_best_take, tag_review, tag_bad, tag_delete, tag_custom_checkbox,
                 current_custom_tags, custom_tags_display,
                 video_preview, model_preview, no_preview],
    )
    
    delete_no_btn.click(
        fn=cancel_delete,
        inputs=[],
        outputs=[delete_confirm_row, tag_delete, tag_status],
    )
    
    # Clear ALL confirmation
    clear_all_tags_btn.click(
        fn=show_clear_all_confirm,
        inputs=[],
        outputs=[clear_all_confirm_row],
    )
    
    clear_all_yes_btn.click(
        fn=confirm_clear_all,
        inputs=[displayed_file_paths, model_filter, tag_filter, limit_slider],
        outputs=[tag_status, file_list, file_count, displayed_file_paths, clear_all_confirm_row],
    )
    
    clear_all_no_btn.click(
        fn=cancel_clear_all,
        inputs=[],
        outputs=[clear_all_confirm_row, tag_status],
    )
    
    # Custom tag - save on Enter (submit)
    custom_tag_input.submit(
        fn=add_custom_tag,
        inputs=[custom_tag_input, current_custom_tags, current_file_path, current_file_model,
                tag_approved, tag_favorite, tag_best_take, tag_review, tag_bad, tag_delete],
        outputs=[current_custom_tags, custom_tag_input, tag_custom_checkbox, custom_tags_display, tag_status],
    )
    
    # Clear tags for current file
    clear_tags_btn.click(
        fn=clear_tags,
        inputs=[current_file_path],
        outputs=[tag_approved, tag_favorite, tag_best_take, tag_review, tag_bad, tag_delete, tag_custom_checkbox,
                 current_custom_tags, custom_tags_display, tag_status],
    )
    
    # Return components for external access
    return {
        "model_filter": model_filter,
        "tag_filter": tag_filter,
        "limit_slider": limit_slider,
        "refresh_btn": refresh_btn,
        "file_list": file_list,
        "file_count": file_count,
        "video_preview": video_preview,
        "model_preview": model_preview,
        "file_info": file_info,
        "tag_approved": tag_approved,
        "tag_favorite": tag_favorite,
        "tag_best_take": tag_best_take,
        "tag_review": tag_review,
        "tag_bad": tag_bad,
        "tag_delete": tag_delete,
        "tag_custom_checkbox": tag_custom_checkbox,
        "custom_tag_input": custom_tag_input,
        "custom_tags_display": custom_tags_display,
        "clear_tags_btn": clear_tags_btn,
        "clear_all_tags_btn": clear_all_tags_btn,
        "tag_status": tag_status,
        "current_file_path": current_file_path,
        "current_file_model": current_file_model,
        "current_custom_tags": current_custom_tags,
        "displayed_file_paths": displayed_file_paths,
        "delete_confirm_row": delete_confirm_row,
        "delete_yes_btn": delete_yes_btn,
        "delete_no_btn": delete_no_btn,
        "clear_all_confirm_row": clear_all_confirm_row,
        "clear_all_yes_btn": clear_all_yes_btn,
        "clear_all_no_btn": clear_all_no_btn,
    }
