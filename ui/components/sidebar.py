"""
Sidebar Navigation Component

Provides a vertical sidebar navigation for the 3D Generation Studio.
Groups models by workflow stage: Input → Create → Refine → Monitor
"""

import gradio as gr
from typing import Dict, List, Callable, Any


# Sidebar navigation items organized by category
SIDEBAR_ITEMS = {
    "input": {
        "label": "📥 Input",
        "items": [
            {"id": "input_image", "icon": "🖼️", "label": "Image", "description": "Upload source image"},
            {"id": "input_video", "icon": "🎥", "label": "Video", "description": "Upload source video"},
        ]
    },
    "create": {
        "label": "🎬 Create",
        "items": [
            {"id": "sharp", "icon": "⚡", "label": "SHARP", "description": "Fast 3DGS (~60s)"},
            {"id": "gen3c", "icon": "🎬", "label": "GEN3C", "description": "Video generation (~10min)"},
            {"id": "lyra", "icon": "🌀", "label": "Lyra", "description": "3DGS from video (~15min)"},
            {"id": "trellis", "icon": "🔷", "label": "TRELLIS.2", "description": "High-quality 3D"},
            {"id": "hunyuan", "icon": "🏔️", "label": "Hunyuan3D", "description": "Image to GLB mesh"},
        ]
    },
    "refine": {
        "label": "🔧 Refine",
        "items": [
            {"id": "mesh_extraction", "icon": "🔷", "label": "Mesh Extract", "description": "3DGS → GLB mesh"},
            {"id": "cleanup", "icon": "✨", "label": "Cleanup", "description": "Clean & optimize mesh"},
            {"id": "convert", "icon": "🔄", "label": "Convert", "description": "Format conversion"},
        ]
    },
    "monitor": {
        "label": "📊 Monitor",
        "items": [
            {"id": "jobs", "icon": "📋", "label": "Jobs", "description": "Job queue & history"},
            {"id": "system", "icon": "💻", "label": "System", "description": "GPU & memory stats"},
            {"id": "settings", "icon": "⚙️", "label": "Settings", "description": "Credentials & config"},
        ]
    },
}


def get_all_page_ids() -> List[str]:
    """Get list of all page IDs for navigation."""
    ids = []
    for category in SIDEBAR_ITEMS.values():
        for item in category["items"]:
            ids.append(item["id"])
    return ids


def create_sidebar_css() -> str:
    """Generate CSS for the sidebar component."""
    return """
    /* Sidebar Container */
    .sidebar-container {
        display: flex;
        flex-direction: column;
        height: 100%;
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #2a2a4a;
        padding: 0;
        min-width: 200px;
        max-width: 220px;
    }
    
    /* Sidebar Header */
    .sidebar-header {
        padding: 16px 12px;
        border-bottom: 1px solid #2a2a4a;
        background: linear-gradient(180deg, #1f1f3a 0%, #1a1a2e 100%);
    }
    
    .sidebar-header h2 {
        margin: 0;
        font-size: 1.1em;
        font-weight: 700;
        color: #ff7c00;
        letter-spacing: 0.5px;
    }
    
    .sidebar-header p {
        margin: 4px 0 0 0;
        font-size: 0.75em;
        color: #888;
    }
    
    /* Category Headers */
    .sidebar-category {
        padding: 12px 12px 6px 12px;
        font-size: 0.7em;
        font-weight: 600;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Navigation Items */
    .sidebar-item {
        display: flex;
        align-items: center;
        padding: 10px 12px;
        margin: 2px 8px;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.15s ease;
        color: #b0b0b0;
        font-size: 0.9em;
    }
    
    .sidebar-item:hover {
        background: rgba(255, 124, 0, 0.1);
        color: #fff;
    }
    
    .sidebar-item.active {
        background: linear-gradient(90deg, rgba(255, 124, 0, 0.2) 0%, rgba(255, 124, 0, 0.05) 100%);
        color: #ff7c00;
        border-left: 3px solid #ff7c00;
        margin-left: 5px;
    }
    
    .sidebar-item-icon {
        font-size: 1.2em;
        margin-right: 10px;
        width: 24px;
        text-align: center;
    }
    
    .sidebar-item-label {
        flex: 1;
        font-weight: 500;
    }
    
    .sidebar-item-badge {
        background: #ff7c00;
        color: #fff;
        font-size: 0.7em;
        padding: 2px 6px;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Sidebar Footer */
    .sidebar-footer {
        margin-top: auto;
        padding: 12px;
        border-top: 1px solid #2a2a4a;
        font-size: 0.75em;
        color: #666;
    }
    
    /* Main Content Area */
    .main-content {
        flex: 1;
        padding: 20px;
        overflow-y: auto;
        background: #0f0f1a;
    }
    
    /* Page Container */
    .page-container {
        max-width: 1000px;
        margin: 0 auto;
    }
    
    /* Page Header */
    .page-header {
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 1px solid #2a2a4a;
    }
    
    .page-header h1 {
        margin: 0;
        font-size: 1.5em;
        font-weight: 700;
        color: #fff;
    }
    
    .page-header p {
        margin: 8px 0 0 0;
        color: #888;
        font-size: 0.9em;
    }
    
    /* Card Style for Settings Groups */
    .settings-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    
    .settings-card-header {
        display: flex;
        align-items: center;
        margin-bottom: 16px;
    }
    
    .settings-card-header h3 {
        margin: 0;
        font-size: 1.1em;
        font-weight: 600;
        color: #fff;
    }
    
    .settings-card-header .icon {
        font-size: 1.3em;
        margin-right: 10px;
    }
    
    /* Action Button Styles */
    .action-button-primary {
        background: linear-gradient(135deg, #ff7c00 0%, #ff5500 100%) !important;
        border: none !important;
        color: #fff !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        font-size: 1em !important;
        transition: all 0.2s ease !important;
    }
    
    .action-button-primary:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(255, 124, 0, 0.3);
    }
    
    .action-button-secondary {
        background: transparent !important;
        border: 1px solid #ff7c00 !important;
        color: #ff7c00 !important;
        font-weight: 500 !important;
        padding: 10px 20px !important;
        border-radius: 8px !important;
    }
    
    .action-button-secondary:hover {
        background: rgba(255, 124, 0, 0.1) !important;
    }
    
    /* Status Indicators */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 500;
    }
    
    .status-badge.success {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
    }
    
    .status-badge.warning {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
    }
    
    .status-badge.error {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
    }
    
    .status-badge.info {
        background: rgba(59, 130, 246, 0.2);
        color: #3b82f6;
    }
    
    /* Input Preview Card */
    .input-preview-card {
        background: #1a1a2e;
        border: 2px dashed #2a2a4a;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .input-preview-card:hover {
        border-color: #ff7c00;
        background: rgba(255, 124, 0, 0.05);
    }
    
    .input-preview-card.has-image {
        border-style: solid;
        border-color: #2a2a4a;
    }
    
    /* Progress Bar */
    .progress-container {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 16px;
        margin-top: 16px;
    }
    
    .progress-bar {
        height: 8px;
        background: #2a2a4a;
        border-radius: 4px;
        overflow: hidden;
    }
    
    .progress-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff7c00 0%, #ff5500 100%);
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Model Selection Cards */
    .model-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 16px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .model-card:hover {
        border-color: #ff7c00;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .model-card.selected {
        border-color: #ff7c00;
        background: rgba(255, 124, 0, 0.1);
    }
    
    .model-card-icon {
        font-size: 2em;
        margin-bottom: 8px;
    }
    
    .model-card-title {
        font-weight: 600;
        color: #fff;
        margin-bottom: 4px;
    }
    
    .model-card-time {
        font-size: 0.8em;
        color: #888;
    }
    """


def create_sidebar() -> Dict[str, Any]:
    """
    Create the sidebar navigation component.
    
    Returns:
        Dictionary containing sidebar components and state.
    """
    components = {}
    
    # Current page state
    components["current_page"] = gr.State(value="sharp")
    
    # Create navigation buttons for each item
    with gr.Column(elem_classes=["sidebar-container"], scale=0, min_width=200):
        # Header
        gr.HTML("""
            <div class="sidebar-header">
                <h2>🎨 3D Studio</h2>
                <p>Multi-model generation</p>
            </div>
        """)
        
        # Navigation items by category
        for category_id, category in SIDEBAR_ITEMS.items():
            gr.HTML(f'<div class="sidebar-category">{category["label"]}</div>')
            
            for item in category["items"]:
                item_id = item["id"]
                # Create a button for each nav item
                btn = gr.Button(
                    f'{item["icon"]} {item["label"]}',
                    elem_id=f"nav_{item_id}",
                    elem_classes=["sidebar-item"],
                    size="sm",
                )
                components[f"nav_{item_id}"] = btn
        
        # Footer with version info
        gr.HTML("""
            <div class="sidebar-footer">
                <div>v2.2 • Sidebar UI</div>
            </div>
        """)
    
    return components

