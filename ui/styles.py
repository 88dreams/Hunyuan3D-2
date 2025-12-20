"""
CSS Styles for 3D Generation Studio

This module contains all custom CSS styling for the Gradio interface.
"""

CUSTOM_CSS = """
/* 3D Generation Studio Custom Styles - Gradio 6.x */

/* Main container styling */
.gradio-container {
    max-width: 1400px !important;
}

/* ============================================
   TAB STYLING - Matching Gradio's Input/Group styling (orange accent)
   ============================================ */

/* Target tab buttons by multiple possible selectors */
button[role="tab"],
.tab-nav button,
.tabs button,
[data-testid="tab-button"],
div[class*="tab"] > button {
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    margin: 0 2px !important;
    border-radius: 8px 8px 0 0 !important;
    background: linear-gradient(180deg, #4a4a4a 0%, #3a3a3a 100%) !important;
    border: 1px solid #555 !important;
    border-bottom: none !important;
    color: #b0b0b0 !important;
    transition: all 0.15s ease !important;
}

/* Hover state */
button[role="tab"]:hover,
.tab-nav button:hover,
.tabs button:hover,
[data-testid="tab-button"]:hover,
div[class*="tab"] > button:hover {
    background: linear-gradient(180deg, #5a5a5a 0%, #4a4a4a 100%) !important;
    color: #d0d0d0 !important;
}

/* Selected/active tab - orange accent like Input headers */
button[role="tab"][aria-selected="true"],
button[role="tab"].selected,
.tab-nav button.selected,
.tab-nav button[aria-selected="true"],
.tabs button.selected,
.tabs button[aria-selected="true"],
[data-testid="tab-button"].selected,
[data-testid="tab-button"][aria-selected="true"],
div[class*="tab"] > button.selected,
div[class*="tab"] > button[aria-selected="true"] {
    font-weight: 700 !important;
    background: linear-gradient(180deg, #2d2d2d 0%, #1f1f1f 100%) !important;
    border: 1px solid #ff7c00 !important;
    border-bottom: 1px solid #1f1f1f !important;
    color: #ff7c00 !important;
    z-index: 1 !important;
}

/* Tab container/navigation area */
[role="tablist"],
.tab-nav,
div[class*="tab-nav"] {
    border-bottom: 1px solid #555 !important;
    padding-bottom: 0 !important;
    background: transparent !important;
}

/* ============================================
   OTHER STYLES
   ============================================ */

/* Scrollable textboxes for logs and progress */
textarea {
    overflow-y: auto !important;
    resize: vertical !important;
}

/* Ensure textbox containers allow scrolling */
.gradio-textbox textarea,
[data-testid="textbox"] textarea {
    overflow-y: auto !important;
    max-height: 400px !important;
}

/* Accordion content should be scrollable */
.accordion-content,
[class*="accordion"] > div {
    overflow-y: auto !important;
    max-height: 500px !important;
}

/* Accordion styling for logs */
.logs-accordion {
    margin-top: 10px;
}

/* Job queue table */
.job-queue-table {
    font-size: 0.9em;
}

/* Status indicators */
.status-ready { color: #22c55e; }
.status-running { color: #f59e0b; }
.status-error { color: #ef4444; }

/* Generate button */
.generate-btn {
    min-height: 50px;
    font-size: 1.1em;
    font-weight: 600;
}

/* Model viewer container */
.model-viewer-container {
    min-height: 300px;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
}

/* Expandable groups */
.gr-group {
    overflow: visible !important;
}
"""

