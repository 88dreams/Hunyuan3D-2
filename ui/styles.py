"""
CSS Styles for 3D Generation Studio

Modern sidebar-based interface with warm cream/yellow theme.
"""

CUSTOM_CSS = """
/* ============================================
   3D Generation Studio - Sidebar UI Theme
   Warm cream/yellow palette
   ============================================ */

/* Root Variables - Dark theme with orange accents */
:root {
    --bg-primary: #0f0f1a;
    --bg-secondary: #1a1a2e;
    --bg-tertiary: #16213e;
    --border-color: #2a2a4a;
    --text-primary: #ffffff;
    --text-secondary: #b0b0b0;
    --text-muted: #888888;
    --accent-primary: #ff7c00;
    --accent-secondary: #ff5500;
    --accent-glow: rgba(255, 124, 0, 0.3);
    --success: #22c55e;
    --warning: #f59e0b;
    --error: #ef4444;
    --info: #3b82f6;
    
    /* Illustration colors */
    --stroke: #2a2a4a;
    --main: #1a1a2e;
    --highlight: #ff7c00;
    --secondary: #1a1a2e;
    --tertiary: #16213e;
    
    /* Button specific */
    --button-bg: #ff7c00;
    --button-text: #ffffff;
    
    /* Card/Group backgrounds */
    --card-bg: #1a1a2e;
    --card-border: #2a2a4a;
    
    /* Base font size increase */
    font-size: 16px;
}

/* Global Container */
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    background: var(--bg-primary) !important;
}

/* ============================================
   HIDDEN TABS - Hide tab bar but show content
   ============================================ */

/* Hide the tab navigation bar completely */
.hidden-tabs > .tab-nav,
.hidden-tabs > div:first-child:has(button) {
    display: none !important;
    height: 0 !important;
    overflow: hidden !important;
    visibility: hidden !important;
    position: absolute !important;
    pointer-events: none !important;
}

/* Ensure tab content takes full space */
.hidden-tabs > .tabitem,
.hidden-tabs > div[class*="tabitem"] {
    padding: 0 !important;
    margin: 0 !important;
}

/* Main Layout - Sidebar + Content */
.main-layout {
    display: flex !important;
    min-height: 100vh !important;
    background: var(--bg-primary) !important;
}

/* ============================================
   SIDEBAR STYLES
   ============================================ */

.sidebar {
    width: 280px !important;
    min-width: 280px !important;
    max-width: 280px !important;
    background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
    border-right: 1px solid var(--border-color) !important;
    display: block !important;
    height: auto !important;
    min-height: 100vh !important;
    position: relative !important;
    overflow: visible !important;
    padding: 0 4px !important;
    padding-bottom: 20px !important;
}

/* Ensure all sidebar content flows naturally */
.sidebar > * {
    position: relative !important;
    width: 100% !important;
    max-width: 100% !important;
}

/* Sidebar Input Section - compact */
.sidebar .gr-group {
    padding: 8px !important;
    margin: 4px 0 !important;
}

.sidebar .gr-image {
    min-height: 140px !important;
}

.sidebar .gr-image .image-container {
    min-height: 120px !important;
}

.sidebar .upload-container {
    min-height: 120px !important;
    padding: 12px !important;
}

/* Scale + Image Info combined box */
.scale-info-box {
    padding: 8px 12px !important;
    margin: 4px 0 !important;
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 6px !important;
}

.scale-info-box .gr-slider {
    margin-bottom: 4px !important;
}

.image-info-text textarea,
.image-info-text input {
    font-size: 0.85em !important;
    color: var(--text-muted) !important;
    padding: 4px 8px !important;
    min-height: 24px !important;
    background: transparent !important;
    border: none !important;
}

/* PLY Input Row with inline refresh button */
.ply-input-row {
    align-items: center !important;
    gap: 8px !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

.ply-input-row > div {
    background: transparent !important;
}

.ply-dropdown {
    flex: 1 !important;
}

.refresh-btn-inline {
    min-width: 44px !important;
    max-width: 44px !important;
    height: 42px !important;
    padding: 0 !important;
    margin: 0 !important;
    margin-top: 24px !important;  /* Align with dropdown input (accounts for label height) */
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.refresh-btn-inline button {
    min-width: 44px !important;
    max-width: 44px !important;
    height: 42px !important;
    padding: 0 !important;
    font-size: 1.4em !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 6px !important;
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-color) !important;
}

.refresh-btn-inline:hover button,
.refresh-btn-inline button:hover {
    background: var(--accent-primary) !important;
    border-color: var(--accent-primary) !important;
}

/* Button grid rows */
.sidebar .button-grid {
    gap: 4px !important;
    padding: 0 4px !important;
    margin-bottom: 4px !important;
}

.sidebar .button-grid .gr-button,
.sidebar .button-grid button {
    min-width: 0 !important;
    padding: 10px 6px !important;
    font-size: 1.15em !important;
    text-transform: uppercase !important;
}

.sidebar-header {
    padding: 12px 12px !important;
    border-bottom: 1px solid var(--border-color) !important;
    background: linear-gradient(180deg, #1f1f3a 0%, var(--bg-secondary) 100%) !important;
}

.sidebar-header h2 {
    margin: 0 !important;
    font-size: 1.3em !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

.sidebar-category {
    padding: 8px 12px 6px 12px !important;
    font-size: 1.3em !important;
    font-weight: 700 !important;
    color: var(--accent-primary) !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    border-bottom: 2px solid var(--accent-primary) !important;
    margin: 6px 8px 8px 8px !important;
    background: linear-gradient(90deg, rgba(255, 124, 0, 0.15) 0%, transparent 100%) !important;
}

/* Add spacing and divider ABOVE the parent container of category labels */
.sidebar .html-container:has(.sidebar-category) {
    margin-top: 12px !important;
    padding-top: 12px !important;
    border-top: 1px solid var(--border-color) !important;
}

/* First category (INPUT) container doesn't need top border/margin */
.sidebar .html-container:has(.sidebar-category):first-of-type {
    border-top: none !important;
    margin-top: 0 !important;
    padding-top: 0 !important;
}

.sidebar-nav button,
.sidebar button {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 12px 8px !important;
    border-radius: 6px !important;
    border: 1px solid var(--border-color) !important;
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    font-size: 1.05em !important;
    font-weight: 600 !important;
    text-align: center !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}

/* Button grid for CREATE section */
.sidebar .button-grid {
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    gap: 6px !important;
    padding: 0 8px !important;
    margin-bottom: 4px !important;
}

.sidebar .button-grid button {
    width: 100% !important;
    padding: 12px 6px !important;
    font-size: 1.15em !important;
    text-transform: uppercase !important;
}

.sidebar-nav button:hover,
.sidebar button:hover {
    background: rgba(255, 124, 0, 0.15) !important;
    border-color: var(--accent-primary) !important;
    color: var(--text-primary) !important;
}

.sidebar-nav button.active,
.sidebar button.selected,
.sidebar-nav button[aria-pressed="true"],
.sidebar-nav.active button,
.sidebar .nav-active,
.sidebar button[data-active="true"] {
    background: var(--accent-primary) !important;
    border-color: var(--accent-primary) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Ensure focus doesn't override active state */
.sidebar button[data-active="true"]:focus {
    background: var(--accent-primary) !important;
    border-color: var(--accent-primary) !important;
    color: #ffffff !important;
}

.sidebar-footer {
    margin-top: auto !important;
    padding: 8px !important;
    border-top: 1px solid var(--border-color) !important;
    font-size: 0.75em !important;
    color: var(--text-muted) !important;
    text-align: center !important;
}

/* ============================================
   MAIN CONTENT AREA
   ============================================ */

.main-content {
    flex: 1 !important;
    padding: 24px 32px !important;
    overflow-y: auto !important;
    background: var(--bg-primary) !important;
    min-height: 100vh !important;
}

.page-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
}

/* Page Header */
.page-header {
    margin-bottom: 28px !important;
    padding-bottom: 20px !important;
    border-bottom: 1px solid var(--border-color) !important;
}

.page-header h1 {
    margin: 0 !important;
    font-size: 1.8em !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}

.page-header p {
    margin: 10px 0 0 0 !important;
    color: var(--text-secondary) !important;
    font-size: 1.05em !important;
    line-height: 1.5 !important;
}

/* ============================================
   CARD COMPONENTS
   ============================================ */

.card, .gr-group, .gr-box {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin-bottom: 16px !important;
}

.card-header {
    display: flex !important;
    align-items: center !important;
    margin-bottom: 16px !important;
    padding-bottom: 12px !important;
    border-bottom: 1px solid var(--border-color) !important;
}

.card-header h3 {
    margin: 0 !important;
    font-size: 1.2em !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

.card-header .icon {
    font-size: 1.4em !important;
    margin-right: 10px !important;
}

/* ============================================
   SETTINGS PAGE - 2-COLUMN LAYOUT
   ============================================ */

.settings-row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 16px !important;
    margin-bottom: 16px !important;
}

.settings-row > .gr-column {
    flex: 1 1 50% !important;
    max-width: 50% !important;
}

.settings-card {
    height: 100% !important;
}

.settings-card p {
    margin: 0 0 12px 0 !important;
    font-size: 1.1em !important;
}

/* ============================================
   INPUT COMPONENTS
   ============================================ */

/* Text inputs */
input[type="text"],
input[type="password"],
input[type="number"],
textarea,
.gr-textbox input,
.gr-textbox textarea {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    padding: 10px 14px !important;
    font-size: 1.0em !important;
    transition: border-color 0.2s ease !important;
}

input[type="text"]:focus,
input[type="password"]:focus,
input[type="number"]:focus,
textarea:focus {
    border-color: var(--accent-primary) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}

/* Labels */
label, .gr-input-label {
    color: var(--text-secondary) !important;
    font-size: 0.95em !important;
    font-weight: 500 !important;
    margin-bottom: 6px !important;
}

/* Dropdowns */
.gr-dropdown, select {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* Sliders */
.gr-slider input[type="range"] {
    accent-color: var(--accent-primary) !important;
}

/* Checkboxes and Radio buttons */
input[type="checkbox"],
input[type="radio"] {
    accent-color: var(--accent-primary) !important;
}

/* ============================================
   BUTTON STYLES
   ============================================ */

/* Primary Button */
.gr-button-primary,
button.primary,
button[variant="primary"] {
    background: var(--button-bg) !important;
    border: 2px solid var(--stroke) !important;
    color: var(--button-text) !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    font-size: 1.05em !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

.gr-button-primary:hover,
button.primary:hover,
button[variant="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px var(--accent-glow) !important;
    background: var(--accent-secondary) !important;
}

/* Secondary Button */
.gr-button-secondary,
button.secondary,
button[variant="secondary"] {
    background: transparent !important;
    border: 1px solid var(--accent-primary) !important;
    color: var(--accent-primary) !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.gr-button-secondary:hover,
button.secondary:hover,
button[variant="secondary"]:hover {
    background: rgba(255, 124, 0, 0.1) !important;
}

/* Stop/Cancel Button */
.gr-button-stop,
button.stop,
button[variant="stop"] {
    background: transparent !important;
    border: 1px solid var(--error) !important;
    color: var(--error) !important;
}

.gr-button-stop:hover,
button.stop:hover,
button[variant="stop"]:hover {
    background: rgba(239, 68, 68, 0.1) !important;
}

/* Small Buttons */
button.sm, button[size="sm"] {
    padding: 6px 12px !important;
    font-size: 0.9em !important;
}

/* Large Buttons */
button.lg, button[size="lg"] {
    padding: 14px 28px !important;
    font-size: 1.15em !important;
}

/* ============================================
   ACCORDION STYLES
   ============================================ */

.gr-accordion {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
}

.gr-accordion > button,
.gr-accordion-header {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 1.0em !important;
    padding: 14px 16px !important;
    border-radius: 12px 12px 0 0 !important;
}

.gr-accordion > button:hover,
.gr-accordion-header:hover {
    background: rgba(255, 124, 0, 0.05) !important;
}

.gr-accordion-content {
    padding: 16px !important;
    border-top: 1px solid var(--border-color) !important;
}

/* ============================================
   TAB STYLES (for nested tabs within pages)
   ============================================ */

.gr-tabs {
    background: transparent !important;
}

button[role="tab"],
.tab-nav button {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-bottom: none !important;
    color: var(--text-secondary) !important;
    padding: 10px 20px !important;
    border-radius: 8px 8px 0 0 !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
}

button[role="tab"]:hover,
.tab-nav button:hover {
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
}

button[role="tab"][aria-selected="true"],
.tab-nav button.selected {
    background: var(--bg-primary) !important;
    border-color: var(--accent-primary) !important;
    color: var(--accent-primary) !important;
    font-weight: 600 !important;
}

.gr-tab-content {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 20px !important;
}

/* ============================================
   STATUS BADGES
   ============================================ */

.status-badge {
    display: inline-flex !important;
    align-items: center !important;
    padding: 4px 12px !important;
    border-radius: 16px !important;
    font-size: 0.8em !important;
    font-weight: 500 !important;
}

.status-success {
    background: rgba(34, 197, 94, 0.15) !important;
    color: var(--success) !important;
}

.status-warning {
    background: rgba(245, 158, 11, 0.15) !important;
    color: var(--warning) !important;
}

.status-error {
    background: rgba(239, 68, 68, 0.15) !important;
    color: var(--error) !important;
}

.status-info {
    background: rgba(59, 130, 246, 0.15) !important;
    color: var(--info) !important;
}

/* ============================================
   IMAGE/VIDEO UPLOAD AREA
   ============================================ */

.gr-image, .gr-video {
    background: var(--bg-secondary) !important;
    border: 2px dashed var(--border-color) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s ease !important;
}

.gr-image:hover, .gr-video:hover {
    border-color: var(--accent-primary) !important;
}

.gr-image img, .gr-video video {
    border-radius: 8px !important;
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* Ensure uploaded image stays visible - override all Gradio defaults */
.sidebar .gr-image img,
.sidebar .gr-image .image-container img,
.sidebar .gr-image [data-testid="image"] img,
.sidebar .image-frame img,
.sidebar [data-testid="image"] img {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    max-width: 100% !important;
    height: auto !important;
    position: relative !important;
    z-index: 10 !important;
}

/* Fix for Gradio image preview and all possible containers */
.sidebar .gr-image .image-preview,
.sidebar .gr-image .uploaded-image,
.sidebar .gr-image .preview-image,
.sidebar .image-frame,
.sidebar [data-testid="image"],
.sidebar .gr-image > div,
.sidebar .gr-image > div > div {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: relative !important;
}

/* Ensure the image container doesn't collapse */
.sidebar .gr-image,
.sidebar .gr-image > div {
    min-height: 100px !important;
    overflow: visible !important;
}

/* Override any hover-only visibility */
.sidebar .gr-image img,
.sidebar .gr-image:hover img,
.sidebar .gr-image:not(:hover) img {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* ============================================
   PROGRESS & LOGS
   ============================================ */

.progress-container {
    background: var(--bg-secondary) !important;
    border-radius: 8px !important;
    padding: 16px !important;
}

.progress-bar {
    height: 8px !important;
    background: var(--bg-primary) !important;
    border-radius: 4px !important;
    overflow: hidden !important;
}

.progress-bar-fill {
    height: 100% !important;
    background: linear-gradient(90deg, var(--accent-primary) 0%, var(--accent-secondary) 100%) !important;
    border-radius: 4px !important;
    transition: width 0.3s ease !important;
}

/* Log textbox */
.logs-box textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.85em !important;
    line-height: 1.5 !important;
    background: var(--bg-primary) !important;
    color: var(--text-secondary) !important;
}

/* ============================================
   MODEL SELECTION CARDS
   ============================================ */

.model-card {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    text-align: center !important;
}

.model-card:hover {
    border-color: var(--accent-primary) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3) !important;
}

.model-card.selected {
    border-color: var(--accent-primary) !important;
    background: rgba(255, 124, 0, 0.1) !important;
}

.model-card-icon {
    font-size: 2.5em !important;
    margin-bottom: 12px !important;
}

.model-card-title {
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    font-size: 1.1em !important;
    margin-bottom: 6px !important;
}

.model-card-time {
    font-size: 0.85em !important;
    color: var(--text-muted) !important;
}

/* ============================================
   DATAFRAME / TABLE STYLES
   ============================================ */

.gr-dataframe {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
}

.gr-dataframe th {
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    padding: 12px !important;
}

.gr-dataframe td {
    color: var(--text-secondary) !important;
    padding: 10px 12px !important;
    border-bottom: 1px solid var(--border-color) !important;
}

.gr-dataframe tr:hover td {
    background: rgba(255, 124, 0, 0.05) !important;
}

/* ============================================
   RESPONSIVE ADJUSTMENTS
   ============================================ */

@media (max-width: 1200px) {
    .sidebar {
        width: 180px !important;
        min-width: 180px !important;
    }
    
    .main-content {
        padding: 20px 24px !important;
    }
}

@media (max-width: 900px) {
    .sidebar {
        width: 60px !important;
        min-width: 60px !important;
    }
    
    .sidebar-header h2,
    .sidebar-header p,
    .sidebar-category,
    .sidebar-nav button span {
        display: none !important;
    }
    
    .sidebar-nav button {
        justify-content: center !important;
        padding: 14px !important;
    }
}

/* ============================================
   UTILITY CLASSES
   ============================================ */

.hidden {
    display: none !important;
}

.flex-row {
    display: flex !important;
    flex-direction: row !important;
    gap: 16px !important;
}

.flex-col {
    display: flex !important;
    flex-direction: column !important;
    gap: 12px !important;
}

.text-center {
    text-align: center !important;
}

.text-muted {
    color: var(--text-muted) !important;
}

.mt-4 {
    margin-top: 16px !important;
}

.mb-4 {
    margin-bottom: 16px !important;
}

.p-4 {
    padding: 16px !important;
}

/* ============================================
   HIDDEN ELEMENTS
   ============================================ */

/* Hide video preview when not visible */
.gr-video[style*="display: none"],
.gr-video.hidden {
    display: none !important;
    height: 0 !important;
    position: absolute !important;
}

/* Reduce gap between output section and page content */
.main-content > .gr-group:first-child {
    margin-bottom: 8px !important;
}

/* ============================================
   SCROLLBAR STYLING
   ============================================ */

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
}
"""
