"""
Tab Components for 3D Generation Studio

This module provides tab UI components for each model:
- Hunyuan3D: Image → GLB mesh (Tencent)
- GEN3C: Image → Video (NVIDIA)
- Lyra: Image/Video → 3DGS/4DGS (NVIDIA)
- SHARP: Image → 3DGS PLY (Apple)
- TRELLIS.2: Image → GLB with PBR (Microsoft)
"""

from ui.tabs.hunyuan_tab import create_hunyuan_tab
from ui.tabs.gen3c_tab import create_gen3c_tab
from ui.tabs.sharp_tab import create_sharp_tab
from ui.tabs.lyra_tab import create_lyra_tab
from ui.tabs.trellis_tab import create_trellis_tab

__all__ = [
    "create_hunyuan_tab",
    "create_gen3c_tab",
    "create_sharp_tab",
    "create_lyra_tab",
    "create_trellis_tab",
]

