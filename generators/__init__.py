"""
Generators Module for 3D Generation Studio

This module provides generation functions for each model:
- Hunyuan3D: Image → GLB mesh (Tencent)
- GEN3C: Image → Video (NVIDIA)
- Lyra: Image/Video → 3DGS/4DGS (NVIDIA)
- SHARP: Image → 3DGS PLY (Apple)
- TRELLIS.2: Image → GLB with PBR (Microsoft)
"""

from generators.hunyuan import (
    run_hunyuan,
    ensure_pipeline,
    get_pipeline,
    HUNYUAN_DEFAULT_OUTPUT_DIR,
    CACHE_DIR,
)

from generators.gen3c import (
    run_gen3c_runpod,
    run_gen3c_serverless,
    run_gen3c_local,
    check_runpod_status,
    check_serverless_status,
    cancel_serverless_job,
    is_runpod_available,
    GEN3C_DEFAULT_CHECKPOINT,
    GEN3C_DEFAULT_OUTPUT_DIR,
)

from generators.sharp import (
    run_sharp_local,
    run_sharp_runpod,
    check_sharp_installation,
    is_sharp_available,
    is_runpod_available as is_sharp_runpod_available,
    SHARP_DEFAULT_OUTPUT_DIR,
)

from generators.lyra import (
    run_lyra_runpod,
    check_lyra_status,
    check_lyra_installation,
    LYRA_DEFAULT_OUTPUT_DIR,
)

from generators.trellis import (
    run_trellis_runpod,
    check_trellis_status,
    check_trellis_installation,
    TRELLIS_DEFAULT_OUTPUT_DIR,
)

__all__ = [
    # Hunyuan
    "run_hunyuan",
    "ensure_pipeline",
    "get_pipeline",
    "HUNYUAN_DEFAULT_OUTPUT_DIR",
    "CACHE_DIR",
    # GEN3C
    "run_gen3c_runpod",
    "run_gen3c_serverless",
    "run_gen3c_local",
    "check_runpod_status",
    "check_serverless_status",
    "cancel_serverless_job",
    "is_runpod_available",
    "GEN3C_DEFAULT_CHECKPOINT",
    "GEN3C_DEFAULT_OUTPUT_DIR",
    # SHARP
    "run_sharp_local",
    "run_sharp_runpod",
    "check_sharp_installation",
    "is_sharp_available",
    "is_sharp_runpod_available",
    "SHARP_DEFAULT_OUTPUT_DIR",
    # Lyra
    "run_lyra_runpod",
    "check_lyra_status",
    "check_lyra_installation",
    "LYRA_DEFAULT_OUTPUT_DIR",
    # TRELLIS.2
    "run_trellis_runpod",
    "check_trellis_status",
    "check_trellis_installation",
    "TRELLIS_DEFAULT_OUTPUT_DIR",
]

