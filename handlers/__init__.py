"""
Generation handlers for the 3D Studio UI.

These functions handle the business logic between UI inputs and generator calls.
They manage:
- Image scaling and preprocessing
- Parameter validation
- Calling the appropriate generator (local or RunPod)
- Experiment logging
- Cleanup of temporary files
"""

from handlers.generation_handlers import (
    handle_sharp_generation,
    handle_gen3c_generation,
    handle_lyra_generation,
    handle_trellis_generation,
    handle_hunyuan_generation,
    handle_mesh_extraction,
    handle_mesh_analyze,
    handle_mesh_cleanup,
)

__all__ = [
    "handle_sharp_generation",
    "handle_gen3c_generation",
    "handle_lyra_generation",
    "handle_trellis_generation",
    "handle_hunyuan_generation",
    "handle_mesh_extraction",
    "handle_mesh_analyze",
    "handle_mesh_cleanup",
]

