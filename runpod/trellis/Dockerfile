# TRELLIS.2 RunPod Serverless Docker Image
# Extends the unified image with TRELLIS.2-specific handler
#
# This image uses the existing unified Docker as base (which already has TRELLIS.2 installed)
# and replaces the handler with a TRELLIS.2-specific one.
#
# v2: Rebuild to fix initialization

FROM 88dreams/gen3c-runpod:v22

# Copy TRELLIS.2-specific handler and scripts
COPY handler_trellis.py /workspace/handler.py
COPY trellis_inference.py /workspace/trellis_inference.py
COPY start_trellis.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

# Environment for TRELLIS.2
ENV PYTHONPATH="/workspace/TRELLIS2:/workspace/GEN3C:${PYTHONPATH}"
ENV TRELLIS_CHECKPOINT_DIR="/runpod-volume/checkpoints/trellis"

# Output directory
RUN mkdir -p /runpod-volume/outputs/trellis

# Default command
CMD ["/workspace/start.sh"]
