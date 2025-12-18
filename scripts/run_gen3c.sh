#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Wrapper to run GEN3C single-image inference from the Hunyuan3D workspace.
#
# Multi-System Configuration (searidge cluster):
#   - searidge02 (CORE): Primary UI host, Ray head
#   - searidge01 (WORKER01): Ray worker
#   - searidge03 (WORKER02): Ray worker
#
# All paths default to NFS shared storage at /srv/searidge_share
#
# Usage:
#   scripts/run_gen3c.sh --input path/to/image.png --video-name my_test \
#       [--guidance 1.0] [--frames 121] [--checkpoint-dir /path/to/checkpoints] \
#       [--extra "--trajectory left --foreground_masking"]
# -----------------------------------------------------------------------------

# =============================================================================
# PATH CONFIGURATION - NFS Shared Storage
# =============================================================================
# These defaults point to the shared NFS mount accessible from all cluster nodes

# Shared storage root
SEARIDGE_SHARE="/srv/searidge_share"

# Project directories on shared storage
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN3C_DIR="${GEN3C_DIR:-${SEARIDGE_SHARE}/projects/GEN3C}"

# Conda environment (same name on all nodes)
# gen3c-rocm uses Python 3.12 (has apex-rocm properly built)
ENV_NAME="${ENV_NAME:-gen3c-rocm}"

# Checkpoint and output directories on shared storage
DEFAULT_CHECKPOINT_DIR="${SEARIDGE_SHARE}/checkpoints/gen3c"
DEFAULT_OUTPUT_DIR="${SEARIDGE_SHARE}/outputs/gen3c"

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
INPUT_IMAGE=""
VIDEO_NAME="gen3c_video"
GUIDANCE="1"
NUM_FRAMES=""
CHECKPOINT_DIR="$DEFAULT_CHECKPOINT_DIR"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
EXTRA_ARGS=()

usage() {
  cat <<USAGE
Usage: run_gen3c.sh --input <image> [options]

Multi-System GEN3C Launcher for searidge cluster
Paths default to NFS shared storage at /srv/searidge_share

Options:
  --input PATH           Source image (required)
  --video-name NAME      Base name for the generated video (default: gen3c_video)
  --guidance VALUE       Guidance scale (default: 1)
  --frames N             Number of frames (must follow 121*N-1 rule; optional)
  --checkpoint-dir PATH  GEN3C checkpoint directory
                         (default: ${DEFAULT_CHECKPOINT_DIR})
  --output-dir PATH      Directory where the resulting video will be copied
                         (default: ${DEFAULT_OUTPUT_DIR})
  --extra "ARGS"         Additional arguments passed verbatim to GEN3C
  -h, --help             Show this help

Environment Variables:
  GEN3C_DIR              GEN3C project directory (default: ${GEN3C_DIR})
  ENV_NAME               Conda environment name (default: gen3c-rocm310)
  HIP_VISIBLE_DEVICES    GPU device ID (default: 0)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT_IMAGE="$2"; shift 2;;
    --video-name)
      VIDEO_NAME="$2"; shift 2;;
    --guidance)
      GUIDANCE="$2"; shift 2;;
    --frames)
      NUM_FRAMES="$2"; shift 2;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"; shift 2;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    --extra)
      EXTRA_ARGS+=($2); shift 2;;
    -h|--help)
      usage; exit 0;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1;;
  esac
done

# =============================================================================
# VALIDATION
# =============================================================================
if [[ -z "$INPUT_IMAGE" ]]; then
  echo "ERROR: --input is required." >&2
  usage
  exit 1
fi

if [[ ! -d "$GEN3C_DIR" ]]; then
  echo "ERROR: GEN3C directory not found at $GEN3C_DIR" >&2
  echo "       Expected on NFS share. Check mount: ls ${SEARIDGE_SHARE}/projects/" >&2
  exit 1
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "ERROR: checkpoint directory not found at $CHECKPOINT_DIR" >&2
  echo "       Expected on NFS share. Check: ls ${SEARIDGE_SHARE}/checkpoints/" >&2
  exit 1
fi

if [[ ! -f "$INPUT_IMAGE" ]]; then
  echo "ERROR: input image not found: $INPUT_IMAGE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# ROCm / GPU CONFIGURATION
# =============================================================================
# Environment variables for AMD ROCm GPUs (RX 6900 XT = gfx1030)
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-max_split_size_mb:512}"
HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
CUDA_HOME_OVERRIDE="${CUDA_HOME_OVERRIDE:-/opt/rocm}"

# Legacy variable name (deprecated but kept for compatibility)
PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-${PYTORCH_ALLOC_CONF}}"

# Force NVIDIA Warp to use CPU mode (not compatible with ROCm)
WARP_DISABLE_CUDA="${WARP_DISABLE_CUDA:-1}"

# Use CPU fallback for ray-triangle intersection (AMD ROCm compatible)
USE_CPU_RAY_INTERSECTION="${USE_CPU_RAY_INTERSECTION:-1}"

# Ensure config module is importable
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# =============================================================================
# GEN3C EXECUTION
# =============================================================================
PY_SCRIPT="$GEN3C_DIR/cosmos_predict1/diffusion/inference/gen3c_single_image.py"
if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "ERROR: expected GEN3C inference script at $PY_SCRIPT" >&2
  exit 1
fi

GEN3C_ARGS=(
  --checkpoint_dir "$CHECKPOINT_DIR"
  --input_image_path "$INPUT_IMAGE"
  --video_save_name "$VIDEO_NAME"
  --guidance "$GUIDANCE"
  --offload_diffusion_transformer
  --offload_tokenizer
  --offload_text_encoder_model
  --disable_prompt_upsampler
  --disable_guardrail
)

if [[ -n "$NUM_FRAMES" ]]; then
  GEN3C_ARGS+=(--num_video_frames "$NUM_FRAMES")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  GEN3C_ARGS+=(${EXTRA_ARGS[@]})
fi

echo "=============================================="
echo "GEN3C Launcher - searidge cluster"
echo "=============================================="
echo "Node: $(hostname)"
echo "GEN3C Dir: $GEN3C_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Conda Env: $ENV_NAME"
echo "GPU Device: $HIP_VISIBLE_DEVICES"
echo "=============================================="

echo "Running GEN3C via conda env '$ENV_NAME'..."
# Include apex-rocm in PYTHONPATH if it exists
APEX_ROCM_DIR="$GEN3C_DIR/.ext/apex-rocm"
if [[ -d "$APEX_ROCM_DIR" ]]; then
  GEN3C_PYTHONPATH="$GEN3C_DIR:$APEX_ROCM_DIR"
else
  GEN3C_PYTHONPATH="$GEN3C_DIR"
fi

set -x
conda run -n "$ENV_NAME" env \
  HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
  PYTORCH_ALLOC_CONF="$PYTORCH_ALLOC_CONF" \
  PYTORCH_HIP_ALLOC_CONF="$PYTORCH_HIP_ALLOC_CONF" \
  HSA_OVERRIDE_GFX_VERSION="$HSA_OVERRIDE_GFX_VERSION" \
  CUDA_HOME="$CUDA_HOME_OVERRIDE" \
  WARP_DISABLE_CUDA="$WARP_DISABLE_CUDA" \
  USE_CPU_RAY_INTERSECTION="$USE_CPU_RAY_INTERSECTION" \
  PYTHONPATH="$GEN3C_PYTHONPATH" \
  python "$PY_SCRIPT" "${GEN3C_ARGS[@]}"
set +x

# =============================================================================
# OUTPUT HANDLING
# =============================================================================
# GEN3C writes to its own videos/ directory, copy to our output location
OUTPUT_VIDEO="$GEN3C_DIR/videos/${VIDEO_NAME}.mp4"
if [[ -f "$OUTPUT_VIDEO" ]]; then
  cp "$OUTPUT_VIDEO" "$OUTPUT_DIR/${VIDEO_NAME}.mp4"
  echo "✅ Copied video to $OUTPUT_DIR/${VIDEO_NAME}.mp4"
else
  echo "WARNING: Expected output video at $OUTPUT_VIDEO was not found." >&2
fi

echo "GEN3C run complete."
