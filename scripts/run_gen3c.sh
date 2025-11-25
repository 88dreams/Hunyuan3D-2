#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Wrapper to run GEN3C single-image inference from the Hunyuan3D workspace.
# Usage:
#   scripts/run_gen3c.sh --input path/to/image.png --video-name my_test \
#       [--guidance 1.0] [--frames 121] [--checkpoint-dir /path/to/checkpoints] \
#       [--extra "--trajectory left --foreground_masking"]
# -----------------------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN3C_DIR=${GEN3C_DIR:-"$HOME/GEN3C"}
ENV_NAME=${ENV_NAME:-"gen3c-rocm"}
DEFAULT_CHECKPOINT_DIR="$GEN3C_DIR/checkpoints"
DEFAULT_OUTPUT_DIR="$PROJECT_ROOT/assets/gen3c_outputs"

INPUT_IMAGE=""
VIDEO_NAME="gen3c_video"
GUIDANCE="1"
NUM_FRAMES=""
CHECKPOINT_DIR="$DEFAULT_CHECKPOINT_DIR"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage: run_gen3c.sh --input <image> [options]

Options:
  --input PATH           Source image (required)
  --video-name NAME      Base name for the generated video (default: gen3c_video)
  --guidance VALUE       Guidance scale (default: 1)
  --frames N             Number of frames (must follow 121*N-1 rule; optional)
  --checkpoint-dir PATH  GEN3C checkpoint directory (default: $HOME/GEN3C/checkpoints)
  --output-dir PATH      Directory where the resulting video will be copied
                         after GEN3C finishes (default: assets/gen3c_outputs)
  --extra "ARGS"         Additional arguments passed verbatim to GEN3C
  -h, --help             Show this help
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

if [[ -z "$INPUT_IMAGE" ]]; then
  echo "ERROR: --input is required." >&2
  usage
  exit 1
fi

if [[ ! -d "$GEN3C_DIR" ]]; then
  echo "ERROR: GEN3C directory not found at $GEN3C_DIR" >&2
  exit 1
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "ERROR: checkpoint directory not found at $CHECKPOINT_DIR" >&2
  exit 1
fi

if [[ ! -f "$INPUT_IMAGE" ]]; then
  echo "ERROR: input image not found: $INPUT_IMAGE" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
PYTORCH_HIP_ALLOC_CONF=${PYTORCH_HIP_ALLOC_CONF:-max_split_size_mb:512}
HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-11.0.0}
CUDA_HOME_OVERRIDE=${CUDA_HOME_OVERRIDE:-/opt/rocm}

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

echo "Running GEN3C via conda env '$ENV_NAME'..."
set -x
conda run -n "$ENV_NAME" env \
  HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
  PYTORCH_HIP_ALLOC_CONF="$PYTORCH_HIP_ALLOC_CONF" \
  HSA_OVERRIDE_GFX_VERSION="$HSA_OVERRIDE_GFX_VERSION" \
  CUDA_HOME="$CUDA_HOME_OVERRIDE" \
  PYTHONPATH="$GEN3C_DIR" \
  python "$PY_SCRIPT" "${GEN3C_ARGS[@]}"
set +x

OUTPUT_VIDEO="$GEN3C_DIR/videos/${VIDEO_NAME}.mp4"
if [[ -f "$OUTPUT_VIDEO" ]]; then
  cp "$OUTPUT_VIDEO" "$OUTPUT_DIR/${VIDEO_NAME}.mp4"
  echo "Copied video to $OUTPUT_DIR/${VIDEO_NAME}.mp4"
else
  echo "WARNING: Expected output video at $OUTPUT_VIDEO was not found." >&2
fi

echo "GEN3C run complete."

