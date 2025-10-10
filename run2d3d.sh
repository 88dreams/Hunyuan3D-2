#!/usr/bin/env bash
set -euo pipefail

# Set your env's Python path (adjust if different)
ENVPY="/home/arkrunr/opt/miniconda3/envs/hunyuan3d/bin/python"
CONDA_PREFIX="$(dirname "$(dirname "$ENVPY")")"

export HSA_OVERRIDE_GFX_VERSION="10.3.0"
export HIP_VISIBLE_DEVICES="0"
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:256"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/x86_64-conda-linux-gnu/lib:${LD_LIBRARY_PATH:-}"

exec "$ENVPY" /home/arkrunr/Hunyuan3D-2-Fork/2d3d.py
