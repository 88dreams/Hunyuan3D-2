# GEN3C (Cosmos) Integration Guide

This document explains how to install NVIDIA TLabs' GEN3C repository locally and prepare it for use from the existing Hunyuan3D UI. Follow the steps exactly so the upstream Transformer Engine, Apex, and MoGe dependencies build correctly inside a Conda-managed Python 3.10 environment.

---

## 1. Clone the Repository

```bash
cd /home/arkrunr
git clone https://github.com/nv-tlabs/GEN3C.git
cd GEN3C
```

Keep the clone outside the main Hunyuan3D repo to avoid polluting its git history.

---

## 2. Python & Conda Requirements

- GEN3C supports **Linux** only (tested on Ubuntu 20.04/22.04/24.04).
- Requires **Python 3.10.x** managed by **Conda**. The system Python (3.13.5) can stay untouched; we just need a dedicated Conda environment that pins Python 3.10.
- Ensure ROCm or CUDA drivers/toolkits that match your GPU are already installed and exposed through environment variables (`HIP_VISIBLE_DEVICES`, `CUDA_HOME`, etc.) as needed.

---

## 3. Create the `cosmos-predict1` Conda Environment

```bash
conda env create --file cosmos-predict1.yaml
conda activate cosmos-predict1
```

This installs Python 3.10.x plus base dependencies defined by NVIDIA TLabs. After activation, all commands below assume the environment is active.

---

## 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs runtime libraries such as PyTorch, Transformer Engine stubs, and other Python packages required for inference.

---

## 5. Patch Transformer Engine Headers inside Conda

Conda environments can hide NVIDIA headers from external builds. Create symbolic links so future builds can find them:

```bash
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
```

---

## 6. Install Transformer Engine, Apex, and MoGe

```bash
pip install transformer-engine[pytorch]==1.12.0
git clone https://github.com/NVIDIA/apex
CUDA_HOME=$CONDA_PREFIX pip install -v --disable-pip-version-check --no-cache-dir \
  --no-build-isolation --config-settings "--build-option=--cpp_ext" \
  --config-settings "--build-option=--cuda_ext" ./apex
pip install git+https://github.com/microsoft/MoGe.git
```

- `CUDA_HOME` (or ROCm equivalent) must point to the Conda prefix so Apex links against the right toolkit.
- `transformer-engine` must match the version from the GEN3C instructions (1.12.0 at time of writing).

---

## 7. Optional: Containerized Setup

If you prefer Docker (with NVIDIA Container Toolkit already installed):

```bash
docker build -f Dockerfile . -t nvcr.io/$USER/cosmos-predict1:latest
```

If you hit permission issues when mounting volumes, run:

```bash
alias share='sudo chown -R ${USER}:users $PWD && sudo chmod g+w $PWD'
share
```

Then retry the container workflow.

---

## 8. Validate the Environment

```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/test_environment.py
```

This upstream smoke test confirms Transformer Engine, Apex, and MoGe all link properly.

---

## 9. Wiring GEN3C into the Existing UI

1. **Add a launcher script** (e.g., `scripts/run_gen3c.sh`) that activates `cosmos-predict1`, sets required env vars (ROCk/CUDA paths, Hugging Face tokens), and runs the GEN3C inference entry point.
2. **Expose configuration** inside `2d3d.py` or `3dbuild.py` to call GEN3C by spawning that launcher script when the user selects the GEN3C backend. Capture stdout/stderr to the existing Gradio log panels.
3. **Define output contracts**: decide where GEN3C writes mesh outputs (e.g., `assets/gen3c_outputs/`). Ensure the Gradio UI watches that path and imports the generated GLB/PLY files once the process completes.
4. **Share caching**: if GEN3C also uses Hugging Face caches, point its `TRANSFORMERS_CACHE` to the same directory used by Hunyuan3D to avoid duplicate downloads.
5. **Document environment activation** for CLI workflows: e.g., `source scripts/run_gen3c.sh --prompt "..."`.

Capture any additional flags (prompt text, reference images, output resolutions) in the UI so end users can seamlessly switch between Hunyuan3D and GEN3C generation paths.

---

## 10. Maintenance Notes

- Track GEN3C updates: re-run the Transformer Engine/Apex build steps whenever upstream requirements change.
- Keep the Conda env lightweight by avoiding extra packages that could conflict with the pinned versions.
- Document ROCm/CUDA versions known to work so other contributors can reproduce results.

Following this guide ensures GEN3C can run locally and be orchestrated through the existing Hunyuan3D UI without destabilizing the current Python 3.13 production environment.

---

## 11. ROCm Alternative (AMD GPUs)

NVIDIA TLabs only documents CUDA builds, but our production hardware relies on AMD GPUs running ROCm 6.12+. Use the helper script we added to this repo to provision a compatible environment.

1. **Run the ROCm setup script**
   ```bash
   cd /home/arkrunr/Hunyuan3D-2-Fork
   ./scripts/setup_gen3c_rocm.sh
   ```
   - Creates (or updates) the `gen3c-rocm` Conda env with Python 3.10.
   - Installs ROCm PyTorch wheels (`torch==2.4.1+rocm6.1`, `torchvision==0.19.1+rocm6.1`).
   - Installs the rest of GEN3C’s dependencies, excluding CUDA-only wheels (`torch`, `torchvision`, `triton`, `warp-lang`, `nvidia-*`).
   - Clones and builds the ROCm fork of Apex from `ROCmSoftwarePlatform/apex`.
   - Installs the Python `amdsmi` bindings so PyTorch can query GPU metadata without crashing on ROCm (match the version to your installed ROCm stack, e.g. `conda run -n gen3c-rocm pip install --force-reinstall "amdsmi==6.4.4"` for ROCm 6.4).
   - **IMPORTANT:** Because Transformer Engine and NVIDIA’s Apex `amp_C` extensions are CUDA-only, GEN3C’s prompt-uplevel/autoregressive stack must run with the PyTorch backend on ROCm. In `cosmos_predict1/autoregressive/configs/...` set `backend: pytorch` (or pass `--prompt_upsampler_backend pytorch`) to avoid TE-only modules.

2. **ROCk/ROCm environment variables**  
   Before launching GEN3C on AMD hardware, export the same variables we rely on for Hunyuan3D:
   ```bash
   export HIP_VISIBLE_DEVICES=0          # pick the desired GPU
   export HSA_OVERRIDE_GFX_VERSION=11.0.0
   export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
   ```

3. **Validate the environment**
   ```bash
   conda run -n gen3c-rocm python scripts/test_environment.py
   ```
   Expect some functionality (Transformer Engine FP8 kernels, NVIDIA ModelOpt optimizations, `warp-lang`) to be unavailable on ROCm. The script will warn about missing `transformer_engine`, which is expected on AMD hardware; Megatron-Core, Diffusers, and ROCm Torch should still load successfully. If you need to capture this behavior in automation, treat a missing Transformer Engine import as informational whenever `torch.version.hip` is defined (see `scripts/test_environment.py` notes below).

4. **Document limitations**  
   Call out in PRs or runbooks when CUDA-only features are disabled so downstream teams know the ROCm backend may have different performance characteristics. Where GEN3C hard-codes CUDA-only imports, wrap them in `try/except` blocks and log a warning pointing to this section.

### 11.1 Launcher Script for Single-Image Inference

Once the environment is ready, trigger GEN3C via the helper script added to this repo:

```bash
cd /home/arkrunr/Hunyuan3D-2-Fork
./scripts/run_gen3c.sh \
  --input /path/to/image.png \
  --video-name lobby_pan \
  --guidance 1 \
  --extra "--trajectory left --foreground_masking"
```

- The script handles `conda run -n gen3c-rocm ...`, exports `HIP_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION`, and `PYTORCH_HIP_ALLOC_CONF`, then copies the generated video into `assets/gen3c_outputs/`.
- Override defaults with environment variables:
  - `GEN3C_DIR` (default `/home/arkrunr/GEN3C`)
  - `ENV_NAME` (default `gen3c-rocm`)
  - `CUDA_HOME_OVERRIDE` (default `/opt/rocm`)
- Pass additional GEN3C CLI flags via `--extra` or a trailing `--`, e.g. camera trajectories, `--foreground_masking`, or offloading options published in the GEN3C README.

Use this same launcher from manual CLI runs or when wiring the Hunyuan3D UI buttons so both paths share the exact defaults.

This workflow keeps the CUDA-centric upstream instructions intact while giving us a reproducible AMD/ROCm path tied into the same UI toggles.

### 11.2 Known ROCm Limitation (Transformer Engine)

- NVIDIA’s Transformer Engine does not currently provide ROCm wheels, so `pip install transformer-engine[...]` will either fail or install CUDA-only stubs that cannot load.
- `scripts/test_environment.py` detects this and now degrades gracefully: on HIP builds it logs a warning instead of exiting with an error. You’ll still get an `[ERROR] transformer_engine` line in interactive runs so operators know the CUDA-only optimizations are unavailable.
- GEN3C inference continues to run without Transformer Engine; only NVIDIA’s FP8 kernels and ModelOpt features are skipped. Document this in any release notes so users on AMD hardware understand the performance trade-off.

---

## 12. Gradio UI Parameter Reference

Use this section as an operator cheat sheet when launching jobs from `2d3d.py`.

### Common Controls
- **Input image**: File picker on the left column. Supply a high-resolution PNG/JPG; transparent PNGs work best if you plan to skip background removal.
- **Backend Selection**: `Hunyuan GLB` (default) runs the local shape pipeline and produces a GLB mesh. `GEN3C Video` offloads to NVIDIA’s Gen3C repo via `scripts/run_gen3c.sh`. Toggling this radio automatically shows only the relevant parameter group and relabels the `Generate` button.
- **System Resources / Generation Progress**: Read-only text boxes that stream CPU/GPU usage and high-level stage updates. If you see no updates for several minutes, check terminal logs for hangs.
- **Output (GLB or MP4)**: Download widget that surfaces whichever artifact the selected backend produces (GLB for Hunyuan, MP4 for GEN3C). The `Logs` panel concatenates stdout, stderr, and friendly hints; always skim it after a run for warnings.

### Hunyuan Controls (visible when `Hunyuan GLB` is selected)
- **Guidance scale (default 9.0)**: Higher values (8–11) force stricter adherence to the input image, which is ideal for architectural interiors demanding clean lines. Lowering to ~7 adds freedom if you want more creative geometry.
- **Inference steps (default 40)**: Controls diffusion iterations. 35–45 balances quality and runtime (~25 minutes end-to-end). Drop to ~25 if you hit queue timeouts; increase toward 60 only if you can wait longer.
- **Seed (default 42)**: Integer for repeatability. Leave blank to randomize outputs; reuse a seed to debug regressions.
- **Model Selection**: `Mini Model (Faster)` auto-sets octree resolution 380 / chunks 6000 for maximum detail that still fits into 16 GB VRAM. `Full Model (Higher Quality)` switches to the full DiT weights with octree 360 / chunks 5000; use it when you can tolerate longer CPU-bound meshing.
- **Use FP16 / Attention slicing / CPU offload**: Memory safety toggles. Keep all three enabled on 16 GB cards. Disable sequentially only if you need to benchmark raw speed and know you have ample VRAM.
- **Remove background automatically**: Calls `hy3dgen.rembg`. Useful when stage photos contain complex seating or rigging you want to drop before 3D reconstruction. Slightly increases preprocessing time; leave off if you already supply an alpha-matted render.
- **Output name / Save Location**: Controls the final `*_shape.glb` filename and destination. By default we write to `assets/hunyuan_outputs`; use the embedded file explorer to browse to a different folder. The directory is created automatically if it doesn’t exist.

### GEN3C Controls (visible when `GEN3C Video` is selected)
- **GEN3C Guidance (default 1.0)**: Maps to the `--guidance` flag in `gen3c_single_image.py`. Values between 0.8 and 1.2 gently steer the video toward the source image while keeping motion natural. Larger than 2.0 can introduce jitter.
- **GEN3C Frames**: Dropdown that feeds `--num_video_frames`. Select one of the supported counts (121, 241, 361, 481). These satisfy NVIDIA’s constraint `(frames - 1) % 120 == 0`; higher values lengthen the clip but consume more memory and time.
- **GEN3C Video Name**: Base name for the MP4 stored under `assets/gen3c_outputs/`. Avoid spaces; the UI appends `.mp4`.
- **GEN3C Checkpoint Directory**: Defaults to `/home/arkrunr/GEN3C/checkpoints`. Override if you mirror checkpoints to another disk. Must contain the `Gen3C-Cosmos-7B`, tokenizer, guardrail, and T5 folders downloaded earlier.
- **Save Location**: Destination for the copied MP4 after the launcher completes (defaults to `assets/gen3c_outputs`). Use the explorer widget to browse to any writable directory; the path will be created if necessary.
- **Additional GEN3C Arguments**: Free-form string forwarded to the launcher’s `--extra`. Use it for camera motions (`--trajectory left/right/full_orbit`), masking flags (`--foreground_masking`), or any other CLI switches described in NVIDIA’s README. You can also append `--` in the textbox to manually pass multiple flags exactly as the upstream script expects.

Tips:
- Always verify the `gen3c-rocm` environment is active and the checkpoints exist before running the GEN3C backend; the UI does not re-install dependencies automatically.
- If GEN3C jobs fail silently, open the `Logs` box for the raw `stdout/stderr` captured from `scripts/run_gen3c.sh`.
- For both backends, set the Gradio timeout high enough (currently 45 minutes) and avoid launching multiple runs concurrently to prevent VRAM exhaustion.

