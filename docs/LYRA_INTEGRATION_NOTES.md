# Lyra Integration Notes for RunPod Serverless

**Created:** 2025-12-22
**Updated:** 2025-12-22
**Purpose:** Track dependency changes for Lyra integration and rollback instructions

---

## Current State

Lyra is already cloned in the unified Docker image (`Dockerfile.unified`) at `/workspace/lyra`.
The base environment (`cosmos-predict1`) is shared with GEN3C.

**Current Docker image:** `88dreams/gen3c-runpod:v22` (stable, without Lyra support)
**New Docker image:** `88dreams/gen3c-runpod:v23` (with Lyra exact dependencies)

---

## Changes Made (v23)

### 1. Dockerfile.unified Updated
- `flash-attn`: 2.7.3 → 2.7.4.post1
- `gsplat`: 1.4.0 → exact commit `73fad53c31ec4d6b088470715a63f432990493de`
- Added `lyra_inference.py` copy to `/workspace/`

### 2. New Files Created
- `runpod/gen3c/lyra_inference.py` - Wrapper script for 2-step Lyra pipeline

### 3. Handler Updated
- `handler_unified.py` - `run_lyra()` now calls `lyra_inference.py` instead of raising error

---

## Dependency Comparison: GEN3C vs Lyra

### Identical (No Changes Needed)
- Python: 3.10
- CUDA: 12.4
- PyTorch: 2.6.0
- NumPy: 1.26.4
- transformers: 4.49.0
- huggingface-hub: 0.29.2
- opencv-python: 4.10.0.84

### Lyra-Specific Packages (Already in Dockerfile.unified)
- flash-attn
- timm==1.0.19
- kiui==0.2.17
- lru-dict==1.3.0
- causal-conv1d (for Mamba)
- gsplat (3DGS renderer)
- fused-ssim
- mpi4py==4.1.0
- plyfile==1.1.2
- deepspeed==0.17.5
- accelerate==1.10.0
- mamba (state space model)

---

## Changes Needed

### 1. gsplat Version
**Current (Dockerfile.unified line ~118):**
```dockerfile
pip install --no-cache-dir gsplat==1.4.0
```

**Lyra requires (specific commit):**
```dockerfile
pip install --no-cache-dir git+https://github.com/nerfstudio-project/gsplat.git@73fad53c31ec4d6b088470715a63f432990493de
```

**Rollback:** Change back to `gsplat==1.4.0`

### 2. flash-attn Version
**Current (Dockerfile.unified line ~106):**
```dockerfile
flash-attn==2.7.3
```

**Lyra requires:**
```dockerfile
flash_attn==2.7.4.post1
```

**Rollback:** Change back to `flash-attn==2.7.3`

---

## Lyra Inference Pipeline

Lyra is a **2-step pipeline**:

### Step 1: SDG (Synthetic Data Generation) - Uses GEN3C
For static (image → 3D):
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=1 \
    cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py \
    --checkpoint_dir checkpoints \
    --num_gpus 1 \
    --input_image_path <input_image> \
    --video_save_folder <output_latents_dir> \
    --foreground_masking \
    --multi_trajectory \
    --total_movement_distance_factor 1.0
```

For dynamic (video → 4D):
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=1 \
    cosmos_predict1/diffusion/inference/gen3c_dynamic_sdg.py \
    --checkpoint_dir checkpoints \
    --vipe_path <input_video> \
    --video_save_folder <output_latents_dir> \
    --disable_prompt_upsampler \
    --num_gpus 1 \
    --foreground_masking \
    --multi_trajectory
```

### Step 2: 3DGS Reconstruction - Lyra Decoder
```bash
accelerate launch sample.py --config configs/demo/lyra_static.yaml
```
or
```bash
accelerate launch sample.py --config configs/demo/lyra_dynamic.yaml
```

**Note:** The YAML config files specify dataset paths. For serverless, we need to:
1. Generate latents in Step 1
2. Modify config to point to generated latents
3. Run Step 2

---

## Checkpoints Needed

### Already on Network Volume (for GEN3C)
- `/workspace/checkpoints/Gen3C-Cosmos-7B/` (~71GB)
- `/workspace/checkpoints/cosmos_predict1/` (tokenizer)

### Need to Download for Lyra
- Lyra 3DGS decoder weights from `nvidia/Lyra` on HuggingFace (~small, 32.75M params)

Download command:
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_lyra_checkpoints.py --checkpoint_dir checkpoints
```

---

## Files Created/Modified (DONE)

### 1. Lyra Inference Script ✅
Created: `runpod/gen3c/lyra_inference.py`
- Wrapper that runs both SDG and 3DGS decoder steps
- Handles temp directories for intermediate latents
- Returns PLY file path

### 2. Handler Update ✅
Modified: `runpod/gen3c/handler_unified.py`
- Updated `run_lyra()` function to call the inference script
- Removed the "not implemented" error

### 3. Dockerfile Update ✅
Modified: `runpod/gen3c/Dockerfile.unified`
- Updated gsplat to Lyra's specific commit
- Updated flash-attn to 2.7.4.post1
- Added COPY for lyra_inference.py

---

## Rollback Instructions

If Lyra integration breaks GEN3C/SHARP:

### Quick Rollback (Revert Docker image)
```bash
# On RunPod endpoint, change Docker image back to:
88dreams/gen3c-runpod:v22
```

### Code Rollback
1. In `Dockerfile.unified`:
   - Line ~106: Change `flash_attn==2.7.4.post1` back to `flash-attn==2.7.3`
   - Line ~118: Change gsplat commit back to `gsplat==1.4.0`

2. In `handler_unified.py`:
   - Revert `run_lyra()` to return "not implemented" error

3. Rebuild:
```bash
cd /home/arkrunr02/Hunyuan3D-2-Fork/runpod/gen3c
docker build -t 88dreams/gen3c-runpod:v23-rollback -f Dockerfile.unified .
docker push 88dreams/gen3c-runpod:v23-rollback
```

---

## Testing Checklist

After Lyra integration, verify these still work:

- [ ] GEN3C video generation
- [ ] SHARP PLY generation
- [ ] S3 upload for large files
- [ ] Lyra static (image → 3DGS)
- [ ] Lyra dynamic (video → 4DGS) [if implemented]

---

## Known Blockers

### 1. DINOv3 Gated Model (TRELLIS.2 issue, not Lyra)
TRELLIS.2 requires `facebook/dinov3-vitl16-pretrain-lvd1689m` which is gated.
User has requested access, waiting for approval.

### 2. Lyra Checkpoints ⚠️ REQUIRED BEFORE TESTING
Need to download Lyra-specific checkpoints to network volume before testing.

---

    ## NEXT STEPS (What Remains)

### Step 1: Build and Push Docker Image v23
```bash
cd /home/arkrunr02/Hunyuan3D-2-Fork/runpod/gen3c
docker build -t 88dreams/gen3c-runpod:v23 -f Dockerfile.unified . 2>&1
docker push 88dreams/gen3c-runpod:v23 2>&1
```

### Step 2: Download Lyra Checkpoints to Network Volume
On a RunPod GPU pod with network volume mounted:
```bash
# Activate environment
source /root/miniforge3/bin/activate cosmos-predict1

# Login to HuggingFace
python -c "from huggingface_hub import login; login()"

# Download Lyra checkpoints
cd /workspace/lyra
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_lyra_checkpoints.py --checkpoint_dir /workspace/checkpoints
```

Expected checkpoint location: `/workspace/checkpoints/lyra/` (maps to `/runpod-volume/checkpoints/lyra/` on serverless)

### Step 3: Update RunPod Endpoint
- Go to RunPod serverless endpoint settings
- Change Docker image to `88dreams/gen3c-runpod:v23`
- Ensure `HF_TOKEN` environment variable is set

### Step 4: Test Lyra
From the Gradio UI:
1. Go to Lyra tab
2. Upload an image
3. Select "Static (Image → 3DGS)" mode
4. Click Generate

### Step 5: Verify GEN3C/SHARP Still Work
After Lyra integration, verify:
- [ ] GEN3C video generation
- [ ] SHARP PLY generation
- [ ] S3 upload for large files

---

## Estimated Lyra Checkpoint Size

Based on HuggingFace model card:
- Lyra 3DGS decoder: ~32.75M parameters (small)
- But also needs GEN3C checkpoints (already have: ~71GB)

The Lyra-specific checkpoints should be relatively small (<1GB).

---

## References

- Lyra GitHub: https://github.com/nv-tlabs/lyra
- Lyra INSTALL.md: https://github.com/nv-tlabs/lyra/blob/main/INSTALL.md
- Lyra Model Weights: https://huggingface.co/nvidia/Lyra
- GEN3C GitHub: https://github.com/nv-tlabs/GEN3C

