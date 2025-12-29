# Hunyuan3D-2-Fork Project Status

**Last Updated:** December 26, 2025

## Project Goal

Build a system that takes a **single 2D image** (specifically architectural interiors) and creates a **high-quality 3D model**. The workflow involves testing multiple AI models to find the best approach for this use case.

---

## Architecture Overview

### Local System
- **Main UI:** `2d3d.py` - Gradio-based interface with tabs for each model
- **Location:** `/home/arkrunr02/Hunyuan3D-2-Fork/`
- **Output Directory:** `/srv/searidge_share/outputs/`

### RunPod Serverless Infrastructure
- **Network Volume ID:** `gg3ercsn6p`
- **Checkpoints Location:** `/runpod-volume/checkpoints/` (maps to `/workspace/checkpoints/` in containers)

### Docker Images (on Docker Hub: `88dreams/`)
| Image | Purpose | Current Version |
|-------|---------|-----------------|
| `gen3c-runpod` | GEN3C, SHARP, Lyra (unified) | v35 (testing) |
| `trellis-runpod` | TRELLIS.2 (dedicated) | v7 |
| `hunyuan-runpod` | Hunyuan3D (dedicated) | v9 |

### RunPod Endpoints
| Model | Endpoint ID | Docker Image |
|-------|-------------|--------------|
| GEN3C/SHARP/Lyra | `1k4wuq26ugtiyu` | `88dreams/gen3c-runpod:v35` |
| TRELLIS.2 | `v26foemsxcj7be` | `88dreams/trellis-runpod:v7` |
| Hunyuan3D | `4wiztgeyjcy1y9` | `88dreams/hunyuan-runpod:v9` |

---

## Models Status

### ✅ GEN3C - WORKING
- **Purpose:** Image → Video generation
- **Status:** Fully functional
- **Output:** MP4 video files
- **S3 Integration:** Working (large files upload to S3, download locally)

### ✅ SHARP - WORKING
- **Purpose:** Image → PLY (3D Gaussian Splat)
- **Status:** Fully functional
- **Output:** PLY files
- **S3 Integration:** Working
- **Note:** Video rendering disabled (was unreliable)

### ⚠️ Lyra - IN PROGRESS (v35 testing)
- **Purpose:** Image → Video → 3DGS (most promising for quality)
- **Pipeline:** 
  1. SDG step (uses GEN3C to generate video)
  2. 3DGS reconstruction (converts video frames to Gaussian splat)
- **Current Issue:** CUDA JIT compilation error with gsplat library
- **Error:** `nvcc warning: incompatible redefinition for option 'compiler-bindir'`
- **Root Cause:** Conda environment compiler conflicts with nvcc

### ⚠️ TRELLIS.2 - PARTIALLY WORKING
- **Purpose:** Image → 3D model (GLB)
- **Status:** Endpoint configured, but requires gated HuggingFace model access
- **Checkpoints:** Manually downloaded to `/workspace/checkpoints/trellis/`
- **Issue:** DINOv3 model is gated on HuggingFace

### ⚠️ Hunyuan3D - WORKING BUT LOW QUALITY
- **Purpose:** Image → 3D model (direct)
- **Status:** Functional but quality insufficient for architectural interiors
- **Supports:** Local execution or RunPod serverless

---

## Key Files

### Main Application
```
/home/arkrunr02/Hunyuan3D-2-Fork/
├── 2d3d.py                    # Main Gradio UI
├── ui/tabs/
│   ├── gen3c_tab.py          # GEN3C UI components
│   ├── sharp_tab.py          # SHARP UI components
│   ├── lyra_tab.py           # Lyra UI components
│   ├── trellis_tab.py        # TRELLIS.2 UI components
│   └── hunyuan_tab.py        # Hunyuan3D UI components
├── generators/
│   ├── gen3c.py              # GEN3C generator logic
│   ├── sharp.py              # SHARP generator logic
│   ├── lyra.py               # Lyra generator logic
│   ├── trellis.py            # TRELLIS.2 generator logic
│   └── hunyuan.py            # Hunyuan3D generator logic
└── runpod/
    ├── runpod_client.py      # Unified RunPod API client
    ├── s3_download.py        # S3 download utilities
    └── __init__.py
```

### Docker/RunPod Files
```
/home/arkrunr02/Hunyuan3D-2-Fork/runpod/
├── gen3c/
│   ├── Dockerfile.unified    # Main Docker image (GEN3C/SHARP/Lyra)
│   ├── handler_unified.py    # RunPod serverless handler
│   ├── start_unified.sh      # Container startup script
│   ├── lyra_inference.py     # Lyra-specific inference wrapper
│   └── server_unified.py     # Server configuration
├── trellis/
│   ├── Dockerfile            # TRELLIS.2 Docker image
│   ├── handler_trellis.py    # TRELLIS.2 handler
│   ├── trellis_inference.py  # TRELLIS.2 inference wrapper
│   └── start_trellis.sh      # Container startup
└── hunyuan/
    ├── Dockerfile            # Hunyuan3D Docker image
    ├── handler_hunyuan.py    # Hunyuan3D handler
    └── start_hunyuan.sh      # Container startup
```

---

## Credential Persistence

Credentials are stored per-model in `~/.runpod_config.json`:
```json
{
  "gen3c_endpoint_id": "...",
  "gen3c_api_key": "...",
  "sharp_endpoint_id": "...",
  "sharp_api_key": "...",
  "lyra_endpoint_id": "...",
  "lyra_api_key": "...",
  "trellis_endpoint_id": "...",
  "trellis_api_key": "...",
  "hunyuan_endpoint_id": "...",
  "hunyuan_api_key": "..."
}
```

---

## S3 Integration

- **Bucket:** `arkrunr`
- **Region:** `us-west-1`
- **Prefix:** `MediaContent/outputs/`
- **IAM User:** `runpod` (has PutObject and GetObject permissions)

Files larger than 8MB are automatically:
1. Uploaded to S3 by the RunPod handler
2. Downloaded locally by the client via presigned URL

---

## Lyra Integration Details

### Two-Step Pipeline
1. **SDG (Synthetic Data Generation):**
   - Uses GEN3C's video generation
   - Generates latent video from input image
   - Output: Video latents in temp directory

2. **3DGS Reconstruction:**
   - Takes video latents
   - Reconstructs 3D Gaussian Splat
   - Uses `sample.py` from Lyra repo

### Checkpoints Required
```
/workspace/checkpoints/
├── lyra/
│   ├── lyra_static.pt        # Main Lyra checkpoint
│   └── lyra_dynamic.pt       # (optional) Dynamic version
└── Gen3C-Cosmos-7B/
    └── Cosmos-Tokenize1-CV8x8x8-720p/
        └── mean_std.pt       # Cosmos tokenizer stats
```

### Symlinks Created (in start_unified.sh)
```
/workspace/checkpoints/Lyra -> /runpod-volume/checkpoints/lyra
/workspace/lyra/checkpoints/Lyra -> /runpod-volume/checkpoints/lyra
/workspace/lyra/checkpoints/cosmos_predict1/Cosmos-Tokenize1-CV8x8x8-720p -> /runpod-volume/checkpoints/Gen3C-Cosmos-7B/Cosmos-Tokenize1-CV8x8x8-720p
```

### Code Patches Applied (in Dockerfile)
```bash
# Patch gs_deferred.py to remove unsupported gsplat arguments
sed -i "s/self.raster_kwargs = {'with_ut': True, 'with_eval3d': True, 'packed': False}/self.raster_kwargs = {'packed': False}/"
sed -i "s/self.raster_kwargs = {'with_ut': False, 'with_eval3d': False, 'packed': False}/self.raster_kwargs = {'packed': False}/"
```

---

## Docker Build History (gen3c-runpod)

| Version | Changes | Result |
|---------|---------|--------|
| v22 | Base working version | GEN3C/SHARP working |
| v23 | Added Lyra dependencies | gsplat commit failed to build |
| v24 | Reverted gsplat to 1.4.0, fixed validation | Lyra validation fixed |
| v25 | Increased subprocess timeout to 2 hours | Timeout fixed |
| v26 | Fixed ckpt_path in config | Still relative path issue |
| v27 | Added symlinks in start_unified.sh | Case sensitivity issue |
| v28 | Set ckpt_path in Python config | OmegaConf override issue |
| v29 | Pass ckpt_path as CLI argument | SDG output path issue |
| v30 | Symlink for diffusion_output | mean_std.pt not found |
| v31 | Pass vae_path CLI, add Cosmos symlinks | with_ut error |
| v32 | Patch gs_deferred.py (remove with_ut) | CUDA JIT compile error |
| v33 | Pre-compile gsplat (import test) | Still JIT error |
| v34 | TORCH_CUDA_ARCH_LIST during install | Still JIT error |
| v35 | Set CC/CXX/CUDAHOSTCXX to system gcc | **TESTING** |

---

## Known Issues & Workarounds

### 1. GPU Architecture Compatibility
- **Issue:** Lyra fails on Blackwell (B200) GPUs
- **Workaround:** Use H100/H200 (Hopper) or A100 (Ampere) GPUs

### 2. Conda Compiler Conflicts
- **Issue:** `nvcc warning: incompatible redefinition for option 'compiler-bindir'`
- **Root Cause:** Conda environment sets conflicting compiler paths
- **Attempted Fix (v35):** Set `CC=/usr/bin/gcc`, `CXX=/usr/bin/g++`, `CUDAHOSTCXX=/usr/bin/g++`

### 3. gsplat JIT Compilation
- **Issue:** gsplat uses torch.utils.cpp_extension.load() for runtime CUDA compilation
- **Problem:** Cannot pre-compile during Docker build (no GPU available)
- **Potential Solutions:**
  - Fix compiler environment (v35 approach)
  - Use non-conda Python environment
  - Build separate Lyra-only container

### 4. HuggingFace Gated Models
- **Issue:** Some models (DINOv3 for TRELLIS) require HuggingFace access approval
- **Solution:** Set `HF_TOKEN` environment variable in RunPod endpoint

---

## Common Commands

### Docker Build
```bash
cd /home/arkrunr02/Hunyuan3D-2-Fork/runpod/gen3c
docker build -t 88dreams/gen3c-runpod:vXX -f Dockerfile.unified .
docker push 88dreams/gen3c-runpod:vXX
```

### Clean Docker
```bash
docker system prune -a -f
docker builder prune --all -f
```

### Check Disk Space
```bash
df -h /
docker system df
```

### Start Local UI
```bash
cd /home/arkrunr02/Hunyuan3D-2-Fork
conda activate gen3c-rocm
python 2d3d.py
```

### SSH to RunPod Storage Pod
```bash
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519
```

---

## Next Steps (if v35 fails)

1. **Option A:** Fix at script level
   - Modify `lyra_inference.py` to set compiler env vars before subprocess call
   - Or modify `start_unified.sh` to unset conda compiler paths

2. **Option B:** Dedicated Lyra container
   - Build new Docker image without conda
   - Use pip/venv for Python environment
   - Deploy as separate RunPod endpoint

3. **Option C:** Pre-built gsplat wheel
   - Find or build gsplat wheel with CUDA kernels pre-compiled
   - Install wheel instead of building from source

---

## Contact & Resources

- **RunPod Dashboard:** https://www.runpod.io/console/serverless
- **Docker Hub:** https://hub.docker.com/u/88dreams
- **S3 Console:** AWS S3 bucket `arkrunr`

---

## Changelog

### December 26, 2025
- Created this status document
- Testing v35 (compiler environment fix for Lyra)
- All other models (GEN3C, SHARP, Hunyuan) working

### December 25, 2025
- Extensive Lyra debugging (v25-v34)
- Fixed path issues, config overrides, symlinks
- Identified gsplat CUDA JIT compilation as root cause

### December 22, 2025
- Set up TRELLIS.2 dedicated endpoint
- Manually downloaded TRELLIS checkpoints
- Fixed transformers version for DINOv3

### December 20-21, 2025
- Implemented S3 integration for large files
- Fixed IAM permissions
- Set up Hunyuan dedicated endpoint

### December 19, 2025
- Initial unified architecture setup
- GEN3C and SHARP working on RunPod serverless

