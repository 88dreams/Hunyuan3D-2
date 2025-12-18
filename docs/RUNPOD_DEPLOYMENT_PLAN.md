# RunPod Cloud Deployment Plan for GEN3C + Hunyuan3D

## Executive Summary

This document outlines a comprehensive plan to deploy the GEN3C and Hunyuan3D inference pipelines to RunPod's GPU cloud infrastructure, enabling fast inference on NVIDIA GPUs (A100/H100) while maintaining the existing web interface for user interaction.

**Current Problem**: Local AMD ROCm-based processing takes 14+ hours for a single GEN3C video generation due to:
1. CPU fallback for NVIDIA Warp (incompatible with ROCm)
2. Limited GPU memory (16GB) requiring extensive offloading
3. Single-system processing bottleneck

**Proposed Solution**: Deploy inference workloads to RunPod's NVIDIA GPUs while keeping the web UI accessible from your local network or hosted on Digital Ocean/Netlify.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │   Local Access  │    │  Digital Ocean  │    │    Netlify      │  │
│  │  (searidge02)   │    │   (Optional)    │    │   (Optional)    │  │
│  │   Gradio UI     │    │   Gradio UI     │    │  Static + API   │  │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘  │
│           │                      │                      │           │
│           └──────────────────────┼──────────────────────┘           │
│                                  │                                   │
│                                  ▼                                   │
│                    ┌─────────────────────────┐                      │
│                    │      API Gateway        │                      │
│                    │  (Job Queue Manager)    │                      │
│                    └───────────┬─────────────┘                      │
└────────────────────────────────┼────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        RUNPOD CLOUD                                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Option A: GPU Pods                        │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │    │
│  │  │  A100 Pod   │  │  A100 Pod   │  │  H100 Pod   │          │    │
│  │  │  (GEN3C)    │  │ (Hunyuan3D) │  │  (GEN3C)    │          │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              OR                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                 Option B: Serverless Endpoints               │    │
│  │  ┌─────────────────────┐  ┌─────────────────────┐           │    │
│  │  │  GEN3C Endpoint     │  │  Hunyuan3D Endpoint │           │    │
│  │  │  (Auto-scaling)     │  │  (Auto-scaling)     │           │    │
│  │  └─────────────────────┘  └─────────────────────┘           │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Decision Points

### 1.1 RunPod Deployment Model Selection

| Feature | GPU Pods | Serverless Endpoints |
|---------|----------|---------------------|
| **Billing** | Per-hour (even when idle) | Per-second (only when processing) |
| **Startup Time** | Instant (always running) | 10-30s cold start |
| **Best For** | Continuous workloads | Sporadic/batch jobs |
| **Complexity** | Lower (SSH access, manual) | Higher (handler code required) |
| **Cost for GEN3C** | ~$1.89/hr (A100 80GB) | ~$0.00031/sec (~$1.12/hr active) |
| **Recommendation** | Development/testing | Production |

**Recommendation**: Start with **GPU Pods** for development and testing, then migrate to **Serverless Endpoints** for production cost efficiency.

### 1.2 Web UI Hosting Options

| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **Local (searidge02)** | No additional cost, existing setup | Limited external access | Internal use |
| **Digital Ocean Droplet** | Full control, persistent | Monthly cost (~$6-12/mo) | API backend |
| **Netlify** | Free tier, CDN, easy deploy | Static only, no backend | Frontend only |
| **Gradio Share** | Free, instant public URL | Temporary (72h), unreliable | Quick demos |

**Recommendation**: 
- **Development**: Keep local Gradio UI on searidge02
- **Production**: Digital Ocean for API backend + Netlify for static frontend (optional)

---

## Phase 2: Docker Image Preparation

### 2.1 Base Image Selection

For NVIDIA GPUs on RunPod, use NVIDIA's official PyTorch containers:

```
Base Image Options:
- nvcr.io/nvidia/pytorch:24.01-py3  (PyTorch 2.2, CUDA 12.3)
- nvcr.io/nvidia/pytorch:24.08-py3  (PyTorch 2.4, CUDA 12.6)
- runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
```

### 2.2 Dockerfile Structure (GEN3C)

The Dockerfile must include:

1. **Base NVIDIA PyTorch image**
2. **System dependencies** (git, ffmpeg, libgl1)
3. **Python dependencies** from GEN3C requirements.txt
4. **Transformer Engine** (CUDA-native, works on NVIDIA)
5. **NVIDIA Apex** (with CUDA extensions)
6. **MoGe** (Microsoft's depth estimation)
7. **Model checkpoints** (pre-downloaded or downloaded at startup)
8. **Handler script** (for serverless) or **inference server** (for pods)

### 2.3 Dockerfile Structure (Hunyuan3D)

Similar structure but simpler:

1. **Base NVIDIA PyTorch image**
2. **System dependencies**
3. **Hunyuan3D Python dependencies**
4. **Model weights** (from HuggingFace)
5. **Handler/server script**

### 2.4 Model Checkpoint Strategy

**Option A: Baked into Image**
- Pros: Fast startup, no download wait
- Cons: Large image (~50-100GB), slow to push/pull
- Best for: Production with dedicated registry

**Option B: Download at Startup**
- Pros: Smaller image, easier updates
- Cons: 5-15 min startup time for cold starts
- Best for: Development, infrequent use

**Option C: Network Volume (RunPod)**
- Pros: Persistent storage, shared across pods
- Cons: Requires RunPod network volume setup
- Best for: Multiple pods sharing checkpoints

**Recommendation**: Use **Network Volume** for checkpoints, keep image lean.

---

## Phase 3: RunPod Account Setup

### 3.1 Prerequisites

1. Create RunPod account at https://runpod.io
2. Add payment method (credit card or crypto)
3. Add initial credits ($25-50 recommended for testing)

### 3.2 API Key Generation

1. Navigate to Settings → API Keys
2. Generate new API key
3. Store securely (needed for programmatic access)

### 3.3 Network Volume Setup (for checkpoints)

1. Navigate to Storage → Network Volumes
2. Create volume:
   - Name: `gen3c-checkpoints`
   - Size: 100GB (for all model weights)
   - Region: Same as your preferred GPU region
3. Note the volume ID for pod attachment

---

## Phase 4: GPU Pod Deployment (Development Path)

### 4.1 Create Pod Template

1. Navigate to Pods → Templates → New Template
2. Configure:
   - **Name**: `gen3c-inference`
   - **Container Image**: Your Docker Hub image (after building)
   - **Docker Command**: Leave empty or set startup script
   - **Expose Ports**: `8080` (API), `7860` (Gradio if included)
   - **Volume Mount**: `/workspace/checkpoints` → network volume

### 4.2 Launch Pod

1. Select GPU type:
   - **A100 80GB** (~$1.89/hr) - Recommended for GEN3C
   - **A100 40GB** (~$1.64/hr) - May work with offloading
   - **RTX 4090** (~$0.74/hr) - Budget option, 24GB VRAM
2. Attach network volume
3. Start pod

### 4.3 Pod Access Methods

- **SSH**: Direct terminal access for debugging
- **HTTP Ports**: Exposed services (API, Gradio)
- **Jupyter**: Built-in notebook interface

---

## Phase 5: Serverless Endpoint Deployment (Production Path)

### 5.1 Handler Script Requirements

RunPod serverless requires a `handler.py` with specific structure:

```python
# Conceptual structure (actual code in Phase 7)
import runpod

def handler(job):
    """
    Process incoming job request.
    
    Input (job["input"]):
        - image_base64: Base64 encoded input image
        - model: "gen3c" or "hunyuan3d"
        - params: Model-specific parameters
    
    Output:
        - For GEN3C: {"video_base64": "...", "status": "success"}
        - For Hunyuan3D: {"mesh_base64": "...", "status": "success"}
    """
    # Load model (cached after first call)
    # Process input
    # Return output
    pass

runpod.serverless.start({"handler": handler})
```

### 5.2 Endpoint Configuration

1. Navigate to Serverless → Endpoints → New Endpoint
2. Configure:
   - **Name**: `gen3c-inference`
   - **Docker Image**: Your image from Docker Hub
   - **GPU Type**: A100 80GB (or H100 for faster)
   - **Max Workers**: 1-3 (based on budget)
   - **Idle Timeout**: 5-30 seconds
   - **Execution Timeout**: 3600 seconds (1 hour for GEN3C)

### 5.3 Endpoint Invocation

```bash
curl -X POST "https://api.runpod.ai/v2/{endpoint_id}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_base64": "...",
      "model": "gen3c",
      "params": {
        "guidance": 1.0,
        "frames": 121,
        "trajectory": "left"
      }
    }
  }'
```

---

## Phase 6: Web UI Modifications

### 6.1 Backend Selection Logic

Modify `2d3d.py` to support multiple backends:

1. **Local** (current): Run on searidge cluster via Ray
2. **RunPod Pod**: SSH/API to running pod
3. **RunPod Serverless**: API call to endpoint

### 6.2 API Client Implementation

Create a new module `runpod_client.py`:

1. **Authentication**: Store API key securely
2. **Job Submission**: Send image + params to endpoint
3. **Status Polling**: Check job status until complete
4. **Result Retrieval**: Download video/mesh when done
5. **Error Handling**: Timeout, failures, retries

### 6.3 UI Changes

Add to Gradio interface:

1. **Backend Selector**: Dropdown (Local/RunPod Pod/RunPod Serverless)
2. **RunPod Status**: Connection status indicator
3. **Cost Estimate**: Show estimated cost before running
4. **Progress**: Poll and display job progress

---

## Phase 7: Implementation Checklist

### 7.1 Docker Image Development

- [ ] Create `Dockerfile.gen3c` for GEN3C inference
- [ ] Create `Dockerfile.hunyuan3d` for Hunyuan3D inference
- [ ] Create `handler.py` for RunPod serverless
- [ ] Create `server.py` for pod-based API server
- [ ] Test locally with `docker build` and `docker run`
- [ ] Push to Docker Hub or RunPod registry

### 7.2 RunPod Setup

- [ ] Create RunPod account and add credits
- [ ] Generate API key
- [ ] Create network volume for checkpoints
- [ ] Upload checkpoints to network volume
- [ ] Create pod template
- [ ] Test pod deployment
- [ ] Create serverless endpoint
- [ ] Test endpoint invocation

### 7.3 Web UI Integration

- [ ] Create `runpod_client.py` module
- [ ] Add backend selection to `2d3d.py`
- [ ] Implement job submission flow
- [ ] Implement status polling
- [ ] Implement result download
- [ ] Add error handling and retries
- [ ] Test end-to-end flow

### 7.4 Optional: External Hosting

- [ ] Set up Digital Ocean droplet (if needed)
- [ ] Deploy API backend to Digital Ocean
- [ ] Set up Netlify project (if needed)
- [ ] Deploy static frontend to Netlify
- [ ] Configure DNS and SSL

---

## Phase 8: Cost Estimation

### 8.1 RunPod Pricing (as of late 2024)

| GPU | On-Demand ($/hr) | Spot ($/hr) | Serverless ($/sec) |
|-----|------------------|-------------|-------------------|
| A100 80GB | $1.89 | $1.14 | $0.00031 |
| A100 40GB | $1.64 | $0.99 | $0.00027 |
| H100 80GB | $3.89 | $2.34 | $0.00065 |
| RTX 4090 | $0.74 | $0.44 | $0.00012 |

### 8.2 Estimated Costs per Job

**GEN3C (121 frames)**:
- Local (current): 14+ hours, $0 (but unusable time)
- A100 80GB: ~15-30 minutes, ~$0.50-1.00
- H100 80GB: ~10-20 minutes, ~$0.65-1.30

**Hunyuan3D (single mesh)**:
- Local (current): ~25-45 minutes, $0
- A100 80GB: ~5-10 minutes, ~$0.15-0.30
- RTX 4090: ~10-15 minutes, ~$0.12-0.18

### 8.3 Monthly Budget Scenarios

| Usage Level | Jobs/Month | Estimated Cost |
|-------------|------------|----------------|
| Light (testing) | 10-20 | $10-30 |
| Moderate | 50-100 | $50-100 |
| Heavy | 200+ | $200+ |

---

## Phase 9: Security Considerations

### 9.1 API Key Management

- Store RunPod API key in environment variable, not in code
- Use `.env` file locally, secrets manager in production
- Rotate keys periodically

### 9.2 Data Privacy

- Images uploaded to RunPod are processed on their infrastructure
- Consider data retention policies
- For sensitive content, use private pods with encryption

### 9.3 Network Security

- Use HTTPS for all API communications
- Implement rate limiting on your API gateway
- Consider IP whitelisting for production endpoints

---

## Phase 10: Maintenance and Monitoring

### 10.1 RunPod Dashboard

- Monitor job queue and worker status
- Track spending and set budget alerts
- Review logs for failed jobs

### 10.2 Local Monitoring

- Keep Prometheus/Grafana for local Ray cluster (development)
- Add RunPod metrics to dashboards (API calls, costs)

### 10.3 Updates

- Rebuild Docker images when dependencies update
- Test new images before deploying to production
- Maintain version tags for rollback capability

---

## Appendix A: File Structure for RunPod Deployment

```
runpod/
├── gen3c/
│   ├── Dockerfile
│   ├── handler.py           # Serverless handler
│   ├── server.py            # Pod API server
│   ├── requirements.txt
│   └── scripts/
│       ├── download_checkpoints.sh
│       └── start.sh
├── hunyuan3d/
│   ├── Dockerfile
│   ├── handler.py
│   ├── server.py
│   ├── requirements.txt
│   └── scripts/
│       └── start.sh
├── shared/
│   ├── utils.py             # Shared utilities
│   └── config.py            # Configuration
└── README.md
```

---

## Appendix B: Checkpoint Files Required

### GEN3C Checkpoints (~50GB total)

```
/checkpoints/
├── Gen3C-Cosmos-7B/
│   ├── model.pt              # Main model weights (~28GB)
│   ├── config.json
│   ├── Cosmos-Tokenize1-CV8x8x8-720p/
│   │   ├── mean_std.pt
│   │   └── ...
│   └── google-t5/
│       └── t5-11b/           # T5 text encoder (~42GB)
└── nvidia/
    └── guardrail/            # Content filter (~2GB)
```

### Hunyuan3D Checkpoints (~20GB total)

```
/checkpoints/
├── tencent/
│   ├── Hunyuan3D-2/
│   │   └── ...
│   └── Hunyuan3D-2mini/
│       └── ...
```

---

## Appendix C: Expected Performance Comparison

| Metric | Local (AMD ROCm) | RunPod A100 | RunPod H100 |
|--------|------------------|-------------|-------------|
| GEN3C 121 frames | 14+ hours | 15-30 min | 10-20 min |
| GEN3C cold start | N/A | 2-5 min | 2-5 min |
| Hunyuan3D mesh | 25-45 min | 5-10 min | 3-7 min |
| VRAM available | 16GB | 80GB | 80GB |
| Warp acceleration | CPU fallback | Native CUDA | Native CUDA |

---

## Next Steps

1. **Review this document** and confirm the approach
2. **Decide on deployment model** (Pods vs Serverless)
3. **Decide on UI hosting** (Local vs Digital Ocean vs Netlify)
4. **Begin Phase 7 implementation** (Docker images first)

When ready, I will create the actual Dockerfile, handler.py, and integration code based on your decisions.

---

*Document created: December 13, 2025*
*Project: Hunyuan3D-2-Fork / GEN3C Integration*
*Author: AI Assistant*

