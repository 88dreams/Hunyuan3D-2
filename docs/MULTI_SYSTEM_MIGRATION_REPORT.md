# Multi-System Migration Report
## Hunyuan3D-2-Fork + GEN3C: Single-System to Multi-Node Architecture

**Generated:** December 11, 2025  
**Prepared for:** LocalAI Multi-System Migration  
**Status:** Phase 1 Complete - Path Externalization

---

## Cluster Reference

| Role | Hostname | IP | User |
|------|----------|-----|------|
| **CORE** (head) | searidge02 | 192.168.88.18 | arkrunr02 |
| **WORKER01** | searidge01 | 192.168.88.17 | arkrunr01 |
| **WORKER02** | searidge03 | 192.168.88.19 | arkrunr03 |

**Shared Storage:** `/srv/searidge_share` (NFS mounted on all nodes)

---

## Executive Summary

This report analyzes the current state of the Hunyuan3D-2-Fork codebase and outlines the necessary steps to transition from a single-system architecture to a multi-node distributed architecture using three AMD ROCm GPU systems (searidge02, searidge01, searidge03) connected via NFS shared storage and Ray for job scheduling.

---

## 1. Current Architecture Overview

### 1.1 System Components

The project currently consists of several interconnected components:

| Component | File(s) | Purpose |
|-----------|---------|---------|
| **Primary Gradio UI** | `2d3d.py` | Main web interface for both Hunyuan3D and GEN3C backends |
| **Original Gradio App** | `gradio_app.py` | Tencent's original Hunyuan3D Gradio interface |
| **API Server** | `api_server.py` | REST API for programmatic access (Blender addon, etc.) |
| **GEN3C Launcher** | `scripts/run_gen3c.sh` | Shell wrapper for GEN3C single-image inference |
| **GEN3C ROCm Setup** | `scripts/setup_gen3c_rocm.sh` | Environment setup for AMD GPUs |
| **CLI Tool** | `3dbuild.py` | Command-line interface for batch processing |
| **Shape Generation** | `hy3dgen/shapegen/` | Core Hunyuan3D shape generation pipeline |

### 1.2 Current Single-System Assumptions

The codebase makes several assumptions that are incompatible with multi-system deployment:

#### **Hardcoded Paths** (NOW FIXED in Phase 1)
```python
# 2d3d.py - NOW uses config/multi_system.yaml
# Old: CACHE_DIR = "/home/arkrunr/.cache/huggingface/hub"
# New: Loads from config → /srv/searidge_share/checkpoints/huggingface

# run2d3d.sh - NOW uses shared paths
# Old: ENVPY="/home/arkrunr/opt/miniconda3/envs/hunyuan3d/bin/python"
# New: Uses conda run with gen3c-rocm310 environment (Python 3.10)
```

#### **Single GPU Assumptions**
```bash
# run2d3d.sh
export HIP_VISIBLE_DEVICES="0"

# run_gen3c.sh
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
```

#### **Local-Only Model Loading**
```python
# gradio_app.py, api_server.py, 2d3d.py
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    model_path,
    subfolder=subfolder,
    cache_dir=CACHE_DIR,  # Local cache only
    ...
)
```

---

## 2. Multi-System Setup (From Documentation)

Per `docs/localai-multi-system-setup.md`, the target architecture is:

### 2.1 Node Roles
| Node | Hostname | IP | User | Role | GPU |
|------|----------|-----|------|------|-----|
| CORE | searidge02 | 192.168.88.18 | arkrunr02 | Ray head, NFS server, web UI host | AMD RX 6900 XT |
| WORKER01 | searidge01 | 192.168.88.17 | arkrunr01 | Ray worker | AMD RX 6900 XT |
| WORKER02 | searidge03 | 192.168.88.19 | arkrunr03 | Ray worker | AMD RX 6900 XT |

### 2.2 Shared Infrastructure
- **NFS Mount:** `/srv/searidge_share` on all nodes
- **Project Location:** `/srv/searidge_share/projects/Hunyuan3D-2-Fork/`
- **Checkpoints:** `/srv/searidge_share/checkpoints/`
- **Inputs:** `/srv/searidge_share/inputs/`
- **Outputs:** `/srv/searidge_share/outputs/`
- **Conda Environment:** `gen3c-rocm310` (Python 3.10, identical on all nodes)
- **Ray Cluster:** Port 6380 on searidge02

### 2.3 Environment Configuration
```bash
# Required on all nodes
export HIP_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=max_split_size_mb:512
```

---

## 3. Gap Analysis

### 3.1 Path Configuration Issues

| Current Value | Required Value | Files Affected |
|---------------|----------------|----------------|
| `/home/arkrunr/.cache/huggingface/hub` | `/srv/localai_share/checkpoints/huggingface` | `2d3d.py` |
| `/home/arkrunr/GEN3C/checkpoints` | `/srv/localai_share/checkpoints` | `2d3d.py`, `scripts/run_gen3c.sh` |
| `/home/arkrunr/GEN3C` | `/srv/localai_share/projects/GEN3C` | `scripts/run_gen3c.sh`, `scripts/setup_gen3c_rocm.sh` |
| `/home/arkrunr/Hunyuan3D-2-Fork` | `/srv/localai_share/projects/Hunyuan3D-2-Fork` | `run2d3d.sh` |
| `$HOME/mambaforge` | Per-node local or shared | Environment activation |

### 3.2 Missing Multi-Node Components

| Component | Status | Description |
|-----------|--------|-------------|
| **Ray Job Launcher** | ❌ Missing | Example exists in docs but not implemented |
| **Distributed Task Queue** | ❌ Missing | No mechanism to distribute jobs across nodes |
| **Node Health Monitoring** | ❌ Missing | No visibility into worker node status |
| **Job Status Tracking** | ❌ Missing | No persistent job state across nodes |
| **Centralized Logging** | ❌ Missing | Logs scattered across nodes |
| **Output Aggregation** | ❌ Missing | No unified output collection |

### 3.3 Gradio UI Limitations

**`2d3d.py` Current Behavior:**
- Runs generation synchronously in the same process
- Blocks UI during generation (45-minute timeout)
- No job queuing or distribution
- System metrics only show local node

**Required Changes:**
- Submit jobs to Ray cluster instead of local execution
- Implement async job status polling
- Display cluster-wide resource utilization
- Support job cancellation across nodes

### 3.4 API Server Limitations

**`api_server.py` Current Behavior:**
- Single-node model worker
- In-process GPU inference
- Local file output only

**Required Changes:**
- Route requests to Ray workers
- Support distributed model loading
- Return results from any node via NFS

### 3.5 GEN3C Launcher Issues

**`scripts/run_gen3c.sh` Current Behavior:**
```bash
GEN3C_DIR=${GEN3C_DIR:-"$HOME/GEN3C"}
ENV_NAME=${ENV_NAME:-"gen3c-rocm310"}
```

**Required Changes:**
```bash
GEN3C_DIR=${GEN3C_DIR:-"/srv/searidge_share/projects/GEN3C"}
ENV_NAME=${ENV_NAME:-"gen3c-rocm310"}
```

---

## 4. Required Code Changes

### 4.1 Configuration Centralization

**COMPLETED:** `config/multi_system.yaml` has been created with:
```yaml
# Paths (actual values)
shared_storage:
  root: /srv/searidge_share
  projects_dir: /srv/searidge_share/projects
  hunyuan_dir: /srv/searidge_share/projects/Hunyuan3D-2-Fork
  gen3c_dir: /srv/searidge_share/projects/GEN3C
  checkpoints_dir: /srv/searidge_share/checkpoints
  hf_cache_dir: /srv/searidge_share/checkpoints/huggingface
  gen3c_checkpoints: /srv/searidge_share/checkpoints/gen3c
  inputs_dir: /srv/searidge_share/inputs
  outputs_dir: /srv/searidge_share/outputs
  hunyuan_outputs: /srv/searidge_share/outputs/hunyuan
  gen3c_outputs: /srv/searidge_share/outputs/gen3c

# Cluster nodes
cluster:
  head_node:
    hostname: searidge02
    ip: 192.168.88.18
    user: arkrunr02
  workers:
    - hostname: searidge01, user: arkrunr01
    - hostname: searidge03, user: arkrunr03

# Ray configuration
ray:
  head_address: "searidge02:6380"
  num_gpus_per_node: 1
  dashboard_port: 8265

# Conda environment
local:
  conda_env: gen3c-rocm310
  python_version: "3.10"
```

### 4.2 File-by-File Changes Required

#### **`2d3d.py`** ✅ PHASE 1 COMPLETE

| Line(s) | Status | Change Made |
|---------|--------|-------------|
| 21 | ✅ Done | Now loads from `config/multi_system.yaml` |
| 26 | ✅ Done | Now uses `/srv/searidge_share/checkpoints/gen3c` |
| 27 | ✅ Done | Now uses `/srv/searidge_share/outputs/hunyuan` |
| 246-406 | ⏳ Phase 2 | `run_hunyuan()` - needs Ray integration |
| 409-480 | ⏳ Phase 2 | `run_gen3c_backend()` - needs Ray integration |

**Phase 2 Functions Needed:**
- `submit_ray_job(job_type, params)` - Submit to Ray cluster
- `poll_job_status(job_id)` - Check job status
- `get_cluster_metrics()` - Aggregate metrics from all nodes

#### **`gradio_app.py`**

| Section | Current | Required Change |
|---------|---------|-----------------|
| Model loading | Local paths | NFS shared paths |
| `SAVE_DIR` | `gradio_cache` | NFS shared cache |
| Pipeline execution | In-process | Ray remote execution |

#### **`api_server.py`**

| Section | Current | Required Change |
|---------|---------|-----------------|
| `ModelWorker.__init__` | Local GPU | Ray actor with GPU resource |
| `generate()` endpoint | Sync local | Async Ray job |
| Output paths | Local `gradio_cache` | NFS shared path |

#### **`scripts/run_gen3c.sh`** ✅ PHASE 1 COMPLETE

| Line | Status | Change Made |
|------|--------|-------------|
| 13 | ✅ Done | Now defaults to `/srv/searidge_share/projects/GEN3C` |
| 16 | ✅ Done | Now defaults to `/srv/searidge_share/outputs/gen3c` |

#### **`scripts/setup_gen3c_rocm.sh`** ✅ PHASE 1 COMPLETE

| Line | Status | Change Made |
|------|--------|-------------|
| 5 | ✅ Done | Now defaults to `/srv/searidge_share/projects/GEN3C` |

#### **`run2d3d.sh`** ✅ PHASE 1 COMPLETE

| Line | Status | Change Made |
|------|--------|-------------|
| 5 | ✅ Done | Now uses `conda run -n gen3c-rocm310` |
| 13 | ✅ Done | Now uses `/srv/searidge_share/projects/Hunyuan3D-2-Fork/2d3d.py` |

#### **`3dbuild.py`**

| Section | Current | Required Change |
|---------|---------|-----------------|
| Model paths | Relative/local | NFS shared paths |
| Output paths | Local | NFS shared paths |

### 4.3 New Files to Create

#### **`config/multi_system.yaml`** ✅ CREATED
- Complete cluster configuration
- All paths for NFS shared storage
- Node definitions with hostnames, IPs, users
- ROCm and Ray settings

#### **`config/paths.py`** ✅ CREATED
- Centralized path configuration
- Environment-aware path resolution
- NFS mount validation
- Config class for easy access

#### **`config/__init__.py`** ✅ CREATED
- Module exports

#### **`tools/gen3c_ray_launcher.py`** ⏳ PHASE 2
Based on the example in `docs/localai-multi-system-setup.md`:
- Ray-based job submission
- Batch processing support
- Progress tracking
- Result aggregation

#### **`tools/ray_workers.py`** ⏳ PHASE 2
- Define Ray actors for Hunyuan3D pipeline
- Define Ray actors for GEN3C pipeline
- GPU resource management
- Model caching across jobs

#### **`tools/cluster_monitor.py`** ⏳ PHASE 3
- Ray cluster health checking
- Node status aggregation
- GPU utilization across cluster

---

## 5. Recommended Migration Steps

### Phase 1: Path Externalization (Low Risk) ✅ COMPLETE
1. ✅ Create `config/multi_system.yaml` with all path configurations
2. ✅ Create `config/paths.py` to load and validate paths
3. ✅ Update all hardcoded paths to use configuration
4. ⏳ Test on single node with NFS paths

### Phase 2: Ray Integration (Medium Risk) - NEXT
1. Create `tools/ray_workers.py` with Ray actors
2. Implement `submit_ray_job()` function
3. Update `2d3d.py` to use Ray for Hunyuan jobs
4. Update `run_gen3c.sh` to work with Ray
5. Test with 2-node cluster

### Phase 3: UI Enhancement (Medium Risk)
1. Add cluster status display to Gradio UI
2. Implement job queue visualization
3. Add job cancellation support
4. Add multi-node progress tracking

### Phase 4: API Server Update (Low Risk)
1. Update `api_server.py` to use Ray backend
2. Add async job endpoints
3. Implement job status polling
4. Test Blender addon with new backend

### Phase 5: Testing & Validation
1. End-to-end testing on 3-node cluster
2. Load testing with concurrent jobs
3. Failure recovery testing
4. Performance benchmarking

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| NFS latency affecting model loading | Medium | High | Pre-load models on each node |
| Ray job failures | Medium | Medium | Implement retry logic |
| Memory exhaustion on workers | High | High | Implement job queuing limits |
| Path configuration errors | Low | High | Validation at startup |
| Inconsistent environments | Medium | High | Automated environment sync |

---

## 7. Testing Checklist

### Pre-Migration
- [ ] Verify NFS mount on all nodes
- [ ] Verify Ray cluster connectivity
- [ ] Verify identical Conda environments
- [ ] Verify GPU access on all nodes
- [ ] Benchmark single-node performance

### Post-Migration
- [ ] Hunyuan3D job runs on any node
- [ ] GEN3C job runs on any node
- [ ] Jobs distributed across nodes
- [ ] Output files accessible from CORE
- [ ] Gradio UI shows cluster status
- [ ] API server routes to workers
- [ ] Blender addon functions correctly
- [ ] Concurrent job handling works
- [ ] Job cancellation works
- [ ] Error handling and recovery works

---

## 8. Summary

The current codebase is designed for single-system operation with hardcoded local paths and synchronous in-process execution. Migrating to the multi-system architecture documented in `localai-multi-system-setup.md` requires:

1. **Path externalization** - Move all hardcoded paths to configuration
2. **Ray integration** - Replace local execution with Ray remote jobs
3. **UI updates** - Add cluster awareness to Gradio interface
4. **API updates** - Make API server cluster-aware
5. **New tooling** - Create Ray launcher and monitoring tools

The migration can be done incrementally, with Phase 1 (path externalization) being the safest starting point that enables testing the NFS infrastructure before adding distributed execution.

---

## Appendix A: File Inventory

### Python Files
| File | Lines | Purpose | Multi-System Impact |
|------|-------|---------|---------------------|
| `2d3d.py` | 739 | Primary Gradio UI | **High** - needs Ray integration |
| `gradio_app.py` | 756 | Original Gradio UI | **High** - needs Ray integration |
| `api_server.py` | 334 | REST API | **High** - needs Ray integration |
| `3dbuild.py` | 177 | CLI tool | **Medium** - path updates |
| `minimal_demo.py` | 34 | Demo script | **Low** - path updates |
| `minimal_vae_demo.py` | 45 | VAE demo | **Low** - path updates |
| `blender_addon.py` | 353 | Blender integration | **Low** - uses API |
| `test_gradio.py` | 19 | Test script | **None** |

### Shell Scripts
| File | Lines | Purpose | Multi-System Impact |
|------|-------|---------|---------------------|
| `run2d3d.sh` | 14 | Launch 2d3d.py | **High** - path updates |
| `scripts/run_gen3c.sh` | 147 | GEN3C launcher | **High** - path updates |
| `scripts/setup_gen3c_rocm.sh` | 98 | ROCm setup | **Medium** - path updates |

### Configuration Files
| File | Purpose | Multi-System Impact |
|------|---------|---------------------|
| `requirements.txt` | Python deps | **Low** - ROCm-specific in docs |
| `setup.py` | Package setup | **None** |

---

## Appendix B: Environment Variables Reference

### Required on All Nodes
```bash
# ROCm configuration (RX 6900 XT = gfx1030)
export HIP_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=max_split_size_mb:512
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Shared storage
export SEARIDGE_SHARE=/srv/searidge_share
export HF_HOME=$SEARIDGE_SHARE/checkpoints/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
```

### Ray-Specific (searidge02 only for head start)
```bash
export RAY_ADDRESS="auto"  # or "searidge02:6380" explicitly
```

