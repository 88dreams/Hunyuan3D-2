# LocalAI Multi-System Setup Guide  
(Ubuntu 24.04.3 • ROCm 7.1.1 • Python 3.12 • PyTorch ROCm 2.9.1)

This guide documents the full working setup for the LocalAI / GEN3C / Hunyuan3D system, using:

- 3 Ubuntu 24.04.3 LTS machines
- AMD Radeon RX 6900 XT GPUs
- ROCm 7.1.1 (Radeon/Ryzen path)
- Python 3.12 (Conda)
- PyTorch 2.9.1 + ROCm 7.1.1 wheels
- Shared NFS storage
- Ray for multi-node scheduling

## Cluster Node Reference

| Role | Hostname | IP | User |
|------|----------|-----|------|
| **CORE** (head) | searidge02 | 192.168.88.18 | arkrunr02 |
| **WORKER01** | searidge01 | 192.168.88.17 | arkrunr01 |
| **WORKER02** | searidge03 | 192.168.88.19 | arkrunr03 |

All three nodes follow the same OS / driver / Conda stack.

**Shared Storage:** `/srv/searidge_share` (NFS mounted on all nodes)

---

## 1. Network & Hostnames

### 1.1 Static IPs

Assign each node a static IP:

- searidge01 (WORKER01): `192.168.88.17`
- searidge02 (CORE): `192.168.88.18`
- searidge03 (WORKER02): `192.168.88.19`

Example Netplan on each node (adjust interface name and IP):

```yaml
# /etc/netplan/01-searidge.yaml
network:
  version: 2
  ethernets:
    eno1:
      addresses: [192.168.88.18/24]   # searidge02 (CORE)
      gateway4: 192.168.88.1
      nameservers:
        addresses: [1.1.1.1,8.8.8.8]
```

On searidge01 / searidge03 use `.11` / `.12` instead of `.10`.

Apply:

```bash
sudo netplan apply
```

### 1.2 /etc/hosts on all nodes

Add:

```text
192.168.88.17 searidge01
192.168.88.18 searidge02
192.168.88.19 searidge03
```

---

## 2. SSH Setup

From searidge02 (as user `arkrunr02`):

```bash
ssh-keygen -t ed25519 -C "arkrunr02@searidge02"
ssh-copy-id arkrunr01@searidge01
ssh-copy-id arkrunr03@searidge03
```

On each node ensure:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

You should be able to:

```bash
ssh arkrunr01@searidge01
ssh arkrunr03@searidge03
```

without a password.

---

## 3. Shared Storage (NFS)

Shared directory (on all nodes) will be:

- `/srv/searidge_share`

### 3.1 On searidge02 (NFS server)

```bash
sudo apt update
sudo apt install -y nfs-kernel-server nfs-common

sudo mkdir -p /srv/searidge_share
sudo chown $USER:$USER /srv/searidge_share
```

`/etc/exports` on searidge02:

```text
/srv/searidge_share 192.168.88.0/24(rw,sync,no_subtree_check)
```

Apply and restart:

```bash
sudo exportfs -ar
sudo systemctl restart nfs-kernel-server
```

### 3.2 On searidge01 and searidge03 (NFS clients only)

```bash
sudo apt update
sudo apt install -y nfs-common

# Make sure NFS server is *not* running here
sudo systemctl stop nfs-kernel-server nfs-server 2>/dev/null || true
sudo systemctl disable nfs-kernel-server nfs-server 2>/dev/null || true
sudo apt remove --purge -y nfs-kernel-server || true
```

Create the mount point:

```bash
sudo mkdir -p /srv/searidge_share
```

`/etc/fstab` on searidge01 and searidge03:

```text
searidge02:/srv/searidge_share /srv/searidge_share nfs defaults,_netdev 0 0
```

Mount:

```bash
sudo mount -a
mount | grep searidge_share
```

You should see:

```text
searidge02:/srv/searidge_share on /srv/searidge_share type nfs ...
```

---

## 4. AMD GPU Driver + ROCm 7.1.1

Perform this on **all three nodes** (searidge02, searidge01, searidge03).

### 4.1 Install amdgpu-install

```bash
sudo apt update
wget https://repo.radeon.com/amdgpu-install/7.1.1/ubuntu/noble/amdgpu-install_7.1.1.70101-1_all.deb
sudo apt install ./amdgpu-install_7.1.1.70101-1_all.deb
```

### 4.2 Install Graphics + ROCm usecase

```bash
sudo amdgpu-install -y --usecase=graphics,rocm
sudo reboot
```

### 4.3 Add user to render/video groups

After reboot on each node:

```bash
groups
sudo usermod -a -G render,video "$LOGNAME"
sudo reboot
```

After the second reboot:

```bash
groups   # should include 'video' and 'render'
```

### 4.4 Verify ROCm

On each node:

```bash
rocminfo | grep -E "Name:|Marketing Name:"
rocm-smi --showproductname --showuse --showtemp
```

You should see an agent corresponding to **AMD Radeon RX 6900 XT**.

---

## 5. Conda (Mambaforge) & gen3c-rocm310 Environment

We use **Mambaforge** and create a shared environment named `gen3c-rocm310` with:

- Python 3.12
- ROCm PyTorch 2.9.1
- NumPy 1.26.4
- GEN3C + Hunyuan dependencies

### 5.1 Install Mambaforge (all nodes)

On each node:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -O /tmp/mamba.sh
bash /tmp/mamba.sh -b -p /home/arkrunr02/mambaforge      # on searidge02
# bash /tmp/mamba.sh -b -p /home/arkrunr01/mambaforge    # on searidge01
# bash /tmp/mamba.sh -b -p /home/arkrunr03/mambaforge    # on searidge03

source ~/mambaforge/etc/profile.d/conda.sh
conda init bash
```

(Log out and back in, or `exec bash`.)

### 5.2 Create gen3c-rocm310 (all nodes)

On each node:

```bash
source ~/mambaforge/etc/profile.d/conda.sh
conda create -y -n gen3c-rocm310 python=3.10 pip wheel setuptools
conda activate gen3c-rocm310310
python -m pip install --upgrade pip wheel
```

---

## 6. Install PyTorch ROCm 7.1.1 (AMD wheels)

Do this on **all nodes** inside `gen3c-rocm310`.

### 6.1 Download wheels (example on searidge02)

```bash
conda activate gen3c-rocm310
mkdir -p /tmp/amd_rocm_wheels
cd /tmp/amd_rocm_wheels

wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/torch-2.9.1%2Brocm7.1.1.lw.git351ff442-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/torchvision-0.24.0%2Brocm7.1.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/triton-3.5.1%2Brocm7.1.1.gita272dfa8-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/torchaudio-2.9.0%2Brocm7.1.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl
```

Copy `/tmp/amd_rocm_wheels` to searidge01 and searidge03 (or repeat the download there).

### 6.2 Install ROCm PyTorch stack (all nodes)

On each node:

```bash
conda activate gen3c-rocm310
cd /tmp/amd_rocm_wheels

python -m pip uninstall -y torch torchvision torchaudio triton

python -m pip install \
  torch-2.9.1+rocm7.1.1.lw.git351ff442-cp312-cp312-linux_x86_64.whl \
  torchvision-0.24.0+rocm7.1.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl \
  torchaudio-2.9.0+rocm7.1.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl \
  triton-3.5.1+rocm7.1.1.gita272dfa8-cp312-cp312-linux_x86_64.whl

python -m pip install "numpy==1.26.4" --force-reinstall
```

### 6.3 Verify ROCm PyTorch

On each node:

```bash
conda activate gen3c-rocm310
python - << 'EOF'
import torch, sys
print("Python version:", sys.version.split()[0])
print("Torch version:", torch.__version__)
print("ROCm visible:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device 0 name:", torch.cuda.get_device_name(0))
EOF
```

Expected:

- Python: `3.12.x`
- Torch: `2.9.1+rocm7.1.1...`
- ROCm visible: `True`
- Device count: `1`
- Device 0 name: `AMD Radeon RX 6900 XT`

---

## 7. PyTorch allocator & ROCm env vars

Use `PYTORCH_ALLOC_CONF` (not the deprecated `PYTORCH_HIP_ALLOC_CONF`).

On each node, create `/etc/profile.d/rocm_localai.sh`:

```bash
sudo bash -c 'cat >/etc/profile.d/rocm_localai.sh' << 'EOF'
export HIP_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=max_split_size_mb:512
# Optional, if needed for specific RDNA/ROCm combinations:
# export HSA_OVERRIDE_GFX_VERSION=10.3.0
EOF
```

Reload:

```bash
source /etc/profile.d/rocm_localai.sh
```

---

## 8. Project Layout on Shared Storage

On **searidge02** (once):

```bash
mkdir -p /srv/searidge_share/projects
mkdir -p /srv/searidge_share/checkpoints
mkdir -p /srv/searidge_share/checkpoints/huggingface
mkdir -p /srv/searidge_share/checkpoints/gen3c
mkdir -p /srv/searidge_share/inputs
mkdir -p /srv/searidge_share/outputs/hunyuan
mkdir -p /srv/searidge_share/outputs/gen3c
mkdir -p /srv/searidge_share/outputs/gradio_cache
mkdir -p /srv/searidge_share/logs
```

Move or rsync existing code:

```bash
# GEN3C
rsync -a /home/arkrunr02/GEN3C/ /srv/searidge_share/projects/GEN3C/

# Hunyuan3D-2-Fork
rsync -a /home/arkrunr02/Hunyuan3D-2-Fork/ /srv/searidge_share/projects/Hunyuan3D-2-Fork/
```

Check that searidge01 and searidge03 see these paths:

```bash
ls /srv/searidge_share/projects/GEN3C
ls /srv/searidge_share/projects/Hunyuan3D-2-Fork
```

All nodes should see identical contents.

Place checkpoints:

```bash
cp -r /home/arkrunr02/GEN3C/checkpoints/* /srv/searidge_share/checkpoints/gen3c/
```

---

## 9. Final Requirements Files

### 9.1 GEN3C `requirements.txt`

Path:

```text
/srv/searidge_share/projects/GEN3C/requirements.txt
```

Content:

```text
attrs==25.1.0
better-profanity==0.7.0
boto3==1.35.99
decord==0.6.0
diffusers==0.32.2
einops==0.8.1
huggingface-hub>=0.33.5,<2.0
hydra-core==1.3.2
imageio[pyav,ffmpeg]==2.37.0
iopath==0.1.10
ipdb==0.13.13
loguru==0.7.2
mediapy==1.2.2
megatron-core==0.14.0
nltk==3.9.1
numpy==1.26.4
nvidia-ml-py==12.535.133
omegaconf==2.3.0
opencv-python==4.10.0.84
pandas==2.2.3
peft==0.14.0
pillow==11.1.0
protobuf>=5,<7
pynvml==12.0.0
pyyaml==6.0.2
retinaface-py==0.0.2
safetensors==0.5.3
scikit-image==0.25.2
sentencepiece==0.2.0
setuptools==76.0.0
termcolor==2.5.0
# torch stack is installed separately as ROCm wheels:
# torch==2.9.1+rocm7.1.1
# torchvision==0.24.0+rocm7.1.1
tqdm==4.66.5
transformers==4.49.0
warp-lang==1.7.2
openexr==3.3.5
```

### 9.2 Hunyuan3D `requirements.txt`

Path:

```text
/srv/searidge_share/projects/Hunyuan3D-2-Fork/requirements.txt
```

Content:

```text
numpy==1.26.4
ninja
pybind11

diffusers==0.32.2
einops==0.8.1
# torch / torchvision / opencv are managed by ROCm + GEN3C:
# opencv-python
# torch
# torchvision
transformers==4.49.0
omegaconf

#sentencepiece
tqdm==4.66.5

# Mesh Processing
trimesh
pymeshlab
pygltflib
xatlas
#kornia
#facexlib

# Training
accelerate
#pytorch_lightning
#scikit-learn
#scikit-image

# Demo only
gradio
fastapi
uvicorn
rembg
onnxruntime
#gevent
#geventhttpclient
```

---

## 10. Install Project Requirements (All Nodes)

On **each** of searidge02, searidge01, searidge03:

```bash
source ~/mambaforge/etc/profile.d/conda.sh
conda activate gen3c-rocm310

# Ensure ROCm torch stack is correct first
python -m pip uninstall -y torch torchvision torchaudio triton
cd /tmp/amd_rocm_wheels
python -m pip install \
  torch-2.9.1+rocm7.1.1.lw.git351ff442-cp312-cp312-linux_x86_64.whl \
  torchvision-0.24.0+rocm7.1.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl \
  torchaudio-2.9.0+rocm7.1.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl \
  triton-3.5.1+rocm7.1.1.gita272dfa8-cp312-cp312-linux_x86_64.whl

python -m pip install "numpy==1.26.4" --force-reinstall

# GEN3C deps
python -m pip install -r /srv/searidge_share/projects/GEN3C/requirements.txt

# Hunyuan deps
python -m pip install -r /srv/searidge_share/projects/Hunyuan3D-2-Fork/requirements.txt
```

Final sanity check (all nodes):

```bash
python - << 'EOF'
import numpy as np, torch, diffusers, transformers
print("NumPy:", np.__version__)
print("Torch:", torch.__version__)
print("ROCm visible:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Diffusers:", diffusers.__version__)
print("Transformers:", transformers.__version__)
EOF
```

Target:

- NumPy: `1.26.4`
- Torch: `2.9.1+rocm7.1.1...`
- ROCm visible: `True`
- Device count: `1`
- Diffusers: `0.32.2`
- Transformers: `4.49.0`

---

## 11. Ray Installation & Cluster Setup

### 11.1 Install Ray (all nodes)

On searidge02, searidge01, searidge03:

```bash
conda activate gen3c-rocm310
python -m pip install "ray[default]==2.52.1"
python -c "import ray; print(ray.__version__)"
```

### 11.2 Start Ray Head on searidge02

On searidge02:

```bash
conda activate gen3c-rocm310
ray stop

ray start \
  --head \
  --port=6380 \
  --num-gpus=1 \
  --dashboard-host=0.0.0.0
```

### 11.3 Start Ray Workers on searidge01 & searidge03

On searidge01:

```bash
conda activate gen3c-rocm310
ray stop

ray start \
  --address='searidge02:6380' \
  --num-gpus=1
```

On searidge03:

```bash
conda activate gen3c-rocm310
ray stop

ray start \
  --address='searidge02:6380' \
  --num-gpus=1
```

### 11.4 Verify Cluster from searidge02

```bash
conda activate gen3c-rocm310
python - << 'EOF'
import ray
ray.init(address="auto")
print("Cluster resources:", ray.cluster_resources())
EOF
```

You should see `'GPU': 3.0` among the resources.

---

## 12. Single-Node Test Run (Baseline)

On searidge02:

```bash
conda activate gen3c-rocm310
export HIP_VISIBLE_DEVICES=0
source /etc/profile.d/rocm_searidge.sh

cd /srv/searidge_share/projects/Hunyuan3D-2-Fork
scripts/run_gen3c.sh \
  --input /srv/searidge_share/inputs/scene.png \
  --video-name scene_core_gpu0 \
  --frames 60 \
  --guidance 1.15 \
  --extra "--num_gpus 1"
```

Ensure the output video is produced and logs show no ROCm or OOM errors.

---

## 13. Ray-Based Multi-Node Rendering (Example)

Example launcher script on searidge02:

```python
# /srv/searidge_share/projects/Hunyuan3D-2-Fork/tools/gen3c_ray_launcher.py
import os, subprocess, pathlib, ray

GEN3C_DIR = "/srv/searidge_share/projects/GEN3C"
HUNYUAN_DIR = "/srv/searidge_share/projects/Hunyuan3D-2-Fork"
WRAPPER = pathlib.Path(HUNYUAN_DIR) / "scripts" / "run_gen3c.sh"
CHECKPOINT_DIR = "/srv/searidge_share/checkpoints/gen3c"

@ray.remote(num_gpus=1)
def render_job(job):
    env = os.environ.copy()
    env["GEN3C_DIR"] = GEN3C_DIR
    env["HIP_VISIBLE_DEVICES"] = "0"

    cmd = [
        str(WRAPPER),
        "--input", job["input_image"],
        "--video-name", job["video_name"],
        "--frames", str(job.get("frames", 121)),
        "--guidance", str(job.get("guidance", 1.15)),
        "--checkpoint-dir", CHECKPOINT_DIR,
        "--extra", "--num_gpus 1",
    ]
    subprocess.run(cmd, check=True, env=env)
    return job["video_name"]

if __name__ == "__main__":
    ray.init(address="auto")

    jobs = [
        {
            "input_image": "/srv/searidge_share/inputs/scene1.png",
            "video_name": "scene1",
            "frames": 121,
            "guidance": 1.15,
        },
        {
            "input_image": "/srv/searidge_share/inputs/scene2.png",
            "video_name": "scene2",
            "frames": 121,
            "guidance": 1.15,
        },
        # add more jobs as needed
    ]

    result = ray.get([render_job.remote(j) for j in jobs])
    print("Completed videos:", result)
```

Run from searidge02:

```bash
conda activate gen3c-rocm310
cd /srv/searidge_share/projects/Hunyuan3D-2-Fork
python tools/gen3c_ray_launcher.py
```

Jobs will be scheduled across searidge02, searidge01, searidge03 GPUs.

---

## 14. Common Troubleshooting Notes

- **Pip resolver warnings** (protobuf / huggingface-hub / opencv)  
  These are expected due to mixed constraints. As long as:
  - Torch is `2.9.1+rocm7.1.1`
  - NumPy is `1.26.4`
  - GEN3C/Hunyuan run correctly  
  you can usually ignore them.

- **Numpy upgraded to 2.x**  
  If any install upgrades NumPy:
  ```bash
  python -m pip install "numpy==1.26.4" --force-reinstall
  ```

- **ROCm not visible in PyTorch**  
  Check:
  ```bash
  rocminfo
  rocm-smi --showproductname
  groups   # must include render, video
  ```
  Then re-check `torch.cuda.is_available()`.

- **NFS issues**  
  - Ensure `nfs-kernel-server` runs only on searidge02.
  - Clients use `nfs-common` and the `/etc/fstab` line pointing to searidge02.

- **Ray issues**  
  - Ensure same Ray version (`2.52.1`) on all nodes.
  - Use `ray stop` on each node before restarting.
  - Verify cluster with `ray status` and `ray.cluster_resources()`.

---

## 15. Monitoring with Prometheus & Grafana

Ray exposes metrics that can be scraped by Prometheus for monitoring cluster health, resource usage, and job performance.

### 15.1 Install Prometheus & Grafana (searidge02 only)

```bash
cd /srv/searidge_share/projects/Hunyuan3D-2-Fork
./scripts/setup_monitoring.sh install
```

This downloads and installs:
- **Prometheus** (v2.54.1) - metrics collection and storage
- **Grafana** (v11.3.0) - visualization dashboards

### 15.2 Start Ray with Metrics Export

Use the new startup script that includes `--metrics-export-port`:

**On searidge02 (head):**
```bash
./scripts/start_ray_cluster.sh head
```

**On searidge01 & searidge03 (workers):**
```bash
./scripts/start_ray_cluster.sh worker
```

### 15.3 Start Monitoring Services

```bash
./scripts/setup_monitoring.sh start
```

### 15.4 Access Dashboards

| Service     | URL                          | Credentials          |
|-------------|------------------------------|----------------------|
| Ray Dashboard | http://192.168.88.18:8265  | (none)               |
| Prometheus  | http://192.168.88.18:9090    | (none)               |
| Grafana     | http://192.168.88.18:3000    | admin / searidge_admin |

### 15.5 Import Ray Dashboards to Grafana

After Ray is running, copy the default dashboards:

```bash
./scripts/setup_monitoring.sh dashboards
```

Or manually:
```bash
cp -r /tmp/ray/session_latest/metrics/grafana/dashboards/* \
  /srv/searidge_share/monitoring/grafana/dashboards/
```

### 15.6 Verify Prometheus Targets

1. Open Prometheus: http://192.168.88.18:9090
2. Go to Status → Targets
3. You should see all 3 nodes (searidge01, 02, 03) listed under the "ray" job

### 15.7 Key Metrics to Monitor

| Metric | Description |
|--------|-------------|
| `ray_node_cpu_utilization` | CPU usage per node |
| `ray_node_mem_used` | Memory usage per node |
| `ray_node_gpus_utilization` | GPU utilization |
| `ray_tasks` | Number of running/pending tasks |
| `ray_actors` | Number of active actors |

### 15.8 Optional: Enable as System Services

To have Prometheus and Grafana start automatically on boot:

```bash
./scripts/setup_monitoring.sh systemd
sudo systemctl enable prometheus grafana
sudo systemctl start prometheus grafana
```

---

This document is the consolidated reference for reproducing the LocalAI multi-system environment on three AMD ROCm nodes:
- **searidge02** (CORE) - arkrunr02
- **searidge01** (WORKER01) - arkrunr01
- **searidge03** (WORKER02) - arkrunr03
