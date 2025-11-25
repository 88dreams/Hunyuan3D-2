# GEN3C Multi-System Rendering Guide

This document explains how to scale GEN3C inference across three Dell Precision 7865 workstations equipped with AMD Ryzen Threadripper PRO 5975WX CPUs and Radeon RX 6900 XT GPUs. It covers hardware preparation, networking, software installation, and several execution patterns so you can balance throughput versus complexity.

## 0. Topology at a Glance

| Hostname (suggested) | Role | GPUs | Notes |
| --- | --- | --- | --- |
| `gen3c-head` | Head/controller, also runs jobs | 1× RX 6900 XT | Hosts the shared storage export and Ray head node |
| `gen3c-worker01` | Worker | 1× RX 6900 XT | Mirrors software stack; joins Ray cluster |
| `gen3c-worker02` | Worker | 1× RX 6900 XT | Mirrors software stack; joins Ray cluster |

All three machines should share:

- Ubuntu 22.04 LTS (kernel 6.14 already installed) with the latest BIOS/firmware.
- A 10 GbE (preferred) or at least 2.5 GbE Ethernet switch. Use Cat6 cabling and assign static IPs (e.g., `192.168.40.10/24`, `.11`, `.12`).
- An NTP source (Chrony) to keep clocks aligned for distributed tooling.

## 1. Hardware Preparation

### 1.1 Validate the Single-GPU Configuration

Because each Precision 7865 Tower ships with a 1000 W PSU, Dell only supports **one** RX 6900 XT per chassis. Confirm the existing GPU is healthy and give it as much thermal headroom as possible:

1. **Power margin**: A single RX 6900 XT can pull ~300 W. Leave ≥250 W of PSU headroom for CPU spikes, memory, and drives. Replace aging PSUs if voltage sag or coil noise appears under load.
2. **Firmware + diagnostics**: Update BIOS/iDRAC and run SupportAssist to verify the PCIe slot and GPU thermals before stressing GEN3C.
3. **Cooling**: Clean dust filters, ensure the GPU shroud is unobstructed, and add intake/exhaust fans if hotspot temps exceed 80 °C during `rocm-smi --stress`.
4. **Firmware toggles**: Even with one GPU, keep “Above 4G Decoding” and “Resizable BAR” enabled for best PCIe mapping stability.

### 1.2 Networking Checklist

1. Connect all three systems plus your router to the switch.
2. Assign static IPs via NetworkManager or netplan:

   ```yaml
   # /etc/netplan/01-gen3c.yaml
   network:
     version: 2
     ethernets:
       eno1:
         addresses: [192.168.40.10/24]
         gateway4: 192.168.40.1
         nameservers:
           addresses: [1.1.1.1, 8.8.8.8]
   ```

3. Add matching hostnames to `/etc/hosts` on each machine for fast name lookups.
4. Enable passwordless SSH between nodes:

   ```bash
   ssh-keygen -t ed25519 -C "$USER@gen3c-head"
   ssh-copy-id gen3c-worker01
   ssh-copy-id gen3c-worker02
   ```

## 2. Base OS and Driver Stack

Perform the following on every host (head + workers).

### 2.1 System Updates

```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

### 2.2 ROCm 6.1 for RDNA2

1. Add the ROCm repository:

   ```bash
   sudo wget https://repo.radeon.com/rocm/rocm.gpg.key -O /etc/apt/trusted.gpg.d/rocm.asc
   echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/61 jammy main' | \
     sudo tee /etc/apt/sources.list.d/rocm.list
   sudo apt update
   ```

2. Install the driver + HIP stack:

   ```bash
   sudo apt install -y rocm-dev rocm-hip-sdk rocminfo rocm-smi
   sudo usermod -aG video,render "$USER"
   ```

3. Reboot, then verify visibility:

   ```bash
   /opt/rocm/bin/rocminfo | grep -i gfx
   /opt/rocm/bin/rocm-smi --showproductname --showhw
   ```

4. Persist the common environment in `/etc/profile.d/rocm.sh`:

   ```bash
   echo 'export HSA_OVERRIDE_GFX_VERSION=11.0.0' | sudo tee /etc/profile.d/rocm.sh
   echo 'export HIP_VISIBLE_DEVICES=0' | sudo tee -a /etc/profile.d/rocm.sh
   echo 'export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512' | sudo tee -a /etc/profile.d/rocm.sh
   ```

Adjust `HIP_VISIBLE_DEVICES` per process at runtime when using multiple GPUs.

## 3. Shared Storage (optional but recommended)

Having checkpoints, inputs, and outputs on a shared path keeps the workers stateless. On `gen3c-head`:

```bash
sudo apt install -y nfs-kernel-server
sudo mkdir -p /srv/gen3c_share
sudo chown "$USER":"$USER" /srv/gen3c_share
```

Edit `/etc/exports`:

```
/srv/gen3c_share 192.168.40.0/24(rw,sync,no_subtree_check)
```

Apply and verify:

```bash
sudo exportfs -ar
sudo systemctl restart nfs-kernel-server
```

On each worker add to `/etc/fstab`:

```
gen3c-head:/srv/gen3c_share  /srv/gen3c_share  nfs  defaults,_netdev  0  0
```

Mount with `sudo mount -a`. Use `/srv/gen3c_share` for checkpoints, prompts, and outputs.

## 4. Software Stack

The code lives under `/home/arkrunr/Hunyuan3D-2-Fork` and the reference GEN3C repo at `/home/arkrunr/GEN3C`.

### 4.1 Conda / Mamba

Install Mambaforge on each host for reproducible environments:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -O /tmp/mamba.sh
bash /tmp/mamba.sh -b -p $HOME/mambaforge
source "$HOME/mambaforge/etc/profile.d/conda.sh"
conda init bash
```

### 4.2 Create the `gen3c-rocm` Environment

```bash
conda create -y -n gen3c-rocm python=3.10
conda activate gen3c-rocm
pip install --upgrade pip wheel setuptools
pip install -r /home/arkrunr/GEN3C/requirements-rocm.txt  # if provided
pip install -r /home/arkrunr/Hunyuan3D-2-Fork/requirements.txt
```

If a ROCm-specific requirements file is not available, install PyTorch ROCm wheels manually:

```bash
pip install torch==2.4.0+rocm61 torchvision==0.19.0+rocm61 --index-url https://download.pytorch.org/whl/rocm6.1
```

### 4.3 Cache Model Weights Once

```bash
cd /home/arkrunr/GEN3C
mkdir -p checkpoints videos
python scripts/download_checkpoints.py  # hypothetical helper; otherwise follow vendor instructions
```

Place checkpoints on `/srv/gen3c_share/checkpoints` so every node can reuse them.

## 5. Execution Patterns

Because each Precision 7865 chassis is capped at one RX 6900 XT, the easiest way to increase throughput is to run more independent prompts in parallel across the three systems. Start with a single-host validation run, then move to manual or Ray-based job sharding.

### 5.1 Option A – Single-Host Baseline (1 GPU)

Use this to smoke-test the stack on any workstation before involving the rest of the cluster.

1. Pin the process to the only GPU (it defaults to GPU 0, but keep it explicit when scripting):

   ```bash
   export HIP_VISIBLE_DEVICES=0
   ```

2. Run the wrapper with `--num_gpus 1` (the default) so logs clearly show the intent:

   ```bash
   cd /home/arkrunr/Hunyuan3D-2-Fork
   GEN3C_DIR=/home/arkrunr/GEN3C \
   HIP_VISIBLE_DEVICES=0 \
   scripts/run_gen3c.sh \
     --input /srv/gen3c_share/inputs/scene.png \
     --video-name scene_gpu0 \
     --frames 242 \
     --guidance 1.15 \
     --checkpoint-dir /srv/gen3c_share/checkpoints \
     --extra "--num_gpus 1"
   ```

Once the single-node path is stable you can repeat the same command on the other two machines (manually or via Ray) to triple throughput with minimal coordination.

### 5.2 Option B – Manual Job Sharding Across Hosts

Best for a small amount of parallelism without extra tooling.

1. Copy or symlink the shared storage to `/srv/gen3c_share`.
2. Prepare a JSONL with prompts/images (`inputs/jobs.jsonl`).
3. On each node, run a different slice:

   ```bash
   conda run -n gen3c-rocm scripts/run_gen3c.sh --input ... --extra "--num_gpus 1 --prompt '...'"
   ```

Use GNU Parallel or a simple shell loop:

```bash
parallel --sshloginfile ~/gen3c_hosts.txt \
  "/home/arkrunr/Hunyuan3D-2-Fork/scripts/run_gen3c.sh --input {1} --video-name {2} --extra \"--num_gpus 1\"" \
  ::: $(ls inputs/*.png) ::: $(ls inputs/*.png | xargs -n1 basename | sed 's/.png//')
```

Pros: zero additional software. Cons: no centralized queueing or retry logic.

### 5.3 Option C – Ray-Based Work Queue (Recommended)

Ray lets you turn the three workstations into a cohesive rendering farm with automatic scheduling, retries, and resource-aware placement.

1. **Install Ray** inside the `gen3c-rocm` environment on every host:

   ```bash
   conda activate gen3c-rocm
   pip install "ray[default]==2.27.0"
   ```

2. **Start the cluster**:

   ```bash
   # On head
   ray start --head --port=6380 --dashboard-host=0.0.0.0 --num-gpus=1

   # On each worker (including the head if you want it to schedule jobs for itself)
   ray start --address='gen3c-head:6380' --num-gpus=1
   ```

3. **Create a submission script (example)** at `/home/arkrunr/Hunyuan3D-2-Fork/tools/gen3c_ray_launcher.py`:

   ```python
   import json
   import os
   import pathlib
   import ray
   import subprocess

   GEN3C_WRAPPER = pathlib.Path("/home/arkrunr/Hunyuan3D-2-Fork/scripts/run_gen3c.sh")

   @ray.remote(num_gpus=1)
   def render_job(job):
       cmd = [
           str(GEN3C_WRAPPER),
           "--input", job["input_image"],
           "--video-name", job["video_name"],
           "--guidance", str(job.get("guidance", 1.0)),
           "--frames", str(job.get("frames", 121)),
           "--checkpoint-dir", job["checkpoint_dir"],
           "--output-dir", job["output_dir"],
           "--extra", job.get("extra_args", "--num_gpus 1"),
       ]
       env = {"GEN3C_DIR": "/home/arkrunr/GEN3C", **os.environ}
       subprocess.run(cmd, check=True, env=env)
       return job["video_name"]

   if __name__ == "__main__":
       ray.init(address="auto")
       jobs = json.load(open("/srv/gen3c_share/jobs.json"))
       pending = [render_job.remote(job) for job in jobs]
       ray.get(pending)
   ```

4. **Launch jobs**:

   ```bash
   conda run -n gen3c-rocm python tools/gen3c_ray_launcher.py
   ```

Ray will keep the queue full, retry failed jobs, and let you observe progress at `http://gen3c-head:8265`.

## 6. Runtime Checklist

1. **Before each session**:
   - `git pull` both `/home/arkrunr/Hunyuan3D-2-Fork` and `/home/arkrunr/GEN3C`.
   - Synchronize checkpoints if you are not using shared storage (`rsync -av gen3c-head:/srv/gen3c_share/checkpoints/ checkpoints/`).
   - Confirm GPUs are idle (`rocm-smi --showuse`).

2. **During execution**:
   - Monitor temperatures via `rocm-smi --showtemp`.
   - Watch memory fragmentation; if `rocblas` OOMs, lower `--num_video_frames` or increase `PYTORCH_HIP_ALLOC_CONF`.

3. **After execution**:
   - Copy outputs from `/srv/gen3c_share/videos` into `Hunyuan3D-2-Fork/assets/gen3c_outputs`.
   - Stop Ray workers (`ray stop`) if you no longer need the cluster.

## 7. Troubleshooting Quick Hits

- **HIP runtime errors**: Ensure every node sourced `/etc/profile.d/rocm.sh` or manually exported `HSA_OVERRIDE_GFX_VERSION=11.0.0`.
- **Ray worker cannot see GPUs**: Check `ray status` and confirm you started Ray inside the Conda environment after exporting `ROCR_VISIBLE_DEVICES`.
- **Slow network copies**: Jumbo frames can help. Set `mtu 9000` on all NICs if the switch supports it.
- **Checkpoint mismatch**: Mount the NFS share read-only on workers to avoid accidental edits.

Following this playbook keeps each Precision 7865’s single GPU saturated while the Ray queue (or manual sharding) ensures all three systems stay busy. Adjust the orchestration layer as your throughput needs evolve (Slurm or Kubernetes are natural upgrades once you exceed a handful of nodes).


