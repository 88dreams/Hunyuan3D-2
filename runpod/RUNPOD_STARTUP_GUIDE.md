# RunPod GEN3C Startup Guide

## Quick Start (After Pod Launches)

### 1. SSH/Terminal into the pod

### 2. Run the startup script:
```bash
bash /workspace/GEN3C/startup.sh
```

Or if the script isn't available, run these commands manually:

---

## Manual Startup Steps

### Step 1: Fix Checkpoint Path (One-time per volume)
The network volume creates a nested directory. Fix with a symlink:

```bash
# Check if symlink already exists
if [ ! -L "/workspace/checkpoints/Gen3C-Cosmos-7B" ]; then
    ln -s /workspace/checkpoints/checkpoints/Gen3C-Cosmos-7B /workspace/checkpoints/Gen3C-Cosmos-7B
    echo "Created checkpoint symlink"
else
    echo "Symlink already exists"
fi
```

### Step 2: Verify Checkpoints
```bash
ls -la /workspace/checkpoints/Gen3C-Cosmos-7B/model.pt
ls -la /workspace/checkpoints/Gen3C-Cosmos-7B/Cosmos-Tokenize1-CV8x8x8-720p/mean_std.pt
ls -la /workspace/checkpoints/Gen3C-Cosmos-7B/google-t5/t5-11b/pytorch_model.bin
```

### Step 3: Fix GPU Detection in Server (One-time per image)
```bash
sed -i 's|os.path.exists("/dev/nvidia0")|__import__("torch").cuda.is_available()|g' /workspace/server.py
```

### Step 4: Kill Any Existing Server
```bash
fuser -k 8000/tcp 2>/dev/null || \
kill $(ps aux | grep -E "python.*server|uvicorn" | grep -v grep | awk '{print $2}') 2>/dev/null || \
true
sleep 2
```

### Step 5: Start the Server
```bash
cd /workspace && python server.py &
```

### Step 6: Verify
```bash
sleep 5
curl http://localhost:8000/health
```

Expected output:
```json
{"status":"healthy","gpu":"available","checkpoint_dir":"/workspace/checkpoints/Gen3C-Cosmos-7B","checkpoint_exists":true}
```

---

## Pod Configuration Reference

| Setting | Value |
|---------|-------|
| **Container Image** | `88dreams/gen3c-runpod:latest` |
| **Container Disk** | 30 GB |
| **Volume Mount Path** | `/workspace/checkpoints` |
| **Expose HTTP Port** | 8000 |
| **Recommended GPU** | A100 80GB |
| **Region** | CA-MTL-1 (or wherever your network volume is) |

---

## Troubleshooting

### "Connection refused" on port 8000
Server isn't running. Start it:
```bash
python /workspace/server.py &
```

### "Address already in use"
Kill existing process:
```bash
fuser -k 8000/tcp 2>/dev/null || true
sleep 2
python /workspace/server.py &
```

### "Checkpoint not found"
Check symlink:
```bash
ls -la /workspace/checkpoints/
# If Gen3C-Cosmos-7B is missing, create symlink:
ln -s /workspace/checkpoints/checkpoints/Gen3C-Cosmos-7B /workspace/checkpoints/Gen3C-Cosmos-7B
```

### GPU shows "not detected"
Fix server.py and restart:
```bash
sed -i 's|os.path.exists("/dev/nvidia0")|__import__("torch").cuda.is_available()|g' /workspace/server.py
fuser -k 8000/tcp 2>/dev/null || true
python /workspace/server.py &
```

---

## Shutdown Checklist

Before terminating the pod:
1. ✅ Any generated videos downloaded?
2. ✅ No important data in `/workspace/outputs/`?
3. ✅ Network volume will persist (checkpoints safe)

The network volume persists between pod sessions. Only the container disk is lost.

