#!/bin/bash
# =============================================================================
# RunPod GEN3C Pod Startup Script
# =============================================================================
# Run this after launching a new pod to set up the environment
# Usage: bash /workspace/GEN3C/startup.sh (if copied to volume)
#    or: curl -s https://raw.githubusercontent.com/.../pod_startup.sh | bash
# =============================================================================

set -e

echo "=========================================="
echo "GEN3C RunPod Startup Script"
echo "=========================================="

# -----------------------------------------------------------------------------
# Step 1: Fix checkpoint symlink (if needed)
# -----------------------------------------------------------------------------
echo ""
echo "[1/5] Checking checkpoints..."

if [ -d "/workspace/checkpoints/Gen3C-Cosmos-7B" ]; then
    echo "  ✓ Gen3C checkpoints found"
else
    echo "  ✗ ERROR: Gen3C checkpoints not found!"
    echo "    Expected at: /workspace/checkpoints/Gen3C-Cosmos-7B"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 2: Verify checkpoints
# -----------------------------------------------------------------------------
echo ""
echo "[2/5] Verifying checkpoints..."

CHECKPOINT_DIR="/workspace/checkpoints/Gen3C-Cosmos-7B"

if [ -f "$CHECKPOINT_DIR/model.pt" ]; then
    echo "  ✓ Main model found ($(du -h $CHECKPOINT_DIR/model.pt | cut -f1))"
else
    echo "  ✗ ERROR: model.pt not found"
    exit 1
fi

if [ -f "$CHECKPOINT_DIR/Cosmos-Tokenize1-CV8x8x8-720p/mean_std.pt" ]; then
    echo "  ✓ Cosmos tokenizer found"
else
    echo "  ✗ ERROR: Cosmos tokenizer not found"
    exit 1
fi

if [ -f "$CHECKPOINT_DIR/google-t5/t5-11b/pytorch_model.bin" ]; then
    echo "  ✓ T5 model found ($(du -h $CHECKPOINT_DIR/google-t5/t5-11b/pytorch_model.bin | cut -f1))"
else
    echo "  ✗ ERROR: T5 model not found"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 3: Fix GPU detection in server.py
# -----------------------------------------------------------------------------
echo ""
echo "[3/5] Fixing GPU detection..."

if [ -f "/workspace/server.py" ]; then
    if grep -q 'os.path.exists("/dev/nvidia0")' /workspace/server.py; then
        sed -i 's|os.path.exists("/dev/nvidia0")|__import__("torch").cuda.is_available()|g' /workspace/server.py
        echo "  ✓ Fixed GPU detection"
    else
        echo "  ✓ GPU detection already fixed"
    fi
else
    echo "  ✗ ERROR: server.py not found"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 4: Kill any existing server
# -----------------------------------------------------------------------------
echo ""
echo "[4/5] Stopping any existing server..."

# Try multiple methods to kill the server
fuser -k 8000/tcp 2>/dev/null || true
pkill -f "python.*server.py" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true
sleep 2
echo "  ✓ Port 8000 cleared"

# -----------------------------------------------------------------------------
# Step 5: Start the server
# -----------------------------------------------------------------------------
echo ""
echo "[5/5] Starting GEN3C server..."

cd /workspace
nohup python server.py > /workspace/server.log 2>&1 &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

# Wait for server to start
echo "  Waiting for server to start..."
sleep 5

# -----------------------------------------------------------------------------
# Verify
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="

HEALTH=$(curl -s http://localhost:8000/health 2>/dev/null || echo "FAILED")

if echo "$HEALTH" | grep -q '"status":"healthy"'; then
    echo "✓ Server is running!"
    echo ""
    echo "Health check response:"
    echo "$HEALTH" | python -m json.tool 2>/dev/null || echo "$HEALTH"
    echo ""
    echo "=========================================="
    echo "GEN3C is ready for inference!"
    echo "=========================================="
    echo ""
    echo "API Endpoints:"
    echo "  Health:   GET  http://localhost:8000/health"
    echo "  Generate: POST http://localhost:8000/generate"
    echo "  Status:   GET  http://localhost:8000/status/{job_id}"
    echo "  Download: GET  http://localhost:8000/download/{job_id}"
    echo ""
    echo "Server logs: tail -f /workspace/server.log"
else
    echo "✗ Server failed to start!"
    echo ""
    echo "Check logs:"
    echo "  cat /workspace/server.log"
    exit 1
fi

