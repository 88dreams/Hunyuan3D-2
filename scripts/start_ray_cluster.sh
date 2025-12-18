#!/bin/bash
# =============================================================================
# Ray Cluster Startup Script
# Searidge Multi-System Setup
# =============================================================================
#
# This script starts Ray on the current node with proper configuration
# for metrics export (Prometheus integration).
#
# Usage:
#   On head node (searidge02):  ./start_ray_cluster.sh head
#   On worker nodes:            ./start_ray_cluster.sh worker
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# CONFIGURATION (from multi_system.yaml)
# =============================================================================
HEAD_IP="192.168.88.18"
HEAD_PORT="6380"
DASHBOARD_PORT="8265"
METRICS_EXPORT_PORT="8080"
NUM_GPUS="1"

# Conda environment
CONDA_ENV="gen3c-rocm"

# =============================================================================
# FUNCTIONS
# =============================================================================

activate_conda() {
    # Try different conda locations
    if [[ -f "$HOME/opt/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/opt/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]]; then
        source "$HOME/mambaforge/etc/profile.d/conda.sh"
    elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    else
        log_error "Could not find conda installation"
        exit 1
    fi
    
    conda activate "$CONDA_ENV"
    log_info "Activated conda environment: $CONDA_ENV"
}

stop_ray() {
    log_info "Stopping any existing Ray processes..."
    ray stop --force 2>/dev/null || true
    sleep 2
}

start_head() {
    log_info "Starting Ray HEAD node..."
    log_info "  Port: $HEAD_PORT"
    log_info "  Dashboard: http://$HEAD_IP:$DASHBOARD_PORT"
    log_info "  Metrics: http://$HEAD_IP:$METRICS_EXPORT_PORT"
    
    ray start \
        --head \
        --port="$HEAD_PORT" \
        --num-gpus="$NUM_GPUS" \
        --dashboard-host="0.0.0.0" \
        --dashboard-port="$DASHBOARD_PORT" \
        --metrics-export-port="$METRICS_EXPORT_PORT" \
        --include-dashboard=true
    
    echo ""
    log_info "Ray HEAD node started successfully!"
    echo ""
    echo "=============================================="
    echo "Ray Cluster Endpoints:"
    echo "  Dashboard:     http://$HEAD_IP:$DASHBOARD_PORT"
    echo "  Metrics:       http://$HEAD_IP:$METRICS_EXPORT_PORT/metrics"
    echo "  Connect:       ray.init(address='$HEAD_IP:$HEAD_PORT')"
    echo ""
    echo "Prometheus SD:   http://$HEAD_IP:$DASHBOARD_PORT/api/prometheus/sd"
    echo "=============================================="
}

start_worker() {
    log_info "Starting Ray WORKER node..."
    log_info "  Connecting to: $HEAD_IP:$HEAD_PORT"
    log_info "  Metrics port: $METRICS_EXPORT_PORT"
    
    ray start \
        --address="$HEAD_IP:$HEAD_PORT" \
        --num-gpus="$NUM_GPUS" \
        --metrics-export-port="$METRICS_EXPORT_PORT"
    
    echo ""
    log_info "Ray WORKER node started successfully!"
    echo ""
    echo "=============================================="
    echo "Worker connected to cluster at $HEAD_IP:$HEAD_PORT"
    echo "Local metrics: http://$(hostname -I | awk '{print $1}'):$METRICS_EXPORT_PORT/metrics"
    echo "=============================================="
}

show_status() {
    log_info "Checking Ray status..."
    ray status 2>/dev/null || log_warn "Ray not running or not connected"
}

# =============================================================================
# MAIN
# =============================================================================

print_usage() {
    echo "Usage: $0 [head|worker|stop|status]"
    echo ""
    echo "Commands:"
    echo "  head    Start as head node (run on searidge02)"
    echo "  worker  Start as worker node (run on searidge01/03)"
    echo "  stop    Stop Ray on this node"
    echo "  status  Show Ray cluster status"
    echo ""
}

# Activate conda first
activate_conda

case "${1:-}" in
    head)
        stop_ray
        start_head
        ;;
    worker)
        stop_ray
        start_worker
        ;;
    stop)
        stop_ray
        log_info "Ray stopped"
        ;;
    status)
        show_status
        ;;
    *)
        print_usage
        exit 1
        ;;
esac

