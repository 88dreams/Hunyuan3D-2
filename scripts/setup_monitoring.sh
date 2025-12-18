#!/bin/bash
# =============================================================================
# Prometheus + Grafana Setup for Ray Cluster Monitoring
# Searidge Multi-System Setup
# =============================================================================
# 
# This script installs and configures Prometheus and Grafana on the head node
# (searidge02) for monitoring the Ray cluster.
#
# Run this script on searidge02 only.
#
# Usage: ./scripts/setup_monitoring.sh [install|start|stop|status]
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config/monitoring"

# Monitoring data directories (on shared storage)
MONITORING_BASE="/srv/searidge_share/monitoring"
PROMETHEUS_DATA="$MONITORING_BASE/prometheus"
GRAFANA_DATA="$MONITORING_BASE/grafana"

# Versions (update as needed)
PROMETHEUS_VERSION="2.54.1"
GRAFANA_VERSION="11.3.0"

# Installation directories
INSTALL_DIR="$HOME/monitoring"
PROMETHEUS_DIR="$INSTALL_DIR/prometheus"
GRAFANA_DIR="$INSTALL_DIR/grafana"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# INSTALLATION
# =============================================================================
install_prometheus() {
    log_info "Installing Prometheus ${PROMETHEUS_VERSION}..."
    
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    
    # Download if not already present
    PROM_TARBALL="prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz"
    if [[ ! -f "$PROM_TARBALL" ]]; then
        log_info "Downloading Prometheus..."
        wget -q "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/${PROM_TARBALL}"
    fi
    
    # Extract
    log_info "Extracting Prometheus..."
    tar xzf "$PROM_TARBALL"
    rm -rf "$PROMETHEUS_DIR"
    mv "prometheus-${PROMETHEUS_VERSION}.linux-amd64" "$PROMETHEUS_DIR"
    
    # Create data directory
    sudo mkdir -p "$PROMETHEUS_DATA"
    sudo chown -R $(whoami):$(whoami) "$PROMETHEUS_DATA"
    
    # Copy config
    cp "$CONFIG_DIR/prometheus.yml" "$PROMETHEUS_DIR/prometheus.yml"
    
    log_info "Prometheus installed to $PROMETHEUS_DIR"
}

install_grafana() {
    log_info "Installing Grafana ${GRAFANA_VERSION}..."
    
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    
    # Download if not already present
    GRAFANA_TARBALL="grafana-${GRAFANA_VERSION}.linux-amd64.tar.gz"
    if [[ ! -f "$GRAFANA_TARBALL" ]]; then
        log_info "Downloading Grafana..."
        wget -q "https://dl.grafana.com/oss/release/${GRAFANA_TARBALL}"
    fi
    
    # Extract
    log_info "Extracting Grafana..."
    tar xzf "$GRAFANA_TARBALL"
    rm -rf "$GRAFANA_DIR"
    mv "grafana-v${GRAFANA_VERSION}" "$GRAFANA_DIR" 2>/dev/null || mv "grafana-${GRAFANA_VERSION}" "$GRAFANA_DIR"
    
    # Create data directories
    sudo mkdir -p "$GRAFANA_DATA"/{logs,plugins,provisioning/datasources,provisioning/dashboards}
    sudo chown -R $(whoami):$(whoami) "$GRAFANA_DATA"
    
    # Copy config
    cp "$CONFIG_DIR/grafana.ini" "$GRAFANA_DIR/conf/custom.ini"
    
    # Create Prometheus datasource provisioning
    cat > "$GRAFANA_DATA/provisioning/datasources/prometheus.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
    editable: false
EOF
    
    # Create dashboard provisioning config
    cat > "$GRAFANA_DATA/provisioning/dashboards/ray.yml" << EOF
apiVersion: 1

providers:
  - name: 'Ray Dashboards'
    orgId: 1
    folder: 'Ray'
    type: file
    disableDeletion: false
    editable: true
    options:
      path: $GRAFANA_DATA/dashboards
EOF
    
    # Create dashboards directory
    mkdir -p "$GRAFANA_DATA/dashboards"
    
    log_info "Grafana installed to $GRAFANA_DIR"
}

setup_ray_dashboards() {
    log_info "Setting up Ray default dashboards..."
    
    # Check if Ray has generated dashboard configs
    RAY_METRICS_DIR="/tmp/ray/session_latest/metrics"
    
    if [[ -d "$RAY_METRICS_DIR/grafana/dashboards" ]]; then
        log_info "Copying Ray default dashboards..."
        cp -r "$RAY_METRICS_DIR/grafana/dashboards/"* "$GRAFANA_DATA/dashboards/" 2>/dev/null || true
    else
        log_warn "Ray metrics directory not found. Start Ray first, then run:"
        log_warn "  cp -r /tmp/ray/session_latest/metrics/grafana/dashboards/* $GRAFANA_DATA/dashboards/"
    fi
}

# =============================================================================
# SERVICE MANAGEMENT
# =============================================================================
start_prometheus() {
    log_info "Starting Prometheus..."
    
    if pgrep -f "prometheus.*config.file" > /dev/null; then
        log_warn "Prometheus is already running"
        return
    fi
    
    cd "$PROMETHEUS_DIR"
    nohup ./prometheus \
        --config.file="$PROMETHEUS_DIR/prometheus.yml" \
        --storage.tsdb.path="$PROMETHEUS_DATA" \
        --storage.tsdb.retention.time=15d \
        --web.listen-address="0.0.0.0:9090" \
        --web.enable-lifecycle \
        > "$PROMETHEUS_DATA/prometheus.log" 2>&1 &
    
    sleep 2
    if pgrep -f "prometheus.*config.file" > /dev/null; then
        log_info "Prometheus started on http://$(hostname -I | awk '{print $1}'):9090"
    else
        log_error "Failed to start Prometheus. Check $PROMETHEUS_DATA/prometheus.log"
    fi
}

stop_prometheus() {
    log_info "Stopping Prometheus..."
    pkill -f "prometheus.*config.file" 2>/dev/null || true
    log_info "Prometheus stopped"
}

start_grafana() {
    log_info "Starting Grafana..."
    
    if pgrep -f "grafana.*server" > /dev/null; then
        log_warn "Grafana is already running"
        return
    fi
    
    cd "$GRAFANA_DIR"
    nohup ./bin/grafana server \
        --config="$GRAFANA_DIR/conf/custom.ini" \
        --homepath="$GRAFANA_DIR" \
        > "$GRAFANA_DATA/logs/grafana-startup.log" 2>&1 &
    
    sleep 3
    if pgrep -f "grafana.*server" > /dev/null; then
        log_info "Grafana started on http://$(hostname -I | awk '{print $1}'):3000"
        log_info "Login: admin / searidge_admin"
    else
        log_error "Failed to start Grafana. Check $GRAFANA_DATA/logs/grafana-startup.log"
    fi
}

stop_grafana() {
    log_info "Stopping Grafana..."
    pkill -f "grafana.*server" 2>/dev/null || true
    log_info "Grafana stopped"
}

show_status() {
    echo "=============================================="
    echo "Monitoring Services Status"
    echo "=============================================="
    
    echo -n "Prometheus: "
    if pgrep -f "prometheus.*config.file" > /dev/null; then
        echo -e "${GREEN}Running${NC} (http://192.168.88.18:9090)"
    else
        echo -e "${RED}Stopped${NC}"
    fi
    
    echo -n "Grafana:    "
    if pgrep -f "grafana.*server" > /dev/null; then
        echo -e "${GREEN}Running${NC} (http://192.168.88.18:3000)"
    else
        echo -e "${RED}Stopped${NC}"
    fi
    
    echo ""
    echo "Ray Dashboard: http://192.168.88.18:8265"
    echo "=============================================="
}

# =============================================================================
# SYSTEMD SERVICE FILES (Optional)
# =============================================================================
create_systemd_services() {
    log_info "Creating systemd service files..."
    
    # Prometheus service
    sudo tee /etc/systemd/system/prometheus.service > /dev/null << EOF
[Unit]
Description=Prometheus Monitoring System
Wants=network-online.target
After=network-online.target

[Service]
User=$(whoami)
Group=$(whoami)
Type=simple
ExecStart=$PROMETHEUS_DIR/prometheus \\
    --config.file=$PROMETHEUS_DIR/prometheus.yml \\
    --storage.tsdb.path=$PROMETHEUS_DATA \\
    --storage.tsdb.retention.time=15d \\
    --web.listen-address=0.0.0.0:9090 \\
    --web.enable-lifecycle
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    # Grafana service
    sudo tee /etc/systemd/system/grafana.service > /dev/null << EOF
[Unit]
Description=Grafana Dashboard
Wants=network-online.target
After=network-online.target prometheus.service

[Service]
User=$(whoami)
Group=$(whoami)
Type=simple
WorkingDirectory=$GRAFANA_DIR
ExecStart=$GRAFANA_DIR/bin/grafana server \\
    --config=$GRAFANA_DIR/conf/custom.ini \\
    --homepath=$GRAFANA_DIR
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    log_info "Systemd services created. Enable with:"
    log_info "  sudo systemctl enable prometheus grafana"
    log_info "  sudo systemctl start prometheus grafana"
}

# =============================================================================
# MAIN
# =============================================================================
print_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  install     Install Prometheus and Grafana"
    echo "  start       Start monitoring services"
    echo "  stop        Stop monitoring services"
    echo "  restart     Restart monitoring services"
    echo "  status      Show service status"
    echo "  systemd     Create systemd service files"
    echo "  dashboards  Copy Ray dashboards to Grafana"
    echo ""
}

case "${1:-}" in
    install)
        install_prometheus
        install_grafana
        setup_ray_dashboards
        echo ""
        log_info "Installation complete!"
        log_info "Start services with: $0 start"
        ;;
    start)
        start_prometheus
        start_grafana
        show_status
        ;;
    stop)
        stop_prometheus
        stop_grafana
        show_status
        ;;
    restart)
        stop_prometheus
        stop_grafana
        sleep 2
        start_prometheus
        start_grafana
        show_status
        ;;
    status)
        show_status
        ;;
    systemd)
        create_systemd_services
        ;;
    dashboards)
        setup_ray_dashboards
        ;;
    *)
        print_usage
        ;;
esac

