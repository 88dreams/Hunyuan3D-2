#!/usr/bin/env python3
"""
Centralized path and configuration management for multi-system deployment.

This module loads configuration from multi_system.yaml and provides
validated paths that work across all cluster nodes.

Usage:
    from config import get_config, get_path
    
    config = get_config()
    checkpoint_dir = get_path('checkpoints_dir')
"""

import os
import socket
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

# Find the config directory relative to this file
CONFIG_DIR = Path(__file__).parent.absolute()
DEFAULT_CONFIG_FILE = CONFIG_DIR / "multi_system.yaml"

# Cache for loaded configuration
_config_cache: Optional[Dict[str, Any]] = None


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


def _expand_path(path: str) -> str:
    """Expand ~ and environment variables in a path."""
    return os.path.expandvars(os.path.expanduser(path))


def _validate_path(path: str, must_exist: bool = False, create: bool = False) -> str:
    """
    Validate and optionally create a path.
    
    Args:
        path: The path to validate
        must_exist: If True, raise error if path doesn't exist
        create: If True, create directory if it doesn't exist
    
    Returns:
        The validated absolute path
    """
    expanded = _expand_path(path)
    abs_path = os.path.abspath(expanded)
    
    if create and not os.path.exists(abs_path):
        os.makedirs(abs_path, exist_ok=True)
    
    if must_exist and not os.path.exists(abs_path):
        raise ConfigurationError(f"Required path does not exist: {abs_path}")
    
    return abs_path


def load_config(config_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_file: Path to config file. Defaults to multi_system.yaml
    
    Returns:
        Configuration dictionary
    """
    global _config_cache
    
    if config_file is None:
        config_file = DEFAULT_CONFIG_FILE
    
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    _config_cache = config
    return config


def get_config(reload: bool = False) -> Dict[str, Any]:
    """
    Get the current configuration, loading if necessary.
    
    Args:
        reload: If True, force reload from file
    
    Returns:
        Configuration dictionary
    """
    global _config_cache
    
    if _config_cache is None or reload:
        load_config()
    
    return _config_cache


def get_path(key: str, create: bool = False, must_exist: bool = False) -> str:
    """
    Get a path from configuration by key name.
    
    Supported keys:
        - shared_root: /srv/searidge_share
        - projects_dir: /srv/searidge_share/projects
        - hunyuan_dir: /srv/searidge_share/projects/Hunyuan3D-2-Fork
        - gen3c_dir: /srv/searidge_share/projects/GEN3C
        - checkpoints_dir: /srv/searidge_share/checkpoints
        - hf_cache_dir: /srv/searidge_share/checkpoints/huggingface
        - gen3c_checkpoints: /srv/searidge_share/checkpoints/gen3c
        - inputs_dir: /srv/searidge_share/inputs
        - outputs_dir: /srv/searidge_share/outputs
        - hunyuan_outputs: /srv/searidge_share/outputs/hunyuan
        - gen3c_outputs: /srv/searidge_share/outputs/gen3c
        - gradio_cache: /srv/searidge_share/outputs/gradio_cache
        - log_dir: /srv/searidge_share/logs
    
    Args:
        key: The path key to look up
        create: If True, create directory if it doesn't exist
        must_exist: If True, raise error if path doesn't exist
    
    Returns:
        The validated path string
    """
    config = get_config()
    
    # Map of key names to config paths
    path_mapping = {
        # Shared storage paths
        'shared_root': ('shared_storage', 'root'),
        'projects_dir': ('shared_storage', 'projects_dir'),
        'hunyuan_dir': ('shared_storage', 'hunyuan_dir'),
        'gen3c_dir': ('shared_storage', 'gen3c_dir'),
        'checkpoints_dir': ('shared_storage', 'checkpoints_dir'),
        'hf_cache_dir': ('shared_storage', 'hf_cache_dir'),
        'gen3c_checkpoints': ('shared_storage', 'gen3c_checkpoints'),
        'inputs_dir': ('shared_storage', 'inputs_dir'),
        'outputs_dir': ('shared_storage', 'outputs_dir'),
        'hunyuan_outputs': ('shared_storage', 'hunyuan_outputs'),
        'gen3c_outputs': ('shared_storage', 'gen3c_outputs'),
        'gradio_cache': ('shared_storage', 'gradio_cache'),
        
        # Logging
        'log_dir': ('logging', 'log_dir'),
        
        # Local paths
        'mambaforge_base': ('local', 'mambaforge_base'),
        
        # Model paths
        'gen3c_checkpoint_dir': ('models', 'gen3c', 'checkpoint_dir'),
    }
    
    if key not in path_mapping:
        raise ConfigurationError(f"Unknown path key: {key}")
    
    # Navigate the config dict
    keys = path_mapping[key]
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            raise ConfigurationError(f"Configuration key not found: {'.'.join(keys)}")
    
    return _validate_path(value, must_exist=must_exist, create=create)


def get_current_node() -> Dict[str, str]:
    """
    Determine which cluster node we're running on.
    
    Returns:
        Dictionary with 'hostname', 'role', 'user', 'ip' for current node
    """
    config = get_config()
    hostname = socket.gethostname()
    
    # Check if we're the head node
    head = config['cluster']['head_node']
    if hostname == head['hostname'] or hostname.startswith(head['hostname']):
        return {
            'hostname': head['hostname'],
            'role': head['role'],
            'user': head['user'],
            'ip': head['ip'],
            'is_head': True
        }
    
    # Check workers
    for worker in config['cluster']['workers']:
        if hostname == worker['hostname'] or hostname.startswith(worker['hostname']):
            return {
                'hostname': worker['hostname'],
                'role': worker['role'],
                'user': worker['user'],
                'ip': worker['ip'],
                'is_head': False
            }
    
    # Unknown node - return defaults
    return {
        'hostname': hostname,
        'role': 'unknown',
        'user': os.environ.get('USER', 'unknown'),
        'ip': '127.0.0.1',
        'is_head': False
    }


def get_rocm_env() -> Dict[str, str]:
    """
    Get ROCm environment variables from configuration.
    
    Returns:
        Dictionary of environment variable names and values
    """
    config = get_config()
    rocm = config.get('rocm', {})
    
    env = {}
    
    if 'hip_visible_devices' in rocm:
        env['HIP_VISIBLE_DEVICES'] = str(rocm['hip_visible_devices'])
    
    if 'pytorch_alloc_conf' in rocm:
        env['PYTORCH_ALLOC_CONF'] = rocm['pytorch_alloc_conf']
    
    if 'hsa_override_gfx_version' in rocm:
        env['HSA_OVERRIDE_GFX_VERSION'] = rocm['hsa_override_gfx_version']
    
    if 'rocm_home' in rocm:
        env['ROCM_HOME'] = rocm['rocm_home']
        env['CUDA_HOME'] = rocm['rocm_home']  # For compatibility
    
    return env


def get_ray_config() -> Dict[str, Any]:
    """
    Get Ray cluster configuration.
    
    Returns:
        Dictionary with Ray settings
    """
    config = get_config()
    return config.get('ray', {})


def is_ray_enabled() -> bool:
    """
    Check if Ray distributed execution is enabled in config.
    
    Returns:
        True if Ray is enabled, False for single-system mode
    """
    ray_config = get_ray_config()
    return ray_config.get('enabled', True)


def should_auto_connect_ray() -> bool:
    """
    Check if Ray should auto-connect on startup.
    
    Returns:
        True if auto-connect is enabled
    """
    ray_config = get_ray_config()
    return ray_config.get('auto_connect', True)


def get_model_config(model_type: str = 'hunyuan') -> Dict[str, Any]:
    """
    Get model configuration.
    
    Args:
        model_type: 'hunyuan' or 'gen3c'
    
    Returns:
        Dictionary with model settings
    """
    config = get_config()
    models = config.get('models', {})
    
    if model_type not in models:
        raise ConfigurationError(f"Unknown model type: {model_type}")
    
    return models[model_type]


def get_webui_config() -> Dict[str, Any]:
    """
    Get web UI configuration.
    
    Returns:
        Dictionary with web UI settings
    """
    config = get_config()
    return config.get('webui', {})


def get_conda_env() -> str:
    """
    Get the Conda environment name to use.
    
    Returns:
        Conda environment name string (default: gen3c-rocm310)
    """
    config = get_config()
    return config.get('local', {}).get('conda_env', 'gen3c-rocm310')


def get_python_version() -> str:
    """
    Get the Python version for the Conda environment.
    
    Returns:
        Python version string (default: 3.10)
    """
    config = get_config()
    return config.get('local', {}).get('python_version', '3.10')


def ensure_output_dirs():
    """
    Ensure all output directories exist.
    Creates them if they don't exist.
    """
    output_keys = [
        'outputs_dir',
        'hunyuan_outputs',
        'gen3c_outputs',
        'gradio_cache',
        'log_dir',
    ]
    
    for key in output_keys:
        try:
            get_path(key, create=True)
        except ConfigurationError:
            pass  # Skip if key not in config


class Config:
    """
    Convenience class for accessing configuration.
    
    Usage:
        cfg = Config()
        print(cfg.hunyuan_dir)
        print(cfg.checkpoints_dir)
    """
    
    def __init__(self, config_file: Optional[str] = None):
        if config_file:
            load_config(config_file)
        else:
            get_config()
    
    @property
    def shared_root(self) -> str:
        return get_path('shared_root')
    
    @property
    def projects_dir(self) -> str:
        return get_path('projects_dir')
    
    @property
    def hunyuan_dir(self) -> str:
        return get_path('hunyuan_dir')
    
    @property
    def gen3c_dir(self) -> str:
        return get_path('gen3c_dir')
    
    @property
    def checkpoints_dir(self) -> str:
        return get_path('checkpoints_dir')
    
    @property
    def hf_cache_dir(self) -> str:
        return get_path('hf_cache_dir')
    
    @property
    def gen3c_checkpoints(self) -> str:
        return get_path('gen3c_checkpoints')
    
    @property
    def inputs_dir(self) -> str:
        return get_path('inputs_dir')
    
    @property
    def outputs_dir(self) -> str:
        return get_path('outputs_dir')
    
    @property
    def hunyuan_outputs(self) -> str:
        return get_path('hunyuan_outputs', create=True)
    
    @property
    def gen3c_outputs(self) -> str:
        return get_path('gen3c_outputs', create=True)
    
    @property
    def gradio_cache(self) -> str:
        return get_path('gradio_cache', create=True)
    
    @property
    def log_dir(self) -> str:
        return get_path('log_dir', create=True)
    
    @property
    def conda_env(self) -> str:
        return get_conda_env()
    
    @property
    def python_version(self) -> str:
        return get_python_version()
    
    @property
    def rocm_env(self) -> Dict[str, str]:
        return get_rocm_env()
    
    @property
    def ray_config(self) -> Dict[str, Any]:
        return get_ray_config()
    
    @property
    def ray_enabled(self) -> bool:
        return is_ray_enabled()
    
    @property
    def ray_auto_connect(self) -> bool:
        return should_auto_connect_ray()
    
    @property
    def current_node(self) -> Dict[str, str]:
        return get_current_node()
    
    @property
    def webui_config(self) -> Dict[str, Any]:
        return get_webui_config()


# Module-level convenience instance
_default_config: Optional[Config] = None


def get_default_config() -> Config:
    """Get or create the default Config instance."""
    global _default_config
    if _default_config is None:
        _default_config = Config()
    return _default_config


if __name__ == '__main__':
    # Test configuration loading
    print("Testing configuration loading...")
    
    cfg = Config()
    
    print(f"\nCurrent node: {cfg.current_node}")
    print(f"\nShared storage root: {cfg.shared_root}")
    print(f"Projects directory: {cfg.projects_dir}")
    print(f"Hunyuan directory: {cfg.hunyuan_dir}")
    print(f"GEN3C directory: {cfg.gen3c_dir}")
    print(f"Checkpoints: {cfg.checkpoints_dir}")
    print(f"HF Cache: {cfg.hf_cache_dir}")
    print(f"Inputs: {cfg.inputs_dir}")
    print(f"Outputs: {cfg.outputs_dir}")
    print(f"\nConda env: {cfg.conda_env}")
    print(f"Python version: {cfg.python_version}")
    print(f"\nROCm env vars: {cfg.rocm_env}")
    print(f"\nRay config: {cfg.ray_config}")

