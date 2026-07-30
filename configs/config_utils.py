"""
Filename: config_utils.py
Purpose: Configuration utilities for loading, merging, and managing YAML configs

Main Components:
- load_yaml_config: Loads YAML configuration files
- merge_configs: Merges multiple configuration dictionaries
- ConfigManager: Class for managing configuration loading with environment variable support
- flatten_config: Flattens nested configuration dictionaries
- resolve_config_paths: Resolves paths with environment variable expansion
- validate_required_paths: Validates presence of required configuration paths

Dependencies: os, yaml, argparse, typing, pathlib
"""

import os
import yaml
import argparse
from typing import Dict, Any, Optional
from pathlib import Path


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries with deep merging."""
    def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    result = {}
    for config in configs:
        if config:
            result = deep_merge(result, config)
    return result


def flatten_config(config: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten nested configuration dictionary."""
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get nested configuration value using dot notation."""
    keys = key.split('.')
    value = config
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default


def resolve_config_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve configuration paths, expanding ~ and environment variables."""
    if isinstance(config, dict):
        return {k: resolve_config_paths(v) for k, v in config.items()}
    elif isinstance(config, str):
        # Expand ~ and environment variables
        return os.path.expanduser(os.path.expandvars(config))
    else:
        return config


class ConfigManager:
    """Manages configuration loading and merging for training scripts."""
    
    def __init__(self, script_name: str):
        self.script_name = script_name
        self.config = {}
        
    def load_config(self, config_path: Optional[str] = None, 
                   load_default: bool = True,
                   load_env: bool = True) -> Dict[str, Any]:
        """Load configuration from YAML files."""
        configs_to_merge = []
        
        # 1. Load env config first (lowest priority)
        if load_env:
            env_config_path = Path(__file__).parent / "env.yaml"
            if env_config_path.exists():
                configs_to_merge.append(load_yaml_config(str(env_config_path)))
        
        # 2. Load default config if requested
        if load_default:
            default_config_path = Path(__file__).parent / "default.yaml"
            if default_config_path.exists():
                configs_to_merge.append(load_yaml_config(str(default_config_path)))
        
        # 3. Load script-specific config (highest priority)
        if config_path:
            configs_to_merge.append(load_yaml_config(config_path))
        else:
            # Try to find default config for this script
            script_config_path = Path(__file__).parent / f"{self.script_name}.yaml"
            if script_config_path.exists():
                configs_to_merge.append(load_yaml_config(str(script_config_path)))
        
        # 4. Merge all configs (later configs override earlier ones)
        self.config = merge_configs(*configs_to_merge)
        
        # 5. Resolve paths
        self.config = resolve_config_paths(self.config)
        
        return self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        return get_config_value(self.config, key, default)
    
    def add_argparse_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add common configuration arguments to argument parser."""
        parser.add_argument("--config", type=str, 
                           help="Path to YAML configuration file")
        return parser
    
    def override_from_args(self, args: argparse.Namespace):
        """Override configuration values from command line arguments."""
        # This method should be customized for each script
        # to map argparse arguments to config keys
        pass


def create_config_parser() -> argparse.ArgumentParser:
    """Create base argument parser with config support."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                       help="Path to YAML configuration file")
    return parser


def validate_required_paths(config: Dict[str, Any], required_paths: list) -> None:
    """Validate that required paths are present in configuration."""
    missing_paths = []
    
    for path_key in required_paths:
        value = get_config_value(config, path_key)
        if not value:
            missing_paths.append(path_key)
    
    if missing_paths:
        paths_str = ", ".join(missing_paths)
        raise ValueError(f"Missing required configuration paths: {paths_str}")


def load_env_config(env_config_path: str|None = None) -> Dict[str, Any]:
    """Load environment configuration from env.yaml."""
    if env_config_path is None:
        env_config_path = Path(__file__).parent / "env.yaml"
    else:
        env_config_path = Path(env_config_path)
    if env_config_path.exists():
        return load_yaml_config(str(env_config_path))
    return {}


def get_path(config: Dict[str, Any], path_key: str, fallback_env_key: Optional[str] = None) -> Optional[str]:
    """
    Get a path from configuration with optional environment variable fallback.
    
    Args:
        config: Configuration dictionary
        path_key: Key to look for in config (e.g., 'paths.caesar_path')
        fallback_env_key: Environment variable name to use as fallback (deprecated)
    
    Returns:
        Path string or None if not found
    """
    # First try to get from config
    path = get_config_value(config, path_key)
    if path:
        return path
    
    # Fallback to environment variable (for backward compatibility)
    if fallback_env_key:
        return os.getenv(fallback_env_key)
    
    return None