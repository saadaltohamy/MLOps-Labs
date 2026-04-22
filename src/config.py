"""
Loads config.yaml and provides a simple accessor for all pipeline settings.
All paths are resolved relative to the project root.
"""

import os
from pathlib import Path

import yaml


def get_project_root() -> Path:
    """Return the project root directory (where config.yaml lives)."""
    # Walk up from this file until we find config.yaml
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "config.yaml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find config.yaml in any parent directory.")


PROJECT_ROOT = get_project_root()


def load_config() -> dict:
    """Load the YAML configuration file and resolve all paths."""
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(relative_path: str) -> Path:
    """Resolve a config-relative path to an absolute path."""
    return PROJECT_ROOT / relative_path


# Load config once at import time
CONFIG = load_config()
