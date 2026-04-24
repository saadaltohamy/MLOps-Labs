"""
Loads config.yaml via Hydra's Compose API.
Supports CLI overrides: python src/train.py training.n_trials=5
All paths are resolved relative to the project root.
"""

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


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


def _parse_cli_overrides() -> list[str]:
    """Extract Hydra-style key=value overrides from sys.argv."""
    return [arg for arg in sys.argv[1:] if "=" in arg and not arg.startswith("-")]


def load_config(overrides: list[str] | None = None) -> dict:
    """Load config.yaml via Hydra with optional overrides.

    Args:
        overrides: Hydra override strings, e.g. ["training.n_trials=5"].
                   If None, auto-detects from CLI args.
    """
    if overrides is None:
        overrides = _parse_cli_overrides()

    config_dir = str(PROJECT_ROOT)
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def resolve_path(relative_path: str) -> Path:
    """Resolve a config-relative path to an absolute path."""
    return PROJECT_ROOT / relative_path


# Load config once at import time
CONFIG = load_config()
