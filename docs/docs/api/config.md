# `src.config`

Configuration loader module. Uses [Hydra's Compose API](https://hydra.cc/docs/advanced/compose_api/) to load `config.yaml` with support for CLI overrides.

---

## Architecture

Unlike the typical `@hydra.main` decorator approach, this module uses Hydra's **Compose API** for maximum flexibility:

- `initialize_config_dir()` points Hydra at the project root
- `compose()` loads `config.yaml` and applies any overrides
- `OmegaConf.to_container()` converts the result to a plain Python dict
- The config is loaded **once at import time** and shared globally as `CONFIG`

This avoids Hydra's side-effects (output directories, working directory changes) while keeping its powerful override system.

---

## Module-Level Constants

### `PROJECT_ROOT`

```python
PROJECT_ROOT: Path
```

The absolute path to the project root directory (where `config.yaml` lives). Resolved at import time by walking up from `config.py` until `config.yaml` is found.

---

### `CONFIG`

```python
CONFIG: dict
```

The parsed configuration dictionary, loaded once at import time via `load_config()`. If CLI arguments contain Hydra-style overrides (e.g. `training.n_trials=5`), they are applied automatically.

**Example usage:**

```python
from src.config import CONFIG, resolve_path

train_csv = resolve_path(CONFIG["data"]["train_csv"])
n_trials = CONFIG["training"]["n_trials"]
```

---

## Functions

### `get_project_root`

```python
def get_project_root() -> Path
```

Walks up the directory tree from the location of `config.py` until it finds a directory containing `config.yaml`.

**Returns:**

| Type | Description |
|------|-------------|
| `Path` | Absolute path to the project root directory |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | If `config.yaml` is not found in any parent directory |

---

### `_parse_cli_overrides`

```python
def _parse_cli_overrides() -> list[str]
```

Extracts Hydra-style `key=value` overrides from `sys.argv`. This is necessary because the Compose API (unlike `@hydra.main`) does not automatically parse CLI arguments.

**Filtering logic:**

- Includes arguments that contain `=`
- Excludes arguments starting with `-` (flags like `--verbose`)

**Returns:**

| Type | Description |
|------|-------------|
| `list[str]` | List of override strings, e.g. `["training.n_trials=5", "training.cv_folds=3"]` |

**Example:**

```bash
# Running:
python -m src.train training.n_trials=5 --verbose

# _parse_cli_overrides() returns:
# ["training.n_trials=5"]
# (--verbose is excluded because it starts with "-")
```

---

### `load_config`

```python
def load_config(overrides: list[str] | None = None) -> dict
```

Loads `config.yaml` via Hydra's Compose API with optional overrides.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `overrides` | `list[str] \| None` | `None` | Hydra override strings. If `None`, auto-detects from CLI args via `_parse_cli_overrides()`. |

**Returns:**

| Type | Description |
|------|-------------|
| `dict` | Parsed YAML configuration as a nested dictionary with all overrides applied |

**Example — programmatic overrides:**

```python
from src.config import load_config

# Override programmatically (ignores CLI args)
cfg = load_config(overrides=["training.n_trials=10", "preprocessing.test_size=0.3"])
print(cfg["training"]["n_trials"])  # 10
```

**Example — automatic CLI detection:**

```python
# When called with no arguments, reads overrides from sys.argv
cfg = load_config()  # picks up CLI args automatically
```

---

### `resolve_path`

```python
def resolve_path(relative_path: str) -> Path
```

Converts a config-relative path string to an absolute path by prepending `PROJECT_ROOT`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `relative_path` | `str` | A path relative to the project root (e.g. `"data/raw/train.csv"`) |

**Returns:**

| Type | Description |
|------|-------------|
| `Path` | Absolute path |

**Example:**

```python
from src.config import resolve_path

path = resolve_path("models/best_model.pkl")
# → /absolute/path/to/mlops-lab0/models/best_model.pkl
```
