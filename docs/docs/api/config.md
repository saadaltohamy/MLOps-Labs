# `src.config`

Configuration loader module. Loads `config.yaml` and provides global accessors for all pipeline settings.

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

The parsed configuration dictionary, loaded once at import time via `load_config()`. All pipeline modules import this to access settings.

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

### `load_config`

```python
def load_config() -> dict
```

Reads and parses the `config.yaml` file located at `PROJECT_ROOT`.

**Returns:**

| Type | Description |
|------|-------------|
| `dict` | Parsed YAML configuration as a nested dictionary |

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
