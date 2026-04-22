# `src.logger`

Centralized colored logger for the Titanic pipeline. All pipeline modules use this instead of `print()` for consistent, color-coded terminal output.

---

## Log Level Colors

| Level | Color | ANSI Code |
|-------|-------|-----------|
| `DEBUG` | Cyan | `\033[36m` |
| `INFO` | Green | `\033[32m` |
| `WARNING` | Yellow | `\033[33m` |
| `ERROR` | Red | `\033[31m` |
| `CRITICAL` | Bold Red | `\033[1;31m` |

---

## Output Format

```
2026-04-22 19:00:03 | INFO | preprocess | Loaded train: (891, 12), test: (418, 11)
```

Format string: `%(asctime)s | %(levelname)s | %(name)s | %(message)s`

---

## Classes

### `ColoredFormatter`

```python
class ColoredFormatter(logging.Formatter)
```

Custom `logging.Formatter` subclass that wraps the log level name and message with ANSI color escape codes based on the log level.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `COLORS` | `dict[int, str]` | Mapping of log level integers to ANSI color codes |
| `RESET` | `str` | ANSI reset code (`\033[0m`) |

**Methods:**

#### `format`

```python
def format(self, record: logging.LogRecord) -> str
```

Applies color codes to the level name and message, then delegates to the parent `Formatter.format()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `record` | `logging.LogRecord` | The log record to format |

**Returns:** Formatted log string with ANSI color codes.

---

## Functions

### `get_logger`

```python
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger
```

Create and return a configured logger with colored console output.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Logger name (typically the module name, e.g. `"train"`, `"preprocess"`) |
| `level` | `int` | `logging.INFO` | Logging level threshold |

**Returns:**

| Type | Description |
|------|-------------|
| `logging.Logger` | Configured logger instance with a `StreamHandler` writing to `sys.stdout` |

**Behavior:**

- Avoids adding duplicate handlers if called multiple times with the same name
- Sets `logger.propagate = False` to prevent duplicate messages from the root logger

**Example:**

```python
from src.logger import get_logger

logger = get_logger("train")

logger.info("Starting training...")        # Green
logger.warning("Missing optional config")  # Yellow
logger.error("File not found!")            # Red
```
