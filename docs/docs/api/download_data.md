# `src.download_data`

Data download module. Downloads the Titanic dataset from Kaggle via `kagglehub`, or skips if the data already exists locally.

---

## Usage

```bash
python src/download_data.py
```

---

## Functions

### `download_data`

```python
def download_data() -> None
```

Downloads the Titanic competition data from Kaggle if `train.csv` and `test.csv` are not already present in the configured `data.raw_dir`.

**Behavior:**

1. Checks if `data/raw/train.csv` and `data/raw/test.csv` exist
2. If both exist → logs a message and returns (skips download)
3. If not → reads `KAGGLE_KEY` from environment, sets the API token, and calls `kagglehub.competition_download("titanic", ...)`

**Environment Variables Required:**

| Variable | Description |
|----------|-------------|
| `KAGGLE_KEY` | Kaggle API key (loaded from `.env`) |

**Config Keys Used:**

| Key | Description |
|-----|-------------|
| `data.raw_dir` | Directory to save downloaded files |
| `data.train_csv` | Expected path of train.csv |
| `data.test_csv` | Expected path of test.csv |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `EnvironmentError` | If `KAGGLE_KEY` is not found in the environment |

**Example:**

```python
from src.download_data import download_data

download_data()
# Logs: "Data already exists at data/raw. Skipping download."
# — or —
# Logs: "Downloading Titanic dataset to data/raw ..."
```
