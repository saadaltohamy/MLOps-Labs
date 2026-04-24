# Configuration

All pipeline settings are managed via [Hydra](https://hydra.cc/) — a powerful configuration framework by Meta Research. The project uses Hydra's **Compose API** to load `config.yaml` at the project root, with full support for **CLI overrides** so you can tweak any setting without editing files.

---

## How It Works

The `src/config.py` module uses Hydra's Compose API (`initialize_config_dir` + `compose`) instead of the `@hydra.main` decorator. This allows:

- **Module-level config** — `CONFIG` is loaded once at import time and shared across all pipeline modules
- **CLI overrides** — any `key=value` argument passed on the command line automatically overrides the corresponding config value
- **No Hydra side-effects** — no `outputs/` directory, no `.hydra/` folder, no working directory changes

```python
from src.config import CONFIG, resolve_path

# Access any config value
n_trials = CONFIG["training"]["n_trials"]
train_csv = resolve_path(CONFIG["data"]["train_csv"])
```

---

## Full Configuration Reference

```yaml
data:
  raw_dir: "data/raw"                  # Directory for raw Kaggle CSVs
  processed_dir: "data/processed"      # Directory for processed train/valid splits
  train_csv: "data/raw/train.csv"      # Path to raw training CSV
  test_csv: "data/raw/test.csv"        # Path to raw test CSV

preprocessing:
  dropped_columns:                     # Columns to drop from features
    - "Cabin"
    - "PassengerId"
    - "Ticket"
    - "Name"
    - "Survived"
  target_column: "Survived"            # Target column name
  numeric_features:                    # Numeric feature columns
    - "Age"
    - "SibSp"
    - "Parch"
    - "Fare"
  categorical_features:                # Categorical feature columns
    - "Pclass"
    - "Sex"
    - "Embarked"
  test_size: 0.2                       # Validation split ratio
  random_state: 42                     # Random seed for reproducibility
  pipeline_path: "models/preprocessing_pipeline.pkl"  # Where to save fitted preprocessors

training:
  n_trials: 30                         # Number of Optuna trials
  cv_folds: 5                          # Number of cross-validation folds
  random_state: 42                     # Random seed
  model_path: "models/best_model.pkl"  # Where to save the best model
  study_name: "titanic_tree_search"    # Optuna study name

reports:
  dir: "reports"                       # Reports output directory
  metrics_file: "reports/metrics.json" # Metrics JSON output path
  optuna_top10_chart: "reports/figures/optuna_top10_accuracy.png"  # Top-10 chart path

models:
  dir: "models"                        # Models output directory
```

---

## CLI Overrides

You can override **any** config value from the command line using Hydra's `key=value` syntax — no need to edit `config.yaml`:

### Basic Overrides

```bash
# Change number of Optuna trials
python -m src.train training.n_trials=5

# Change validation split ratio
python -m src.preprocess preprocessing.test_size=0.3

# Override multiple values at once
python -m src.train training.n_trials=50 training.cv_folds=10

# Change model output path
python -m src.train training.model_path="models/experiment_v2.pkl"
```

### Nested Key Syntax

Use dot notation to reach nested config values:

```bash
# Override the raw data directory
python -m src.download_data data.raw_dir="data/custom_raw"

# Override the study name
python -m src.train training.study_name="my_custom_study"
```

### Override Lists

```bash
# Override numeric features
python -m src.preprocess 'preprocessing.numeric_features=["Age", "Fare"]'
```

---

## Override Examples by Pipeline Step

| Step | Example Command |
|------|-----------------|
| **Download** | `python -m src.download_data data.raw_dir="data/custom"` |
| **Preprocess** | `python -m src.preprocess preprocessing.test_size=0.3` |
| **Train** | `python -m src.train training.n_trials=50 training.cv_folds=10` |
| **Test** | `python -m src.test_model training.model_path="models/experiment.pkl"` |

---

## Environment Variables

Environment variables are loaded from the `.env` file at the project root. These are **separate from Hydra config** and control external service credentials.

| Variable | Required | Description |
|----------|----------|-------------|
| `KAGGLE_USERNAME` | Yes (for download) | Your Kaggle username |
| `KAGGLE_KEY` | Yes (for download) | Your Kaggle API key |
| `WANDB_API_KEY` | No | Weights & Biases API key — enables experiment tracking |
| `WANDB_PROJECT` | No | wandb project name (default: `mlops-lab0`) |

!!! tip
    Copy `.env.example` to `.env` and fill in your values:
    ```bash
    cp .env.example .env
    ```

---

## Customizing the Pipeline

### Quick experiments via CLI (no file changes needed)

```bash
# Fast trial run — 5 trials, 3 folds
python -m src.train training.n_trials=5 training.cv_folds=3

# Full production run — 100 trials, 10 folds
python -m src.train training.n_trials=100 training.cv_folds=10
```

### Permanent changes via `config.yaml`

Edit `config.yaml` directly for changes you want to persist:

```yaml
training:
  n_trials: 50  # Increase for more thorough search
```

### Add or remove features

```yaml
preprocessing:
  numeric_features: ["Age", "SibSp", "Parch", "Fare"]
  categorical_features: ["Pclass", "Sex", "Embarked"]
```

!!! warning
    If you modify feature lists, make sure the corresponding columns exist in the raw CSV, and re-run preprocessing before training.

---

## How Config Loading Works Internally

```
┌─────────────────────────────────────────────────┐
│           python -m src.train                    │
│               training.n_trials=5                │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. src/config.py is imported                    │
│  2. get_project_root() finds config.yaml         │
│  3. _parse_cli_overrides() extracts              │
│     ["training.n_trials=5"] from sys.argv        │
│  4. Hydra compose() loads config.yaml            │
│     and applies the overrides                    │
│  5. CONFIG dict is available globally            │
│                                                  │
│  Result: CONFIG["training"]["n_trials"] == 5     │
│          (instead of the default 30)             │
│                                                  │
└─────────────────────────────────────────────────┘
```

!!! note
    Config is loaded **once at import time** (`CONFIG = load_config()` at module level). All modules that import `CONFIG` share the same overridden values for that process run.
