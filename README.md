# 🚢 Titanic ML Pipeline — MLOps Labs

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>
<a target="_blank" href="https://saadaltohamy.github.io/MLOps-Labs/">
    <img src="https://img.shields.io/badge/docs-MKDocs-blue?logo=readthedocs" />
</a>

A production-ready, modular machine learning pipeline for the Titanic survival prediction task.
Built as part of the **ITI MLOps Course — Labs**.

📖 **[Full Documentation →](https://saadaltohamy.github.io/MLOps-Labs/)**

---

## ✨ Features

- **Modular pipeline** — Download → Preprocess → Validate → Train → Test → Inference
- **Optuna hyperparameter optimization** across 6 model families (RF, ExtraTrees, GBT, HistGBT, XGBoost, CatBoost)
- **Weights & Biases integration** — optional experiment tracking (auto-enabled if `WANDB_API_KEY` is set)
- **Data validation tests** — `pytest` checks run after preprocessing; training only proceeds if all tests pass
- **Colored logging** — all pipeline output uses a centralized logger with color-coded log levels
- **Hydra config management** — all settings in `config.yaml` with **CLI overrides** (`training.n_trials=5`)
- **CLI inference** — predict on new CSV data with a single command

---

## 📁 Project Structure

```
MLOps-Labs/
├── config.yaml                     # Central configuration (paths, features, training params)
├── .env                            # Environment variables (Kaggle + wandb keys)
├── .env.example                    # Template for .env
├── run_pipeline.sh                 # One-command pipeline orchestration
│
├── src/                            # Pipeline source modules
│   ├── config.py                   # Hydra Compose API config loader (supports CLI overrides)
│   ├── logger.py                   # Centralized colored logger
│   ├── download_data.py            # Download data from Kaggle
│   ├── preprocess.py               # Build sklearn pipelines, split & save data
│   ├── train.py                    # Optuna HPO + model training + wandb logging
│   ├── test_model.py               # Evaluate best model on validation set
│   └── inference.py                # CLI tool for predictions on new data
│
├── tests/
│   └── test_data.py                # Data validation tests (post-preprocessing)
│
├── models/                         # Serialized artifacts (generated)
│   ├── best_model.pkl              # Best trained model
│   └── preprocessing_pipeline.pkl  # Fitted sklearn preprocessors
│
├── reports/                        # Metrics & figures (generated)
│   ├── metrics.json                # Accuracy, ROC-AUC scores
│   ├── predictions.csv             # Inference output
│   └── figures/
│       └── optuna_top10_accuracy.png
│
├── data/
│   ├── raw/                        # Original CSVs from Kaggle
│   └── processed/                  # Train/valid splits after preprocessing
│
├── notebooks/                      # Original Jupyter notebook
│   └── processing_titanic.ipynb
│
├── docs/                           # MKDocs documentation
│   ├── mkdocs.yml
│   └── docs/
│
└── pyproject.toml                  # Project metadata & dependencies
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/saadaltohamy/MLOps-Labs.git
cd MLOps-Labs
```

### 2. Install uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Install dependencies

**Using uv (recommended):**

```bash
uv sync
```

> This reads `pyproject.toml` + `uv.lock`, creates a `.venv`, and installs all dependencies in one step.

**Using pip (alternative):**

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
pip install -e .
```

### 4. Set up environment variables

Copy the example and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key

# Optional — wandb tracking is enabled automatically when this is set
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=mlops-lab0
```

> **Note:** If `WANDB_API_KEY` is not set, the pipeline runs normally — wandb logging is simply skipped.

### 5. Run the full pipeline

```bash
bash run_pipeline.sh
```

This executes all steps in order:

| Step | Command | Description |
|------|---------|-------------|
| 1 | `python -m src.download_data` | Downloads Titanic data from Kaggle (skips if already present) |
| 2 | `python -m src.preprocess` | Builds preprocessing pipelines, splits data, saves artifacts |
| 3 | `pytest tests/test_data.py -v` | Validates processed data — **pipeline stops if tests fail** |
| 4 | `python -m src.train` | Runs Optuna HPO (30 trials), saves best model + metrics + chart |
| 5 | `python -m src.test_model` | Evaluates on validation set, reports accuracy + ROC-AUC |

---

## 🔧 Running Individual Steps

You can run any step independently:

```bash
# Download data
python -m src.download_data

# Preprocess
python -m src.preprocess

# Validate data
python -m pytest tests/test_data.py -v

# Train (with default config)
python -m src.train

# Train with CLI overrides (Hydra)
python -m src.train training.n_trials=50 training.cv_folds=10

# Test
python -m src.test_model

# Inference on new data
python -m src.inference --input data/raw/test.csv --output reports/predictions.csv
```

---

## 🔮 Inference

Use the inference module to predict on any CSV file with the same schema as the Titanic dataset:

```bash
python src/inference.py --input <path_to_csv> --output <output_path>
```

**Example:**

```bash
python src/inference.py --input data/raw/test.csv --output reports/predictions.csv
```

Output CSV contains:

| PassengerId | Survived |
|-------------|----------|
| 892         | 0        |
| 893         | 1        |
| ...         | ...      |

---

## ⚙️ Configuration (Hydra)

Configuration is managed via [Hydra](https://hydra.cc/) using the **Compose API**. All settings live in [`config.yaml`](config.yaml):

```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"

preprocessing:
  dropped_columns: ["Cabin", "PassengerId", "Ticket", "Name", "Survived"]
  target_column: "Survived"
  numeric_features: ["Age", "SibSp", "Parch", "Fare"]
  categorical_features: ["Pclass", "Sex", "Embarked"]
  test_size: 0.2

training:
  n_trials: 30          # Number of Optuna trials
  cv_folds: 5           # Cross-validation folds
  model_path: "models/best_model.pkl"
```

### CLI Overrides

Override **any** config value from the command line — no file edits needed:

```bash
# Quick experiment with fewer trials
python -m src.train training.n_trials=5

# Override multiple values
python -m src.train training.n_trials=100 training.cv_folds=10

# Change split ratio for preprocessing
python -m src.preprocess preprocessing.test_size=0.3
```

Overrides use **dot notation** to reach nested keys. The override is applied on top of `config.yaml` defaults — the file itself is not modified.

---

## 🧪 Data Validation Tests

After preprocessing, the following tests run automatically:

| Test | Description |
|------|-------------|
| `test_data_files_exist` | All processed CSV files exist on disk |
| `test_preprocessing_pipeline_exists` | Pickle file with fitted preprocessors exists |
| `test_no_missing_values_after_transform` | No NaN values after pipeline transformation |
| `test_correct_shape` | Train and valid have the expected number of columns |
| `test_target_is_binary` | Target values are only 0 or 1 |
| `test_no_duplicate_rows` | Warns about feature-level duplicates (expected for Titanic) |
| `test_train_valid_split_ratio` | Split ratio matches configured `test_size` |

---

## 📊 Outputs

After a full pipeline run, you'll find:

| File | Content |
|------|---------|
| `models/best_model.pkl` | Serialized best model (with preprocessing pipeline) |
| `models/preprocessing_pipeline.pkl` | Fitted sklearn `ColumnTransformer` pipelines |
| `reports/metrics.json` | Training & validation accuracy, ROC-AUC scores |
| `reports/figures/optuna_top10_accuracy.png` | Bar chart of top 10 Optuna trials |
| `reports/predictions.csv` | Inference predictions (after running inference) |

---

## 📖 Documentation

Full API documentation is available via MKDocs:

```bash
cd docs
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **scikit-learn** | Preprocessing pipelines & ensemble models |
| **XGBoost** | Gradient boosting classifier |
| **CatBoost** | Categorical-aware boosting |
| **Optuna** | Hyperparameter optimization |
| **Hydra** | Configuration management with CLI overrides |
| **Weights & Biases** | Experiment tracking (optional) |
| **pytest** | Data validation testing |
| **MKDocs** | Project documentation |
| **kagglehub** | Kaggle data download |

---

## 📜 License

This project is part of the ITI MLOps Course curriculum.
