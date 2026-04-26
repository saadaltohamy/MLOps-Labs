# Getting Started

This guide walks you through setting up and running the Titanic ML pipeline from scratch.

---

## Prerequisites

- **Python 3.12+**
- A **Kaggle account** with an API key ([how to get one](https://www.kaggle.com/docs/api))
- *(Optional)* A **Weights & Biases** account for experiment tracking
- *(Optional)* An **MLflow tracking server** for Optuna trial logging

---

## 1. Clone the Repository

```bash
git clone https://github.com/saadaltohamy/mlops-lab0.git
cd mlops-lab0
```

---

## 2. Install uv (recommended)

[uv](https://docs.astral.sh/uv/) is a blazing-fast Python package manager built in Rust. Install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 3. Install Dependencies

### Option A: Using uv (recommended)

```bash
uv sync
```

This reads `pyproject.toml` + `uv.lock`, creates a `.venv`, and installs all dependencies in one step.

To activate the environment afterwards:

```bash
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### Option B: Using pip

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
pip install -e .
```

This installs all required packages defined in `pyproject.toml`, including:

- scikit-learn, XGBoost, CatBoost
- Optuna + optuna-integration[wandb]
- Hydra (configuration management)
- pandas, matplotlib, seaborn
- pytest, python-dotenv

---

## 4. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key

# Optional — wandb logging auto-enables when this key is set
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=mlops-lab0

# Optional — Optuna trials are logged to MLflow when this URI is set
MLFLOW_TRACKING_URI=http://localhost:5000
```

!!! note
    If `WANDB_API_KEY` or `MLFLOW_TRACKING_URI` is not set, the pipeline runs normally and that tracking integration is skipped with a warning.

---

## 5. Run the Full Pipeline

```bash
bash run_pipeline.sh
```

This runs all steps in sequence:

1. **Download** — fetches Titanic data from Kaggle (skips if data already exists)
2. **Preprocess** — builds sklearn pipelines, splits data, saves artifacts
3. **Validate** — runs pytest data quality checks (**stops pipeline on failure**)
4. **Train** — runs Optuna HPO (30 trials), saves best model + metrics + chart
5. **Test** — evaluates best model on validation set

---

## 6. Run Inference

After training, predict on new data:

```bash
python src/inference.py --input data/raw/test.csv --output reports/predictions.csv
```

---

## Running Individual Steps

Each module can be run independently:

```bash
python -m src.download_data       # Step 1: Download data
python -m src.preprocess           # Step 2: Preprocess
python -m pytest tests/test_data.py -v  # Step 3: Validate
python -m src.train                # Step 4: Train
python -m src.test_model           # Step 5: Test
```

### Using CLI Overrides (Hydra)

Override any config value without editing `config.yaml`:

```bash
# Quick run with 5 trials instead of 30
python -m src.train training.n_trials=5

# Override multiple values
python -m src.train training.n_trials=50 training.cv_folds=10

# Change split ratio
python -m src.preprocess preprocessing.test_size=0.3
```

See [Configuration](configuration.md) for full override syntax and examples.
