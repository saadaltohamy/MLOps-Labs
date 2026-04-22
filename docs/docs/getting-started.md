# Getting Started

This guide walks you through setting up and running the Titanic ML pipeline from scratch.

---

## Prerequisites

- **Python 3.12+**
- A **Kaggle account** with an API key ([how to get one](https://www.kaggle.com/docs/api))
- *(Optional)* A **Weights & Biases** account for experiment tracking

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
- pandas, matplotlib, seaborn
- pytest, python-dotenv, pyyaml

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
```

!!! note
    If `WANDB_API_KEY` is not set, the pipeline runs normally — wandb logging is simply skipped with a warning.

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
python src/download_data.py       # Step 1: Download data
python src/preprocess.py          # Step 2: Preprocess
python -m pytest tests/test_data.py -v  # Step 3: Validate
python src/train.py               # Step 4: Train
python src/test_model.py          # Step 5: Test
```
