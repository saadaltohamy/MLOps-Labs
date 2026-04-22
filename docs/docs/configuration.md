# Configuration

All pipeline settings are controlled via the `config.yaml` file at the project root.

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

## Environment Variables

Environment variables are loaded from the `.env` file at the project root.

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

### Change the number of Optuna trials

Edit `config.yaml`:

```yaml
training:
  n_trials: 50  # Increase for more thorough search
```

### Change the train/validation split ratio

```yaml
preprocessing:
  test_size: 0.3  # 70/30 split instead of 80/20
```

### Add or remove features

```yaml
preprocessing:
  numeric_features: ["Age", "SibSp", "Parch", "Fare"]
  categorical_features: ["Pclass", "Sex", "Embarked"]
```

!!! warning
    If you modify feature lists, make sure the corresponding columns exist in the raw CSV, and re-run preprocessing before training.
