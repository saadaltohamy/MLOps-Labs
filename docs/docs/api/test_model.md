# `src.test_model`

Testing module. Loads the best trained model, evaluates it on the validation set, and reports accuracy and ROC-AUC.

---

## Usage

```bash
python src/test_model.py
```

---

## Functions

### `prepare_catboost_frame`

```python
def prepare_catboost_frame(
    frame: pd.DataFrame,
    categorical_features: list[str]
) -> pd.DataFrame
```

Prepares a DataFrame for CatBoost by filling NaN values in categorical columns with `"missing"` and casting them to string type.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `frame` | `pd.DataFrame` | Input DataFrame |
| `categorical_features` | `list[str]` | Names of categorical columns to prepare |

**Returns:** A copy of the DataFrame with categorical columns cleaned.

---

### `run_testing`

```python
def run_testing() -> None
```

Main testing entry point. Evaluates the saved best model on the validation set.

**Steps:**

1. Load `X_valid` and `y_valid` from `data/processed/`
2. Load the model bundle from `models/best_model.pkl`
3. Load the preprocessing pipeline bundle (for `categorical_features`)
4. Prepare data based on model type (CatBoost requires special handling)
5. Generate predictions and predicted probabilities
6. Calculate accuracy and ROC-AUC
7. Append validation metrics to `reports/metrics.json`

**Config Keys Used:**

| Key | Description |
|-----|-------------|
| `data.processed_dir` | Directory containing processed data |
| `training.model_path` | Path to the saved model |
| `preprocessing.pipeline_path` | Path to the preprocessing pipeline pickle |
| `reports.metrics_file` | Path to the metrics JSON file |

**Metrics Output:**

The following keys are added/updated in `reports/metrics.json`:

```json
{
  "validation_accuracy": 0.7821,
  "validation_roc_auc": 0.8240
}
```

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | If the model pickle file does not exist |
