# `src.train`

Training module. Runs Optuna hyperparameter optimization across 6 model families, saves the best model, generates metrics and charts, and optionally logs Optuna trials to Weights & Biases and MLflow.

---

## Usage

```bash
python src/train.py
```

---

## Constants

### `MODEL_NAMES`

```python
MODEL_NAMES: list[str] = [
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "xgboost",
    "catboost",
]
```

The list of model families that Optuna searches across.

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

### `sample_model_params`

```python
def sample_model_params(trial: optuna.Trial) -> dict
```

Samples hyperparameters for a given Optuna trial. First selects a model family via `trial.suggest_categorical`, then samples family-specific hyperparameters.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `trial` | `optuna.Trial` | The current Optuna trial object |

**Returns:**

| Type | Description |
|------|-------------|
| `dict` | Dictionary containing `"model_name"` and all sampled hyperparameters |

**Hyperparameter Ranges by Model:**

| Model | Key Parameters |
|-------|---------------|
| Random Forest | `n_estimators` (200–700), `max_depth` (3–16), `min_samples_split`, `min_samples_leaf`, `max_features` |
| Extra Trees | Same ranges as Random Forest |
| Gradient Boosting | `n_estimators` (100–500), `learning_rate` (0.01–0.2), `max_depth` (2–6), `subsample` (0.6–1.0) |
| Hist Gradient Boosting | `max_iter` (150–600), `learning_rate` (0.01–0.2), `max_leaf_nodes` (15–63), `l2_regularization` |
| XGBoost | `n_estimators` (150–700), `learning_rate` (0.01–0.2), `max_depth` (3–10), `reg_alpha`, `reg_lambda` |
| CatBoost | `iterations` (200–800), `learning_rate` (0.01–0.2), `depth` (4–10), `l2_leaf_reg`, `bagging_temperature` |

---

### `build_model_from_params`

```python
def build_model_from_params(
    params: dict,
    onehot_preprocessor: ColumnTransformer,
    hist_preprocessor: ColumnTransformer,
    scale_pos_weight: float,
    hist_categorical_feature_idx: list[int],
    categorical_features: list[str],
    X_train: pd.DataFrame,
    X_train_catboost: pd.DataFrame,
) -> tuple
```

Constructs a full estimator (with preprocessing pipeline) from sampled parameters.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `params` | `dict` | Hyperparameters from `sample_model_params()` |
| `onehot_preprocessor` | `ColumnTransformer` | Fitted OneHotEncoder-based preprocessor |
| `hist_preprocessor` | `ColumnTransformer` | Fitted OrdinalEncoder-based preprocessor |
| `scale_pos_weight` | `float` | Class imbalance ratio for XGBoost |
| `hist_categorical_feature_idx` | `list[int]` | Categorical column indices for HistGBT |
| `categorical_features` | `list[str]` | Categorical column names for CatBoost |
| `X_train` | `pd.DataFrame` | Training features (standard) |
| `X_train_catboost` | `pd.DataFrame` | Training features (CatBoost-prepared) |

**Returns:**

| Type | Description |
|------|-------------|
| `tuple[estimator, DataFrame, dict]` | `(model_or_pipeline, X_to_use, fit_kwargs)` |

**Model-to-Preprocessor Mapping:**

| Model | Preprocessor | Wrapping |
|-------|-------------|----------|
| Random Forest | `onehot_preprocessor` | `sklearn.Pipeline` |
| Extra Trees | `onehot_preprocessor` | `sklearn.Pipeline` |
| Gradient Boosting | `onehot_preprocessor` | `sklearn.Pipeline` |
| Hist Gradient Boosting | `hist_preprocessor` | `sklearn.Pipeline` |
| XGBoost | `onehot_preprocessor` | `sklearn.Pipeline` |
| CatBoost | None (native) | Standalone `CatBoostClassifier` |

---

### `_init_wandb_if_available`

```python
def _init_wandb_if_available() -> wandb | None
```

Checks if `WANDB_API_KEY` is set in the environment. If so, calls `wandb.login()` and returns the `wandb` module. If not, logs a warning and returns `None`.

**Returns:**

| Type | Description |
|------|-------------|
| `wandb` module or `None` | The wandb module if available, else `None` |

---

### `_init_mlflow_if_available`

```python
def _init_mlflow_if_available(study_name: str) -> MLflowCallback | None
```

Checks if `MLFLOW_TRACKING_URI` is set in the environment. If so, configures MLflow tracking and returns an Optuna `MLflowCallback`. If not, logs a warning and returns `None`.

**Returns:**

| Type | Description |
|------|-------------|
| `MLflowCallback` or `None` | The Optuna MLflow callback if enabled, else `None` |

---

### `run_training`

```python
def run_training() -> None
```

Main training entry point. Orchestrates the full Optuna optimization workflow.

**Steps:**

1. Load processed data (`X_train`, `y_train`) from `data/processed/`
2. Load preprocessing pipelines from pickle
3. Initialize wandb if API key is available
4. Initialize MLflow if `MLFLOW_TRACKING_URI` is available
5. Define the Optuna objective function (5-fold StratifiedKFold, ROC-AUC metric)
6. Run Optuna study with `n_trials` from config
7. Retrain the best model on the full training set
8. Save model bundle to pickle
9. Save metrics to `reports/train_metrics.json`
10. Generate top-10 trials bar chart

**Config Keys Used:**

| Key | Description |
|-----|-------------|
| `training.n_trials` | Number of Optuna trials |
| `training.cv_folds` | Number of cross-validation folds |
| `training.random_state` | Random seed |
| `training.model_path` | Where to save the best model |
| `training.study_name` | Optuna study name |

**Output Files:**

| File | Content |
|------|---------|
| `models/best_model.pkl` | Serialized best model bundle |
| `reports/train_metrics.json` | Training metrics (accuracy, ROC-AUC, best params) |
| `reports/figures/optuna_top10_accuracy.png` | Top 10 trials bar chart |

**Model Bundle Contents:**

```python
{
    "model": estimator,          # Trained model (Pipeline or CatBoostClassifier)
    "model_name": str,           # e.g. "xgboost"
    "best_roc_auc_cv": float,    # Best cross-validation ROC-AUC
    "best_params": dict,         # Best hyperparameters
}
```
