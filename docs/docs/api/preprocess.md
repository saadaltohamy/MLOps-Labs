# `src.preprocess`

Preprocessing module. Loads raw data, builds sklearn preprocessing pipelines, splits data into train/validation sets, and saves all artifacts.

---

## Usage

```bash
python src/preprocess.py
```

---

## Functions

### `build_preprocessors`

```python
def build_preprocessors(
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[ColumnTransformer, ColumnTransformer]
```

Builds two sklearn `ColumnTransformer` preprocessing pipelines:

1. **`onehot_preprocessor`** — uses `SimpleImputer(median)` for numerics + `OneHotEncoder` for categoricals. Used by: Random Forest, Extra Trees, Gradient Boosting, XGBoost.
2. **`hist_preprocessor`** — uses `SimpleImputer(median)` for numerics + `OrdinalEncoder` for categoricals. Used by: Hist Gradient Boosting.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `numeric_features` | `list[str]` | Column names of numeric features (e.g. `["Age", "Fare"]`) |
| `categorical_features` | `list[str]` | Column names of categorical features (e.g. `["Sex", "Embarked"]`) |

**Returns:**

| Type | Description |
|------|-------------|
| `tuple[ColumnTransformer, ColumnTransformer]` | `(onehot_preprocessor, hist_preprocessor)` |

**Pipeline Details:**

```
onehot_preprocessor:
  ├─ num: SimpleImputer(strategy="median")
  └─ cat: SimpleImputer(strategy="most_frequent") → OneHotEncoder(handle_unknown="ignore")

hist_preprocessor:
  ├─ num: SimpleImputer(strategy="median")
  └─ cat: SimpleImputer(strategy="most_frequent") → OrdinalEncoder(handle_unknown="use_encoded_value")
```

---

### `run_preprocessing`

```python
def run_preprocessing() -> None
```

Main preprocessing pipeline entry point.

**Steps:**

1. Load raw `train.csv` and `test.csv` from paths in config
2. Drop columns specified in `preprocessing.dropped_columns`
3. Extract target column (`Survived`)
4. Split into train/validation using `train_test_split` with stratification
5. Build and fit both preprocessors on `X_train`
6. Compute helper values: `scale_pos_weight`, `hist_categorical_feature_idx`
7. Save processed CSVs to `data/processed/`
8. Save preprocessing pipeline bundle to pickle

**Config Keys Used:**

| Key | Description |
|-----|-------------|
| `data.train_csv` | Path to raw training CSV |
| `data.test_csv` | Path to raw test CSV |
| `data.processed_dir` | Output directory for processed data |
| `preprocessing.dropped_columns` | Columns to exclude from features |
| `preprocessing.target_column` | Target column name |
| `preprocessing.numeric_features` | Numeric feature column names |
| `preprocessing.categorical_features` | Categorical feature column names |
| `preprocessing.test_size` | Validation split ratio |
| `preprocessing.random_state` | Random seed |
| `preprocessing.pipeline_path` | Path to save the pipeline pickle |

**Output Files:**

| File | Content |
|------|---------|
| `data/processed/X_train.csv` | Training features |
| `data/processed/X_valid.csv` | Validation features |
| `data/processed/y_train.csv` | Training labels |
| `data/processed/y_valid.csv` | Validation labels |
| `data/processed/X_test_competition.csv` | Competition test features |
| `data/processed/test_passenger_ids.csv` | PassengerIds for inference |
| `models/preprocessing_pipeline.pkl` | Fitted preprocessor bundle |

**Pickle Bundle Contents:**

The saved pickle file contains a dictionary with:

```python
{
    "onehot_preprocessor": ColumnTransformer,   # Fitted onehot pipeline
    "hist_preprocessor": ColumnTransformer,      # Fitted ordinal pipeline
    "scale_pos_weight": float,                   # Class weight ratio
    "hist_categorical_feature_idx": list[int],   # Categorical column indices
    "numeric_features": list[str],               # Feature names
    "categorical_features": list[str],           # Feature names
}
```

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | If `train.csv` is not found at the configured path |
