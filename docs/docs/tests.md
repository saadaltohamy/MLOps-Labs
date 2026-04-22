# Data Validation Tests

Data validation tests run automatically after preprocessing (Step 3 in the pipeline).
If any test fails, the pipeline stops — training will **not** proceed with invalid data.

---

## Running Tests

```bash
python -m pytest tests/test_data.py -v
```

---

## Test Descriptions

### `test_data_files_exist`

Verifies that all expected processed data files exist on disk:

- `data/processed/X_train.csv`
- `data/processed/X_valid.csv`
- `data/processed/y_train.csv`
- `data/processed/y_valid.csv`
- `data/processed/X_test_competition.csv`
- `data/processed/test_passenger_ids.csv`

---

### `test_preprocessing_pipeline_exists`

Verifies that the fitted preprocessing pipeline pickle file exists at the path specified in `config.yaml` (default: `models/preprocessing_pipeline.pkl`).

---

### `test_no_missing_values_after_transform`

Applies the fitted `onehot_preprocessor` to both `X_train` and `X_valid`, then asserts that the transformed arrays contain **no NaN values**.

This validates that the `SimpleImputer` steps in the preprocessing pipeline are working correctly.

---

### `test_correct_shape`

Asserts that:

1. `X_train` and `X_valid` have the **same number of columns**
2. The column count matches the total number of configured features (`numeric_features` + `categorical_features`)

---

### `test_target_is_binary`

Asserts that `y_train` and `y_valid` only contain values **0** and **1**.

---

### `test_no_duplicate_rows`

Checks for duplicate rows in `X_train`. Because we drop identifier columns (Name, Ticket, Cabin, PassengerId), some passengers may naturally have identical feature values.

!!! note
    This test **warns** rather than fails, since feature-level duplicates are expected for the Titanic dataset after dropping identifiers.

---

### `test_train_valid_split_ratio`

Asserts that the actual validation split ratio approximately matches the configured `test_size` (within ±5% tolerance).

---

## Fixtures

The tests use `pytest` module-scoped fixtures to load data efficiently:

| Fixture | Description |
|---------|-------------|
| `processed_dir` | Resolved path to `data/processed/` |
| `X_train` | Loaded `X_train.csv` as a DataFrame |
| `X_valid` | Loaded `X_valid.csv` as a DataFrame |
| `y_train` | Loaded `y_train.csv` as a Series |
| `y_valid` | Loaded `y_valid.csv` as a Series |
| `pipeline_bundle` | Loaded preprocessing pipeline pickle dict |
