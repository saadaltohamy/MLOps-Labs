# `src.inference`

Inference module. CLI tool that receives a CSV file as input, runs predictions using the saved best model, and outputs a CSV with `PassengerId` and `Survived` columns.

---

## Usage

```bash
python src/inference.py --input <input_csv> --output <output_csv>
```

**Example:**

```bash
python src/inference.py --input data/raw/test.csv --output reports/predictions.csv
```

---

## CLI Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--input` | `-i` | Yes | Path to input CSV file (must contain `PassengerId` column) |
| `--output` | `-o` | Yes | Path to output CSV file |

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

### `run_inference`

```python
def run_inference(input_csv: str, output_csv: str) -> None
```

Runs inference on a given CSV file using the saved best model.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_csv` | `str` | Path to the input CSV file |
| `output_csv` | `str` | Path to save the predictions CSV |

**Steps:**

1. Load the model bundle from `models/best_model.pkl`
2. Load the preprocessing pipeline bundle (for `categorical_features`)
3. Read the input CSV
4. Extract `PassengerId` column
5. Drop columns listed in `preprocessing.dropped_columns`
6. Prepare data for CatBoost if needed
7. Generate integer predictions (0 or 1)
8. Save output CSV with `PassengerId` and `Survived`

**Output CSV Format:**

| Column | Type | Description |
|--------|------|-------------|
| `PassengerId` | int | Passenger identifier from input |
| `Survived` | int | Predicted survival (0 or 1) |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | If the model pickle or input CSV does not exist |
| `ValueError` | If the input CSV does not contain a `PassengerId` column |

---

### `main`

```python
def main() -> None
```

CLI entry point. Parses `--input` and `--output` arguments using `argparse`, then calls `run_inference()`.
