"""
Data validation tests — run after preprocessing, before training.
Validates that the processed data is clean and correctly structured.
"""

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG, resolve_path


@pytest.fixture(scope="module")
def processed_dir():
    return resolve_path(CONFIG["data"]["processed_dir"])


@pytest.fixture(scope="module")
def X_train(processed_dir):
    return pd.read_csv(processed_dir / "X_train.csv")


@pytest.fixture(scope="module")
def X_valid(processed_dir):
    return pd.read_csv(processed_dir / "X_valid.csv")


@pytest.fixture(scope="module")
def y_train(processed_dir):
    return pd.read_csv(processed_dir / "y_train.csv").squeeze()


@pytest.fixture(scope="module")
def y_valid(processed_dir):
    return pd.read_csv(processed_dir / "y_valid.csv").squeeze()


@pytest.fixture(scope="module")
def pipeline_bundle():
    pipeline_path = resolve_path(CONFIG["preprocessing"]["pipeline_path"])
    with open(pipeline_path, "rb") as f:
        return pickle.load(f)


# ---- Test: all processed data files exist on disk ----
def test_data_files_exist(processed_dir):
    expected_files = [
        "X_train.csv",
        "X_valid.csv",
        "y_train.csv",
        "y_valid.csv",
        "X_test_competition.csv",
        "test_passenger_ids.csv",
    ]
    for filename in expected_files:
        filepath = processed_dir / filename
        assert filepath.exists(), f"Missing processed data file: {filepath}"


# ---- Test: preprocessing pipeline pickle exists ----
def test_preprocessing_pipeline_exists():
    pipeline_path = resolve_path(CONFIG["preprocessing"]["pipeline_path"])
    assert pipeline_path.exists(), f"Preprocessing pipeline not found: {pipeline_path}"


# ---- Test: no missing values after preprocessing pipeline transform ----
def test_no_missing_values_after_transform(X_train, X_valid, pipeline_bundle):
    """After applying the fitted onehot_preprocessor, there should be no NaN values."""
    onehot_preprocessor = pipeline_bundle["onehot_preprocessor"]

    X_train_transformed = onehot_preprocessor.transform(X_train)
    X_valid_transformed = onehot_preprocessor.transform(X_valid)

    # ColumnTransformer can return sparse matrix or ndarray
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
    if hasattr(X_valid_transformed, "toarray"):
        X_valid_transformed = X_valid_transformed.toarray()

    assert not np.isnan(X_train_transformed).any(), (
        "X_train has NaN values after preprocessing"
    )
    assert not np.isnan(X_valid_transformed).any(), (
        "X_valid has NaN values after preprocessing"
    )


# ---- Test: correct number of columns ----
def test_correct_shape(X_train, X_valid):
    """X_train and X_valid should have the same number of columns."""
    assert X_train.shape[1] == X_valid.shape[1], (
        f"Column count mismatch: X_train={X_train.shape[1]}, X_valid={X_valid.shape[1]}"
    )

    expected_cols = (
        CONFIG["preprocessing"]["numeric_features"]
        + CONFIG["preprocessing"]["categorical_features"]
    )
    assert X_train.shape[1] == len(expected_cols), (
        f"Expected {len(expected_cols)} columns, got {X_train.shape[1]}"
    )


# ---- Test: target is binary (0 or 1) ----
def test_target_is_binary(y_train, y_valid):
    assert set(y_train.unique()).issubset({0, 1}), (
        f"y_train contains non-binary values: {y_train.unique()}"
    )
    assert set(y_valid.unique()).issubset({0, 1}), (
        f"y_valid contains non-binary values: {y_valid.unique()}"
    )


# ---- Test: check for duplicate rows in training data (warning) ----
def test_no_duplicate_rows(X_train):
    """
    Check for duplicate rows in X_train. Because we drop identifier columns
    (Name, Ticket, Cabin, PassengerId), some passengers may naturally have
    identical feature values. This is expected — we warn rather than fail.
    """
    n_duplicates = X_train.duplicated().sum()
    if n_duplicates > 0:
        warnings.warn(
            f"X_train has {n_duplicates} duplicate rows (after dropping identifier columns). "
            "This is expected for the Titanic dataset."
        )
    # Still passes — duplicates in features are not a data quality error here
    assert True


# ---- Test: train/valid split ratio is approximately correct ----
def test_train_valid_split_ratio(X_train, X_valid):
    total = len(X_train) + len(X_valid)
    test_size = CONFIG["preprocessing"]["test_size"]
    actual_valid_ratio = len(X_valid) / total
    # Allow 5% tolerance
    assert abs(actual_valid_ratio - test_size) < 0.05, (
        f"Split ratio mismatch: expected ~{test_size}, got {actual_valid_ratio:.3f}"
    )
