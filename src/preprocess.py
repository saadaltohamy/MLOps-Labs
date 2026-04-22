"""
Preprocessing module:
  - Loads raw train/test CSVs
  - Builds sklearn ColumnTransformer pipelines (onehot + hist)
  - Splits data into train/valid
  - Saves processed data to CSV and preprocessing pipelines to pickle
"""

import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG, resolve_path
from src.logger import get_logger

logger = get_logger("preprocess")


def build_preprocessors(
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[ColumnTransformer, ColumnTransformer]:
    """Build onehot and hist (ordinal) preprocessors."""
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_onehot_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    categorical_ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )

    onehot_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_onehot_transformer, categorical_features),
        ]
    )

    hist_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_ordinal_transformer, categorical_features),
        ]
    )

    return onehot_preprocessor, hist_preprocessor


def run_preprocessing() -> None:
    """Main preprocessing pipeline."""
    cfg_preprocess = CONFIG["preprocessing"]
    cfg_data = CONFIG["data"]

    # --- Load raw data ---
    train_csv = resolve_path(cfg_data["train_csv"])
    test_csv = resolve_path(cfg_data["test_csv"])

    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}. Run download_data.py first.")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")

    # --- Extract features and target ---
    dropped_cols = cfg_preprocess["dropped_columns"]
    target_col = cfg_preprocess["target_column"]
    numeric_features = cfg_preprocess["numeric_features"]
    categorical_features = cfg_preprocess["categorical_features"]
    test_size = cfg_preprocess["test_size"]
    random_state = cfg_preprocess["random_state"]

    X = train_df.drop(dropped_cols, axis=1).copy()
    y = train_df[target_col].copy()
    X_test_competition = test_df.drop(
        [c for c in dropped_cols if c != target_col], axis=1
    ).copy()
    test_passenger_ids = test_df["PassengerId"].copy()

    # --- Train/valid split ---
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Split: X_train={X_train.shape}, X_valid={X_valid.shape}")

    # --- Build and fit preprocessors ---
    onehot_preprocessor, hist_preprocessor = build_preprocessors(
        numeric_features, categorical_features
    )
    # Fit on X_train so they are ready for transform
    onehot_preprocessor.fit(X_train)
    hist_preprocessor.fit(X_train)
    logger.info("Preprocessors fitted on X_train.")

    # --- Compute helper values needed during training ---
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    hist_categorical_feature_idx = list(
        range(len(numeric_features), len(numeric_features) + len(categorical_features))
    )

    # --- Save processed data ---
    processed_dir = resolve_path(cfg_data["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_valid.to_csv(processed_dir / "X_valid.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_valid.to_csv(processed_dir / "y_valid.csv", index=False)
    X_test_competition.to_csv(processed_dir / "X_test_competition.csv", index=False)
    test_passenger_ids.to_csv(processed_dir / "test_passenger_ids.csv", index=False)
    logger.info(f"Processed data saved to {processed_dir}")

    # --- Save preprocessing pipelines as pickle ---
    pipeline_path = resolve_path(cfg_preprocess["pipeline_path"])
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline_bundle = {
        "onehot_preprocessor": onehot_preprocessor,
        "hist_preprocessor": hist_preprocessor,
        "scale_pos_weight": scale_pos_weight,
        "hist_categorical_feature_idx": hist_categorical_feature_idx,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline_bundle, f)
    logger.info(f"Preprocessing pipelines saved to {pipeline_path}")


if __name__ == "__main__":
    run_preprocessing()
