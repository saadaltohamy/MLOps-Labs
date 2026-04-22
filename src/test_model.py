"""
Testing module:
  - Loads the best trained model from pickle
  - Evaluates it on X_valid / y_valid
  - Reports accuracy and ROC-AUC
  - Saves results to reports/metrics.json
"""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG, resolve_path
from src.logger import get_logger

logger = get_logger("test_model")


def prepare_catboost_frame(
    frame: pd.DataFrame, categorical_features: list[str]
) -> pd.DataFrame:
    prepared = frame.copy()
    for column in categorical_features:
        prepared[column] = prepared[column].fillna("missing").astype(str)
    return prepared


def run_testing() -> None:
    cfg_data = CONFIG["data"]
    cfg_train = CONFIG["training"]
    cfg_preprocess = CONFIG["preprocessing"]
    cfg_reports = CONFIG["reports"]

    # --- Load processed validation data ---
    processed_dir = resolve_path(cfg_data["processed_dir"])
    X_valid = pd.read_csv(processed_dir / "X_valid.csv")
    y_valid = pd.read_csv(processed_dir / "y_valid.csv").squeeze()

    # --- Load model bundle ---
    model_path = resolve_path(cfg_train["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

    with open(model_path, "rb") as f:
        model_bundle = pickle.load(f)

    model = model_bundle["model"]
    model_name = model_bundle["model_name"]

    # --- Load preprocessing pipeline for categorical_features ---
    pipeline_path = resolve_path(cfg_preprocess["pipeline_path"])
    with open(pipeline_path, "rb") as f:
        pipeline_bundle = pickle.load(f)
    categorical_features = pipeline_bundle["categorical_features"]

    # --- Prepare data based on model type ---
    if model_name == "catboost":
        X_eval = prepare_catboost_frame(X_valid, categorical_features)
    else:
        X_eval = X_valid

    # --- Predict ---
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    # --- Calculate metrics ---
    accuracy = float(accuracy_score(y_valid, y_pred))
    roc_auc = float(roc_auc_score(y_valid, y_proba))

    logger.info(f"Model: {model_name}")
    logger.info(f"Validation Accuracy: {accuracy:.4f}")
    logger.info(f"Validation ROC-AUC:  {roc_auc:.4f}")

    # --- Save/update metrics ---
    metrics_file = resolve_path(cfg_reports["metrics_file"])
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing metrics if present
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    metrics["validation_accuracy"] = accuracy
    metrics["validation_roc_auc"] = roc_auc

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    run_testing()
