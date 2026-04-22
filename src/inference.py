"""
Inference module:
  - Receives a CSV file path via argparse
  - Loads the best model and preprocessing pipeline
  - Predicts Survived for each passenger
  - Outputs a CSV with PassengerId and Survived
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG, resolve_path
from src.logger import get_logger

logger = get_logger("inference")


def prepare_catboost_frame(
    frame: pd.DataFrame, categorical_features: list[str]
) -> pd.DataFrame:
    prepared = frame.copy()
    for column in categorical_features:
        prepared[column] = prepared[column].fillna("missing").astype(str)
    return prepared


def run_inference(input_csv: str, output_csv: str) -> None:
    cfg_train = CONFIG["training"]
    cfg_preprocess = CONFIG["preprocessing"]

    # --- Load model ---
    model_path = resolve_path(cfg_train["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

    with open(model_path, "rb") as f:
        model_bundle = pickle.load(f)

    model = model_bundle["model"]
    model_name = model_bundle["model_name"]

    # --- Load preprocessing pipeline bundle ---
    pipeline_path = resolve_path(cfg_preprocess["pipeline_path"])
    with open(pipeline_path, "rb") as f:
        pipeline_bundle = pickle.load(f)
    categorical_features = pipeline_bundle["categorical_features"]

    # --- Read input CSV ---
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    # Extract PassengerId before dropping columns
    if "PassengerId" not in df.columns:
        raise ValueError("Input CSV must contain a 'PassengerId' column.")
    passenger_ids = df["PassengerId"].copy()

    # Drop columns that shouldn't be features
    dropped_cols = cfg_preprocess["dropped_columns"]
    cols_to_drop = [c for c in dropped_cols if c in df.columns]
    X = df.drop(cols_to_drop, axis=1).copy()

    # --- Prepare data based on model type ---
    if model_name == "catboost":
        X = prepare_catboost_frame(X, categorical_features)

    # --- Predict ---
    predictions = model.predict(X)

    # Ensure integer predictions
    predictions = predictions.astype(int)

    # --- Build output ---
    output_df = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions,
    })

    # --- Save output ---
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    logger.info(f"Total predictions: {len(output_df)}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on a Titanic CSV file.")
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input CSV file (with PassengerId)"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to output CSV file"
    )
    args = parser.parse_args()

    run_inference(args.input, args.output)


if __name__ == "__main__":
    main()
