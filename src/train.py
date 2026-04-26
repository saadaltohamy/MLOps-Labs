"""
Training module:
  - Loads preprocessed data and preprocessing pipelines from pickle
  - Runs Optuna hyperparameter optimization across multiple model families
  - Saves the best model to pickle
  - Outputs metrics (accuracy, roc_auc) to reports/train_metrics.json
  - Generates a bar chart of top-10 trials by accuracy
  - Optionally logs Optuna trials to Weights & Biases and MLflow
"""

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI environments

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import wandb
import mlflow
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from optuna.integration import WeightsAndBiasesCallback
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG, resolve_path
from src.logger import get_logger

logger = get_logger("train")

load_dotenv(resolve_path(".env"))

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

MODEL_NAMES = [
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "xgboost",
    "catboost",
]


def prepare_catboost_frame(
    frame: pd.DataFrame, categorical_features: list[str]
) -> pd.DataFrame:
    prepared = frame.copy()
    for column in categorical_features:
        prepared[column] = prepared[column].fillna("missing").astype(str)
    return prepared


def sample_model_params(trial: optuna.Trial) -> dict:
    model_name = trial.suggest_categorical("model_name", MODEL_NAMES)
    params: dict = {"model_name": model_name}

    if model_name == "random_forest":
        params.update(
            {
                "n_estimators": trial.suggest_int("rf_n_estimators", 200, 700),
                "max_depth": trial.suggest_int("rf_max_depth", 3, 16),
                "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "rf_max_features", ["sqrt", "log2", None]
                ),
            }
        )
    elif model_name == "extra_trees":
        params.update(
            {
                "n_estimators": trial.suggest_int("et_n_estimators", 200, 700),
                "max_depth": trial.suggest_int("et_max_depth", 3, 16),
                "min_samples_split": trial.suggest_int("et_min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("et_min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "et_max_features", ["sqrt", "log2", None]
                ),
            }
        )
    elif model_name == "gradient_boosting":
        params.update(
            {
                "n_estimators": trial.suggest_int("gb_n_estimators", 100, 500),
                "learning_rate": trial.suggest_float("gb_learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("gb_max_depth", 2, 6),
                "min_samples_split": trial.suggest_int("gb_min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("gb_min_samples_leaf", 1, 10),
                "subsample": trial.suggest_float("gb_subsample", 0.6, 1.0),
                "max_features": trial.suggest_categorical(
                    "gb_max_features", ["sqrt", "log2", None]
                ),
            }
        )
    elif model_name == "hist_gradient_boosting":
        params.update(
            {
                "learning_rate": trial.suggest_float("hgb_learning_rate", 0.01, 0.2, log=True),
                "max_iter": trial.suggest_int("hgb_max_iter", 150, 600),
                "max_leaf_nodes": trial.suggest_int("hgb_max_leaf_nodes", 15, 63),
                "max_depth": trial.suggest_int("hgb_max_depth", 3, 12),
                "min_samples_leaf": trial.suggest_int("hgb_min_samples_leaf", 10, 60),
                "l2_regularization": trial.suggest_float(
                    "hgb_l2_regularization", 1e-4, 10.0, log=True
                ),
                "max_bins": trial.suggest_int("hgb_max_bins", 64, 255),
            }
        )
    elif model_name == "xgboost":
        params.update(
            {
                "n_estimators": trial.suggest_int("xgb_n_estimators", 150, 700),
                "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
                "min_child_weight": trial.suggest_float("xgb_min_child_weight", 1.0, 10.0),
                "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-4, 10.0, log=True),
            }
        )
    else:  # catboost
        params.update(
            {
                "iterations": trial.suggest_int("cat_iterations", 200, 800),
                "learning_rate": trial.suggest_float("cat_learning_rate", 0.01, 0.2, log=True),
                "depth": trial.suggest_int("cat_depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("cat_l2_leaf_reg", 1.0, 10.0),
                "random_strength": trial.suggest_float(
                    "cat_random_strength", 1e-3, 10.0, log=True
                ),
                "bagging_temperature": trial.suggest_float(
                    "cat_bagging_temperature", 0.0, 5.0
                ),
                "border_count": trial.suggest_int("cat_border_count", 32, 255),
            }
        )

    return params


def build_model_from_params(
    params: dict,
    onehot_preprocessor: ColumnTransformer,
    hist_preprocessor: ColumnTransformer,
    scale_pos_weight: float,
    hist_categorical_feature_idx: list[int],
    categorical_features: list[str],
    X_train: pd.DataFrame,
    X_train_catboost: pd.DataFrame,
):
    """Return (estimator, X_to_use, fit_kwargs) based on model_name."""
    model_name = params["model_name"]

    if model_name == "random_forest":
        estimator = Pipeline(
            steps=[
                ("preprocessor", clone(onehot_preprocessor)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        min_samples_split=params["min_samples_split"],
                        min_samples_leaf=params["min_samples_leaf"],
                        max_features=params["max_features"],
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )
        return estimator, X_train, {}

    if model_name == "extra_trees":
        estimator = Pipeline(
            steps=[
                ("preprocessor", clone(onehot_preprocessor)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        min_samples_split=params["min_samples_split"],
                        min_samples_leaf=params["min_samples_leaf"],
                        max_features=params["max_features"],
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )
        return estimator, X_train, {}

    if model_name == "gradient_boosting":
        estimator = Pipeline(
            steps=[
                ("preprocessor", clone(onehot_preprocessor)),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=params["n_estimators"],
                        learning_rate=params["learning_rate"],
                        max_depth=params["max_depth"],
                        min_samples_split=params["min_samples_split"],
                        min_samples_leaf=params["min_samples_leaf"],
                        subsample=params["subsample"],
                        max_features=params["max_features"],
                        random_state=42,
                    ),
                ),
            ]
        )
        return estimator, X_train, {}

    if model_name == "hist_gradient_boosting":
        estimator = Pipeline(
            steps=[
                ("preprocessor", clone(hist_preprocessor)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=params["learning_rate"],
                        max_iter=params["max_iter"],
                        max_leaf_nodes=params["max_leaf_nodes"],
                        max_depth=params["max_depth"],
                        min_samples_leaf=params["min_samples_leaf"],
                        l2_regularization=params["l2_regularization"],
                        max_bins=params["max_bins"],
                        categorical_features=hist_categorical_feature_idx,
                        early_stopping=False,
                        random_state=42,
                    ),
                ),
            ]
        )
        return estimator, X_train, {}

    if model_name == "xgboost":

        estimator = Pipeline(
            steps=[
                ("preprocessor", clone(onehot_preprocessor)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=params["n_estimators"],
                        learning_rate=params["learning_rate"],
                        max_depth=params["max_depth"],
                        min_child_weight=params["min_child_weight"],
                        subsample=params["subsample"],
                        colsample_bytree=params["colsample_bytree"],
                        reg_alpha=params["reg_alpha"],
                        reg_lambda=params["reg_lambda"],
                        scale_pos_weight=scale_pos_weight,
                        tree_method="hist",
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        return estimator, X_train, {}

    # catboost

    estimator = CatBoostClassifier(
        iterations=params["iterations"],
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        l2_leaf_reg=params["l2_leaf_reg"],
        random_strength=params["random_strength"],
        bagging_temperature=params["bagging_temperature"],
        border_count=params["border_count"],
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        verbose=0,
        allow_writing_files=False,
        random_state=42,
    )
    return estimator, X_train_catboost, {"cat_features": categorical_features}


# ---------------------------------------------------------------------------
# wandb helpers
# ---------------------------------------------------------------------------

def _init_wandb_if_available():
    """Return wandb module if WANDB_API_KEY is set, else None."""
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        logger.warning("WANDB_API_KEY not set — skipping wandb logging.")
        return None
    wandb.login(key=api_key)
    return wandb


def _init_mlflow_if_available(study_name: str):
    """Return MLflow callback if MLFLOW_TRACKING_URI is set, else None."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        logger.warning("MLFLOW_TRACKING_URI not set - skipping MLflow logging.")
        return None

    from optuna.integration import MLflowCallback

    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking enabled: {tracking_uri}")
    return MLflowCallback(
        tracking_uri=tracking_uri,
        metric_name="accuracy",
        mlflow_kwargs={"experiment_name": study_name, "nested": True},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_training() -> None:
    cfg_train = CONFIG["training"]
    cfg_data = CONFIG["data"]
    cfg_preprocess = CONFIG["preprocessing"]
    cfg_reports = CONFIG["reports"]

    # --- Load processed data ---
    processed_dir = resolve_path(cfg_data["processed_dir"])
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze()

    # --- Load preprocessing pipelines ---
    pipeline_path = resolve_path(cfg_preprocess["pipeline_path"])
    with open(pipeline_path, "rb") as f:
        bundle = pickle.load(f)

    onehot_preprocessor = bundle["onehot_preprocessor"]
    hist_preprocessor = bundle["hist_preprocessor"]
    scale_pos_weight = bundle["scale_pos_weight"]
    hist_categorical_feature_idx = bundle["hist_categorical_feature_idx"]
    categorical_features = bundle["categorical_features"]

    # Prepare CatBoost-compatible frame
    X_train_catboost = prepare_catboost_frame(X_train, categorical_features)

    # --- Cross-validation setup ---
    n_trials = cfg_train["n_trials"]
    cv_folds = cfg_train["cv_folds"]
    random_state = cfg_train["random_state"]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # --- wandb ---
    wandb_mod = _init_wandb_if_available()
    wandb_callback = None
    if wandb_mod is not None:
        wandb_callback = WeightsAndBiasesCallback(
            metric_name="roc_auc",
            as_multirun=True,
            wandb_kwargs={"project": os.getenv("WANDB_PROJECT", "mlops-lab0")},
        )

    mlflow_callback = _init_mlflow_if_available(cfg_train["study_name"])

    # --- Optuna objective ---
    def objective(trial: optuna.Trial) -> float:
        params = sample_model_params(trial)
        estimator, search_train, fit_kwargs = build_model_from_params(
            params,
            onehot_preprocessor,
            hist_preprocessor,
            scale_pos_weight,
            hist_categorical_feature_idx,
            categorical_features,
            X_train,
            X_train_catboost,
        )

        fold_scores = []
        for fold_idx, (train_idx, valid_idx) in enumerate(
            cv.split(search_train, y_train), start=1
        ):
            estimator_fold = clone(estimator)
            X_fold_train = search_train.iloc[train_idx]
            X_fold_valid = search_train.iloc[valid_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_valid = y_train.iloc[valid_idx]

            estimator_fold.fit(X_fold_train, y_fold_train, **fit_kwargs)
            y_valid_proba = estimator_fold.predict_proba(X_fold_valid)[:, 1]
            fold_score = roc_auc_score(y_fold_valid, y_valid_proba)
            fold_scores.append(fold_score)
            mlflow.log_metric(f"fold_{fold_idx}_roc_auc", fold_score)
            trial.report(float(np.mean(fold_scores)), step=fold_idx)

        mean_auc = float(np.mean(fold_scores))
        std_auc = float(np.std(fold_scores))

        mlflow.log_metric("cv_mean_roc_auc", mean_auc)
        mlflow.log_metric("cv_std_roc_auc", std_auc)

        return mean_auc
    # --- Run study ---
    study = optuna.create_study(
        study_name=cfg_train["study_name"],
        direction="maximize",
    )

    callbacks: list[Any] = []
    if wandb_callback is not None:
        callbacks.append(wandb_callback)
    if mlflow_callback is not None:
        callbacks.append(mlflow_callback)

    logger.info(f"Starting Optuna study with {n_trials} trials ...")
    study.optimize(objective, n_trials=n_trials, callbacks=callbacks)

    # --- Retrain best model on full training set ---
    best_params = study.best_trial.params
    # Rebuild params dict (add model_name which optuna stored)
    best_params_full = {"model_name": best_params.pop("model_name")}
    # Strip prefix from param names
    for key, value in best_params.items():
        # Remove model prefix like "rf_", "et_", "gb_", etc.
        parts = key.split("_", 1)
        if len(parts) > 1 and parts[0] in ("rf", "et", "gb", "hgb", "xgb", "cat"):
            clean_key = parts[1]
        else:
            clean_key = key
        best_params_full[clean_key] = value

    best_model, best_X_train, best_fit_kwargs = build_model_from_params(
        best_params_full,
        onehot_preprocessor,
        hist_preprocessor,
        scale_pos_weight,
        hist_categorical_feature_idx,
        categorical_features,
        X_train,
        X_train_catboost,
    )
    logger.info(f"Best trial: {study.best_trial.number} | ROC-AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {best_params_full}")

    best_model.fit(best_X_train, y_train, **best_fit_kwargs)

    # --- Save best model ---
    model_path = resolve_path(cfg_train["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": best_model,
        "model_name": best_params_full["model_name"],
        "best_roc_auc_cv": study.best_value,
        "best_params": best_params_full,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    logger.info(f"Best model saved to {model_path}")

    # --- Save metrics ---
    reports_dir = resolve_path(cfg_reports["dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = resolve_path(cfg_reports["train_metrics_file"])

    # Compute training accuracy
    y_train_pred = best_model.predict(best_X_train)
    train_accuracy = float(accuracy_score(y_train, y_train_pred))
    train_roc_auc = float(
        roc_auc_score(y_train, best_model.predict_proba(best_X_train)[:, 1])
    )

    metrics = {
        "train_accuracy": train_accuracy,
        "train_roc_auc": train_roc_auc,
        "best_cv_roc_auc": study.best_value,
        "best_model": best_params_full["model_name"],
        "best_params": {k: v for k, v in best_params_full.items() if k != "model_name"},
        "n_trials": n_trials,
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Training metrics saved to {metrics_file}")

    # --- Generate top-10 trials chart ---
    chart_path = resolve_path(cfg_reports["optuna_top10_chart"])
    chart_path.parent.mkdir(parents=True, exist_ok=True)

    trials_df = study.trials_dataframe()
    top10 = trials_df.nlargest(min(10, len(trials_df)), "value")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        [f"Trial {t}" for t in top10["number"]],
        top10["value"],
        color="#4C72B0",
    )
    ax.set_xlabel("ROC-AUC (CV)")
    ax.set_title("Top 10 Optuna Trials by ROC-AUC")
    ax.invert_yaxis()
    for bar, val in zip(bars, top10["value"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    logger.info(f"Top-10 chart saved to {chart_path}")

    # --- Finalize wandb ---
    if wandb_mod is not None:
        try:
            wandb_mod.finish()
        except Exception:
            pass

    logger.info("Training complete!")


if __name__ == "__main__":
    run_training()
