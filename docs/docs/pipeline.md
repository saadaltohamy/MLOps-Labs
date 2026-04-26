# Pipeline Overview

The Titanic ML pipeline is a sequential workflow that automates the full machine learning lifecycle — from raw data to predictions. It can be run either through `run_pipeline.sh` or reproduced stage-by-stage through DVC using `dvc.yaml`.

---

## DVC Integration

The repository includes a DVC pipeline with these stages:

1. `download`
2. `preprocess`
3. `validate`
4. `train`
5. `evaluate`

Before using `dvc pull` or `dvc push`, configure the existing `origin` remote locally:

```bash
dvc remote modify origin --local access_key_id c355...2ffc
dvc remote modify origin --local secret_access_key c355...2ffc
```

Run the pipeline with:

```bash
dvc repro
```

---

## Pipeline Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    run_pipeline.sh                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: download_data.py                                    │
│  ├─ Check if data/raw/train.csv exists                       │
│  ├─ If not → download from Kaggle via kagglehub              │
│  └─ Output: data/raw/train.csv, data/raw/test.csv            │
│                          ↓                                   │
│  Step 2: preprocess.py                                       │
│  ├─ Load raw CSVs                                            │
│  ├─ Drop columns (Cabin, PassengerId, Ticket, Name)          │
│  ├─ Build two sklearn ColumnTransformers:                    │
│  │   ├─ onehot_preprocessor (for RF, ET, GB, XGB)            │
│  │   └─ hist_preprocessor (for HistGBT — uses OrdinalEncoder)│
│  ├─ Train/valid split (80/20, stratified)                    │
│  ├─ Save CSVs → data/processed/                              │
│  └─ Save pipelines → models/preprocessing_pipeline.pkl       │
│                          ↓                                   │
│  Step 3: pytest tests/test_data.py                           │
│  ├─ Validate: no missing values after transform              │
│  ├─ Validate: correct column count                           │
│  ├─ Validate: binary target (0 or 1)                         │
│  ├─ Validate: split ratio matches config                     │
│  └─ ⛔ Pipeline STOPS if any test fails                      │
│                          ↓                                   │
│  Step 4: train.py                                            │
│  ├─ Load processed data + preprocessing pipelines            │
│  ├─ Run Optuna study (n_trials from config)                  │
│  ├─ Search across 6 model families:                          │
│  │   RF, ExtraTrees, GBT, HistGBT, XGBoost, CatBoost         │
│  ├─ 5-fold StratifiedKFold cross-validation                  │
│  ├─ Optional: log to Weights & Biases and MLflow             │
│  ├─ Retrain best model on full train+valid set               │
│  ├─ Save model → models/best_model.pkl                       │
│  ├─ Save metrics → reports/train_metrics.json                │
│  └─ Save chart → reports/figures/optuna_top10_accuracy.png   │
│                          ↓                                   │
│  Step 5: test_model.py                                       │
│  ├─ Load best model from pickle                              │
│  ├─ Predict on X_valid                                       │
│  ├─ Report accuracy + ROC-AUC                                │
│  └─ Update reports/metrics.json                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Model Families

The Optuna study searches across these 6 model families:

| Model | Library | Preprocessor Used |
|-------|---------|-------------------|
| Random Forest | scikit-learn | OneHotEncoder |
| Extra Trees | scikit-learn | OneHotEncoder |
| Gradient Boosting | scikit-learn | OneHotEncoder |
| Hist Gradient Boosting | scikit-learn | OrdinalEncoder |
| XGBoost | xgboost | OneHotEncoder |
| CatBoost | catboost | Native categorical handling |

---

## Generated Outputs

| File | Description |
|------|-------------|
| `models/best_model.pkl` | Serialized best model (includes preprocessing pipeline) |
| `models/preprocessing_pipeline.pkl` | Fitted sklearn `ColumnTransformer` objects |
| `reports/train_metrics.json` | Training accuracy, ROC-AUC, best CV accuracy, best params |
| `reports/metrics.json` | Validation accuracy and ROC-AUC |
| `reports/figures/optuna_top10_accuracy.png` | Bar chart of top 10 Optuna trials by accuracy |
| `data/processed/*.csv` | Processed train/valid splits |

---

## Logging

All pipeline modules use a centralized colored logger (`src/logger.py`).

Log output format:

```
2026-04-22 19:00:03 | INFO | preprocess | Loaded train: (891, 12), test: (418, 11)
```

Color coding:

| Level | Color |
|-------|-------|
| `DEBUG` | Cyan |
| `INFO` | Green |
| `WARNING` | Yellow |
| `ERROR` | Red |
| `CRITICAL` | Bold Red |
