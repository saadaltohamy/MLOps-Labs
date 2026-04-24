# 🚢 Titanic ML Pipeline

Welcome to the documentation for **MLOps Labs** — a modular, production-ready machine learning pipeline for the Titanic survival prediction task.

Built as part of the **ITI MLOps Course**.

---

## Features

- **Modular pipeline** — Download → Preprocess → Validate → Train → Test → Inference
- **Optuna hyperparameter optimization** across 6 model families
- **Weights & Biases integration** — optional experiment tracking
- **Data validation tests** — pipeline stops if data quality checks fail
- **Colored logging** — centralized logger with color-coded log levels
- **Hydra config management** — all settings in `config.yaml` with CLI overrides
- **CLI inference** — predict on new data with one command

---

## Quick Links

| Section | Description |
|---------|-------------|
| [Getting Started](getting-started.md) | Installation, setup, and first run |
| [Pipeline Overview](pipeline.md) | Step-by-step pipeline flow |
| [Configuration](configuration.md) | Hydra config, CLI overrides, environment variables |
| [API Reference](api/config.md) | Module and function documentation |
| [Data Validation Tests](tests.md) | Test descriptions and usage |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| scikit-learn | Preprocessing pipelines & ensemble models |
| XGBoost | Gradient boosting classifier |
| CatBoost | Categorical-aware boosting |
| Optuna | Hyperparameter optimization |
| Hydra | Configuration management with CLI overrides |
| Weights & Biases | Experiment tracking (optional) |
| pytest | Data validation testing |
| MKDocs | Project documentation |
| kagglehub | Kaggle data download |
