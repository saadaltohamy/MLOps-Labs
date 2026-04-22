#!/bin/bash
# ===========================================================================
# Titanic ML Pipeline — Full Orchestration Script
# ===========================================================================
# Runs the pipeline in order:
#   1. Download data (skips if already present)
#   2. Preprocessing (build pipelines, split data, save as pickle + CSV)
#   3. Data validation tests (pytest — pipeline STOPS if tests fail)
#   4. Training (Optuna HPO, save best model, generate charts)
#   5. Testing (evaluate on validation set, output final metrics)
# ===========================================================================

set -e  # Exit immediately on any error

# Move to project root (directory containing this script)
cd "$(dirname "$0")"

echo ""
echo "============================================"
echo "  Step 1: Downloading data"
echo "============================================"
python src/download_data.py

echo ""
echo "============================================"
echo "  Step 2: Preprocessing"
echo "============================================"
python src/preprocess.py

echo ""
echo "============================================"
echo "  Step 3: Data validation tests"
echo "============================================"
python -m pytest tests/test_data.py -v

echo ""
echo "============================================"
echo "  Step 4: Training (Optuna)"
echo "============================================"
python src/train.py

echo ""
echo "============================================"
echo "  Step 5: Testing (Validation evaluation)"
echo "============================================"
python src/test_model.py

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "============================================"
echo ""
echo "  Model saved to:   models/best_model.pkl"
echo "  Metrics saved to: reports/metrics.json"
echo "  Chart saved to:   reports/figures/optuna_top10_accuracy.png"
echo ""
