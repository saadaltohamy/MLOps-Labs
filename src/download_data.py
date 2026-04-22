"""
Download the Titanic dataset from Kaggle via kagglehub.
Skips download if train.csv and test.csv already exist locally.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so we can import src.config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kagglehub
import kagglehub.config
from dotenv import load_dotenv

from src.config import CONFIG, resolve_path
from src.logger import get_logger

logger = get_logger("download_data")

# Load environment variables (.env) for Kaggle credentials
load_dotenv(resolve_path(".env"))


def download_data() -> None:
    """Download Titanic competition data from Kaggle if not already present."""
    raw_dir = resolve_path(CONFIG["data"]["raw_dir"])
    train_csv = resolve_path(CONFIG["data"]["train_csv"])
    test_csv = resolve_path(CONFIG["data"]["test_csv"])

    # Check if data already exists locally
    if train_csv.exists() and test_csv.exists():
        logger.info(f"Data already exists at {raw_dir}. Skipping download.")
        return

    # Ensure raw directory exists
    raw_dir.mkdir(parents=True, exist_ok=True)

    kaggle_key = os.environ.get("KAGGLE_KEY")
    if not kaggle_key:
        raise EnvironmentError(
            "KAGGLE_KEY not found in environment. "
            "Please set it in .env or as an environment variable."
        )
    kagglehub.config.set_kaggle_api_token(kaggle_key)

    logger.info(f"Downloading Titanic dataset to {raw_dir} ...")
    kagglehub.competition_download("titanic", output_dir=str(raw_dir), force_download=True)
    logger.info(f"Download complete. Files in: {raw_dir}")


if __name__ == "__main__":
    download_data()
