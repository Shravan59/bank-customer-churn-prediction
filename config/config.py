"""
config.py
=========

Centralized configuration for the
Predictive Modeling and Risk Scoring for Bank Customer Churn project.

All project modules should import configuration values from this file
instead of hardcoding paths, filenames, or constants.

Python Version
--------------
3.12+

Author
------
Shravan Pandey
"""

from __future__ import annotations

from pathlib import Path
import logging

# =============================================================================
# Project Information
# =============================================================================

PROJECT_NAME: str = "Predictive Modeling and Risk Scoring for Bank Customer Churn"

PROJECT_VERSION: str = "1.0.0"

PROJECT_DESCRIPTION: str = (
    "End-to-end machine learning project for predicting bank customer churn "
    "using data preprocessing, feature engineering, model training, "
    "evaluation, explainability, visualization, and Streamlit deployment."
)

PROJECT_AUTHOR: str = "Shravan Pandey"

APP_NAME = PROJECT_NAME
APP_VERSION = PROJECT_VERSION

# =============================================================================
# Project Paths
# =============================================================================

CONFIG_DIR: Path = Path(__file__).resolve().parent

PROJECT_ROOT: Path = CONFIG_DIR.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"

MODEL_DIR: Path = PROJECT_ROOT / "models"

REPORT_DIR: Path = PROJECT_ROOT / "reports"

VISUALS_DIR: Path = PROJECT_ROOT / "visuals"
EDA_VISUALS_DIR: Path = VISUALS_DIR / "eda"
MODEL_VISUALS_DIR: Path = VISUALS_DIR / "model"
DASHBOARD_VISUALS_DIR: Path = VISUALS_DIR / "dashboard"

ASSETS_DIR: Path = PROJECT_ROOT / "assets"
ICONS_DIR: Path = ASSETS_DIR / "icons"

NOTEBOOK_DIR: Path = PROJECT_ROOT / "notebooks"

PAGES_DIR: Path = PROJECT_ROOT / "pages"

UTILS_DIR: Path = PROJECT_ROOT / "utils"

SRC_DIR: Path = PROJECT_ROOT / "src"

LOG_DIR: Path = PROJECT_ROOT / "logs"

PREDICTIONS_DIR: Path = PROJECT_ROOT / "predictions"

# =============================================================================
# Dataset Configuration
# =============================================================================

DATASET_FILENAME: str = "Bank Customer Churn Prediction.csv"

RAW_DATA_PATH: Path = RAW_DATA_DIR / DATASET_FILENAME

SAMPLE_INPUT_PATH: Path = SAMPLE_DATA_DIR / "sample_input.csv"

TARGET_COLUMN: str = "Exited"

# =============================================================================
# Additional Paths
# =============================================================================

DATASET_PATH: Path = RAW_DATA_PATH

MODEL_ARTIFACTS_DIR: Path = MODEL_DIR

INTERNSHIP_NAME: str = "Unified Mentor Data Science Internship"

INTERNSHIP_ORGANIZATION: str = "Unified Mentor"

IDENTIFIER_COLUMNS: list[str] = [
    "CustomerId",
    "Surname",
]

CATEGORICAL_COLUMNS: list[str] = [
    "Geography",
    "Gender",
]

NUMERICAL_COLUMNS: list[str] = [
    "Year",
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

BINARY_COLUMNS: list[str] = [
    "HasCrCard",
    "IsActiveMember",
]

# =============================================================================
# Machine Learning Configuration
# =============================================================================

RANDOM_STATE: int = 42

TEST_SIZE: float = 0.20

CV_FOLDS: int = 5

SCORING_METRIC: str = "roc_auc"

CLASSIFICATION_MODELS: list[str] = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
]

# =============================================================================
# Artifact Paths
# =============================================================================

# =============================================================================
# Artifact Directory
# =============================================================================

ARTIFACTS_DIR: Path = MODEL_DIR

BEST_MODEL_PATH: Path = MODEL_DIR / "best_model.pkl"

PREPROCESSING_PIPELINE_PATH: Path = (
    MODEL_DIR / "preprocessing_pipeline.pkl"
)

FEATURE_COLUMNS_PATH: Path = (
    MODEL_DIR / "feature_columns.pkl"
)

MODEL_METRICS_PATH: Path = (
    MODEL_DIR / "model_metrics.json"
)

MODEL_METADATA_PATH: Path = (
    MODEL_DIR / "model_metadata.json"
)

MODEL_COMPARISON_PATH: Path = (
    MODEL_DIR / "model_comparison.csv"
)

TRAINING_RESULTS_PATH: Path = (
    MODEL_DIR / "training_results.csv"
)

# =============================================================================
# Processed Dataset Paths
# =============================================================================

CLEANED_DATA_PATH: Path = (
    PROCESSED_DATA_DIR / "cleaned_data.csv"
)

ENGINEERED_TRAIN_PATH: Path = (
    PROCESSED_DATA_DIR / "engineered_train.csv"
)

ENGINEERED_TEST_PATH: Path = (
    PROCESSED_DATA_DIR / "engineered_test.csv"
)

X_TRAIN_PROCESSED_PATH: Path = (
    PROCESSED_DATA_DIR / "X_train_processed.csv"
)

X_TEST_PROCESSED_PATH: Path = (
    PROCESSED_DATA_DIR / "X_test_processed.csv"
)

Y_TRAIN_PATH: Path = (
    PROCESSED_DATA_DIR / "y_train.csv"
)

Y_TEST_PATH: Path = (
    PROCESSED_DATA_DIR / "y_test.csv"
)

# =============================================================================
# Visualization Settings
# =============================================================================

FIGURE_SIZE: tuple[int, int] = (12, 6)

DPI: int = 300

DEFAULT_CMAP: str = "Blues"

STYLE: str = "ggplot"

# =============================================================================
# Logging Configuration
# =============================================================================

LOG_FILE: Path = LOG_DIR / "project.log"

LOG_FORMAT: str = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

LOG_LEVEL: int = logging.INFO

# =============================================================================
# Directory Management
# =============================================================================


def ensure_directories() -> None:
    """
    Create all required project directories.
    """

    directories: list[Path] = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAMPLE_DATA_DIR,
        MODEL_DIR,
        REPORT_DIR,
        VISUALS_DIR,
        EDA_VISUALS_DIR,
        MODEL_VISUALS_DIR,
        DASHBOARD_VISUALS_DIR,
        ASSETS_DIR,
        ICONS_DIR,
        NOTEBOOK_DIR,
        PAGES_DIR,
        UTILS_DIR,
        SRC_DIR,
        LOG_DIR,
        PREDICTIONS_DIR,
        ARTIFACTS_DIR,
    ]

    for directory in directories:
        print(f"Checking: {directory}")

        if directory.exists():
            print(
                f"  Exists | is_dir={directory.is_dir()} | is_file={directory.is_file()}"
            )

        directory.mkdir(parents=True, exist_ok=True)