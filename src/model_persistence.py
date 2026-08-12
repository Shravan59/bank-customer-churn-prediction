"""
model_persistence.py
====================

Production-ready model persistence module for the
Predictive Modeling and Risk Scoring for Bank Customer Churn project.

Responsibilities
----------------
1. Save machine learning artifacts.
2. Load persisted artifacts.
3. Validate artifact integrity.
4. Generate model metadata.
5. Provide centralized persistence utilities.

This module intentionally does NOT perform:
- preprocessing
- feature engineering
- model training
- model evaluation
- explainability
- prediction

Python Version
--------------
3.12+
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pandas as pd

from config.config import (
    MODEL_DIR,
    PROJECT_NAME,
    PROJECT_VERSION,
    RANDOM_STATE,
    TARGET_COLUMN,
)

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# =============================================================================
# Artifact Paths
# =============================================================================

BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"

PREPROCESSING_PIPELINE_PATH = (
    MODEL_DIR / "preprocessing_pipeline.pkl"
)

FEATURE_COLUMNS_PATH = (
    MODEL_DIR / "feature_columns.pkl"
)

MODEL_METRICS_PATH = (
    MODEL_DIR / "model_metrics.json"
)

MODEL_METADATA_PATH = (
    MODEL_DIR / "model_metadata.json"
)

TRAINING_RESULTS_PATH = MODEL_DIR / "training_results.csv"

# =============================================================================
# Generic Utilities
# =============================================================================


def artifact_exists(path: Path) -> bool:
    """
    Check whether an artifact exists.

    Parameters
    ----------
    path : Path
        Artifact path.

    Returns
    -------
    bool
        True if artifact exists.
    """

    exists = path.exists()

    logger.info(
        "Artifact %s: %s",
        path.name,
        "FOUND" if exists else "NOT FOUND",
    )

    return exists


# =============================================================================
# Validation Utilities
# =============================================================================


def validate_model(
    model: BaseEstimator,
) -> None:
    """
    Validate a trained model.

    Parameters
    ----------
    model : BaseEstimator
        Trained model.

    Raises
    ------
    ValueError
        If validation fails.
    """

    if model is None:
        raise ValueError(
            "Model cannot be None."
        )

    if not isinstance(
        model,
        BaseEstimator,
    ):
        raise TypeError(
            "Object is not a valid scikit-learn estimator."
        )

    if not hasattr(model, "predict"):
        raise ValueError(
            "Estimator does not implement predict()."
        )

    logger.info(
        "Model validation passed."
    )


def validate_pipeline(
    pipeline: Pipeline | ColumnTransformer,
) -> None:
    """
    Validate preprocessing pipeline.

    Parameters
    ----------
    pipeline : Pipeline | ColumnTransformer
        Fitted preprocessing pipeline.

    Raises
    ------
    ValueError
        If validation fails.
    """

    if pipeline is None:
        raise ValueError(
            "Pipeline cannot be None."
        )

    if not isinstance(
        pipeline,
        (Pipeline, ColumnTransformer),
    ):
        raise TypeError(
            "Invalid preprocessing pipeline."
        )

    logger.info(
        "Pipeline validation passed."
    )


def validate_feature_columns(
    feature_columns: list[str],
) -> None:
    """
    Validate feature column list.

    Parameters
    ----------
    feature_columns : list[str]
        Feature names.

    Raises
    ------
    ValueError
        If validation fails.
    """

    if not feature_columns:
        raise ValueError(
            "Feature column list cannot be empty."
        )

    if not all(
        isinstance(col, str)
        for col in feature_columns
    ):
        raise TypeError(
            "Feature names must be strings."
        )

    logger.info(
        "Feature column validation passed."
    )

# =============================================================================
# Save Functions
# =============================================================================


def save_model(
    model: BaseEstimator,
    model_path: Path = BEST_MODEL_PATH,
) -> Path:
    """
    Save the trained machine learning model.

    Parameters
    ----------
    model : BaseEstimator
        Trained model.

    model_path : Path, default=BEST_MODEL_PATH
        Destination path.

    Returns
    -------
    Path
        Saved model path.
    """

    logger.info("Saving trained model...")

    validate_model(model)

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if model_path.exists():
        logger.warning(
            "Replacing existing model artifact: %s",
            model_path.name,
        )

    joblib.dump(
        model,
        model_path,
    )

    logger.info(
        "Model saved successfully: %s",
        model_path,
    )

    return model_path


def save_preprocessing_pipeline(
    pipeline: Pipeline | ColumnTransformer,
    pipeline_path: Path = PREPROCESSING_PIPELINE_PATH,
) -> Path:
    """
    Save the preprocessing pipeline.

    Parameters
    ----------
    pipeline : Pipeline | ColumnTransformer
        Fitted preprocessing pipeline.

    pipeline_path : Path, default=PREPROCESSING_PIPELINE_PATH
        Destination path.

    Returns
    -------
    Path
        Saved pipeline path.
    """

    logger.info("Saving preprocessing pipeline...")

    validate_pipeline(pipeline)

    pipeline_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if pipeline_path.exists():
        logger.warning(
            "Replacing existing preprocessing pipeline."
        )

    joblib.dump(
        pipeline,
        pipeline_path,
    )

    logger.info(
        "Pipeline saved successfully: %s",
        pipeline_path,
    )

    return pipeline_path


def save_feature_columns(
    feature_columns: list[str],
    feature_path: Path = FEATURE_COLUMNS_PATH,
) -> Path:
    """
    Save selected feature column names.

    Parameters
    ----------
    feature_columns : list[str]
        Feature names.

    feature_path : Path, default=FEATURE_COLUMNS_PATH
        Destination path.

    Returns
    -------
    Path
        Saved artifact path.
    """

    logger.info("Saving feature columns...")

    validate_feature_columns(
        feature_columns,
    )

    feature_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if feature_path.exists():
        logger.warning(
            "Replacing existing feature column artifact."
        )

    joblib.dump(
        feature_columns,
        feature_path,
    )

    logger.info(
        "Feature columns saved successfully."
    )

    return feature_path


def save_model_metrics(
    metrics: dict[str, Any],
    metrics_path: Path = MODEL_METRICS_PATH,
) -> Path:
    """
    Save evaluation metrics as JSON.

    Parameters
    ----------
    metrics : dict[str, Any]
        Evaluation metrics.

    metrics_path : Path, default=MODEL_METRICS_PATH
        Destination path.

    Returns
    -------
    Path
        Saved metrics file.
    """

    logger.info("Saving model metrics...")

    if metrics is None:
        raise ValueError(
            "Metrics dictionary cannot be None."
        )

    if not isinstance(metrics, dict):
        raise TypeError(
            "Metrics must be provided as a dictionary."
        )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if metrics_path.exists():
        logger.warning(
            "Replacing existing metrics artifact."
        )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    logger.info(
        "Model metrics saved successfully."
    )

    return metrics_path

# =============================================================================
# Load Functions
# =============================================================================


def load_model_metrics(
    metrics_path: Path = TRAINING_RESULTS_PATH,
) -> dict[str, Any]:
    """
    Load model metrics from training_results.csv.

    Parameters
    ----------
    metrics_path : Path
        Path to training_results.csv.

    Returns
    -------
    dict[str, Any]
        Dictionary containing model metrics.
    """

    logger.info("Loading model metrics...")

    if not artifact_exists(metrics_path):
        raise FileNotFoundError(
            f"Metrics artifact not found: {metrics_path}"
        )

    try:

        metrics_df = pd.read_csv(metrics_path)

        if metrics_df.empty:
            raise ValueError(
                "Training results file is empty."
            )

        # Take the best/first model row
        metrics = metrics_df.iloc[0].to_dict()

        logger.info(
            "Model metrics loaded successfully."
        )

        return metrics

    except Exception as exc:

        logger.exception(
            "Unable to load metrics."
        )

        raise RuntimeError(
            "Failed to load model metrics."
        ) from exc

def load_model(
    model_path: Path = BEST_MODEL_PATH,
) -> BaseEstimator:
    """
    Load trained machine learning model.
    """

    logger.info("Loading trained model...")

    if not artifact_exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    try:

        model = joblib.load(model_path)

        validate_model(model)

        logger.info(
            "Model loaded successfully."
        )

        return model

    except Exception as exc:

        logger.exception(
            "Unable to load trained model."
        )

        raise RuntimeError(
            "Failed to deserialize trained model."
        ) from exc


def load_preprocessing_pipeline(
    pipeline_path: Path = PREPROCESSING_PIPELINE_PATH,
) -> Pipeline | ColumnTransformer:
    """
    Load the preprocessing pipeline.

    Parameters
    ----------
    pipeline_path : Path, default=PREPROCESSING_PIPELINE_PATH
        Serialized pipeline path.

    Returns
    -------
    Pipeline | ColumnTransformer
        Loaded preprocessing pipeline.
    """

    logger.info("Loading preprocessing pipeline...")

    if not artifact_exists(pipeline_path):
        raise FileNotFoundError(
            f"Pipeline artifact not found: {pipeline_path}"
        )

    try:

        pipeline = joblib.load(
            pipeline_path
        )

        validate_pipeline(
            pipeline,
        )

        logger.info(
            "Preprocessing pipeline loaded successfully."
        )

        return pipeline

    except Exception as exc:

        logger.exception(
            "Unable to load preprocessing pipeline."
        )

        raise RuntimeError(
            "Failed to deserialize preprocessing pipeline."
        ) from exc


def load_feature_columns(
    feature_path: Path = FEATURE_COLUMNS_PATH,
) -> list[str]:
    """
    Load saved feature column names.

    Parameters
    ----------
    feature_path : Path, default=FEATURE_COLUMNS_PATH
        Serialized feature column path.

    Returns
    -------
    list[str]
        Feature column names.
    """

    logger.info(
        "Loading feature columns..."
    )

    if not artifact_exists(feature_path):
        raise FileNotFoundError(
            f"Feature artifact not found: {feature_path}"
        )

    try:

        feature_columns = joblib.load(
            feature_path
        )

        validate_feature_columns(
            feature_columns
        )

        logger.info(
            "Feature columns loaded successfully."
        )

        return feature_columns

    except Exception as exc:

        logger.exception(
            "Unable to load feature columns."
        )

        raise RuntimeError(
            "Failed to deserialize feature columns."
        ) from exc


def load_model_metrics(
    metrics_path: Path = MODEL_METRICS_PATH,
) -> dict[str, Any]:
    """
    Load saved model metrics.

    Parameters
    ----------
    metrics_path : Path, default=MODEL_METRICS_PATH
        JSON metrics file.

    Returns
    -------
    dict[str, Any]
        Evaluation metrics dictionary.
    """

    logger.info(
        "Loading model metrics..."
    )

    if not artifact_exists(metrics_path):
        raise FileNotFoundError(
            f"Metrics artifact not found: {metrics_path}"
        )

    try:

        with metrics_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            metrics = json.load(file)

        if not isinstance(
            metrics,
            dict,
        ):
            raise TypeError(
                "Metrics file is invalid."
            )

        logger.info(
            "Model metrics loaded successfully."
        )

        return metrics

    except Exception as exc:

        logger.exception(
            "Unable to load metrics."
        )

        raise RuntimeError(
            "Failed to deserialize model metrics."
        ) from exc


# =============================================================================
# Metadata Utilities
# =============================================================================


def generate_model_metadata(
    model: BaseEstimator,
    feature_columns: list[str],
) -> dict[str, Any]:
    """
    Generate model metadata.

    Parameters
    ----------
    model : BaseEstimator
        Trained model.

    feature_columns : list[str]
        Ordered feature names.

    Returns
    -------
    dict[str, Any]
        Model metadata dictionary.
    """

    logger.info(
        "Generating model metadata..."
    )

    validate_model(model)

    validate_feature_columns(
        feature_columns
    )

    metadata = {
        "project_name": PROJECT_NAME,
        "project_version": PROJECT_VERSION,
        "training_timestamp": datetime.now().isoformat(),
        "model_type": type(model).__name__,
        "number_of_input_features": len(feature_columns),
        "feature_names": feature_columns,
        "target_column": TARGET_COLUMN,
        "random_seed": RANDOM_STATE,
    }

    logger.info(
        "Model metadata generated successfully."
    )

    return metadata

    

def save_model_metadata(
    metadata: dict[str, Any],
    metadata_path: Path = MODEL_METADATA_PATH,
) -> Path:
    """
    Save model metadata.

    Parameters
    ----------
    metadata : dict[str, Any]
        Metadata dictionary.

    metadata_path : Path, default=MODEL_METADATA_PATH
        Destination JSON path.

    Returns
    -------
    Path
        Saved metadata file.
    """

    logger.info(
        "Saving model metadata..."
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    
# =============================================================================
# Orchestration Functions
# =============================================================================


def save_all_artifacts(
    *,
    model: BaseEstimator,
    preprocessing_pipeline: Pipeline | ColumnTransformer,
    feature_columns: list[str],
    metrics: dict[str, Any],
) -> dict[str, Path]:
    """
    Save all machine learning artifacts.

    Parameters
    ----------
    model : BaseEstimator
        Trained machine learning model.

    preprocessing_pipeline : Pipeline | ColumnTransformer
        Fitted preprocessing pipeline.

    feature_columns : list[str]
        Ordered feature names.

    metrics : dict[str, Any]
        Model evaluation metrics.

    Returns
    -------
    dict[str, Path]
        Dictionary containing the saved artifact paths.
    """

    logger.info("=" * 70)
    logger.info("Saving all model artifacts...")
    logger.info("=" * 70)

    model_path = save_model(model)

    pipeline_path = save_preprocessing_pipeline(
        preprocessing_pipeline
    )

    feature_path = save_feature_columns(
        feature_columns
    )

    metrics_path = save_model_metrics(
        metrics
    )

    metadata = generate_model_metadata(
        model=model,
        feature_columns=feature_columns,
    )

    metadata_path = save_model_metadata(
        metadata
    )

    logger.info("All artifacts saved successfully.")

    return {
        "model": model_path,
        "pipeline": pipeline_path,
        "feature_columns": feature_path,
        "metrics": metrics_path,
        "metadata": metadata_path,
    }


def load_all_artifacts() -> dict[str, Any]:
    """
    Load every persisted artifact.

    Returns
    -------
    dict[str, Any]
        Dictionary containing all loaded artifacts.
    """

    logger.info("=" * 70)
    logger.info("Loading all model artifacts...")
    logger.info("=" * 70)

    model = joblib.load(BEST_MODEL_PATH)

    pipeline = load_preprocessing_pipeline()

    feature_columns = load_feature_columns()

    metrics = load_model_metrics()

    if not artifact_exists(MODEL_METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata artifact not found: {MODEL_METADATA_PATH}"
        )

    try:

        with MODEL_METADATA_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(file)

    except Exception as exc:

        logger.exception(
            "Unable to load model metadata."
        )

        raise RuntimeError(
            "Failed to deserialize model metadata."
        ) from exc

    logger.info(
        "All artifacts loaded successfully."
    )

    return {
        "model": model,
        "preprocessing_pipeline": pipeline,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "metadata": metadata,
    }


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """
    Demonstrate artifact loading.

    Notes
    -----
    This module is intended to be imported by the training,
    evaluation, explainability, visualization, and prediction
    modules. Running it directly performs an integrity check
    by attempting to load all persisted artifacts.
    """

    try:

        artifacts = load_all_artifacts()

        logger.info("=" * 70)
        logger.info("Artifact Integrity Check Completed Successfully")
        logger.info("=" * 70)

        logger.info(
            "Loaded Model           : %s",
            type(artifacts["model"]).__name__,
        )

        logger.info(
            "Loaded Pipeline        : %s",
            type(
                artifacts["preprocessing_pipeline"]
            ).__name__,
        )

        logger.info(
            "Input Features         : %d",
            len(
                artifacts["feature_columns"]
            ),
        )

        logger.info(
            "Target Column          : %s",
            artifacts["metadata"].get(
                "target_column"
            ),
        )

        logger.info(
            "Project               : %s",
            artifacts["metadata"].get(
                "project_name"
            ),
        )

        logger.info(
            "Version               : %s",
            artifacts["metadata"].get(
                "project_version"
            ),
        )

        logger.info(
            "Training Timestamp    : %s",
            artifacts["metadata"].get(
                "training_timestamp"
            ),
        )

        logger.info("=" * 70)

    except Exception as exc:

        logger.exception(
            "Model persistence module failed: %s",
            exc,
        )
        raise


if __name__ == "__main__":
    main()