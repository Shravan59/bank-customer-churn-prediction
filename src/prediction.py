"""
prediction.py
=============

Production-ready inference module for the
Predictive Modeling and Risk Scoring for Bank Customer Churn project.

Responsibilities
----------------
1. Load trained model artifacts.
2. Validate new customer input.
3. Apply the saved preprocessing pipeline.
4. Perform single and batch predictions.
5. Generate prediction summaries.
6. Export prediction results.

This module intentionally does NOT perform:
- model training
- preprocessing pipeline fitting
- model evaluation
- explainability
- visualization

Python Version
--------------
3.12+
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.model_persistence import (
    load_feature_columns,
    load_model,
    load_preprocessing_pipeline,
)

from config.config import (
    CATEGORICAL_COLUMNS,
    MODEL_DIR,
    PREDICTIONS_DIR,
    SAMPLE_INPUT_PATH,
)

from src.model_persistence import (
    load_feature_columns,
    load_model,
    load_preprocessing_pipeline,
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
# Constants
# =============================================================================

EXPECTED_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

NUMERICAL_COLUMNS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

BINARY_COLUMNS = [
    "HasCrCard",
    "IsActiveMember",
]

VALID_GENDERS = {
    "Male",
    "Female",
}

VALID_GEOGRAPHIES = {
    "France",
    "Germany",
    "Spain",
}

# =============================================================================
# Artifact Loading
# =============================================================================


def load_prediction_artifacts() -> tuple[
    BaseEstimator,
    Pipeline | ColumnTransformer,
    list[str],
]:
    """
    Load all artifacts required for inference.

    Returns
    -------
    tuple
        (
            trained_model,
            preprocessing_pipeline,
            feature_columns,
        )

    Raises
    ------
    RuntimeError
        If any required artifact cannot be loaded.
    """

    logger.info("Loading prediction artifacts...")

    try:

        model = load_model()

        preprocessing_pipeline = (
            load_preprocessing_pipeline()
        )

        feature_columns = (
            load_feature_columns()
        )

        logger.info(
            "Prediction artifacts loaded successfully."
        )

        return (
            model,
            preprocessing_pipeline,
            feature_columns,
        )

    except Exception as exc:

        logger.exception(
            "Unable to load prediction artifacts."
        )

        raise RuntimeError(
            "Prediction artifacts could not be loaded."
        ) from exc


# =============================================================================
# Input Validation
# =============================================================================



def validate_prediction_input(
    data: pd.DataFrame | dict[str, Any],
) -> pd.DataFrame:
    """
    Validate customer data before inference.

    Parameters
    ----------
    data : pandas.DataFrame | dict[str, Any]
        Input customer data.

    Returns
    -------
    pandas.DataFrame
        Validated dataframe.

    Raises
    ------
    TypeError
        If the input type is unsupported.

    ValueError
        If validation fails.
    """

    logger.info("Validating prediction input...")

    if isinstance(data, dict):
        data = pd.DataFrame([data])

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame or dictionary."
        )

    if data.empty:
        raise ValueError(
            "Prediction input is empty."
        )

    if data.columns.duplicated().any():
        duplicate_columns = (
            data.columns[data.columns.duplicated()]
            .tolist()
        )

        raise ValueError(
            f"Duplicate columns detected: {duplicate_columns}"
        )

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    unexpected_columns = [
        column
        for column in data.columns
        if column not in EXPECTED_COLUMNS
    ]

    if unexpected_columns:
        raise ValueError(
            f"Unexpected columns detected: {unexpected_columns}"
        )

    if data.isna().any().any():
        raise ValueError(
            "Prediction input contains missing values."
        )

    for column in BINARY_COLUMNS:

        invalid_values = (
            set(data[column].unique()) - {0, 1}
        )

        if invalid_values:
            raise ValueError(
                f"Invalid values in '{column}': "
                f"{sorted(invalid_values)}"
            )

    if not (
        set(data["Gender"].unique())
        <= VALID_GENDERS
    ):
        raise ValueError(
            "Gender contains invalid category values."
        )

    if not (
        set(data["Geography"].unique())
        <= VALID_GEOGRAPHIES
    ):
        raise ValueError(
            "Geography contains invalid category values."
        )

    numeric_columns = [
        column
        for column in NUMERICAL_COLUMNS
        if column in data.columns
    ]

    for column in numeric_columns:

        if not pd.api.types.is_numeric_dtype(
            data[column]
        ):
            raise TypeError(
                f"Column '{column}' must be numeric."
            )

    logger.info(
        "Prediction input validation completed successfully."
    )

    return data.copy()
# =============================================================================
# Input Preprocessing
# =============================================================================


def preprocess_input(
    data: pd.DataFrame | dict[str, Any],
    preprocessing_pipeline: Pipeline | ColumnTransformer,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Preprocess customer data using the saved preprocessing pipeline.

    Parameters
    ----------
    data : pandas.DataFrame | dict[str, Any]
        Customer data.

    preprocessing_pipeline : Pipeline | ColumnTransformer
        Fitted preprocessing pipeline.

    feature_columns : list[str]
        Engineered feature names generated during training.

    Returns
    -------
    pandas.DataFrame
        Processed dataframe ready for inference.

    Raises
    ------
    RuntimeError
        If preprocessing fails.
    """

    logger.info("Preprocessing prediction input...")

    validated_data = validate_prediction_input(data)

    try:

        transformed = preprocessing_pipeline.transform(
            validated_data
        )

        processed_df = pd.DataFrame(
            transformed,
            columns=feature_columns,
            index=validated_data.index,
        )

        logger.info(
            "Prediction preprocessing completed successfully."
        )

        return processed_df

    except Exception as exc:

        logger.exception(
            "Prediction preprocessing failed."
        )

        raise RuntimeError(
            "Unable to preprocess prediction input."
        ) from exc


# =============================================================================
# Single Prediction
# =============================================================================


def predict_single(
    customer: pd.DataFrame | dict[str, Any],
) -> dict[str, Any]:
    """
    Predict churn for a single customer.

    Parameters
    ----------
    customer : pandas.DataFrame | dict[str, Any]
        Single customer record.

    Returns
    -------
    dict[str, Any]
        Prediction result.
    """

    logger.info("Running single customer prediction...")

    (
        model,
        preprocessing_pipeline,
        feature_columns,
    ) = load_prediction_artifacts()

    processed = preprocess_input(
        customer,
        preprocessing_pipeline,
        feature_columns,
    )

    prediction = int(
        model.predict(processed)[0]
    )

    probability: float | None = None
    confidence: float | None = None

    if hasattr(model, "predict_proba"):

        probability = float(
            model.predict_proba(processed)[0][1]
        )

        confidence = max(
            probability,
            1.0 - probability,
        )

    elif hasattr(model, "decision_function"):

        score = float(
            model.decision_function(processed)[0]
        )

        probability = float(
            1.0 / (1.0 + np.exp(-score))
        )

        confidence = max(
            probability,
            1.0 - probability,
        )

    if probability is None:

        risk_level = "Unknown"

    elif probability < 0.30:

        risk_level = "Low"

    elif probability <= 0.70:

        risk_level = "Medium"

    else:

        risk_level = "High"

    result = {
        "prediction": prediction,
        "probability": probability,
        "risk_level": risk_level,
        "confidence": confidence,
    }

    logger.info(
        "Single prediction completed successfully."
    )

    return result


# =============================================================================
# Batch Prediction
# =============================================================================


def predict_batch(
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict churn for multiple customers.

    Parameters
    ----------
    customers : pandas.DataFrame
        Customer records.

    Returns
    -------
    pandas.DataFrame
        Original dataframe with prediction results appended.
    """

    logger.info(
        "Running batch prediction..."
    )

    (
        model,
        preprocessing_pipeline,
        feature_columns,
    ) = load_prediction_artifacts()

    validated = validate_prediction_input(
        customers
    )

    processed = preprocess_input(
        validated,
        preprocessing_pipeline,
        feature_columns,
    )

    predictions = model.predict(
        processed
    )

    probabilities = None
    confidence = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            processed
        )[:, 1]

        confidence = np.maximum(
            probabilities,
            1.0 - probabilities,
        )

    elif hasattr(model, "decision_function"):

        scores = model.decision_function(
            processed
        )

        probabilities = (
            1.0 / (1.0 + np.exp(-scores))
        )

        confidence = np.maximum(
            probabilities,
            1.0 - probabilities,
        )

    results = validated.copy()

    results["Prediction"] = predictions

    results["Probability"] = probabilities

    results["Confidence"] = confidence

    if probabilities is not None:

        results["Risk Level"] = np.select(
            [
                probabilities < 0.30,
                probabilities <= 0.70,
                probabilities > 0.70,
            ],
            [
                "Low",
                "Medium",
                "High",
            ],
            default="Unknown",
        )

    else:

        results["Risk Level"] = "Unknown"

    logger.info(
        "Batch prediction completed successfully."
    )

    return results
# =============================================================================
# Prediction Summary
# =============================================================================


def generate_prediction_summary(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """
    Generate summary statistics for batch prediction results.

    Parameters
    ----------
    predictions : pandas.DataFrame
        DataFrame returned by ``predict_batch()``.

    Returns
    -------
    dict[str, Any]
        Prediction summary.
    """

    logger.info(
        "Generating prediction summary..."
    )

    if predictions.empty:
        raise ValueError(
            "Prediction dataframe is empty."
        )

    total_customers = len(predictions)

    churn_count = int(
        (predictions["Prediction"] == 1).sum()
    )

    retention_count = int(
        (predictions["Prediction"] == 0).sum()
    )

    average_probability = None

    if predictions["Probability"].notna().any():

        average_probability = float(
            predictions["Probability"].mean()
        )

    summary = {
        "number_of_customers": total_customers,
        "predicted_churn_count": churn_count,
        "predicted_retention_count": retention_count,
        "average_probability": average_probability,
        "high_risk_customers": int(
            (predictions["Risk Level"] == "High").sum()
        ),
        "medium_risk_customers": int(
            (predictions["Risk Level"] == "Medium").sum()
        ),
        "low_risk_customers": int(
            (predictions["Risk Level"] == "Low").sum()
        ),
    }

    logger.info(
        "Prediction summary generated successfully."
    )

    return summary


# =============================================================================
# Export Predictions
# =============================================================================


def export_predictions(
    predictions: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """
    Export prediction results to CSV.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Prediction dataframe.

    output_path : Path | None, optional
        Destination CSV path.

    Returns
    -------
    pathlib.Path
        Saved CSV path.
    """

    logger.info(
        "Exporting prediction results..."
    )

    if predictions.empty:
        raise ValueError(
            "Prediction dataframe is empty."
        )

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path is None:

        output_path = (
            PREDICTIONS_DIR
            / "customer_churn_predictions.csv"
        )

    predictions.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        "Prediction results exported successfully: %s",
        output_path,
    )

    return output_path


# =============================================================================
# Prediction Demo
# =============================================================================


def run_prediction_demo() -> pd.DataFrame:
    """
    Execute a complete prediction demonstration.

    Workflow
    --------
    1. Load sample input.
    2. Generate predictions.
    3. Generate prediction summary.
    4. Export predictions.

    Returns
    -------
    pandas.DataFrame
        Prediction dataframe.
    """

    logger.info("=" * 70)
    logger.info(
        "Running prediction demonstration..."
    )
    logger.info("=" * 70)

    if not SAMPLE_INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Sample input file not found:\n{SAMPLE_INPUT_PATH}"
        )

    sample_data = pd.read_csv(
        SAMPLE_INPUT_PATH
    )

    predictions = predict_batch(
        sample_data
    )

    summary = generate_prediction_summary(
        predictions
    )

    export_predictions(
        predictions
    )

    logger.info("Prediction Summary")

    for key, value in summary.items():

        logger.info(
            "%s : %s",
            key,
            value,
        )

    logger.info(
        "Prediction demonstration completed successfully."
    )

    return predictions
# =============================================================================
# Main Workflow
# =============================================================================


def main() -> None:
    """
    Execute the complete prediction workflow.

    This function serves as the standalone entry point for the
    prediction module. It loads a sample input dataset, performs
    batch inference, generates summary statistics, exports the
    prediction results, and logs the workflow status.

    Raises
    ------
    Exception
        Propagates any exception encountered during prediction.
    """

    try:

        logger.info("=" * 70)
        logger.info("Starting Customer Churn Prediction Pipeline")
        logger.info("=" * 70)

        predictions = run_prediction_demo()

        summary = generate_prediction_summary(
            predictions
        )

        logger.info("")
        logger.info("=" * 70)
        logger.info("Prediction Summary")
        logger.info("=" * 70)

        logger.info(
            "Customers Processed      : %d",
            summary["number_of_customers"],
        )

        logger.info(
            "Predicted Churn         : %d",
            summary["predicted_churn_count"],
        )

        logger.info(
            "Predicted Retention     : %d",
            summary["predicted_retention_count"],
        )

        if summary["average_probability"] is not None:

            logger.info(
                "Average Churn Probability : %.4f",
                summary["average_probability"],
            )

        logger.info(
            "High Risk Customers     : %d",
            summary["high_risk_customers"],
        )

        logger.info(
            "Medium Risk Customers   : %d",
            summary["medium_risk_customers"],
        )

        logger.info(
            "Low Risk Customers      : %d",
            summary["low_risk_customers"],
        )

        logger.info("=" * 70)
        logger.info("Prediction Pipeline Completed Successfully")
        logger.info("=" * 70)

    except Exception as exc:

        logger.exception(
            "Prediction pipeline failed: %s",
            exc,
        )

        raise


if __name__ == "__main__":
    main()