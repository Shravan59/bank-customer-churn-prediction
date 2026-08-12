"""
model_evaluation.py
===================

Production-ready model evaluation module for the
Predictive Modeling and Risk Scoring for Bank Customer Churn project.

Responsibilities
----------------
1. Load the trained model.
2. Load the processed test dataset.
3. Validate evaluation inputs.
4. Generate predictions and probabilities.
5. Compute evaluation metrics.
6. Generate evaluation visualizations.
7. Save evaluation artifacts.

This module intentionally does NOT perform:
- data preprocessing
- feature engineering
- model training
- hyperparameter tuning
- model explainability
- prediction deployment

Author
------
Internship Project

Python Version
--------------
3.11+
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    REPORT_DIR,
    TARGET_COLUMN,
    VISUALS_DIR,
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
# Project Paths
# =============================================================================

BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"

X_TEST_PATH = PROCESSED_DATA_DIR / "X_test_engineered.csv"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.csv"

EVALUATION_REPORT_PATH = (
    REPORT_DIR / "model_evaluation_report.csv"
)

CLASSIFICATION_REPORT_PATH = (
    REPORT_DIR / "classification_report.csv"
)

CONFUSION_MATRIX_PATH = (
    VISUALS_DIR / "evaluation" / "confusion_matrix.png"
)

ROC_CURVE_PATH = (
    VISUALS_DIR / "evaluation" / "roc_curve.png"
)

PR_CURVE_PATH = (
    VISUALS_DIR / "evaluation" / "precision_recall_curve.png"
)

# =============================================================================
# Model Loading
# =============================================================================


def load_model(
    model_path: Path = BEST_MODEL_PATH,
) -> BaseEstimator:
    """
    Load the trained machine learning model.

    Parameters
    ----------
    model_path : Path, optional
        Path to the serialized model.

    Returns
    -------
    BaseEstimator
        Loaded trained model.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.

    TypeError
        If the loaded object is not a scikit-learn estimator.
    """

    logger.info("Loading trained model...")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    model = joblib.load(model_path)

    if not isinstance(model, BaseEstimator):
        raise TypeError(
            "Loaded object is not a valid scikit-learn estimator."
        )

    logger.info("Model loaded successfully.")

    return model


# =============================================================================
# Test Data Loading
# =============================================================================


def load_test_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load engineered testing dataset.

    Returns
    -------
    tuple
        (
            X_test,
            y_test
        )

    Raises
    ------
    FileNotFoundError
        If any required dataset is missing.
    """

    logger.info("Loading testing dataset...")

    required_files = [
        X_TEST_PATH,
        Y_TEST_PATH,
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{file_path}"
            )

    X_test = pd.read_csv(X_TEST_PATH)

    y_test = pd.read_csv(
        Y_TEST_PATH
    )[TARGET_COLUMN]

    logger.info(
        "Testing samples: %d",
        len(X_test),
    )

    logger.info(
        "Number of features: %d",
        X_test.shape[1],
    )

    return X_test, y_test


# =============================================================================
# Input Validation
# =============================================================================


def validate_inputs(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Validate evaluation inputs.

    Parameters
    ----------
    model : BaseEstimator
        Trained model.

    X_test : pd.DataFrame
        Testing feature matrix.

    y_test : pd.Series
        Ground-truth labels.

    Raises
    ------
    ValueError
        If validation fails.
    """

    logger.info("Validating evaluation inputs...")

    if model is None:
        raise ValueError(
            "Loaded model is None."
        )

    if X_test.empty:
        raise ValueError(
            "Testing feature dataset is empty."
        )

    if y_test.empty:
        raise ValueError(
            "Testing target dataset is empty."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "Feature and target sizes do not match."
        )

    if X_test.isna().any().any():
        raise ValueError(
            "Testing features contain missing values."
        )

    if y_test.isna().any():
        raise ValueError(
            "Testing target contains missing values."
        )

    if not hasattr(model, "predict"):
        raise ValueError(
            "Loaded model does not implement predict()."
        )

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Loaded model does not implement predict_proba()."
        )

    logger.info(
        "Evaluation input validation completed successfully."
    )

# =============================================================================
# Imports Required for Model Evaluation
# =============================================================================

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# =============================================================================
# Prediction Generation
# =============================================================================


def generate_predictions(
    model: BaseEstimator,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions and prediction probabilities.

    Parameters
    ----------
    model : BaseEstimator
        Trained classification model.

    X_test : pd.DataFrame
        Testing feature matrix.

    Returns
    -------
    tuple
        (
            predicted_labels,
            predicted_probabilities
        )
    """

    logger.info("Generating predictions...")

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    logger.info(
        "Generated predictions for %d observations.",
        len(predictions),
    )

    return predictions, probabilities


# =============================================================================
# Metric Computation
# =============================================================================


def compute_metrics(
    y_test: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[
    dict[str, float],
    np.ndarray,
    pd.DataFrame,
]:
    """
    Compute evaluation metrics.

    Parameters
    ----------
    y_test : pd.Series
        Ground-truth labels.

    predictions : np.ndarray
        Predicted labels.

    probabilities : np.ndarray
        Predicted probabilities.

    Returns
    -------
    tuple
        (
            metrics_dictionary,
            confusion_matrix_array,
            classification_report_dataframe
        )
    """

    logger.info("Computing evaluation metrics...")

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    report = pd.DataFrame(
        classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    logger.info("Evaluation Metrics")

    for metric, value in metrics.items():

        logger.info(
            "%-12s : %.4f",
            metric,
            value,
        )

    return (
        metrics,
        matrix,
        report,
    )


# =============================================================================
# Confusion Matrix Visualization
# =============================================================================


def plot_confusion_matrix(
    matrix: np.ndarray,
) -> None:
    """
    Generate and save the confusion matrix figure.

    Parameters
    ----------
    matrix : np.ndarray
        Confusion matrix.
    """

    logger.info("Saving confusion matrix...")

    CONFUSION_MATRIX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(6, 6),
    )

    image = axis.imshow(
        matrix,
        interpolation="nearest",
    )

    figure.colorbar(image)

    axis.set_title("Confusion Matrix")

    axis.set_xlabel("Predicted Label")

    axis.set_ylabel("Actual Label")

    tick_labels = ["Retained", "Exited"]

    axis.set_xticks([0, 1])

    axis.set_yticks([0, 1])

    axis.set_xticklabels(tick_labels)

    axis.set_yticklabels(tick_labels)

    for row in range(matrix.shape[0]):

        for column in range(matrix.shape[1]):

            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=11,
            )

    figure.tight_layout()

    figure.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    logger.info("Confusion matrix saved successfully.")

# =============================================================================
# ROC Curve
# =============================================================================


def plot_roc_curve(
    y_test: pd.Series,
    probabilities: np.ndarray,
) -> float:
    """
    Generate and save the ROC Curve.

    Parameters
    ----------
    y_test : pd.Series
        Ground-truth labels.

    probabilities : np.ndarray
        Predicted probabilities.

    Returns
    -------
    float
        ROC-AUC score calculated from the ROC curve.
    """

    logger.info("Generating ROC Curve...")

    ROC_CURVE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        probabilities,
    )

    roc_auc = auc(
        false_positive_rate,
        true_positive_rate,
    )

    figure, axis = plt.subplots(
        figsize=(7, 6),
    )

    axis.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f"ROC-AUC = {roc_auc:.4f}",
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
    )

    axis.set_xlabel("False Positive Rate")

    axis.set_ylabel("True Positive Rate")

    axis.set_title("Receiver Operating Characteristic")

    axis.legend(loc="lower right")

    axis.grid(alpha=0.30)

    figure.tight_layout()

    figure.savefig(
        ROC_CURVE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    logger.info("ROC Curve saved successfully.")

    return roc_auc


# =============================================================================
# Precision-Recall Curve
# =============================================================================


def plot_precision_recall_curve(
    y_test: pd.Series,
    probabilities: np.ndarray,
) -> float:
    """
    Generate and save the Precision-Recall Curve.

    Parameters
    ----------
    y_test : pd.Series
        Ground-truth labels.

    probabilities : np.ndarray
        Predicted probabilities.

    Returns
    -------
    float
        Area under the Precision-Recall Curve.
    """

    logger.info("Generating Precision-Recall Curve...")

    PR_CURVE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities,
    )

    pr_auc = auc(
        recall,
        precision,
    )

    figure, axis = plt.subplots(
        figsize=(7, 6),
    )

    axis.plot(
        recall,
        precision,
        linewidth=2,
        label=f"PR-AUC = {pr_auc:.4f}",
    )

    axis.set_xlabel("Recall")

    axis.set_ylabel("Precision")

    axis.set_title("Precision-Recall Curve")

    axis.legend(loc="lower left")

    axis.grid(alpha=0.30)

    figure.tight_layout()

    figure.savefig(
        PR_CURVE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    logger.info(
        "Precision-Recall Curve saved successfully."
    )

    return pr_auc


# =============================================================================
# Save Evaluation Results
# =============================================================================


def save_evaluation_results(
    metrics: dict[str, float],
    report: pd.DataFrame,
) -> None:
    """
    Save evaluation metrics and classification report.

    Parameters
    ----------
    metrics : dict[str, float]
        Model evaluation metrics.

    report : pd.DataFrame
        Classification report.
    """

    logger.info("Saving evaluation results...")

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_dataframe = pd.DataFrame(
        [metrics]
    )

    metrics_dataframe.to_csv(
        EVALUATION_REPORT_PATH,
        index=False,
    )

    report.to_csv(
        CLASSIFICATION_REPORT_PATH,
        index=True,
    )

    logger.info(
        "Evaluation metrics saved successfully."
    )

# =============================================================================
# Model Evaluation Pipeline
# =============================================================================


def run_model_evaluation() -> dict[str, Any]:
    """
    Execute the complete model evaluation workflow.

    Workflow
    --------
    1. Load the trained model.
    2. Load the processed test dataset.
    3. Validate evaluation inputs.
    4. Generate predictions and probabilities.
    5. Compute evaluation metrics.
    6. Generate evaluation visualizations.
    7. Save evaluation reports.

    Returns
    -------
    dict[str, Any]
        Dictionary containing evaluation results.
    """

    logger.info("=" * 70)
    logger.info("Starting Model Evaluation Pipeline")
    logger.info("=" * 70)

    model = load_model()

    X_test, y_test = load_test_data()

    validate_inputs(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    predictions, probabilities = generate_predictions(
        model=model,
        X_test=X_test,
    )

    (
        metrics,
        confusion_matrix_array,
        classification_report_df,
    ) = compute_metrics(
        y_test=y_test,
        predictions=predictions,
        probabilities=probabilities,
    )

    roc_auc = plot_roc_curve(
        y_test=y_test,
        probabilities=probabilities,
    )

    pr_auc = plot_precision_recall_curve(
        y_test=y_test,
        probabilities=probabilities,
    )

    plot_confusion_matrix(
        matrix=confusion_matrix_array,
    )

    save_evaluation_results(
        metrics=metrics,
        report=classification_report_df,
    )

    logger.info("=" * 70)
    logger.info("Model Evaluation Completed Successfully")
    logger.info("=" * 70)

    logger.info(
        "Accuracy  : %.4f",
        metrics["Accuracy"],
    )

    logger.info(
        "Precision : %.4f",
        metrics["Precision"],
    )

    logger.info(
        "Recall    : %.4f",
        metrics["Recall"],
    )

    logger.info(
        "F1 Score  : %.4f",
        metrics["F1 Score"],
    )

    logger.info(
        "ROC-AUC   : %.4f",
        roc_auc,
    )

    logger.info(
        "PR-AUC    : %.4f",
        pr_auc,
    )

    return {
        "model": model,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix_array,
        "classification_report": classification_report_df,
        "predictions": predictions,
        "probabilities": probabilities,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """
    Execute the model evaluation module as a standalone script.
    """

    try:

        results = run_model_evaluation()

        logger.info("\nEvaluation Metrics")
        logger.info("-" * 70)

        for metric, value in results["metrics"].items():
            logger.info("%-12s : %.4f", metric, value)

        logger.info("-" * 70)
        logger.info(
            "Evaluation reports saved to: %s",
            REPORT_DIR,
        )

        logger.info(
            "Evaluation visualizations saved to: %s",
            VISUALS_DIR / "evaluation",
        )

    except Exception as exc:

        logger.exception(
            "Model evaluation failed: %s",
            exc,
        )
        raise


if __name__ == "__main__":
    main()