"""
model_visualization.py
======================

Production-ready visualization module for the
Predictive Modeling and Risk Scoring for Bank Customer Churn project.

Responsibilities
----------------
1. Load saved evaluation artifacts.
2. Load explainability artifacts.
3. Validate loaded artifacts.
4. Generate publication-quality model visualizations.
5. Save all figures under visuals/model/.

This module intentionally does NOT perform:
- data preprocessing
- feature engineering
- model training
- model evaluation
- model explainability
- prediction

Python Version
--------------
3.12+
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from config.config import (
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    REPORT_DIR,
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

MODEL_VISUALS_DIR = VISUALS_DIR / "model"

BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"

FEATURE_IMPORTANCE_PATH = (
    REPORT_DIR / "feature_importance.csv"
)

MODEL_COMPARISON_PATH = (
    MODEL_DIR / "model_comparison.csv"
)

EVALUATION_METRICS_PATH = (
    REPORT_DIR / "model_evaluation_report.csv"
)

CLASSIFICATION_REPORT_PATH = (
    REPORT_DIR / "classification_report.csv"
)

X_TEST_PATH = (
    PROCESSED_DATA_DIR / "X_test_engineered.csv"
)

Y_TEST_PATH = (
    PROCESSED_DATA_DIR / "y_test.csv"
)

SHAP_VALUES_PATH = (
    MODEL_DIR / "shap_values.pkl"
)

# =============================================================================
# Artifact Loading
# =============================================================================


def load_artifacts() -> dict[str, Any]:
    """
    Load all artifacts required for visualization.

    Returns
    -------
    dict[str, Any]
        Dictionary containing loaded artifacts.

    Raises
    ------
    FileNotFoundError
        If any required artifact is missing.
    """

    logger.info("Loading visualization artifacts...")

    required_files = [
        BEST_MODEL_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
        FEATURE_IMPORTANCE_PATH,
        EVALUATION_METRICS_PATH,
        CLASSIFICATION_REPORT_PATH,
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required artifact not found:\n{file_path}"
            )

    artifacts: dict[str, Any] = {
        "model": joblib.load(BEST_MODEL_PATH),
        "X_test": pd.read_csv(X_TEST_PATH),
        "y_test": pd.read_csv(Y_TEST_PATH).squeeze(),
        "feature_importance": pd.read_csv(
            FEATURE_IMPORTANCE_PATH
        ),
        "evaluation_metrics": pd.read_csv(
            EVALUATION_METRICS_PATH
        ),
        "classification_report": pd.read_csv(
            CLASSIFICATION_REPORT_PATH
        ),
    }

    if MODEL_COMPARISON_PATH.exists():

        artifacts["model_comparison"] = pd.read_csv(
            MODEL_COMPARISON_PATH
        )

    else:

        artifacts["model_comparison"] = None

    if SHAP_VALUES_PATH.exists():

        artifacts["shap_values"] = joblib.load(
            SHAP_VALUES_PATH
        )

    else:

        artifacts["shap_values"] = None

    logger.info("Artifacts loaded successfully.")

    return artifacts


# =============================================================================
# Validation
# =============================================================================


def validate_inputs(
    artifacts: dict[str, Any],
) -> None:
    """
    Validate loaded visualization artifacts.

    Parameters
    ----------
    artifacts : dict[str, Any]
        Loaded project artifacts.

    Raises
    ------
    ValueError
        If any required artifact is invalid.
    """

    logger.info("Validating visualization artifacts...")

    if artifacts["X_test"].empty:
        raise ValueError(
            "X_test dataset is empty."
        )

    if artifacts["y_test"].empty:
        raise ValueError(
            "y_test dataset is empty."
        )

    if artifacts["feature_importance"].empty:
        raise ValueError(
            "Feature importance table is empty."
        )

    if len(artifacts["X_test"]) != len(
        artifacts["y_test"]
    ):
        raise ValueError(
            "Mismatch between X_test and y_test."
        )

    MODEL_VISUALS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Visualization artifact validation completed."
    )

# =============================================================================
# Confusion Matrix Visualization
# =============================================================================

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
)


def plot_confusion_matrix(
    artifacts: dict[str, Any],
) -> None:
    """
    Generate and save the confusion matrix.

    Parameters
    ----------
    artifacts : dict[str, Any]
        Loaded visualization artifacts.
    """

    logger.info("Generating confusion matrix...")

    model = artifacts["model"]
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]

    predictions = model.predict(X_test)

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    fig, ax = plt.subplots(
        figsize=(6, 6),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Retained", "Exited"],
    )

    display.plot(
        ax=ax,
        colorbar=True,
    )

    ax.set_title(
        "Confusion Matrix",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        MODEL_VISUALS_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        "Confusion matrix saved successfully."
    )


# =============================================================================
# ROC Curve
# =============================================================================


def plot_roc_curve(
    artifacts: dict[str, Any],
) -> None:
    """
    Generate and save the ROC curve.

    Parameters
    ----------
    artifacts : dict[str, Any]
        Loaded visualization artifacts.
    """

    logger.info("Generating ROC curve...")

    model = artifacts["model"]
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities,
    )

    roc_auc = auc(
        fpr,
        tpr,
    )

    fig, ax = plt.subplots(
        figsize=(7, 6),
    )

    RocCurveDisplay(
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc,
        estimator_name="Best Model",
    ).plot(ax=ax)

    ax.grid(alpha=0.30)

    ax.set_title(
        f"ROC Curve (AUC = {roc_auc:.4f})",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        MODEL_VISUALS_DIR / "roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        "ROC curve saved successfully."
    )


# =============================================================================
# Precision-Recall Curve
# =============================================================================


def plot_precision_recall_curve(
    artifacts: dict[str, Any],
) -> None:
    """
    Generate and save the Precision-Recall curve.

    Parameters
    ----------
    artifacts : dict[str, Any]
        Loaded visualization artifacts.
    """

    logger.info(
        "Generating Precision-Recall curve..."
    )

    model = artifacts["model"]
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities,
    )

    pr_auc = auc(
        recall,
        precision,
    )

    fig, ax = plt.subplots(
        figsize=(7, 6),
    )

    PrecisionRecallDisplay(
        precision=precision,
        recall=recall,
    ).plot(ax=ax)

    ax.grid(alpha=0.30)

    ax.set_title(
        f"Precision-Recall Curve (AUC = {pr_auc:.4f})",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        MODEL_VISUALS_DIR / "precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        "Precision-Recall curve saved successfully."
    )


# =============================================================================
# Feature Importance Visualization
# =============================================================================


def plot_feature_importance(
    artifacts: dict[str, Any],
) -> None:
    """
    Generate and save the Top-20 feature importance chart.

    Parameters
    ----------
    artifacts : dict[str, Any]
        Loaded visualization artifacts.
    """

    logger.info(
        "Generating feature importance plot..."
    )

    importance_df = (
        artifacts["feature_importance"]
        .sort_values(
            by="Importance",
            ascending=False,
        )
        .head(20)
    )

    fig, ax = plt.subplots(
        figsize=(10, 7),
    )

    ax.barh(
        importance_df["Feature"][::-1],
        importance_df["Importance"][::-1],
    )

    ax.set_xlabel("Importance")

    ax.set_ylabel("Feature")

    ax.set_title(
        "Top 20 Feature Importance",
        fontsize=14,
    )

    ax.grid(
        axis="x",
        alpha=0.30,
    )

    fig.tight_layout()

    fig.savefig(
        MODEL_VISUALS_DIR / "feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        "Feature importance plot saved successfully."
    )

# =============================================================================
# SHAP Summary Visualization
# =============================================================================


def plot_shap_summary(
    artifacts: dict[str, Any],
) -> None:
    """
    Generate and save SHAP summary and SHAP feature importance plots.

    Parameters
    ----------
    artifacts : dict[str, Any]
        Loaded visualization artifacts.
    """

    logger.info("Generating SHAP visualizations...")

    shap_values = artifacts.get("shap_values")

    if shap_values is None:
        logger.warning(
            "SHAP values not found. Skipping SHAP visualizations."
        )
        return

    X_test = artifacts["X_test"]

    # -------------------------------------------------------------------------
    # SHAP Summary Plot
    # -------------------------------------------------------------------------

    plt.figure(figsize=(10, 7))

    shap.summary_plot(
        shap_values,
        X_test,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        MODEL_VISUALS_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------------------------------------------------------
    # SHAP Feature Importance Plot
    # -------------------------------------------------------------------------

    plt.figure(figsize=(10, 7))

    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        MODEL_VISUALS_DIR / "shap_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "SHAP visualizations saved successfully."
    )


# =============================================================================
# Model Comparison Visualization
# =============================================================================


def plot_model_comparison(
    artifacts: dict[str, Any],
) -> None:
    """
    Generate model performance comparison chart.

    Parameters
    ----------
    artifacts : dict[str, Any]
        Loaded visualization artifacts.
    """

    comparison_df = artifacts.get(
        "model_comparison"
    )

    if comparison_df is None or comparison_df.empty:

        logger.info(
            "Model comparison data unavailable. Skipping comparison plot."
        )
        return

    logger.info(
        "Generating model comparison visualization..."
    )

    metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
    ]

    fig, ax = plt.subplots(
        figsize=(11, 6),
    )

    x = np.arange(len(comparison_df))

    width = 0.15

    for index, metric in enumerate(metrics):

        ax.bar(
            x + (index - 2) * width,
            comparison_df[metric],
            width=width,
            label=metric,
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        comparison_df["Model"],
        rotation=15,
        ha="right",
    )

    ax.set_ylabel("Score")

    ax.set_ylim(0, 1.05)

    ax.set_title(
        "Model Performance Comparison",
        fontsize=14,
    )

    ax.grid(
        axis="y",
        alpha=0.30,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        MODEL_VISUALS_DIR / "model_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        "Model comparison visualization saved successfully."
    )


# =============================================================================
# Save All Visualizations
# =============================================================================


def save_visualizations(
    artifacts: dict[str, Any],
) -> None:
    """
    Generate and save all model visualizations.

    Parameters
    ----------
    artifacts : dict[str, Any]
        Loaded visualization artifacts.
    """

    logger.info("Generating visualization suite...")

    plot_confusion_matrix(artifacts)

    plot_roc_curve(artifacts)

    plot_precision_recall_curve(artifacts)

    plot_feature_importance(artifacts)

    plot_shap_summary(artifacts)

    plot_model_comparison(artifacts)

    logger.info(
        "All model visualizations generated successfully."
    )

# =============================================================================
# Model Visualization Pipeline
# =============================================================================


def run_model_visualization() -> dict[str, Any]:
    """
    Execute the complete model visualization workflow.

    Workflow
    --------
    1. Load visualization artifacts.
    2. Validate loaded artifacts.
    3. Generate publication-quality visualizations.
    4. Save all figures.

    Returns
    -------
    dict[str, Any]
        Loaded artifacts after successful visualization generation.
    """

    logger.info("=" * 70)
    logger.info("Starting Model Visualization Pipeline")
    logger.info("=" * 70)

    artifacts = load_artifacts()

    validate_inputs(artifacts)

    save_visualizations(artifacts)

    logger.info("=" * 70)
    logger.info("Model Visualization Pipeline Completed Successfully")
    logger.info("=" * 70)

    generated_files = [
        "confusion_matrix.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "feature_importance.png",
    ]

    if artifacts.get("shap_values") is not None:
        generated_files.extend(
            [
                "shap_summary.png",
                "shap_feature_importance.png",
            ]
        )

    if artifacts.get("model_comparison") is not None:
        generated_files.append("model_comparison.png")

    logger.info("Generated Visualization Files:")

    for file_name in generated_files:
        logger.info("  • %s", file_name)

    logger.info(
        "Visualization directory: %s",
        MODEL_VISUALS_DIR,
    )

    return artifacts


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """
    Execute the model visualization module.

    This function generates all publication-quality
    figures for the trained machine learning model
    using the saved evaluation and explainability
    artifacts.
    """

    try:

        run_model_visualization()

        logger.info("")
        logger.info("-" * 70)
        logger.info("Model Visualization Summary")
        logger.info("-" * 70)

        logger.info(
            "All visualizations have been successfully generated."
        )

        logger.info(
            "Output directory: %s",
            MODEL_VISUALS_DIR,
        )

        logger.info(
            "Image format : PNG"
        )

        logger.info(
            "Resolution   : 300 DPI"
        )

        logger.info(
            "Layout        : Tight"
        )

        logger.info("-" * 70)

    except Exception as exc:

        logger.exception(
            "Model visualization failed: %s",
            exc,
        )
        raise


if __name__ == "__main__":
    main()