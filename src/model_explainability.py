"""
model_explainability.py
=======================

Production-ready model explainability module for the
Predictive Modeling and Risk Scoring for Bank Customer Churn project.

Responsibilities
----------------
1. Load the trained model.
2. Load engineered datasets.
3. Load selected feature names.
4. Validate loaded artifacts.
5. Compute global feature importance.
6. Generate SHAP explanations.
7. Generate business-friendly feature interpretations.
8. Save explainability visualizations.

This module intentionally does NOT perform:
- preprocessing
- feature engineering
- model training
- model evaluation
- prediction

Author
------
Internship Project

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
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from config.config import REPORT_DIR

from config.config import (
    MODEL_DIR,
    PROCESSED_DATA_DIR,
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

FEATURE_NAMES_PATH = (
    MODEL_DIR / "selected_features.pkl"
)

X_TEST_PATH = (
    PROCESSED_DATA_DIR / "X_test_engineered.csv"
)

MODEL_VISUALS_DIR = VISUALS_DIR / "model"

FEATURE_IMPORTANCE_PATH = (
    MODEL_VISUALS_DIR / "feature_importance.png"
)

SHAP_SUMMARY_PATH = (
    MODEL_VISUALS_DIR / "shap_summary.png"
)

SHAP_BAR_PATH = (
    MODEL_VISUALS_DIR / "shap_bar.png"
)

SUPPORTED_MODELS = (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)

# =============================================================================
# Artifact Loading
# =============================================================================


def load_artifacts() -> tuple[
    BaseEstimator,
    pd.DataFrame,
    list[str],
]:
    """
    Load all artifacts required for model explainability.

    Returns
    -------
    tuple
        (
            trained_model,
            X_test,
            feature_names
        )

    Raises
    ------
    FileNotFoundError
        If any required artifact is missing.
    """

    logger.info("Loading explainability artifacts...")

    required_files = [
        BEST_MODEL_PATH,
        FEATURE_NAMES_PATH,
        X_TEST_PATH,
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required artifact not found:\n{file_path}"
            )

    model = joblib.load(BEST_MODEL_PATH)

    X_test = pd.read_csv(X_TEST_PATH)

    feature_names = joblib.load(
        FEATURE_NAMES_PATH
    )

    logger.info(
        "Artifacts loaded successfully."
    )

    logger.info(
        "Test Samples : %d",
        len(X_test),
    )

    logger.info(
        "Features : %d",
        len(feature_names),
    )

    return (
        model,
        X_test,
        feature_names,
    )


# =============================================================================
# Validation
# =============================================================================


def validate_inputs(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    feature_names: list[str],
) -> None:
    """
    Validate explainability artifacts.

    Parameters
    ----------
    model : BaseEstimator
        Trained model.

    X_test : pd.DataFrame
        Engineered testing dataset.

    feature_names : list[str]
        Selected feature names.

    Raises
    ------
    ValueError
        If validation fails.
    """

    logger.info(
        "Validating explainability artifacts..."
    )

    if not isinstance(
        model,
        SUPPORTED_MODELS,
    ):
        raise TypeError(
            "Unsupported estimator type."
        )

    if X_test.empty:
        raise ValueError(
            "Testing dataset is empty."
        )

    if len(feature_names) == 0:
        raise ValueError(
            "Feature name list is empty."
        )

    if X_test.shape[1] != len(feature_names):
        raise ValueError(
            "Feature count mismatch between "
            "dataset and selected features."
        )

    if X_test.isna().any().any():
        raise ValueError(
            "Testing dataset contains missing values."
        )

    logger.info(
        "Validation completed successfully."
    )

# =============================================================================
# Feature Importance
# =============================================================================


def compute_feature_importance(
    model: BaseEstimator,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Compute global feature importance for the trained model.

    Tree-based models use ``feature_importances_`` while linear models use the
    absolute value of model coefficients.

    Parameters
    ----------
    model : BaseEstimator
        Trained machine learning model.

    feature_names : list[str]
        Engineered feature names.

    Returns
    -------
    pd.DataFrame
        Feature importance dataframe sorted in descending order.
    """

    logger.info("Computing global feature importance...")

    if hasattr(model, "feature_importances_"):

        importance = np.asarray(
            model.feature_importances_,
            dtype=float,
        )

    elif hasattr(model, "coef_"):

        coefficients = np.asarray(model.coef_)

        if coefficients.ndim > 1:
            coefficients = coefficients[0]

        importance = np.abs(coefficients)

    else:
        raise ValueError(
            "The supplied model does not expose "
            "feature importance information."
        )

    feature_importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance,
        }
    )

    feature_importance_df.sort_values(
        by="Importance",
        ascending=False,
        inplace=True,
    )

    feature_importance_df.reset_index(
        drop=True,
        inplace=True,
    )

    logger.info(
        "Computed importance for %d features.",
        len(feature_importance_df),
    )

    return feature_importance_df
    
    



# =============================================================================
# SHAP Analysis
# =============================================================================


def generate_shap_analysis(
    model: BaseEstimator,
    X_test: pd.DataFrame,
) -> tuple[Any, Any]:
    """
    Generate SHAP explainer and SHAP values.

    Automatically selects TreeExplainer for tree-based models and falls back to
    the generic SHAP Explainer for linear models.

    Parameters
    ----------
    model : BaseEstimator
        Trained machine learning model.

    X_test : pd.DataFrame
        Engineered testing feature matrix.

    Returns
    -------
    tuple
        (
            explainer,
            shap_values
        )
    """

    logger.info("Generating SHAP explanations...")

    tree_models = (
        DecisionTreeClassifier,
        RandomForestClassifier,
        GradientBoostingClassifier,
    )

    try:

        if isinstance(model, tree_models):

            logger.info(
                "Using shap.TreeExplainer."
            )

            explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(X_test)

        else:

            logger.info(
                "Using shap.Explainer."
            )

            explainer = shap.Explainer(
                model,
                X_test,
            )

            shap_values = explainer(X_test)

    except Exception as exc:

        logger.exception(
            "Unable to generate SHAP values."
        )

        raise RuntimeError(
            "SHAP analysis failed."
        ) from exc

    logger.info(
        "SHAP analysis completed successfully."
    )

    return explainer, shap_values
# =============================================================================
# Visualization Saving
# =============================================================================


def save_visualizations(
    feature_importance_df: pd.DataFrame,
    shap_values: Any,
    X_test: pd.DataFrame,
) -> None:
    """
    Save feature importance, SHAP visualizations, and feature importance CSV.

    Parameters
    ----------
    feature_importance_df : pd.DataFrame
        Sorted feature importance dataframe.

    shap_values : Any
        SHAP values generated by the explainer.

    X_test : pd.DataFrame
        Engineered testing dataset.
    """

    logger.info("Saving explainability visualizations...")

    # ---------------------------------------------------------------------
    # Create required directories
    # ---------------------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_VISUALS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------------
    # Save Feature Importance CSV
    # ---------------------------------------------------------------------

    feature_importance_df.to_csv(
        REPORT_DIR / "feature_importance.csv",
        index=False,
    )

    logger.info(
        "Feature importance CSV saved."
    )

    # ---------------------------------------------------------------------
    # Feature Importance Plot
    # ---------------------------------------------------------------------

    top_features = feature_importance_df.head(20)

    fig, ax = plt.subplots(
        figsize=(10, 7),
    )

    ax.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1],
    )

    ax.set_title("Top 20 Feature Importances")

    ax.set_xlabel("Importance")

    fig.tight_layout()

    fig.savefig(
        FEATURE_IMPORTANCE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        "Feature importance plot saved."
    )

    # ---------------------------------------------------------------------
    # SHAP Summary Plot
    # ---------------------------------------------------------------------

    plt.figure(figsize=(10, 7))

    shap.summary_plot(
        shap_values,
        X_test,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        SHAP_SUMMARY_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "SHAP summary plot saved."
    )

    # ---------------------------------------------------------------------
    # SHAP Bar Plot
    # ---------------------------------------------------------------------

    plt.figure(figsize=(10, 7))

    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        SHAP_BAR_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "SHAP bar plot saved."
    )
# =============================================================================
# Business Insights
# =============================================================================


def generate_business_insights(
    feature_importance_df: pd.DataFrame,
) -> list[str]:
    """
    Generate business-friendly interpretations for the most
    influential model features.

    Parameters
    ----------
    feature_importance_df : pd.DataFrame
        Ranked feature importance dataframe.

    Returns
    -------
    list[str]
        Business interpretation statements.
    """

    logger.info(
        "Generating business insights..."
    )

    insights: list[str] = []

    top_features = feature_importance_df.head(10)

    total_importance = (
        top_features["Importance"].sum()
        if top_features["Importance"].sum() > 0
        else 1.0
    )

    for _, row in top_features.iterrows():

        contribution = (
            row["Importance"] / total_importance
        ) * 100

        insight = (
            f"{row['Feature']} is one of the strongest "
            f"predictors of customer churn, contributing "
            f"approximately {contribution:.2f}% of the "
            f"importance among the top 10 features."
        )

        insights.append(insight)

    logger.info(
        "Generated %d business insights.",
        len(insights),
    )

    return insights
# =============================================================================
# Model Explainability Pipeline
# =============================================================================


def run_model_explainability() -> tuple[
    pd.DataFrame,
    Any,
]:
    """
    Execute the complete model explainability workflow.

    Workflow
    --------
    1. Load trained model and required artifacts.
    2. Validate inputs.
    3. Compute global feature importance.
    4. Generate SHAP explanations.
    5. Save explainability visualizations.
    6. Generate business insights.

    Returns
    -------
    tuple
        (
            feature_importance_df,
            shap_values,
        )
    """

    logger.info("=" * 70)
    logger.info("Starting Model Explainability Pipeline")
    logger.info("=" * 70)

    (
        model,
        X_test,
        feature_names,
    ) = load_artifacts()

    validate_inputs(
        model=model,
        X_test=X_test,
        feature_names=feature_names,
    )

    feature_importance_df = compute_feature_importance(
        model=model,
        feature_names=feature_names,
    )

    (
        _,
        shap_values,
    ) = generate_shap_analysis(
        model=model,
        X_test=X_test,
    )

    save_visualizations(
        feature_importance_df=feature_importance_df,
        shap_values=shap_values,
        X_test=X_test,
    )

    insights = generate_business_insights(
        feature_importance_df=feature_importance_df,
    )

    logger.info("=" * 70)
    logger.info("Model Explainability Completed Successfully")
    logger.info("=" * 70)

    logger.info("Top 10 Important Features")

    for index, row in (
        feature_importance_df.head(10).iterrows()
    ):
        logger.info(
            "%2d. %-35s %.6f",
            index + 1,
            row["Feature"],
            row["Importance"],
        )

    logger.info("-" * 70)

    for insight in insights:
        logger.info("%s", insight)

    return (
        feature_importance_df,
        shap_values,
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """
    Execute the model explainability module.

    This function serves as the standalone entry point
    for generating feature importance analysis, SHAP
    explanations, and business-friendly interpretations.
    """

    try:

        (
            feature_importance_df,
            _,
        ) = run_model_explainability()

        logger.info("")
        logger.info("Top Feature Importance Summary")
        logger.info("-" * 70)

        logger.info(
            "\n%s",
            feature_importance_df.head(10).to_string(
                index=False
            ),
        )

        logger.info("-" * 70)
        logger.info(
            "Visualizations saved to: %s",
            MODEL_VISUALS_DIR,
        )

    except Exception as exc:

        logger.exception(
            "Model explainability failed: %s",
            exc,
        )
        raise


if __name__ == "__main__":
    main()