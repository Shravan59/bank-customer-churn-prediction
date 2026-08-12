"""
exploratory_data_analysis.py
============================

Production-ready Exploratory Data Analysis (EDA) module for the
Predictive Modeling and Risk Scoring for Bank Customer Churn project.

Responsibilities
----------------
1. Load the cleaned dataset.
2. Generate dataset profiling information.
3. Perform exploratory data analysis.
4. Generate business-oriented insights.
5. Save all visualizations for reporting and dashboard usage.

This module intentionally performs ONLY exploratory analysis.

It does NOT perform:
- data preprocessing
- feature engineering
- model training
- model evaluation
- model explainability
- prediction

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
from typing import Any

import pandas as pd

from config.config import (
    PROCESSED_DATA_DIR,
    VISUALS_DIR,
    TARGET_COLUMN,
)

from utils.visualization import (
    save_plot,
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
# Project Constants
# =============================================================================

EDA_VISUALS_DIR: Path = VISUALS_DIR / "eda"

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

CATEGORICAL_COLUMNS: list[str] = [
    "Geography",
    "Gender",
]

CLEANED_DATA_PATH: Path = (
    PROCESSED_DATA_DIR / "cleaned_data.csv"
)

# =============================================================================
# Data Loading
# =============================================================================


def load_cleaned_data(
    file_path: Path = CLEANED_DATA_PATH,
) -> pd.DataFrame:
    """
    Load the cleaned customer churn dataset.

    Parameters
    ----------
    file_path : Path, optional
        Path to the cleaned dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset does not exist.

    ValueError
        If the dataset is empty.
    """

    logger.info("Loading cleaned dataset...")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    dataframe = pd.read_csv(file_path)

    if dataframe.empty:
        raise ValueError(
            "The cleaned dataset is empty."
        )

    logger.info(
        "Dataset loaded successfully (%d rows, %d columns).",
        dataframe.shape[0],
        dataframe.shape[1],
    )

    return dataframe


# =============================================================================
# Dataset Summary
# =============================================================================


def generate_dataset_summary(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Generate a comprehensive dataset summary.

    This function reports only descriptive information and
    does not modify the dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Cleaned dataset.

    Returns
    -------
    dict[str, Any]
        Dictionary containing dataset statistics.
    """

    logger.info("Generating dataset summary...")

    summary: dict[str, Any] = {
        "shape": dataframe.shape,
        "rows": dataframe.shape[0],
        "columns": dataframe.shape[1],
        "column_names": list(dataframe.columns),
        "data_types": dataframe.dtypes.astype(str).to_dict(),
        "missing_values": dataframe.isna().sum().to_dict(),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "unique_values": dataframe.nunique().to_dict(),
        "memory_usage_mb": round(
            dataframe.memory_usage(deep=True).sum()
            / (1024 ** 2),
            2,
        ),
        "numerical_summary": dataframe[
            NUMERICAL_COLUMNS
        ].describe().round(2),
    }

    logger.info("Dataset Summary")
    logger.info("-" * 60)
    logger.info("Rows              : %d", summary["rows"])
    logger.info("Columns           : %d", summary["columns"])
    logger.info(
        "Memory Usage (MB) : %.2f",
        summary["memory_usage_mb"],
    )
    logger.info(
        "Duplicate Rows    : %d",
        summary["duplicate_rows"],
    )

    total_missing = sum(
        summary["missing_values"].values()
    )

    if total_missing == 0:
        logger.info("Missing Values    : None")
    else:
        logger.warning(
            "Total Missing Values: %d",
            total_missing,
        )

    logger.info(
        "Target Distribution:\n%s",
        dataframe[TARGET_COLUMN]
        .value_counts()
        .to_string(),
    )

    EDA_VISUALS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_table = (
        pd.DataFrame(
            {
                "Metric": [
                    "Rows",
                    "Columns",
                    "Memory Usage (MB)",
                    "Duplicate Rows",
                    "Missing Values",
                ],
                "Value": [
                    summary["rows"],
                    summary["columns"],
                    summary["memory_usage_mb"],
                    summary["duplicate_rows"],
                    total_missing,
                ],
            }
        )
    )

    save_plot(
        dataframe=summary_table,
        output_path=EDA_VISUALS_DIR / "dataset_summary.png",
        title="Dataset Summary",
    )

    logger.info("Dataset summary generated successfully.")

    return summary
from typing import Any

import matplotlib.pyplot as plt
import plotly.express as px


# =============================================================================
# Target Variable Analysis
# =============================================================================


def analyze_target_variable(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Analyze the target variable distribution.

    Generates:
        - Count plot
        - Percentage distribution plot

    Parameters
    ----------
    dataframe : pd.DataFrame
        Cleaned dataset.

    Returns
    -------
    dict[str, Any]
        Target distribution statistics.
    """

    logger.info("Analyzing target variable...")

    output_dir = EDA_VISUALS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = dataframe[TARGET_COLUMN].value_counts().sort_index()

    percentages = (
        dataframe[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    summary = {
        "counts": counts.to_dict(),
        "percentages": percentages.to_dict(),
    }

    # ----------------------------------------------------------
    # Count Plot
    # ----------------------------------------------------------

    fig = px.bar(
        x=["Retained", "Exited"],
        y=counts.values,
        text=counts.values,
        title="Customer Churn Distribution",
        labels={
            "x": "Customer Status",
            "y": "Customers",
        },
    )

    fig.update_traces(textposition="outside")

    fig.write_image(
        output_dir / "target_distribution.png",
        width=900,
        height=600,
        scale=3,
    )

    # ----------------------------------------------------------
    # Percentage Plot
    # ----------------------------------------------------------

    fig = px.pie(
        values=percentages.values,
        names=["Retained", "Exited"],
        title="Customer Churn Percentage",
    )

    fig.write_image(
        output_dir / "target_percentage.png",
        width=900,
        height=600,
        scale=3,
    )

    logger.info("Target analysis completed.")

    return summary


# =============================================================================
# Numerical Feature Analysis
# =============================================================================


def analyze_numerical_features(
    dataframe: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Analyze numerical variables.

    Generates for every numerical feature:

    - Histogram
    - Boxplot

    Parameters
    ----------
    dataframe : pd.DataFrame

    Returns
    -------
    dict
        Dictionary containing summary statistics.
    """

    logger.info("Analyzing numerical features...")

    statistics: dict[str, pd.DataFrame] = {}

    output_dir = EDA_VISUALS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for column in NUMERICAL_COLUMNS:

        if column not in dataframe.columns:
            continue

        statistics[column] = (
            dataframe[column]
            .describe()
            .round(2)
            .to_frame(name=column)
        )

        # ------------------------------------------------------
        # Histogram
        # ------------------------------------------------------

        fig = px.histogram(
            dataframe,
            x=column,
            nbins=30,
            title=f"{column} Distribution",
        )

        fig.write_image(
            output_dir / f"{column.lower()}_distribution.png",
            width=900,
            height=600,
            scale=3,
        )

        # ------------------------------------------------------
        # Boxplot
        # ------------------------------------------------------

        fig = px.box(
            dataframe,
            y=column,
            title=f"{column} Boxplot",
        )

        fig.write_image(
            output_dir / f"{column.lower()}_boxplot.png",
            width=900,
            height=600,
            scale=3,
        )

    logger.info("Numerical analysis completed.")

    return statistics


# =============================================================================
# Categorical Feature Analysis
# =============================================================================


def analyze_categorical_features(
    dataframe: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Analyze categorical variables.

    Generates:

    - Count plot
    - Percentage distribution

    Parameters
    ----------
    dataframe : pd.DataFrame

    Returns
    -------
    dict
        Frequency tables.
    """

    logger.info("Analyzing categorical features...")

    results: dict[str, pd.DataFrame] = {}

    output_dir = EDA_VISUALS_DIR

    for column in CATEGORICAL_COLUMNS:

        if column not in dataframe.columns:
            continue

        frequency = (
            dataframe[column]
            .value_counts()
            .rename("Count")
            .to_frame()
        )

        frequency["Percentage"] = (
            dataframe[column]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
            .values
        )

        results[column] = frequency

        # ------------------------------------------------------
        # Count Plot
        # ------------------------------------------------------

        fig = px.bar(
            frequency.reset_index(),
            x="index",
            y="Count",
            text="Count",
            title=f"{column} Distribution",
            labels={
                "index": column,
            },
        )

        fig.update_traces(textposition="outside")

        fig.write_image(
            output_dir / f"{column.lower()}_distribution.png",
            width=900,
            height=600,
            scale=3,
        )

        # ------------------------------------------------------
        # Percentage Plot
        # ------------------------------------------------------

        fig = px.pie(
            names=frequency.index,
            values=frequency["Percentage"],
            title=f"{column} Percentage Distribution",
        )

        fig.write_image(
            output_dir / f"{column.lower()}_percentage.png",
            width=900,
            height=600,
            scale=3,
        )

    logger.info("Categorical analysis completed.")

    return results
import numpy as np


# =============================================================================
# Correlation Analysis
# =============================================================================


def correlation_analysis(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform Pearson correlation analysis on numerical features.

    A correlation heatmap is generated and saved to disk.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Cleaned dataset.

    Returns
    -------
    pd.DataFrame
        Pearson correlation matrix.
    """

    logger.info("Performing correlation analysis...")

    output_dir = EDA_VISUALS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    available_columns = [
        column
        for column in NUMERICAL_COLUMNS + [TARGET_COLUMN]
        if column in dataframe.columns
    ]

    correlation_matrix = (
        dataframe[available_columns]
        .corr(method="pearson")
        .round(2)
    )

    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        color_continuous_scale="RdBu",
        aspect="auto",
        title="Pearson Correlation Heatmap",
    )

    fig.write_image(
        output_dir / "correlation_heatmap.png",
        width=1000,
        height=850,
        scale=3,
    )

    logger.info("Correlation analysis completed.")

    return correlation_matrix


# =============================================================================
# Outlier Analysis
# =============================================================================


def outlier_analysis(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform IQR-based outlier analysis.

    Outliers are reported only.
    No rows are removed.

    Parameters
    ----------
    dataframe : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Outlier summary.
    """

    logger.info("Performing outlier analysis...")

    results: list[dict[str, Any]] = []

    for column in NUMERICAL_COLUMNS:

        if column not in dataframe.columns:
            continue

        q1 = dataframe[column].quantile(0.25)
        q3 = dataframe[column].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (
            (dataframe[column] < lower)
            | (dataframe[column] > upper)
        )

        outlier_count = int(mask.sum())

        results.append(
            {
                "Feature": column,
                "Q1": round(q1, 2),
                "Q3": round(q3, 2),
                "IQR": round(iqr, 2),
                "Lower Bound": round(lower, 2),
                "Upper Bound": round(upper, 2),
                "Outliers": outlier_count,
                "Outlier (%)": round(
                    outlier_count / len(dataframe) * 100,
                    2,
                ),
            }
        )

    outlier_df = pd.DataFrame(results)

    fig = px.bar(
        outlier_df,
        x="Feature",
        y="Outliers",
        text="Outliers",
        title="IQR Outlier Analysis",
    )

    fig.update_traces(textposition="outside")

    fig.write_image(
        EDA_VISUALS_DIR / "outlier_analysis.png",
        width=1000,
        height=600,
        scale=3,
    )

    logger.info("Outlier analysis completed.")

    return outlier_df


# =============================================================================
# Bivariate Analysis
# =============================================================================


def bivariate_analysis(
    dataframe: pd.DataFrame,
) -> dict[str, Path]:
    """
    Generate business-oriented bivariate visualizations.

    Generated Visualizations
    ------------------------
    - Exited vs Geography
    - Exited vs Gender
    - Exited vs Age
    - Exited vs Balance
    - Exited vs CreditScore
    - Exited vs NumOfProducts
    - Exited vs IsActiveMember
    - Exited vs HasCrCard

    Parameters
    ----------
    dataframe : pd.DataFrame

    Returns
    -------
    dict
        Mapping of chart names to saved file paths.
    """

    logger.info("Generating bivariate analysis...")

    saved_files: dict[str, Path] = {}

    # ----------------------------------------------------------
    # Target vs Geography
    # ----------------------------------------------------------

    fig = px.histogram(
        dataframe,
        x="Geography",
        color=TARGET_COLUMN,
        barmode="group",
        title="Customer Churn by Geography",
    )

    path = EDA_VISUALS_DIR / "target_vs_geography.png"

    fig.write_image(path, width=900, height=600, scale=3)

    saved_files["target_vs_geography"] = path

    # ----------------------------------------------------------
    # Target vs Gender
    # ----------------------------------------------------------

    fig = px.histogram(
        dataframe,
        x="Gender",
        color=TARGET_COLUMN,
        barmode="group",
        title="Customer Churn by Gender",
    )

    path = EDA_VISUALS_DIR / "target_vs_gender.png"

    fig.write_image(path, width=900, height=600, scale=3)

    saved_files["target_vs_gender"] = path

    # ----------------------------------------------------------
    # Numerical Features vs Target
    # ----------------------------------------------------------

    numerical_targets = [
        "Age",
        "Balance",
        "CreditScore",
        "NumOfProducts",
        "IsActiveMember",
        "HasCrCard",
    ]

    for feature in numerical_targets:

        if feature not in dataframe.columns:
            continue

        if dataframe[feature].nunique() <= 10:

            fig = px.box(
                dataframe,
                x=TARGET_COLUMN,
                y=feature,
                title=f"{feature} vs {TARGET_COLUMN}",
            )

        else:

            fig = px.violin(
                dataframe,
                x=TARGET_COLUMN,
                y=feature,
                box=True,
                title=f"{feature} vs {TARGET_COLUMN}",
            )

        file_name = (
            f"target_vs_{feature.lower()}.png"
        )

        path = EDA_VISUALS_DIR / file_name

        fig.write_image(
            path,
            width=900,
            height=600,
            scale=3,
        )

        saved_files[file_name] = path

    logger.info("Bivariate analysis completed.")

    return saved_files
# =============================================================================
# Business Insight Generation
# =============================================================================


def generate_business_insights(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Generate concise business insights from the EDA results.

    Insights are computed dynamically from the dataset rather than
    being hardcoded.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Cleaned dataset.

    Returns
    -------
    list[str]
        Generated business insights.
    """

    logger.info("Generating business insights...")

    insights: list[str] = []

    # -----------------------------------------------------------------
    # Target Distribution
    # -----------------------------------------------------------------

    target_percentage = (
        dataframe[TARGET_COLUMN]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    churn_rate = float(target_percentage.get(1, 0.0))
    retained_rate = float(target_percentage.get(0, 0.0))

    insights.append(
        f"Customer retention rate is {retained_rate:.2f}% "
        f"while churn rate is {churn_rate:.2f}%."
    )

    # -----------------------------------------------------------------
    # Geography
    # -----------------------------------------------------------------

    if "Geography" in dataframe.columns:

        geography_summary = (
            dataframe
            .groupby("Geography")[TARGET_COLUMN]
            .mean()
            .mul(100)
            .round(2)
        )

        highest_geo = geography_summary.idxmax()
        highest_geo_rate = geography_summary.max()

        insights.append(
            f"{highest_geo} has the highest observed churn rate "
            f"({highest_geo_rate:.2f}%)."
        )

    # -----------------------------------------------------------------
    # Gender
    # -----------------------------------------------------------------

    if "Gender" in dataframe.columns:

        gender_summary = (
            dataframe
            .groupby("Gender")[TARGET_COLUMN]
            .mean()
            .mul(100)
            .round(2)
        )

        highest_gender = gender_summary.idxmax()

        insights.append(
            f"{highest_gender} customers exhibit a higher "
            "average churn rate."
        )

    # -----------------------------------------------------------------
    # Active Membership
    # -----------------------------------------------------------------

    if "IsActiveMember" in dataframe.columns:

        activity_summary = (
            dataframe
            .groupby("IsActiveMember")[TARGET_COLUMN]
            .mean()
            .mul(100)
            .round(2)
        )

        if (
            len(activity_summary) == 2
            and activity_summary.loc[0] > activity_summary.loc[1]
        ):
            insights.append(
                "Inactive members churn more frequently than "
                "active members."
            )

    # -----------------------------------------------------------------
    # Products
    # -----------------------------------------------------------------

    if "NumOfProducts" in dataframe.columns:

        product_summary = (
            dataframe
            .groupby("NumOfProducts")[TARGET_COLUMN]
            .mean()
            .mul(100)
            .round(2)
        )

        highest_product_group = product_summary.idxmax()

        insights.append(
            f"Customers with {highest_product_group} product(s) "
            "show the highest average churn."
        )

    # -----------------------------------------------------------------
    # Age
    # -----------------------------------------------------------------

    if "Age" in dataframe.columns:

        average_age = (
            dataframe
            .groupby(TARGET_COLUMN)["Age"]
            .mean()
            .round(2)
        )

        if (
            len(average_age) == 2
            and average_age.loc[1] > average_age.loc[0]
        ):
            insights.append(
                "Customers who churn are older on average "
                "than retained customers."
            )

    logger.info(
        "Generated %d business insights.",
        len(insights),
    )

    return insights


# =============================================================================
# EDA Pipeline
# =============================================================================


def run_eda() -> dict[str, Any]:
    """
    Execute the complete exploratory data analysis workflow.

    Returns
    -------
    dict[str, Any]
        Dictionary containing all EDA outputs.
    """

    logger.info("=" * 70)
    logger.info("Starting Exploratory Data Analysis")
    logger.info("=" * 70)

    dataframe = load_cleaned_data()

    summary = generate_dataset_summary(dataframe)

    target_summary = analyze_target_variable(dataframe)

    numerical_summary = analyze_numerical_features(dataframe)

    categorical_summary = analyze_categorical_features(dataframe)

    correlation_matrix = correlation_analysis(dataframe)

    outlier_summary = outlier_analysis(dataframe)

    bivariate_results = bivariate_analysis(dataframe)

    business_insights = generate_business_insights(dataframe)

    logger.info("=" * 70)
    logger.info("EDA completed successfully.")
    logger.info(
        "Visualizations saved to: %s",
        EDA_VISUALS_DIR,
    )
    logger.info("=" * 70)

    return {
        "dataset_summary": summary,
        "target_analysis": target_summary,
        "numerical_analysis": numerical_summary,
        "categorical_analysis": categorical_summary,
        "correlation_matrix": correlation_matrix,
        "outlier_analysis": outlier_summary,
        "bivariate_analysis": bivariate_results,
        "business_insights": business_insights,
    }


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """
    Entry point for standalone execution.
    """

    try:

        results = run_eda()

        logger.info("Business Insights")

        for index, insight in enumerate(
            results["business_insights"],
            start=1,
        ):
            logger.info("%d. %s", index, insight)

    except Exception as exc:

        logger.exception(
            "EDA execution failed: %s",
            exc,
        )
        raise


if __name__ == "__main__":
    main()