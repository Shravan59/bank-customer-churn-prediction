"""
Reusable visualization utilities for the Bank Customer Churn Prediction project.

This module centralizes all plotting logic used throughout the project,
providing consistent styling, figure export, and reusable visualization
functions for exploratory data analysis, machine learning evaluation,
model explainability, and the Streamlit dashboard.

The module does not contain any preprocessing, training, prediction,
or Streamlit page logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from config.config import (
    DEFAULT_CMAP,
    DPI,
    FIGURE_SIZE,
    STYLE,
)
from utils.helpers import ensure_directory

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Visualization Constants
# ---------------------------------------------------------------------

DEFAULT_EDGE_COLOR = "black"
DEFAULT_ALPHA = 0.8
DEFAULT_ROTATION = 45

# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------


def set_visual_style() -> None:
    """
    Configure the global plotting style.

    Applies a consistent visual appearance across all matplotlib
    and seaborn visualizations.

    Raises
    ------
    RuntimeError
        If style configuration fails.
    """
    try:
        plt.style.use(STYLE)
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = FIGURE_SIZE
        plt.rcParams["figure.dpi"] = DPI
        plt.rcParams["axes.titlesize"] = 14
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        plt.rcParams["legend.fontsize"] = 10
        logger.info("Visualization style configured successfully.")

    except Exception as exc:
        logger.exception("Unable to configure plotting style.")
        raise RuntimeError(
            "Failed to configure visualization style."
        ) from exc


# ---------------------------------------------------------------------
# Saving Utilities
# ---------------------------------------------------------------------


def save_figure(
    figure: plt.Figure,
    filename: str,
    directory: str | Path,
) -> Path:
    """
    Save a matplotlib figure.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure object.

    filename : str
        Output filename.

    directory : str or Path
        Destination directory.

    Returns
    -------
    pathlib.Path
        Saved figure path.

    Raises
    ------
    RuntimeError
        If saving fails.
    """
    try:
        output_dir = ensure_directory(directory)
        output_path = output_dir / filename

        figure.tight_layout()
        figure.savefig(
            output_path,
            dpi=DPI,
            bbox_inches="tight",
        )
        plt.close(figure)

        logger.info("Saved figure: %s", output_path)

        return output_path

    except Exception as exc:
        logger.exception("Figure saving failed.")
        raise RuntimeError(
            f"Unable to save figure '{filename}'."
        ) from exc


# ---------------------------------------------------------------------
# Basic Charts
# ---------------------------------------------------------------------


def create_bar_chart(
    dataframe: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> plt.Figure:
    """
    Create a reusable vertical bar chart.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        ax.bar(
            dataframe[x],
            dataframe[y],
            edgecolor=DEFAULT_EDGE_COLOR,
            alpha=DEFAULT_ALPHA,
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=DEFAULT_ROTATION)

        return fig

    except Exception as exc:
        logger.exception("Bar chart generation failed.")
        raise RuntimeError(
            "Unable to create bar chart."
        ) from exc


def create_horizontal_bar_chart(
    dataframe: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> plt.Figure:
    """
    Create a horizontal bar chart.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        ax.barh(
            dataframe[y],
            dataframe[x],
            edgecolor=DEFAULT_EDGE_COLOR,
            alpha=DEFAULT_ALPHA,
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        return fig

    except Exception as exc:
        logger.exception("Horizontal bar chart failed.")
        raise RuntimeError(
            "Unable to create horizontal bar chart."
        ) from exc


def create_histogram(
    dataframe: pd.DataFrame,
    column: str,
    bins: int = 30,
    title: str | None = None,
) -> plt.Figure:
    """
    Create a histogram.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        ax.hist(
            dataframe[column],
            bins=bins,
            edgecolor=DEFAULT_EDGE_COLOR,
            alpha=DEFAULT_ALPHA,
        )

        ax.set_title(title or f"{column} Distribution")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        return fig

    except Exception as exc:
        logger.exception("Histogram creation failed.")
        raise RuntimeError(
            f"Unable to create histogram for '{column}'."
        ) from exc


def create_boxplot(
    dataframe: pd.DataFrame,
    column: str,
    title: str | None = None,
) -> plt.Figure:
    """
    Create a boxplot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.boxplot(
            dataframe[column].dropna(),
            vert=True,
            patch_artist=True,
        )

        ax.set_title(title or f"{column} Boxplot")
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.3)

        return fig

    except Exception as exc:
        logger.exception("Boxplot creation failed.")
        raise RuntimeError(
            f"Unable to create boxplot for '{column}'."
        ) from exc


def create_countplot(
    dataframe: pd.DataFrame,
    column: str,
    title: str | None = None,
) -> plt.Figure:
    """
    Create a categorical count plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        counts = dataframe[column].value_counts()

        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        ax.bar(
            counts.index.astype(str),
            counts.values,
            edgecolor=DEFAULT_EDGE_COLOR,
            alpha=DEFAULT_ALPHA,
        )

        ax.set_title(title or f"{column} Distribution")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=DEFAULT_ROTATION)

        return fig

    except Exception as exc:
        logger.exception("Count plot creation failed.")
        raise RuntimeError(
            f"Unable to create count plot for '{column}'."
        ) from exc

# ---------------------------------------------------------------------
# Statistical & EDA Charts
# ---------------------------------------------------------------------


def create_correlation_heatmap(
    dataframe: pd.DataFrame,
    method: str = "pearson",
    cmap: str = DEFAULT_CMAP,
) -> plt.Figure:
    """
    Create a correlation heatmap.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe.

    method : str, default="pearson"
        Correlation method.

    cmap : str
        Matplotlib colormap.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    RuntimeError
        If the plot cannot be generated.
    """
    try:
        numeric_df = dataframe.select_dtypes(include=np.number)

        correlation = numeric_df.corr(method=method)

        fig, ax = plt.subplots(figsize=(10, 8))

        image = ax.imshow(
            correlation,
            cmap=cmap,
            interpolation="nearest",
            aspect="auto",
        )

        ax.set_xticks(range(len(correlation.columns)))
        ax.set_xticklabels(
            correlation.columns,
            rotation=90,
        )

        ax.set_yticks(range(len(correlation.columns)))
        ax.set_yticklabels(correlation.columns)

        ax.set_title("Correlation Heatmap")

        plt.colorbar(image)

        return fig

    except Exception as exc:
        logger.exception("Correlation heatmap creation failed.")
        raise RuntimeError(
            "Unable to create correlation heatmap."
        ) from exc


def create_pie_chart(
    dataframe: pd.DataFrame,
    column: str,
    title: str | None = None,
) -> plt.Figure:
    """
    Create a categorical pie chart.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe.

    column : str
        Column name.

    title : str, optional
        Chart title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        counts = dataframe[column].value_counts()

        fig, ax = plt.subplots(figsize=(7, 7))

        ax.pie(
            counts.values,
            labels=counts.index.astype(str),
            autopct="%1.1f%%",
            startangle=90,
        )

        ax.set_title(title or f"{column} Distribution")

        return fig

    except Exception as exc:
        logger.exception("Pie chart generation failed.")
        raise RuntimeError(
            f"Unable to create pie chart for '{column}'."
        ) from exc


def create_scatter_plot(
    dataframe: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> plt.Figure:
    """
    Create a scatter plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        ax.scatter(
            dataframe[x],
            dataframe[y],
            alpha=0.6,
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        return fig

    except Exception as exc:
        logger.exception("Scatter plot failed.")
        raise RuntimeError(
            "Unable to create scatter plot."
        ) from exc


def create_line_chart(
    dataframe: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> plt.Figure:
    """
    Create a line chart.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        ax.plot(
            dataframe[x],
            dataframe[y],
            linewidth=2,
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        return fig

    except Exception as exc:
        logger.exception("Line chart generation failed.")
        raise RuntimeError(
            "Unable to create line chart."
        ) from exc


# ---------------------------------------------------------------------
# EDA Visualizations
# ---------------------------------------------------------------------


def create_churn_distribution_plot(
    dataframe: pd.DataFrame,
    target_column: str,
) -> plt.Figure:
    """
    Create target class distribution plot.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset.

    target_column : str
        Target variable.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        counts = dataframe[target_column].value_counts().sort_index()

        labels = [
            "Retained",
            "Exited",
        ] if len(counts) == 2 else counts.index.astype(str)

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.bar(
            labels,
            counts.values,
            edgecolor=DEFAULT_EDGE_COLOR,
            alpha=DEFAULT_ALPHA,
        )

        ax.set_title("Customer Churn Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Number of Customers")
        ax.grid(True, alpha=0.3)

        return fig

    except Exception as exc:
        logger.exception("Target distribution plot failed.")
        raise RuntimeError(
            "Unable to create churn distribution plot."
        ) from exc


def create_probability_distribution_plot(
    probabilities: np.ndarray | pd.Series,
    bins: int = 20,
) -> plt.Figure:
    """
    Plot predicted probability distribution.

    Parameters
    ----------
    probabilities : array-like
        Prediction probabilities.

    bins : int
        Number of histogram bins.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        values = np.asarray(probabilities)

        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        ax.hist(
            values,
            bins=bins,
            edgecolor=DEFAULT_EDGE_COLOR,
            alpha=DEFAULT_ALPHA,
        )

        ax.set_title("Prediction Probability Distribution")
        ax.set_xlabel("Probability")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        return fig

    except Exception as exc:
        logger.exception(
            "Probability distribution plot failed."
        )
        raise RuntimeError(
            "Unable to create probability distribution plot."
        ) from exc


def save_plotly_chart(
    figure: Any,
    filename: str,
    directory: str | Path,
) -> Path:
    """
    Save a Plotly visualization as HTML.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure
        Plotly figure.

    filename : str
        Output filename.

    directory : str or Path
        Output directory.

    Returns
    -------
    pathlib.Path

    Raises
    ------
    RuntimeError
        If saving fails.
    """
    try:
        output_dir = ensure_directory(directory)
        output_path = output_dir / filename

        figure.write_html(
            str(output_path),
            include_plotlyjs="cdn",
        )

        logger.info(
            "Saved Plotly chart to %s",
            output_path,
        )

        return output_path

    except Exception as exc:
        logger.exception("Plotly export failed.")
        raise RuntimeError(
            f"Unable to save Plotly chart '{filename}'."
        ) from exc


def create_plotly_bar_chart(
    dataframe: pd.DataFrame,
    x: str,
    y: str,
    title: str,
):
    """
    Create an interactive Plotly bar chart.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    try:
        return px.bar(
            dataframe,
            x=x,
            y=y,
            title=title,
        )

    except Exception as exc:
        logger.exception("Plotly bar chart failed.")
        raise RuntimeError(
            "Unable to create Plotly bar chart."
        ) from exc

# ---------------------------------------------------------------------
# Machine Learning Visualizations
# ---------------------------------------------------------------------

from sklearn.metrics import auc


def create_confusion_matrix_plot(
    confusion_matrix: np.ndarray,
    class_labels: list[str] | tuple[str, str] = ("Retained", "Exited"),
) -> plt.Figure:
    """
    Create a confusion matrix visualization.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Confusion matrix.

    class_labels : list[str] | tuple[str, str], default=("Retained", "Exited")
        Class labels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        fig, ax = plt.subplots(figsize=(6, 5))

        image = ax.imshow(
            confusion_matrix,
            interpolation="nearest",
            cmap=DEFAULT_CMAP,
        )

        plt.colorbar(image)

        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_yticks(np.arange(len(class_labels)))

        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

        threshold = confusion_matrix.max() / 2

        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    str(confusion_matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white"
                    if confusion_matrix[i, j] > threshold
                    else "black",
                )

        return fig

    except Exception as exc:
        logger.exception("Unable to create confusion matrix plot.")
        raise RuntimeError(
            "Confusion matrix visualization failed."
        ) from exc


def create_roc_curve_plot(
    fpr: np.ndarray,
    tpr: np.ndarray,
) -> plt.Figure:
    """
    Create ROC curve.

    Parameters
    ----------
    fpr : np.ndarray
        False positive rate.

    tpr : np.ndarray
        True positive rate.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"AUC = {roc_auc:.3f}",
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1.5,
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        return fig

    except Exception as exc:
        logger.exception("ROC curve creation failed.")
        raise RuntimeError(
            "Unable to create ROC curve."
        ) from exc


def create_feature_importance_plot(
    feature_importance: pd.DataFrame,
    top_n: int = 20,
) -> plt.Figure:
    """
    Plot feature importance.

    Parameters
    ----------
    feature_importance : pd.DataFrame
        DataFrame with Feature and Importance columns.

    top_n : int
        Number of features.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        df = (
            feature_importance
            .sort_values("Importance", ascending=False)
            .head(top_n)
        )

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.barh(
            df["Feature"],
            df["Importance"],
            alpha=DEFAULT_ALPHA,
        )

        ax.invert_yaxis()

        ax.set_title(f"Top {top_n} Feature Importance")
        ax.set_xlabel("Importance")
        ax.grid(True, alpha=0.3)

        return fig

    except Exception as exc:
        logger.exception("Feature importance plotting failed.")
        raise RuntimeError(
            "Unable to plot feature importance."
        ) from exc


def create_shap_summary_plot(
    shap_module,
    shap_values: Any,
    features: pd.DataFrame,
    max_display: int = 20,
) -> plt.Figure:
    """
    Generate SHAP summary plot.

    Parameters
    ----------
    shap_module :
        Imported shap module.

    shap_values :
        SHAP values.

    features : pd.DataFrame
        Feature matrix.

    max_display : int
        Number of displayed features.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        plt.figure(figsize=(10, 7))

        shap_module.summary_plot(
            shap_values,
            features,
            show=False,
            max_display=max_display,
        )

        return plt.gcf()

    except Exception as exc:
        logger.exception("SHAP summary plot failed.")
        raise RuntimeError(
            "Unable to generate SHAP summary plot."
        ) from exc


def create_model_comparison_chart(
    comparison_df: pd.DataFrame,
    metric: str = "ROC-AUC",
) -> plt.Figure:
    """
    Create model comparison chart.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Model comparison dataframe.

    metric : str
        Metric column.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        ax.bar(
            comparison_df["Model"],
            comparison_df[metric],
            alpha=DEFAULT_ALPHA,
        )

        ax.set_title(f"Model Comparison ({metric})")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=30)

        return fig

    except Exception as exc:
        logger.exception("Model comparison chart failed.")
        raise RuntimeError(
            "Unable to create model comparison chart."
        ) from exc


# ---------------------------------------------------------------------
# Dashboard Helpers
# ---------------------------------------------------------------------


def create_dashboard_metric_card_data(
    title: str,
    value: Any,
    delta: Any | None = None,
    help_text: str | None = None,
) -> dict[str, Any]:
    """
    Prepare dashboard metric card data.

    Parameters
    ----------
    title : str
        Card title.

    value : Any
        Primary metric value.

    delta : Any, optional
        Delta value.

    help_text : str, optional
        Tooltip/help text.

    Returns
    -------
    dict[str, Any]
        Dictionary compatible with Streamlit metric cards.
    """
    return {
        "label": title,
        "value": value,
        "delta": delta,
        "help": help_text,
    }