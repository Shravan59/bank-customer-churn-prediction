"""
Executive Dashboard for the Bank Customer Churn Prediction project.

This Streamlit page provides a high-level business overview using the
processed dataset and previously generated project artifacts.

The dashboard is intentionally presentation-only and does not perform
preprocessing, model training, evaluation, or explainability.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config.config import (
    CLEANED_DATA_PATH,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    PROJECT_DESCRIPTION,
    PROJECT_NAME,
    PROJECT_VERSION,
    PROJECT_AUTHOR,
    TARGET_COLUMN,
)
from utils.helpers import (
    display_project_info,
    memory_usage,
    read_csv,
    validate_dataframe,
)
from utils.visualization import (
    create_churn_distribution_plot,
    create_countplot,
    create_correlation_heatmap,
)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Executive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Cached Data Loader
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)

def load_dashboard_data() -> pd.DataFrame:
    """
    Load the processed dataset used by the dashboard.

    Returns
    -------
    pandas.DataFrame
        Validated dataset.

    Raises
    ------
    RuntimeError
        If dataset loading fails.
    """
    try:
        logger.info("Loading processed dashboard dataset.")

        dataframe = read_csv(CLEANED_DATA_PATH)

        validate_dataframe(
            dataframe=dataframe,
            required_columns=[TARGET_COLUMN],
        )

        logger.info("Dashboard dataset loaded successfully.")

        return dataframe

    except Exception as exc:
        logger.exception("Dashboard dataset loading failed.")
        raise RuntimeError(
            "Unable to load dashboard dataset."
        ) from exc


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------


def render_sidebar(dataframe: pd.DataFrame) -> None:
    """
    Render dashboard sidebar.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Loaded processed dataset.
    """
    with st.sidebar:

        st.title("📊 Executive Dashboard")

        st.divider()

        st.subheader("Project Information")

        metadata = display_project_info()

        st.markdown(f"**Project:** {metadata['project_name']}")
        st.markdown(f"**Version:** {metadata['version']}")
        st.markdown(f"**Author:** {metadata['author']}")

        st.divider()

        st.subheader("Dataset")

        st.markdown(f"**Rows:** {len(dataframe):,}")
        st.markdown(f"**Columns:** {dataframe.shape[1]}")
        st.markdown(f"**Target:** `{TARGET_COLUMN}`")

        st.divider()

        st.subheader("Navigation")

        st.markdown(
            """
- Executive Overview
- KPI Summary
- Dataset Statistics
- Customer Distribution
- Numerical Overview
- Correlation Analysis
- Business Insights
- Pipeline Overview
"""
        )

        st.divider()

        st.caption(
            "This page presents business-level insights using the "
            "saved processed dataset and previously generated artifacts."
        )


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------


def render_header() -> None:
    """
    Render dashboard page header.
    """
    logger.info("Rendering dashboard header.")

    container = st.container(border=True)

    with container:

        left, right = st.columns([4, 1])

        with left:
            st.title("🏦 Predictive Modeling and Risk Scoring for Bank Customer Churn")

            st.markdown(PROJECT_DESCRIPTION)

        with right:
            st.metric(
                label="Version",
                value=PROJECT_VERSION,
            )

            st.caption(
                f"Updated\n\n{datetime.now():%d %b %Y}"
            )

        st.info(
            "This executive dashboard provides a concise overview of "
            "customer churn behaviour, dataset statistics, and project insights."
        )


# ---------------------------------------------------------------------
# Dataset Summary
# ---------------------------------------------------------------------


def render_dataset_summary(dataframe: pd.DataFrame) -> None:
    """
    Display dataset summary information.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataset.
    """
    logger.info("Rendering dataset overview.")

    with st.expander(
        "📁 Dataset Overview",
        expanded=True,
    ):

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### Dataset Information")

            summary = pd.DataFrame(
                {
                    "Metric": [
                        "Rows",
                        "Columns",
                        "Memory Usage",
                        "Missing Values",
                        "Duplicate Rows",
                        "Target Variable",
                    ],
                    "Value": [
                        len(dataframe),
                        dataframe.shape[1],
                        memory_usage(dataframe),
                        int(dataframe.isna().sum().sum()),
                        int(dataframe.duplicated().sum()),
                        TARGET_COLUMN,
                    ],
                }
            )

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
            )

        with col2:

            st.markdown("#### Feature Categories")

            feature_summary = pd.DataFrame(
                {
                    "Feature Type": [
                        "Numerical",
                        "Categorical",
                    ],
                    "Count": [
                        len(NUMERICAL_COLUMNS),
                        len(CATEGORICAL_COLUMNS),
                    ],
                }
            )

            st.dataframe(
                feature_summary,
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------
# Placeholder Functions (Implemented in Phase 2 & 3)
# ---------------------------------------------------------------------


def render_kpi_cards(dataframe: pd.DataFrame) -> None:
    """Implemented in Phase 2."""
    raise NotImplementedError


def render_distribution_section(dataframe: pd.DataFrame) -> None:
    """Implemented in Phase 2."""
    raise NotImplementedError


def render_numerical_section(dataframe: pd.DataFrame) -> None:
    """Implemented in Phase 2."""
    raise NotImplementedError


def render_correlation_section(dataframe: pd.DataFrame) -> None:
    """Implemented in Phase 2."""
    raise NotImplementedError


def render_business_insights(dataframe: pd.DataFrame) -> None:
    """Implemented in Phase 2."""
    raise NotImplementedError


def render_pipeline() -> None:
    """Implemented in Phase 3."""
    raise NotImplementedError


def render_footer() -> None:
    """Implemented in Phase 3."""
    raise NotImplementedError


def main() -> None:
    """Implemented in Phase 3."""
    raise NotImplementedError
# ---------------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------------


def render_kpi_cards(dataframe: pd.DataFrame) -> None:
    """
    Render executive KPI cards.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Processed customer dataset.
    """
    logger.info("Rendering KPI cards.")

    total_customers = len(dataframe)
    total_features = dataframe.shape[1] - 1

    churn_rate = (
        dataframe[TARGET_COLUMN].mean() * 100
        if TARGET_COLUMN in dataframe.columns
        else 0.0
    )

    retention_rate = 100 - churn_rate

    st.subheader("📈 Executive KPIs")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}",
    )

    col2.metric(
        "Total Features",
        total_features,
    )

    col3.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%",
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Retention Rate",
        f"{retention_rate:.2f}%",
    )

    col5.metric(
        "Numerical Features",
        len(NUMERICAL_COLUMNS),
    )

    col6.metric(
        "Categorical Features",
        len(CATEGORICAL_COLUMNS),
    )


# ---------------------------------------------------------------------
# Distribution Section
# ---------------------------------------------------------------------


def render_distribution_section(dataframe: pd.DataFrame) -> None:
    """
    Render customer distribution charts.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Processed dataset.
    """
    logger.info("Rendering distribution charts.")

    st.subheader("👥 Customer Distribution")

    tab1, tab2, tab3 = st.tabs(
        [
            "Churn",
            "Gender",
            "Geography",
        ]
    )

    with tab1:

        figure = create_churn_distribution_plot(
            dataframe,
            TARGET_COLUMN,
        )

        st.pyplot(
            figure,
            use_container_width=True,
        )

    with tab2:

        figure = create_countplot(
            dataframe,
            column="Gender",
            title="Gender Distribution",
        )

        st.pyplot(
            figure,
            use_container_width=True,
        )

    with tab3:

        figure = create_countplot(
            dataframe,
            column="Geography",
            title="Geography Distribution",
        )

        st.pyplot(
            figure,
            use_container_width=True,
        )


# ---------------------------------------------------------------------
# Numerical Section
# ---------------------------------------------------------------------


def render_numerical_section(
    dataframe: pd.DataFrame,
) -> None:
    """
    Display descriptive statistics for important numerical features.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Processed dataset.
    """
    logger.info("Rendering numerical overview.")

    st.subheader("📊 Numerical Overview")

    selected_features = [
        "Age",
        "CreditScore",
        "Balance",
        "EstimatedSalary",
    ]

    available = [
        feature
        for feature in selected_features
        if feature in dataframe.columns
    ]

    if not available:
        st.info("Required numerical features are unavailable.")
        return

    st.dataframe(
        dataframe[available].describe().T.round(2),
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Correlation Section
# ---------------------------------------------------------------------


def render_correlation_section(
    dataframe: pd.DataFrame,
) -> None:
    """
    Render Pearson correlation heatmap.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Processed dataset.
    """
    logger.info("Rendering correlation heatmap.")

    st.subheader("📌 Correlation Overview")

    figure = create_correlation_heatmap(
        dataframe,
    )

    st.pyplot(
        figure,
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Business Insights
# ---------------------------------------------------------------------


def render_business_insights(
    dataframe: pd.DataFrame,
) -> None:
    """
    Generate concise business insights from dataset statistics.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Processed dataset.
    """
    logger.info("Rendering business insights.")

    st.subheader("💡 Business Insights")

    churn_rate = dataframe[TARGET_COLUMN].mean() * 100

    active_rate = (
        dataframe["IsActiveMember"].mean() * 100
        if "IsActiveMember" in dataframe.columns
        else None
    )

    average_age = (
        dataframe["Age"].mean()
        if "Age" in dataframe.columns
        else None
    )

    geography_text = ""

    if "Geography" in dataframe.columns:
        top_geo = dataframe["Geography"].value_counts().idxmax()
        geography_text = (
            f"Most customers are from **{top_geo}**."
        )

    insights: list[str] = [
        (
            f"Approximately **{churn_rate:.2f}%** "
            "of customers have exited."
        ),
        geography_text,
    ]

    if active_rate is not None:
        insights.append(
            f"**{active_rate:.2f}%** of customers are active members."
        )

    if average_age is not None:
        insights.append(
            f"The average customer age is **{average_age:.1f} years**."
        )

    insights.append(
        "Customer churn prediction can support targeted "
        "retention campaigns and proactive customer engagement."
    )

    for insight in insights:
        if insight:
            st.success(insight)

# ---------------------------------------------------------------------
# Project Pipeline
# ---------------------------------------------------------------------


def render_pipeline() -> None:
    """
    Render the end-to-end machine learning pipeline.

    Returns
    -------
    None
    """
    logger.info("Rendering project pipeline.")

    st.subheader("⚙️ End-to-End Project Pipeline")

    st.code(
        """
Raw Data
    │
    ▼
Data Preprocessing
    │
    ▼
Exploratory Data Analysis (EDA)
    │
    ▼
Feature Engineering
    │
    ▼
Model Training
    │
    ▼
Model Evaluation
    │
    ▼
Model Explainability (SHAP)
    │
    ▼
Prediction
        """,
        language="text",
    )

    with st.expander("📌 Pipeline Description", expanded=False):

        pipeline_df = pd.DataFrame(
            {
                "Stage": [
                    "Raw Data",
                    "Preprocessing",
                    "EDA",
                    "Feature Engineering",
                    "Training",
                    "Evaluation",
                    "Explainability",
                    "Prediction",
                ],
                "Purpose": [
                    "Load customer information.",
                    "Prepare data for modelling.",
                    "Understand customer behaviour.",
                    "Generate model-ready features.",
                    "Train multiple machine learning algorithms.",
                    "Compare model performance using evaluation metrics.",
                    "Interpret model predictions using SHAP.",
                    "Predict future customer churn risk.",
                ],
            }
        )

        st.dataframe(
            pipeline_df,
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------


def render_footer() -> None:
    """
    Render the dashboard footer.

    Returns
    -------
    None
    """
    logger.info("Rendering footer.")

    st.divider()

    left, center, right = st.columns(3)

    with left:
        st.caption(f"👨‍💻 Author: {PROJECT_AUTHOR}")

    with center:
        st.caption(f"📦 Version: {PROJECT_VERSION}")

    with right:
        st.caption("🎓 Unified Mentor Internship Project")

    st.markdown(
        f"""
<div style="
text-align:center;
color:#6b7280;
font-size:13px;
padding-top:15px;
padding-bottom:10px;
">

<strong>{PROJECT_NAME}</strong>

<br><br>

Built with ❤️ using
Python • Pandas • NumPy • Scikit-learn • Plotly • Streamlit

<br><br>

© 2026 {PROJECT_AUTHOR}. All Rights Reserved.

</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------


def main() -> None:
    """
    Execute the Executive Dashboard.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the dashboard fails to render.
    """
    try:
        logger.info("Launching Executive Dashboard.")

        dataframe = load_dashboard_data()

        render_sidebar(dataframe)

        render_header()

        st.divider()

        render_kpi_cards(dataframe)

        st.divider()

        render_dataset_summary(dataframe)

        st.divider()

        render_distribution_section(dataframe)

        st.divider()

        render_numerical_section(dataframe)

        st.divider()

        render_correlation_section(dataframe)

        st.divider()

        render_business_insights(dataframe)

        st.divider()

        render_pipeline()

        st.divider()

        render_footer()

        logger.info(
            "Executive Dashboard rendered successfully."
        )

    except Exception as exc:
        logger.exception(
            "Executive Dashboard failed."
        )

        st.error(
            "An unexpected error occurred while loading the Executive Dashboard."
        )

        st.exception(exc)

        raise RuntimeError(
            "Failed to render Executive Dashboard."
        ) from exc


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()