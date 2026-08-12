"""
app.py
======

Streamlit application entry point for the
Predictive Modeling and Risk Scoring for Bank Customer Churn project.

Responsibilities
----------------
1. Configure the Streamlit application.
2. Load global styling resources.
3. Display project branding.
4. Render the sidebar.
5. Route execution to page components.

This module intentionally does NOT:
- Load machine learning models
- Perform predictions
- Execute preprocessing
- Train or evaluate models
- Perform explainability

Python Version
--------------
3.12+
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from config.config import (
    ASSETS_DIR,
    PROJECT_NAME,
    PROJECT_VERSION,
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
# Asset Paths
# =============================================================================

CSS_FILE = ASSETS_DIR / "style.css"
LOGO_FILE = ASSETS_DIR / "logo.png"

# =============================================================================
# Streamlit Page Configuration
# =============================================================================

st.set_page_config(
    page_title=PROJECT_NAME,
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Helper Functions
# =============================================================================


def load_css(css_path: Path = CSS_FILE) -> None:
    """
    Load the global CSS stylesheet.

    Parameters
    ----------
    css_path : Path, default=CSS_FILE
        Path to the CSS file.
    """

    try:

        if css_path.exists():

            with css_path.open(
                "r",
                encoding="utf-8",
            ) as css_file:

                st.markdown(
                    f"<style>{css_file.read()}</style>",
                    unsafe_allow_html=True,
                )

            logger.info(
                "CSS stylesheet loaded successfully."
            )

        else:

            logger.warning(
                "CSS file not found: %s",
                css_path,
            )

    except Exception as exc:

        logger.exception(
            "Failed to load CSS: %s",
            exc,
        )


def load_logo(logo_path: Path = LOGO_FILE) -> None:
    """
    Display the project logo if available.

    Parameters
    ----------
    logo_path : Path, default=LOGO_FILE
        Path to the logo image.
    """

    try:

        if logo_path.exists():

            st.image(
                str(logo_path),
                use_container_width=True,
            )

            logger.info(
                "Logo loaded successfully."
            )

        else:

            logger.warning(
                "Logo not found: %s",
                logo_path,
            )

    except Exception as exc:

        logger.exception(
            "Unable to load logo: %s",
            exc,
        )


def render_sidebar() -> None:
    """
    Render the application sidebar.
    """

    with st.sidebar:

        load_logo()

        st.title("Navigation")

        st.info(
            "Use the navigation menu on the left to explore "
            "different sections of the project."
        )

        st.divider()

        st.subheader("Project")

        st.markdown(f"**Name:** {PROJECT_NAME}")
        st.markdown(f"**Version:** {PROJECT_VERSION}")
        st.markdown(f"**Target:** {TARGET_COLUMN}")

        st.divider()

        st.subheader("Dataset")

        st.markdown(
            """
- Bank Customer Churn Dataset
- Binary Classification
- Customer-Level Records
            """
        )

        st.divider()

        st.subheader("Model")

        st.markdown(
            """
- Multiple ML Algorithms
- Best Model Selection
- Explainable AI (SHAP)
            """
        )

        st.divider()

        st.subheader("Author")

        st.markdown(
            """
**Internship Project**

Predictive Modeling and Risk Scoring for Bank Customer Churn
            """
        )
# =============================================================================
# Landing Page Components
# =============================================================================


def render_hero_section() -> None:
    """
    Render the application hero section.
    """

    logger.info("Rendering hero section.")

    with st.container():

        left_col, right_col = st.columns([3, 1], gap="large")

        with left_col:

            st.title(
                "🏦 Predictive Modeling and Risk Scoring for Bank Customer Churn"
            )

            st.markdown(
                """
Developed as an **end-to-end Machine Learning project** to help financial
institutions identify customers who are likely to leave the bank.

The solution combines **data preprocessing, exploratory data analysis,
feature engineering, machine learning, explainability, and interactive
visualization** into a single production-ready workflow.
                """
            )

            st.success(
                "🎯 Goal: Predict customer churn early so banks can improve customer retention and reduce revenue loss."
            )

        with right_col:

            st.metric(
                label="Problem Type",
                value="Classification",
            )

            st.metric(
                label="Target",
                value=TARGET_COLUMN,
            )

            st.metric(
                label="Project Version",
                value=PROJECT_VERSION,
            )


# =============================================================================
# Project Overview
# =============================================================================


def render_project_overview() -> None:
    """
    Render the project overview section.
    """

    logger.info("Rendering project overview.")

    st.header("📖 Project Overview")

    st.markdown(
        """
Customer churn is one of the most significant business challenges in the
banking industry. Acquiring a new customer is considerably more expensive
than retaining an existing one.

This project leverages supervised machine learning techniques to estimate
the probability that a customer will leave the bank based on demographic,
financial, and behavioral attributes.

The resulting predictions enable business teams to proactively identify
high-risk customers and implement personalized retention strategies.
        """
    )


# =============================================================================
# Business Problem
# =============================================================================


def render_business_problem() -> None:
    """
    Display the business problem statement.
    """

    logger.info("Rendering business problem section.")

    st.header("💼 Business Problem")

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            """
### Current Challenges

- Increasing customer churn
- Loss of recurring revenue
- High customer acquisition costs
- Limited proactive intervention
- Difficulty identifying at-risk customers
            """
        )

    with col2:

        st.success(
            """
### Proposed Solution

- Predict customer churn probability
- Identify high-risk customers
- Support retention campaigns
- Improve business decision-making
- Enable data-driven customer management
            """
        )


# =============================================================================
# Project Objectives
# =============================================================================


def render_project_objectives() -> None:
    """
    Display project objectives.
    """

    logger.info("Rendering project objectives.")

    st.header("🎯 Project Objectives")

    primary, secondary = st.columns(2)

    with primary:

        st.subheader("Primary Objectives")

        st.markdown(
            """
- Predict customer churn accurately
- Compare multiple ML algorithms
- Select the best-performing model
- Improve customer retention strategy
- Deliver explainable predictions
            """
        )

    with secondary:

        st.subheader("Secondary Objectives")

        st.markdown(
            """
- Perform comprehensive EDA
- Generate business insights
- Build reusable ML pipeline
- Develop an interactive Streamlit application
- Produce internship-ready documentation
            """
        )


# =============================================================================
# Workflow Section
# =============================================================================


def render_workflow_section() -> None:
    """
    Display the end-to-end project workflow.
    """

    logger.info("Rendering workflow section.")

    st.header("⚙️ End-to-End Workflow")

    st.code(
        """
Raw Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Exploratory Data Analysis
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
Model Explainability
      │
      ▼
Customer Churn Prediction
        """,
        language="text",
    )


# =============================================================================
# Technologies Used
# =============================================================================


def render_technologies_section() -> None:
    """
    Display technologies used in the project.
    """

    logger.info("Rendering technologies section.")

    st.header("🛠 Technologies Used")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
### 💻 Programming

- Python 3.12
- NumPy
- Pandas
            """
        )

    with col2:

        st.markdown(
            """
### 🤖 Machine Learning

- Scikit-learn
- SHAP
- Joblib
            """
        )

    with col3:

        st.markdown(
            """
### 📈 Visualization

- Streamlit
- Matplotlib
- Plotly
            """
        )


# =============================================================================
# Key Features
# =============================================================================


def render_key_features() -> None:
    """
    Display project highlights.
    """

    logger.info("Rendering key features.")

    st.header("⭐ Key Features")

    feature_1, feature_2, feature_3 = st.columns(3)

    with feature_1:

        st.success(
            """
### 📊 Data Analytics

- Data Profiling
- Exploratory Analysis
- Correlation Analysis
- Business Insights
            """
        )

    with feature_2:

        st.success(
            """
### 🤖 Machine Learning

- Multiple Algorithms
- Hyperparameter Tuning
- Model Comparison
- Performance Evaluation
            """
        )

    with feature_3:

        st.success(
            """
### 🚀 Deployment

- Interactive Dashboard
- Risk Prediction
- Explainable AI
- Production-ready Pipeline
            """
        )


# =============================================================================
# Landing Page
# =============================================================================


def render_landing_page() -> None:
    """
    Render the complete landing page.
    """

    logger.info("Rendering landing page.")

    render_hero_section()

    st.divider()

    render_project_overview()

    st.divider()

    render_business_problem()

    st.divider()

    render_project_objectives()

    st.divider()

    render_workflow_section()

    st.divider()

    render_technologies_section()

    st.divider()

    render_key_features()

# =============================================================================
# KPI Section
# =============================================================================


def render_kpi_cards() -> None:
    """
    Render high-level project KPI cards.
    """

    logger.info("Rendering KPI cards.")

    st.header("📊 Project Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Dataset Size",
            value="10,000",
            delta="Customers",
        )

    with col2:
        st.metric(
            label="Features",
            value="11",
        )

    with col3:
        st.metric(
            label="Target",
            value=TARGET_COLUMN,
        )

    with col4:
        st.metric(
            label="Models Trained",
            value="4",
        )

    with col5:
        st.metric(
            label="Best Model",
            value="Auto Selected",
        )


# =============================================================================
# Dataset Summary
# =============================================================================


def render_dataset_summary() -> None:
    """
    Render dataset summary information.
    """

    logger.info("Rendering dataset summary.")

    st.header("🗂 Dataset Summary")

    left_col, right_col = st.columns(2)

    with left_col:

        st.markdown(
            """
### Dataset Information

- **Rows:** 10,000
- **Columns:** 11
- **Target Variable:** Exited
- **Problem Type:** Binary Classification
            """
        )

    with right_col:

        st.markdown(
            """
### Feature Breakdown

- **Numerical Features:** 9
- **Categorical Features:** 2
- **Identifier Columns:** Removed
- **Missing Values:** None
            """
        )


# =============================================================================
# ML Pipeline Architecture
# =============================================================================


def render_pipeline_architecture() -> None:
    """
    Render the machine learning pipeline architecture.
    """

    logger.info("Rendering pipeline architecture.")

    st.header("🏗 Machine Learning Pipeline")

    st.code(
        """
Raw Customer Data
        │
        ▼
Data Preprocessing
        │
        ▼
Exploratory Data Analysis
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
Customer Churn Prediction
        """,
        language="text",
    )


# =============================================================================
# Repository Structure
# =============================================================================


def render_repository_structure() -> None:
    """
    Display the project repository structure.
    """

    logger.info("Rendering repository structure.")

    st.header("📁 Repository Structure")

    st.code(
        """
bank_customer_churn_prediction/
│
├── app.py
├── assets/
├── config/
├── data/
├── models/
├── notebooks/
├── pages/
├── reports/
├── src/
├── utils/
├── visuals/
├── requirements.txt
├── README.md
└── LICENSE
        """,
        language="text",
    )


# =============================================================================
# Future Scope
# =============================================================================


def render_future_scope() -> None:
    """
    Display future enhancement opportunities.
    """

    logger.info("Rendering future scope.")

    st.header("🚀 Future Scope")

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            """
### Technical Enhancements

- Automated Model Retraining
- Cloud Deployment
- REST API Integration
- CI/CD Pipeline
- Docker Support
            """
        )

    with col2:

        st.success(
            """
### Business Enhancements

- Real-Time Predictions
- Customer Segmentation
- Personalized Retention Strategies
- Campaign Optimization
- Executive BI Dashboard
            """
        )


# =============================================================================
# Footer
# =============================================================================


def render_footer() -> None:
    """
    Render the application footer.
    """

    logger.info("Rendering footer.")

    st.divider()

    st.markdown(
        f"""
---
### 📌 Project Information

**Project:** {PROJECT_NAME}

**Version:** {PROJECT_VERSION}

**Internship:** Unified Mentor

**Domain:** Banking Analytics & Machine Learning

Developed using **Python, Scikit-learn, Streamlit, SHAP, Pandas, Plotly, and Matplotlib**.
"""
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """
    Execute the Streamlit application.

    Workflow
    --------
    1. Load global stylesheet.
    2. Render sidebar.
    3. Display landing page.
    4. Display KPI cards.
    5. Display dataset summary.
    6. Display ML pipeline architecture.
    7. Display repository overview.
    8. Display future scope.
    9. Display footer.
    """

    logger.info("Starting Streamlit application.")

    load_css()

    render_sidebar()

    render_landing_page()

    st.divider()

    render_kpi_cards()

    st.divider()

    render_dataset_summary()

    st.divider()

    render_pipeline_architecture()

    st.divider()

    render_repository_structure()

    st.divider()

    render_future_scope()

    render_footer()

    logger.info("Streamlit application loaded successfully.")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    main()