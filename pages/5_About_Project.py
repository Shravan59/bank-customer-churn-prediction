"""
About Project Streamlit Page.

This module implements the "About Project" page for the
Predictive Modeling and Risk Scoring for Bank Customer Churn
application.

The page provides comprehensive project documentation,
including:

- Project overview
- Business problem statement
- Project objectives
- Dataset summary
- End-to-end machine learning pipeline
- Project architecture
- Technology stack
- Implemented features
- Business value
- Author information
- Internship details
- Future scope

The implementation follows production-quality software
engineering practices and is fully compatible with the
existing Streamlit multipage application architecture.

Notes
-----
This module is read-only and does not perform model training,
prediction, or artifact generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from config.config import (
    APP_NAME,
    APP_VERSION,
    DATASET_PATH,
    INTERNSHIP_NAME,
    INTERNSHIP_ORGANIZATION,
    MODEL_ARTIFACTS_DIR,
    PROJECT_AUTHOR,
    PROJECT_DESCRIPTION,
    PROJECT_NAME,
    PROJECT_VERSION,
    TARGET_COLUMN,
)

from utils.helpers import (
    format_number,
    load_dataset,
)

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Configuration Imports
# -----------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(MODEL_ARTIFACTS_DIR).parent

APPLICATION_TITLE: str = APP_NAME
APPLICATION_VERSION: str = APP_VERSION

ABOUT_PAGE_TITLE: str = "📘 About Project"

ABOUT_PAGE_DESCRIPTION: str = (
    "Comprehensive documentation of the end-to-end Machine Learning "
    "project for Predictive Modeling and Risk Scoring for "
    "Bank Customer Churn."
)
def render_page_header() -> None:
    """
    Render the page header.

    Displays the page title, project description, version information,
    and project context for the About Project page.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the page header cannot be rendered.
    """
    logger.info("Rendering About Project page header.")

    try:
        st.title("📘 About Project")

        st.caption(
            "Predictive Modeling and Risk Scoring for Bank Customer Churn"
        )

        st.markdown(
            """
This page provides comprehensive documentation of the project,
including the business problem, dataset, machine learning workflow,
system architecture, technology stack, implemented features, business
value, and future roadmap.
"""
        )

        st.info(
            f"""
**Project:** {PROJECT_NAME}

**Version:** {PROJECT_VERSION}

**Application:** {APPLICATION_TITLE} ({APPLICATION_VERSION})

**Author:** {PROJECT_AUTHOR}
"""
        )

        st.breadcrumb = getattr(st, "breadcrumb", None)

        st.markdown(
            f"""
**Project Context**

`{PROJECT_NAME}` → **About Project**

{PROJECT_DESCRIPTION}
"""
        )

        st.divider()

        logger.info("Page header rendered successfully.")

    except Exception as exc:
        logger.exception("Failed to render page header.")
        raise RuntimeError(
            "Unable to render page header."
        ) from exc


def render_project_overview() -> None:
    """
    Render the project overview section.

    Presents the project objectives, motivation, expected outcomes,
    and executive summary.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the overview cannot be rendered.
    """
    logger.info("Rendering project overview.")

    try:
        st.header("📖 Project Overview")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(
                """
### Executive Summary

Customer churn is one of the most significant business challenges faced
by financial institutions. Losing valuable customers directly impacts
revenue, profitability, and long-term business growth.

This project develops a complete end-to-end Machine Learning solution
capable of predicting whether a customer is likely to leave the bank
based on demographic information, financial characteristics, and
customer behaviour.

The solution combines data preprocessing, exploratory data analysis,
feature engineering, model training, model evaluation, explainability,
and interactive Streamlit dashboards into a production-ready analytical
application.
"""
            )

        with col2:
            overview_metrics = pd.DataFrame(
                {
                    "Metric": [
                        "Domain",
                        "Application",
                        "Pipeline",
                        "Deployment",
                    ],
                    "Value": [
                        "Banking",
                        "Customer Churn",
                        "End-to-End ML",
                        "Streamlit",
                    ],
                }
            )

            st.dataframe(
                overview_metrics,
                hide_index=True,
                use_container_width=True,
            )

        objectives = pd.DataFrame(
            {
                "Objective": [
                    "Predict customer churn accurately",
                    "Support proactive customer retention",
                    "Reduce customer attrition",
                    "Improve business decision-making",
                    "Provide explainable AI insights",
                    "Enable interactive analytics",
                ]
            }
        )

        st.subheader("Project Objectives")

        st.dataframe(
            objectives,
            hide_index=True,
            use_container_width=True,
        )

        st.success(
            "The project delivers an end-to-end production-ready machine "
            "learning solution for customer churn prediction."
        )

        logger.info("Project overview rendered successfully.")

    except Exception as exc:
        logger.exception("Failed to render project overview.")
        raise RuntimeError(
            "Unable to render project overview."
        ) from exc


def render_problem_statement() -> None:
    """
    Render the business problem statement.

    Displays the business challenge, objectives, expected benefits,
    and project motivation.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the problem statement cannot be rendered.
    """
    logger.info("Rendering business problem statement.")

    try:
        st.header("🎯 Business Problem Statement")

        st.markdown(
            """
Financial institutions continuously lose customers due to competition,
changing customer expectations, pricing strategies, service quality,
and evolving financial products.

Traditional approaches identify churn only after customers have already
left, making retention efforts expensive and less effective.

The objective of this project is to leverage Machine Learning to
identify customers at risk of churning before attrition occurs, enabling
timely and targeted intervention.
"""
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Business Challenges")

            st.markdown(
                """
- High customer acquisition costs
- Increasing customer attrition
- Reduced customer lifetime value
- Limited visibility into churn risk
- Inefficient retention campaigns
- Difficulty prioritizing high-risk customers
"""
            )

        with col2:
            st.subheader("Project Goals")

            st.markdown(
                """
- Predict customer churn accurately
- Identify high-risk customers early
- Improve customer retention strategies
- Support data-driven decision making
- Reduce operational costs
- Increase long-term profitability
"""
            )

        problem_summary = pd.DataFrame(
            {
                "Business Aspect": [
                    "Industry",
                    "Primary Challenge",
                    "Solution",
                    "Expected Outcome",
                ],
                "Description": [
                    "Banking",
                    "Customer Churn",
                    "Machine Learning Prediction",
                    "Improved Customer Retention",
                ],
            }
        )

        st.subheader("Problem Summary")

        st.dataframe(
            problem_summary,
            hide_index=True,
            use_container_width=True,
        )

        with st.expander(
            "📌 Why Customer Churn Prediction Matters",
            expanded=False,
        ):
            st.markdown(
                """
Early identification of customers likely to leave enables banks to:

- Prioritize retention resources effectively.
- Deliver personalized customer engagement.
- Improve customer satisfaction.
- Increase customer lifetime value.
- Reduce revenue loss.
- Strengthen competitive advantage.

Machine Learning transforms historical customer data into actionable
business intelligence, allowing organizations to move from reactive
decision-making to proactive customer relationship management.
"""
            )

        logger.info("Business problem statement rendered successfully.")

    except Exception as exc:
        logger.exception("Failed to render problem statement.")
        raise RuntimeError(
            "Unable to render business problem statement."
        ) from exc

def render_dataset_information() -> None:
    """
    Render dataset information.

    Displays a comprehensive summary of the dataset used for customer
    churn prediction, including dataset location, dimensions,
    feature composition, target information, and basic statistics.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If dataset information cannot be rendered.
    """
    logger.info("Rendering dataset information.")

    try:
        st.header("🗂️ Dataset Information")

        dataset_exists = Path(DATASET_PATH).exists()

        if dataset_exists:
            try:
                dataframe = load_dataset(DATASET_PATH)
            except Exception:
                dataframe = pd.read_csv(DATASET_PATH)
        else:
            dataframe = None

        if dataframe is None:
            st.warning("Dataset could not be loaded.")
            return

        rows, columns = dataframe.shape

        numeric_columns = dataframe.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_columns = dataframe.select_dtypes(
            exclude="number"
        ).columns.tolist()

        missing_values = int(dataframe.isna().sum().sum())

        duplicate_rows = int(dataframe.duplicated().sum())

        memory_usage = (
            dataframe.memory_usage(deep=True).sum()
            / (1024 ** 2)
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", format_number(rows))

        with col2:
            st.metric("Columns", format_number(columns))

        with col3:
            st.metric(
                "Numeric Features",
                len(numeric_columns),
            )

        with col4:
            st.metric(
                "Categorical Features",
                len(categorical_columns),
            )

        summary = pd.DataFrame(
            {
                "Property": [
                    "Dataset Path",
                    "Rows",
                    "Columns",
                    "Memory Usage (MB)",
                    "Missing Values",
                    "Duplicate Rows",
                    "Target Column",
                ],
                "Value": [
                    str(DATASET_PATH),
                    rows,
                    columns,
                    f"{memory_usage:.2f}",
                    missing_values,
                    duplicate_rows,
                    TARGET_COLUMN,
                ],
            }
        )

        st.subheader("Dataset Summary")

        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True,
        )

        with st.expander(
            "Available Features",
            expanded=False,
        ):
            feature_df = pd.DataFrame(
                {
                    "Feature": dataframe.columns,
                    "Data Type": dataframe.dtypes.astype(str),
                }
            )

            st.dataframe(
                feature_df,
                hide_index=True,
                use_container_width=True,
            )

        if TARGET_COLUMN in dataframe.columns:

            distribution = (
                dataframe[TARGET_COLUMN]
                .value_counts(dropna=False)
                .reset_index()
            )

            distribution.columns = [
                "Target",
                "Count",
            ]

            fig = px.pie(
                distribution,
                names="Target",
                values="Count",
                hole=0.45,
                title="Target Variable Distribution",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        logger.info("Dataset information rendered successfully.")

    except Exception as exc:
        logger.exception(
            "Failed to render dataset information."
        )
        raise RuntimeError(
            "Unable to render dataset information."
        ) from exc


def render_machine_learning_pipeline() -> None:
    """
    Render the end-to-end machine learning pipeline.

    Displays the complete workflow followed throughout the project,
    from raw data ingestion to deployment.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the pipeline section cannot be rendered.
    """
    logger.info("Rendering machine learning pipeline.")

    try:
        st.header("🤖 End-to-End Machine Learning Pipeline")

        pipeline_steps = pd.DataFrame(
            {
                "Stage": [
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                    "10",
                ],
                "Pipeline Step": [
                    "Business Understanding",
                    "Data Collection",
                    "Data Preprocessing",
                    "Exploratory Data Analysis",
                    "Feature Engineering",
                    "Model Training",
                    "Model Evaluation",
                    "Model Explainability",
                    "Model Deployment",
                    "Interactive Streamlit Dashboard",
                ],
            }
        )

        st.dataframe(
            pipeline_steps,
            hide_index=True,
            use_container_width=True,
        )

        fig = px.line(
            pipeline_steps,
            x="Stage",
            y="Pipeline Step",
            markers=True,
            title="Machine Learning Workflow",
        )

        fig.update_layout(
            xaxis_title="Pipeline Stage",
            yaxis_title="Workflow",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        tabs = st.tabs(
            [
                "Preprocessing",
                "Training",
                "Evaluation",
                "Deployment",
            ]
        )

        with tabs[0]:
            st.markdown(
                """
### Data Preprocessing

- Missing value handling
- Duplicate removal
- Data cleaning
- Feature validation
- Data transformation
- Encoding
- Scaling
"""
            )

        with tabs[1]:
            st.markdown(
                """
### Model Training

- Train/Test split
- Model selection
- Hyperparameter optimization
- Cross-validation
- Model persistence
"""
            )

        with tabs[2]:
            st.markdown(
                """
### Model Evaluation

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Explainability
"""
            )

        with tabs[3]:
            st.markdown(
                """
### Deployment

- Persisted model loading
- Interactive prediction
- Batch prediction
- Dashboard analytics
- Business recommendations
"""
            )

        with st.expander(
            "📌 Pipeline Benefits",
            expanded=False,
        ):
            st.markdown(
                """
The implemented pipeline follows industry-standard machine learning
best practices:

- Modular architecture
- Reproducible workflow
- Automated preprocessing
- Explainable predictions
- Production-ready deployment
- Interactive visualization
- Business-friendly reporting
"""
            )

        logger.info(
            "Machine learning pipeline rendered successfully."
        )

    except Exception as exc:
        logger.exception(
            "Failed to render machine learning pipeline."
        )
        raise RuntimeError(
            "Unable to render machine learning pipeline."
        ) from exc

# =============================================================================
# Project Architecture
# =============================================================================


def render_project_architecture() -> None:
    """
    Render the project architecture section.

    Displays the logical folder structure, modular design philosophy,
    and component responsibilities for the end-to-end machine learning
    project.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the project architecture section cannot be rendered.
    """
    logger.info("Rendering project architecture.")

    try:
        st.header("🏗️ Project Architecture")

        st.markdown(
            """
The project follows a modular, production-oriented architecture designed
to maximize maintainability, scalability, reusability, and separation of
concerns.
            """
        )

        architecture = """
Project/
│
├── app.py
│
├── assets/
│
├── config/
│   └── config.py
│
├── data/
│
├── artifacts/
│
├── pages/
│   ├── 1_Home.py
│   ├── 2_Dataset_Analysis.py
│   ├── 3_Model_Performance.py
│   ├── 4_Customer_Churn_Prediction.py
│   └── 5_About_Project.py
│
├── utils/
│   ├── helpers.py
│   ├── visualization.py
│   ├── validation.py
│   ├── preprocessing.py
│   └── logger.py
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── model_explainability.py
│   ├── model_visualization.py
│   ├── model_reporting.py
│   └── model_persistence.py
│
├── requirements.txt
├── Dockerfile
├── pyproject.toml
└── README.md
        """

        st.code(
            architecture,
            language="text",
        )

        responsibilities = pd.DataFrame(
            {
                "Module": [
                    "config",
                    "utils",
                    "pages",
                    "artifacts",
                    "data_preprocessing",
                    "feature_engineering",
                    "model_training",
                    "model_evaluation",
                    "model_persistence",
                    "deployment",
                ],
                "Responsibility": [
                    "Centralized project configuration",
                    "Reusable helper utilities",
                    "Interactive Streamlit dashboard pages",
                    "Persisted machine learning artifacts",
                    "Data cleaning and preprocessing",
                    "Feature transformation and selection",
                    "Model training and optimization",
                    "Performance evaluation and validation",
                    "Model serialization and loading",
                    "Prediction and inference workflow",
                ],
            }
        )

        st.subheader("📂 Module Responsibilities")

        st.dataframe(
            responsibilities,
            use_container_width=True,
            hide_index=True,
        )

        architecture_chart = pd.DataFrame(
            {
                "Layer": [
                    "Presentation",
                    "Presentation",
                    "Business Logic",
                    "Business Logic",
                    "Machine Learning",
                    "Machine Learning",
                    "Persistence",
                    "Configuration",
                ],
                "Component": [
                    "Streamlit Pages",
                    "Dashboard Assets",
                    "Utility Functions",
                    "Validation",
                    "Training Pipeline",
                    "Prediction Engine",
                    "Saved Artifacts",
                    "Config Module",
                ],
            }
        )

        fig = px.sunburst(
            architecture_chart,
            path=["Layer", "Component"],
            title="Layered Project Architecture",
        )

        fig.update_layout(
            margin=dict(
                t=60,
                l=10,
                r=10,
                b=10,
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        with st.expander(
            "📌 Architecture Highlights",
            expanded=False,
        ):
            st.markdown(
                """
### Highlights

- Modular project architecture
- Clean separation of concerns
- Reusable utility modules
- Independent Streamlit pages
- Centralized configuration management
- Production-ready machine learning pipeline
- Persisted preprocessing and model artifacts
- Scalable deployment structure
- Comprehensive logging and exception handling
- Easy maintenance and future extensibility
                """
            )

        logger.info("Project architecture rendered successfully.")

    except Exception as exc:
        logger.exception(
            "Failed to render project architecture."
        )
        raise RuntimeError(
            "Unable to render project architecture."
        ) from exc


# =============================================================================
# Technology Stack
# =============================================================================


def render_technology_stack() -> None:
    """
    Render the technology stack section.

    Displays the software technologies, machine learning libraries,
    visualization tools, and deployment framework used throughout
    the project.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the technology stack cannot be rendered.
    """
    logger.info("Rendering technology stack.")

    try:
        st.header("🛠️ Technology Stack")

        technologies = pd.DataFrame(
            {
                "Category": [
                    "Programming Language",
                    "Data Processing",
                    "Machine Learning",
                    "Visualization",
                    "Dashboard",
                    "Model Explainability",
                    "Model Persistence",
                    "Configuration",
                    "Logging",
                    "Version Control",
                ],
                "Technology": [
                    "Python 3.12+",
                    "Pandas & NumPy",
                    "Scikit-learn",
                    "Plotly Express",
                    "Streamlit",
                    "SHAP",
                    "Joblib / Pickle",
                    "config.py",
                    "logging",
                    "Git",
                ],
            }
        )

        st.dataframe(
            technologies,
            use_container_width=True,
            hide_index=True,
        )

        category_counts = (
            technologies.groupby("Category")
            .size()
            .reset_index(name="Count")
        )

        fig = px.bar(
            category_counts,
            x="Category",
            y="Count",
            text="Count",
            title="Technology Categories",
        )

        fig.update_traces(
            textposition="outside",
        )

        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Number of Technologies",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💻 Core Libraries")

            st.markdown(
                """
- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly Express
- Streamlit
- SHAP
- Joblib
                """
            )

        with col2:
            st.subheader("⚙️ Engineering Practices")

            st.markdown(
                """
- PEP 8 Compliance
- Complete Type Hints
- NumPy-style Docstrings
- Structured Logging
- Comprehensive Exception Handling
- Modular Architecture
- Reusable Components
- Production-ready Design
                """
            )

        with st.expander(
            "🚀 Why This Technology Stack?",
            expanded=False,
        ):
            st.markdown(
                """
This project leverages widely adopted open-source technologies that are
well suited for enterprise-grade machine learning applications.

### Benefits

- Rapid model development
- Reliable data preprocessing
- Interactive analytical dashboards
- Explainable machine learning
- High-performance visualization
- Production-ready deployment
- Easy maintenance and scalability
- Reproducible machine learning workflows
                """
            )

        logger.info("Technology stack rendered successfully.")

    except Exception as exc:
        logger.exception(
            "Failed to render technology stack."
        )
        raise RuntimeError(
            "Unable to render technology stack."
        ) from exc

def render_project_features() -> None:
    """
    Render the project features section.

    Displays the major functional capabilities implemented within the
    customer churn prediction system.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the project features section cannot be rendered.
    """
    logger.info("Rendering project features.")

    try:
        st.header("✨ Project Features")

        feature_groups = {
            "📊 Data Analysis": [
                "Dataset exploration",
                "Missing value analysis",
                "Duplicate analysis",
                "Correlation analysis",
                "Outlier detection",
            ],
            "🤖 Machine Learning": [
                "Feature engineering",
                "Model training",
                "Model evaluation",
                "Model comparison",
                "Model persistence",
            ],
            "📈 Visualization": [
                "Interactive Plotly charts",
                "Performance dashboards",
                "Probability visualization",
                "Business insights",
            ],
            "🚀 Deployment": [
                "Single prediction",
                "Batch prediction",
                "Download predictions",
                "Interactive Streamlit UI",
            ],
        }

        cols = st.columns(2)

        for index, (title, items) in enumerate(feature_groups.items()):
            with cols[index % 2]:
                with st.container(border=True):
                    st.subheader(title)
                    for item in items:
                        st.markdown(f"- {item}")

        feature_df = pd.DataFrame(
            {
                "Category": [
                    category
                    for category, values in feature_groups.items()
                    for _ in values
                ],
                "Feature": [
                    value
                    for values in feature_groups.values()
                    for value in values
                ],
            }
        )

        fig = px.treemap(
            feature_df,
            path=["Category", "Feature"],
            title="Implemented Features",
        )

        st.plotly_chart(fig, use_container_width=True)

        logger.info("Project features rendered successfully.")

    except Exception as exc:
        logger.exception("Failed to render project features.")
        raise RuntimeError(
            "Unable to render project features."
        ) from exc


def render_business_value() -> None:
    """
    Render the business value section.

    Summarizes the expected organizational value delivered by the
    customer churn prediction solution.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the business value section cannot be rendered.
    """
    logger.info("Rendering business value.")

    try:
        st.header("💼 Business Value")

        values = pd.DataFrame(
            {
                "Business Benefit": [
                    "Early churn detection",
                    "Customer retention",
                    "Marketing optimization",
                    "Operational efficiency",
                    "Decision support",
                    "Revenue protection",
                ],
                "Impact": [
                    "High",
                    "High",
                    "Medium",
                    "Medium",
                    "High",
                    "High",
                ],
            }
        )

        st.dataframe(
            values,
            hide_index=True,
            use_container_width=True,
        )

        impact_counts = (
            values["Impact"]
            .value_counts()
            .rename_axis("Impact")
            .reset_index(name="Count")
        )

        fig = px.bar(
            impact_counts,
            x="Impact",
            y="Count",
            text="Count",
            title="Business Impact Distribution",
        )

        fig.update_traces(textposition="outside")

        st.plotly_chart(fig, use_container_width=True)

        st.success(
            "The solution enables proactive customer retention by "
            "identifying customers at risk before churn occurs."
        )

        with st.expander("Executive Summary"):
            st.markdown(
                """
- Reduce customer attrition.
- Improve customer lifetime value.
- Prioritize high-risk customers.
- Enable proactive retention campaigns.
- Support data-driven decision making.
- Increase long-term profitability.
"""
            )

        logger.info("Business value rendered successfully.")

    except Exception as exc:
        logger.exception("Failed to render business value.")
        raise RuntimeError(
            "Unable to render business value."
        ) from exc


def render_author_information() -> None:
    """
    Render author and project information.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the author information cannot be rendered.
    """
    logger.info("Rendering author information.")

    try:
        st.header("👨‍💻 Author Information")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Project Version", PROJECT_VERSION)
            st.metric("Application", APP_VERSION)

        with col2:
            author_df = pd.DataFrame(
                {
                    "Field": [
                        "Author",
                        "Project",
                        "Internship",
                        "Organization",
                        "Version",
                    ],
                    "Value": [
                        PROJECT_AUTHOR,
                        PROJECT_NAME,
                        INTERNSHIP_NAME,
                        INTERNSHIP_ORGANIZATION,
                        PROJECT_VERSION,
                    ],
                }
            )

            st.dataframe(
                author_df,
                hide_index=True,
                use_container_width=True,
            )

        st.info(
            "This project demonstrates a complete production-ready "
            "machine learning workflow for customer churn prediction."
        )

        logger.info("Author information rendered successfully.")

    except Exception as exc:
        logger.exception("Failed to render author information.")
        raise RuntimeError(
            "Unable to render author information."
        ) from exc


def render_future_scope() -> None:
    """
    Render future enhancement opportunities.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the future scope section cannot be rendered.
    """
    logger.info("Rendering future scope.")

    try:
        st.header("🚀 Future Scope")

        enhancements = pd.DataFrame(
            {
                "Enhancement": [
                    "Real-time prediction API",
                    "Automated model retraining",
                    "Hyperparameter optimization",
                    "Deep learning models",
                    "Explainable AI dashboards",
                    "Cloud deployment",
                    "CI/CD integration",
                    "MLOps pipeline",
                ],
                "Priority": [
                    "High",
                    "High",
                    "Medium",
                    "Medium",
                    "High",
                    "Medium",
                    "Medium",
                    "High",
                ],
            }
        )

        st.dataframe(
            enhancements,
            hide_index=True,
            use_container_width=True,
        )

        roadmap = (
            enhancements["Priority"]
            .value_counts()
            .rename_axis("Priority")
            .reset_index(name="Count")
        )

        fig = px.pie(
            roadmap,
            names="Priority",
            values="Count",
            hole=0.45,
            title="Future Enhancement Roadmap",
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Recommended Roadmap", expanded=True):
            st.markdown(
                """
### Short Term
- Improve feature engineering.
- Automate retraining.
- Add monitoring dashboards.

### Medium Term
- Deploy REST APIs.
- Integrate cloud storage.
- Improve explainability.

### Long Term
- Full MLOps implementation.
- Real-time streaming predictions.
- Enterprise deployment.
- Continuous model monitoring.
"""
            )

        logger.info("Future scope rendered successfully.")

    except Exception as exc:
        logger.exception("Failed to render future scope.")
        raise RuntimeError(
            "Unable to render future scope."
        ) from exc

def render_footer() -> None:
    """
    Render the application footer.

    Displays project information, internship details, technology stack,
    version information, and copyright attribution.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the footer cannot be rendered.
    """
    logger.info("Rendering About Project footer.")

    try:
        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.caption("📂 Project")
            st.markdown(f"**{PROJECT_NAME}**")
            st.caption(f"Version {PROJECT_VERSION}")

        with col2:
            st.caption("👨‍💻 Author")
            st.markdown(f"**{PROJECT_AUTHOR}**")
            st.caption(INTERNSHIP_NAME)

        with col3:
            st.caption("🏢 Organization")
            st.markdown(f"**{INTERNSHIP_ORGANIZATION}**")
            st.caption("Production Ready Streamlit Dashboard")

        st.markdown(
            """
<div style="text-align:center;padding-top:20px;padding-bottom:10px;">
<h4>Predictive Modeling and Risk Scoring for Bank Customer Churn</h4>

End-to-End Machine Learning Project built with
Python • Pandas • NumPy • Scikit-learn • Plotly • Streamlit

<br><br>

© 2026 All Rights Reserved.

</div>
""",
            unsafe_allow_html=True,
        )

        logger.info("Footer rendered successfully.")

    except Exception as exc:
        logger.exception("Failed to render footer.")
        raise RuntimeError(
            "Unable to render footer."
        ) from exc


def main() -> None:
    """
    Execute the About Project page.

    Coordinates rendering of all About Project sections in the intended
    order while providing centralized exception handling and logging.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the page fails to render successfully.
    """
    logger.info("Starting About Project page.")

    try:
        render_page_header()

        st.divider()

        render_project_overview()

        st.divider()

        render_problem_statement()

        st.divider()

        render_dataset_information()

        st.divider()

        render_machine_learning_pipeline()

        st.divider()

        render_project_architecture()

        st.divider()

        render_technology_stack()

        st.divider()

        render_project_features()

        st.divider()

        render_business_value()

        st.divider()

        render_author_information()

        st.divider()

        render_future_scope()

        st.divider()

        render_footer()

        logger.info(
            "About Project page rendered successfully."
        )

    except Exception as exc:
        logger.exception(
            "Unexpected error while rendering About Project page."
        )

        st.error(
            "An unexpected error occurred while rendering the About Project page."
        )

        st.exception(exc)

        raise RuntimeError(
            "About Project page execution failed."
        ) from exc


if __name__ == "__main__":
    main()