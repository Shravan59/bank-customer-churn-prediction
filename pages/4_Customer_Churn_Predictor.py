"""
pages/4_Customer_Churn_Prediction.py
===================================

Interactive Streamlit page for customer churn inference.

This module provides an end-to-end interface for performing customer churn
predictions using persisted machine learning artifacts produced by the project
pipeline. It supports both single-customer prediction and batch prediction from
CSV files while presenting prediction probabilities and executive business
recommendations.

The module integrates with the centralized project configuration, reusable
helper utilities, persisted preprocessing pipeline, trained model, and saved
feature metadata. All inference is performed using previously trained models;
this page does not perform model training or feature engineering.

Features
--------
- Persisted model artifact loading and validation
- Interactive single-customer prediction
- Batch prediction using uploaded CSV files
- Prediction probability visualization
- Downloadable batch prediction results
- Executive summary and business recommendations
- Responsive Streamlit dashboard
- Session state support
- Comprehensive logging and exception handling

Notes
-----
This module is intended to be executed as part of the Streamlit multipage
application for the project:

    Predictive Modeling and Risk Scoring for Bank Customer Churn

Author
------
Configured through ``config.config``.

Version
-------
Configured through ``config.config``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from config.config import (
    BEST_MODEL_PATH,
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    CLEANED_DATA_PATH,
    FEATURE_COLUMNS_PATH,
    MODEL_METRICS_PATH,
    NUMERICAL_COLUMNS,
    PREPROCESSING_PIPELINE_PATH,
    PROJECT_AUTHOR,
    PROJECT_DESCRIPTION,
    PROJECT_NAME,
    PROJECT_VERSION,
    RANDOM_STATE,
    TARGET_COLUMN,
)

from utils.helpers import (
    calculate_file_size,
    check_random_state,
    display_project_info,
    ensure_directory,
    file_exists,
    format_metric,
    get_timestamp,
    load_json,
    load_pickle,
    read_csv,
    save_json,
    save_pickle,
    validate_columns,
    validate_dataframe,
    validate_model_file,
    validate_prediction_input,
    validate_probability,
)

# =============================================================================
# Logging Configuration
# =============================================================================

logger = logging.getLogger(__name__)

if not logger.handlers:
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(funcName)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

logger.setLevel(logging.INFO)
logger.propagate = False

# =============================================================================
# Configuration Imports / Module Constants
# =============================================================================

MODEL_PATH: Path = BEST_MODEL_PATH
PIPELINE_PATH: Path = PREPROCESSING_PIPELINE_PATH
FEATURE_PATH: Path = FEATURE_COLUMNS_PATH
METRICS_PATH: Path = MODEL_METRICS_PATH
DATA_PATH: Path = CLEANED_DATA_PATH

SESSION_MODEL_KEY: str = "model_artifacts"
SESSION_TEMPLATE_KEY: str = "prediction_template"

SUPPORTED_UPLOAD_TYPES: tuple[str, ...] = ("csv",)

DEFAULT_DOWNLOAD_FILENAME: str = (
    "customer_churn_predictions.csv"
)
def load_model_artifacts() -> dict[str, Any]:
    """
    Load and validate persisted machine learning artifacts.

    This function loads the trained model, preprocessing pipeline,
    feature metadata, and optional evaluation artifacts required for
    customer churn prediction. Successfully loaded artifacts are cached
    in Streamlit session state to avoid repeated disk I/O.

    Returns
    -------
    dict[str, Any]
        Dictionary containing validated model artifacts.

    Raises
    ------
    RuntimeError
        If one or more required artifacts cannot be loaded or validated.
    """
    logger.info("Loading persisted model artifacts.")

    try:
        if SESSION_MODEL_KEY in st.session_state:
            logger.info("Using cached model artifacts.")
            return st.session_state[SESSION_MODEL_KEY]

        required_files = {
            "model": MODEL_PATH,
            "pipeline": PIPELINE_PATH,
            "feature_columns": FEATURE_PATH,
        }

        for artifact_name, artifact_path in required_files.items():
            validate_model_file(artifact_path)

            if not file_exists(artifact_path):
                raise FileNotFoundError(
                    f"Required artifact '{artifact_name}' was not found: "
                    f"{artifact_path}"
                )

        model = load_pickle(MODEL_PATH)

        if not isinstance(model, BaseEstimator):
            raise TypeError(
                "Loaded model is not a valid scikit-learn estimator."
            )

        pipeline = load_pickle(PIPELINE_PATH)

        if not isinstance(
            pipeline,
            (Pipeline, ColumnTransformer),
        ):
            raise TypeError(
                "Loaded preprocessing pipeline is invalid."
            )

        feature_columns = load_pickle(FEATURE_PATH)

        if not isinstance(feature_columns, (list, tuple)):
            raise TypeError(
                "Feature column metadata must be a list."
            )

        feature_columns = list(feature_columns)

        if not feature_columns:
            raise ValueError("Feature column list is empty.")

        metrics: dict[str, Any] = {}

        if file_exists(METRICS_PATH):
            try:
                metrics = load_pickle(METRICS_PATH)
            except Exception:
                logger.warning(
                    "Unable to load model metrics artifact."
                )

        artifacts: dict[str, Any] = {
            "model": model,
            "pipeline": pipeline,
            "feature_columns": feature_columns,
            "metrics": metrics,
            "model_path": MODEL_PATH,
            "pipeline_path": PIPELINE_PATH,
            "feature_path": FEATURE_PATH,
            "loaded_at": get_timestamp(),
        }

        st.session_state[SESSION_MODEL_KEY] = artifacts

        logger.info("Model artifacts loaded successfully.")

        return artifacts

    except Exception as exc:
        logger.exception("Failed to load model artifacts.")
        raise RuntimeError(
            "Unable to load persisted prediction artifacts."
        ) from exc


@st.cache_data(show_spinner=False)
def load_prediction_template() -> pd.DataFrame:
    """
    Load a prediction template for user input.

    The template is generated from the processed dataset by removing
    identifier and target columns, leaving only model input features.
    It serves as the reference schema for both single-customer and
    batch prediction workflows.

    Returns
    -------
    pandas.DataFrame
        Empty dataframe preserving the expected feature structure.

    Raises
    ------
    RuntimeError
        If the template cannot be created.
    """
    logger.info("Loading prediction template.")

    try:
        dataframe = read_csv(DATA_PATH)

        validate_dataframe(dataframe)

        drop_columns = [
            column
            for column in dataframe.columns
            if column == TARGET_COLUMN
            or column.lower() in {"id", "customerid", "rownumber"}
        ]

        template = dataframe.drop(
            columns=drop_columns,
            errors="ignore",
        ).head(0)

        st.session_state[SESSION_TEMPLATE_KEY] = template

        logger.info(
            "Prediction template loaded with %d features.",
            len(template.columns),
        )

        return template

    except Exception as exc:
        logger.exception(
            "Failed to create prediction template."
        )
        raise RuntimeError(
            "Unable to load the prediction template."
        ) from exc

def render_page_header() -> None:
    """
    Render the page header.

    Displays the page title, project information, description, workflow
    summary, and usage instructions for customer churn prediction.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the page header cannot be rendered.
    """
    logger.info("Rendering Customer Churn Prediction page header.")

    try:
        st.title("🎯 Customer Churn Prediction")

        st.caption(
            f"{PROJECT_NAME} • Version {PROJECT_VERSION}"
        )

        st.markdown(
            PROJECT_DESCRIPTION,
        )

        st.info(
            """
This dashboard enables real-time customer churn prediction using the
trained machine learning model.

### Supported Prediction Modes

- 👤 Single Customer Prediction
- 📄 Batch Prediction (CSV Upload)

### Capabilities

- Automated preprocessing
- Probability estimation
- Business interpretation
- Downloadable prediction results
- Executive insights
"""
        )

        with st.expander(
            "ℹ️ Prediction Workflow",
            expanded=False,
        ):
            st.markdown(
                """
1. Load the persisted model artifacts.

2. Enter customer information or upload a CSV file.

3. Validate all inputs.

4. Apply the preprocessing pipeline.

5. Generate churn predictions.

6. Display prediction probabilities.

7. Download prediction results.
"""
            )

        breadcrumb_col1, breadcrumb_col2 = st.columns([4, 1])

        with breadcrumb_col1:
            st.markdown(
                f"**Project:** {PROJECT_NAME}"
            )

        with breadcrumb_col2:
            st.markdown(
                f"**Version:** {PROJECT_VERSION}"
            )

        st.divider()

        logger.info("Page header rendered successfully.")

    except Exception as exc:
        logger.exception(
            "Unable to render page header."
        )
        raise RuntimeError(
            "Failed to render page header."
        ) from exc


def render_sidebar() -> None:
    """
    Render the application sidebar.

    Displays model information, artifact availability, prediction options,
    batch prediction settings, display preferences, and project metadata.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the sidebar cannot be rendered.
    """
    logger.info("Rendering sidebar.")

    try:
        artifacts = load_model_artifacts()

        st.sidebar.title("⚙️ Prediction Controls")

        st.sidebar.subheader("🤖 Model Information")

        model = artifacts.get("model")

        st.sidebar.write(
            f"**Estimator:** {type(model).__name__}"
        )

        st.sidebar.write(
            f"**Random State:** {RANDOM_STATE}"
        )

        st.sidebar.write(
            f"**Features:** {len(artifacts['feature_columns'])}"
        )

        st.sidebar.divider()

        st.sidebar.subheader("📦 Artifact Status")

        artifact_status = pd.DataFrame(
            {
                "Artifact": [
                    "Model",
                    "Pipeline",
                    "Feature Columns",
                    "Metrics",
                ],
                "Status": [
                    "✅ Loaded",
                    "✅ Loaded",
                    "✅ Loaded",
                    (
                        "✅ Loaded"
                        if artifacts.get("metrics")
                        else "⚠ Optional"
                    ),
                ],
            }
        )

        st.sidebar.dataframe(
            artifact_status,
            hide_index=True,
            use_container_width=True,
        )

        st.sidebar.divider()

        st.sidebar.subheader("🔮 Prediction Options")

        st.session_state["prediction_mode"] = (
            st.sidebar.radio(
                "Prediction Mode",
                (
                    "Single Customer",
                    "Batch Prediction",
                ),
                index=0,
            )
        )

        st.session_state[
            "show_probability"
        ] = st.sidebar.checkbox(
            "Show Prediction Probability",
            value=True,
        )

        st.session_state[
            "show_recommendations"
        ] = st.sidebar.checkbox(
            "Show Business Recommendations",
            value=True,
        )

        st.sidebar.divider()

        st.sidebar.subheader("📄 Batch Prediction")

        st.session_state[
            "preview_batch"
        ] = st.sidebar.checkbox(
            "Preview Uploaded Dataset",
            value=True,
        )

        st.session_state[
            "download_predictions"
        ] = st.sidebar.checkbox(
            "Enable CSV Download",
            value=True,
        )

        st.sidebar.caption(
            "Supported format: CSV"
        )

        st.sidebar.divider()

        st.sidebar.subheader("🎨 Display Settings")

        st.session_state[
            "expanded_results"
        ] = st.sidebar.checkbox(
            "Expand Prediction Results",
            value=True,
        )

        st.session_state[
            "show_probability_chart"
        ] = st.sidebar.checkbox(
            "Show Probability Visualization",
            value=True,
        )

        st.sidebar.divider()

        st.sidebar.subheader("📘 Project Information")

        project_info = display_project_info()

        if isinstance(project_info, dict):
            for key, value in project_info.items():
                st.sidebar.caption(
                    f"**{key}:** {value}"
                )
        else:
            st.sidebar.caption(project_info)

        st.sidebar.markdown("---")

        st.sidebar.caption(
            f"**Project:** {PROJECT_NAME}"
        )

        st.sidebar.caption(
            f"**Version:** {PROJECT_VERSION}"
        )

        st.sidebar.caption(
            f"**Author:** {PROJECT_AUTHOR}"
        )

        logger.info("Sidebar rendered successfully.")

    except Exception as exc:
        logger.exception(
            "Failed to render sidebar."
        )
        raise RuntimeError(
            "Unable to render application sidebar."
        ) from exc

def render_single_prediction_form() -> tuple[int | str | None, float | None, pd.DataFrame | None]:
    """
    Render a schema-agnostic single customer prediction form.

    The form is constructed dynamically using persisted model artifacts and
    optional metadata. When metadata describing feature types and categorical
    values is available, appropriate Streamlit widgets are selected
    automatically. Otherwise, generic numeric inputs are created for every
    discovered feature.

    The function performs the complete inference workflow:

    1. Load persisted artifacts.
    2. Build the input form dynamically.
    3. Validate user input.
    4. Create a one-row inference dataframe.
    5. Apply the persisted preprocessing pipeline.
    6. Generate prediction.
    7. Compute prediction probability (if supported).

    Returns
    -------
    tuple[int | str | None, float | None, pandas.DataFrame | None]
        Tuple containing:

        - prediction
        - prediction probability (None if unavailable)
        - one-row input dataframe

        Returns (None, None, None) until the prediction button is pressed.

    Raises
    ------
    RuntimeError
        If the prediction workflow cannot be completed successfully.
    """
    logger.info("Rendering dynamic single prediction form.")

    try:
        artifacts = load_model_artifacts()

        model = artifacts["model"]
        pipeline = artifacts["pipeline"]
        feature_names = list(artifacts["feature_columns"])

        metadata: dict[str, Any] = {}

        possible_metadata = [
            MODEL_PATH.parent / "feature_info.json",
            MODEL_PATH.parent / "metadata.json",
            MODEL_PATH.parent / "feature_metadata.json",
            MODEL_PATH.parent / "model_metadata.json",
        ]

        for metadata_path in possible_metadata:
            if file_exists(metadata_path):
                try:
                    metadata = load_json(metadata_path)
                    logger.info(
                        "Loaded metadata from %s",
                        metadata_path.name,
                    )
                    break
                except Exception:
                    logger.warning(
                        "Unable to load metadata from %s",
                        metadata_path,
                    )

        st.header("👤 Single Customer Prediction")

        st.caption(
            "Complete the dynamically generated input form below."
        )

        with st.form(
            key="single_prediction_form",
            clear_on_submit=False,
        ):
            input_values: dict[str, Any] = {}

            feature_info = metadata.get("features", {})

            for feature in feature_names:

                info = feature_info.get(feature, {})

                feature_type = str(
                    info.get(
                        "type",
                        info.get("dtype", "numeric"),
                    )
                ).lower()

                default = info.get("default", 0)

                help_text = info.get(
                    "description",
                    f"Input value for {feature}",
                )

                if (
                    feature_type in {
                        "categorical",
                        "category",
                        "object",
                        "string",
                    }
                    and info.get("categories")
                ):

                    categories = list(info["categories"])

                    default_index = 0

                    if default in categories:
                        default_index = categories.index(default)

                    input_values[feature] = st.selectbox(
                        label=feature,
                        options=categories,
                        index=default_index,
                        help=help_text,
                    )

                elif feature_type in {
                    "bool",
                    "boolean",
                }:

                    input_values[feature] = st.checkbox(
                        feature,
                        value=bool(default),
                        help=help_text,
                    )

                elif feature_type in {
                    "int",
                    "integer",
                    "int32",
                    "int64",
                }:

                    input_values[feature] = st.number_input(
                        label=feature,
                        value=int(default),
                        step=1,
                        help=help_text,
                    )

                else:

                    input_values[feature] = st.number_input(
                        label=feature,
                        value=float(default),
                        help=help_text,
                    )

            submitted = st.form_submit_button(
                "🔮 Predict Customer Churn",
                use_container_width=True,
            )

        if not submitted:
            return None, None, None

        inference_dataframe = pd.DataFrame(
            [input_values],
            columns=feature_names,
        )

        validate_prediction_input(
            inference_dataframe,
            feature_names,
        )

        transformed = pipeline.transform(
            inference_dataframe,
        )

        prediction = model.predict(transformed)[0]

        probability: float | None = None

        if hasattr(model, "predict_proba"):
            probability = float(
                model.predict_proba(transformed)[0][1]
            )
            validate_probability(probability)

        logger.info(
            "Single prediction completed successfully."
        )

        return (
            prediction,
            probability,
            inference_dataframe,
        )

    except Exception as exc:
        logger.exception(
            "Single customer prediction failed."
        )
        raise RuntimeError(
            "Failed to generate customer churn prediction."
        ) from exc

def predict_single_customer(
    input_dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Predict customer churn for a single customer record.

    This function performs end-to-end inference using the persisted
    preprocessing pipeline and trained model loaded from project artifacts.
    The implementation is schema-agnostic and relies on the persisted
    feature names and optional metadata rather than hardcoded dataset
    columns.

    Parameters
    ----------
    input_dataframe : pandas.DataFrame
        Single-row dataframe containing customer information.

    Returns
    -------
    dict[str, Any]
        Dictionary containing prediction results with the following keys:

        - ``prediction`` : int | str
        - ``prediction_label`` : str
        - ``prediction_probability`` : float | None
        - ``confidence`` : float | None
        - ``processed_features`` : Any
        - ``input_dataframe`` : pandas.DataFrame
        - ``feature_count`` : int
        - ``model_name`` : str
        - ``prediction_timestamp`` : str

    Raises
    ------
    RuntimeError
        If prediction cannot be completed successfully.
    """
    logger.info("Starting single customer prediction.")

    try:
        if input_dataframe is None:
            raise ValueError("Input dataframe cannot be None.")

        if input_dataframe.empty:
            raise ValueError("Input dataframe is empty.")

        artifacts = load_model_artifacts()

        model = artifacts["model"]
        pipeline = artifacts["pipeline"]
        feature_names = list(artifacts["feature_columns"])

        missing_features = [
            feature
            for feature in feature_names
            if feature not in input_dataframe.columns
        ]

        if missing_features:
            raise ValueError(
                "Missing required feature(s): "
                f"{', '.join(missing_features)}"
            )

        inference_dataframe = input_dataframe.loc[
            :, feature_names
        ].copy()

        inference_dataframe = inference_dataframe.replace(
            {
                np.inf: np.nan,
                -np.inf: np.nan,
            }
        )

        if inference_dataframe.isnull().any().any():
            raise ValueError(
                "Input contains missing values."
            )

        logger.info(
            "Applying persisted preprocessing pipeline."
        )

        processed_features = pipeline.transform(
            inference_dataframe
        )

        logger.info("Generating model prediction.")

        prediction = model.predict(processed_features)[0]

        probability: float | None = None
        confidence: float | None = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(
                processed_features
            )[0]

            if len(probabilities) >= 2:
                probability = float(probabilities[1])
            else:
                probability = float(np.max(probabilities))

            confidence = float(np.max(probabilities))

        prediction_label = (
            "Churn"
            if str(prediction).lower() in {"1", "true", "yes", "churn"}
            else "No Churn"
        )

        result: dict[str, Any] = {
            "prediction": prediction,
            "prediction_label": prediction_label,
            "prediction_probability": probability,
            "confidence": confidence,
            "processed_features": processed_features,
            "input_dataframe": inference_dataframe,
            "feature_count": len(feature_names),
            "model_name": type(model).__name__,
            "prediction_timestamp": datetime.now().isoformat(),
        }

        logger.info(
            "Prediction completed successfully. "
            "Prediction=%s Probability=%s",
            prediction,
            probability,
        )

        return result

    except Exception as exc:
        logger.exception(
            "Unexpected error during single customer prediction."
        )
        raise RuntimeError(
            "Failed to perform single customer prediction."
        ) from exc

def render_prediction_result(
    prediction_result: dict[str, Any],
) -> None:
    """
    Render the single customer prediction results.

    Displays the prediction outcome, prediction probability (when available),
    confidence score, summary KPI cards, and a concise business interpretation.

    Parameters
    ----------
    prediction_result : dict[str, Any]
        Dictionary returned by ``predict_single_customer()``.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the prediction result cannot be rendered.
    """
    logger.info("Rendering prediction result.")

    try:
        if not prediction_result:
            st.info(
                "Submit the prediction form to view the prediction results."
            )
            return

        prediction = prediction_result.get("prediction")
        prediction_label = prediction_result.get(
            "prediction_label",
            str(prediction),
        )
        probability = prediction_result.get(
            "prediction_probability"
        )
        confidence = prediction_result.get("confidence")
        model_name = prediction_result.get(
            "model_name",
            "Unknown",
        )
        timestamp = prediction_result.get(
            "prediction_timestamp",
            "N/A",
        )

        st.header("📋 Prediction Result")

        churn_labels = {
            "1",
            "true",
            "yes",
            "churn",
        }

        is_churn = (
            str(prediction).strip().lower() in churn_labels
            or prediction_label.strip().lower() == "churn"
        )

        if is_churn:
            st.error(
                "⚠️ The model predicts that this customer is likely to churn."
            )
        else:
            st.success(
                "✅ The model predicts that this customer is likely to be retained."
            )

        metric_cols = st.columns(4)

        with metric_cols[0]:
            st.metric(
                "Prediction",
                prediction_label,
            )

        with metric_cols[1]:
            st.metric(
                "Probability",
                (
                    f"{probability:.2%}"
                    if probability is not None
                    else "N/A"
                ),
            )

        with metric_cols[2]:
            st.metric(
                "Confidence",
                (
                    f"{confidence:.2%}"
                    if confidence is not None
                    else "N/A"
                ),
            )

        with metric_cols[3]:
            st.metric(
                "Model",
                model_name,
            )

        summary_df = pd.DataFrame(
            {
                "Attribute": [
                    "Prediction",
                    "Probability",
                    "Confidence",
                    "Model",
                    "Prediction Time",
                ],
                "Value": [
                    prediction_label,
                    (
                        f"{probability:.4f}"
                        if probability is not None
                        else "N/A"
                    ),
                    (
                        f"{confidence:.4f}"
                        if confidence is not None
                        else "N/A"
                    ),
                    model_name,
                    timestamp,
                ],
            }
        )

        st.subheader("Prediction Summary")

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )

        if probability is not None:
            if probability >= 0.80:
                risk_level = "Very High"
            elif probability >= 0.60:
                risk_level = "High"
            elif probability >= 0.40:
                risk_level = "Moderate"
            elif probability >= 0.20:
                risk_level = "Low"
            else:
                risk_level = "Very Low"

            st.info(
                f"**Estimated Churn Risk:** {risk_level}"
            )

        with st.expander(
            "💼 Business Interpretation",
            expanded=True,
        ):
            if is_churn:
                st.markdown(
                    """
- The customer exhibits characteristics associated with churn.
- Consider proactive retention campaigns.
- Review customer engagement and support history.
- Personalized offers or loyalty incentives may reduce churn risk.
- Monitor this customer closely over the coming period.
"""
                )
            else:
                st.markdown(
                    """
- The customer appears likely to remain with the bank.
- Continue maintaining customer satisfaction.
- Explore opportunities for cross-selling and upselling.
- Maintain regular engagement to preserve loyalty.
"""
                )

        logger.info("Prediction result rendered successfully.")

    except Exception as exc:
        logger.exception(
            "Failed to render prediction result."
        )
        raise RuntimeError(
            "Unable to render prediction result."
        ) from exc

def render_probability_visualization(
    prediction_result: dict[str, Any],
) -> None:
    """
    Render prediction probability visualizations.

    Displays an interactive probability gauge and probability comparison
    charts for the predicted customer using Plotly Express where applicable.
    If prediction probabilities are unavailable, an informational message is
    shown instead.

    Parameters
    ----------
    prediction_result : dict[str, Any]
        Dictionary returned by ``predict_single_customer()``.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the probability visualization cannot be rendered.
    """
    logger.info("Rendering prediction probability visualization.")

    try:
        if not prediction_result:
            st.info(
                "Run a prediction to visualize prediction probabilities."
            )
            return

        probability = prediction_result.get(
            "prediction_probability"
        )

        if probability is None:
            st.info(
                "The loaded model does not support probability estimation."
            )
            return

        probability = float(probability)
        probability = max(0.0, min(1.0, probability))

        churn_probability = probability * 100.0
        retention_probability = (1.0 - probability) * 100.0

        st.header("📊 Prediction Probability")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric(
                "Churn Probability",
                f"{churn_probability:.2f}%",
            )

        with metric_col2:
            st.metric(
                "Retention Probability",
                f"{retention_probability:.2f}%",
            )

        with metric_col3:
            if churn_probability >= 80:
                risk_level = "Very High"
            elif churn_probability >= 60:
                risk_level = "High"
            elif churn_probability >= 40:
                risk_level = "Moderate"
            elif churn_probability >= 20:
                risk_level = "Low"
            else:
                risk_level = "Very Low"

            st.metric(
                "Risk Level",
                risk_level,
            )

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=churn_probability,
                    number={
                        "suffix": "%",
                        "font": {"size": 36},
                    },
                    title={
                        "text": "Churn Probability",
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                        },
                        "bar": {
                            "color": "#1f77b4",
                        },
                        "steps": [
                            {
                                "range": [0, 20],
                                "color": "#d4edda",
                            },
                            {
                                "range": [20, 40],
                                "color": "#fff3cd",
                            },
                            {
                                "range": [40, 60],
                                "color": "#ffe8a1",
                            },
                            {
                                "range": [60, 80],
                                "color": "#f8d7da",
                            },
                            {
                                "range": [80, 100],
                                "color": "#f5c6cb",
                            },
                        ],
                    },
                )
            )

            gauge.update_layout(
                height=400,
                margin=dict(
                    l=20,
                    r=20,
                    t=50,
                    b=20,
                ),
            )

            st.plotly_chart(
                gauge,
                use_container_width=True,
            )

        with chart_col2:
            probability_df = pd.DataFrame(
                {
                    "Outcome": [
                        "Retained",
                        "Churn",
                    ],
                    "Probability": [
                        retention_probability,
                        churn_probability,
                    ],
                }
            )

            bar_chart = px.bar(
                probability_df,
                x="Outcome",
                y="Probability",
                text="Probability",
                title="Prediction Probability Distribution",
            )

            bar_chart.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside",
            )

            bar_chart.update_layout(
                yaxis_title="Probability (%)",
                xaxis_title="Prediction Outcome",
                height=400,
            )

            st.plotly_chart(
                bar_chart,
                use_container_width=True,
            )

        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Churn Probability",
                    "Retention Probability",
                    "Predicted Class",
                    "Risk Level",
                ],
                "Value": [
                    f"{churn_probability:.2f}%",
                    f"{retention_probability:.2f}%",
                    prediction_result.get(
                        "prediction_label",
                        "Unknown",
                    ),
                    risk_level,
                ],
            }
        )

        st.subheader("Probability Summary")

        st.dataframe(
            summary_df,
            hide_index=True,
            use_container_width=True,
        )

        with st.expander(
            "💼 Business Interpretation",
            expanded=True,
        ):
            st.markdown(
                f"""
- **Predicted churn probability:** **{churn_probability:.2f}%**
- **Predicted retention probability:** **{retention_probability:.2f}%**
- Customers with higher churn probabilities should be prioritized for retention initiatives.
- Combine probability estimates with business value before taking action.
- Probability scores can support customer segmentation, campaign prioritization, and proactive intervention strategies.
- Confidence should always be interpreted alongside model performance metrics and domain expertise.
"""
            )

        logger.info(
            "Prediction probability visualization rendered successfully."
        )

    except Exception as exc:
        logger.exception(
            "Failed to render probability visualization."
        )
        raise RuntimeError(
            "Unable to render prediction probability visualization."
        ) from exc

def render_batch_prediction(
    dataframe: pd.DataFrame | None = None,
) -> None:
    """
    Render the batch customer churn prediction interface.

    This function enables users to upload a CSV file containing customer
    records, validates the uploaded dataset against the persisted model
    artifacts, performs batch inference, visualizes prediction results,
    and provides a downloadable CSV containing predictions.

    The implementation is schema-agnostic and discovers the required feature
    schema dynamically from the persisted model artifacts.

    Parameters
    ----------
    dataframe : pandas.DataFrame | None, default=None
        Optional dataframe to use instead of uploading a CSV file.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If an unexpected error occurs during batch prediction.
    """
    logger.info("Rendering batch prediction interface.")

    try:
        st.header("📄 Batch Customer Churn Prediction")

        uploaded_file = None

        if dataframe is None:
            uploaded_file = st.file_uploader(
                label="Upload Customer Dataset (CSV)",
                type=["csv"],
                accept_multiple_files=False,
                help="Upload a CSV containing customer records for batch prediction.",
            )

            if uploaded_file is None:
                st.info("Upload a CSV file to begin batch prediction.")
                return

            if uploaded_file.size == 0:
                st.error("The uploaded file is empty.")
                return

            try:
                dataframe = pd.read_csv(uploaded_file)
            except pd.errors.EmptyDataError:
                st.error("The uploaded CSV file is empty.")
                return
            except pd.errors.ParserError:
                st.error("Unable to parse the uploaded CSV file.")
                return
            except Exception as exc:
                st.error(f"Invalid CSV file: {exc}")
                return

        if dataframe.empty:
            st.warning("The supplied dataset contains no records.")
            return

        artifacts = load_model_artifacts()

        model = artifacts.get("model")
        pipeline = artifacts.get("pipeline")

        feature_names = list(
            artifacts.get("feature_columns", [])
        )

        metadata = (
            artifacts.get("metadata")
            or artifacts.get("feature_metadata")
            or {}
        )

        st.subheader("Dataset Preview")

        st.dataframe(
            dataframe.head(),
            use_container_width=True,
        )

        st.caption(
            f"Rows: {len(dataframe):,} | Columns: {dataframe.shape[1]}"
        )

        validation_rows: list[tuple[str, str]] = []

        missing_columns: list[str] = []
        unexpected_columns: list[str] = []

        if feature_names:

            missing_columns = [
                feature
                for feature in feature_names
                if feature not in dataframe.columns
            ]

            unexpected_columns = [
                column
                for column in dataframe.columns
                if column not in feature_names
            ]

            validation_rows.extend(
                [
                    (
                        "Required Features",
                        str(len(feature_names)),
                    ),
                    (
                        "Missing Columns",
                        str(len(missing_columns)),
                    ),
                    (
                        "Unexpected Columns",
                        str(len(unexpected_columns)),
                    ),
                ]
            )

        validation_df = pd.DataFrame(
            validation_rows,
            columns=[
                "Validation",
                "Result",
            ],
        )

        st.subheader("Validation Summary")

        st.dataframe(
            validation_df,
            hide_index=True,
            use_container_width=True,
        )

        if missing_columns:
            st.error(
                "The uploaded dataset is missing required feature columns."
            )
            st.write(missing_columns)
            return

        if unexpected_columns:
            st.warning(
                "Unexpected columns detected. They will be ignored."
            )
            st.write(unexpected_columns)

        inference_df = dataframe.copy()

        if feature_names:
            inference_df = inference_df.loc[:, feature_names]

        inference_df = inference_df.replace(
            {
                np.inf: np.nan,
                -np.inf: np.nan,
            }
        )

        if inference_df.isnull().any().any():
            st.error(
                "The uploaded dataset contains missing values."
            )
            return

        if pipeline is not None:
            processed = pipeline.transform(inference_df)
        else:
            processed = inference_df

        predictions = model.predict(processed)

        probabilities: np.ndarray | None = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(processed)[:, 1]

        results_df = dataframe.copy()

        results_df["Prediction"] = predictions

        label_map = {
            0: "Retained",
            1: "Churn",
            False: "Retained",
            True: "Churn",
        }

        results_df["Prediction Label"] = [
            label_map.get(
                prediction,
                str(prediction),
            )
            for prediction in predictions
        ]

        if probabilities is not None:

            results_df["Churn Probability"] = probabilities

            def _risk(probability: float) -> str:
                if probability >= 0.70:
                    return "High Risk"
                if probability >= 0.40:
                    return "Medium Risk"
                return "Low Risk"

            results_df["Risk Category"] = [
                _risk(probability)
                for probability in probabilities
            ]

        else:
            results_df["Risk Category"] = "Unknown"

        st.subheader("Prediction Results")

        st.dataframe(
            results_df,
            use_container_width=True,
        )

        total_customers = len(results_df)

        churn_count = int(
            (
                results_df["Prediction Label"]
                == "Churn"
            ).sum()
        )

        retained_count = total_customers - churn_count

        average_probability = (
            float(results_df["Churn Probability"].mean())
            if "Churn Probability" in results_df.columns
            else None
        )

        metric_columns = st.columns(4)

        metric_columns[0].metric(
            "Customers",
            f"{total_customers:,}",
        )

        metric_columns[1].metric(
            "Predicted Churn",
            f"{churn_count:,}",
        )

        metric_columns[2].metric(
            "Retained",
            f"{retained_count:,}",
        )

        metric_columns[3].metric(
            "Average Probability",
            (
                f"{average_probability:.2%}"
                if average_probability is not None
                else "N/A"
            ),
        )

        distribution = (
            results_df["Prediction Label"]
            .value_counts()
            .rename_axis("Prediction")
            .reset_index(name="Count")
        )

        prediction_chart = px.bar(
            distribution,
            x="Prediction",
            y="Count",
            text="Count",
            title="Prediction Class Distribution",
        )

        st.plotly_chart(
            prediction_chart,
            use_container_width=True,
        )

        if "Churn Probability" in results_df.columns:

            probability_chart = px.histogram(
                results_df,
                x="Churn Probability",
                nbins=25,
                title="Probability Distribution",
            )

            st.plotly_chart(
                probability_chart,
                use_container_width=True,
            )

        risk_distribution = (
            results_df["Risk Category"]
            .value_counts()
            .rename_axis("Risk Category")
            .reset_index(name="Count")
        )

        risk_chart = px.bar(
            risk_distribution,
            x="Risk Category",
            y="Count",
            text="Count",
            title="Risk Category Distribution",
        )

        st.plotly_chart(
            risk_chart,
            use_container_width=True,
        )

        csv_bytes = results_df.to_csv(
            index=False,
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction Results",
            data=csv_bytes,
            file_name="batch_customer_churn_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

        logger.info(
            "Batch prediction completed successfully for %d records.",
            total_customers,
        )

    except Exception as exc:
        logger.exception(
            "Unexpected error during batch prediction."
        )
        st.error(
            "An unexpected error occurred while processing batch prediction."
        )
        raise RuntimeError(
            "Batch prediction failed."
        ) from exc

def render_prediction_summary(
    prediction_results: pd.DataFrame | None,
) -> None:
    """
    Render an executive summary of customer churn prediction results.

    This function summarizes batch prediction outputs, computes key business
    metrics, displays interactive KPI cards, presents an executive summary
    table, and visualizes prediction outcomes using Plotly Express.

    Parameters
    ----------
    prediction_results : pandas.DataFrame | None
        DataFrame containing prediction results generated by the batch
        prediction workflow. The function dynamically discovers available
        prediction-related columns and gracefully handles missing optional
        columns.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If an unexpected error occurs while rendering the summary.
    """
    logger.info("Rendering prediction summary section.")

    try:
        st.header("📈 Prediction Summary")

        if prediction_results is None:
            st.info("No prediction results available.")
            return

        if prediction_results.empty:
            st.warning("Prediction results are empty.")
            return

        total_records = len(prediction_results)

        prediction_column = None
        probability_column = None
        risk_column = None

        prediction_candidates = (
            "Prediction Label",
            "Prediction",
            "Predicted Label",
            "Predicted_Class",
        )

        probability_candidates = (
            "Churn Probability",
            "Probability",
            "Prediction Probability",
        )

        risk_candidates = (
            "Risk Category",
            "Risk",
        )

        for column in prediction_candidates:
            if column in prediction_results.columns:
                prediction_column = column
                break

        for column in probability_candidates:
            if column in prediction_results.columns:
                probability_column = column
                break

        for column in risk_candidates:
            if column in prediction_results.columns:
                risk_column = column
                break

        churn_count = 0
        retained_count = 0

        if prediction_column is not None:

            prediction_series = (
                prediction_results[prediction_column]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            churn_count = int(
                prediction_series.isin(
                    [
                        "1",
                        "true",
                        "yes",
                        "churn",
                    ]
                ).sum()
            )

            retained_count = total_records - churn_count

        average_probability = None

        if probability_column is not None:
            average_probability = float(
                prediction_results[
                    probability_column
                ].mean()
            )

        summary_columns = st.columns(4)

        with summary_columns[0]:
            st.metric(
                "Total Customers",
                f"{total_records:,}",
            )

        with summary_columns[1]:
            st.metric(
                "Predicted Churn",
                f"{churn_count:,}",
            )

        with summary_columns[2]:
            st.metric(
                "Predicted Retained",
                f"{retained_count:,}",
            )

        with summary_columns[3]:
            st.metric(
                "Average Probability",
                (
                    f"{average_probability:.2%}"
                    if average_probability is not None
                    else "N/A"
                ),
            )

        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Total Customers",
                    "Predicted Churn",
                    "Predicted Retained",
                    "Average Churn Probability",
                    "Prediction Coverage",
                ],
                "Value": [
                    total_records,
                    churn_count,
                    retained_count,
                    (
                        f"{average_probability:.2%}"
                        if average_probability is not None
                        else "N/A"
                    ),
                    "100%",
                ],
            }
        )

        st.subheader("Executive Summary")

        st.dataframe(
            summary_df,
            hide_index=True,
            use_container_width=True,
        )

        if prediction_column is not None:

            distribution_df = (
                prediction_results[prediction_column]
                .value_counts()
                .rename_axis("Prediction")
                .reset_index(name="Count")
            )

            prediction_chart = px.pie(
                distribution_df,
                names="Prediction",
                values="Count",
                hole=0.45,
                title="Prediction Distribution",
            )

            st.plotly_chart(
                prediction_chart,
                use_container_width=True,
            )

        if probability_column is not None:

            probability_chart = px.histogram(
                prediction_results,
                x=probability_column,
                nbins=25,
                title="Prediction Probability Distribution",
            )

            st.plotly_chart(
                probability_chart,
                use_container_width=True,
            )

        if risk_column is not None:

            risk_distribution = (
                prediction_results[risk_column]
                .value_counts()
                .rename_axis("Risk Category")
                .reset_index(name="Count")
            )

            risk_chart = px.bar(
                risk_distribution,
                x="Risk Category",
                y="Count",
                text="Count",
                title="Risk Category Distribution",
            )

            risk_chart.update_traces(
                textposition="outside",
            )

            st.plotly_chart(
                risk_chart,
                use_container_width=True,
            )

        with st.expander(
            "📋 Executive Interpretation",
            expanded=True,
        ):
            churn_rate = (
                (churn_count / total_records) * 100
                if total_records > 0
                else 0.0
            )

            st.markdown(
                f"""
- **Customers Analysed:** **{total_records:,}**
- **Predicted Churn Rate:** **{churn_rate:.2f}%**
- **Predicted Retention Rate:** **{100 - churn_rate:.2f}%**
- Customers classified as **High Risk** should be prioritized for retention initiatives.
- Prediction probabilities can be used to rank customers for targeted intervention campaigns.
- Results should be interpreted alongside business knowledge and existing customer engagement strategies.
"""
            )

        logger.info(
            "Prediction summary rendered successfully."
        )

    except Exception as exc:
        logger.exception(
            "Failed to render prediction summary."
        )
        raise RuntimeError(
            "Unable to render prediction summary."
        ) from exc

def render_business_recommendations(
    prediction_results: pd.DataFrame | None = None,
) -> None:
    """
    Render executive business recommendations based on churn prediction results.

    This function provides actionable business recommendations derived from
    batch prediction outputs. When prediction results are available, summary
    statistics are computed dynamically to tailor the recommendations.
    Otherwise, general best-practice recommendations for customer churn
    management are displayed.

    Parameters
    ----------
    prediction_results : pandas.DataFrame | None, default=None
        Batch prediction results containing prediction labels, probabilities,
        and optional risk categories.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If an unexpected error occurs while rendering recommendations.
    """
    logger.info("Rendering business recommendations.")

    try:
        st.header("💼 Executive Business Recommendations")

        churn_count = 0
        retained_count = 0
        total_customers = 0
        churn_rate = 0.0
        average_probability: float | None = None

        if prediction_results is not None and not prediction_results.empty:

            total_customers = len(prediction_results)

            prediction_column = next(
                (
                    column
                    for column in (
                        "Prediction Label",
                        "Prediction",
                        "Predicted Label",
                        "Predicted_Class",
                    )
                    if column in prediction_results.columns
                ),
                None,
            )

            probability_column = next(
                (
                    column
                    for column in (
                        "Churn Probability",
                        "Probability",
                        "Prediction Probability",
                    )
                    if column in prediction_results.columns
                ),
                None,
            )

            if prediction_column is not None:
                prediction_series = (
                    prediction_results[prediction_column]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

                churn_count = int(
                    prediction_series.isin(
                        {
                            "1",
                            "true",
                            "yes",
                            "churn",
                        }
                    ).sum()
                )

                retained_count = total_customers - churn_count

                churn_rate = (
                    churn_count / total_customers * 100
                    if total_customers > 0
                    else 0.0
                )

            if probability_column is not None:
                average_probability = float(
                    prediction_results[
                        probability_column
                    ].mean()
                )

        metric_columns = st.columns(4)

        with metric_columns[0]:
            st.metric(
                "Customers Analysed",
                f"{total_customers:,}",
            )

        with metric_columns[1]:
            st.metric(
                "Predicted Churn",
                f"{churn_count:,}",
            )

        with metric_columns[2]:
            st.metric(
                "Retention Rate",
                (
                    f"{100 - churn_rate:.2f}%"
                    if total_customers > 0
                    else "N/A"
                ),
            )

        with metric_columns[3]:
            st.metric(
                "Average Risk",
                (
                    f"{average_probability:.2%}"
                    if average_probability is not None
                    else "N/A"
                ),
            )

        st.subheader("Key Recommendations")

        recommendations = [
            {
                "Priority": "High",
                "Area": "Customer Retention",
                "Recommendation": (
                    "Prioritize customers predicted to churn using targeted "
                    "retention campaigns and proactive relationship management."
                ),
            },
            {
                "Priority": "High",
                "Area": "Customer Engagement",
                "Recommendation": (
                    "Provide personalized offers, loyalty rewards, and "
                    "financial incentives to high-risk customers."
                ),
            },
            {
                "Priority": "Medium",
                "Area": "Customer Experience",
                "Recommendation": (
                    "Investigate customer complaints, service quality, and "
                    "support interactions to reduce dissatisfaction."
                ),
            },
            {
                "Priority": "Medium",
                "Area": "Portfolio Management",
                "Recommendation": (
                    "Use prediction probabilities to prioritize intervention "
                    "resources for customers with the highest expected risk."
                ),
            },
            {
                "Priority": "Medium",
                "Area": "Marketing",
                "Recommendation": (
                    "Develop personalized marketing campaigns for different "
                    "customer segments based on predicted churn risk."
                ),
            },
            {
                "Priority": "Low",
                "Area": "Model Monitoring",
                "Recommendation": (
                    "Continuously monitor model performance and retrain using "
                    "new customer behaviour data."
                ),
            },
        ]

        recommendation_df = pd.DataFrame(recommendations)

        st.dataframe(
            recommendation_df,
            hide_index=True,
            use_container_width=True,
        )

        if (
            prediction_results is not None
            and not prediction_results.empty
            and "Risk Category" in prediction_results.columns
        ):
            risk_summary = (
                prediction_results["Risk Category"]
                .value_counts()
                .rename_axis("Risk Category")
                .reset_index(name="Customers")
            )

            chart = px.bar(
                risk_summary,
                x="Risk Category",
                y="Customers",
                text="Customers",
                title="Customer Risk Segmentation",
            )

            chart.update_traces(
                textposition="outside",
            )

            st.plotly_chart(
                chart,
                use_container_width=True,
            )

        if total_customers > 0:

            if churn_rate >= 40:
                st.error(
                    "The predicted churn rate is high. Immediate retention "
                    "initiatives should be prioritized."
                )
            elif churn_rate >= 20:
                st.warning(
                    "A moderate proportion of customers are predicted to "
                    "churn. Targeted intervention campaigns are recommended."
                )
            else:
                st.success(
                    "Predicted churn is relatively low. Continue monitoring "
                    "customer engagement while maintaining proactive "
                    "retention strategies."
                )

        else:
            st.info(
                "Run customer predictions to receive personalized business "
                "recommendations."
            )

        with st.expander(
            "📖 Executive Guidance",
            expanded=True,
        ):
            st.markdown(
                """
### Strategic Actions

- Focus retention efforts on customers with the highest predicted churn probability.
- Combine model predictions with business knowledge before making operational decisions.
- Prioritize personalized communication instead of generic campaigns.
- Monitor customer satisfaction, complaints, and product usage continuously.
- Retrain the prediction model periodically using newly available customer data.
- Measure campaign effectiveness using conversion, retention, and lifetime value metrics.

### Operational Recommendations

- Establish automated churn monitoring dashboards.
- Schedule periodic batch predictions.
- Integrate predictions into CRM workflows.
- Trigger alerts for high-risk customers.
- Evaluate retention campaign ROI regularly.
"""
            )

        logger.info(
            "Business recommendations rendered successfully."
        )

    except Exception as exc:
        logger.exception(
            "Failed to render business recommendations."
        )
        raise RuntimeError(
            "Unable to render business recommendations."
        ) from exc

def render_footer() -> None:
    """
    Render the dashboard footer.

    Displays project metadata, author information, internship details,
    technology stack, and copyright attribution in a professional,
    responsive footer suitable for the Streamlit application.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the footer cannot be rendered.
    """
    logger.info("Rendering dashboard footer.")

    try:
        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.caption("👨‍💻 **Author**")
            st.markdown(f"**{PROJECT_AUTHOR}**")

        with col2:
            st.caption("📂 **Project**")
            st.markdown(f"**{PROJECT_NAME}**")
            st.caption(f"Version: **{PROJECT_VERSION}**")

        with col3:
            st.caption("🎓 **Internship**")
            internship_name = globals().get(
                "INTERNSHIP_NAME",
                "Machine Learning Internship",
            )
            internship_org = globals().get(
                "INTERNSHIP_ORGANIZATION",
                "",
            )

            st.markdown(f"**{internship_name}**")

            if internship_org:
                st.caption(internship_org)

        st.markdown("")

        tech_stack = globals().get(
            "TECHNOLOGY_STACK",
            [
                "Python 3.12+",
                "Streamlit",
                "Scikit-learn",
                "Pandas",
                "NumPy",
                "Plotly",
            ],
        )

        if isinstance(tech_stack, (list, tuple, set)):
            tech_text = " • ".join(map(str, tech_stack))
        else:
            tech_text = str(tech_stack)

        st.caption(f"**Technology Stack:** {tech_text}")

        current_year = datetime.now().year

        st.markdown(
            f"""
<div style="text-align:center;
            padding-top:20px;
            padding-bottom:10px;
            color:#6c757d;
            font-size:0.90rem;">

<b>{PROJECT_NAME}</b><br>

Version {PROJECT_VERSION}<br><br>

Predictive Modeling and Risk Scoring for Bank Customer Churn Dashboard
built using Streamlit and Machine Learning.<br><br>

© {current_year} {PROJECT_AUTHOR}. All Rights Reserved.

</div>
""",
            unsafe_allow_html=True,
        )

        logger.info("Dashboard footer rendered successfully.")

    except Exception as exc:
        logger.exception(
            "Failed to render dashboard footer."
        )
        raise RuntimeError(
            "Unable to render dashboard footer."
        ) from exc

def main() -> None:
    """
    Execute the Customer Churn Prediction page.

    This function coordinates the complete Streamlit workflow for customer
    churn prediction, including:

    - Loading model artifacts.
    - Rendering the page header and sidebar.
    - Performing single customer prediction.
    - Displaying prediction results.
    - Rendering probability visualizations.
    - Performing batch prediction.
    - Displaying prediction summaries.
    - Rendering business recommendations.
    - Displaying the application footer.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If an unexpected error occurs during page execution.
    """
    logger.info("Starting Customer Churn Prediction page.")

    try:
        # Initialize required artifacts
        load_model_artifacts()

        # Header
        render_page_header()

        # Sidebar
        render_sidebar()

        st.divider()

        # -----------------------------
        # Single Customer Prediction
        # -----------------------------
        prediction, probability, input_dataframe = (
            render_single_prediction_form()
        )

        prediction_result: dict[str, Any] | None = None

        if input_dataframe is not None:
            prediction_result = predict_single_customer(
                input_dataframe=input_dataframe
            )

            # Preserve probability from form if available
            if (
                probability is not None
                and prediction_result.get(
                    "prediction_probability"
                )
                is None
            ):
                prediction_result[
                    "prediction_probability"
                ] = probability

        if prediction_result is not None:
            st.divider()

            render_prediction_result(
                prediction_result
            )

            st.divider()

            render_probability_visualization(
                prediction_result
            )

        st.divider()

        # -----------------------------
        # Batch Prediction
        # -----------------------------
        render_batch_prediction()

        batch_results = st.session_state.get(
            "batch_prediction_results"
        )

        if isinstance(batch_results, pd.DataFrame):

            st.divider()

            render_prediction_summary(
                batch_results
            )

            st.divider()

            render_business_recommendations(
                batch_results
            )

        else:
            st.divider()
            render_business_recommendations()

        st.divider()

        render_footer()

        logger.info(
            "Customer Churn Prediction page rendered successfully."
        )

    except Exception as exc:
        logger.exception(
            "Unexpected error while rendering Customer Churn Prediction page."
        )

        st.error(
            "An unexpected error occurred while running the Customer Churn Prediction page."
        )

        st.exception(exc)

        raise RuntimeError(
            "Customer Churn Prediction page execution failed."
        ) from exc


if __name__ == "__main__":
    main()