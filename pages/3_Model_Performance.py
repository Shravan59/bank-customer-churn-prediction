"""
pages/3_Model_Performance.py

Compact Model Performance Dashboard
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS = PROJECT_ROOT / "models"
DATA = PROJECT_ROOT / "data" / "processed"

MODEL_PATH = MODELS / "best_model.pkl"
PIPELINE_PATH = MODELS / "preprocessing_pipeline.pkl"

X_TEST = DATA / "X_test_processed.csv"
Y_TEST = DATA / "y_test.csv"

# -------------------------------------------------------
# PAGE
# -------------------------------------------------------

st.set_page_config(
    page_title="Model Performance",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Model Performance Dashboard")

# -------------------------------------------------------
# LOAD FILES
# -------------------------------------------------------

try:

    model = joblib.load(MODEL_PATH)

    X = pd.read_csv(X_TEST)

    y = pd.read_csv(Y_TEST).iloc[:, 0]

except Exception as e:

    st.error(e)

    st.stop()

# -------------------------------------------------------
# PREDICTIONS
# -------------------------------------------------------

y_pred = model.predict(X)

if hasattr(model, "predict_proba"):

    y_prob = model.predict_proba(X)[:, 1]

else:

    y_prob = None

# -------------------------------------------------------
# METRICS
# -------------------------------------------------------

accuracy = accuracy_score(y, y_pred)

precision = precision_score(y, y_pred)

recall = recall_score(y, y_pred)

f1 = f1_score(y, y_pred)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy:.2%}")

col2.metric("Precision", f"{precision:.2%}")

col3.metric("Recall", f"{recall:.2%}")

col4.metric("F1 Score", f"{f1:.2%}")

st.divider()

# -------------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------------

st.subheader("Confusion Matrix")

cm = confusion_matrix(y, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual No", "Actual Yes"],
    columns=["Pred No", "Pred Yes"],
)

fig = px.imshow(
    cm_df,
    text_auto=True,
    color_continuous_scale="Blues",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.dataframe(
    cm_df,
    use_container_width=True,
)

st.divider()
# -------------------------------------------------------
# CLASSIFICATION REPORT
# -------------------------------------------------------

st.subheader("Classification Report")

report = classification_report(
    y,
    y_pred,
    output_dict=True,
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df,
    use_container_width=True,
)

st.divider()

# -------------------------------------------------------
# PREDICTION DISTRIBUTION
# -------------------------------------------------------

st.subheader("Prediction Distribution")

pred_df = pd.DataFrame(
    {
        "Prediction": y_pred
    }
)

pred_df["Prediction"] = pred_df["Prediction"].replace(
    {
        0: "No Churn",
        1: "Churn",
    }
)

fig = px.histogram(
    pred_df,
    x="Prediction",
    color="Prediction",
    text_auto=True,
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.divider()

# -------------------------------------------------------
# BASIC SUMMARY
# -------------------------------------------------------

st.subheader("Model Summary")

summary = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "Total Test Samples",
        ],
        "Value": [
            round(accuracy,4),
            round(precision,4),
            round(recall,4),
            round(f1,4),
            len(y),
        ],
    }
)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
)

st.divider()
# -------------------------------------------------------
# ROC CURVE
# -------------------------------------------------------

if y_prob is not None:

    st.subheader("ROC Curve")

    fpr, tpr, _ = roc_curve(
        y,
        y_prob,
    )

    auc = roc_auc_score(
        y,
        y_prob,
    )

    roc_df = pd.DataFrame(
        {
            "False Positive Rate": fpr,
            "True Positive Rate": tpr,
        }
    )

    fig = px.line(
        roc_df,
        x="False Positive Rate",
        y="True Positive Rate",
        title=f"ROC Curve (AUC = {auc:.4f})",
    )

    fig.add_scatter(
        x=[0,1],
        y=[0,1],
        mode="lines",
        name="Random",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.metric(
        "ROC AUC",
        f"{auc:.4f}",
    )

    st.divider()

# -------------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------------

st.subheader("Feature Importance")

importance = None

if hasattr(model,"feature_importances_"):

    importance = model.feature_importances_

elif hasattr(model,"coef_"):

    importance = np.abs(
        model.coef_[0]
    )

if importance is not None:

    feature_names = list(X.columns)

    imp_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance,
        }
    )

    imp_df = imp_df.sort_values(
        "Importance",
        ascending=False,
    )

    top = imp_df.head(15)

    fig = px.bar(
        top,
        x="Importance",
        y="Feature",
        orientation="h",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.dataframe(
        top,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Feature Importance not available for this model."
    )

st.divider()

# -------------------------------------------------------
# BUSINESS INSIGHTS
# -------------------------------------------------------

st.subheader("Business Insights")

st.success(
    f"""
Accuracy : {accuracy:.2%}

Precision : {precision:.2%}

Recall : {recall:.2%}

F1 Score : {f1:.2%}

This model can be used to identify customers
who are likely to churn so the bank can take
retention actions.
"""
)

st.divider()
# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------

st.divider()

st.markdown(
    """
---
### 📌 Project Information

**Project:** Predictive Modeling and Risk Scoring for Bank Customer Churn

**Technology:** Python • Scikit-Learn • Streamlit • Plotly

**Internship Project**
"""
)

st.success("✅ Model Performance Dashboard Loaded Successfully")
