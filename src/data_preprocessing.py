"""
data_preprocessing.py

Compact preprocessing pipeline for
Bank Customer Churn Prediction.
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "bank_customer_churn.csv"

PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"

PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)


TARGET = "Exited"


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():

    print("Loading dataset...")

    df = pd.read_csv(RAW_DATA)

    # Save cleaned dataset
    df.to_csv(
        PROCESSED / "cleaned_data.csv",
        index=False,
    )

    # Remove identifier columns if present
    drop_cols = [
        c
        for c in ["CustomerId", "Surname"]
        if c in df.columns
    ]

    df = df.drop(columns=drop_cols)

    # Remove Year if constant
    if "Year" in df.columns:
        if df["Year"].nunique() <= 1:
            df = df.drop(columns=["Year"])

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    categorical = [
        c
        for c in ["Geography", "Gender"]
        if c in X.columns
    ]

    numeric = [
        c
        for c in X.columns
        if c not in categorical
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
        ]
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor)
        ]
    )

    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)

    feature_names = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    X_train = pd.DataFrame(
        X_train,
        columns=feature_names,
    )

    X_test = pd.DataFrame(
        X_test,
        columns=feature_names,
    )

    X_train.to_csv(
        PROCESSED / "X_train_processed.csv",
        index=False,
    )

    X_test.to_csv(
        PROCESSED / "X_test_processed.csv",
        index=False,
    )

    y_train.to_frame(name=TARGET).to_csv(
        PROCESSED / "y_train.csv",
        index=False,
    )

    y_test.to_frame(name=TARGET).to_csv(
        PROCESSED / "y_test.csv",
        index=False,
    )

    joblib.dump(
        pipeline,
        MODELS / "preprocessing_pipeline.pkl",
    )

    joblib.dump(
        list(feature_names),
        MODELS / "feature_columns.pkl",
    )

    print("\nDone.")
    print("Files created:\n")
    print("cleaned_data.csv")
    print("X_train_processed.csv")
    print("X_test_processed.csv")
    print("y_train.csv")
    print("y_test.csv")
    print("preprocessing_pipeline.pkl")
    print("feature_columns.pkl")


if __name__ == "__main__":
    main()