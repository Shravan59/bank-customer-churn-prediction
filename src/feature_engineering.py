"""
feature_engineering.py
Simple Feature Engineering
"""

from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED = PROJECT_ROOT / "data" / "processed"

X_TRAIN = PROCESSED / "X_train_processed.csv"
X_TEST = PROCESSED / "X_test_processed.csv"

OUT_TRAIN = PROCESSED / "X_train_engineered.csv"
OUT_TEST = PROCESSED / "X_test_engineered.csv"

FEATURE_FILE = PROJECT_ROOT / "models" / "selected_features.pkl"


def add_features(df):

    df = df.copy()

    if {"Balance", "EstimatedSalary"}.issubset(df.columns):
        df["BalanceSalaryRatio"] = (
            df["Balance"] /
            (df["EstimatedSalary"] + 1)
        )

    if {"NumOfProducts", "Tenure"}.issubset(df.columns):
        df["ProductsPerTenure"] = (
            df["NumOfProducts"] /
            (df["Tenure"] + 1)
        )

    if {"CreditScore", "Age"}.issubset(df.columns):
        df["CreditScoreAgeRatio"] = (
            df["CreditScore"] /
            (df["Age"] + 1)
        )

    return df


def main():

    print("Loading processed data...")

    X_train = pd.read_csv(X_TRAIN)
    X_test = pd.read_csv(X_TEST)

    X_train = add_features(X_train)
    X_test = add_features(X_test)

    X_train.to_csv(
        OUT_TRAIN,
        index=False,
    )

    X_test.to_csv(
        OUT_TEST,
        index=False,
    )

    joblib.dump(
        list(X_train.columns),
        FEATURE_FILE,
    )

    print("\nDone.")
    print("Files Created:\n")
    print("X_train_engineered.csv")
    print("X_test_engineered.csv")
    print("selected_features.pkl")


if __name__ == "__main__":
    main()