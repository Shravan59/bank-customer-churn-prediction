"""
model_training.py
Simple Model Training
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"

X_train = pd.read_csv(
    PROCESSED / "X_train_engineered.csv"
)

X_test = pd.read_csv(
    PROCESSED / "X_test_engineered.csv"
)

y_train = pd.read_csv(
    PROCESSED / "y_train.csv"
)["Exited"]

y_test = pd.read_csv(
    PROCESSED / "y_test.csv"
)["Exited"]

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(random_state=42),

    "Gradient Boosting":
        GradientBoostingClassifier(random_state=42),

}

results = []

best_auc = 0
best_model = None

print("Training Models...\n")

for name, model in models.items():

    model.fit(
        X_train,
        y_train,
    )

    pred = model.predict(X_test)

    prob = model.predict_proba(
        X_test
    )[:,1]

    accuracy = accuracy_score(
        y_test,
        pred,
    )

    precision = precision_score(
        y_test,
        pred,
    )

    recall = recall_score(
        y_test,
        pred,
    )

    f1 = f1_score(
        y_test,
        pred,
    )

    auc = roc_auc_score(
        y_test,
        prob,
    )

    results.append({

        "Model":name,
        "Accuracy":accuracy,
        "Precision":precision,
        "Recall":recall,
        "F1":f1,
        "ROC_AUC":auc,

    })

    print(
        f"{name}  ROC_AUC={auc:.4f}"
    )

    if auc > best_auc:

        best_auc = auc
        best_model = model

results = pd.DataFrame(results)

results.to_csv(
    MODELS/"model_comparison.csv",
    index=False,
)

results.to_csv(
    MODELS/"training_results.csv",
    index=False,
)

joblib.dump(
    best_model,
    MODELS/"best_model.pkl",
)

print("\nDone.\n")

print(results)

print("\nBest Model Saved Successfully.")