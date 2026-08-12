# 🏦 Bank Customer Churn Prediction

End-to-end Machine Learning project for predicting bank customer churn and assigning customer risk levels.

## 📌 Overview

This project uses customer demographic, financial, and behavioral data to predict whether a customer is likely to leave the bank.

The system includes:

- Data preprocessing and validation
- Exploratory Data Analysis
- Feature engineering
- Multiple ML classification models
- Hyperparameter tuning
- Model evaluation
- SHAP-based Explainable AI
- Single and batch customer prediction
- Risk scoring
- Interactive Streamlit dashboard
- Persisted ML models and preprocessing pipeline

## 🤖 Machine Learning

Models evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The best-performing model is saved as:

`models/best_model.pkl`

## 📊 Dataset

**Domain:** Banking  
**Problem:** Binary Classification  
**Target:** `Exited`

Important features include:

`CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, and `EstimatedSalary`.

Identifier columns such as `CustomerId` and `Surname` are excluded from modeling.

## 🧠 Explainable AI

SHAP is used to understand the major factors influencing churn predictions.

Generated outputs include:

- SHAP Summary Plot
- SHAP Feature Importance
- Feature Importance Analysis
- ROC Curve
- Precision-Recall Curve
- Confusion Matrix

## 🌐 Streamlit Dashboard

The application provides:

- Executive Dashboard
- Dataset Analysis
- Model Performance
- Customer Churn Prediction
- Project Information

Users can enter individual customer details or process multiple customers and receive:

- Churn prediction
- Churn probability
- Confidence score
- Risk level

## 📁 Project Structure

```text
bank_customer_churn_prediction/
├── app.py
├── assets/
├── config/
├── data/
├── models/
├── notebooks/
├── pages/
├── predictions/
├── reports/
├── src/
├── utils/
├── visuals/
├── requirements.txt
├── README.md
└── LICENSE
````

## ⚙️ Installation

```bash
git clone https://github.com/your-username/bank_customer_churn_prediction.git
cd bank_customer_churn_prediction
pip install -r requirements.txt
```

## ▶️ Run Dashboard

```bash
streamlit run app.py
```

The application will be available at:

`http://localhost:8501`

## 🔄 ML Workflow

```text
Raw Dataset
     ↓
Data Validation
     ↓
Preprocessing
     ↓
EDA
     ↓
Feature Engineering
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Best Model Selection
     ↓
SHAP Explainability
     ↓
Model Persistence
     ↓
Prediction Engine
     ↓
Streamlit Dashboard
```

## 💼 Business Impact

The system helps banks:

* Identify high-risk customers
* Prioritize retention campaigns
* Understand churn drivers
* Support data-driven decisions
* Reduce potential customer loss

## 🛠️ Technology Stack

Python • Pandas • NumPy • Scikit-learn • SHAP • Matplotlib • Plotly • Streamlit • Joblib

## 👨‍💻 Author

**Shravan Pandey**
Data Science & Machine Learning Intern

## 📜 License

MIT License
'@ | Set-Content -Encoding UTF8 README.md

git add -A
git add -f data models predictions reports visuals assets/logo.png
git status
git commit -m "Finalize bank customer churn prediction project"
git push origin main

