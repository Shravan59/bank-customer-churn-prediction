# 🏦 Predictive Modeling and Risk Scoring for Bank Customer Churn

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-blue?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly)
![SHAP](https://img.shields.io/badge/Explainable-AI-purple)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

---

# 📊 Project Banner

> **End-to-End Machine Learning Solution for Predicting Customer Churn in the Banking Industry**

A production-ready machine learning application that enables banks to identify customers at high risk of leaving the organization through predictive analytics, explainable AI, and an interactive Streamlit dashboard.

---

# 📖 Project Description

Customer churn is one of the most significant challenges faced by modern financial institutions. Losing existing customers directly impacts revenue, profitability, and long-term business growth.

This project develops a complete machine learning pipeline capable of predicting customer churn using historical customer information. The solution follows an industry-standard workflow beginning with data preprocessing and ending with deployment through a professional Streamlit web application.

The repository demonstrates the complete lifecycle of a real-world supervised machine learning project including:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training
- Model Evaluation
- Explainable AI using SHAP
- Model Persistence
- Customer Churn Prediction
- Interactive Streamlit Dashboard

The project has been designed to be:

- Internship Submission Ready
- GitHub Portfolio Ready
- ATS Friendly
- Modular
- Production Ready
- Easy to Maintain
- Easy to Extend

---

# 💼 Business Problem

Banks invest substantial resources in acquiring new customers. However, retaining existing customers is significantly more cost-effective than acquiring new ones.

Without an intelligent prediction system, organizations often identify customer churn only after customers have already left.

### Business Challenges

- Increasing customer attrition
- Revenue loss due to churn
- High acquisition costs
- Inefficient marketing campaigns
- Poor customer retention strategies
- Difficulty identifying high-risk customers

---

## Business Need

Develop a predictive system capable of identifying customers likely to leave the bank before churn occurs, enabling proactive intervention and personalized retention strategies.

---

# 🎯 Project Objectives

## Primary Objectives

- Predict whether a customer will churn.
- Compare multiple machine learning algorithms.
- Select the best-performing model.
- Explain predictions using Explainable AI.
- Deploy an interactive Streamlit dashboard.

---

## Secondary Objectives

- Understand customer behavior through EDA.
- Identify major churn drivers.
- Build reusable preprocessing pipelines.
- Produce professional visualizations.
- Enable batch customer predictions.
- Export prediction reports.

---

# 🏗 Project Architecture

```text
                        +----------------------+
                        |   Raw CSV Dataset    |
                        +----------+-----------+
                                   |
                                   ▼
                    Data Preprocessing Module
                                   |
                                   ▼
                Exploratory Data Analysis (EDA)
                                   |
                                   ▼
                    Feature Engineering Module
                                   |
                                   ▼
                     Machine Learning Training
                                   |
                                   ▼
                    Model Evaluation & Metrics
                                   |
                                   ▼
                      Explainable AI (SHAP)
                                   |
                                   ▼
                     Model Persistence Layer
                                   |
                                   ▼
                       Prediction Engine
                                   |
                                   ▼
                  Interactive Streamlit Dashboard
```

---

# 🔄 End-to-End Workflow

```text
Bank Customer Dataset
          │
          ▼
Data Validation
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
Train-Test Split
          │
          ▼
Train Multiple ML Models
          │
          ▼
Hyperparameter Tuning
          │
          ▼
Model Evaluation
          │
          ▼
Best Model Selection
          │
          ▼
SHAP Explainability
          │
          ▼
Save Artifacts
          │
          ▼
Prediction Module
          │
          ▼
Streamlit Dashboard
```

---

# 🎯 Key Project Highlights

- ✅ End-to-End Machine Learning Pipeline
- ✅ Production-Ready Python Code
- ✅ Modular Repository Structure
- ✅ Automated Data Processing
- ✅ Multiple Classification Models
- ✅ Hyperparameter Optimization
- ✅ Explainable AI (SHAP)
- ✅ Model Persistence
- ✅ Batch & Single Predictions
- ✅ Interactive Streamlit Dashboard
- ✅ High-Quality Visualizations
- ✅ GitHub Portfolio Ready
- ✅ Internship Submission Ready

---

# 📌 Project Scope

This repository demonstrates the complete implementation of a customer churn prediction system using classical supervised machine learning techniques.

The project focuses on:

- Customer analytics
- Business intelligence
- Predictive modeling
- Explainable AI
- Interactive reporting
- Decision support

while maintaining a lightweight architecture suitable for an internship-scale project (4–5 days of development).

---

# 📁 Project Folder Structure

```text
bank_customer_churn_prediction/
│
├── app.py
│
├── config/
│   └── config.py
│
├── assets/
│   ├── style.css
│   ├── logo.png
│   └── icons/
│
├── data/
│   ├── raw/
│   │   └── Bank Customer Churn Prediction.csv
│   │
│   ├── processed/
│   │   ├── cleaned_data.csv
│   │   ├── X_train_processed.csv
│   │   ├── X_test_processed.csv
│   │   ├── engineered_train.csv
│   │   ├── engineered_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   │
│   └── sample/
│       └── sample_input.csv
│
├── models/
│   ├── best_model.pkl
│   ├── preprocessing_pipeline.pkl
│   ├── feature_columns.pkl
│   ├── model_metrics.json
│   ├── model_metadata.json
│   ├── model_comparison.csv
│   └── training_results.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── pages/
│   ├── Executive_Dashboard.py
│   ├── Dataset_Analysis.py
│   ├── Model_Performance.py
│   ├── Customer_Churn_Prediction.py
│   └── About_Project.py
│
├── reports/
│
├── src/
│   ├── data_preprocessing.py
│   ├── exploratory_data_analysis.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── model_explainability.py
│   ├── model_visualization.py
│   ├── model_persistence.py
│   └── prediction.py
│
├── utils/
│   ├── helpers.py
│   └── visualization.py
│
├── visuals/
│   ├── eda/
│   ├── model/
│   └── dashboard/
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# 📊 Dataset Information

| Property | Value |
|------------|------|
| Dataset | Bank Customer Churn |
| Domain | Banking |
| Problem Type | Binary Classification |
| Target Variable | **Exited** |
| Dataset Format | CSV |
| Learning Type | Supervised Learning |

The dataset contains customer demographic, financial, and behavioral information used to predict whether a customer is likely to leave the bank.

---

# 📋 Feature Description

| Feature | Description | Type |
|----------|-------------|------|
| CreditScore | Customer credit score | Numerical |
| Geography | Country of residence | Categorical |
| Gender | Customer gender | Categorical |
| Age | Customer age | Numerical |
| Tenure | Years with the bank | Numerical |
| Balance | Current account balance | Numerical |
| NumOfProducts | Number of bank products | Numerical |
| HasCrCard | Owns a credit card (0/1) | Binary |
| IsActiveMember | Active member indicator | Binary |
| EstimatedSalary | Estimated annual salary | Numerical |
| Year | Record year | Numerical |
| Exited | Customer churn label | Target |

> **Removed Identifier Columns**
>
> - CustomerId
> - Surname

These columns are removed during preprocessing because they do not contribute to predictive performance.

---

# 🛠 Technology Stack

| Category | Technologies |
|-----------|--------------|
| Programming Language | Python 3.12 |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Explainable AI | SHAP |
| Visualization | Matplotlib, Plotly |
| Dashboard | Streamlit |
| Model Persistence | Joblib |
| Configuration | pathlib, JSON |
| Documentation | Markdown |

---

# ⚙️ Data Preprocessing Summary

The preprocessing pipeline is designed to be reusable for both model training and future inference.

### Steps Performed

- Dataset loading
- Data validation
- Removal of identifier columns
- Feature separation
- Train-test split using stratification
- One-Hot Encoding for categorical features
- Standardization of numerical features
- Pipeline persistence
- Processed dataset export

### Encoding Strategy

| Feature | Method |
|----------|--------|
| Geography | OneHotEncoder |
| Gender | OneHotEncoder (Binary Drop) |

### Scaling

The following numerical features are standardized using **StandardScaler**:

- CreditScore
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary
- Year

---

# 📈 Exploratory Data Analysis (EDA)

The EDA module provides a detailed understanding of customer behavior and churn patterns.

### Analyses Performed

- Dataset profiling
- Missing value analysis
- Duplicate record analysis
- Numerical feature distributions
- Categorical feature distributions
- Target variable analysis
- Correlation analysis
- Outlier analysis
- Bivariate analysis
- Business insight generation

### Generated Visualizations

- Target distribution
- Histograms
- Boxplots
- Count plots
- Correlation heatmap
- Churn vs Geography
- Churn vs Gender
- Churn vs Age
- Churn vs Balance
- Churn vs Credit Score
- Churn vs Number of Products
- Churn vs Active Member
- Churn vs Credit Card

All visualizations are automatically saved under:

```text
visuals/
└── eda/
```

---

# 🧩 Feature Engineering Summary

Feature engineering builds upon the cleaned and preprocessed data while avoiding redundant transformations.

### Implemented Operations

- Feature validation
- Data consistency checks
- Evidence-based derived feature creation (where applicable)
- Optional feature selection
- Engineered dataset persistence

### Engineering Goals

- Improve predictive performance
- Preserve data integrity
- Maintain compatibility with the preprocessing pipeline
- Prepare optimized datasets for model training

### Output Artifacts

```text
data/
└── processed/
    ├── engineered_train.csv
    └── engineered_test.csv
```

---

# 📌 Data Pipeline Overview

```text
Raw Dataset
      │
      ▼
Validation
      │
      ▼
Preprocessing
      │
      ▼
EDA
      │
      ▼
Feature Engineering
      │
      ▼
Engineered Dataset
      │
      ▼
Model Training
```

---

# ✅ Key Deliverables of This Stage

- Centralized preprocessing pipeline
- Reusable feature engineering workflow
- Comprehensive EDA reports
- High-quality visualizations
- Production-ready processed datasets
- Modular project architecture

---

# 🤖 Machine Learning Pipeline

The project follows a modular machine learning workflow where each stage is implemented as an independent production-ready Python module.

```text
Processed Dataset
        │
        ▼
Feature Engineering
        │
        ▼
Model Training
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Model Comparison
        │
        ▼
Best Model Selection
        │
        ▼
Model Evaluation
        │
        ▼
Explainable AI (SHAP)
        │
        ▼
Model Persistence
        │
        ▼
Prediction Engine
        │
        ▼
Streamlit Dashboard
```

---

# 🧠 Models Implemented

The following supervised machine learning algorithms were implemented and compared.

| Model | Purpose |
|--------|---------|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Interpretable non-linear classifier |
| Random Forest | Ensemble tree-based classifier |
| Gradient Boosting | Boosted ensemble classifier |

---

## Hyperparameter Tuning

Lightweight hyperparameter optimization is performed for:

- Decision Tree
- Random Forest

using **GridSearchCV** with:

- 5-fold Cross Validation
- ROC-AUC scoring

The best estimator is automatically selected based on:

1. ROC-AUC Score
2. F1 Score

---

# 📊 Model Evaluation

Every trained model is evaluated using multiple classification metrics.

## Evaluation Metrics

| Metric | Purpose |
|----------|---------|
| Accuracy | Overall prediction accuracy |
| Precision | Correct positive predictions |
| Recall | Ability to detect churn |
| F1 Score | Precision–Recall balance |
| ROC-AUC | Overall ranking performance |

---

## Evaluation Reports

The evaluation module automatically generates:

- Classification Report
- Confusion Matrix
- ROC Curve
- Precision–Recall Curve
- Model Comparison Table
- Evaluation Metrics Summary

Generated reports are saved under the project artifacts directory.

---

# 📈 Model Comparison

Each algorithm is compared using identical train-test data.

Example comparison table:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------|-----------|------------|-----------|------------|------------|
| Logistic Regression | ✓ | ✓ | ✓ | ✓ | ✓ |
| Decision Tree | ✓ | ✓ | ✓ | ✓ | ✓ |
| Random Forest | ✓ | ✓ | ✓ | ✓ | ✓ |
| Gradient Boosting | ✓ | ✓ | ✓ | ✓ | ✓ |

The highest ROC-AUC model is automatically persisted as:

```text
models/
└── best_model.pkl
```

---

# 🔍 Explainable AI (SHAP)

Model transparency is achieved using **SHAP (SHapley Additive exPlanations)**.

The explainability module automatically detects the estimator type and chooses the appropriate SHAP explainer.

Supported models:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

---

## Generated Explainability Artifacts

- SHAP Summary Plot
- SHAP Bar Plot
- Global Feature Importance
- Top Feature Ranking
- Business Interpretation Report

These artifacts help explain **why** the model predicts customer churn rather than only **what** it predicts.

---

# 📉 Model Visualizations

The visualization module produces publication-quality figures suitable for reports and presentations.

## Generated Visualizations

### Evaluation

- ROC Curve
- Precision–Recall Curve
- Confusion Matrix

### Explainability

- SHAP Summary Plot
- SHAP Feature Importance
- Global Feature Importance

### Model Comparison

- Performance Comparison Chart
- Metric Comparison Plot

All visualizations are exported at **300 DPI** for high-quality reporting.

Directory:

```text
visuals/
└── model/
```

---

# 🎯 Prediction Workflow

The prediction module performs inference using previously saved artifacts.

## Loaded Artifacts

```text
best_model.pkl

preprocessing_pipeline.pkl

feature_columns.pkl
```

---

## Single Customer Prediction

Input:

```text
One customer record
```

Output:

```json
{
  "prediction": 1,
  "probability": 0.82,
  "risk_level": "High",
  "confidence": 0.82
}
```

---

## Batch Prediction

Input:

```text
Multiple customer records (CSV/DataFrame)
```

Output:

- Original dataset
- Prediction
- Probability
- Confidence
- Risk Level

---

## Prediction Summary

Automatically generated statistics include:

- Number of customers
- Predicted churn customers
- Predicted retained customers
- Average churn probability
- High-risk customers
- Medium-risk customers
- Low-risk customers

Prediction results can be exported directly to CSV.

---

# 🌐 Streamlit Dashboard

The project includes a professional multi-page Streamlit dashboard.

## Dashboard Features

### 🏠 Home

- Project overview
- Business problem
- Objectives
- Workflow
- Technologies
- KPIs

---

### 📊 Dataset Analysis

- Dataset summary
- Feature distributions
- Correlation analysis
- EDA visualizations
- Business insights

---

### 🤖 Model Performance

- Evaluation metrics
- ROC Curve
- Precision–Recall Curve
- Confusion Matrix
- Feature Importance
- SHAP Analysis
- Model Comparison

---

### 🔮 Customer Churn Prediction

- Single customer prediction
- Batch prediction
- Risk categorization
- Probability estimation
- Prediction export

---

### ℹ️ About Project

- Repository information
- Project architecture
- Workflow
- Technologies
- Internship details

---

# 📂 Generated Project Outputs

The complete pipeline automatically generates the following artifacts.

## Processed Data

```text
data/processed/
```

Contains:

- Cleaned dataset
- Processed training dataset
- Processed testing dataset
- Engineered datasets

---

## Models

```text
models/
```

Contains:

- Best trained model
- Preprocessing pipeline
- Feature columns
- Metrics
- Metadata
- Training results

---

## Visualizations

```text
visuals/
├── eda/
├── model/
└── dashboard/
```

Contains:

- EDA plots
- Evaluation plots
- Explainability plots
- Dashboard assets

---

## Prediction Results

```text
predictions/
```

Contains exported prediction CSV files.

---

# 🏆 Key Technical Highlights

- Production-ready architecture
- Modular Python implementation
- Reusable preprocessing pipeline
- Multiple ML algorithms
- Hyperparameter tuning
- Explainable AI
- Model persistence
- Batch inference
- Interactive dashboard
- Clean documentation
- Internship-ready codebase
- GitHub portfolio quality

---
# 🚀 Installation Guide

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/bank_customer_churn_prediction.git

cd bank_customer_churn_prediction
```

---

# 🐍 Virtual Environment Setup

## Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

## Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

# 📦 Install Project Dependencies

Install all required libraries using:

```bash
pip install -r requirements.txt
```

---

# ▶ Running Individual Modules

The project follows a modular architecture, allowing each stage of the machine learning pipeline to be executed independently.

## Data Preprocessing

```bash
python src/data_preprocessing.py
```

---

## Exploratory Data Analysis

```bash
python src/exploratory_data_analysis.py
```

---

## Feature Engineering

```bash
python src/feature_engineering.py
```

---

## Model Training

```bash
python src/model_training.py
```

---

## Model Evaluation

```bash
python src/model_evaluation.py
```

---

## Model Explainability

```bash
python src/model_explainability.py
```

---

## Model Visualization

```bash
python src/model_visualization.py
```

---

## Prediction

```bash
python src/prediction.py
```

---

# ⚡ Running the Complete Pipeline

Execute the modules sequentially:

```text
1. data_preprocessing.py

↓

2. exploratory_data_analysis.py

↓

3. feature_engineering.py

↓

4. model_training.py

↓

5. model_evaluation.py

↓

6. model_explainability.py

↓

7. model_visualization.py

↓

8. prediction.py
```

This workflow generates all required artifacts, evaluation reports, explainability outputs, and prediction utilities.

---

# 🌐 Running the Streamlit Dashboard

Launch the interactive dashboard with:

```bash
streamlit run app.py
```

After launching, open the URL displayed in the terminal (typically `http://localhost:8501`) in your browser.

---

# 🔮 Example Prediction Workflow

### Step 1

Prepare a CSV file containing customer records with the required input features.

↓

### Step 2

Run:

```bash
python src/prediction.py
```

↓

### Step 3

The module will:

- Load the trained model
- Load the preprocessing pipeline
- Validate the input data
- Generate predictions
- Estimate churn probabilities
- Assign customer risk levels
- Export prediction results

↓

### Step 4

Prediction results are saved as a CSV file for further analysis or business use.

---

# 📁 Project Outputs

The project automatically generates the following outputs during execution.

## Processed Data

```text
data/
└── processed/
```

Includes:

- Cleaned dataset
- Processed train/test datasets
- Engineered datasets

---

## Trained Models

```text
models/
```

Contains:

- `best_model.pkl`
- `preprocessing_pipeline.pkl`
- `feature_columns.pkl`
- `model_metrics.json`
- `model_metadata.json`
- `model_comparison.csv`
- `training_results.csv`

---

## Visualizations

```text
visuals/
├── eda/
├── model/
└── dashboard/
```

Includes:

- EDA charts
- Model evaluation plots
- SHAP explainability plots
- Feature importance charts
- Dashboard assets

---

## Prediction Results

```text
predictions/
```

Contains exported prediction files with:

- Predicted class
- Churn probability
- Confidence score
- Risk category

---

# 🔮 Future Improvements

Potential enhancements for future versions include:

- Deep Learning–based churn prediction models
- Automated hyperparameter optimization using Optuna
- Real-time prediction APIs using FastAPI
- Docker containerization
- CI/CD integration with GitHub Actions
- Cloud deployment (Azure, AWS, or GCP)
- MLflow experiment tracking
- Automated data validation
- Feature drift monitoring
- Model performance monitoring
- Scheduled model retraining
- Interactive business reporting
- Advanced customer segmentation
- Cost-sensitive learning for churn prevention
- Integration with banking CRM platforms

---

# 💼 Business Impact

This project enables financial institutions to:

- Identify customers at high risk of churn.
- Prioritize retention strategies based on predicted risk.
- Improve customer lifetime value.
- Reduce customer acquisition costs through proactive retention.
- Support data-driven decision-making using interpretable machine learning.
- Increase operational efficiency through automation.
- Provide actionable insights using explainable AI.

---

# 🎓 Learning Outcomes

This project demonstrates practical experience in:

- End-to-end machine learning workflow
- Data preprocessing and validation
- Exploratory data analysis
- Feature engineering
- Classification modeling
- Hyperparameter tuning
- Model evaluation
- Explainable AI (SHAP)
- Model persistence
- Production-ready inference pipelines
- Streamlit application development
- Software engineering best practices
- Modular Python architecture
- Version control and documentation

---

# 📜 License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this project in accordance with the terms of the license.

See the `LICENSE` file for additional details.

---

# 🙏 Acknowledgements

Special thanks to:

- Unified Mentor Internship Program
- Scikit-learn Development Team
- SHAP Contributors
- Streamlit Team
- Plotly Developers
- Pandas Community
- NumPy Community
- Open Source Python Ecosystem

Their tools and resources made this project possible.

---

# 👨‍💻 Author Information

**Project Title**

Predictive Modeling and Risk Scoring for Bank Customer Churn

**Author**

Shravan Pandey

**Role**

Data Science & Machine Learning Intern

**Project Type**

Industry-Oriented End-to-End Machine Learning Project

**Primary Domain**

Banking Analytics

---

# 📬 Contact Information

- **Author:** Shravan Pandey
- **GitHub:** https://github.com/your-username
- **LinkedIn:** https://www.linkedin.com/in/your-profile
- **Email:** your.email@example.com

> Replace the GitHub, LinkedIn, and email placeholders above with your actual profile links before publishing the repository.

---

# ⭐ Support the Project

If you found this project useful:

- ⭐ Star the repository
- 🍴 Fork the project
- 🛠️ Contribute improvements
- 🐛 Report issues
- 💡 Suggest new features

Your support helps improve the project and encourages further open-source contributions.

---

# 📌 Repository Status

| Attribute | Status |
|-----------|--------|
| Project Type | End-to-End Machine Learning |
| Domain | Banking |
| Problem | Customer Churn Prediction |
| Development Status | Completed |
| Code Quality | Production Ready |
| Documentation | Complete |
| Streamlit Dashboard | Included |
| Explainable AI | SHAP |
| License | MIT |

---

## 🎯 Conclusion

This project delivers a complete, production-ready machine learning solution for predicting bank customer churn. It combines robust data preprocessing, comprehensive exploratory data analysis, feature engineering, multiple classification models, rigorous evaluation, explainable AI, and an interactive Streamlit dashboard within a clean, modular architecture.

Designed as an internship-quality project, it demonstrates both machine learning expertise and professional software engineering practices, making it suitable for GitHub portfolios, resume projects, LinkedIn showcases, and technical interviews.

---