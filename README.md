1. Project Overview

Business Problem: Customer churn is costly for telecom providers. Retaining existing customers is cheaper than acquiring new ones.

Goal: Predict which customers are at risk of churn, explain why churn happens, and provide actionable recommendations for reducing churn.

Dataset: Telco Customer Churn (Kaggle)

~7,000 customers, demographics, services, contracts, billing, churn label.

2. Project Pipeline

Data Ingestion & Cleaning (Phase 2)

Handled missing values (TotalCharges → numeric).

Encoded categorical variables (one-hot encoding).

Train-test split (70:30).

Exploratory Data Analysis (Phase 3)

Categorical: churn % by internet service, contract, payment method.

Numerical: tenure, monthly charges, total charges (binning + stats tests).

Interaction effects: contract × payment method, internet service × tech support.

Modeling & Drivers (Phase 4)

Logistic Regression (interpretable coefficients).

Random Forest / XGBoost (feature importance).

SHAP values for explainability.

Key drivers confirmed: contract type, payment method, internet service, tenure.

Advanced Analysis

Engagement Score → composite metric (tenure + services + billing).

Survival Analysis → Kaplan–Meier & Cox models predict when churn happens.

End-to-End Skills (Phase 5)

Packaged pipeline: ingestion → preprocessing → modeling → prediction.

Experiment tracking (MLflow/DVC).

Docker for reproducibility.

Deployment (Phase 8)

Streamlit app → predict churn probability + SHAP explanations.

Power BI dashboard → interactive churn insights for executives.

3. Key Findings

Baseline churn rate = 26.6% (~1 in 4 customers).

High-risk groups:

Month-to-month contracts → 43% churn.

Fiber optic internet → 42% churn.

Electronic check payment → 45% churn.

Low-risk groups:

Two-year contracts → 3% churn.

Customers with Tech Support → 15% churn.

Numerical insights:

Tenure < 6 months → 54% churn vs 5+ years → 7%.

Monthly charges $80–95 → 37% churn vs $20 → 9%.

4. Business Impact

By targeting the top 10% at-risk customers (identified by model), the company could reduce churn by ~20%.

With average monthly revenue of ~$70 per customer, preventing churn of half of these customers could save $210k/year.

Actionable levers:

Promote longer-term contracts.

Provide incentives to fiber optic users.

Reduce friction for high-bill customers.

Expand support services (TechSupport, OnlineSecurity).

5. Tools & Technologies

Python: pandas, numpy, scikit-learn, xgboost, lifelines, shap.

Visualization: matplotlib, seaborn, Power BI.

Deployment: Streamlit, Docker.

Experiment Tracking: MLflow / DVC.

6. Limitations & Future Work

Dataset is static and clean (not fully realistic).

Lacks behavioral features (usage frequency, complaints, customer support logs).

Future improvements:

Add real-time usage & support ticket data.

Expand survival analysis with richer features.

Deploy model on cloud (AWS/GCP) for production.
