Telco Customer Churn Prediction – End-to-End ML Project

🚀 An end-to-end Machine Learning project to predict customer churn for a telecom company, with deployment (Streamlit) and business insights (Power BI).

🔹 Problem Statement

Customer churn is a critical problem in the telecom industry — losing existing customers directly impacts revenue.
The goal of this project is to:

Predict whether a customer will churn (Yes/No).

Identify key drivers of churn using explainability techniques.

Provide actionable business insights through dashboards.

Deploy the model into a user-friendly web app.

🎯 Project Objectives

✔️ Perform EDA to explore churn patterns.
✔️ Build multiple ML models (Logistic Regression, Random Forest, XGBoost).
✔️ Optimize using hyperparameter tuning.
✔️ Evaluate with Recall, F1-score, ROC-AUC (not just Accuracy).
✔️ Apply SHAP explainability for feature importance.
✔️ Deploy as an interactive Streamlit web app.
✔️ Create a Power BI dashboard for business storytelling.

⚙️ Tech Stack

Python → pandas, NumPy, scikit-learn, XGBoost, SHAP

Visualization → matplotlib, seaborn

Dashboarding → Power BI

Deployment → Streamlit, Hugging Face Spaces

Version Control → Git, GitHub

📊 Exploratory Data Analysis (EDA)

Some key patterns discovered:

Customers with month-to-month contracts churn significantly more.

Higher MonthlyCharges correlate with churn.

Longer tenure customers are more loyal.

Payment methods like Electronic check show higher churn rates.

(Visuals will be included in notebooks + Power BI dashboard 📈)

🤖 Machine Learning Models
Logistic Regression
Random Forest
XGBoost 

➡️ Best model: XGBoost, optimized with hyperparameter tuning.

🔍 Model Explainability

Using SHAP values, we identified the most important features:

Contract type
Tenure
MonthlyCharges
Payment method

This helps the business understand why customers are churning.

🌐 Deployment

✅ Streamlit App → Interactive app where users can input customer details and get churn prediction + explanation.

✅ Power BI Dashboard → Business-friendly insights for stakeholders.

🔗 Live Demo (Coming Soon)
🔗 Power BI Dashboard (Coming Soon)

📂 Project Structure
Customer-Churn-ML-End-to-End/
│
├── data/                # Dataset
├── notebooks/           # Jupyter notebooks (EDA, modeling, explainability)
├── streamlit/           # Streamlit app code
├── powerbi/             # Power BI dashboard file
├── models/              # Saved ML models
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

🚀 How to Run Locally

Clone the repo:

git clone https://github.com/azmat-s/Customer-Churn-ML-end-to-end.git
cd Customer-Churn-ML-end-to-end


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run streamlit/app.py

🏆 Key Learnings

Hands-on experience with end-to-end ML pipelines.

Importance of model explainability (SHAP).

Deploying ML solutions with Streamlit + GitHub + Hugging Face.

Building interactive BI dashboards for storytelling.

✨ Future Work

Extend to an MLOps pipeline with MLflow, Docker, and CI/CD.

Deploy API version with FastAPI / Flask.

Add real-time monitoring for churn drift detection.

