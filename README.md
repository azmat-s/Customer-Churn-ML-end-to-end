Telco Customer Churn Prediction – End-to-End ML Project

🚀 This repository documents my journey of building an end-to-end Machine Learning project to predict customer churn for a telecom company.

Currently, I have completed Phase 1: Project Setup & Planning.

🔹 Phase 1: Setup & Planning
📝 Step 1.1 – Understand the Business Problem

Churn definition: Customer churn refers to when a customer stops using a company’s services.

Why it matters: For telecom companies, churn directly impacts revenue. Acquiring a new customer is often more expensive than retaining an existing one.

Business need: If we can predict which customers are likely to churn, the company can take proactive actions (discounts, offers, better support) to retain them.

Output: Clear definition of churn and its business impact.

🎯 Step 1.2 – Define Goals & Metrics

Goal: Build a machine learning model that predicts whether a customer will churn.

Success criteria: Since churn datasets are usually imbalanced (fewer churners than non-churners), using accuracy alone is misleading.

Instead, we will focus on:

Recall (Sensitivity) → Catch as many churners as possible.

Precision → Ensure predictions are correct (avoid wasting offers on loyal customers).

F1-score → Balance between precision and recall.

ROC-AUC → Overall model quality across thresholds.

Output: Defined performance metrics (F1, ROC-AUC) to measure model success.

📂 Step 1.3 – Setup Project Folder

To keep the project organized, I created a structured folder layout:

Customer-Churn-ML-End-to-End/
│
├── data/                # Dataset
├── notebooks/           # Jupyter notebooks
├── streamlit/           # Streamlit app code
├── powerbi/             # Power BI dashboard
├── models/              # Saved ML models
└── README.md            # Documentation


Output: A GitHub-ready folder structure for end-to-end development.

✅ Summary of Phase 1
Step	Task	Tools	Output
1.1	Understand the business problem	—	Defined churn and business impact
1.2	Define goals & metrics	—	Selected F1-score, Recall, Precision, ROC-AUC
1.3	Setup project folder	GitHub	Created /data, /notebooks, /streamlit, /powerbi
🎯 Next Steps

Phase 2 → Data cleaning & preprocessing

Phase 3 → Exploratory Data Analysis (EDA)

Phase 4 → Model building & evaluation

Phase 5 → Deployment (Streamlit) + Power BI dashboard

👨‍💻 Author

📌 Azmat S. – Aspiring Data Scientist | Machine Learning Enthusiast
