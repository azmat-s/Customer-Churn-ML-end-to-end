# Customer Churn Prediction — End-to-End ML Project

## 1. Project Overview

**Business Problem:** Customer churn is costly for telecom providers. Retaining existing customers is cheaper than acquiring new ones.

**Goal:** Predict which customers are at risk of churn, explain why churn happens, and provide actionable recommendations for reducing churn.

**Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — ~7,000 customers with demographics, services, contracts, billing, and churn label.

---

## 2. Project Structure

```
├── api/
│   ├── main.py              # FastAPI app — /predict, /health, serves frontend
│   └── static/
│       └── index.html       # Dark-themed prediction UI
├── data/
│   ├── raw/                 # Original Kaggle dataset
│   └── processed/           # Cleaned train/test splits
├── models/
│   ├── lr_pipeline.joblib   # Logistic Regression pipeline (active)
│   └── xgb_pipeline.joblib  # XGBoost pipeline
├── notebooks/
│   ├── 01-Data cleaning Preprocessing.ipynb
│   ├── 02-Exploratory Data Analysis.ipynb
│   └── 03-Modelling_and_Eval.ipynb
├── reports/
│   └── figures/             # SHAP plots and EDA charts
├── requirements.txt
└── README.md
```

---

## 3. Project Pipeline

### Data Cleaning & Preprocessing
- Handled missing values (`TotalCharges` → numeric)
- Encoded categorical variables (one-hot encoding)
- Train-test split (70:30)

### Exploratory Data Analysis
- Churn rate by internet service, contract type, payment method
- Tenure, monthly charges, total charges (binning + stats tests)
- Interaction effects: contract × payment method, internet service × tech support

### Modelling & Evaluation
- Logistic Regression (interpretable coefficients)
- XGBoost (feature importance)
- SHAP values for explainability
- Key drivers: contract type, payment method, internet service, tenure

### API & Frontend
- FastAPI REST API with `/predict` endpoint
- Dark-themed web UI — fill in 19 customer fields, get instant churn probability with color-coded risk level and actionable recommendation
- Run locally with: `python -m uvicorn api.main:app --reload` then open `http://127.0.0.1:8000`

---

## 4. Key Findings

**Baseline churn rate: 26.6%** (~1 in 4 customers)

| Segment | Churn Rate |
|---|---|
| Month-to-month contract | 43% |
| Fiber optic internet | 42% |
| Electronic check payment | 45% |
| Two-year contract | 3% |
| Customers with Tech Support | 15% |

**Numerical insights:**
- Tenure < 6 months → 54% churn vs 5+ years → 7%
- Monthly charges $80–95 → 37% churn vs ~$20 → 9%

---

## 5. Business Impact

By targeting the top 10% at-risk customers identified by the model, the company could reduce churn by ~20%.

With average monthly revenue of ~$70/customer, preventing churn of half those customers could save **$210k/year**.

**Actionable levers:**
- Promote longer-term contracts
- Provide incentives to fiber optic users
- Reduce friction for high-bill customers
- Expand support services (TechSupport, OnlineSecurity)

---

## 6. Tools & Technologies

| Category | Tools |
|---|---|
| Data & ML | pandas, numpy, scikit-learn, xgboost, shap, lifelines |
| Visualization | matplotlib, seaborn, Power BI |
| API | FastAPI, uvicorn |
| Notebooks | Jupyter |

---

## 7. Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API + frontend
python -m uvicorn api.main:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

---

## 8. Limitations & Future Work

- Dataset is static and clean (not fully realistic)
- Lacks behavioral features (usage frequency, complaints, support logs)

**Future improvements:**
- Add real-time usage & support ticket data
- Expand survival analysis with richer features
- Deploy to cloud (AWS/GCP) for production
- Add model monitoring and drift detection
