# Customer Churn Prediction — End-to-End ML System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-2.1-blue" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/SHAP-Explainable-ff69b4" />
</p>

---

## Overview

A production-grade machine learning system that predicts telecom customer churn using **Logistic Regression**, deployed as a live **FastAPI** REST API with a dark-themed web frontend. The project covers the full ML lifecycle — from data cleaning and EDA through model comparison and SHAP explainability to Dockerized deployment.

> **Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — ~7,000 customers with demographics, service subscriptions, billing, and churn labels.

---

## Live Demo

🔗 **Coming soon**

---

## Key Results

| Metric | Logistic Regression | XGBoost | Random Forest |
|---|:---:|:---:|:---:|
| **AUC-ROC** | **0.836** ✅ | 0.831 | 0.824 |
| Interpretability | High (coefficients + SHAP) | Medium | Low |
| Inference Speed | Fast | Medium | Slow |

- **3 models compared** — Logistic Regression selected for best AUC-ROC + interpretability
- **SHAP explainability** reveals the top churn drivers at both global and individual prediction levels

### Top Churn Drivers

| Driver | Impact |
|---|---|
| 📄 Month-to-month contracts | 43% churn rate (vs 3% for two-year) |
| ⏳ Short tenure (< 6 months) | 54% churn rate |
| 💰 High monthly charges ($80–95) | 37% churn rate |
| 💳 Electronic check payment | 45% churn rate |
| 🌐 Fiber optic (no add-ons) | 42% churn rate |

**Baseline churn rate: 26.6%** — roughly 1 in 4 customers

---

## Project Structure

```
Customer churn ML end to end/
│
├── api/
│   ├── main.py                  # FastAPI — /predict, /health, serves frontend
│   └── static/
│       └── index.html           # Dark-themed prediction web UI
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Original Kaggle dataset
│   ├── telco_cleaned.csv                       # Cleaned dataset
│   ├── X_train.csv / X_test.csv                # Feature splits
│   └── y_train.csv / y_test.csv                # Label splits
│
├── models/
│   ├── lr_pipeline.joblib       # ✅ Logistic Regression pipeline (active)
│   └── xgb_pipeline.joblib     # XGBoost pipeline
│
├── notebooks/
│   ├── 01-Data cleaning Preprocessing.ipynb
│   ├── 02-Exploratory Data Analysis.ipynb
│   └── 03-Modelling_and_Eval.ipynb
│
├── reports/
│   ├── data_preprocessing.md    # Cleaning & preprocessing notes
│   ├── eda_summary.md           # EDA findings
│   ├── figures/                 # 31 charts: SHAP, EDA, risk profiles
│   └── tables/
│
├── Streamlit/                   # (Planned) Streamlit dashboard
├── power Bi/                    # (Planned) Power BI dashboard
├── src/                         # (Planned) Source modules
│
├── Dockerfile                   # Production container config
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## ML Pipeline

### 1 · Data Cleaning & Preprocessing
- Converted `TotalCharges` to numeric, handled missing values
- One-hot encoded all categorical variables
- 70/30 train-test split

### 2 · Exploratory Data Analysis
- Churn rate analysis by contract type, internet service, payment method
- Statistical tests (chi-square) for categorical features
- Binning analysis for tenure, monthly charges, total charges
- Interaction effects: contract × payment method, internet service × tech support

### 3 · Modelling & Evaluation
- **Logistic Regression** — interpretable coefficients, best AUC-ROC (0.836)
- **XGBoost** — gradient boosting with feature importance
- **SHAP values** — global and local explainability
- High-risk and low-risk customer profiling

### 4 · API & Web Frontend
- FastAPI REST API with `/predict` and `/health` endpoints
- Dark-themed web UI — fill in 19 customer fields, get instant churn probability
- Color-coded risk levels (🟢 Low / 🟡 Medium / 🔴 High) with actionable recommendations

### 5 · Dockerized Deployment
- Lightweight `python:3.11-slim` base image
- Single-command build and run

---

## Quick Start

### Option A — Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/Customer-Churn-ML-end-to-end.git
cd Customer-Churn-ML-end-to-end

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the API server
python -m uvicorn api.main:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

### Option B — Run with Docker 🐳

```bash
# Build the image
docker build -t churn-predictor .

# Run the container
docker run -p 8000:8000 churn-predictor
```

Open **http://localhost:8000** in your browser.

---

## API Reference

### `GET /health`
Health check endpoint.

```json
{ "status": "healthy" }
```

### `POST /predict`
Submit customer features to get churn prediction.

**Request body** (JSON with 19 fields):
```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "Yes",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.50,
  "TotalCharges": 1026.00
}
```

**Response:**
```json
{
  "churn_probability": 0.82,
  "prediction": "Yes",
  "risk_level": "High"
}
```

---

## Business Impact

By targeting the **top 10% at-risk customers** identified by the model, the company could reduce churn by ~20%.

With average monthly revenue of ~$70/customer, preventing churn of half those customers could save an estimated **$210k/year**.

**Actionable levers:**
- 📄 Promote longer-term contracts (two-year: 3% churn vs month-to-month: 43%)
- 💰 Provide billing incentives for high-charge customers
- 🛡️ Bundle security & tech support add-ons with fiber optic plans
- 💳 Encourage auto-pay over electronic checks

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Data & ML** | pandas · NumPy · scikit-learn · XGBoost · SHAP |
| **Visualization** | Matplotlib · Seaborn · Power BI |
| **API** | FastAPI · Uvicorn · aiofiles |
| **Deployment** | Docker · Python 3.11 |
| **Notebooks** | Jupyter |

---

## Limitations & Future Work

**Current limitations:**
- Static dataset — no real-time data pipeline
- Lacks behavioral features (usage frequency, complaints, support logs)

**Planned improvements:**
- ☁️ Cloud deployment (AWS / GCP / Azure)
- 📊 Streamlit / Power BI interactive dashboards
- 📡 Real-time usage & support ticket data integration
- 🔄 Model monitoring, drift detection & automated retraining
- 🧪 Expand survival analysis with richer features

---

## License

This project is for educational and portfolio purposes. Dataset provided by IBM via [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

---

<p align="center">
  Built with ❤️ by <strong>Saniya</strong>
</p>
