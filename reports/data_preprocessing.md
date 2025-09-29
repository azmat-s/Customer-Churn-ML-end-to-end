
# Customer Churn – Data Preprocessing Summary

## 1. Data Loading & Inspection
- Dataset: **WA_Fn-UseC_-Telco-Customer-Churn.csv**
- Shape: 7043 rows × 21 columns
- Initial inspection showed customerID column (non-predictive).

---

## 2. Data Cleaning
- Dropped column: **customerID**
- Converted **TotalCharges** → numeric
- Found 11 missing values in TotalCharges → handled (removed or imputed)
- After cleaning: dataset ready for feature engineering

---

## 3. Feature Categorization
- **Categorical features (16):**
  gender, SeniorCitizen, Partner, Dependents, PhoneService,
  MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
  DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
  Contract, PaperlessBilling, PaymentMethod

- **Numerical features (3):**
  tenure, MonthlyCharges, TotalCharges

- **Target:** Churn (Yes/No → 1/0)

---

## 4. Target Encoding
- Converted **Churn** to binary (Yes = 1, No = 0)
- Class distribution:
  - Stayed (0): ~73.4%
  - Churned (1): ~26.6%
- ⚠️ Noted class imbalance → will matter in model training

---

## 5. Train-Test Split
- Train set: 5625 rows (80%)
- Test set: 1407 rows (20%)
- Stratified split preserved churn ratio:
  - Train churn % ≈ 26.6%
  - Test churn % ≈ 26.6%

---

## ✅ Deliverables
- Clean dataset with correct dtypes
- Features separated into categorical & numerical
- Target encoded
- Train/test sets created and saved
