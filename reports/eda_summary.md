
# Customer Churn – Exploratory Data Analysis (EDA) Summary

## 1. Baseline
- Overall churn rate in training data = **26.6%**  
- About 1 in 4 customers leave.

---

## 2. Top High-Risk Groups
![High Risk](../reports/figures/high_risk_summary.png)

- Fiber optic customers churn at **41.8%** (~1.6× baseline).  
- Month-to-month contracts churn at **43%** vs only **3%** for two-year contracts.  
- Electronic check users churn at **45%**, the highest of all payment methods.

---

## 3. Top Low-Risk Groups
![Low Risk](../reports/figures/low_risk_summary.png)

- Two-year contracts churn at just **3%**.  
- Customers with TechSupport churn at **15%** vs 42% without it.  
- Senior citizens with long tenure are less risky than new younger customers.

---

## 4. Feature Significance (Chi-square)
![Chi Square](../reports/figures/chi_square_results.png)

- **Contract, InternetService, PaymentMethod** are the strongest churn predictors (p < 1e-100).  
- **PhoneService, Gender** show no significant impact.

---

## 5. Key Interactions
![Contract × PaymentMethod](../reports/figures/churn_contract_payment.png)  
- 🔴 Month-to-month + Electronic check churn = **54%**  
- 🟢 Two-year + Mailed check churn = **1%**

![InternetService × TechSupport](../reports/figures/churn_internet_techsupport.png)  
- 🔴 Fiber optic + No TechSupport churn = **49%**  
- 🟢 DSL + TechSupport churn = **9%**

---

## 6. Numerical Insights

### Tenure
![Tenure bins](../reports/figures/churn_by_bins_tenure.png)  
![Tenure summary](../reports/figures/num_summary_tenure.png)  

- 🔴 Customers in their **first 6 months churn 54%**.  
- 🟢 Customers with **5+ years tenure churn only 7%**.  
- Median tenure = **10 months (churned)** vs **38 months (retained)**.

### Monthly Charges
![MonthlyCharges bins](../reports/figures/churn_by_bins_MonthlyCharges.png)  
![MonthlyCharges summary](../reports/figures/num_summary_MonthlyCharges.png)  

- 🔴 Bills **$80–95** → churn ~37%.  
- 🟢 Bills **$20–25** → churn ~9%.  
- Higher charges strongly increase churn.

### Total Charges
![TotalCharges bins](../reports/figures/churn_by_bins_TotalCharges.png)  
![TotalCharges summary](../reports/figures/num_summary_TotalCharges.png)  

- 🔴 Very low lifetime spend (<$300) churns **46%**.  
- 🟢 High lifetime spend ($4500+) churns **14%**.  
- Loyal customers naturally have higher total charges.

---

## 7. Takeaways
- **Contract type, internet service, and payment method** are the strongest churn drivers.  
- **New customers with high bills** are at the greatest risk.  
- **Support services (TechSupport, OnlineSecurity)** dramatically reduce churn.  
- Retention focus:  
  - Customers in their first year.  
  - Customers on month-to-month + electronic check.  
  - Fiber optic users without support.  

---
