# Loan-Payback-Probability-using-Machine-Learning

This project builds a high-performance machine learning model to predict whether a customer will **pay back a loan** using real-world tabular data with heavy categorical features.

The solution demonstrates advanced feature engineering, categorical handling, CatBoost optimization, and model interpretability.

---

## 📌 Problem Statement

Financial institutions must assess the risk of loan default before approving applications.  
The goal of this project is to predict:

> **Will a customer pay back the loan or default?**

This is a **binary classification problem** evaluated using **AUC score**.

---

## 🚀 Why CatBoost?

The dataset contains many categorical variables with messy formatting.  
CatBoost was chosen because:

- Handles categorical features **without one-hot encoding**
- Avoids data leakage using ordered boosting
- Works extremely well on tabular financial data
- Requires minimal preprocessing compared to XGBoost/LightGBM

---

## 🧹 Data Cleaning Challenges Solved

The raw data had issues like:

- Categorical values stored as: `'Single'`, `'Married'` (extra quotes)
- Mixed data types
- Missing values
- Skewed numerical features

Solutions implemented:

- String cleaning for all object columns
- Power transformation for skewed numerical data
- Intelligent categorical combinations like:
  - `edu_emp_combo`
  - `purpose_grade_combo`

---

## 🧠 Feature Engineering

Created meaningful business features:
```
| Feature | Description |
|---|---|
| edu_emp_combo | Education + Employment risk signal |
| purpose_grade_combo | Loan purpose + credit grade |
| Recency based metrics | Customer recent activity |
| Numerical normalization | Using PowerTransformer |
```
---

## 🏗️ Modeling Pipeline

1. Train/Validation split
2. Creation of CatBoost `Pool` with categorical features
3. Hyperparameter tuning using CatBoost `grid_search`
4. Evaluation using AUC
5. Feature importance visualization
6. Training vs Validation AUC monitoring

---

## ⚙️ Hyperparameter Tuning

```python
param_grid = {
    'depth':[4,5,6,7,8],
    'learning_rate':[0.01,0.03,0.05],
    'iterations':[1500,2000,2500],
    'l2_leaf_reg':[1,3,5]
}
