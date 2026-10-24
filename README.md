# Customer Churn Prediction — Machine Learning Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churnpred-ml.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB1424.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-268B07.svg)](https://lightgbm.readthedocs.io/)

> 🌐 **Live Web Application**: Explore the deployed dashboard at **[https://churnpred-ml.streamlit.app/](https://churnpred-ml.streamlit.app/)**  
> Test customer profiles, tune decision thresholds, view Confusion Matrices, and inspect real-time SHAP waterfall attributions directly in your browser without local installation.

---

## 📌 Project Overview

Customer churn occurs when subscribers cancel their service contracts. Because acquiring new customers costs significantly more than retaining existing ones, early churn detection is critical.

This project delivers an end-to-end customer churn prediction solution using the **IBM Telco Customer Churn** dataset:
- Evaluates **4 core algorithms** (Logistic Regression, Random Forest, XGBoost, LightGBM).
- Demonstrates an empirical **SMOTE** experiment applied strictly to training partitions to prevent data leakage.
- Provides interactive evaluation diagnostics (**Confusion Matrix**, **ROC-AUC curves**, **custom decision threshold tuning**).
- Explains model decisions globally and locally using **SHAP** (SHapley Additive exPlanations).
- Deploys as an interactive **Streamlit** dashboard.

---

## 📁 Repository Structure

```text
ChurnPred/
├── data/
│   ├── dataset.csv            # Raw IBM Telco benchmark dataset (7,043 rows)
│   └── sample_customers.csv   # Sample test records for quick UI testing
├── models/
│   ├── model_artifacts.joblib # Serialized models, scaler, and preprocessor
│   └── metrics.json           # Benchmark evaluation results (Baseline vs SMOTE)
├── notebooks/
│   └── churn_prediction.ipynb # Step-by-step exploratory data analysis & modeling
├── src/
│   └── train_and_export.py    # Standalone reproducible training & export pipeline
├── app.py                     # Interactive Streamlit dashboard application
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Git ignore rules
```

---

## 📊 1. Dataset & Preprocessing

- **Dataset**: IBM Telco Customer Churn benchmark (7,043 rows, 21 columns).
- **Target Variable**: `Churn` (`Yes`: 1,869 [26.54%], `No`: 5,174 [73.46%]).
- **Missing Value Handling**: Imputed 11 whitespace entries (`" "`) in `TotalCharges` for new customers (`tenure = 0`) to `0.0`.
- **Feature Engineering**:
  - Dropped non-predictive `customerID`.
  - Binary and one-hot encoding for categorical variables (`drop_first=True` to avoid multicollinearity).
  - Feature scaling with `StandardScaler` on numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`).
  - Stratified 80/20 train/test split.

---

## 🤖 2. Models Evaluated

Four classifiers trained under both **Baseline (No SMOTE)** and **With SMOTE** settings:
1. **Logistic Regression** (`max_iter=1000`)
2. **Random Forest** (`n_estimators=100`, `max_depth=8`)
3. **XGBoost** (`n_estimators=100`, `max_depth=4`, `learning_rate=0.1`)
4. **LightGBM** (`n_estimators=100`, `max_depth=4`, `learning_rate=0.1`)

---

## ⚖️ 3. The SMOTE Experiment

To tackle target class imbalance (73.5% non-churn vs. 26.5% churn):
- **Zero Data Leakage**: SMOTE is applied **strictly to the training partition**; the test set remains completely untouched.
- **Evaluation**: Both settings are tested on the identical held-out test partition (1,409 customers).

### Benchmark Results

| Model | Setting | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | Baseline (No SMOTE) | 0.8062 | 0.6593 | 0.5588 | 0.6049 | 0.8422 |
| **Logistic Regression** | With SMOTE | 0.7374 | 0.5038 | 0.7112 | 0.5898 | 0.8192 |
| **Random Forest** | Baseline (No SMOTE) | 0.8041 | 0.6738 | 0.5080 | 0.5793 | 0.8438 |
| **Random Forest** | With SMOTE | 0.7516 | 0.5217 | **0.7701** | **0.6220** | 0.8408 |
| **XGBoost** | Baseline (No SMOTE) | 0.7984 | 0.6471 | 0.5294 | 0.5824 | **0.8447** |
| **XGBoost** | With SMOTE | 0.7509 | 0.5215 | **0.7460** | **0.6139** | 0.8344 |
| **LightGBM** | Baseline (No SMOTE) | 0.8048 | 0.6634 | 0.5374 | 0.5938 | 0.8440 |
| **LightGBM** | With SMOTE | 0.7530 | 0.5247 | **0.7380** | **0.6133** | 0.8338 |

### Key Business Takeaway
- **Recall increased by +20% to +26%**: Random Forest Recall rose from **50.8% to 77.0%**, and XGBoost Recall rose from **52.9% to 74.6%**, catching far more at-risk customers.
- **Precision trade-off**: False positives increased as the model grew more sensitive, dropping precision from ~66% to ~52%.
- **Decision Rationale**: In retention campaigns, missing a churner costs far more than offering a discount to a loyal customer. SMOTE represents an intentional engineering trade-off favoring high Recall.

---

## 🔍 4. SHAP Explainability

Using **SHAP TreeExplainer** on the primary production model (`XGBoost With SMOTE`):
1. **Global Importance**: Top drivers include `Contract_Two year`, `Contract_One year`, `tenure`, `InternetService_Fiber optic`, and `MonthlyCharges`.
2. **Directional Impact**:
   - **Increased Churn Risk**: Month-to-month contracts, short tenure (< 6 months), Fiber optic internet, and higher monthly charges.
   - **Increased Retention**: One/two-year contracts, tenure > 36 months, and active tech support / online security subscriptions.
3. **Local Explanations**: Waterfall plots illustrate exactly how individual features contribute positively or negatively to any customer's risk probability.

---

## 🚀 5. How to Run

### Live Demo (No Setup Needed)
Access the live deployment on Streamlit Community Cloud:  
👉 **[https://churnpred-ml.streamlit.app/](https://churnpred-ml.streamlit.app/)**

### Local Installation
```bash
# 1. Clone repository
git clone https://github.com/Code-X-Slayer/ChurnPred.git
cd ChurnPred

# 2. Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt
```

### Running the Streamlit App Locally
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Running the Training Pipeline (Optional)
To retrain all 8 model variants and re-export artifacts:
```bash
python src/train_and_export.py
```

### Running the Jupyter Notebook
```bash
jupyter notebook notebooks/churn_prediction.ipynb
```

---

## 🎯 6. Streamlit Dashboard Features

The dashboard provides five interactive tabs:

1. **📋 Overview**: High-level KPIs, class imbalance visualization, and dataset summary.
2. **⚖️ Model Comparison & SMOTE**: Benchmark performance table and interactive metric comparison bar charts.
3. **📊 Model Evaluation & ROC**: Model picker, dynamic decision threshold slider (0.10–0.90), live Confusion Matrix, and ROC curve with operating point marker.
4. **🔍 SHAP Explainability**: Global feature importance bar chart and SHAP beeswarm summary plot.
5. **🎯 Live Customer Prediction**: Profile configuration form with one-click quick-fill samples, real-time churn probability, risk status badges, and local SHAP waterfall plots.
