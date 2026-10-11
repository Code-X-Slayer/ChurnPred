# Customer Churn Prediction — Mini ML Project

A practical, beginner-friendly machine learning project that predicts customer churn in the telecommunications industry using the **IBM Telco Customer Churn** dataset.

The project evaluates 4 classification models, demonstrates an empirical **SMOTE** balancing experiment applied strictly to training data, explains predictions using **SHAP** (SHapley Additive exPlanations), and delivers an interactive **Streamlit** dashboard.

---

## 📖 Comprehensive Reviewer Guide
> **Looking for a deep-dive explanation for an interview or code review?**
> Check out [REVIEWER_GUIDE.md](file:///c:/Users/Vijay%20Karthik/Downloads/ChurnPred/ChurnPred/REVIEWER_GUIDE.md) for an end-to-end walkthrough of all code, the mathematical mechanics of SMOTE, a concrete customer case study with step-by-step SHAP waterfall calculations, and common reviewer Q&A.

---

## 📁 Repository Structure

```text
ChurnPred/
├── REVIEWER_GUIDE.md        # Comprehensive technical guide for code reviews / viva
├── churn_prediction.ipynb   # Complete, executed end-to-end Jupyter Notebook
├── app.py                   # Clean, interactive Streamlit dashboard
├── train_and_export.py      # Pipeline script to train models and export artifacts
├── dataset.csv              # IBM Telco Customer Churn dataset (7,043 rows)
├── requirements.txt         # Project dependencies
├── model_artifacts.joblib   # Serialized models, scaler, and preprocessor
├── metrics.json             # Benchmark metrics for Baseline vs SMOTE
└── sample_customers.csv     # Test sample records for rapid testing
```

---

## 📊 1. Dataset & Preprocessing

- **Dataset**: IBM Telco Customer Churn benchmark (7,043 customer rows, 21 columns).
- **Target Variable**: `Churn` (`Yes`: 1,869 [26.5%], `No`: 5,174 [73.5%]).
- **Missing Value Handling**: The `TotalCharges` feature contains 11 whitespace entries (`" "`) corresponding to new customers with `tenure = 0`. These are logically converted to numeric and filled with `0.0`.
- **Feature Engineering**:
  - Dropped non-predictive `customerID`.
  - Binary encoding for binary indicators (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`).
  - One-hot encoding for multi-category features (`Contract`, `InternetService`, `PaymentMethod`, etc.).
  - Feature scaling with `StandardScaler` on numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`).
  - Stratified 80/20 train/test split.

---

## 🤖 2. Models Evaluated

Four classifiers were trained using standard, clean configurations:
1. **Logistic Regression** (`max_iter=1000`)
2. **Random Forest** (`n_estimators=100`, `max_depth=8`)
3. **XGBoost** (`n_estimators=100`, `max_depth=4`, `learning_rate=0.1`)
4. **LightGBM** (`n_estimators=100`, `max_depth=4`, `learning_rate=0.1`)

---

## ⚖️ 3. The SMOTE Experiment

To address class imbalance (~73.5% non-churn vs ~26.5% churn), we compared performance **Without SMOTE** vs. **With SMOTE**:
- **Critical Rule**: SMOTE is applied **strictly to the training partition** to prevent data leakage.
- **Evaluation**: Both settings are evaluated on the identical, untouched test set (1,409 customers).

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

### Did SMOTE Help?
- **Recall increased dramatically (+20% to +26%)**: Random Forest Recall jumped from **50.8% to 77.0%**, and XGBoost Recall jumped from **52.9% to 74.6%**. The model flags significantly more at-risk customers.
- **Precision decreased (-12% to -15%)**: More false positives occurred as the decision threshold became more sensitive.
- **Accuracy dropped slightly (~80% to ~75%)**: Driven by the higher false positive rate.
- **F1-Score improved for ensemble models**: Random Forest F1 rose from 0.579 to 0.622.
- **Conclusion**: SMOTE does *not* automatically improve all metrics. It is an intentional trade-off: in customer retention, failing to identify a churner is typically much more costly than offering a retention discount to a loyal customer, making higher Recall commercially desirable.

---

## 🔍 4. SHAP Explainability

Using **SHAP TreeExplainer** on the top model:
1. **Global Feature Importance**: Top predictors include `Contract_Two year`, `Contract_One year`, `tenure`, `InternetService_Fiber optic`, and `MonthlyCharges`.
2. **Direction of Impact (Beeswarm)**:
   - **High Churn Risk**: Month-to-month contract, short tenure (< 6 months), Fiber optic internet, high monthly charges.
   - **Low Churn Risk / Retention**: 1- or 2-year contracts, long tenure (> 36 months), online tech support.
3. **Local Customer Explanations**: Waterfall plots reveal the exact positive and negative contributors to an individual customer's churn risk score.

---

## 🚀 5. How to Run

### Installation
Ensure Python 3.10+ is installed, then install the dependencies:
```bash
pip install -r requirements.txt
```

### Running the Jupyter Notebook
Open and explore the step-by-step analysis:
```bash
jupyter notebook churn_prediction.ipynb
```

### Running the Streamlit App
Launch the interactive web application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🎯 6. Streamlit Dashboard Features
- **Overview Tab**: Dataset dimensions, class balance chart, and project background.
- **Model Comparison Tab**: Side-by-side metric tables, interactive bar charts, and SMOTE trade-off analysis.
- **SHAP Explainability Tab**: Global feature importance and summary beeswarm plots with business recommendations.
- **Live Customer Prediction Tab**: Interactive customer profile form, instant churn probability gauge, risk classification badge, and custom SHAP waterfall chart.
