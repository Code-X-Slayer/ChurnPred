import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #3498db;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .model-info-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #dcdfe6;
    }
    .churn-badge-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 8px 14px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
        font-size: 1.05rem;
    }
    .churn-badge-low {
        background-color: #d4edda;
        color: #155724;
        padding: 8px 14px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    artifacts = joblib.load("model_artifacts.joblib")
    return artifacts

artifacts = load_artifacts()
all_models = artifacts["all_models"]
best_model_name = artifacts.get("best_model_name", "XGBoost (With SMOTE)")
scaler = artifacts["scaler"]
feature_names = artifacts["feature_names"]
cat_options = artifacts["cat_options"]
results_df = artifacts["results_df"]
X_test_scaled = artifacts["X_test_scaled"]
y_test = artifacts["y_test"]
X_train_scaled = artifacts["X_train_scaled"]

@st.cache_resource
def get_global_shap_explainer(_model, X_sample):
    explainer = shap.TreeExplainer(_model)
    shap_vals = explainer(X_sample)
    return explainer, shap_vals

default_xgb_model = all_models["XGBoost_smote"]
global_explainer, test_shap_values = get_global_shap_explainer(default_xgb_model, X_test_scaled[:300])

def get_customer_shap_explanation(model, model_key, X_sample):
    if "Logistic" in model_key:
        explainer = shap.LinearExplainer(model, X_train_scaled.iloc[:100])
        sv = explainer(X_sample)
        return sv[0]
    elif "Random Forest" in model_key:
        explainer = shap.TreeExplainer(model)
        sv = explainer(X_sample)
        exp = shap.Explanation(
            values=sv.values[0, :, 1],
            base_values=sv.base_values[0, 1],
            data=sv.data[0],
            feature_names=feature_names
        )
        return exp
    else:
        explainer = shap.TreeExplainer(model)
        sv = explainer(X_sample)
        return sv[0]

MODEL_OPTIONS = {
    "XGBoost (With SMOTE) ⭐ Recommended": "XGBoost_smote",
    "XGBoost (Baseline - No SMOTE)": "XGBoost_base",
    "Random Forest (With SMOTE)": "Random Forest_smote",
    "Random Forest (Baseline - No SMOTE)": "Random Forest_base",
    "LightGBM (With SMOTE)": "LightGBM_smote",
    "LightGBM (Baseline - No SMOTE)": "LightGBM_base",
    "Logistic Regression (With SMOTE)": "Logistic Regression_smote",
    "Logistic Regression (Baseline - No SMOTE)": "Logistic Regression_base"
}

st.sidebar.title("⚙️ Model Controls")
selected_model_label = st.sidebar.selectbox(
    "Active Prediction Model:",
    list(MODEL_OPTIONS.keys()),
    index=0,
    help="Choose which algorithm powers the Live Customer Prediction and SHAP explanation."
)
selected_model_key = MODEL_OPTIONS[selected_model_label]
active_model = all_models[selected_model_key]

model_family = selected_model_key.split("_")[0]
is_smote = "With SMOTE" if "smote" in selected_model_key else "Baseline (No SMOTE)"
active_metrics = results_df[
    (results_df["Model"] == model_family) & (results_df["Setting"] == is_smote)
].iloc[0]

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📈 Performance: {model_family}")
st.sidebar.caption(f"Setting: **{is_smote}** (Evaluated on 1,409 Test Customers)")
st.sidebar.write(f"• **Recall**: `{active_metrics['Recall']:.1%}`")
st.sidebar.write(f"• **Precision**: `{active_metrics['Precision']:.1%}`")
st.sidebar.write(f"• **F1-Score**: `{active_metrics['F1-Score']:.3f}`")
st.sidebar.write(f"• **ROC-AUC**: `{active_metrics['ROC-AUC']:.3f}`")
st.sidebar.write(f"• **Accuracy**: `{active_metrics['Accuracy']:.1%}`")
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Switch between Baseline and SMOTE models in the sidebar to compare how churn risk probabilities and SHAP explanations respond.")

st.title("📉 Customer Churn Prediction Dashboard")
st.caption("Machine Learning Mini-Project using the IBM Telco Customer Churn Dataset")

tabs = st.tabs([
    "📋 Overview",
    "⚖️ Model Comparison & SMOTE",
    "🔍 SHAP Explainability",
    "🎯 Live Customer Prediction"
])

with tabs[0]:
    st.header("Project & Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", "7,043")
    with col2:
        st.metric("Total Features", "20")
    with col3:
        st.metric("Churn Rate", "26.54%", delta="-73.46% Retained", delta_color="inverse")
    with col4:
        st.metric("Active Model", f"{model_family} ({'SMOTE' if 'smote' in selected_model_key else 'Base'})")

    st.markdown("""
    ### Business Problem
    Customer churn occurs when customers discontinue their subscription with a service provider. 
    Acquiring new customers is significantly more expensive than retaining existing ones. 
    This project builds and evaluates machine learning models to detect at-risk customers early, 
    quantifies the impact of handling class imbalance via **SMOTE**, and uncovers the key drivers of churn using **SHAP explainability**.
    """)

    st.subheader("Churn Distribution (Class Imbalance)")
    col_chart1, col_chart2 = st.columns([1, 1])
    
    with col_chart1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        counts = [5174, 1869]
        labels = ["Retained (No)", "Churned (Yes)"]
        colors = ["#2ecc71", "#e74c3c"]
        bars = ax.bar(labels, counts, color=colors, width=0.5)
        ax.set_ylabel("Customer Count")
        ax.set_title("Customer Churn Distribution")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{yval:,} ({yval/sum(counts):.1%})", ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, 6000)
        st.pyplot(fig)
        plt.close(fig)

    with col_chart2:
        st.markdown("""
        #### Key Data Observations:
        - **Imbalanced Target**: ~26.5% of customers churned while ~73.5% remained loyal.
        - **Missing Value Handling**: `TotalCharges` contained 11 blank whitespace entries for brand new customers with `tenure = 0`. These were imputed with `0.0`.
        - **Feature Types**: Demographics (gender, partner, dependents), account information (tenure, contract, payment method), and subscribed services (internet, security, tech support).
        """)

with tabs[1]:
    st.header("Model Performance & SMOTE Experiment")
    st.markdown("""
    Four standard classifiers were evaluated under two settings:
    1. **Baseline (No SMOTE)**: Trained on standard training data with natural class distribution.
    2. **With SMOTE**: Synthetic Minority Over-sampling Technique applied **strictly to the training partition** to prevent data leakage.
    Both settings were evaluated on the **identical, untouched test set** (1,409 customers).
    """)

    st.subheader("Performance Comparison Table")
    styled_df = results_df.copy()
    st.dataframe(styled_df, use_container_width=True)

    st.subheader("Visualizing SMOTE vs Baseline Across Metrics")
    
    metric_choice = st.selectbox(
        "Select Metric to Compare:",
        ["Recall", "Precision", "F1-Score", "Accuracy", "ROC-AUC"],
        index=0
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(
        data=results_df,
        x="Model",
        y=metric_choice,
        hue="Setting",
        palette=["#3498db", "#e67e22"],
        ax=ax
    )
    ax.set_title(f"{metric_choice} Comparison: Baseline vs With SMOTE", fontsize=12)
    ax.set_ylim(0, 1.0)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height / 2.),
                        ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    ### Honest Findings: Did SMOTE Help?
    
    - **Significant Recall Boost (+20% to +26%)**:
      - Random Forest Recall increased from **50.8%** to **77.0%**.
      - XGBoost Recall increased from **52.9%** to **74.6%**.
      - The models learned balanced representations of churners, drastically reducing false negatives (missed churners).
    
    - **Precision Trade-off (-12% to -15%)**:
      - Because the models flag churn more proactively, false positive rates increased. Precision dropped from ~66% to ~51-52%.
    
    - **Overall Accuracy & ROC-AUC**:
      - Accuracy dropped from ~80% to ~75% due to additional false alarms.
      - ROC-AUC remained stable (~0.83 - 0.84), showing ranking quality is preserved.
    
    > **Business Takeaway**: SMOTE does **not** unconditionally boost all metrics. It is an intentional engineering lever to prioritize **Recall** when the cost of losing an at-risk customer exceeds the cost of offering a retention incentive.
    """)

with tabs[2]:
    st.header("SHAP Explainability (Model Interpretability)")
    st.markdown(f"""
    Explanations computed using **TreeExplainer** on `{best_model_name}` to reveal how individual features influence churn predictions.
    """)

    col_shap1, col_shap2 = st.columns(2)
    
    with col_shap1:
        st.subheader("Global Feature Importance")
        st.caption("Top 10 features with greatest average impact on model decisions")
        fig, ax = plt.subplots(figsize=(6, 4))
        shap.plots.bar(test_shap_values, max_display=10, show=False)
        st.pyplot(fig)
        plt.close(fig)

    with col_shap2:
        st.subheader("SHAP Summary (Beeswarm) Plot")
        st.caption("Red = high feature value, Blue = low feature value. Right = increases churn.")
        fig, ax = plt.subplots(figsize=(6, 4))
        shap.plots.beeswarm(test_shap_values, max_display=10, show=False)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("""
    ### Strategic Insights from SHAP:
    1. **Contract Type**: `Contract_Two year` and `Contract_One year` strongly push predictions toward **retention** (negative SHAP values). Month-to-month contracts are the single largest risk factor.
    2. **Tenure**: Long tenure significantly protects against churn, whereas tenure under 6 months strongly increases churn probability.
    3. **Internet Service (Fiber Optic)**: Customers on Fiber Optic show higher churn propensity, frequently linked to higher monthly costs or service dissatisfaction.
    4. **Monthly & Total Charges**: Higher monthly charges increase churn risk unless offset by long tenure or multi-year commitment contracts.
    """)

with tabs[3]:
    st.header("🎯 Live Customer Prediction & Local SHAP Explanation")
    st.markdown("Customize a customer profile, select your preferred machine learning model, and inspect the real-time probability along with individual feature attributions.")

    st.markdown("### 1. Active Model Selection")
    c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)
    c_m1.metric("Selected Model", model_family)
    c_m2.metric("Training Strategy", is_smote)
    c_m3.metric("Recall (Test)", f"{active_metrics['Recall']:.1%}")
    c_m4.metric("Precision (Test)", f"{active_metrics['Precision']:.1%}")
    c_m5.metric("ROC-AUC (Test)", f"{active_metrics['ROC-AUC']:.3f}")

    st.markdown("### 2. Customer Profile Configuration")
    if os.path.exists("sample_customers.csv"):
        sample_df = pd.read_csv("sample_customers.csv")
        sample_choice = st.selectbox(
            "⚡ Quick-Fill with a Sample Test Customer (or choose Custom Input):",
            ["Custom Input"] + [f"Customer #{idx} (Actual: {row['Churn']}, Contract: {row['Contract']}, Tenure: {row['tenure']} mo, Monthly: ${row['MonthlyCharges']})" for idx, row in sample_df.iterrows()]
        )
    else:
        sample_choice = "Custom Input"

    preset_data = None
    if sample_choice != "Custom Input":
        selected_idx = int(sample_choice.split("#")[1].split(" ")[0])
        preset_data = sample_df.loc[selected_idx].to_dict()

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", ["Female", "Male"], index=0 if not preset_data else (0 if preset_data["gender"]=="Female" else 1))
            senior = st.selectbox("Senior Citizen", [0, 1], index=0 if not preset_data else int(preset_data["SeniorCitizen"]))
            partner = st.selectbox("Partner", ["Yes", "No"], index=0 if not preset_data else (0 if preset_data["Partner"]=="Yes" else 1))
            dependents = st.selectbox("Dependents", ["Yes", "No"], index=1 if not preset_data else (0 if preset_data["Dependents"]=="Yes" else 1))

        with c2:
            st.markdown("**Account & Billing**")
            tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12 if not preset_data else int(preset_data["tenure"]))
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0 if not preset_data else ["Month-to-month", "One year", "Two year"].index(preset_data["Contract"]))
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if not preset_data else (0 if preset_data["PaperlessBilling"]=="Yes" else 1))
            payment = st.selectbox("Payment Method", cat_options["PaymentMethod"], index=0 if not preset_data else cat_options["PaymentMethod"].index(preset_data["PaymentMethod"]))
            monthly = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=70.0 if not preset_data else float(preset_data["MonthlyCharges"]))
            total = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(tenure * monthly) if not preset_data else float(preset_data["TotalCharges"]))

        with c3:
            st.markdown("**Subscribed Services**")
            phone = st.selectbox("Phone Service", ["Yes", "No"], index=0 if not preset_data else (0 if preset_data["PhoneService"]=="Yes" else 1))
            multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=0 if not preset_data else ["No", "Yes", "No phone service"].index(preset_data["MultipleLines"]))
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=1 if not preset_data else ["DSL", "Fiber optic", "No"].index(preset_data["InternetService"]))
            security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=0 if not preset_data else ["No", "Yes", "No internet service"].index(preset_data["OnlineSecurity"]))
            backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=0 if not preset_data else ["No", "Yes", "No internet service"].index(preset_data["OnlineBackup"]))
            protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=0 if not preset_data else ["No", "Yes", "No internet service"].index(preset_data["DeviceProtection"]))
            tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=0 if not preset_data else ["No", "Yes", "No internet service"].index(preset_data["TechSupport"]))
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=0 if not preset_data else ["No", "Yes", "No internet service"].index(preset_data["StreamingTV"]))
            movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=0 if not preset_data else ["No", "Yes", "No internet service"].index(preset_data["StreamingMovies"]))

        submitted = st.form_submit_button("🔮 Predict Customer Churn", use_container_width=True)

    if submitted:
        input_data = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": tech,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }

        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)
        
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])

        churn_prob = float(active_model.predict_proba(input_encoded)[0, 1])
        prediction = int(churn_prob >= 0.5)

        st.markdown("---")
        st.subheader("3. Prediction Results & Model Explanation")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown(f"**Evaluated by**: `{selected_model_label}`")
            st.metric("Churn Probability", f"{churn_prob:.1%}")
            st.progress(churn_prob)
            if prediction == 1:
                st.markdown('<div class="churn-badge-high">⚠️ HIGH RISK OF CHURN</div>', unsafe_allow_html=True)
                st.warning("Recommendation: Proactive intervention required. Offer contract extension incentives or plan optimization.")
            else:
                st.markdown('<div class="churn-badge-low">✅ LOW RISK / RETAINED</div>', unsafe_allow_html=True)
                st.success("Recommendation: Customer is stable. Standard relationship maintenance and periodic check-ins.")

        with res_col2:
            st.markdown(f"#### Local SHAP Explanation ({model_family})")
            st.caption("Shows how each attribute pushed churn risk higher (red) or lower (blue).")
            
            try:
                single_shap_exp = get_customer_shap_explanation(active_model, selected_model_key, input_encoded)
                fig, ax = plt.subplots(figsize=(8, 4.5))
                shap.plots.waterfall(single_shap_exp, max_display=8, show=False)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"SHAP explanation computation error: {e}")
