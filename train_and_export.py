import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_and_export():
    print("Loading dataset...")
    df = pd.read_csv("dataset.csv")

    # Clean TotalCharges (11 blank spaces from tenure=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)

    df["Churn_Binary"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    feature_df = df.drop(columns=["customerID", "Churn", "Churn_Binary"])
    y = df["Churn_Binary"]

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [c for c in feature_df.columns if c not in num_cols]

    cat_options = {col: sorted(feature_df[col].dropna().unique().tolist()) for col in cat_cols}

    X_encoded = pd.get_dummies(feature_df, drop_first=True)
    feature_names = X_encoded.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    # SMOTE applied strictly to training data to avoid data leakage
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    print(f"Training distribution before SMOTE: Non-Churn = {sum(y_train==0)}, Churn = {sum(y_train==1)}")
    print(f"Training distribution after SMOTE:  Non-Churn = {sum(y_train_smote==0)}, Churn = {sum(y_train_smote==1)}")
    print(f"Test set (untouched):              Non-Churn = {sum(y_test==0)}, Churn = {sum(y_test==1)}")

    model_factories = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "XGBoost": lambda: XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric="logloss", random_state=42),
        "LightGBM": lambda: LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)
    }

    results = []
    trained_models = {}

    for name, factory in model_factories.items():
        print(f"Training {name}...")
        model_base = factory()
        model_base.fit(X_train_scaled, y_train)
        y_pred_b = model_base.predict(X_test_scaled)
        y_proba_b = model_base.predict_proba(X_test_scaled)[:, 1]

        results.append({
            "Model": name,
            "Setting": "Baseline (No SMOTE)",
            "Accuracy": round(float(accuracy_score(y_test, y_pred_b)), 4),
            "Precision": round(float(precision_score(y_test, y_pred_b)), 4),
            "Recall": round(float(recall_score(y_test, y_pred_b)), 4),
            "F1-Score": round(float(f1_score(y_test, y_pred_b)), 4),
            "ROC-AUC": round(float(roc_auc_score(y_test, y_proba_b)), 4),
        })
        trained_models[f"{name}_base"] = model_base

        model_smote = factory()
        model_smote.fit(X_train_smote, y_train_smote)
        y_pred_s = model_smote.predict(X_test_scaled)
        y_proba_s = model_smote.predict_proba(X_test_scaled)[:, 1]

        results.append({
            "Model": name,
            "Setting": "With SMOTE",
            "Accuracy": round(float(accuracy_score(y_test, y_pred_s)), 4),
            "Precision": round(float(precision_score(y_test, y_pred_s)), 4),
            "Recall": round(float(recall_score(y_test, y_pred_s)), 4),
            "F1-Score": round(float(f1_score(y_test, y_pred_s)), 4),
            "ROC-AUC": round(float(roc_auc_score(y_test, y_proba_s)), 4),
        })
        trained_models[f"{name}_smote"] = model_smote

    results_df = pd.DataFrame(results)
    print("\n--- Model Benchmark Results ---")
    print(results_df.to_string(index=False))

    with open("metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    best_model = trained_models["XGBoost_smote"]

    # Sample a few test records for quick UI testing
    raw_test_indices = X_test.index
    raw_test_customers = df.loc[raw_test_indices].copy()
    sample_customers = pd.concat([
        raw_test_customers[raw_test_customers["Churn"] == "Yes"].head(5),
        raw_test_customers[raw_test_customers["Churn"] == "No"].head(5)
    ])
    sample_customers.to_csv("sample_customers.csv", index=False)

    artifacts = {
        "best_model": best_model,
        "best_model_name": "XGBoost (With SMOTE)",
        "all_models": trained_models,
        "scaler": scaler,
        "feature_names": feature_names,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_options": cat_options,
        "results_df": results_df,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "y_train": y_train,
    }
    joblib.dump(artifacts, "model_artifacts.joblib")
    print("\nArtifacts successfully exported to 'model_artifacts.joblib' and 'metrics.json'!")

if __name__ == "__main__":
    train_and_export()
