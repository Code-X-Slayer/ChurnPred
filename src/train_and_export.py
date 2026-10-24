import json
from pathlib import Path
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

# Resolve base paths relative to script location
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def resolve_data_path(filename: str) -> Path:
    """Finds target data file in data/ directory or repo root."""
    primary = DATA_DIR / filename
    return primary if primary.exists() else (BASE_DIR / filename)


def train_and_export():
    # Ensure destination directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load customer churn dataset
    dataset_path = resolve_data_path("dataset.csv")
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # Impute missing TotalCharges for new customers (tenure = 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)

    # Map target label to binary integers (1 = Churn, 0 = Retained)
    df["Churn_Binary"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Separate feature matrix and target series
    feature_df = df.drop(columns=["customerID", "Churn", "Churn_Binary"], errors="ignore")
    y = df["Churn_Binary"]

    # Define continuous and categorical feature names
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [c for c in feature_df.columns if c not in num_cols]

    # Cache unique category options for Streamlit UI dropdowns
    cat_options = {col: sorted(feature_df[col].dropna().unique().tolist()) for col in cat_cols}

    # One-hot encode categorical features (drop first category to avoid multicollinearity)
    X_encoded = pd.get_dummies(feature_df, drop_first=True)
    feature_names = X_encoded.columns.tolist()

    # Stratified 80/20 train/test split to preserve class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize numerical features using training set statistics
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    # Apply SMOTE strictly on training partition to prevent data leakage
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    # Log class distribution shifts across partitions
    print(f"Training distribution before SMOTE: Non-Churn = {sum(y_train == 0)}, Churn = {sum(y_train == 1)}")
    print(f"Training distribution after SMOTE:  Non-Churn = {sum(y_train_smote == 0)}, Churn = {sum(y_train_smote == 1)}")
    print(f"Test set (untouched):              Non-Churn = {sum(y_test == 0)}, Churn = {sum(y_test == 1)}")

    # Model factory definitions with fixed random seed
    model_factories = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "XGBoost": lambda: XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric="logloss", random_state=42),
        "LightGBM": lambda: LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)
    }

    results = []
    trained_models = {}

    # Train and evaluate each algorithm under Baseline and SMOTE settings
    for name, factory in model_factories.items():
        print(f"Training {name}...")

        # 1. Baseline Model (No SMOTE)
        model_base = factory()
        model_base.fit(X_train_scaled, y_train)
        y_pred_b = model_base.predict(X_test_scaled)
        y_proba_b = model_base.predict_proba(X_test_scaled)[:, 1]

        # Record baseline test metrics
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

        # 2. SMOTE-Trained Model
        model_smote = factory()
        model_smote.fit(X_train_smote, y_train_smote)
        y_pred_s = model_smote.predict(X_test_scaled)
        y_proba_s = model_smote.predict_proba(X_test_scaled)[:, 1]

        # Record SMOTE test metrics
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

    # Format benchmark table for logging
    results_df = pd.DataFrame(results)
    print("\n--- Model Benchmark Results ---")
    print(results_df.to_string(index=False))

    # Save benchmark metrics to JSON file
    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    # Designate primary recommended production model
    best_model = trained_models["XGBoost_smote"]

    # Sample representative test profiles for UI quick-testing
    raw_test_indices = X_test.index
    raw_test_customers = df.loc[raw_test_indices].copy()
    sample_customers = pd.concat([
        raw_test_customers[raw_test_customers["Churn"] == "Yes"].head(5),
        raw_test_customers[raw_test_customers["Churn"] == "No"].head(5)
    ])
    sample_path = DATA_DIR / "sample_customers.csv"
    sample_customers.to_csv(sample_path, index=False)

    # Package all runtime artifacts, preprocessor, and metadata
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
    model_artifact_path = MODELS_DIR / "model_artifacts.joblib"
    joblib.dump(artifacts, model_artifact_path)
    print(f"\nArtifacts successfully exported to '{model_artifact_path}' and '{metrics_path}'!")


if __name__ == "__main__":
    train_and_export()
