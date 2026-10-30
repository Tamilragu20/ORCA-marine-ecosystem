"""
ORCA Machine Learning Training Module
Trains multiple models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting),
trains Isolation Forest for anomaly detection and K-Means for unsupervised oceanographic grouping,
and serializes artifacts to models/.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "orca_integrated_ecosystem.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Scientifically defensible predictive features (strictly excluding direct target constituents to prevent target leakage)
FEATURE_COLS = [
    "depth",
    "temperature",
    "salinity",
    "chlorophyll",
    "phosphate",
    "nitrate",
    "silicate",
    "temp_depth_gradient",
    "sigma_theta_proxy",
    "np_ratio",
    "primary_productivity_proxy",
    "seasonal_sin",
    "seasonal_cos",
    "species_richness",
    "shannon_diversity",
    "latitude",
    "longitude"
]

TARGET_COL = "ecosystem_condition" # 0: Healthy, 1: Moderate Stress, 2: High Stress


def prepare_train_test_data(
    test_year_threshold: int = 2017,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list]:
    """
    Loads dataset and partitions using strict Temporal Forward-Holdout splitting:
    - Train: historical baseline years 2000–2016 (approx 77% of data)
    - Test: forward holdout years 2017–2021 (approx 23% of data)
    Fits StandardScaler strictly on the training partition to prevent any data leakage.
    """
    df = pd.read_csv(DATA_PATH)
    
    # Ensure year is integer
    df["year"] = pd.to_datetime(df["date"]).dt.year.astype(int)
    
    # Drop rows with NaNs in feature or target columns
    df_clean = df.dropna(subset=FEATURE_COLS + [TARGET_COL, "year"]).copy()

    # Temporal split
    train_mask = df_clean["year"] < test_year_threshold
    test_mask = df_clean["year"] >= test_year_threshold

    df_train_full = df_clean[train_mask].copy()
    df_test = df_clean[test_mask].copy()

    # Split train into historical core (2000-2014) and validation (2015-2016)
    val_mask = df_train_full["year"] >= (test_year_threshold - 2)
    df_train = df_train_full[~val_mask].copy()
    df_val = df_train_full[val_mask].copy()

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train[TARGET_COL].values.astype(int)

    X_val = df_val[FEATURE_COLS].values
    y_val = df_val[TARGET_COL].values.astype(int)

    X_test = df_test[FEATURE_COLS].values
    y_test = df_test[TARGET_COL].values.astype(int)

    # Standardize features using Train statistics ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler, FEATURE_COLS


def train_and_compare_models() -> Dict[str, Any]:
    """Trains candidate classifiers, cross-validates, evaluates on test set, and serializes best model."""
    print("=" * 65)
    print("ORCA MACHINE LEARNING: TRAINING & MODEL COMPARISON")
    print("=" * 65)

    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = prepare_train_test_data()
    print(f"Dataset Split Sizes: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")

    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42)
    }

    results = {}
    best_model_name = None
    best_cv_f1 = -1.0
    fitted_models = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in candidate_models.items():
        print(f"\n--- Training {name} ---")
        # 5-fold Cross-Validation on Train set
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
        print(f"5-Fold CV Macro F1: {cv_mean:.4f} (+/- {cv_std:.4f})")

        # Fit on Train
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Predict on Test
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr") if y_prob is not None else 0.0
        except Exception:
            auc = 0.0

        print(f"Test Accuracy:  {acc:.4f}")
        print(f"Test Precision: {prec:.4f}")
        print(f"Test Recall:    {rec:.4f}")
        print(f"Test F1-Score:  {f1:.4f}")
        print(f"Test ROC-AUC:   {auc:.4f}")

        results[name] = {
            "cv_macro_f1_mean": round(cv_mean, 4),
            "cv_macro_f1_std": round(cv_std, 4),
            "test_accuracy": round(float(acc), 4),
            "test_precision": round(float(prec), 4),
            "test_recall": round(float(rec), 4),
            "test_f1": round(float(f1), 4),
            "test_roc_auc": round(float(auc), 4),
        }

        # Select champion model based on cross-validation performance on training set
        if cv_mean > best_cv_f1:
            best_cv_f1 = cv_mean
            best_model_name = name

    print(f"\n>>> Best Performing Model (by 5-fold CV F1): {best_model_name} (CV Macro F1 = {best_cv_f1:.4f})")
    best_classifier = fitted_models[best_model_name]

    # -------------------------------------------------------------
    # Train Unsupervised Models (Isolation Forest & K-Means)
    # -------------------------------------------------------------
    print("\n--- Training Unsupervised Models ---")
    print("Training Isolation Forest for oceanographic anomaly detection...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)

    print("Training K-Means (k=3) for unsupervised ecosystem clustering...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_train)

    # -------------------------------------------------------------
    # Feature Importances
    # -------------------------------------------------------------
    if hasattr(best_classifier, "feature_importances_"):
        raw_importances = best_classifier.feature_importances_
        importance_dict = {
            feat: round(float(imp), 4) 
            for feat, imp in sorted(zip(feature_names, raw_importances), key=lambda x: x[1], reverse=True)
        }
    elif hasattr(best_classifier, "coef_"):
        raw_coefs = np.mean(np.abs(best_classifier.coef_), axis=0)
        norm_coefs = raw_coefs / np.sum(raw_coefs)
        importance_dict = {
            feat: round(float(imp), 4)
            for feat, imp in sorted(zip(feature_names, norm_coefs), key=lambda x: x[1], reverse=True)
        }
    else:
        importance_dict = {}

    # -------------------------------------------------------------
    # Serialize Model Artifacts
    # -------------------------------------------------------------
    print("\nSaving trained models and scalers...")
    classifier_path = os.path.join(MODELS_DIR, "orca_classifier.joblib")
    scaler_path = os.path.join(MODELS_DIR, "orca_scaler.joblib")
    iso_path = os.path.join(MODELS_DIR, "anomaly_detector.joblib")
    kmeans_path = os.path.join(MODELS_DIR, "clustering_model.joblib")
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")

    joblib.dump(best_classifier, classifier_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(iso_forest, iso_path)
    joblib.dump(kmeans, kmeans_path)

    metadata = {
        "best_model_name": best_model_name,
        "features": feature_names,
        "classes": {0: "Healthy", 1: "Moderate Stress", 2: "High Stress"},
        "feature_importances": importance_dict,
        "model_comparison": results,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "target_col": TARGET_COL,
        "random_state": 42
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artifacts successfully saved to {MODELS_DIR}")
    return metadata


if __name__ == "__main__":
    train_and_compare_models()
