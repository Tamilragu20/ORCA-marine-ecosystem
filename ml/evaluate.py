"""
ORCA Model Evaluation Module
Computes detailed classification metrics, per-class performance, confusion matrices,
feature importances, and outputs evaluation reports and figures.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from ml.train import prepare_train_test_data, FEATURE_COLS, TARGET_COL

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def evaluate_orca_models():
    print("=" * 65)
    print("ORCA MACHINE LEARNING: COMPREHENSIVE EVALUATION")
    print("=" * 65)

    # Load artifacts
    classifier_path = os.path.join(MODELS_DIR, "orca_classifier.joblib")
    scaler_path = os.path.join(MODELS_DIR, "orca_scaler.joblib")
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")

    if not os.path.exists(classifier_path):
        raise FileNotFoundError("Model artifacts not found. Run ml/train.py first.")

    # Ensure scikit-learn Cython loss extensions are mapped in sys.modules across environments
    import sys
    try:
        import sklearn._loss._loss as _c_loss
        if "_loss" not in sys.modules:
            sys.modules["_loss"] = _c_loss
    except (ImportError, AttributeError):
        pass

    model = joblib.load(classifier_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = prepare_train_test_data()

    class_names = ["Healthy", "Moderate Stress", "High Stress"]
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # 1. Classification Metrics
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    auc_score = roc_auc_score(y_test, y_prob, multi_class="ovr")

    clf_report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    clf_report_text = classification_report(y_test, y_pred, target_names=class_names)
    print(f"\nModel: {meta.get('best_model_name')}")
    print("\nClassification Report:\n", clf_report_text)

    # 2. Confusion Matrix Figure
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar=False)
    # Add percentage annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j + 0.5, i + 0.7, f"({cm_norm[i, j]*100:.1f}%)", ha="center", va="center", color="#475569", fontsize=9)
    plt.title(f"Confusion Matrix: {meta.get('best_model_name')} on Marine Test Set", pad=15)
    plt.xlabel("Predicted Ecosystem Condition")
    plt.ylabel("Ground-Truth Scientific Condition")
    plt.tight_layout()
    p_cm = os.path.join(FIG_DIR, "09_confusion_matrix.png")
    plt.savefig(p_cm, dpi=200)
    plt.close()
    print(f"Saved: {p_cm}")

    # 3. Model Comparison Metrics Bar Chart
    comp = meta.get("model_comparison", {})
    if comp:
        df_comp = pd.DataFrame(comp).T.reset_index().rename(columns={"index": "Model"})
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df_comp))
        width = 0.20
        
        ax.bar(x - 1.5*width, df_comp["test_accuracy"], width, label="Accuracy", color="#0284c7")
        ax.bar(x - 0.5*width, df_comp["test_precision"], width, label="Precision", color="#0d9488")
        ax.bar(x + 0.5*width, df_comp["test_recall"], width, label="Recall", color="#6366f1")
        ax.bar(x + 1.5*width, df_comp["test_f1"], width, label="Macro F1", color="#f59e0b")
        
        ax.set_ylabel("Metric Score (0.0 - 1.0)")
        ax.set_title("Comparison of Machine Learning Models on Marine Ecosystem Test Set", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(df_comp["Model"])
        ax.set_ylim(0.5, 1.05)
        ax.legend(loc="lower right")
        plt.tight_layout()
        p_comp = os.path.join(FIG_DIR, "10_model_comparison_metrics.png")
        plt.savefig(p_comp, dpi=200)
        plt.close()
        print(f"Saved: {p_comp}")

    # 4. Feature Importance Chart
    importances = meta.get("feature_importances", {})
    if importances:
        feat_df = pd.DataFrame(list(importances.items()), columns=["Feature", "Importance"])
        feat_df = feat_df.sort_values(by="Importance", ascending=True).tail(12)

        plt.figure(figsize=(10, 6))
        plt.barh(feat_df["Feature"], feat_df["Importance"], color="#0284c7")
        plt.title(f"Top 12 Contributing Features ({meta.get('best_model_name')})", pad=12)
        plt.xlabel("Gini / Mean Decrease in Impurity")
        plt.tight_layout()
        p_imp = os.path.join(FIG_DIR, "11_feature_importance.png")
        plt.savefig(p_imp, dpi=200)
        plt.close()
        print(f"Saved: {p_imp}")

    # 5. Multi-Class ROC Curves
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    colors = ["#10b981", "#f59e0b", "#ef4444"]
    for i, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"{cname} (AUC = {roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Multi-Class ROC Curves (One-vs-Rest): {meta.get('best_model_name')}", pad=12)
    plt.legend(loc="lower right")
    plt.tight_layout()
    p_roc = os.path.join(FIG_DIR, "12_roc_curves.png")
    plt.savefig(p_roc, dpi=200)
    plt.close()
    print(f"Saved: {p_roc}")

    # Save detailed evaluation report markdown
    eval_report_path = os.path.join(REPORTS_DIR, "model_evaluation_report.md")
    with open(eval_report_path, "w", encoding="utf-8") as f:
        f.write("# ORCA Model Evaluation & Benchmarking Report\n\n")
        f.write(f"- **Champion Model Selected**: {meta.get('best_model_name')}\n")
        f.write(f"- **Training Dataset**: {len(y_train):,} historical observations (Years 2000–2016)\n")
        f.write(f"- **Validation Dataset**: {len(y_val):,} observations (Years 2015–2016)\n")
        f.write(f"- **Evaluation Dataset**: {len(y_test):,} temporal holdout observations (Years 2017–2021) from California Current Ecosystem (CalCOFI / OBIS)\n")
        f.write(f"- **Test Accuracy**: {acc:.4f}\n")
        f.write(f"- **Test Macro Precision**: {prec_macro:.4f}\n")
        f.write(f"- **Test Macro Recall**: {rec_macro:.4f}\n")
        f.write(f"- **Test Macro F1-Score**: {f1_macro:.4f}\n")
        f.write(f"- **Multi-Class ROC-AUC (OvR)**: {auc_score:.4f}\n\n")
        f.write("### Per-Class Detailed Performance\n\n")
        f.write("| Ecosystem Condition | Precision | Recall | F1-Score | Support |\n")
        f.write("|---|---|---|---|---|\n")
        for cname in class_names:
            cstats = clf_report_dict[cname]
            f.write(f"| **{cname}** | {cstats['precision']:.3f} | {cstats['recall']:.3f} | {cstats['f1-score']:.3f} | {int(cstats['support'])} |\n")
        f.write("\n### Candidate Model Benchmark\n\n")
        f.write("| Model | 5-Fold CV Macro F1 | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for m_name, m_metrics in comp.items():
            f.write(f"| {m_name} | {m_metrics['cv_macro_f1_mean']:.4f} | {m_metrics['test_accuracy']:.4f} | {m_metrics['test_precision']:.4f} | {m_metrics['test_recall']:.4f} | {m_metrics['test_f1']:.4f} | {m_metrics['test_roc_auc']:.4f} |\n")
        
        f.write("\n### Scientific Explainability & Contributing Factors\n\n")
        f.write("Top environmental predictors ranked by tree importance:\n")
        for f_name, f_imp in list(importances.items())[:8]:
            f.write(f"- **{f_name}**: {f_imp * 100:.2f}%\n")
        
        f.write("\n### Limitations & Methodological Integrity\n")
        f.write("- **Observational Scope**: Sensor profiles reflect the California Current System; extrapolating to Arctic or tropical coral reefs requires regional calibration.\n")
        f.write("- **Correlation vs. Causation**: ML features reflect statistical associations with ecological distress, not laboratory-proven causality.\n")
        f.write("- **Sampling Depth**: Data density is highest in the upper 200m euphotic zone where photosynthetic activity and pelagic fisheries occur.\n")

    print(f"\nModel evaluation report written to {eval_report_path}")
    return clf_report_dict


if __name__ == "__main__":
    evaluate_orca_models()
