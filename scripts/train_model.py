#!/usr/bin/env python
"""
TechPulse ML Training Pipeline
Trains and evaluates Logistic Regression and Random Forest classifiers
for tech job role prediction.

Usage: python scripts/train_model.py
Models saved to: models/ (only if new model beats existing one — saved as _v2)
Reports saved to: reports/
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed dataset.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("TechPulse ML Training Pipeline")
print("=" * 60)

# ─── 1. Load & clean data ────────────────────────────────────────────────────
print("\n[1/7] Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df.drop_duplicates(subset=["job_id"]).copy()
print(f"    Rows after dedup: {len(df)}")

# ─── 2. Parse skills ─────────────────────────────────────────────────────────
def parse_skills(s):
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [x.strip().lower() for x in str(s).split("|") if x.strip()]

df["parsed_skills"] = df["skills"].apply(parse_skills)
df = df[df["parsed_skills"].map(len) > 0].copy()
print(f"    Rows with ≥1 skill: {len(df)}")

# ─── 3. Target column ────────────────────────────────────────────────────────
df = df[df["job_role"].notna()].copy()
print(f"    Rows with job_role: {len(df)}")

# Class distribution
class_dist = df["job_role"].value_counts()
print("\n[2/7] Class distribution:")
for role, cnt in class_dist.items():
    pct = cnt / len(df) * 100
    print(f"    {role:<45} {cnt:>4} ({pct:.1f}%)")

# Check imbalance
max_ratio = class_dist.max() / class_dist.min()
print(f"\n    Imbalance ratio (max/min): {max_ratio:.2f}x")
if max_ratio > 3:
    print("    ⚠ Significant imbalance detected — using class_weight='balanced'")

# ─── 4. Feature encoding ─────────────────────────────────────────────────────
print("\n[3/7] Encoding features with MultiLabelBinarizer...")
encoder = MultiLabelBinarizer()
X = encoder.fit_transform(df["parsed_skills"].tolist())
y = df["job_role"].astype(str).to_numpy()
print(f"    Skill vocabulary size: {len(encoder.classes_)}")
print(f"    Feature matrix: {X.shape}")

# ─── 5. Stratified train/test split ──────────────────────────────────────────
print("\n[4/7] Stratified 80/20 split (random_state=42)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ─── 6. Train models ─────────────────────────────────────────────────────────
print("\n[5/7] Training models...")

models_to_train = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
    ),
}

# Optional XGBoost
try:
    import xgboost as xgb
    models_to_train["XGBoost"] = xgb.XGBClassifier(
        n_estimators=200, random_state=42, use_label_encoder=False,
        eval_metric="mlogloss", n_jobs=-1
    )
    print("    XGBoost found — including in comparison.")
except ImportError:
    print("    XGBoost not installed — skipping.")

# ─── 7. Evaluate each model ──────────────────────────────────────────────────
print("\n[6/7] Evaluating models...")

evaluation_results = {}
best_model_name = None
best_model = None
best_weighted_f1 = -1.0

for name, clf in models_to_train.items():
    print(f"\n  → Training {name}...")
    clf.fit(X_train, y_train)

    # Predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    macro_f1 = f1_score(y_test, y_pred_test, average="macro")
    weighted_f1 = f1_score(y_test, y_pred_test, average="weighted")
    macro_prec = precision_score(y_test, y_pred_test, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, y_pred_test, average="macro", zero_division=0)

    overfit_gap = train_acc - test_acc
    overfit_warning = overfit_gap > 0.15

    clf_report = classification_report(y_test, y_pred_test, output_dict=True)
    cm = confusion_matrix(y_test, y_pred_test, labels=list(clf.classes_)).tolist()

    evaluation_results[name] = {
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "overfit_gap": round(overfit_gap, 4),
        "overfit_warning": overfit_warning,
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "classification_report": clf_report,
        "confusion_matrix": cm,
        "classes": list(clf.classes_)
    }

    print(f"    Train accuracy:   {train_acc:.4f}")
    print(f"    Test  accuracy:   {test_acc:.4f}")
    print(f"    Overfit gap:      {overfit_gap:.4f}" + (" ⚠ OVERFIT" if overfit_warning else ""))
    print(f"    Macro  F1:        {macro_f1:.4f}")
    print(f"    Weighted F1:      {weighted_f1:.4f}")
    print(f"    Macro Precision:  {macro_prec:.4f}")
    print(f"    Macro Recall:     {macro_rec:.4f}")

    if weighted_f1 > best_weighted_f1:
        best_weighted_f1 = weighted_f1
        best_model_name = name
        best_model = clf

print(f"\n  🏆 Best model: {best_model_name} (Weighted F1 = {best_weighted_f1:.4f})")

# ─── 8. Save best model (v2 only — do NOT overwrite existing) ────────────────
print("\n[7/7] Saving best model and encoder...")
model_v2_path = MODELS_DIR / "job_role_model_v2.pkl"
encoder_v2_path = MODELS_DIR / "skill_encoder_v2.pkl"
joblib.dump(best_model, model_v2_path)
joblib.dump(encoder, encoder_v2_path)
print(f"    Model saved: {model_v2_path}")
print(f"    Encoder saved: {encoder_v2_path}")
print("    NOTE: Existing job_role_model.pkl and skill_encoder.pkl were NOT overwritten.")

# ─── 9. Save evaluation reports ──────────────────────────────────────────────
# Full evaluation JSON
eval_output = {
    "best_model": best_model_name,
    "best_weighted_f1": round(best_weighted_f1, 4),
    "train_size": int(X_train.shape[0]),
    "test_size": int(X_test.shape[0]),
    "n_classes": int(len(set(y))),
    "class_distribution": class_dist.to_dict(),
    "imbalance_ratio": round(float(max_ratio), 2),
    "models": evaluation_results
}

eval_json_path = REPORTS_DIR / "model_evaluation.json"
with open(eval_json_path, "w") as f:
    json.dump(eval_output, f, indent=2)
print(f"    Evaluation JSON: {eval_json_path}")

# Classification report text
clf_report_txt_path = REPORTS_DIR / "classification_report.txt"
with open(clf_report_txt_path, "w") as f:
    f.write(f"TechPulse ML Evaluation — Best Model: {best_model_name}\n")
    f.write(f"Weighted F1: {best_weighted_f1:.4f}\n")
    f.write("=" * 60 + "\n\n")
    for model_name, res in evaluation_results.items():
        f.write(f"Model: {model_name}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Train Accuracy : {res['train_accuracy']:.4f}\n")
        f.write(f"Test  Accuracy : {res['test_accuracy']:.4f}\n")
        f.write(f"Overfit Gap    : {res['overfit_gap']:.4f}\n")
        f.write(f"Macro F1       : {res['macro_f1']:.4f}\n")
        f.write(f"Weighted F1    : {res['weighted_f1']:.4f}\n")
        f.write(f"Macro Precision: {res['macro_precision']:.4f}\n")
        f.write(f"Macro Recall   : {res['macro_recall']:.4f}\n\n")
        # Per-class metrics
        clf_rpt = res["classification_report"]
        f.write(f"Per-class metrics:\n")
        for cls in res["classes"]:
            if cls in clf_rpt:
                m = clf_rpt[cls]
                f.write(f"  {cls:<45} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1-score']:.3f} N={m['support']}\n")
        f.write("\n")

print(f"    Classification report: {clf_report_txt_path}")

# Confusion matrices as CSVs
for model_name, res in evaluation_results.items():
    cm_df = pd.DataFrame(
        res["confusion_matrix"],
        index=res["classes"],
        columns=res["classes"]
    )
    cm_path = REPORTS_DIR / f"confusion_matrix_{model_name.lower()}.csv"
    cm_df.to_csv(cm_path)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"  Best model:   {best_model_name}")
print(f"  Weighted F1:  {best_weighted_f1:.4f}")
print(f"  Model v2:     {model_v2_path}")
print(f"  Encoder v2:   {encoder_v2_path}")
print(f"  Reports dir:  {REPORTS_DIR}")
print("=" * 60)
