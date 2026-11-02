#!/usr/bin/env python
"""
TechPulse Existing Model Validator
Loads the EXISTING models (not v2) and evaluates them on the same
stratified 80/20 split used in train_model.py.

Usage: python scripts/validate_model.py
Reports saved to: reports/existing_model_evaluation.json
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "job_role_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "skill_encoder.pkl"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("TechPulse Existing Model Validator")
print("=" * 60)
print(f"scikit-learn version: {sklearn.__version__}")

# Check for sklearn version mismatch warning
print("\n[!] Note: The existing model was pickled with sklearn 1.6.1.")
print(f"    Current sklearn: {sklearn.__version__}")
if sklearn.__version__ != "1.6.1":
    print("    ⚠ VERSION MISMATCH — model may produce warnings on load.")
    print("    Consider retraining with scripts/train_model.py to generate _v2 models.")
else:
    print("    ✓ Versions match — no compatibility issues expected.")

# ─── 1. Load models ───────────────────────────────────────────────────────────
print("\n[1/5] Loading existing models...")
try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print(f"    Loaded model: {MODEL_PATH.name}")
    print(f"    Model type: {type(model).__name__}")
    print(f"    Encoder classes: {len(encoder.classes_)}")
    print(f"    Model classes: {list(model.classes_)}")
except Exception as e:
    print(f"    ✗ Failed to load models: {e}")
    sys.exit(1)

# ─── 2. Load & prepare data ──────────────────────────────────────────────────
print("\n[2/5] Loading and preparing dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df.drop_duplicates(subset=["job_id"]).copy()

def parse_skills(s):
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [x.strip().lower() for x in str(s).split("|") if x.strip()]

df["parsed_skills"] = df["skills"].apply(parse_skills)
df = df[df["parsed_skills"].map(len) > 0].copy()
df = df[df["job_role"].notna()].copy()
print(f"    Dataset rows: {len(df)}")
print(f"    Class distribution:")
for role, cnt in df["job_role"].value_counts().items():
    print(f"      {role:<45} {cnt:>4}")

# ─── 3. Encode & split (same as train_model.py: random_state=42, 80/20) ──────
print("\n[3/5] Encoding and splitting (stratified 80/20, random_state=42)...")
X = encoder.transform(df["parsed_skills"].tolist())
y = df["job_role"].astype(str).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"    Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ─── 4. Evaluate existing model ──────────────────────────────────────────────
print("\n[4/5] Evaluating existing model...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
macro_f1 = f1_score(y_test, y_pred_test, average="macro")
weighted_f1 = f1_score(y_test, y_pred_test, average="weighted")
macro_prec = precision_score(y_test, y_pred_test, average="macro", zero_division=0)
macro_rec = recall_score(y_test, y_pred_test, average="macro", zero_division=0)
overfit_gap = train_acc - test_acc

print(f"    Train accuracy:   {train_acc:.4f}")
print(f"    Test  accuracy:   {test_acc:.4f}")
print(f"    Overfit gap:      {overfit_gap:.4f}" + (" ⚠ OVERFIT" if overfit_gap > 0.15 else ""))
print(f"    Macro  F1:        {macro_f1:.4f}")
print(f"    Weighted F1:      {weighted_f1:.4f}")
print(f"    Macro Precision:  {macro_prec:.4f}")
print(f"    Macro Recall:     {macro_rec:.4f}")

clf_report = classification_report(y_test, y_pred_test, output_dict=True)
cm = confusion_matrix(y_test, y_pred_test, labels=list(model.classes_)).tolist()

# ─── 5. Test with sample inputs ──────────────────────────────────────────────
print("\n[5/5] Testing with sample inputs...")
sample_inputs = [
    ["python", "machine-learning", "deep-learning", "pytorch", "sql"],
    ["kubernetes", "docker", "aws", "terraform", "ci-cd", "linux"],
    ["javascript", "react", "node.js", "typescript", "html-css"],
    ["product-management", "project-management", "communication"]
]

sample_results = []
for skills in sample_inputs:
    try:
        X_sample = encoder.transform([skills])
        pred = model.predict(X_sample)[0]
        proba = model.predict_proba(X_sample)[0]
        top_prob = round(float(proba.max()) * 100, 1)
        sample_results.append({
            "skills": skills,
            "predicted_role": pred,
            "confidence_pct": top_prob
        })
        print(f"    {', '.join(skills[:3])}... → {pred} ({top_prob}%)")
    except Exception as e:
        print(f"    ✗ Prediction failed for {skills}: {e}")
        sample_results.append({"skills": skills, "error": str(e)})

# ─── Save report ──────────────────────────────────────────────────────────────
report = {
    "model_file": str(MODEL_PATH),
    "encoder_file": str(ENCODER_PATH),
    "model_type": type(model).__name__,
    "sklearn_version_current": sklearn.__version__,
    "sklearn_version_at_pickle": "1.6.1",
    "version_mismatch": sklearn.__version__ != "1.6.1",
    "dataset_rows": int(len(df)),
    "train_size": int(X_train.shape[0]),
    "test_size": int(X_test.shape[0]),
    "train_accuracy": round(train_acc, 4),
    "test_accuracy": round(test_acc, 4),
    "overfit_gap": round(overfit_gap, 4),
    "overfit_warning": overfit_gap > 0.15,
    "macro_f1": round(macro_f1, 4),
    "weighted_f1": round(weighted_f1, 4),
    "macro_precision": round(macro_prec, 4),
    "macro_recall": round(macro_rec, 4),
    "classification_report": clf_report,
    "confusion_matrix": cm,
    "classes": list(model.classes_),
    "sample_predictions": sample_results
}

report_path = REPORTS_DIR / "existing_model_evaluation.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n    ✓ Report saved: {report_path}")
print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
print(f"  Test accuracy:  {test_acc:.4f}")
print(f"  Weighted F1:    {weighted_f1:.4f}")
print(f"  Report:         {report_path}")
print("=" * 60)
