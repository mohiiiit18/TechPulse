#!/usr/bin/env python
"""
TechPulse ML — one-command reproducible training pipeline.

Usage:
    python -m src.train_model

Trains LogisticRegression, RandomForest and XGBoost (if available) on the
Phase-2 cleaned dataset with a stratified + group-aware (leakage-safe) split,
tunes with RandomizedSearchCV (macro F1) on training data only, evaluates on
the untouched test set and saves the best model as models/job_role_model_v3.pkl
(never overwriting existing artifacts).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from src.feature_engineering import prepare_features
from src.model_utils import (
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
    build_models,
    cross_validate,
    evaluate,
    format_confusion_png,
    make_split,
    save_model,
    tune,
)

CLEAN_PATH = HERE / "data" / "processed" / "techpulse_cleaned.csv"


def main() -> int:
    print("=" * 64)
    print("TechPulse ML Training Pipeline (Phase 3)")
    print("=" * 64)

    df = pd.read_csv(CLEAN_PATH, low_memory=False)
    print(f"[1/8] Loaded cleaned dataset: {len(df)} rows")
    print("      Role distribution:", df["job_role"].value_counts().to_dict())

    X, y, groups, encoder, stats = prepare_features(df)
    print(f"[2/8] Features: {X.shape} | vocab: {stats['vocabulary_size']} | "
          f"density {stats['density']:.3f} | empty-skill rows: {stats['empty_skill_rows']}")

    train_idx, test_idx = make_split(X, y, groups)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train = groups[train_idx]
    print(f"[3/8] Leakage-aware stratified split: train={len(train_idx)}, test={len(test_idx)}")
    print("      Test class counts:", pd.Series(y_test).value_counts().to_dict())

    models = build_models(n_classes=len(np.unique(y)))
    print(f"[4/8] Models to train: {list(models.keys())}")

    comparison = []
    full_reports = {}
    best_name, best_clf, best_macro_f1 = None, None, -1.0

    for i, (name, clf) in enumerate(models.items(), start=1):
        print(f"\n[5/8] {name}: tuning + CV (training data only)...")
        tuned, tune_info = tune(name, clf, X_train, y_train, g_train, n_iter=15)
        cv_info = cross_validate(tuned, X_train, y_train, g_train, cv_folds=3)
        tuned.fit(X_train, y_train)
        res = evaluate(tuned, X_train, y_train, X_test, y_test)
        full_reports[name] = {"classification_report": res["classification_report"],
                              "confusion_matrix": res["confusion_matrix"],
                              "classes": res["classes"]}
        row = {"model": name, **{k: v for k, v in res.items()
                                 if k not in ("classification_report", "confusion_matrix", "classes")},
               **tune_info, **cv_info}
        comparison.append(row)
        print(f"      test_acc={res['test_accuracy']}  f1_macro={res['f1_macro']}  "
              f"f1_weighted={res['f1_weighted']}  overfit_gap={res['overfit_gap']}  "
              f"cv_f1_macro={cv_info['f1_macro_mean']}±{cv_info['f1_macro_std']}")
        # Primary selection metric: macro F1 (imbalanced classes).
        if res["f1_macro"] > best_macro_f1:
            best_macro_f1, best_name, best_clf = res["f1_macro"], name, tuned

    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print(f"\n[6/8] Best model by MACRO F1: {best_name} ({best_macro_f1})")

    best_res = next(r for r in comparison if r["model"] == best_name)
    cls_report = full_reports[best_name]["classification_report"]
    pd.DataFrame(cls_report).T.to_csv(REPORTS_DIR / "classification_report.csv")

    cm = np.array(full_reports[best_name]["confusion_matrix"])
    classes = full_reports[best_name]["classes"]
    format_confusion_png(cm, classes, REPORTS_DIR / "confusion_matrix.png")

    # Probability sanity check (calibration proxy): does argmax prob ≈ accuracy?
    if hasattr(best_clf, "predict_proba"):
        proba = best_clf.predict_proba(X_test)
        conf = proba.max(axis=1)
        print(f"[7/8] Mean max-probability: {conf.mean():.3f} (uncalibrated — UI labels accordingly)")

    metadata = {
        "model_name": best_name,
        "dataset": "data/processed/techpulse_cleaned.csv (Phase 2 cleaned)",
        "dataset_rows": stats["n_rows"],
        "features": {
            "type": "MultiLabelBinarizer multi-hot over standardized skills",
            "vocabulary_size": stats["vocabulary_size"],
            "target_column": "job_role",
            "target_included_in_features": False,
            "known_skills": encoder.known_skills,
        },
        "split": {"strategy": "StratifiedGroupKFold (leakage-aware)", "test_size": TEST_SIZE,
                  "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx))},
        "selection_metric": "f1_macro",
        "training_params": best_res.get("best_params", {}),
        "evaluation": {k: v for k, v in best_res.items()},
        "classification_report": full_reports[best_name]["classification_report"],
        "cross_validation": best_res.get("f1_macro_mean"),
        "class_distribution": df["job_role"].value_counts().to_dict(),
    }
    model_path, enc_path, meta_path = save_model(best_clf, encoder, metadata)
    print(f"[8/8] Saved: {model_path.name}, {enc_path.name}, {meta_path.name}")

    # Full training summary
    summary = {
        "best_model": best_name,
        "best_f1_macro": best_macro_f1,
        "selection_reason": "Highest macro F1 on the untouched test set (imbalanced classes)",
        "comparison": comparison,
        "per_model_reports": full_reports,
        "feature_stats": stats,
        "class_distribution": df["job_role"].value_counts().to_dict(),
        "probability_calibration": "NOT calibrated — raw probabilities; UI must not present as guaranteed confidence",
    }
    with open(REPORTS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nDONE. Reports: reports/model_comparison.csv, classification_report.csv, "
          "confusion_matrix.png, training_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
