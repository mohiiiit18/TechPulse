 #!/usr/bin/env python
"""
TechPulse ML — standalone model evaluation.

Evaluates ALL saved models (production + v3 candidate) on the SAME
stratified, group-aware test split and writes a comparison report.

Usage: python -m src.evaluate_model
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from src.feature_engineering import prepare_features
from src.model_utils import REPORTS_DIR, evaluate, format_confusion_png, make_split

CLEAN_PATH = HERE / "data" / "processed" / "techpulse_cleaned.csv"
MODELS_DIR = HERE / "models"

CANDIDATES = [
    ("production (v2 RandomForest)", MODELS_DIR / "job_role_model_v2.pkl", MODELS_DIR / "skill_encoder_v2.pkl"),
    ("v3 candidate", MODELS_DIR / "job_role_model_v3.pkl", MODELS_DIR / "skill_encoder_v3.pkl"),
]


def main() -> int:
    df = pd.read_csv(CLEAN_PATH, low_memory=False)
    rows = []
    base_encoder = None

    for label, mp, ep in CANDIDATES:
        if not (mp.exists() and ep.exists()):
            print(f"skip {label}: artifacts missing")
            continue
        clf = joblib.load(mp)
        encoder = joblib.load(ep)
        if base_encoder is None:
            # Build split once with the first candidate's encoder so all models
            # share the identical train/test rows.
            X, y, groups, _, _ = prepare_features(df, encoder=encoder)
            train_idx, test_idx = make_split(X, y, groups)
        else:
            X, y, groups, _, _ = prepare_features(df, encoder=encoder)
        res = evaluate(clf, X[train_idx], y[train_idx], X[test_idx], y[test_idx])
        rows.append({"model": label,
                     **{k: v for k, v in res.items()
                        if k not in ("classification_report", "confusion_matrix", "classes")}})
        format_confusion_png(np.array(res["confusion_matrix"]), res["classes"],
                             REPORTS_DIR / f"confusion_matrix_{label.split()[0]}.png")
        print(f"{label}: acc={res['test_accuracy']} f1_macro={res['f1_macro']} "
              f"f1_weighted={res['f1_weighted']} overfit_gap={res['overfit_gap']}")

    out = REPORTS_DIR / "model_comparison_all.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
