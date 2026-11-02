"""
TechPulse Phase 7 — Evaluate the saved salary model on the same split.

Generates: actual-vs-predicted plot, residual distribution, feature
importance, per-category (role / country) performance where sample sizes
permit. Reports saved to reports/.

Run: python src/evaluate_salary_model.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from src.salary_preprocessing import build_salary_dataset
from src.salary_feature_engineering import make_salary_features

REPORTS = HERE / "reports"


def main():
    meta = json.loads((HERE / "models" / "salary_model_metadata.json").read_text())
    model = joblib.load(HERE / "models" / "salary_prediction_model.pkl")
    prep = joblib.load(HERE / "models" / "salary_preprocessor.pkl")

    df = pd.read_csv(HERE / "data" / "processed" / "techpulse_cleaned.csv", low_memory=False)
    salary_df, _ = build_salary_dataset(df)
    X, y, groups, feature_names, enc = make_salary_features(salary_df)

    # Reconstruct the identical split (GroupShuffleSplit, seed 42) from training
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
    pred = model.predict(X_te)

    mae = mean_absolute_error(y_te, pred)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    r2 = r2_score(y_te, pred)
    print(f"Loaded model: {meta['model_name']} | test MAE={mae:.0f} RMSE={rmse:.0f} R2={r2:.3f}")

    # Per-category performance where sample sizes permit (>= 10 rows)
    seg = salary_df.iloc[te_idx][["job_role", "country"]].copy()
    seg["actual"] = y_te.values
    seg["pred"] = pred
    seg_out = {}
    for col in ["job_role", "country"]:
        for val, g in seg.groupby(col):
            if len(g) >= 10:
                seg_out[f"{col}={val}"] = {
                    "n": int(len(g)),
                    "mae": round(float(mean_absolute_error(g["actual"], g["pred"])), 0),
                }
    print("Segment performance (n>=10):")
    for k, v in seg_out.items():
        print(f"  {k}: n={v['n']} MAE={v['mae']:.0f}")

    # Plots
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_te, pred, alpha=0.5, s=18)
    lim = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
    ax.plot(lim, lim, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual salary (USD/yr)")
    ax.set_ylabel("Predicted salary (USD/yr)")
    ax.set_title(f"Actual vs Predicted — {meta['model_name']} (MAE={mae:,.0f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "salary_actual_vs_predicted.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_te.values - pred, bins=25, color="#4c8bf5", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual (actual − predicted, USD)")
    ax.set_title("Residual distribution (test split)")
    fig.tight_layout()
    fig.savefig(REPORTS / "salary_residuals.png", dpi=120)
    plt.close(fig)

    summary = {
        "model": meta["model_name"], "test_mae": round(float(mae), 1),
        "test_rmse": round(rmse, 1), "test_r2": round(float(r2), 4),
        "segments": seg_out,
        "feature_importance_top15": meta.get("feature_importance_top15", {}),
    }
    (REPORTS / "salary_model_evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: reports/salary_actual_vs_predicted.png, reports/salary_residuals.png, "
          f"reports/salary_model_evaluation.json")


if __name__ == "__main__":
    main()
