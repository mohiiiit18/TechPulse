"""
TechPulse Phase 7 — Train & evaluate salary prediction models.

Trains Linear Regression (baseline), RandomForest, GradientBoosting and
XGBoost (if installed) with a consistent group-aware 80/20 split and 5-fold
group-aware CV. Selects by test MAE (reliable generalization, not training
accuracy). Saves model + preprocessor + metadata under models/.

Run: python src/train_salary_model.py
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from src.salary_preprocessing import MODEL_CURRENCY, MODEL_PERIOD, PREPROCESSING_POLICY, build_salary_dataset
from src.salary_feature_engineering import make_salary_features

MODELS_DIR = HERE / "models"
RANDOM_STATE = 42


def evaluate(name, model, X_tr, y_tr, X_te, y_te, groups_tr, results):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    r2 = r2_score(y_te, pred)
    cv_pred = cross_val_predict(model, X_tr, y_tr, groups=groups_tr,
                                cv=GroupKFold(n_splits=5))
    cv_mae = mean_absolute_error(y_tr, cv_pred)
    results[name] = {
        "test_mae": round(float(mae), 1), "test_rmse": round(rmse, 1),
        "test_r2": round(float(r2), 4), "cv5_mae": round(float(cv_mae), 1),
        "train_mae": round(float(mean_absolute_error(y_tr, model.predict(X_tr))), 1),
    }
    print(f"  {name:<18} test MAE={mae:>8.0f}  RMSE={rmse:>9.0f}  R2={r2:.3f}  CV5 MAE={cv_mae:.0f}")
    return pred


def main():
    print("TechPulse Phase 7 — Salary model training")
    df = pd.read_csv(HERE / "data" / "processed" / "techpulse_cleaned.csv", low_memory=False)
    salary_df, prep_info = build_salary_dataset(df)
    print(f"Usable records: {prep_info['final_rows']} (trimmed {prep_info['rows_trimmed_out']} extreme outliers)")

    X, y, groups, feature_names, enc = make_salary_features(salary_df)
    idx = np.arange(len(X))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
    g_tr = groups.iloc[tr_idx]
    print(f"Train={len(X_tr)}, Test={len(X_te)}, features={X.shape[1]}")

    results: dict = {}
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, max_features=0.4,
            random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.9, colsample_bytree=0.8, random_state=RANDOM_STATE,
            eval_metric="mae")
        print("XGBoost available — included in comparison.")
    except ImportError:
        print("XGBoost not installed — skipping (not required).")

    print("\nModel comparison (test split):")
    for name, model in models.items():
        evaluate(name, model, X_tr, y_tr, X_te, y_te, g_tr, results)

    # Select by test MAE (generalization), tie-break by CV MAE
    best_name = min(results, key=lambda n: (results[n]["test_mae"], results[n]["cv5_mae"]))
    best_model = models[best_name]
    print(f"\nSelected: {best_name} (lowest test MAE)")

    # Prediction range = empirical residual quantiles on the test set,
    # clearly labelled as error-based (not a statistical CI).
    best_pred = best_model.predict(X_te)
    resid = (y_te - best_pred).values
    range_lo, range_hi = np.quantile(resid, [0.1, 0.9])
    print(f"Empirical 10-90% residual range: [{range_lo:+.0f}, {range_hi:+.0f}]")

    # Feature importance for tree model
    importance = {}
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        top = sorted(zip(feature_names, imp), key=lambda t: -t[1])[:15]
        importance = {k: round(float(v), 4) for k, v in top}

    # Persist
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "salary_prediction_model.pkl")
    joblib.dump({"encoder": enc, "feature_names": feature_names},
                MODELS_DIR / "salary_preprocessor.pkl")
    metadata = {
        "model_name": best_name,
        "dataset_version": "data/processed/techpulse_cleaned.csv",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "feature_names_count": len(feature_names),
        "currency": MODEL_CURRENCY,
        "salary_period": MODEL_PERIOD,
        "target_definition": "(salary_norm_min + salary_norm_max) / 2, annual USD",
        "preprocessing_policy": PREPROCESSING_POLICY,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "metrics": results, "selected": best_name,
        "prediction_range_pct10_90": {"lower_add": float(range_lo), "upper_add": float(range_hi)},
        "feature_importance_top15": importance,
        "library_versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__,
                             "numpy": np.__version__},
    }
    (MODELS_DIR / "salary_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved: salary_prediction_model.pkl, salary_preprocessor.pkl, salary_model_metadata.json")


if __name__ == "__main__":
    main()
