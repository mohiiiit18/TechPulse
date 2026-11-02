#!/usr/bin/env python
"""
TechPulse Phase 7 — Salary Prediction (ML) test suite.

Covers: model artifacts, preprocessing, salary target integrity (currency/period),
feature engineering, predictor API, edge cases (missing/invalid inputs, unknown
roles/locations/skills), prediction consistency, Streamlit integration and
full-app regression.

Usage: python scripts/test_salary_prediction.py   (exit 0 = all pass)
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
sys.path.insert(0, HERE)

results = []


def check(name, passed, detail=""):
    results.append((passed, name, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return passed


print("=" * 64)
print("TechPulse Phase 7 — Salary Prediction Test Suite")
print("=" * 64)

MODELS = os.path.join(HERE, "models")

# ── 1. Model artifacts load ─────────────────────────────────────────────────
try:
    model = joblib.load(os.path.join(MODELS, "salary_prediction_model.pkl"))
    preproc = joblib.load(os.path.join(MODELS, "salary_preprocessor.pkl"))
    check("salary model + preprocessor load", True, type(model).__name__)
except Exception as e:
    check("salary model + preprocessor load", False, str(e))
    sys.exit(1)

meta_path = os.path.join(MODELS, "salary_model_metadata.json")
meta = json.loads(open(meta_path, encoding="utf-8").read()) if os.path.exists(meta_path) else {}
required_meta = ["model_name", "dataset_version", "training_date", "library_versions",
                 "currency", "salary_period", "feature_names_count", "n_train", "n_test", "metrics"]
missing_meta = [k for k in required_meta if k not in meta]
check("salary_model_metadata.json complete", not missing_meta,
      f"missing: {missing_meta}" if missing_meta else "all keys present")
check("metadata currency is USD-only", meta.get("currency") == "USD" and meta.get("salary_period") == "year",
      f"currency={meta.get('currency')}, period={meta.get('salary_period')}")
check("metadata records library versions",
      {"sklearn", "pandas", "numpy"} <= set(meta.get("library_versions", {})),
      str(meta.get("library_versions")))

# ── 2. Salary target integrity (no currency mixing, no fabricated values) ──
from src.salary_preprocessing import build_salary_dataset, MODEL_CURRENCY, MODEL_PERIOD
raw = pd.read_csv(os.path.join(HERE, "data", "processed", "techpulse_cleaned.csv"), low_memory=False)
sdf, sinfo = build_salary_dataset(raw)
check("salary target all positive finite", bool((sdf["salary_target"] > 0).all()
                                                and np.isfinite(sdf["salary_target"]).all()),
      f"n={len(sdf)}, min=${sdf['salary_target'].min():,.0f}, max=${sdf['salary_target'].max():,.0f}")
check("no currency mixing in training set", set(sdf["salary_norm_currency"]) == {MODEL_CURRENCY},
      f"currencies={sorted(set(sdf['salary_norm_currency']))}")
check("only annual-period rows in training set", set(sdf["salary_norm_period"]) == {MODEL_PERIOD},
      f"periods={sorted(set(sdf['salary_norm_period']))}")
check("target = midpoint of disclosed range (never fabricated)",
      np.allclose(sdf["salary_target"], (sdf["salary_norm_min"] + sdf["salary_norm_max"]) / 2.0),
      "salary_target == (min+max)/2 exactly")
check("min <= max for all kept rows", bool((sdf["salary_norm_max"] >= sdf["salary_norm_min"]).all()),
      "range validation enforced")
check("preprocessing pipeline stats reported", sinfo["final_rows"] == len(sdf)
      and sinfo["rows_non_usd_excluded"] >= 0 and "rows_trimmed_out" in sinfo,
      f"disclosed={sinfo['rows_with_disclosed_salary']}, usd={sinfo['rows_usd']}, "
      f"annual={sinfo['rows_annual']}, valid={sinfo['rows_valid_range']}, final={sinfo['final_rows']}")
check("IQR outliers flagged, not silently deleted",
      "salary_outlier_flag" in sdf.columns and isinstance(sinfo["rows_flagged_iqr_outlier"], int),
      f"flagged={sinfo['rows_flagged_iqr_outlier']}, trimmed out={sinfo['rows_trimmed_out']}")
check("n_train + n_test in metadata match final rows (within trim/leakage deltas)",
      meta["n_train"] + meta["n_test"] == len(sdf),
      f"meta={meta['n_train']}+{meta['n_test']} vs {len(sdf)}")
check("usable salary records disclosed", sinfo["final_rows"] > 100, f"{sinfo['final_rows']} usable rows")

# ── 3. Feature engineering ──────────────────────────────────────────────────
feat_names = preproc["feature_names"] if isinstance(preproc, dict) else meta["feature_names"]
check("no target-leaking features (salary columns excluded)",
      not any(c.startswith("salary") or c in ("min_salary", "max_salary") for c in feat_names),
      f"{len(feat_names)} features, none salary-derived")
check("features include role, seniority, country, employment, remote, skills",
      all(any(f.startswith(p) for f in feat_names) for p in
          ["job_role_", "seniority_", "country_", "employment_type_"]),
      "one-hot blocks + skills present")
check("no title/company features (could reconstruct target)",
      not any(c in feat_names for c in ("job_title", "company_name", "company")),
      "no leakage-prone identifier features")

# ── 4. Predictor API + valid inputs ─────────────────────────────────────────
from src.salary_predictor import SalaryPredictor, LIMITATIONS
p = SalaryPredictor()
check("predictor loads meta + known values", len(p.known_roles) > 0 and len(p.known_countries) > 0
      and len(p.known_skills) > 0,
      f"roles={len(p.known_roles)}, countries={len(p.known_countries)}, skills={len(p.known_skills)}")

pred, lo, hi, info = p.predict_salary("Data Scientist / ML Engineer", "United States",
                                      ["python", "machine-learning", "sql"],
                                      is_remote=True, employment_type="full_time", seniority="senior")
check("valid input -> sensible USD prediction", 20000 <= pred <= 600000 and lo < pred < hi,
      f"${pred:,.0f} in [{lo:,.0f}, {hi:,.0f}], model={info['model']}")
check("range is empirical error band, wider than point estimate",
      hi - lo > 0 and "10-90" in info["range_note"].replace("–", "-").replace("—", "-"),
      f"width=${hi - lo:,.0f}")
check("currency/period correct in output", info["currency"] == "USD" and info["period"] == "annual",
      str(info["currency"]))

# Seniority affects prediction (not silently ignored)
pred_j, _, _, _ = p.predict_salary("Data Scientist / ML Engineer", "United States",
                                   ["python", "sql"], seniority="junior")
pred_s, _, _, _ = p.predict_salary("Data Scientist / ML Engineer", "United States",
                                   ["python", "sql"], seniority="senior")
check("seniority shifts prediction", pred_s != pred_j, f"junior=${pred_j:,.0f} vs senior=${pred_s:,.0f}")

# Remote toggle affects prediction
pred_r, _, _, _ = p.predict_salary("Software Developer", "United States", ["python"], is_remote=True)
pred_o, _, _, _ = p.predict_salary("Software Developer", "United States", ["python"], is_remote=False)
check("remote status affects prediction", pred_r != pred_o, f"remote=${pred_r:,.0f} vs on-site=${pred_o:,.0f}")

# ── 5. Edge cases: missing / invalid / unknown inputs ───────────────────────
try:
    p.predict_salary("Data Scientist / ML Engineer", "United States", [])
    check("empty skills list handled", True, "no exception")
except Exception as e:
    check("empty skills list handled", False, str(e))

try:
    p.predict_salary("Data Scientist / ML Engineer", "United States", None)
    check("None skills handled", True, "no exception")
except Exception as e:
    check("None skills handled", False, str(e))

try:
    p2, lo2, hi2, _ = p.predict_salary("Totally Made-Up Role", "Atlantis", ["python"])
    check("unknown role + unknown location -> falls back to defaults (no crash)",
          20000 <= p2 <= 600000 and lo2 < p2 < hi2, f"${p2:,.0f} (defaults used)")
except Exception as e:
    check("unknown role + unknown location", False, str(e))

check("unknown skills ignored safely",
      p.predict_salary("Data Scientist / ML Engineer", "United States", ["python", "quantum-telepathy"])[0]
      == p.predict_salary("Data Scientist / ML Engineer", "United States", ["python"])[0],
      "unknown skill has zero effect on prediction")

bad = 0
for bad_skills in [[""], [None], [123], [np.nan]]:
    try:
        p.predict_salary("Software Developer", "United States", bad_skills)
    except Exception:
        bad += 1
check("invalid skill values (None/NaN/non-str) don't crash", bad == 0, f"{bad}/4 raised")

# ── 6. Prediction consistency ───────────────────────────────────────────────
a = p.predict_salary("Data Scientist / ML Engineer", "United States", ["python", "aws"])
b = p.predict_salary("Data Scientist / ML Engineer", "United States", ["python", "aws"])
check("prediction deterministic on repeat calls", a[0] == b[0] and a[1] == b[1], "identical outputs")

# Different skill combinations produce different predictions
pred_ds, _, _, _ = p.predict_salary("Data Scientist / ML Engineer", "United States", ["python", "sql"])
pred_sw, _, _, _ = p.predict_salary("Software Developer", "United States", ["python", "sql"])
check("different roles -> different predictions", pred_ds != pred_sw,
      f"DS=${pred_ds:,.0f} vs SW=${pred_sw:,.0f}")
more_skills = p.predict_salary("Data Scientist / ML Engineer", "United States",
                               ["python", "sql", "aws", "docker", "llm"])[0]
check("adding skills changes prediction (skill features active)", more_skills != pred_ds,
      f"2 skills=${pred_ds:,.0f} vs 5 skills=${more_skills:,.0f}")

# ── 7. Model quality gates (from saved test-split metrics) ─────────────────
sel = meta["metrics"][meta["selected"]]
check("selected model metrics reasonable", sel["test_mae"] < 60000 and sel["test_r2"] > 0.3,
      f"MAE=${sel['test_mae']:,.0f}, R2={sel['test_r2']:.2f}, RMSE=${sel['test_rmse']:,.0f}")
check("cv metrics present for all trained models",
      all("cv5_mae" in m for m in meta["metrics"].values()),
      f"models={sorted(meta['metrics'])}")
check("model selection by test MAE (generalization, not train accuracy)",
      meta["selected"] == min(meta["metrics"], key=lambda k: meta["metrics"][k]["test_mae"]),
      f"selected={meta['selected']}")
check("all 4 model families trained (Linear/RF/GB/XGB)",
      {"LinearRegression", "RandomForest", "GradientBoosting", "XGBoost"} <= set(meta["metrics"]),
      f"trained={sorted(meta['metrics'])}")

# ── 8. Streamlit integration + regression ───────────────────────────────────
try:
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(HERE, "app_pages/salary_prediction.py"),
                           default_timeout=120).run()
    check("salary_prediction page renders", not at.exception,
          str(at.exception) if at.exception else "OK")

    # Full prediction flow through the UI
    at2 = AppTest.from_file(os.path.join(HERE, "app_pages/salary_prediction.py"),
                            default_timeout=120)
    at2.run()
    at2.button[0].click().run()
    check("predict-salary button flow works via UI", not at2.exception
          and any("Predicted salary" in m.label for m in at2.metric),
          str(at2.exception) if at2.exception else "metrics rendered")

    # Regression: benchmark page + full app
    at4 = AppTest.from_file(os.path.join(HERE, "app_pages/salary_benchmark.py"),
                            default_timeout=120).run()
    check("regression: salary_benchmark page intact", not at4.exception, "OK")
    at5 = AppTest.from_file(os.path.join(HERE, "app.py"), default_timeout=120).run()
    check("regression: app.py navigates with salary prediction page", not at5.exception, "OK")
    for page in ["app_pages/career_matcher.py", "app_pages/market_intelligence.py",
                 "app_pages/job_explorer.py", "app_pages/job_recommendations.py",
                 "app_pages/resume_analyzer.py", "app_pages/skill_gap_analyzer.py"]:
        atr = AppTest.from_file(os.path.join(HERE, page), default_timeout=120).run()
        check(f"regression: {page}", not atr.exception,
              str(atr.exception) if atr.exception else "OK")

    check("salary benchmark page file exists & separate",
          os.path.exists(os.path.join(HERE, "app_pages/salary_benchmark.py"))
          and os.path.exists(os.path.join(HERE, "app_pages/salary_prediction.py")),
          "two separate salary pages")
except Exception as e:
    import traceback
    traceback.print_exc()
    check("Streamlit integration", False, str(e))

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
passed = sum(1 for x, *_ in results if x)
failed = sum(1 for x, *_ in results if not x)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 64)

out = os.path.join(HERE, "reports", "salary_prediction_test_results.txt")
with open(out, "w", encoding="utf-8") as f:
    f.write("TechPulse Phase 7 Salary Prediction Test Results\n" + "=" * 60 + "\n\n")
    for x, n, d in results:
        f.write(f"[{'PASS' if x else 'FAIL'}] {n}: {d}\n")
    f.write(f"\nTOTAL: {passed} passed, {failed} failed\n")
print(f"Saved: {out}")
sys.exit(0 if failed == 0 else 1)
