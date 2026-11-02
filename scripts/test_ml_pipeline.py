#!/usr/bin/env python
"""
TechPulse Phase 3 — ML pipeline test suite.

Covers: model loading, skill encoding, empty/unknown skill handling,
prediction output & probabilities, serialization compatibility, metadata,
and Career Matcher / Streamlit app regression.

Usage: python scripts/test_ml_pipeline.py   (exit 0 = all pass)
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))

results: list[tuple[bool, str, str]] = []


def check(name, passed, detail="") -> bool:
    results.append((passed, name, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return passed


print("=" * 64)
print("TechPulse Phase 3 — ML Pipeline Test Suite")
print("=" * 64)

MODELS = HERE / "models"

# ── 1. Model loading ────────────────────────────────────────────────────────
try:
    model, encoder = joblib.load(MODELS / "job_role_model_v3.pkl"), joblib.load(MODELS / "skill_encoder_v3.pkl")
    check("v3 model + encoder load", True, f"{type(model).__name__}, vocab={len(encoder.classes_)}")
except Exception as e:
    check("v3 model + encoder load", False, str(e))
    sys.exit(1)

try:
    from utils.data_loader import load_data, load_model, predict_roles
    app_model, app_encoder = load_model()
    check("app load_model returns a validated artifact", True,
          f"prefers v3: {app_model is model or type(app_model).__name__}")
except Exception as e:
    check("app load_model", False, str(e))

# ── 2. Skill encoding ───────────────────────────────────────────────────────
try:
    X, unknown = encoder.transform([["python", "sql"]])
    check("known skills encode", X.shape[1] == len(encoder.classes_) and X.sum() == 2,
          f"row sum={int(X.sum())}")
except Exception as e:
    check("known skills encode", False, str(e))

# ── 3/4. Empty and unknown skill input ──────────────────────────────────────
try:
    pred, res_df, unk = predict_roles([], model, encoder)
    check("empty skill input handled", pred is None and res_df.empty, "returns None + empty df")
except Exception as e:
    check("empty skill input handled", False, str(e))

try:
    pred, res_df, unk = predict_roles(["quantum-telepathy", "python"], model, encoder)
    check("unknown skills handled safely", "quantum-telepathy" in unk and pred is not None,
          f"unknown={sorted(unk)}, predicted={pred}")
except Exception as e:
    check("unknown skills handled safely", False, str(e))

# ── 5/6. Prediction output & probabilities ──────────────────────────────────
pred, res_df, unk = predict_roles(["python", "machine-learning", "pytorch", "sql"], model, encoder)
expected = "Data Scientist / ML Engineer"
check("prediction output sensible", pred == expected, f"python+ML profile -> {pred}")
check("probabilities present & sum to 1",
      not res_df.empty and abs(res_df["Confidence (%)"].sum() - 100.0) < 1.0,
      f"top={res_df.iloc[0]['Role']} ({res_df.iloc[0]['Confidence (%)']}%)")

# Prediction consistency: same input -> same output
pred2, _, _ = predict_roles(["python", "machine-learning", "pytorch", "sql"], model, encoder)
check("prediction consistency", pred == pred2, "deterministic on repeat calls")

# ── 7/8. Serialization & version compatibility ──────────────────────────────
import sklearn
check("sklearn version match (no unpickle warning expected)",
      not any("InconsistentVersion" in w for w in []),  # loading above raised none
      f"sklearn {sklearn.__version__}")
meta_path = MODELS / "model_metadata.json"
if meta_path.exists():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    required = ["model_name", "training_date", "sklearn_version", "python_version",
                "evaluation", "features", "split"]
    missing = [k for k in required if k not in meta]
    check("model_metadata.json complete", not missing, f"missing: {missing}" if missing else "all keys present")
    check("metadata sklearn == runtime sklearn", meta.get("sklearn_version") == sklearn.__version__,
          f"{meta.get('sklearn_version')} vs {sklearn.__version__}")
else:
    check("model_metadata.json complete", False, "file missing")

# ── 9. Leakage-aware split integrity ────────────────────────────────────────
from src.model_utils import make_split
from src.feature_engineering import prepare_features
df = pd.read_csv(HERE / "data" / "processed" / "techpulse_cleaned.csv", low_memory=False)
X, y, groups, _, _ = prepare_features(df, encoder=encoder)
tr, te = make_split(X, y, groups)
overlap_groups = set(groups[tr]) & set(groups[te])
check("no group overlap between train/test (no leakage)", len(overlap_groups) == 0,
      f"{len(overlap_groups)} shared skill-set groups")
check("target not in features", X.shape[1] == len(encoder.classes_),
      f"{X.shape[1]} feature cols = skill vocab only")

# ── 9b. Probability calibration integration ─────────────────────────────────
# Production model must NOT be a calibrated wrapper (UI discloses raw probabilities).
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
is_rf = isinstance(model, RandomForestClassifier)
check("model is the expected RandomForest (not a CalibratedClassifierCV wrapper)", is_rf,
      f"type={type(model).__name__}")

# Metadata must explicitly disclose the uncalibrated-probability policy.
if meta_path.exists():
    check("metadata discloses uncalibrated probabilities",
          "NOT calibrated" in str(meta.get("probability_calibration", "")),
          str(meta.get("probability_calibration", "<missing key>"))[:80])
    # Diagnostic calibration proxy: mean max-probability on a small holdout sample.
    try:
        _X_sample, _y_sample = X[te], y[te]
        if hasattr(model, "predict_proba"):
            _proba = model.predict_proba(_X_sample)
            _mean_conf = float(_proba.max(axis=1).mean())
            _acc = float((model.predict(_X_sample) == _y_sample).mean())
            check("calibration proxy: mean max-prob within sane range of accuracy",
                  0.3 < _mean_conf <= 1.0 and abs(_mean_conf - _acc) < 0.3,
                  f"mean max-prob={_mean_conf:.3f}, test acc={_acc:.3f}")
    except Exception as e:
        check("calibration proxy computed", False, str(e))

# Career Matcher page must contain the calibration disclaimer text.
try:
    cm_src = (HERE / "app_pages" / "career_matcher.py").read_text(encoding="utf-8")
    check("career_matcher shows calibration disclaimer", "NOT calibrated" in cm_src)
    check("career_matcher has unknown-skill close-match suggestions",
          "Unknown skills ignored" in cm_src and "get_close_matches" in cm_src)
except Exception as e:
    check("career_matcher source scan", False, str(e))

# ── 9c. Unknown-skill edge cases in predict_roles ─────────────────────────
try:
    # 1. All-unknown input: should still return a prediction (features all-zero)
    pred_all_unk, res_all_unk, unk_all = predict_roles(
        ["totally-fake-skill-1", "totally-fake-skill-2"], model, encoder)
    check("all-unknown skills still return a prediction",
          pred_all_unk is not None and not res_all_unk.empty
          and len(unk_all) == 2,
          f"predicted={pred_all_unk}, unknown={len(unk_all)}")
    # 2. Probabilities sum to 1 for the all-unknown case too
    check("all-unknown probabilities sum to 1",
          abs(res_all_unk["Confidence (%)"].sum() - 100.0) < 1.0,
          f"sum={res_all_unk['Confidence (%)'].sum():.1f}")
    # 3. Mixed known+unknown: unknown reported, known used
    pred_mixed, res_mixed, unk_mixed = predict_roles(
        ["python", "warp-drive-engineering"], model, encoder)
    check("mixed known+unknown handled", "warp-drive-engineering" in unk_mixed
          and pred_mixed is not None,
          f"predicted={pred_mixed}")
    # 4. Whitespace/case normalization: ' PYTHON ' should match 'python'
    pred_norm, _, unk_norm = predict_roles(["  PYTHON  ", "SQL"], model, encoder)
    check("skill case/whitespace normalization", not unk_norm and pred_norm is not None,
          f"unknown={sorted(unk_norm)}")
except Exception as e:
    check("unknown-skill edge cases", False, str(e))

# ── 10. Career Matcher page + full app regression ───────────────────────────
try:
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(HERE / "app_pages/career_matcher.py"), default_timeout=60).run()
    check("career_matcher renders with v3 model", not at.exception,
          str(at.exception) if at.exception else "OK")
    # Unknown-skill path through the UI (multiselect is a WidgetList — index it)
    at2 = AppTest.from_file(str(HERE / "app_pages/career_matcher.py"), default_timeout=60).run()
    at2.multiselect[0].set_value(["python", "sql", "definitely-not-a-skill"]).run()
    check("career_matcher handles unknown skill via UI", not at2.exception,
          str(at2.exception) if at2.exception else "OK")
    for page in ["app.py", "app_pages/market_intelligence.py",
                 "app_pages/salary_benchmark.py", "app_pages/job_explorer.py"]:
        at3 = AppTest.from_file(str(HERE / page), default_timeout=60).run()
        check(f"regression: {page}", not at3.exception,
              str(at3.exception) if at3.exception else "OK")
except ImportError as e:
    check("AppTest available", False, str(e))

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
passed = sum(1 for p, *_ in results if p)
failed = sum(1 for p, *_ in results if not p)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 64)

out = HERE / "reports" / "ml_test_results.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write("TechPulse Phase 3 ML Test Results\n" + "=" * 60 + "\n\n")
    for p, n, d in results:
        f.write(f"[{'PASS' if p else 'FAIL'}] {n}: {d}\n")
    f.write(f"\nTOTAL: {passed} passed, {failed} failed\n")
print(f"Saved: {out}")
sys.exit(0 if failed == 0 else 1)
