"""
TechPulse Phase 7 — Salary predictor (inference only, no training here).

Loads the saved model + preprocessor and exposes predict_salary() for the
Streamlit page. Handles unknown roles/locations/skills safely (never crashes,
never invents values) and returns an empirical error-based range taken from
the saved metadata (10–90% residual quantiles on the test split).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
MODELS_DIR = HERE / "models"

LIMITATIONS = [
    "Predictions are model estimates from historical job postings (up to Aug 2026), not offers.",
    "Expected error: roughly ±$24k (MAE) on typical postings; the range shown is an "
    "empirical 10–90% error band from the test split — not a statistical confidence interval.",
    "R² ≈ 0.59 on the test split: the model explains ~59% of salary variance; many factors "
    "(company size, experience level, negotiation) are not in the dataset.",
    "Accuracy is lower for some roles (e.g. DevOps) and locations (e.g. non-US)",
    "USD-only model: non-USD postings are excluded from training and never converted.",
    "Seniority is a parsed title token — 'unspecified' is a real observed category, not a guess.",
]


class SalaryPredictor:
    """Loads the saved model + preprocessor and predicts annual USD salaries."""

    def __init__(self):
        self.model = joblib.load(MODELS_DIR / "salary_prediction_model.pkl")
        self.prep = joblib.load(MODELS_DIR / "salary_preprocessor.pkl")
        self.meta = json.loads((MODELS_DIR / "salary_model_metadata.json").read_text())
        self.feature_names: list[str] = self.prep["feature_names"]
        self.enc = self.prep["encoder"]
        self.range_lo = self.meta["prediction_range_pct10_90"]["lower_add"]
        self.range_hi = self.meta["prediction_range_pct10_90"]["upper_add"]

    @property
    def known_roles(self) -> list[str]:
        return [c.removeprefix("job_role_") for c in self.feature_names
                if c.startswith("job_role_")]

    @property
    def known_countries(self) -> list[str]:
        return [c.removeprefix("country_") for c in self.feature_names
                if c.startswith("country_")]

    @property
    def known_skills(self) -> list[str]:
        return list(self.enc.classes_)

    @property
    def known_employment_types(self) -> list[str]:
        return [c.removeprefix("employment_type_") for c in self.feature_names
                if c.startswith("employment_type_")]

    def _vector(self, role, seniority, country, employment_type, is_remote, skills):
        row = pd.Series(0, index=self.feature_names, dtype=float)
        # One-hot categorical features; unknown values are safely ignored (all-zero)
        for prefix, value in [("job_role", role), ("seniority", seniority),
                              ("country", country), ("employment_type", employment_type)]:
            col = f"{prefix}_{value}"
            if col in row.index:
                row[col] = 1
        if "is_remote" in row.index:
            row["is_remote"] = 1 if is_remote else 0
        # Skills: only known encoder classes set; unknown skills ignored
        for s in (skills or []):
            if s in self.enc.classes_:
                col = f"skill_{s}"
                if col in row.index:
                    row[col] = 1
        return row.to_frame().T

    def predict_salary(self, role, country, skills, is_remote=False,
                       employment_type=None, seniority="unspecified"):
        """Return (prediction, lower, upper, info) for annual USD salary."""
        X = self._vector(role, seniority, country, employment_type or "", is_remote, skills)
        pred = float(np.atleast_1d(self.model.predict(X))[0])
        return pred, pred + self.range_lo, pred + self.range_hi, {
            "currency": "USD", "period": "annual",
            "model": self.meta["model_name"],
            "range_note": "Empirical 10–90% residual range from test split (error-based, not a CI)",
        }
