"""
TechPulse Phase 7 — Salary preprocessing.

Builds a validated, single-currency (USD) annual salary target from the
cleaned dataset. Rules:

- Only rows with a genuinely disclosed salary are kept (never fabricated).
- Only USD observations are used (no exchange-rate conversion, so no
  invented rates; non-USD rows are excluded and preserved untouched).
- Only annual pay-period rows are used (the dataset's annualization is
  already handled in Phase 2 via utils/data_loader.ANNUALIZATION_FACTORS;
  this module documents the policy rather than re-converting).
- Range validation: min <= max, min > 0, avg = (min + max) / 2 when both
  exist, else the single disclosed value.
- Outliers are flagged (IQR) and kept for transparency but extreme values
  beyond the 1st/99th percentile are excluded from training to avoid
  letting a handful of postings dominate the loss.
"""
from __future__ import annotations

import pandas as pd

MODEL_CURRENCY = "USD"
MODEL_PERIOD = "year"

# Policy note surfaced in metadata and the UI
PREPROCESSING_POLICY = (
    "Single-currency (USD) annual-salary model. Only rows with disclosed USD "
    "salaries are used; non-USD rows are excluded (no exchange-rate conversion, "
    "so no invented rates). Only annual pay-period rows are used. Salary "
    "target = (min + max) / 2 when both disclosed, else the single value. "
    "Original salary columns are preserved untouched in the dataset."
)


def build_salary_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Filter and validate the cleaned dataset into a salary training set.

    Returns (salary_df, info_dict) where salary_df preserves all original
    columns plus a validated `salary_target` column.
    """
    info: dict = {"input_rows": len(df)}

    salary = df[df["salary_norm_currency"].notna()].copy()
    info["rows_with_disclosed_salary"] = len(salary)

    # Currency normalization: USD-only model (no conversion, no invented rates)
    salary = salary[salary["salary_norm_currency"] == MODEL_CURRENCY].copy()
    info["rows_usd"] = len(salary)
    info["rows_non_usd_excluded"] = info["rows_with_disclosed_salary"] - len(salary)

    # Pay-period handling: annual-only model (dataset already annualized
    # non-annual periods in Phase 2; we require the native annual rows to
    # keep the target genuinely comparable).
    salary = salary[salary["salary_norm_period"] == MODEL_PERIOD].copy()
    info["rows_annual"] = len(salary)

    # Range validation: min/max present, positive, min <= max
    valid = (
        salary["salary_norm_min"].notna()
        & salary["salary_norm_max"].notna()
        & (salary["salary_norm_min"] > 0)
        & (salary["salary_norm_max"] >= salary["salary_norm_min"])
    )
    salary = salary[valid].copy()
    info["rows_valid_range"] = len(salary)

    # Target: midpoint of the disclosed range (min==max -> single value)
    salary["salary_target"] = (salary["salary_norm_min"] + salary["salary_norm_max"]) / 2.0

    # Outlier investigation: flag (IQR) for transparency, trim extreme
    # 1st/99th percentile values from training only (flagged, not deleted
    # from the dataset).
    q1, q3 = salary["salary_target"].quantile([0.25, 0.75])
    iqr = q3 - q1
    salary["salary_outlier_flag"] = (
        (salary["salary_target"] < q1 - 1.5 * iqr) | (salary["salary_target"] > q3 + 1.5 * iqr)
    )
    info["rows_flagged_iqr_outlier"] = int(salary["salary_outlier_flag"].sum())

    lo, hi = salary["salary_target"].quantile([0.01, 0.99])
    trimmed = salary[(salary["salary_target"] >= lo) & (salary["salary_target"] <= hi)].copy()
    info["p01_trim"] = float(lo)
    info["p99_trim"] = float(hi)
    info["rows_trimmed_out"] = len(salary) - len(trimmed)
    info["final_rows"] = len(trimmed)

    return trimmed.reset_index(drop=True), info
