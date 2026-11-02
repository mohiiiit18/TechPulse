#!/usr/bin/env python
"""
TechPulse Phase 2 — Dataset Cleaning & Data Quality Engineering.

Pipeline: Inspect -> Clean -> Validate -> Save -> Integrate -> Test.

Policy (hard rules):
  * The original dataset file is NEVER modified or overwritten.
  * No missing value is ever fabricated; gaps are preserved (Unknown/NaN).
  * Only CONFIRMED duplicates are removed (same job_id or same URL).
  * No currency conversion — no invented exchange rates.
  * Outliers are flagged, never auto-deleted.

Inputs :  data/processed dataset.csv   (treated as the raw source snapshot)
Outputs:  data/raw/processed dataset.csv       (immutable raw copy)
          data/processed/techpulse_cleaned.csv (cleaned dataset)
          data/reports/*.json / *.csv          (quality, missing, dup, outlier, before/after)

Usage: python scripts/clean_dataset.py
"""

from __future__ import annotations

import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_IN = BASE_DIR / "data" / "processed dataset.csv"
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "data" / "reports"
for d in (RAW_DIR, CLEAN_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Documented annualization factors — conversion ASSUMPTIONS, not exchange rates.
ANNUALIZATION_FACTORS = {"year": 1.0, "hour": 2000.0, "month": 12.0}

# Seniority extraction — preserved as its own column, titles are NOT merged away.
SENIORITY_PATTERNS = [
    ("intern", r"(?i)\bintern(ship)?\b"),
    ("chief", r"(?i)\b(chief|cta|cto|cpo)\b"),
    ("head", r"(?i)\bhead of\b"),
    ("principal", r"(?i)\bprincipal\b"),
    ("staff", r"(?i)\bstaff\b"),
    ("lead", r"(?i)\blead\b"),
    ("senior", r"(?i)\b(senior|sr\.?)\b"),
    ("junior", r"(?i)\b(junior|jr\.?|graduate|entry[\s-]?level)\b"),
]

# Skill standardization map — ONLY unambiguous equivalents.
# Applied on lowercase tokens; original tokens preserved in skills_original.
SKILL_ALIASES = {
    "ml": "machine-learning",
    "machine learning": "machine-learning",
    "natural language processing": "nlp",
    "javascript": "javascript",  # canonical spelling (JS handled below)
    "js": "javascript",
    "nodejs": "node.js",
    "node": "node.js",
    "reactjs": "react",
    "react.js": "react",
    "nextjs": "next.js",
    "postgres": "postgresql",
    "postgres sql": "postgresql",
    "k8s": "kubernetes",
    "tf": "terraform",  # safe: terraform is the only single 'tf' skill in this dataset
    "ci/cd": "ci-cd",
    "c icd": "ci-cd",
    "graphql": "graphql",
    "pytorch": "pytorch",
    "tf-learning": "tensorflow",
    "tensor flow": "tensorflow",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "big-data": "big-data",
    "etl": "etl",
    "html/css": "html-css",
    "html css": "html-css",
    "ui/ux": "ui-ux",
    "ui ux": "ui-ux",
    "qa": "qa-testing",
    "product mgmt": "product-management",
    "product manager": "product-management",
    "proj-mgmt": "project-management",
}


def parse_skills(s) -> list[str]:
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [x.strip().lower() for x in str(s).split("|") if x.strip()]


def print_header(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# ─────────────────────────────────────────────────────────────────────────────
# 1. INSPECT
# ─────────────────────────────────────────────────────────────────────────────
print_header("[1/7] INSPECT — raw dataset profile")
df_raw = pd.read_csv(RAW_IN, low_memory=False)
print(f"    Records: {len(df_raw)} | Columns: {len(df_raw.columns)}")

# Work on a copy; raw file on disk is never touched.
df = df_raw.copy()

# Normalize obvious whitespace noise in text columns (no semantic change).
text_cols = [c for c in df.columns if df[c].dtype == object]
for c in text_cols:
    df[c] = df[c].apply(lambda v: re.sub(r"\s+", " ", str(v)).strip() if pd.notna(v) else v)

n_empty_titles = int((df["title"].astype(str).str.strip() == "").sum())
skills_lists_before = df["skills"].apply(parse_skills)
n_empty_skills_records = int(skills_lists_before.map(len).eq(0).sum())
n_unique_skills_raw = len({s for lst in skills_lists_before for s in lst})

inspection = {
    "records": int(len(df)),
    "columns": int(len(df.columns)),
    "unique_job_roles": int(df["job_role"].nunique()),
    "job_roles": df["job_role"].value_counts().to_dict(),
    "unique_titles": int(df["title"].nunique()),
    "unique_skills_raw": int(n_unique_skills_raw),
    "employment_types": {str(k): int(v) for k, v in df["employment_type"].value_counts(dropna=False).items()},
    "currency_distribution": {str(k): int(v) for k, v in df["salary_currency"].value_counts(dropna=False).items()},
    "period_distribution": {str(k): int(v) for k, v in df["salary_period"].value_counts(dropna=False).items()},
    "countries_top10": df["country"].value_counts().head(10).to_dict(),
    "empty_titles": n_empty_titles,
    "records_with_no_skills": n_empty_skills_records,
    "duplicate_job_id": int(df.duplicated(subset=["job_id"]).sum()),
    "duplicate_url": int(df.duplicated(subset=["url"]).sum()),
    "negative_salary_min": int((pd.to_numeric(df["salary_min"], errors="coerce") < 0).sum()),
    "salary_min_gt_max": int(
        (pd.to_numeric(df["salary_min"], errors="coerce") > pd.to_numeric(df["salary_max"], errors="coerce")).sum()
    ),
    "invalid_dates": int(pd.to_datetime(df["posted_at"], errors="coerce", utc=True).isna().sum()),
}
print(f"    Roles: {inspection['unique_job_roles']} | Skills: {n_unique_skills_raw} | Titles: {inspection['unique_titles']}")
print(f"    Empty titles: {n_empty_titles} | Records w/o skills: {n_empty_skills_records}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CLEAN
# ─────────────────────────────────────────────────────────────────────────────
print_header("[2/7] CLEAN")

# 2a. Missing categorical values -> explicit "Unknown" (never fabricated real data).
for col in ["country", "city", "company_industry"]:
    if col in df.columns:
        n_before = int(df[col].isna().sum())
        df[col] = df[col].fillna("Unknown")
        print(f"    {col}: {n_before} missing -> 'Unknown'")

# 2b. Skills: distinguish genuinely-empty vs missing, and standardize aliases.
df["skills_original"] = df["skills"]  # preserve raw pipe-separated skills verbatim


def standardize_skill(tok: str) -> str:
    return SKILL_ALIASES.get(tok, tok)


def clean_skill_list(s):
    toks = parse_skills(s)
    seen, out = set(), []
    for t in toks:
        std = standardize_skill(t)
        if std and std not in seen:
            seen.add(std)
            out.append(std)
    return out


df["parsed_skills_std"] = df["skills"].apply(clean_skill_list)
df["skills_std"] = df["parsed_skills_std"].apply(lambda lst: "|".join(lst) if lst else "")
df["skills_status"] = np.where(
    df["parsed_skills_std"].map(len) > 0, "specified", "missing"  # distinguishes empty vs unknown
)
n_alias_hits = int(sum(
    1 for lst in skills_lists_before for t in lst if t in SKILL_ALIASES and SKILL_ALIASES[t] != t
))
print(f"    Skills: {n_alias_hits} alias tokens standardized; "
      f"{n_unique_skills_raw} raw tokens -> {len({s for lst in df['parsed_skills_std'] for s in lst})} canonical")

# 2c. Job title standardization: normalize casing/whitespace, extract seniority.
#     Titles are normalized for CONSISTENCY ONLY — fundamentally different roles are
#     not merged; the ML target (job_role) is untouched.
def extract_seniority(title: str) -> str:
    for label, pat in SENIORITY_PATTERNS:
        if re.search(pat, str(title)):
            return label
    return "unspecified"


df["title_clean"] = df["title"].apply(lambda t: re.sub(r"\s+", " ", str(t)).strip().lower())
df["seniority"] = df["title"].apply(extract_seniority)
print(f"    Titles normalized: {df_raw['title'].nunique()} -> {df['title_clean'].nunique()} distinct cleaned titles")
print(f"    Seniority extracted: {df['seniority'].value_counts().to_dict()}")

# 2d. Salary: numeric coercion + normalized annual-equivalent USD columns.
#     Original salary_min/max/currency/period are preserved untouched.
#     Normalized values exist ONLY for USD rows (no exchange rates invented).
for c in ("salary_min", "salary_max"):
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["salary_norm_min"] = np.nan
df["salary_norm_max"] = np.nan
df["salary_norm_currency"] = pd.Series(pd.NA, index=df.index, dtype="string")
df["salary_norm_period"] = pd.Series(pd.NA, index=df.index, dtype="string")

usd = df["salary_currency"].eq("USD")
for period, factor in ANNUALIZATION_FACTORS.items():
    sel = usd & df["salary_period"].eq(period)
    df.loc[sel, "salary_norm_min"] = df.loc[sel, "salary_min"] * factor
    df.loc[sel, "salary_norm_max"] = df.loc[sel, "salary_max"] * factor
    df.loc[sel, "salary_norm_currency"] = "USD"
    df.loc[sel, "salary_norm_period"] = "year"
df["salary_norm_avg"] = (df["salary_norm_min"] + df["salary_norm_max"]) / 2
print(f"    Salary normalized (USD annualized): {int(df['salary_norm_avg'].notna().sum())} of {len(df)} rows")
print("    Non-USD rows: original values preserved, NO conversion (no invented FX rates)")

# 2e. Dates & freshness.
posted = pd.to_datetime(df["posted_at"], errors="coerce", utc=True)
df["posted_date"] = posted
today = pd.Timestamp.now(tz="UTC")
df["posting_age_days"] = (today - posted).dt.days


def freshness(age):
    if pd.isna(age):
        return "unknown"
    if age <= 30:
        return "fresh (<=30d)"
    if age <= 90:
        return "recent (31-90d)"
    if age <= 365:
        return "aging (91-365d)"
    return "stale (>365d)"


df["freshness"] = df["posting_age_days"].apply(freshness)
print(f"    Freshness: {df['freshness'].value_counts().to_dict()}")

# 2f. Confirmed duplicate removal (same job_id OR same URL only).
before = len(df)
dup_mask = df.duplicated(subset=["job_id"]) | df.duplicated(subset=["url"])
dup_report_rows = df.loc[dup_mask, ["job_id", "title", "company_name", "url"]].copy()
df = df[~dup_mask].copy()
removed = before - len(df)
print(f"    Duplicates removed (confirmed by job_id/url): {removed}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. VALIDATE
# ─────────────────────────────────────────────────────────────────────────────
print_header("[3/7] VALIDATE")
checks = {
    "no_missing_job_id": bool(df["job_id"].notna().all()),
    "no_duplicate_job_id": bool(not df.duplicated(subset=["job_id"]).any()),
    "no_duplicate_url": bool(not df.duplicated(subset=["url"]).any()),
    "no_empty_titles": bool((df["title_clean"].str.strip() != "").all()),
    "no_negative_salaries": bool((df[["salary_min", "salary_max"]].min(skipna=True) >= 0).all()),
    # Compare only rows where BOTH bounds are present; NaN-vs-number comparisons
    # evaluate False and would wrongly fail otherwise.
    "salary_min_le_max": bool(
        (df.loc[df["salary_min"].notna() & df["salary_max"].notna(), "salary_min"]
         <= df.loc[df["salary_min"].notna() & df["salary_max"].notna(), "salary_max"]).all()
    ),
    "dates_parse_or_missing": bool(df["posted_date"].notna().sum() + df["posted_at"].isna().sum() == len(df)),
    "salary_period_values_valid": bool(df["salary_period"].dropna().isin(["year", "month", "hour"]).all()),
    "salary_currency_iso_like": bool(
        df["salary_currency"].dropna().astype(str).str.match(r"^[A-Z]{3}$").all()
    ),
    "skills_format_valid": bool(
        df["skills_std"].apply(lambda s: s == "" or all(re.match(r"^[a-z0-9.#+-]+$", t) for t in s.split("|"))).all()
    ),
    "freshness_no_unknown_when_date_present": bool(
        df.loc[df["posted_date"].notna(), "freshness"].ne("unknown").all()
    ),
}
for name, ok in checks.items():
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}")

# Salary outlier report (IQR on normalized USD annual) — flagged only.
out = df[df["salary_norm_avg"].notna()].copy()
q1, q3 = out["salary_norm_avg"].quantile([0.25, 0.75])
iqr = q3 - q1
lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
out_mask = (out["salary_norm_avg"] < lo_fence) | (out["salary_norm_avg"] > hi_fence)
outlier_df = out.loc[out_mask, [
    "job_id", "title_clean", "company_name", "salary_min", "salary_max",
    "salary_currency", "salary_period", "salary_norm_avg"
]]
print(f"    Salary outliers (IQR fences {lo_fence:,.0f} / {hi_fence:,.0f}): {len(outlier_df)} flagged, 0 removed")

validation_pass = all(checks.values())

# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVE
# ─────────────────────────────────────────────────────────────────────────────
print_header("[4/7] SAVE")
# Immutable raw snapshot (copied from source; source itself untouched).
raw_snapshot = RAW_DIR / "processed dataset.csv"
if not raw_snapshot.exists():
    raw_snapshot.write_bytes(RAW_IN.read_bytes())
    print(f"    Raw snapshot preserved: {raw_snapshot}")

clean_path = CLEAN_DIR / "techpulse_cleaned.csv"
df.to_csv(clean_path, index=False)
print(f"    Cleaned dataset: {clean_path} ({len(df)} rows, {len(df.columns)} cols)")

after_missing = {
    c: {"count": int(df[c].isna().sum()), "pct": round(df[c].isna().mean() * 100, 2)}
    for c in df.columns if df[c].isna().sum() > 0
}
missing_report = {
    "before_cleaning": {
        c: {"count": int(df_raw[c].isna().sum()), "pct": round(df_raw[c].isna().mean() * 100, 2)}
        for c in df_raw.columns if df_raw[c].isna().sum() > 0
    },
    "after_cleaning_note": "Country/city/industry NaNs were relabelled 'Unknown' in text columns; "
                           "salary NaNs are preserved (never fabricated); parsed_skills_std empty "
                           "means genuinely no skills listed (skills_status='missing').",
    "after_cleaning_nan_columns": after_missing,
}

with open(REPORTS_DIR / "missing_value_report.json", "w") as f:
    json.dump(missing_report, f, indent=2, default=str)

with open(REPORTS_DIR / "duplicate_report.json", "w") as f:
    json.dump({
        "policy": "Remove only confirmed duplicates (same job_id or same URL).",
        "duplicate_job_id_before": inspection["duplicate_job_id"],
        "duplicate_url_before": inspection["duplicate_url"],
        "suspected_same_title_company_location": int(df_raw.duplicated(subset=["title", "company_name", "location_raw"]).sum()),
        "suspected_note": "Suspected groups have unique job_ids and URLs (multi-region postings / re-posts); retained.",
        "confirmed_duplicates_removed": removed,
        "rows_removed": [r.to_dict() for _, r in dup_report_rows.iterrows()],
    }, f, indent=2, default=str)

outlier_df.to_csv(REPORTS_DIR / "salary_outlier_report.csv", index=False)

# Per-column salary stats before/after
def salary_stats(d, col):
    s = pd.to_numeric(d[col], errors="coerce").dropna()
    if s.empty:
        return {}
    return {"n": int(len(s)), "min": float(s.min()), "p25": float(s.quantile(.25)),
            "median": float(s.median()), "p75": float(s.quantile(.75)), "p90": float(s.quantile(.9)),
            "max": float(s.max())}


before_after = {
    "rows_before": int(len(df_raw)),
    "rows_after": int(len(df)),
    "rows_removed": int(removed),
    "columns_before": int(len(df_raw.columns)),
    "columns_after": int(len(df.columns)),
    "new_columns": [c for c in df.columns if c not in df_raw.columns],
    "skills_raw_unique": inspection["unique_skills_raw"],
    "skills_canonical_unique": len({s for lst in df["parsed_skills_std"] for s in lst}),
    "alias_tokens_standardized": n_alias_hits,
    "titles_before": inspection["unique_titles"],
    "titles_after_clean": int(df["title_clean"].nunique()),
    "salary_stats_usd_min_raw": salary_stats(df_raw[df_raw["salary_currency"] == "USD"], "salary_min"),
    "salary_stats_norm_usd_year": salary_stats(df[df["salary_norm_currency"] == "USD"], "salary_norm_avg"),
    "salary_usd_period_breakdown": {
        str(k): int(v) for k, v in
        df[df["salary_currency"] == "USD"]["salary_period"].value_counts(dropna=False).items()
    },
    "validation_all_passed": validation_pass,
    "validation_checks": checks,
    "generated_at": datetime.now(timezone.utc).isoformat(),
}
with open(REPORTS_DIR / "before_after_summary.json", "w") as f:
    json.dump(before_after, f, indent=2, default=str)

quality_report = {**inspection, **before_after, "salary_outliers": {
    "n_flagged": int(len(outlier_df)), "lower_fence": round(float(lo_fence), 2),
    "upper_fence": round(float(hi_fence), 2), "removed": 0,
}}
with open(REPORTS_DIR / "data_quality_report.json", "w") as f:
    json.dump(quality_report, f, indent=2, default=str)
print(f"    Reports -> {REPORTS_DIR} (quality, missing, duplicates, outliers, before/after)")

print_header("[5/7] DONE — integration is in utils/data_loader.py; run scripts/run_tests.py")
