#!/usr/bin/env python
"""
TechPulse Dataset Quality Report
Analyzes the processed dataset and produces reports/data_quality_report.json.
Does NOT modify the dataset.

Usage: python scripts/data_quality_report.py
Report saved to: reports/data_quality_report.json
"""

import json
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed dataset.csv"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("TechPulse Dataset Quality Report")
print("=" * 60)

df_raw = pd.read_csv(DATA_PATH, low_memory=False)

# ─── 1. Basic stats ──────────────────────────────────────────────────────────
print("\n[1/8] Basic statistics...")
total_rows = len(df_raw)
unique_job_ids = df_raw["job_id"].nunique() if "job_id" in df_raw.columns else None
dup_count = total_rows - unique_job_ids if unique_job_ids is not None else 0
print(f"    Total rows:      {total_rows}")
print(f"    Unique job_ids:  {unique_job_ids}")
print(f"    Duplicate rows:  {dup_count}")

# ─── 2. Missing values ───────────────────────────────────────────────────────
print("\n[2/8] Missing values per column...")
missing_report = {}
for col in df_raw.columns:
    n_missing = int(df_raw[col].isna().sum())
    pct_missing = round(n_missing / total_rows * 100, 2)
    missing_report[col] = {"count": n_missing, "pct": pct_missing}
    if n_missing > 0:
        print(f"    {col:<35} {n_missing:>5} missing ({pct_missing:.1f}%)")

# ─── 3. Salary analysis ──────────────────────────────────────────────────────
print("\n[3/8] Salary analysis...")
currency_counts = {}
period_counts = {}
usd_outliers = {}

if "salary_currency" in df_raw.columns:
    currency_counts = df_raw["salary_currency"].value_counts(dropna=False).to_dict()
    print("    Currency distribution:")
    for k, v in sorted(currency_counts.items(), key=lambda x: -x[1]):
        kstr = str(k) if pd.notna(k) else "NaN"
        print(f"      {kstr:<10} {v}")

if "salary_period" in df_raw.columns:
    period_counts = df_raw["salary_period"].value_counts(dropna=False).to_dict()
    print("    Period distribution:")
    for k, v in sorted(period_counts.items(), key=lambda x: -x[1]):
        kstr = str(k) if pd.notna(k) else "NaN"
        print(f"      {kstr:<10} {v}")

# Outlier detection (IQR on USD year salaries)
if "salary_min" in df_raw.columns and "salary_currency" in df_raw.columns and "salary_period" in df_raw.columns:
    usd_year = df_raw[(df_raw["salary_currency"] == "USD") & (df_raw["salary_period"] == "year")]["salary_min"].dropna()
    if len(usd_year) > 0:
        Q1 = usd_year.quantile(0.25)
        Q3 = usd_year.quantile(0.75)
        IQR = Q3 - Q1
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR
        n_outliers = int(((usd_year < lower_fence) | (usd_year > upper_fence)).sum())
        usd_outliers = {
            "n_usd_year_records": int(len(usd_year)),
            "Q1": round(float(Q1), 2),
            "Q3": round(float(Q3), 2),
            "IQR": round(float(IQR), 2),
            "lower_fence": round(float(lower_fence), 2),
            "upper_fence": round(float(upper_fence), 2),
            "n_outliers": n_outliers,
            "pct_outliers": round(n_outliers / len(usd_year) * 100, 2)
        }
        print(f"\n    USD/year salary outliers (IQR method): {n_outliers} of {len(usd_year)}")

# Additional outlier check on annualized USD salaries (any period) - PHASE 6
if "salary_min" in df_raw.columns:
    _sal = pd.to_numeric(df_raw["salary_min"], errors="coerce")
    _usd_any = df_raw[(df_raw["salary_currency"] == "USD") & _sal.notna()]
    _vals = pd.to_numeric(_usd_any["salary_min"], errors="coerce")
    _n_implausible = int(((_vals < 1000) & (df_raw.loc[_vals.index, "salary_period"] == "year")).sum())
    print(f"    Implausible USD/year salaries (< 1,000/yr): {_n_implausible}")

# ─── 4. Top skills ───────────────────────────────────────────────────────────
print("\n[4/8] Top 20 skills...")
def parse_skills(s):
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [x.strip().lower() for x in str(s).split("|") if x.strip()]

df_raw["_parsed"] = df_raw["skills"].apply(parse_skills)
all_skills = df_raw["_parsed"].explode().dropna()
top_skills = all_skills.value_counts().head(20).to_dict()
suspicious_skills = [s for s in all_skills.unique() if len(str(s)) > 50 or not str(s).replace("-", "").replace(".", "").replace("#", "").replace("+", "").isalnum()]
print("    Top 20:", list(top_skills.keys()))
if suspicious_skills[:10]:
    print(f"    Suspicious skill values ({len(suspicious_skills)} total): {suspicious_skills[:5]}")

# ─── 5. Job role distribution ────────────────────────────────────────────────
print("\n[5/8] Job role distribution...")
role_dist = {}
if "job_role" in df_raw.columns:
    role_dist = df_raw["job_role"].value_counts(dropna=False).to_dict()
    for k, v in role_dist.items():
        print(f"    {str(k):<45} {v}")

# ─── 6. Date range and staleness ─────────────────────────────────────────────
print("\n[6/8] Date range analysis...")
date_info = {}
for date_col in ["posted_at", "scraped_at"]:
    if date_col in df_raw.columns:
        dates = pd.to_datetime(df_raw[date_col], errors="coerce", utc=True)
        n_valid = int(dates.notna().sum())
        if n_valid > 0:
            min_date = dates.min()
            max_date = dates.max()
            # Use UTC-aware cutoff to match tz-aware dates
            now = pd.Timestamp.now(tz="UTC")
            cutoff_2yr = now - pd.Timedelta(days=730)
            n_stale = int((dates < cutoff_2yr).sum())
            date_info[date_col] = {
                "n_valid": n_valid,
                "min": str(min_date.date()),
                "max": str(max_date.date()),
                "n_stale_over_2yrs": n_stale,
                "pct_stale": round(n_stale / n_valid * 100, 2)
            }
            print(f"    {date_col}: {min_date.date()} -> {max_date.date()} | Stale (>2yr): {n_stale} ({round(n_stale/n_valid*100,1)}%)")
        else:
            date_info[date_col] = {"n_valid": 0}
            print(f"    {date_col}: no valid dates")

# ─── 7. Country distribution ─────────────────────────────────────────────────
print("\n[7/8] Country distribution (top 10)...")
country_info = {}
if "country" in df_raw.columns:
    n_missing_country = int(df_raw["country"].isna().sum())
    country_counts = df_raw["country"].value_counts().head(10).to_dict()
    country_info = {"top_10": country_counts, "missing_count": n_missing_country}
    print(f"    Missing country: {n_missing_country}")
    for k, v in country_counts.items():
        print(f"    {k:<30} {v}")

# ─── 8. Remote split ─────────────────────────────────────────────────────────
print("\n[8/8] Remote vs non-remote...")
remote_info = {}
if "is_remote" in df_raw.columns:
    vc = df_raw["is_remote"].value_counts(dropna=False).to_dict()
    remote_info = {str(k): int(v) for k, v in vc.items()}
    print(f"    {remote_info}")

# ─── Build full report ────────────────────────────────────────────────────────
report = {
    "generated_at": datetime.now().isoformat(),
    "basic_stats": {
        "total_rows": total_rows,
        "unique_job_ids": unique_job_ids,
        "duplicate_rows": dup_count
    },
    "missing_values": missing_report,
    "salary_analysis": {
        "currency_distribution": {str(k): int(v) for k, v in currency_counts.items()},
        "period_distribution": {str(k): int(v) for k, v in period_counts.items()},
        "usd_year_outliers": usd_outliers
    },
    "top_20_skills": {k: int(v) for k, v in top_skills.items()},
    "suspicious_skill_values_count": len(suspicious_skills),
    "suspicious_skill_samples": suspicious_skills[:10],
    "job_role_distribution": {str(k): int(v) for k, v in role_dist.items()},
    "date_analysis": date_info,
    "country_distribution": country_info,
    "remote_split": remote_info
}

report_path = REPORTS_DIR / "data_quality_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2, default=str)

print(f"\n    ✓ Report saved: {report_path}")
print("\n" + "=" * 60)
print("DATA QUALITY REPORT COMPLETE")
print("=" * 60)
print(f"  Total rows:     {total_rows}")
print(f"  Duplicates:     {dup_count}")
print(f"  Roles:          {len(role_dist)}")
print(f"  Report:         {report_path}")
print("=" * 60)
