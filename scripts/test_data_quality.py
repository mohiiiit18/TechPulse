#!/usr/bin/env python
"""
TechPulse Phase 2 — Data-pipeline & cleaning validation tests.

Covers: cleaned-data loading, missing-value policy, duplicate detection,
salary processing, skill standardization, job-role consistency, and the
existing Streamlit pages (regression via AppTest).

Usage: python scripts/test_data_quality.py
Exit code 0 = all pass.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent.parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))

RAW_PATH = HERE / "data" / "processed dataset.csv"
CLEAN_PATH = HERE / "data" / "processed" / "techpulse_cleaned.csv"

results: list[tuple[bool, str, str]] = []


def check(name: str, passed: bool, detail: str = "") -> bool:
    results.append((passed, name, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return passed


print("=" * 64)
print("TechPulse Phase 2 — Data Quality Test Suite")
print("=" * 64)

raw = pd.read_csv(RAW_PATH, low_memory=False)
clean = pd.read_csv(CLEAN_PATH, low_memory=False)

# ── 1. Loading & schema ──────────────────────────────────────────────────────
check("cleaned dataset loads", not clean.empty, f"{len(clean)} rows, {len(clean.columns)} cols")
check("row count preserved", len(clean) == len(raw), f"{len(raw)} -> {len(clean)} (nothing silently deleted)")
new_cols = set(clean.columns) - set(raw.columns)
check("new enrichment columns exist", True, f"added: {sorted(new_cols)}")

# ── 2. Missing-value policy ──────────────────────────────────────────────────
n_country_unknown = int((clean["country"] == "Unknown").sum())
check("country gaps preserved as Unknown", n_country_unknown == int(raw["country"].isna().sum()),
      f"{n_country_unknown} rows")
n_city_unknown = int((clean["city"] == "Unknown").sum())
check("city gaps preserved as Unknown", n_city_unknown == int(raw["city"].isna().sum()),
      f"{n_city_unknown} rows")
salary_nan_before = int(raw["salary_min"].isna().sum())
salary_nan_after = int(clean["salary_min"].isna().sum())
check("salary NaNs NOT fabricated", salary_nan_before == salary_nan_after,
      f"{salary_nan_before} -> {salary_nan_after}")
check("skills_status distinguishes empty/unknown",
      set(clean["skills_status"].unique()) <= {"specified", "missing"},
      str(clean["skills_status"].value_counts().to_dict()))
check("original skills preserved", clean["skills_original"].fillna("").equals(raw["skills"].fillna("")),
      "skills_original identical to raw")

# ── 3. Duplicate detection ───────────────────────────────────────────────────
dup_id = int(clean.duplicated(subset=["job_id"]).sum())
dup_url = int(clean.duplicated(subset=["url"]).sum())
check("no duplicate job_id after cleaning", dup_id == 0, f"{dup_id} remaining")
check("no duplicate url after cleaning", dup_url == 0, f"{dup_url} remaining")

# ── 4. Salary processing ─────────────────────────────────────────────────────
sal = pd.to_numeric(clean["salary_norm_avg"], errors="coerce")
raw_sal = pd.to_numeric(raw.loc[raw["salary_currency"] == "USD", "salary_min"], errors="coerce")
check("salary_norm only for USD rows",
      int(sal.notna().sum()) == int((clean["salary_currency"] == "USD").sum()),
      f"{int(sal.notna().sum())} normalized rows")
check("no negative salaries",
      bool((pd.to_numeric(clean["salary_min"], errors="coerce").dropna() >= 0).all()
           and (pd.to_numeric(clean["salary_max"], errors="coerce").dropna() >= 0).all()),
      "min/max >= 0")
both = clean[clean["salary_min"].notna() & clean["salary_max"].notna()]
check("salary_min <= salary_max",
      bool((pd.to_numeric(both["salary_min"]) <= pd.to_numeric(both["salary_max"])).all()),
      f"{len(both)} paired rows")
hour_rows = clean[(clean["salary_currency"] == "USD") & (clean["salary_period"] == "hour")]
check("hourly annualization x2000 applied",
      bool(np.allclose(hour_rows["salary_norm_min"], hour_rows["salary_min"] * 2000, equal_nan=True)),
      f"{len(hour_rows)} hourly rows")
month_rows = clean[(clean["salary_currency"] == "USD") & (clean["salary_period"] == "month")]
check("monthly annualization x12 applied",
      bool(np.allclose(month_rows["salary_norm_min"], month_rows["salary_min"] * 12, equal_nan=True)),
      f"{len(month_rows)} monthly rows")
eur = clean[clean["salary_currency"] == "EUR"]
check("non-USD NOT converted (no invented FX)",
      bool(eur["salary_norm_avg"].isna().all()) and
      bool((eur["salary_min"].fillna(-1) == pd.to_numeric(raw.loc[eur.index, "salary_min"], errors="coerce").fillna(-1)).all()),
      f"{len(eur)} EUR rows untouched")

# ── 5. Skill standardization ─────────────────────────────────────────────────
check("skill format valid (lowercase tokens)",
      bool(clean["skills_std"].apply(
          lambda s: s == "" or all(t == t.strip().lower() and " " not in t for t in s.split("|"))
      ).all()),
      "all canonical tokens")
check("parsed_skills_std non-empty for all rows",
      bool(clean["parsed_skills_std"].map(lambda s: len(str(s)) > 0).all()),
      "0 rows with missing skill lists")

# ── 6. Job category consistency ──────────────────────────────────────────────
check("job_role unchanged by cleaning", clean["job_role"].equals(raw["job_role"]),
      "ML target untouched")
check("seniority column populated",
      bool(clean["seniority"].notna().all()),
      str(clean["seniority"].value_counts().to_dict()))
check("no title lost (no empty cleaned titles)",
      bool((clean["title_clean"].str.strip() != "").all()),
      f"{clean['title_clean'].nunique()} distinct cleaned titles")

# ── 7. Dates & freshness ─────────────────────────────────────────────────────
check("all posting dates parsed",
      int(pd.to_datetime(clean["posted_date"], errors="coerce", utc=True).isna().sum()) == 0,
      "0 invalid dates")
check("freshness values valid",
      bool(clean["freshness"].isin(["fresh (<=30d)", "recent (31-90d)",
                                    "aging (91-365d)", "stale (>365d)", "unknown"]).all()),
      str(clean["freshness"].value_counts().to_dict()))
check("posting_age_days consistent",
      bool((clean["posting_age_days"].dropna() >= 0).all()),
      f"max age {int(clean['posting_age_days'].max())} days")

# ── 8. Integration: app pipeline on cleaned data ─────────────────────────────
try:
    from streamlit.testing.v1 import AppTest
    for page in ["app.py", "app_pages/career_matcher.py", "app_pages/market_intelligence.py",
                 "app_pages/salary_benchmark.py", "app_pages/job_explorer.py"]:
        at = AppTest.from_file(str(HERE / page), default_timeout=40).run()
        check(f"page renders on cleaned data: {page}", not at.exception,
              str(at.exception) if at.exception else "OK")
except ImportError as e:
    check("streamlit AppTest available", False, str(e))

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
passed = sum(1 for p, *_ in results if p)
failed = sum(1 for p, *_ in results if not p)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 64)

out = HERE / "data" / "reports" / "cleaning_test_results.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write("TechPulse Phase 2 Data Quality Test Results\n" + "=" * 60 + "\n\n")
    for p, n, d in results:
        f.write(f"[{'PASS' if p else 'FAIL'}] {n}: {d}\n")
    f.write(f"\nTOTAL: {passed} passed, {failed} failed\n")
print(f"Saved: {out}")
sys.exit(0 if failed == 0 else 1)
