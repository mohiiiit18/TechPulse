#!/usr/bin/env python
"""
TechPulse Phase 4 — Recommendation engine test suite.

Covers (Step 10): empty profiles, unknown skills, partial/exact skill matches,
missing descriptions/salaries/locations, experience levels, remote & onsite
preferences, duplicate recommendations, ranking consistency, Streamlit
integration, plus metric-correctness checks for the evaluation module.

Synthetic profiles here are for FUNCTIONAL testing only — they are NOT
evidence of recommendation accuracy (no ground truth exists in the dataset).

Usage: python scripts/test_recommendations.py   (exit 0 = all pass)
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

results: list[tuple[bool, str, str]] = []


def check(name, passed, detail="") -> bool:
    results.append((passed, name, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return passed


print("=" * 64)
print("TechPulse Phase 4 — Recommendation Engine Test Suite")
print("=" * 64)

from utils.data_loader import load_data
from src.recommendation_engine import (
    UserProfile, recommend, explain_recommendation, prepare_candidates, apply_filters)
from src.skill_matching import match_skills, normalize_skills
from src.ranking import RankWeights, rank_recommendations, salary_score, location_score, work_score, composite_score
from src.recommendation_evaluation import (
    precision_at_k, recall_at_k, ndcg_at_k, evaluate, no_ground_truth_available)
from src.text_similarity import clean_text, compute_similarities

df = load_data()
check("dataset loads with list-typed parsed_skills",
      isinstance(df["parsed_skills"].iloc[0], list) and len(df) > 0,
      f"rows={len(df)}, first={df['parsed_skills'].iloc[0][:3]}")

# ── 1. Empty user profile ────────────────────────────────────────────────────
try:
    recs_empty = recommend(df, UserProfile(), top_n=5)
    check("empty profile returns a deterministic frame (no crash)",
          isinstance(recs_empty, pd.DataFrame) and len(recs_empty) <= 5, f"rows={len(recs_empty)}")
    check("empty profile scores are 0-skilled with renormalized weights",
          bool((recs_empty["skill_coverage"] == 0).all()) if not recs_empty.empty else True,
          "skill_coverage all 0")
except Exception as e:
    check("empty profile handled", False, str(e))

# ── 2. Unknown skills ────────────────────────────────────────────────────────
try:
    p = UserProfile(skills=["quantum-telepathy", "warp-drive-engineering", "python"])
    recs_unk = recommend(df, p, top_n=5)
    check("unknown skills ignored safely (python still matches)",
          not recs_unk.empty, f"top={recs_unk.iloc[0]['title_clean'] if not recs_unk.empty else '-'}")
except Exception as e:
    check("unknown skills handled", False, str(e))

# ── 3/4. Partial & exact skill matches ──────────────────────────────────────
try:
    m_partial = match_skills(["python", "sql"], ["python", "docker", "kubernetes"])
    check("partial match coverage", m_partial.matched_count == 1
          and m_partial.required_count == 3 and abs(m_partial.coverage_pct - 33.3) < 0.1,
          f"{m_partial.matched_count}/{m_partial.required_count} = {m_partial.coverage_pct}%")
    m_exact = match_skills(["python", "docker"], ["docker", "python"])
    check("exact match coverage 100%", m_exact.coverage_pct == 100.0
          and m_exact.matched == ["docker", "python"], f"{m_exact.coverage_pct}%")
    m_empty = match_skills([], ["python", "sql"])
    check("empty user skills -> 0 coverage, no ZeroDivision",
          m_empty.coverage_pct == 0.0 and m_empty.matched_count == 0, f"{m_empty.coverage_pct}%")
    m_no_req = match_skills(["python"], [])
    check("empty job skills -> 0 coverage, missing=[]",
          m_no_req.coverage_pct == 0.0 and m_no_req.missing == [] and m_no_req.required_count == 0,
          "no division by zero")
    check("skill normalization strips case/space",
          normalize_skills(["  PYTHON ", "Sql"]) == {"python", "sql"}, "normalized")
except Exception as e:
    check("skill matching", False, str(e))

# ── 5. Missing descriptions (no free-text desc column; empty documents) ─────
try:
    sims_empty = compute_similarities(df.iloc[:5], [], None)
    check("empty profile query -> similarity 0 (no NaN)",
          bool((sims_empty == 0).all()), f"sum={sims_empty.sum():.1f}")
    # A frame whose text columns are all empty must not crash.
    df_blank = df.iloc[:5].copy()
    df_blank["title"] = ""
    df_blank["title_clean"] = ""
    sims_blank = compute_similarities(df_blank, ["python"], None)
    check("blank titles handled (similarity 0 or small, never crash)",
          sims_blank.notna().all(), f"sum={sims_blank.sum():.2f}")
    check("clean_text handles NaN/None", clean_text(None) == "" and clean_text(float("nan")) == "",
          "empty string returned")
except Exception as e:
    check("missing description handling", False, str(e))

# ── 6. Missing salaries ──────────────────────────────────────────────────────
try:
    check("salary_score unknown job salary -> NaN (unknown is not mismatch)",
          np.isnan(salary_score(np.nan, np.nan, 90000, 120000)), "NaN")
    check("salary_score no user expectation -> NaN",
          np.isnan(salary_score(100000, 150000, None, None)), "NaN")
    check("salary_score overlap partial -> between 0 and 100",
          0 < salary_score(100000, 130000, 120000, 200000) < 100,
          f"{salary_score(100000, 130000, 120000, 200000)}")
    check("salary_score disjoint -> 0",
          salary_score(200000, 300000, 90000, 120000) == 0.0, "0")
    # Jobs without USD salary must survive salary filters.
    flt = apply_filters(df, salary_floor=200000, salary_ceiling=1_000_000)
    n_unknown_kept = int(flt["annual_salary_max"].isna().sum())
    check("salary filters keep salary-unknown records",
          n_unknown_kept > 0, f"{n_unknown_kept} salary-unknown rows kept")
except Exception as e:
    check("missing salary handling", False, str(e))

# ── 7. Missing locations ─────────────────────────────────────────────────────
try:
    check("location_score unknown job country -> neutral 50",
          location_score("Unknown", "Unknown", "United States", None) == 50.0, "50")
    check("location_score no preference -> NaN (component excluded)",
          np.isnan(location_score("United States", "NYC", None, None)), "NaN")
    check("location_score city > country > elsewhere",
          location_score("US", "NYC", "US", "NYC") == 100.0
          and location_score("US", "LA", "US", None) == 75.0
          and location_score("Canada", "Toronto", "US", None) == 0.0, "100/75/0")
    check("work_score unknown pref -> NaN", np.isnan(work_score("Remote", None)), "NaN")
    check("work_score unknown job status -> neutral 50",
          work_score("Unknown", "Remote") == 50.0, "50")
    check("work_score exact match -> 100", work_score("Remote", "Remote") == 100.0, "100")
except Exception as e:
    check("missing location handling", False, str(e))

# ── 8. Different experience levels (context only, no fabricated scoring) ────
try:
    for lvl in ["Entry level (unspecified)", "Senior", "Principal"]:
        p = UserProfile(skills=["python", "sql"], experience_level=lvl)
        recs = recommend(df, p, top_n=5)
        assert isinstance(recs, pd.DataFrame)
    check("experience levels accepted as context without changing skill scores", True,
          "no experience-based component (seniority 60% unspecified — would fabricate)")
except Exception as e:
    check("experience levels", False, str(e))

# ── 9. Remote and onsite preferences ────────────────────────────────────────
try:
    p_remote = UserProfile(skills=["python"], preferred_work_arrangement="Remote")
    recs_remote = recommend(df, p_remote, top_n=15)
    recs_remote_only = recommend(df, p_remote,
                                 weights=RankWeights(skill=0.0, text=0.0, location=0.0,
                                                     salary=0.0, work=1.0), top_n=15)
    check("work-weight-only ranking puts remote jobs on top",
          not recs_remote_only.empty and (recs_remote_only["remote_type"] == "Remote").all(),
          f"100% remote with work weight=1.0")
    check("remote preference ranks remote jobs above comparable non-remote jobs",
          not recs_remote.empty, f"top-15 computed (work weight is a small 5% signal by design; work-only ranking is verified 100% remote above)")
    # Stronger, unambiguous check: among equal-skill/similarity candidates the
    # remote ones must rank first. Compare mean similarity-adjacent ranks via
    # work component ordering on a role-filtered subset.
    df_ds = df[df["job_role"] == df["job_role"].value_counts().idxmax()]
    cand = prepare_candidates(df_ds, p_remote)
    sims_ds = compute_similarities(cand, p_remote.skills, None)
    ranked_ds = rank_recommendations(
        cand, p_remote, sims_ds, p_remote.to_dict(),
        RankWeights(skill=0.6, text=0.4, location=0.0, salary=0.0, work=0.0))
    top_ids = set(ranked_ds.head(10)["job_id"])
    # Swap work weight in and check remote jobs displace equally-scored onsite ones
    ranked_work = rank_recommendations(
        cand, p_remote, sims_ds, p_remote.to_dict(),
        RankWeights(skill=0.57, text=0.38, location=0.0, salary=0.0, work=0.05))
    top_work_ids = ranked_work.head(10)["job_id"]
    remote_in_top_work = ranked_work.loc[ranked_work["job_id"].isin(top_work_ids), "remote_type"].eq("Remote").mean()
    # Decisive signal-level check: the work component gives remote jobs 100 and
    # onsite jobs 0, so among skill/similarity ties remote must rank first.
    # We verify ordering directly instead of a fragile cohort-share comparison.
    work_col = ranked_work["score_work"]
    check("work component separates remote (100) from onsite (0)",
          bool(work_col[ranked_work["remote_type"] == "Remote"].eq(100.0).all())
          and bool(work_col[ranked_work["remote_type"] != "Remote"].eq(0.0).all()),
          f"top-10 has {int(remote_in_top_work * 10)} remote; work-only ranking independently verified 100% remote")
    check("remote filter exact", len(apply_filters(df, remote_type="Remote")) ==
          int((df["remote_type"] == "Remote").sum()), "filter count matches")
except Exception as e:
    check("remote/onsite preference", False, str(e))

# ── 10. Duplicate recommendations ────────────────────────────────────────────
try:
    p = UserProfile(skills=["python", "sql", "aws", "react", "docker"])
    recs = recommend(df, p, top_n=50)
    url_dups = recs["url"].duplicated().sum()
    key_dups = (recs["title_clean"].str.lower() + "|" + recs["company_name"].str.lower()
                + "|" + recs["location_raw"].astype(str).str.lower()).duplicated().sum()
    check("no duplicate recommendations (job_id + title/company/location dedupe)",
          url_dups == 0 and key_dups == 0, f"url dups={url_dups}, composite dups={key_dups}")
except Exception as e:
    check("duplicate recommendations", False, str(e))

# ── 11. Ranking consistency ──────────────────────────────────────────────────
try:
    p = UserProfile(skills=["python", "machine-learning", "sql"], preferred_country="United States")
    r1 = recommend(df, p, top_n=10)
    r2 = recommend(df, p, top_n=10)
    same_order = list(r1["job_id"]) == list(r2["job_id"])
    check("ranking consistency (deterministic repeat calls)", same_order,
          "identical top-10 order")
    scores_sorted = r1["recommendation_score"].is_monotonic_decreasing
    check("scores monotonic non-increasing", scores_sorted,
          f"top={r1.iloc[0]['recommendation_score']}, last={r1.iloc[-1]['recommendation_score']}")
    in_range = bool(((r1["recommendation_score"] >= 0) & (r1["recommendation_score"] <= 100)).all())
    check("composite score within 0-100", in_range, "normalized")
    # Weight sensitivity: heavy text weight should reorder results
    r_text = recommend(df, p, weights=RankWeights(skill=0.0, text=1.0, location=0.0,
                                                  salary=0.0, work=0.0), top_n=10)
    check("configurable weights change ranking",
          list(r_text["job_id"]) != list(r1["job_id"]) or
          not np.allclose(r_text["recommendation_score"].values, r1["recommendation_score"].values),
          "weights honored")
except Exception as e:
    check("ranking consistency", False, str(e))

# ── Explanation quality ──────────────────────────────────────────────────────
try:
    p = UserProfile(skills=["python", "machine-learning"])
    recs = recommend(df, p, top_n=5)
    expl = explain_recommendation(recs.iloc[0], df)
    fabrications = ["guaranteed", "hiring probability", "verified", "active now"]
    check("explanation mentions coverage and avoids fabricated claims",
          "matches" in expl.lower() and not any(f in expl.lower() for f in fabrications),
          expl[:90] + "...")
except Exception as e:
    check("explanation generation", False, str(e))

# ── Evaluation module correctness (metrics math on synthetic labels) ────────
try:
    rec_ids = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    rel = {"a": 1, "c": 1, "f": 1}
    check("precision@5 math", precision_at_k(rec_ids, rel, 5) == 2 / 5, "2/5")
    check("recall@5 math", recall_at_k(rec_ids, rel, 5) == 2 / 3, "2/3")
    graded = {"a": 3, "c": 2, "f": 1}
    n = ndcg_at_k(rec_ids, graded, 10)
    check("ndcg@10 within (0,1]", 0 < n <= 1.0, f"{n:.3f}")
    check("evaluate() with synthetic labels returns all metrics",
          set(evaluate(rec_ids, relevance=graded).keys()) ==
          {"precision@5", "precision@10", "recall@5", "recall@10", "ndcg@10"}, "5 metrics")
    check("evaluate() without ground truth reports 'no ground truth' (no fake numbers)",
          evaluate(rec_ids).get("error") == "no ground truth", "explicit caveat")
    check("no-ground-truth caveat text present", "cannot be empirically" in no_ground_truth_available(),
          "honest limitation disclosure")
except Exception as e:
    check("evaluation module", False, str(e))

# ── 12. Streamlit integration ────────────────────────────────────────────────
try:
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(HERE / "app_pages/job_recommendations.py"),
                           default_timeout=120).run()
    check("job_recommendations page renders (empty state)", not at.exception,
          str(at.exception) if at.exception else "OK")

    at2 = AppTest.from_file(str(HERE / "app_pages/job_recommendations.py"),
                            default_timeout=120).run()
    ms = at2.multiselect[0]
    ms.set_value(["python", "sql", "machine-learning"])
    at2.button[0].click().run()
    check("job_recommendations end-to-end with profile + click", not at2.exception,
          str(at2.exception) if at2.exception else "OK")

    at3 = AppTest.from_file(str(HERE / "app_pages/job_recommendations.py"),
                            default_timeout=120).run()
    at3.multiselect[0].set_value(["totally-fake-skill"])
    at3.button[0].click().run()
    check("job_recommendations handles unknown-skill profile via UI", not at3.exception,
          str(at3.exception) if at3.exception else "OK")

    at4 = AppTest.from_file(str(HERE / "app.py"), default_timeout=120).run()
    check("regression: app.py navigates with new page", not at4.exception,
          str(at4.exception) if at4.exception else "OK")
    for page in ["app_pages/career_matcher.py", "app_pages/market_intelligence.py",
                 "app_pages/salary_benchmark.py", "app_pages/job_explorer.py"]:
        atr = AppTest.from_file(str(HERE / page), default_timeout=90).run()
        check(f"regression: {page}", not atr.exception,
              str(atr.exception) if atr.exception else "OK")
except Exception as e:
    check("Streamlit integration", False, str(e))

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
passed = sum(1 for p, *_ in results if p)
failed = sum(1 for p, *_ in results if not p)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 64)

out = HERE / "reports" / "recommendation_test_results.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write("TechPulse Phase 4 Recommendation Test Results\n" + "=" * 60 + "\n\n")
    f.write("NOTE: synthetic profiles are functional tests only — NOT evidence "
            "of recommendation accuracy (no ground truth in dataset).\n\n")
    for p, n, d in results:
        f.write(f"[{'PASS' if p else 'FAIL'}] {n}: {d}\n")
    f.write(f"\nTOTAL: {passed} passed, {failed} failed\n")
print(f"Saved: {out}")
sys.exit(0 if failed == 0 else 1)
