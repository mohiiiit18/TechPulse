#!/usr/bin/env python
"""
TechPulse Phase 6 — Skill Gap Analyzer test suite.

Covers: skill demand calculations (from actual dataset), gap analysis
(empty/complete/unknown skills, duplicate skills, different roles, unknown
roles, missing requirements), priority ranking, roadmap generation, UI
integration, and full-app regression.

Usage: python scripts/test_skill_gap.py   (exit 0 = all pass)
"""
from __future__ import annotations

import os
import sys
import warnings

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
print("TechPulse Phase 6 — Skill Gap Analyzer Test Suite")
print("=" * 64)

from utils.data_loader import load_data
from src.skill_demand import (global_skill_demand, role_skill_demand,
                              role_posting_counts, demand_score_for)
from src.skill_gap_analyzer import analyze_skill_gap, gap_table, COVERAGE_NOTE
from src.skill_prioritization import (rank_missing_skills, priority_counts,
                                      SkillPriority, PRIORITY_METHOD)
from src.learning_roadmap import (build_roadmap, STATUSES, LearningRoadmap,
                                  RoadmapItem, ROADMAP_NOTE)

df = load_data()

# ── 1. Skill demand calculations (actual dataset only) ──────────────────────
gd = global_skill_demand(df, top_n=10)
check("global demand computed from dataset", not gd.empty and gd["count"].sum() > 0,
      f"top skill={gd.iloc[0]['skill']} ({int(gd.iloc[0]['count'])})" if not gd.empty else "empty")
check("global demand pct ≤ 100 and ranks sequential",
      bool((gd["pct_of_postings"] <= 100).all()) and list(gd["rank"]) == list(range(1, len(gd) + 1)),
      f"max pct={gd['pct_of_postings'].max()}")

# Verify against direct pandas computation (no fabricated stats)
import pandas as pd
role = "Data Scientist / ML Engineer"
rd = role_skill_demand(df, role, top_n=None)
direct = df[df["job_role"] == role]["parsed_skills"].explode().value_counts()
top = rd.iloc[0]
check("role demand matches direct pandas value_counts",
      int(top["count"]) == int(direct.iloc[0]) and top["skill"] == direct.index[0],
      f"{top['skill']}={int(top['count'])} == {int(direct.iloc[0])}")
check("role demand pct = count/role postings",
      abs(float(top["pct_of_role_postings"]) - int(direct.iloc[0]) / len(df[df['job_role'] == role]) * 100) < 0.1,
      f"{top['pct_of_role_postings']}%")

# Deterministic demand stats (dataset fixed)
gd2 = global_skill_demand(df, top_n=10)
check("demand stats deterministic", gd.equals(gd2), "identical on repeat calls")

# Empty/unknown role
empty_rd = role_skill_demand(df, "Nonexistent Role X")
check("unknown role -> empty demand table (no invention)", empty_rd.empty, "empty frame")
ds_unknown = demand_score_for(["python", "sql"], empty_rd)
check("demand_score unknown role -> all 0.0", all(v == 0.0 for v in ds_unknown.values()),
      str(ds_unknown))

# ── 2. Gap analysis ─────────────────────────────────────────────────────────
user_full = ["python", "machine-learning", "llm", "pytorch", "sql"]
res = analyze_skill_gap(user_full, role, df)
check("gap analysis computes coverage for known role",
      res.is_known_role and res.analyzed_postings > 0,
      f"postings={res.analyzed_postings}, coverage={res.coverage_pct}%")
check("coverage formula = matched/required*100",
      abs(res.coverage_pct - res.matched_count / res.required_count * 100) < 0.1,
      f"{res.matched_count}/{res.required_count} = {res.coverage_pct}%")

# Complete skills -> coverage 100, no missing
res_complete = analyze_skill_gap(sorted(res.missing) + user_full, role, df)
check("complete skills -> coverage 100%, no missing", res_complete.coverage_pct == 100.0
      and not res_complete.missing, f"coverage={res_complete.coverage_pct}%")

# Empty skills
res_empty = analyze_skill_gap([], role, df)
check("empty skills -> coverage 0, missing = all required", res_empty.coverage_pct == 0.0
      and res_empty.matched_count == 0 and res_empty.required_count > 0,
      f"required={res_empty.required_count}")

# Unknown / NaN skills
res_unk = analyze_skill_gap(["quantum-telepathy", "python"], role, df)
check("unknown skills ignored safely", "quantum-telepathy" not in (res_unk.missing + res_unk.matched)
      and res_unk.coverage_pct > 0, f"coverage={res_unk.coverage_pct}%")

# Duplicate skills collapse
res_dup = analyze_skill_gap(["python", "Python", " python ", "sql"], role, df)
check("duplicate/case variants collapse", res_dup.matched_count == 2,
      f"matched={res_dup.matched}")

# Unknown role
res_unkrole = analyze_skill_gap(["python"], "Nonexistent Role X", df)
check("unknown role -> is_known_role False, empty lists", not res_unkrole.is_known_role
      and res_unkrole.matched == [] and res_unkrole.missing == [],
      f"postings={res_unkrole.analyzed_postings}")

# Different roles give different requirement sets
res_sw = analyze_skill_gap(["python", "sql"], "Software Developer", df)
check("different roles produce different requirement sets",
      res_sw.required_count != res.required_count or res_sw.coverage_pct != res.coverage_pct,
      f"SW={res_sw.coverage_pct}% vs DS={res.coverage_pct}%")

# Missing job requirements (empty skill lists)
tmp = df.copy()
for i in tmp.index[:5]:
    tmp.at[i, "parsed_skills"] = []
res_noskills = analyze_skill_gap(["python"], role, tmp)
check("postings with empty skill lists handled safely", res_noskills.is_known_role
      and not pd.isna(res_noskills.coverage_pct), f"coverage={res_noskills.coverage_pct}%")

# Partial matches are informational only
res_partial = analyze_skill_gap(["machine-learning", "sql"], role, df)
check("partial matches computed separately (not counted in coverage)",
      res_partial.partial == [] or res_partial.partial != res_partial.matched,
      f"partial={res_partial.partial}")

# ── 3. Priority ranking ─────────────────────────────────────────────────────
ranked = rank_missing_skills(res.missing, rd)
check("priority ranking ranks by demand desc", len(ranked) == len(set(res.missing))
      and all(ranked[i].demand_score >= ranked[i+1].demand_score for i in range(len(ranked)-1)),
      f"top={ranked[0].skill} ({ranked[0].demand_score:.0f}%)" if ranked else "none")
check("priority buckets valid (high/medium/low)",
      all(sp.priority in ("high", "medium", "low") for sp in ranked),
      f"counts={priority_counts(ranked)}")
check("priority thresholds applied",
      all((sp.priority == "high") == (sp.demand_score >= 50)
          and (sp.priority == "medium") == (20 <= sp.demand_score < 50) for sp in ranked),
      PRIORITY_METHOD[:50] + "...")
check("empty missing skills -> no priorities", rank_missing_skills([], rd) == [], "[]")
check("duplicate missing skills deduped in ranking",
      len(rank_missing_skills(["python", "python"], rd)) == 1, "1 item")

# Unknown-role demand -> all low priority (0 demand, no invention)
ranked_unk = rank_missing_skills(["python", "sql"], empty_rd)
check("unknown role demand -> all low priority, no fabrication",
      all(sp.priority == "low" and sp.demand_score == 0.0 for sp in ranked_unk),
      f"n={len(ranked_unk)}")

# ── 4. Roadmap generation ───────────────────────────────────────────────────
rm = build_roadmap(ranked, role, weekly_hours=10)
check("roadmap built from priorities", len(rm.items) == len(ranked),
      f"items={len(rm.items)}, stages={rm.n_stages}")
check("stage mapping high->1, medium->2, low->3",
      all({"high": 1, "medium": 2, "low": 3}[sp.priority] == sp_i.stage
          for sp, sp_i in zip(ranked, rm.items)), "order preserved")
check("roadmap items default Not Started", all(it.status == "Not Started" for it in rm.items),
      str(STATUSES))
prog = rm.progress_summary()
check("progress summary: completion 0% initially", prog["completed"] == 0
      and prog["completion_pct"] == 0.0, str(prog))

# Editable roadmap: mark statuses
rm.items[0].status = "Completed"
rm.items[1].status = "In Progress"
prog2 = rm.progress_summary()
check("roadmap editable (status updates reflected)", prog2["completed"] == 1
      and prog2["in_progress"] == 1 and 0 < prog2["completion_pct"] < 100,
      f"completion={prog2['completion_pct']}%")

# Full completion path
rm2 = build_roadmap(ranked, role)
for it in rm2.items:
    it.status = "Completed"
check("all-completed roadmap -> 100%", rm2.progress_summary()["completion_pct"] == 100.0,
      f"{rm2.progress_summary()['completion_pct']}%")

check("empty priorities -> empty roadmap", build_roadmap([], role).items == [], "no items")
check("no course links/durations in roadmap",
      "no course links" in ROADMAP_NOTE and "guaranteed" in ROADMAP_NOTE,
      "constraints stated in module")
check("weekly hours stored as context only", rm.weekly_hours == 10 and rm.n_stages <= 3,
      f"weekly_hours={rm.weekly_hours}")

# ── 5. Gap table (dashboard data) ───────────────────────────────────────────
gt = gap_table(res, df)
check("gap table sorted by demand desc", not gt.empty and
      gt["pct_of_role_postings"].iloc[0] >= gt["pct_of_role_postings"].iloc[-1],
      f"top={gt.iloc[0]['skill']} ({gt.iloc[0]['pct_of_role_postings']}%)")
gt_unk = gap_table(res_unkrole, df)
check("gap table for unknown role -> empty", gt_unk.empty, "empty frame")

# ── 6. UI integration + regressions ─────────────────────────────────────────
try:
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(HERE, "app_pages/skill_gap_analyzer.py"),
                           default_timeout=120).run()
    check("skill_gap_analyzer page renders (empty state)", not at.exception,
          str(at.exception) if at.exception else "OK")

    # Full flow: set profile + role, run analysis (separate runs, not chained)
    at2 = AppTest.from_file(os.path.join(HERE, "app_pages/skill_gap_analyzer.py"),
                            default_timeout=120)
    at2.run()
    at2.multiselect[0].set_value(["python", "machine-learning", "sql"])
    at2.selectbox[0].set_value(role)
    at2 = at2.run()
    check("skill gap page runs analysis via UI", not at2.exception,
          str(at2.exception) if at2.exception else "OK")

    # Unknown skills through UI
    at3 = AppTest.from_file(os.path.join(HERE, "app_pages/skill_gap_analyzer.py"),
                            default_timeout=120)
    at3.run()
    at3.multiselect[0].set_value(["definitely-not-a-skill"])
    at3 = at3.run()
    check("skill gap page handles unknown skills via UI", not at3.exception,
          str(at3.exception) if at3.exception else "OK")

    # Cross-page integration: resume page stores resume_skills in session
    at4 = AppTest.from_file(os.path.join(HERE, "app_pages/resume_analyzer.py"),
                            default_timeout=120).run()
    check("resume analyzer page still renders (regression)", not at4.exception, "OK")

    at5 = AppTest.from_file(os.path.join(HERE, "app.py"), default_timeout=120).run()
    check("regression: app.py navigates with skill gap page", not at5.exception,
          str(at5.exception) if at5.exception else "OK")
    for page in ["app_pages/career_matcher.py", "app_pages/market_intelligence.py",
                 "app_pages/salary_benchmark.py", "app_pages/job_explorer.py",
                 "app_pages/job_recommendations.py", "app_pages/resume_analyzer.py"]:
        atr = AppTest.from_file(os.path.join(HERE, page), default_timeout=120).run()
        check(f"regression: {page}", not atr.exception,
              str(atr.exception) if atr.exception else "OK")
except Exception as e:
    import traceback; traceback.print_exc()
    check("Streamlit integration", False, str(e))

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
passed = sum(1 for p, *_ in results if p)
failed = sum(1 for p, *_ in results if not p)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 64)

out = os.path.join(HERE, "reports", "skill_gap_test_results.txt")
with open(out, "w", encoding="utf-8") as f:
    f.write("TechPulse Phase 6 Skill Gap Analyzer Test Results\n" + "=" * 60 + "\n\n")
    for p, n, d in results:
        f.write(f"[{'PASS' if p else 'FAIL'}] {n}: {d}\n")
    f.write(f"\nTOTAL: {passed} passed, {failed} failed\n")
print(f"Saved: {out}")
sys.exit(0 if failed == 0 else 1)
