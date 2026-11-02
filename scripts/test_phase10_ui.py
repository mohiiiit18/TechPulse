#!/usr/bin/env python
"""UI smoke test: render all active application pages headless (AppTest)."""
from streamlit.testing.v1 import AppTest
import sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PAGES = [
    "app_pages/overview.py",
    "app_pages/career_matcher.py",
    "app_pages/job_recommendations.py",
    "app_pages/resume_analyzer.py",
    "app_pages/skill_gap_analyzer.py",
    "app_pages/salary_benchmark.py",
    "app_pages/salary_prediction.py",
    "app_pages/market_intelligence.py",
    "app_pages/job_explorer.py",
    "app_pages/career_assistant.py",
]

results = []
for rel in PAGES:
    path = str(ROOT / rel)
    try:
        at = AppTest.from_file(path, default_timeout=60)
        at.run()
        errs = [e.message for e in at.exception]
        results.append((path, "OK" if not errs else f"EXCEPTION: {errs[0][:180]}"))
    except Exception as e:
        results.append((path, f"RUN ERROR: {e}"))

for p, r in results:
    print(f"[{'PASS' if r == 'OK' else 'FAIL'}] {p}: {r}")
n_fail = sum(1 for _, r in results if r != "OK")
print(f"RESULTS: {len(results) - n_fail} passed, {n_fail} failed")
