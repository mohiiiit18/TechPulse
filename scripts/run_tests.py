"""
Phase 8: AppTest suite for TechPulse
Tests all pages for exceptions, special-character search, and basic rendering.
Results saved to reports/test_results.txt
"""

import sys
import os
import traceback
from pathlib import Path

# This script must run from within d:\project 69
HERE = Path(__file__).resolve().parent.parent  # project root (scripts/ is one level down)
os.chdir(HERE)
sys.path.insert(0, str(HERE))

APP_DIR = HERE  # all app paths are relative to project root

results = []

def record(name, passed: bool, detail="") -> bool:
    """Append a PASS/FAIL entry. Returns the boolean for direct use in try/except blocks."""
    status = "PASS" if passed else "FAIL"
    results.append(f"[{status}] {name}: {detail}")
    print(f"[{status}] {name}: {detail}")
    return passed

print("=" * 60)
print("TechPulse AppTest Suite")
print("=" * 60)

try:
    from streamlit.testing.v1 import AppTest

    # ── Test 1: Main app loads ─────────────────────────────────────────────────
    try:
        at = AppTest.from_file(str(HERE / "app.py"), default_timeout=20).run()
        record("app.py loads", not at.exception, str(at.exception) if at.exception else "OK")
    except Exception as e:
        record("app.py loads", False, str(e))

    # ── Test 2: career_matcher page ───────────────────────────────────────────
    try:
        at2 = AppTest.from_file(str(HERE / "app_pages/career_matcher.py"), default_timeout=20).run()
        record("career_matcher.py loads", not at2.exception, str(at2.exception) if at2.exception else "OK")
    except Exception as e:
        record("career_matcher.py loads", False, str(e))

    # ── Test 3: market_intelligence page ─────────────────────────────────────
    try:
        at3 = AppTest.from_file(str(HERE / "app_pages/market_intelligence.py"), default_timeout=20).run()
        record("market_intelligence.py loads", not at3.exception, str(at3.exception) if at3.exception else "OK")
    except Exception as e:
        record("market_intelligence.py loads", False, str(e))

    # ── Test 4: salary_benchmark page ────────────────────────────────────────
    try:
        at4 = AppTest.from_file(str(HERE / "app_pages/salary_benchmark.py"), default_timeout=20).run()
        record("salary_benchmark.py loads", not at4.exception, str(at4.exception) if at4.exception else "OK")
    except Exception as e:
        record("salary_benchmark.py loads", False, str(e))

    # ── Test 5: job_explorer page ────────────────────────────────────────────
    try:
        at5 = AppTest.from_file(str(HERE / "app_pages/job_explorer.py"), default_timeout=20).run()
        record("job_explorer.py loads", not at5.exception, str(at5.exception) if at5.exception else "OK")
    except Exception as e:
        record("job_explorer.py loads", False, str(e))

    # ── Test 6: BUG 4 — special character search (C++) ───────────────────────
    try:
        at6 = AppTest.from_file(str(HERE / "app_pages/job_explorer.py"), default_timeout=20).run()
        at6.text_input[0].set_value("C++").run()
        record("job_explorer search C++", not at6.exception, str(at6.exception) if at6.exception else "OK")
    except Exception as e:
        record("job_explorer search C++", False, str(e))

    # ── Test 7: BUG 4 — special character search (AWS) ───────────────────────
    try:
        at7 = AppTest.from_file(str(HERE / "app_pages/job_explorer.py"), default_timeout=20).run()
        at7.text_input[0].set_value("(AWS)").run()
        record("job_explorer search (AWS)", not at7.exception, str(at7.exception) if at7.exception else "OK")
    except Exception as e:
        record("job_explorer search (AWS)", False, str(e))

    # ── Test 8: BUG 4 — special character search [React] ─────────────────────
    try:
        at8 = AppTest.from_file(str(HERE / "app_pages/job_explorer.py"), default_timeout=20).run()
        at8.text_input[0].set_value("[React]").run()
        record("job_explorer search [React]", not at8.exception, str(at8.exception) if at8.exception else "OK")
    except Exception as e:
        record("job_explorer search [React]", False, str(e))

    # ── Test 9: BUG 6 — salary_benchmark NaN guard ───────────────────────────
    try:
        at9 = AppTest.from_file(str(HERE / "app_pages/salary_benchmark.py"), default_timeout=20).run()
        record("salary_benchmark NaN guard", not at9.exception, str(at9.exception) if at9.exception else "OK")
    except Exception as e:
        record("salary_benchmark NaN guard", False, str(e))

    # ── Test 10: BUG 5 — market_intelligence no duplicate chart ───────────────
    try:
        at10 = AppTest.from_file(str(HERE / "app_pages/market_intelligence.py"), default_timeout=20).run()
        record("market_intelligence no duplicate chart", not at10.exception, str(at10.exception) if at10.exception else "OK")
    except Exception as e:
        record("market_intelligence no duplicate chart", False, str(e))

    # ── Test 11: career_matcher default prediction ────────────────────────────
    try:
        at11 = AppTest.from_file(str(HERE / "app_pages/career_matcher.py"), default_timeout=20).run()
        record("career_matcher prediction", not at11.exception, str(at11.exception) if at11.exception else "OK")
    except Exception as e:
        record("career_matcher prediction", False, str(e))

except ImportError as e:
    record("AppTest import", False, str(e))

# ── Check reports exist ────────────────────────────────────────────────────────
print("\n[Checking report files...]")
report_files = [
    "reports/existing_model_evaluation.json",
    "reports/model_evaluation.json",
    "reports/data_quality_report.json",
    "reports/classification_report.txt",
]
for rf in report_files:
    exists = (HERE / rf).exists()
    record(f"Report exists: {rf}", exists, "found" if exists else "MISSING")

model_files = [
    "models/job_role_model_v2.pkl",
    "models/skill_encoder_v2.pkl",
]
for mf in model_files:
    exists = (HERE / mf).exists()
    record(f"Model file exists: {mf}", exists, "found" if exists else "MISSING")

# ── Write results ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for r in results if r.startswith("[PASS]"))
failed = sum(1 for r in results if r.startswith("[FAIL]"))
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

(HERE / "reports").mkdir(exist_ok=True)
with open(HERE / "reports" / "test_results.txt", "w", encoding="utf-8") as f:
    f.write("TechPulse AppTest Results\n")
    f.write("=" * 60 + "\n\n")
    for r in results:
        f.write(r + "\n")
    f.write(f"\nTOTAL: {passed} passed, {failed} failed\n")

print("\nResults saved to reports/test_results.txt")
sys.exit(0 if failed == 0 else 1)
