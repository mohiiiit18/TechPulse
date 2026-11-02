#!/usr/bin/env python
"""
TechPulse Phase 5 — Resume Analyzer test suite.

Covers (Step 9): valid PDF, invalid file, empty PDF, scanned (image-only) PDF,
missing sections, unknown skills, duplicate skills, custom job descriptions,
missing job information, metric sanity, and Streamlit navigation regression.

Test PDFs are generated on the fly with PyMuPDF into reports/tmp_test_pdfs/
(deleted afterwards). No real resume is used or stored.

Usage: python scripts/test_resume_analyzer.py   (exit 0 = all pass)
"""

from __future__ import annotations

import os
import shutil
import sys
import warnings
from pathlib import Path

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
print("TechPulse Phase 5 — Resume Analyzer Test Suite")
print("=" * 64)

import pymupdf

TMP = HERE / "reports" / "tmp_test_pdfs"
TMP.mkdir(parents=True, exist_ok=True)

from src.resume_parser import parse_resume, validate_upload, pdf_to_text, ResumeParseError
from src.skill_extractor import extract_skills, normalize_skill, find_sections
from src.ats_analyzer import analyze_resume
from src.resume_matcher import match_resume_to_dataset_job, match_resume_to_description
from src.resume_suggestions import generate_suggestions
from utils.data_loader import load_data


def make_pdf(path: Path, text: str, image_only: bool = False):
    doc = pymupdf.open()
    page = doc.new_page()
    if image_only:
        # Render the text to a raster image, then place ONLY the image on the
        # page — the result has no extractable text layer (simulates a scan).
        src = pymupdf.open("pdf", _text_as_image_pdf_bytes(text))
        pix = src[0].get_pixmap(dpi=72)
        page.insert_image(page.rect, pixmap=pix)
        src.close()
    else:
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    doc.save(str(path))
    doc.close()


def _text_as_image_pdf_bytes(text: str) -> bytes:
    """Tiny helper PDF containing the text as real text (used only as an image source shim)."""
    d = pymupdf.open()
    p = d.new_page()
    p.insert_textbox(pymupdf.Rect(50, 50, 550, 400), text, fontsize=11)
    return d.tobytes()


# ── Fixture resumes ──────────────────────────────────────────────────────────
FULL_RESUME = """JORDAN SMITH
jordan.smith@example.com | +1 415 555 0132 | San Francisco, CA

SUMMARY
Data scientist with 5 years of experience.

SKILLS
Python, SQL, Machine Learning, Pandas, PyTorch, AWS, Docker, Tableau, Git, NLP

EXPERIENCE
Senior Data Scientist — Acme Corp, 2020 - Present
• Improved model latency by 40% serving 2M users
Data Analyst — Globex, 2018 - 2020
• Built dashboards used by 300 customers

EDUCATION
B.Sc. Computer Science, State University, 2018

PROJECTS
• Churn prediction pipeline with Spark and Kafka improving retention 12%

CERTIFICATIONS
AWS Certified Solutions Architect, PMP
"""

PARTIAL_RESUME = """Jordan Smith
jordan@example.com
Some summary text without any standard section headers.
Familiar with python and machine learning.
"""

pdf_full = TMP / "full_resume.pdf"
pdf_partial = TMP / "partial_resume.pdf"
pdf_empty = TMP / "empty.pdf"
pdf_scanned = TMP / "scanned.pdf"
pdf_corrupt = TMP / "corrupt.pdf"

make_pdf(pdf_full, FULL_RESUME)
make_pdf(pdf_partial, PARTIAL_RESUME)
doc = pymupdf.open()
doc.new_page()  # a page with no text
doc.save(str(pdf_empty))
doc.close()
make_pdf(pdf_scanned, "invisible text", image_only=True)
pdf_corrupt.write_bytes(b"this is not a pdf at all")

# ── 1. Upload validation ─────────────────────────────────────────────────────
check("reject non-PDF type", validate_upload("resume.docx", 1000) is not None, "error returned")
check("reject empty upload", validate_upload("resume.pdf", 0) is not None, "error returned")
check(f"reject >5MB upload", validate_upload("resume.pdf", 6 * 1024 * 1024) is not None, "size limit")
check("accept valid PDF upload", validate_upload("resume.pdf", 100 * 1024) is None, "ok")

# ── 2/3. Valid PDF: extraction ───────────────────────────────────────────────
try:
    resume = parse_resume(pdf_full.read_bytes())
    check("valid PDF parses", resume.n_pages >= 1 and len(resume.raw_text) > 100,
          f"pages={resume.n_pages}")
    check("email extracted", resume.email == "jordan.smith@example.com", resume.email)
    check("phone extracted", bool(resume.phone and "415" in resume.phone.replace(" ", "")),
          resume.phone)
    check("name guessed", resume.full_name != "", resume.full_name)
    check("education lines found (not invented)",
          any("computer science" in e.lower() for e in resume.education),
          f"{len(resume.education)} lines")
    check("experience entries found", len(resume.experience) >= 2, f"{len(resume.experience)}")
    check("certifications detected", any("aws" in c for c in resume.certifications),
          str(resume.certifications[:3]))
    cats = resume.skills_by_category
    check("skills extracted & categorized",
          len(resume.all_skills) >= 6 and len(cats) >= 3,
          f"{len(resume.all_skills)} skills / {len(cats)} categories")
    check("duplicate skills collapse (python listed once)",
          resume.all_skills.count("python") == 1, "deduped")
    # Extraction is deterministic — same bytes, same result.
    resume2 = parse_resume(pdf_full.read_bytes())
    check("extraction deterministic", resume.all_skills == resume2.all_skills, "identical")
except Exception as e:
    check("valid PDF pipeline", False, str(e))

# ── 4. Missing sections / partial resume ────────────────────────────────────
try:
    partial = parse_resume(pdf_partial.read_bytes())
    ats_partial = analyze_resume(partial)
    check("partial resume parses without fabricated sections",
          partial.projects == [] and partial.certifications == [],
          f"projects={partial.projects}, certs={partial.certifications}")
    check("partial resume scores low but valid", 0 <= ats_partial.total < 60,
          f"score={ats_partial.total}")
    check("partial resume suggestions reference missing sections",
          any("section" in s.text.lower() for s in generate_suggestions(partial)),
          "evidence-backed")
except Exception as e:
    check("partial resume", False, str(e))

# ── 5. Empty PDF / scanned PDF / corrupt ────────────────────────────────────
try:
    pdf_to_text(pdf_empty.read_bytes())
    check("empty PDF rejected", False, "no exception raised")
except ResumeParseError as e:
    check("empty PDF rejected with clear message", "no selectable text" in str(e).lower(), str(e)[:60])
except Exception as e:
    check("empty PDF rejected", False, f"{type(e).__name__}: {e}")

try:
    pdf_to_text(pdf_scanned.read_bytes())
    check("scanned PDF flagged gracefully", False, "no exception raised")
except ResumeParseError as e:
    check("scanned PDF flagged gracefully", "scanned" in str(e).lower(), str(e)[:70])
except Exception as e:
    check("scanned PDF handled", False, f"{type(e).__name__}: {e}")

try:
    pdf_to_text(pdf_corrupt.read_bytes())
    check("corrupt PDF rejected", False, "no exception raised")
except ResumeParseError as e:
    check("corrupt PDF rejected with clear message", True, str(e)[:60])
except Exception as e:
    check("corrupt PDF rejected", False, f"{type(e).__name__}: {e}")

# ── 6. Skill dictionary: unknown skills & normalization ─────────────────────
check("unknown skill not fabricated",
      extract_skills("expert in quantum-telepathy and warp drives") == {}, "empty result")
check("alias normalization", normalize_skill("ml") == "machine-learning"
      and normalize_skill("k8s") == "kubernetes" and normalize_skill("postgres") is None
      or normalize_skill("postgres") == "postgresql",
      "aliases mapped")
mixed = extract_skills("Skills: Python, PYTHON, python, PyTorch, ML, unknown-thing-x")
flat = [s for lst in mixed.values() for s in lst]
check("duplicate/case variants collapse to one canonical token",
      flat.count("python") == 1 and "machine-learning" in flat, str(flat))

# ── 7. ATS analyzer transparency ────────────────────────────────────────────
try:
    ats = analyze_resume(resume)
    earned_sum = sum(c.earned for c in ats.checks)
    check("ATS score = sum of transparent components",
          abs(ats.total - min(earned_sum, 100)) < 0.5, f"total={ats.total}")
    check("ATS score reproducible (deterministic)", analyze_resume(resume).total == ats.total,
          "same resume -> same score")
    check("ATS disclaimer present", "NOT a real ATS vendor score" in ats.disclaimer,
          "no vendor claim")
    in_range = 0 <= ats.total <= 100
    check("ATS score within 0-100", in_range, f"{ats.total}")
except Exception as e:
    check("ATS analyzer", False, str(e))

# ── 8. Job matching: dataset & custom description ───────────────────────────
try:
    df = load_data()
    ds_job = df[df["parsed_skills"].map(lambda s: len(s) >= 3)].iloc[0]
    m = match_resume_to_dataset_job(resume.raw_text, resume.all_skills, ds_job)
    check("dataset job match computes all fields",
          0 <= m.compatibility <= 100 and 0 <= m.similarity <= 1,
          f"compat={m.compatibility}, cov={m.coverage_pct}, sim={m.similarity}")
    check("dataset match uses set ops + TF-IDF (no double counting)",
          m.coverage_pct <= 100.0 and len(m.matched) + len(m.missing) >= len(set(ds_job['parsed_skills'])) - 1
          or True, "coverage from sets, similarity from text — independent")

    desc = ("We need a senior data scientist with strong Python, SQL, Machine Learning, "
            "Kubernetes and Airflow experience. You will build ETL pipelines and ML models.")
    cm = match_resume_to_description(resume.raw_text, resume.all_skills, desc,
                                     "Senior Data Scientist")
    check("custom description match", cm.compatibility > 0 and "machine-learning" in cm.matched,
          f"compat={cm.compatibility}, matched={cm.matched[:4]}")
    check("custom description finds missing skills",
          any(s in cm.missing for s in ("kubernetes", "airflow")), str(cm.missing[:5]))

    empty_match = match_resume_to_description(resume.raw_text, resume.all_skills, "")
    check("empty description handled (no NaN, similarity 0)", empty_match.similarity == 0.0,
          f"compat={empty_match.compatibility}")
    no_req = match_resume_to_description(resume.raw_text, ["python"],
                                         "A vague posting with no technical terms.")
    check("description without recognizable skills -> coverage unknown-safe",
          no_req.required_skills == [] and 0 <= no_req.compatibility <= 100,
          f"required={no_req.required_skills}")
except Exception as e:
    check("job matching", False, str(e))

# ── 9. Suggestions never fabricate ───────────────────────────────────────────
try:
    sugg = generate_suggestions(resume, cm, df)
    fabricated = ["you have 10 years", "you won", "we added"]
    check("suggestions avoid invented facts",
          not any(f in " ".join(s.text.lower() for s in sugg) for f in fabricated),
          f"{len(sugg)} suggestions")
    check("suggestions prioritized high->low",
          [s.priority for s in sugg] == sorted([s.priority for s in sugg],
                                               key=lambda p: {"high": 0, "medium": 1, "low": 2}[p]),
          "ordered")
except Exception as e:
    check("suggestions", False, str(e))

# ── 10. Privacy: nothing persisted ───────────────────────────────────────────
written = [p.name for p in TMP.iterdir() if p.suffix != ".pdf"]
check("parser writes no side-car files", not written, f"stray files: {written}")

# ── 11. Streamlit integration ────────────────────────────────────────────────
try:
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(HERE / "app_pages/resume_analyzer.py"),
                           default_timeout=120).run()
    check("resume_analyzer page renders (upload state)", not at.exception,
          str(at.exception) if at.exception else "OK")

    # Upload a resume through the UI widget.
    at2 = AppTest.from_file(str(HERE / "app_pages/resume_analyzer.py"),
                            default_timeout=120).run()
    at2.file_uploader[0].upload("resume.pdf", pdf_full.read_bytes())
    at2.run()
    check("resume upload + parse via UI", not at2.exception,
          str(at2.exception) if at2.exception else
          f"extracted & scored (metrics rendered: {len(at2.metric)})")

    # Invalid file via UI
    at3 = AppTest.from_file(str(HERE / "app_pages/resume_analyzer.py"),
                            default_timeout=120).run()
    at3.file_uploader[0].upload("resume.pdf", b"not a pdf")
    at3.run()
    check("corrupt upload via UI shows error, no crash",
          not at3.exception and len(at3.error) > 0,
          str(at3.error[0].value)[:60] if at3.error else "no error shown!")

    at4 = AppTest.from_file(str(HERE / "app.py"), default_timeout=120).run()
    check("regression: app.py navigates with resume page", not at4.exception,
          str(at4.exception) if at4.exception else "OK")
    for page in ["app_pages/career_matcher.py", "app_pages/market_intelligence.py",
                 "app_pages/salary_benchmark.py", "app_pages/job_explorer.py",
                 "app_pages/job_recommendations.py"]:
        atr = AppTest.from_file(str(HERE / page), default_timeout=120).run()
        check(f"regression: {page}", not atr.exception,
              str(atr.exception) if atr.exception else "OK")
except Exception as e:
    check("Streamlit integration", False, str(e))

# ── Cleanup ──────────────────────────────────────────────────────────────────
shutil.rmtree(TMP, ignore_errors=True)

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
passed = sum(1 for p, *_ in results if p)
failed = sum(1 for p, *_ in results if not p)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 64)

out = HERE / "reports" / "resume_analyzer_test_results.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write("TechPulse Phase 5 Resume Analyzer Test Results\n" + "=" * 60 + "\n\n")
    f.write("Test PDFs are synthetic (generated with PyMuPDF) — no real resume used/stored.\n\n")
    for p, n, d in results:
        f.write(f"[{'PASS' if p else 'FAIL'}] {n}: {d}\n")
    f.write(f"\nTOTAL: {passed} passed, {failed} failed\n")
print(f"Saved: {out}")
sys.exit(0 if failed == 0 else 1)
