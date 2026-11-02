"""TechPulse Phase 5 — Transparent heuristic ATS-style analysis.

This is NOT a real ATS vendor score (Greenhouse/Lever/Taleo et al. are
proprietary) and does NOT guarantee interview selection. Every sub-score is
deterministic, derived only from the uploaded resume's content, and the full
breakdown is always shown to the user.

Scoring (100 total):
    Contact info          20   email + phone (+ bonus: name found)
    Skills section        25   labeled section present + breadth of skills
    Experience section    25   labeled section + dated entries
    Education section     15   labeled section + parseable degree lines
    Projects              10   section present + items
    Certifications         5   any recognized certification keywords
Section organization/readability is folded into each section's sub-score
(header found = structure points) rather than double-counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.resume_parser import ResumeData

WEIGHTS = {
    "contact": 20,
    "skills": 25,
    "experience": 25,
    "education": 15,
    "projects": 10,
    "certifications": 5,
}


@dataclass
class ATSCheck:
    name: str
    earned: float
    maximum: float
    detail: str


@dataclass
class ATSResult:
    total: float
    checks: list[ATSCheck] = field(default_factory=list)
    disclaimer: str = (
        "This is TechPulse's transparent heuristic resume score computed from your "
        "document's structure and content. It is NOT a real ATS vendor score and does "
        "NOT guarantee selection — each company's ATS (Greenhouse, Lever, Taleo, ...) "
        "scores differently and is proprietary."
    )


def analyze_resume(resume: ResumeData) -> ATSResult:
    """Deterministic heuristic evaluation. No randomness, no ML, no fabrication:
    every point maps to something literally found in the document."""
    checks: list[ATSCheck] = []

    # ── Contact (20) ──
    contact = 0.0
    parts = []
    if resume.email:
        contact += 10
        parts.append("email found")
    else:
        parts.append("no email detected — add one")
    if resume.phone:
        contact += 7
        parts.append("phone found")
    else:
        parts.append("no phone detected — add one")
    if resume.full_name:
        contact += 3
        parts.append("name line found")
    else:
        parts.append("clear full-name line at top not detected")
    checks.append(ATSCheck("Contact information", contact, WEIGHTS["contact"], "; ".join(parts)))

    # ── Skills (25) ──
    n_skills = len(resume.all_skills)
    n_categories = len(resume.skills_by_category)
    skills_score = 0.0
    skill_parts = []
    if "skills" in resume.sections:
        skills_score += 10
        skill_parts.append("labeled skills section found")
    else:
        skill_parts.append("no labeled skills section — add one ('Technical Skills')")
    # Breadth: up to 15 pts, 3 pts per category represented (max 5 categories).
    breadth = min(15.0, 3.0 * n_categories)
    skills_score += breadth
    skill_parts.append(f"{n_skills} known skills across {n_categories} categories (+{breadth:.0f})")
    checks.append(ATSCheck("Skills", skills_score, WEIGHTS["skills"], "; ".join(skill_parts)))

    # ── Experience (25) ──
    exp_score = 0.0
    exp_parts = []
    if "experience" in resume.sections:
        exp_score += 10
        exp_parts.append("labeled experience section found")
    else:
        exp_parts.append("no labeled experience section — add one ('Work Experience')")
    n_dated = len(resume.experience)
    exp_score += min(15.0, 5.0 * n_dated)
    exp_parts.append(f"{n_dated} dated/bulleted entries parsed (+{min(15.0, 5.0 * n_dated):.0f})")
    checks.append(ATSCheck("Experience", exp_score, WEIGHTS["experience"], "; ".join(exp_parts)))

    # ── Education (15) ──
    edu_score = 0.0
    edu_parts = []
    if "education" in resume.sections:
        edu_score += 7
        edu_parts.append("labeled education section found")
    else:
        edu_parts.append("no labeled education section")
    n_edu = len(resume.education)
    edu_score += min(8.0, 4.0 * n_edu)
    edu_parts.append(f"{n_edu} degree/diploma lines detected (+{min(8.0, 4.0 * n_edu):.0f})")
    checks.append(ATSCheck("Education", edu_score, WEIGHTS["education"], "; ".join(edu_parts)))

    # ── Projects (10) ──
    proj_score = min(10.0, 5.0 * bool(resume.sections.get("projects")) + 2.5 * len(resume.projects))
    proj_detail = (f"section={'yes' if 'projects' in resume.sections else 'no'}, "
                   f"{len(resume.projects)} bulleted/dated items")
    checks.append(ATSCheck("Projects", proj_score, WEIGHTS["projects"], proj_detail))

    # ── Certifications (5) ──
    cert_score = min(5.0, 2.5 * len(resume.certifications))
    checks.append(ATSCheck("Certifications", cert_score, WEIGHTS["certifications"],
                           f"{len(resume.certifications)} recognized certification keyword(s)"))

    total = sum(c.earned for c in checks)
    return ATSResult(total=round(min(total, 100.0), 1), checks=checks)
