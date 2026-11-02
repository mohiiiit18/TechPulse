"""TechPulse Phase 5 — Resume improvement suggestions.

Every suggestion is grounded in the actual parsed resume and/or the matched
job's required skills. Nothing about work experience, achievements or
qualifications is invented — when evidence is missing, the suggestion says
what to ADD, not what the user supposedly has.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.resume_parser import ResumeData
from src.resume_matcher import JobMatchResult
from src.skill_matching import strongest_matches


@dataclass
class Suggestion:
    category: str      # "Skills" | "Sections" | "Impact" | "Job match"
    priority: str      # "high" | "medium" | "low"
    text: str


def generate_suggestions(resume: ResumeData,
                         match: JobMatchResult | None = None,
                         df=None) -> list[Suggestion]:
    """Rule-based, evidence-backed suggestions."""
    out: list[Suggestion] = []

    # ── Missing sections (from actual section detection) ──
    section_labels = {
        "experience": ("Work Experience", "high"),
        "education": ("Education", "high"),
        "skills": ("Technical Skills", "high"),
        "projects": ("Projects", "medium"),
        "certifications": ("Certifications", "low"),
    }
    missing_sections = [lbl for key, (lbl, _) in section_labels.items()
                        if key not in resume.sections]
    for label, (_, prio) in section_labels.items():
        if label not in resume.sections:
            out.append(Suggestion(
                "Sections", prio,
                f"No **{label}** section header was detected. ATS parsers rely on standard "
                f"headings — add a clearly labeled \"{label}\" section."))

    # ── Contact gaps ──
    if not resume.email:
        out.append(Suggestion("Sections", "high",
                              "No email address was detected. Put a professional email in the header."))
    if not resume.phone:
        out.append(Suggestion("Sections", "medium",
                              "No phone number was detected. Add one to the header."))

    # ── Skill gaps vs matched job ──
    if match is not None and match.missing:
        out.append(Suggestion(
            "Job match", "high",
            f"For **{match.job_title}**, you're missing {len(match.missing)} required skill(s): "
            f"{', '.join(match.missing[:8])}. Consider upskilling in the top ones or "
            f"highlighting equivalent experience you already have (don't list tools you haven't used)."))
    if match is not None and match.matched:
        top = strongest_matches(match.matched, df,
                                role=None if df is None else None, top_n=3)
        out.append(Suggestion(
            "Job match", "low",
            f"Lead with your strongest matched skills ({', '.join(top or match.matched[:3])}) "
            f"near the top of the resume — coverage for this role is {match.coverage_pct:.0f}%."))

    # ── Impact/quantification (evidence-based wording advice, not invented facts) ──
    import re
    number_lines = 0
    bullet_texts = resume.experience + resume.projects
    for item in bullet_texts:
        if re.search(r"\d+\s*(%|percent|x|users|customers|records|ms\b|hours|days)", item.lower()) \
                or re.search(r"\$\s?\d", item):
            number_lines += 1
    if bullet_texts and number_lines == 0:
        out.append(Suggestion(
            "Impact", "medium",
            "None of your experience/project bullets contain measurable results (%, $, counts, "
            "latency...). Where TRUE in your history, rewrite bullets like \"Improved X by N%\" — "
            "only include numbers you can substantiate."))
    elif bullet_texts and number_lines < max(1, len(bullet_texts) // 4):
        out.append(Suggestion(
            "Impact", "low",
            f"Only {number_lines} of {len(bullet_texts)} bullets appear quantified. Strengthen the "
            f"rest where the underlying facts support it."))

    # ── Weak project descriptions (very short bullets) ──
    short_bullets = [b for b in resume.projects if len(b) < 40]
    if short_bullets:
        out.append(Suggestion(
            "Impact", "medium",
            f"{len(short_bullets)} project bullet(s) are very short (<40 chars). Expand them with "
            f"your specific role, stack and outcome — but only with what you actually did."))

    if not out:
        out.append(Suggestion("General", "low",
                              "No structural gaps detected. Tailor keywords per application and keep "
                              "quantified, truthful bullets."))

    # Priority: high → medium → low
    order = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda s: order[s.priority])
    return out
