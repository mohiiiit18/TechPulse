"""TechPulse Phase 5 — Resume ↔ job matching.

Reuses Phase-4 machinery (skill_matching set operations + TF-IDF similarity)
so both components count once:
  - skill matching (matched/missing/coverage) via src.skill_matching.match_skills
  - text similarity via src.text_similarity vectorizer machinery

Job source A: existing dataset record (skills from parsed_skills; text doc from
title/skills — the dataset has NO free description column, per Phase 4 findings).
Job source B: pasted custom job description (skills extracted with the Phase-5
dictionary; full description used for TF-IDF).

Compatibility score = 60% skill coverage + 40% TF-IDF similarity (both 0-100,
no double counting: coverage comes from sets, similarity from text).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.skill_extractor import extract_skills
from src.skill_matching import match_skills
from src.text_similarity import build_content_document, clean_text


@dataclass
class JobMatchResult:
    job_title: str
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    coverage_pct: float = 0.0
    similarity: float = 0.0          # TF-IDF cosine 0-1
    compatibility: float = 0.0       # 0-100 blended score
    required_skills: list[str] = field(default_factory=list)


def _dataset_job_text(row: pd.Series) -> str:
    doc = build_content_document(row)
    return doc or str(row.get("title", ""))


def match_resume_to_dataset_job(resume_text: str, resume_skills: list[str],
                                job_row: pd.Series) -> JobMatchResult:
    """Match a parsed resume against one dataset job record."""
    job_skills = list(job_row.get("parsed_skills") or [])
    m = match_skills(resume_skills, job_skills)

    # TF-IDF on the job's content document vs the resume text (shared cleaning).
    job_doc = _dataset_job_text(job_row)
    resume_doc = clean_text(resume_text)
    sim = _tfidf_similarity(resume_doc, job_doc)

    return JobMatchResult(
        job_title=str(job_row.get("title", "Unknown title")),
        matched=m.matched,
        missing=m.missing,
        coverage_pct=m.coverage_pct,
        similarity=round(float(sim), 4),
        compatibility=round(0.6 * m.coverage_pct + 0.4 * sim * 100.0, 1),
        required_skills=sorted(set(job_skills)),
    )


def match_resume_to_description(resume_text: str, resume_skills: list[str],
                                description: str, job_title: str = "Custom job") -> JobMatchResult:
    """Match a parsed resume against a pasted job description."""
    required = sorted({s for skills in extract_skills(description).values() for s in skills})
    m = match_skills(resume_skills, required)

    sim = _tfidf_similarity(clean_text(resume_text), clean_text(description))

    return JobMatchResult(
        job_title=job_title or "Custom job",
        matched=m.matched,
        missing=m.missing,
        coverage_pct=m.coverage_pct,
        similarity=round(float(sim), 4),
        compatibility=round(0.6 * m.coverage_pct + 0.4 * sim * 100.0, 1),
        required_skills=required,
    )


def _tfidf_similarity(doc_a: str, doc_b: str) -> float:
    """Cosine similarity of two documents. Empty/blank docs → 0.0 (no NaN)."""
    if not doc_a.strip() or not doc_b.strip():
        return 0.0
    try:
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)
        tfidf = vec.fit_transform([doc_a, doc_b])
        return float(cosine_similarity(tfidf[0], tfidf[1])[0][0])
    except ValueError:
        return 0.0
