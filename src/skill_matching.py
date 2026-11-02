"""TechPulse Phase 4 — Skill matching module.

Set-operation based skill matching between a user profile and each job's
required skills. Pure functions, no side effects, safe on empty lists.

Skills in the dataset live in `parsed_skills` (list[str], lowercase,
standardized) — restored by utils/data_loader.load_data(). User skills are
normalized identically (strip + lowercase) so set operations line up.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def normalize_skills(skills) -> set[str]:
    """Normalize an iterable of skills to a lowercase stripped set.

    Accepts None / NaN-ish values safely.
    """
    if skills is None:
        return set()
    out = set()
    for s in skills:
        if s is None:
            continue
        tok = str(s).strip().lower()
        if tok and tok != "nan":
            out.add(tok)
    return out


@dataclass
class SkillMatchResult:
    """Result of matching a user's skills against one job's required skills."""

    matched: list[str] = field(default_factory=list)   # intersection, sorted
    missing: list[str] = field(default_factory=list)   # required-but-absent, sorted
    matched_count: int = 0
    required_count: int = 0
    coverage_pct: float = 0.0                          # 0-100


def match_skills(user_skills, job_skills) -> SkillMatchResult:
    """Compute the skill match between a user and one job.

    Formula (Step 3):
        Skill Coverage = (matching required skills / total required skills) × 100

    Empty user skills or empty job skills → coverage 0.0, never ZeroDivision.
    """
    user_set = normalize_skills(user_skills)
    job_set = normalize_skills(job_skills)

    if not job_set:
        # No known requirements: nothing to match against (unknown, not mismatch)
        return SkillMatchResult(matched=[], missing=[], matched_count=0,
                                required_count=0, coverage_pct=0.0)

    matched = sorted(user_set & job_set)
    missing = sorted(job_set - user_set)
    matched_count = len(matched)
    coverage = (matched_count / len(job_set)) * 100.0  # safe: job_set non-empty

    return SkillMatchResult(
        matched=matched,
        missing=missing,
        matched_count=matched_count,
        required_count=len(job_set),
        coverage_pct=round(coverage, 1),
    )


def strongest_matches(matched: list[str], df, role: str | None = None, top_n: int = 4) -> list[str]:
    """Order a user's matched skills by dataset-wide demand (frequency).

    Used for explanation text ("your strongest matches are ..."). Falls back
    to alphabetical order if the dataframe/skills column is unavailable.
    """
    if not matched:
        return []
    if df is None or "parsed_skills" not in df.columns:
        return sorted(matched)[:top_n]

    pool = df[df["job_role"] == role] if role is not None and "job_role" in df.columns else df
    freq = pool["parsed_skills"].explode().dropna().value_counts() if not pool.empty else None

    if freq is None or freq.empty:
        return sorted(matched)[:top_n]
    return sorted(matched, key=lambda s: (-int(freq.get(s, 0)), s))[:top_n]
