"""TechPulse Phase 6 — Skill gap analysis.

Compares user skills against required skills for a target role, reusing
Phase-4 set operations (src.skill_matching) — one matching implementation,
no duplicated logic.

Coverage formula (safe on empty inputs):
    Skill Coverage = (matched required skills / total required skills) × 100
A role with no postings / no skills yields an EMPTY result (unknown), never a
fabricated 0% requirement, and coverage is explicitly NOT a hiring probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.skill_demand import role_skill_demand
from src.skill_matching import match_skills, normalize_skills


@dataclass
class SkillGapResult:
    """Aggregated gap analysis for one user profile × one target role."""

    target_role: str
    analyzed_postings: int = 0      # role postings in the dataset (0 if role unknown)
    user_skills: list = field(default_factory=list)
    matched: list = field(default_factory=list)      # sorted intersection
    missing: list = field(default_factory=list)      # required-but-absent, sorted
    partial: list = field(default_factory=list)      # substring-related pairs (dataset-signal only)
    matched_count: int = 0
    required_count: int = 0
    coverage_pct: float = 0.0
    is_known_role: bool = False      # False when the role has no postings


def analyze_skill_gap(user_skills, target_role: str, df: pd.DataFrame) -> SkillGapResult:
    """Compare user skills to the aggregated skill requirements of a role.

    The role's requirement set = union of skills across ALL postings for that
    role in the cleaned dataset. If the role is unknown or has no postings the
    result is flagged `is_known_role=False` with empty matched/missing lists.
    """
    user_set = normalize_skills(user_skills)
    res = SkillGapResult(target_role=str(target_role), user_skills=sorted(user_set))

    if df is None or "job_role" not in df.columns or "parsed_skills" not in df.columns:
        return res

    role_df = df[df["job_role"] == target_role]
    res.analyzed_postings = len(role_df)
    if role_df.empty:
        return res  # unknown role — nothing invented

    required = set(role_df["parsed_skills"].explode().dropna().astype(str))
    required = {s for s in required if s.strip() and s != "nan"}
    res.is_known_role = True

    m = match_skills(user_set, required)
    res.matched = m.matched
    res.missing = m.missing
    res.matched_count = m.matched_count
    res.required_count = m.required_count
    res.coverage_pct = m.coverage_pct

    # "Partial" matches: dataset-signal only. A required skill is partially
    # covered when the user knows a skill whose canonical token shares a
    # hyphenated base (e.g. user 'machine-learning' vs required 'ml-engineering'
    # share the token 'machine'). This is informational, NOT counted in coverage.
    for req in m.missing:
        base = req.split("-")[0]
        if len(base) >= 5 and any(u.split("-")[0] == base for u in user_set):
            res.partial.append(req)
    res.partial = sorted(set(res.partial) - set(res.missing) | set()) or res.partial
    return res


def gap_table(result: SkillGapResult, df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Missing-skill table joined with per-role demand for the dashboard.

    Columns: skill, pct_of_role_postings (0 when skill absent from role pool),
    priority bucket is added downstream by skill_prioritization.
    """
    if not result.is_known_role or not result.missing:
        return pd.DataFrame(columns=["skill", "pct_of_role_postings"])
    demand = role_skill_demand(df, result.target_role, top_n=None)
    pct = dict(zip(demand["skill"], demand["pct_of_role_postings"]))
    out = pd.DataFrame({
        "skill": sorted(result.missing),
        "pct_of_role_postings": [round(float(pct.get(s, 0.0)), 1) for s in sorted(result.missing)],
    })
    return out.sort_values(["pct_of_role_postings", "skill"], ascending=[False, True]).head(top_n)


COVERAGE_NOTE = (
    "Skill coverage = matched required skills ÷ total required skills for the "
    "role's postings in this dataset. It is a curriculum-planning indicator "
    "only — NOT a hiring probability or an offer prediction."
)
