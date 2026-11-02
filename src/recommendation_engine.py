"""TechPulse Phase 4 — Intelligent Job Recommendation Engine.

Content-based recommender combining set-based skill matching, TF-IDF cosine
text similarity and configurable weighted ranking, with per-job explanations
and filters. Built on the Phase-2 cleaned dataset (data columns verified:
parsed_skills, title/title_clean, country, city, remote_type, annual_salary_*,
posted_date — there is NO free-text job-description column).

Public API:
    UserProfile          — validated user profile dataclass
    prepare_candidates() — filter + skill-match the corpus
    recommend()          — full pipeline: prepare → similarity → rank → explain
    explain_recommendation() — human-readable reason (dataset facts only)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import pandas as pd

from src.ranking import RankWeights, rank_recommendations
from src.skill_matching import match_skills, normalize_skills, strongest_matches
from src.text_similarity import compute_similarities


@dataclass
class UserProfile:
    """User profile for recommendations. Empty/incomplete profiles are valid:
    unknown components are excluded from scoring with weight renormalization."""

    skills: list[str] = field(default_factory=list)
    preferred_role: str | None = None
    experience_level: str | None = None        # informational (see ranking note)
    preferred_country: str | None = None
    preferred_city: str | None = None
    preferred_work_arrangement: str | None = None  # "Remote" | "On-site / Hybrid"
    salary_min: float | None = None            # USD annualized expectation
    salary_max: float | None = None

    def is_empty(self) -> bool:
        return not any([self.skills, self.preferred_role, self.preferred_country,
                        self.preferred_city, self.preferred_work_arrangement,
                        self.salary_min is not None, self.salary_max is not None])

    def to_dict(self) -> dict:
        return asdict(self)


# Filters (Step 7)
def apply_filters(df: pd.DataFrame, job_role: str | None = None,
                  country: str | None = None, remote_type: str | None = None,
                  salary_floor: float | None = None, salary_ceiling: float | None = None,
                  company: str | None = None,
                  min_skill_coverage: float | None = None) -> pd.DataFrame:
    """Corpus-level filters. `salary_floor/ceiling` bound the job's disclosed
    USD-annualized range (records without USD salary are kept — unknown ≠ excluded)."""
    out = df
    if job_role and "job_role" in out.columns:
        out = out[out["job_role"] == job_role]
    if country:
        out = out[out["country"].str.lower() == country.strip().lower()]
    if remote_type and "remote_type" in out.columns:
        out = out[out["remote_type"].str.lower() == remote_type.strip().lower()]
    if company and "company_name" in out.columns:
        out = out[out["company_name"].str.lower().str.contains(company.strip().lower(), na=False)]
    if salary_floor is not None and "annual_salary_max" in out.columns:
        # Keep jobs whose range can reach the floor, plus salary-unknown jobs.
        has_sal = out["annual_salary_max"].notna()
        out = out[~has_sal | (out["annual_salary_max"] >= salary_floor)]
    if salary_ceiling is not None and "annual_salary_min" in out.columns:
        has_sal = out["annual_salary_min"].notna()
        out = out[~has_sal | (out["annual_salary_min"] <= salary_ceiling)]
    if min_skill_coverage is not None and "skill_coverage" in out.columns:
        out = out[out["skill_coverage"] >= float(min_skill_coverage)]
    return out


def prepare_candidates(df: pd.DataFrame, profile: UserProfile) -> pd.DataFrame:
    """Annotate every job with its per-job skill match result."""
    out = df.copy()
    user_skills = normalize_skills(profile.skills)

    matched_lists, missing_lists, m_counts, r_counts, coverages = [], [], [], [], []
    for job_skills in out["parsed_skills"]:
        m = match_skills(list(user_skills), job_skills)
        matched_lists.append(m.matched)
        missing_lists.append(m.missing)
        m_counts.append(m.matched_count)
        r_counts.append(m.required_count)
        coverages.append(m.coverage_pct)

    out["matched_skills"] = matched_lists
    out["missing_skills"] = missing_lists
    out["matched_count"] = m_counts
    out["required_count"] = r_counts
    out["skill_coverage"] = coverages
    return out


def explain_recommendation(row: pd.Series, df: pd.DataFrame) -> str:
    """Human-readable explanation using ONLY dataset-derived facts."""
    name = str(row.get("title_clean") or row.get("title") or "this position").strip()
    matched = list(row.get("matched_skills") or [])
    missing = list(row.get("missing_skills") or [])
    coverage = float(row.get("skill_coverage") or 0.0)

    if not matched and not missing:
        return f"{name}: this record discloses no standardized skill requirements, so the skill component is unknown — ranked on text and preference signals only."

    if not matched:
        parts = [f"{name}: shares none of the {len(missing)} skills required by this record"]
        if missing:
            parts.append(f"(e.g. {', '.join(missing[:4])})")
        parts.append("consider it only if you plan to learn those skills.")
        return " ".join(parts)

    lead = strongest_matches(matched, df, role=row.get("job_role"), top_n=4)
    lead_names = ", ".join(s.title() for s in lead) if lead else ", ".join(matched[:4])
    text = (f"{name.title()}: this position matches {len(matched)} of the "
            f"{row.get('required_count', len(matched))} skills it requires "
            f"({coverage:.0f}% coverage). Your strongest matches are {lead_names}.")
    if missing:
        text += f" Consider learning {', '.join(missing[:4])} to improve your compatibility."
    else:
        text += " You already cover every skill this record discloses."
    return text


def recommend(df: pd.DataFrame, profile: UserProfile,
              weights: RankWeights | None = None,
              filters: dict | None = None,
              top_n: int = 10,
              dedupe: bool = True) -> pd.DataFrame:
    """Full recommendation pipeline.

    Empty profile → ranked on text/location/salary/work components that ARE
    available (skills scored 0 but weights renormalize over available ones —
    with an empty profile text similarity is 0 too, so results fall back to a
    deterministic dataset order and the caller should surface a warning).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    candidates = prepare_candidates(df, profile)
    if filters:
        candidates = apply_filters(candidates, **(filters or {}))
    if candidates.empty:
        return candidates

    sims = compute_similarities(candidates, profile.skills, profile.preferred_role)
    ranked = rank_recommendations(candidates, profile, sims, profile.to_dict(), weights)

    if dedupe:
        # Duplicate recommendations guard: identical postings can appear under
        # different job_ids; dedupe on (title, company, location).
        loc = ranked.get("location_raw", pd.Series("", index=ranked.index)).fillna("").astype(str).str.lower()
        key = (ranked["title_clean"].fillna("").str.lower() + "|"
               + ranked["company_name"].fillna("").str.lower() + "|" + loc)
        ranked = ranked.loc[~key.duplicated(keep="first")]

    return ranked.head(top_n).reset_index(drop=True)
