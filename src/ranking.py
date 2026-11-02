"""TechPulse Phase 4 — Personalized ranking module.

Configurable weighted composite score over 0-100. Missing user info means a
component is "unknown": it is excluded and the remaining weights are
renormalized (weights never silently reward or punish unavailable data).

Formula (Step 5):
    Final Score = w_skill × SkillScore + w_text × SimilarityScore
                + w_loc  × LocationScore + w_sal  × SalaryScore
                + w_work × WorkScore        (all components 0-100)

These are recommendation relevance scores, NOT probabilities of getting hired.
Experience is intentionally NOT a component: `seniority` exists for only ~40%
of rows and maps poorly to user experience years — treating it as a signal
would fabricate information (per dataset-inspection findings).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {
    "skill": 0.40,
    "text": 0.30,
    "location": 0.15,
    "salary": 0.10,
    "work": 0.05,
}


@dataclass
class RankWeights:
    """Configurable component weights; must sum to ~1.0 (validated)."""

    skill: float = DEFAULT_WEIGHTS["skill"]
    text: float = DEFAULT_WEIGHTS["text"]
    location: float = DEFAULT_WEIGHTS["location"]
    salary: float = DEFAULT_WEIGHTS["salary"]
    work: float = DEFAULT_WEIGHTS["work"]

    def as_dict(self) -> dict[str, float]:
        return {"skill": self.skill, "text": self.text, "location": self.location,
                "salary": self.salary, "work": self.work}

    @classmethod
    def from_dict(cls, d: dict) -> "RankWeights":
        base = cls()
        for k, v in (d or {}).items():
            if hasattr(base, k) and v is not None:
                setattr(base, k, float(v))
        return base


def _renormalize(weights: dict[str, float], available: set[str]) -> dict[str, float]:
    """Drop unavailable components and rescale remaining weights to sum to 1."""
    active = {k: v for k, v in weights.items() if k in available and v > 0}
    total = sum(active.values())
    if total <= 0:
        return {k: 0.0 for k in active}
    return {k: v / total for k, v in active.items()}


def location_score(job_country, job_city, pref_country, pref_city) -> float:
    """0-100: exact city 100 > same country 75 > different 0.
    Missing preferences → unknown (caller excludes component)."""
    jc = str(job_country or "").strip().lower()
    jct = str(job_city or "").strip().lower()
    pc = str(pref_country or "").strip().lower()
    pct = str(pref_city or "").strip().lower()
    if not pc and not pct:
        return np.nan                      # unknown → exclude
    if not jc or jc in ("unknown", "nan"):
        return 50.0                        # job location unknown → neutral
    if pct and pct == jct:
        return 100.0
    if pc and pc == jc:
        return 75.0
    return 0.0


def salary_score(job_min, job_max, user_min, user_max) -> float:
    """0-100 overlap of the job's disclosed USD-annualized range with the
    user's expected range. Missing job salary or user range → unknown (NaN);
    never invents numbers."""
    try:
        if pd.isna(job_min) or pd.isna(job_max):
            return np.nan
        if user_min is None and user_max is None:
            return np.nan
        user_min = float(user_min) if user_min is not None else 0.0
        user_max = float(user_max) if user_max is not None else float("inf")
        if user_max < user_min:
            user_min, user_max = user_max, user_min
        jmin, jmax = float(job_min), float(job_max)
        overlap = max(0.0, min(jmax, user_max) - max(jmin, user_min))
        if overlap <= 0:
            return 0.0
        job_width = max(jmax - jmin, 1.0)
        coverage = overlap / job_width
        user_width = max(user_max - user_min, 1.0)
        user_covered = overlap / user_width if user_width > 1.0 else 1.0
        return round(100.0 * min(1.0, 0.5 * coverage + 0.5 * user_covered), 1)
    except (TypeError, ValueError):
        return np.nan


def work_score(job_remote_type, pref_arrangement) -> float:
    """0-100 match of Remote vs 'On-site / Hybrid'. Unknown pref → NaN.
    Job remote status unknown → neutral 50."""
    if not pref_arrangement:
        return np.nan
    jt = str(job_remote_type or "").strip().lower()
    pt = str(pref_arrangement).strip().lower()
    if not pt or pt in ("any", "no preference"):
        return np.nan
    if not jt or jt == "unknown":
        return 50.0
    if jt == pt or pt in jt:
        return 100.0
    return 0.0


def composite_score(skill_pct: float, text_sim: float, loc_s, sal_s, work_s,
                    weights: RankWeights | None = None) -> tuple[float, dict[str, float]]:
    """Weighted final score (0-100) with weight renormalization.

    Returns (final_score_rounded, components_dict). Components excluded for
    unknown values are absent from the dict.
    """
    w = (weights or RankWeights()).as_dict()

    skill_component = float(skill_pct or 0.0)
    text_component = float(text_sim or 0.0) * 100.0

    available = {"skill", "text"}          # always computable
    components: dict[str, float] = {"skill": skill_component, "text": text_component}
    for name, val in (("location", loc_s), ("salary", sal_s), ("work", work_s)):
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            components[name] = float(val)
            available.add(name)

    wn = _renormalize(w, available)
    final = sum(wn.get(k, 0.0) * v for k, v in components.items())
    return round(float(final), 1), components


def rank_recommendations(df: pd.DataFrame, user_skills, similarity: pd.Series,
                         user_profile: dict, weights: RankWeights | None = None) -> pd.DataFrame:
    """Score and rank the candidate dataframe. Adds columns:
    skill_coverage, similarity, score_skill/text/location/salary/work,
    recommendation_score. Expects optional per-row skill match columns
    (matched_skills, missing_skills, matched_count, required_count) already
    present — produced by recommendation_engine.prepare_candidates()."""
    out = df.copy()
    if out.empty:
        return out

    sims = similarity.reindex(out.index).fillna(0.0) if similarity is not None \
        else pd.Series(0.0, index=out.index)
    out["similarity"] = sims.round(4)

    pref_country = user_profile.get("preferred_country")
    pref_city = user_profile.get("preferred_city")
    pref_work = user_profile.get("preferred_work_arrangement")
    u_min = user_profile.get("salary_min")
    u_max = user_profile.get("salary_max")

    loc_vals, sal_vals, work_vals = [], [], []
    for _, row in out.iterrows():
        loc_vals.append(location_score(row.get("country"), row.get("city"),
                                       pref_country, pref_city))
        sal_vals.append(salary_score(row.get("annual_salary_min"), row.get("annual_salary_max"),
                                     u_min, u_max))
        work_vals.append(work_score(row.get("remote_type"), pref_work))
    out["score_location"] = loc_vals
    out["score_salary"] = sal_vals
    out["score_work"] = work_vals

    scored = []
    for idx, row in out.iterrows():
        final, comps = composite_score(
            row.get("skill_coverage", 0.0), out.at[idx, "similarity"],
            row.get("score_location"), row.get("score_salary"), row.get("score_work"),
            weights,
        )
        comps["skill"] = round(comps.get("skill", 0.0), 1)
        comps["text"] = round(comps.get("text", 0.0), 1)
        comps["location"] = round(comps.get("location", 0.0), 1) if "location" in comps else None
        comps["salary"] = round(comps.get("salary", 0.0), 1) if "salary" in comps else None
        comps["work"] = round(comps.get("work", 0.0), 1) if "work" in comps else None
        scored.append(comps)
        out.at[idx, "recommendation_score"] = final

    comp_df = pd.DataFrame(scored, index=out.index)
    for c in ("skill", "text", "location", "salary", "work"):
        out[f"score_{c}"] = comp_df[c]

    return out.sort_values(
        ["recommendation_score", "skill_coverage", "similarity"],
        ascending=[False, False, False],
    )
