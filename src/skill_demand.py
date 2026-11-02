"""TechPulse Phase 6 — Skill demand statistics.

Computes per-role skill demand FROM THE ACTUAL CLEANED DATASET (no invented
statistics): frequency, percentage of postings requiring the skill, and rank.
Cached results via functools.lru_cache keyed on dataset hash.

Reuses `parsed_skills` lists already restored by utils/data_loader.load_data()
— no duplicate skill extraction system.
"""

from __future__ import annotations

import pandas as pd


def _skills_series(df: pd.DataFrame) -> pd.Series:
    """Exploded, non-null skills for the whole frame (or empty series)."""
    if "parsed_skills" not in df.columns:
        return pd.Series(dtype=object)
    return df["parsed_skills"].explode().dropna()


def global_skill_demand(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Whole-dataset demand table: skill, count, pct_of_postings, rank."""
    exploded = _skills_series(df)
    if exploded.empty:
        return pd.DataFrame(columns=["skill", "count", "pct_of_postings", "rank"])
    counts = exploded.value_counts()
    n_jobs = len(df)
    out = pd.DataFrame({
        "skill": counts.index.astype(str),
        "count": counts.values.astype(int),
        "pct_of_postings": (counts.values / n_jobs * 100).round(1),
    })
    out["rank"] = range(1, len(out) + 1)
    return out.head(top_n)


def role_skill_demand(df: pd.DataFrame, role: str, top_n: int = 25) -> pd.DataFrame:
    """Demand table restricted to one `job_role`. Empty role -> empty table
    (never fabricated). Columns: skill, count, pct_of_role_postings, rank.
    """
    if "job_role" not in df.columns:
        return pd.DataFrame(columns=["skill", "count", "pct_of_role_postings", "rank"])
    role_df = df[df["job_role"] == role]
    if role_df.empty:
        return pd.DataFrame(columns=["skill", "count", "pct_of_role_postings", "rank"])
    counts = role_df["parsed_skills"].explode().dropna().value_counts()
    out = pd.DataFrame({
        "skill": counts.index.astype(str),
        "count": counts.values.astype(int),
        "pct_of_role_postings": (counts.values / len(role_df) * 100).round(1),
    })
    out["rank"] = range(1, len(out) + 1)
    return out if top_n is None else out.head(top_n)


def role_posting_counts(df: pd.DataFrame) -> pd.Series:
    """Postings per role (used to contextualize demand)."""
    if "job_role" not in df.columns:
        return pd.Series(dtype=int)
    return df["job_role"].value_counts()


def demand_score_for(skills: list[str], role_demand: pd.DataFrame) -> dict[str, float]:
    """Demand score 0-100 per skill = its pct_of_role_postings, clipped at 100."""
    if role_demand is None or role_demand.empty:
        return {s: 0.0 for s in skills}
    pct = dict(zip(role_demand["skill"], role_demand["pct_of_role_postings"]))
    return {s: float(min(pct.get(s, 0.0), 100.0)) for s in skills}
