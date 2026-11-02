"""
TechPulse Phase 8 — Knowledge retrieval from existing TechPulse data.

The dataset (data/processed/techpulse_cleaned.csv) and the trained ML models
are the only sources of truth. Retrieval is TF-IDF keyword search over job
records (no LLM required, no fabricated facts). If a query has no reasonable
match the retriever says so explicitly instead of inventing an answer.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.skill_demand import global_skill_demand, role_posting_counts, role_skill_demand


def _job_document(row) -> str:
    """Content document per job: title, role, company, location, skills."""
    parts = [
        str(row.get("title_clean") or row.get("title") or ""),
        str(row.get("job_role") or ""),
        str(row.get("company_name") or ""),
        str(row.get("country") or ""),
        str(row.get("remote_type") or ""),
        " ".join(row.get("parsed_skills") or []),
    ]
    return " ".join(p for p in parts if p and p.lower() != "nan")


@lru_cache(maxsize=4)
def _fit_retriever(documents: tuple[str, ...]):
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    matrix = vec.fit_transform(list(documents))
    return vec, matrix


def search_jobs(df: pd.DataFrame, query: str, top_n: int = 5) -> pd.DataFrame:
    """TF-IDF retrieval over job records. Empty/irrelevant query → empty frame."""
    if df is None or df.empty or not (query or "").strip():
        return pd.DataFrame()
    docs = tuple(_job_document(r) for r in df.to_dict("records"))
    vec, matrix = _fit_retriever(docs)
    q = vec.transform([query])
    sims = cosine_similarity(q, matrix).ravel()
    out = df.copy()
    out["_relevance"] = sims
    out = out[out["_relevance"] > 0.05].sort_values("_relevance", ascending=False)
    return out.head(top_n)


def dataset_summary(df: pd.DataFrame) -> dict:
    """Role list + counts from the dataset (used to ground assistant answers)."""
    roles = role_posting_counts(df)
    return {
        "n_jobs": int(len(df)),
        "roles": {k: int(v) for k, v in roles.items()},
        "n_skills": int(df["parsed_skills"].explode().nunique()),
    }


def role_profile(df: pd.DataFrame, role: str) -> dict:
    """Everything the assistant knows about one role — dataset facts only."""
    rd = df[df["job_role"] == role]
    if rd.empty:
        return {"known": False}
    demand = role_skill_demand(df, role, top_n=10)
    usd = rd[rd["salary_norm_currency"] == "USD"]["salary_norm_avg"]
    return {
        "known": True,
        "postings": int(len(rd)),
        "top_skills": list(demand["skill"].head(10)),
        "skill_demand_pct": {r["skill"]: float(r["pct_of_role_postings"])
                             for _, r in demand.iterrows()},
        "salary_stats": (None if usd.dropna().empty else {
            "currency": "USD", "n": int(usd.notna().sum()),
            "p25": float(usd.quantile(0.25)), "median": float(usd.median()),
            "p75": float(usd.quantile(0.75)),
        }),
        "remote_share": float((rd["remote_type"] == "Remote").mean()) if len(rd) else 0.0,
    }


def most_demanded_skills(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Dataset-wide skill demand (delegates to Phase-6 module)."""
    return global_skill_demand(df, top_n=top_n)
