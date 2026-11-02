"""TechPulse Phase 4 — Recommendation evaluation.

Computes ranking metrics (Precision@K, Recall@K, NDCG@K) GIVEN binary or
graded relevance labels. The TechPulse dataset contains **no ground-truth
relevance labels and no user interaction data** — see the caveat printed by
`evaluate()` and the `no_ground_truth_available()` helper.

`evaluate()` accepts either:
  - `relevance`: dict {job_id or positional index: graded relevance (int ≥ 0)}, or
  - `relevant_ids`: iterable of relevant ids (binary relevance).
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if k <= 0 or not recommended:
        return 0.0
    top = recommended[:k]
    return sum(1 for r in top if r in relevant) / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    top = recommended[:k]
    return sum(1 for r in top if r in relevant) / len(relevant)


def ndcg_at_k(recommended: list, graded_relevance: Mapping, k: int) -> float:
    """NDCG@10 over graded relevance (dict id → gain ≥ 0)."""
    if k <= 0 or not graded_relevance:
        return 0.0
    dcg = 0.0
    for i, r in enumerate(recommended[:k]):
        gain = float(graded_relevance.get(r, 0))
        dcg += (2 ** gain - 1) / math.log2(i + 2)
    ideal = sorted(graded_relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** float(g) - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(recommended_ids: list, relevance: Mapping | None = None,
             relevant_ids: Iterable | None = None,
             ks: tuple = (5, 10)) -> dict:
    """Return metric dict. If neither relevance source is given, returns an
    explicit 'cannot validate' result instead of fake numbers."""
    if relevance is None and relevant_ids is None:
        return {"error": "no ground truth", "metrics": None}
    graded = dict(relevance) if relevance is not None else \
        {rid: 1 for rid in relevant_ids}
    relevant = {rid for rid, g in graded.items() if g > 0}

    out: dict = {}
    for k in ks:
        out[f"precision@{k}"] = round(precision_at_k(recommended_ids, relevant, k), 4)
        out[f"recall@{k}"] = round(recall_at_k(recommended_ids, relevant, k), 4)
    kmax = max(ks)
    out[f"ndcg@{kmax}"] = round(ndcg_at_k(recommended_ids, graded, kmax), 4)
    return out


def no_ground_truth_available() -> str:
    return (
        "The TechPulse dataset contains historical job postings only — there are no "
        "ground-truth relevance labels and no user interaction data (clicks, saves, "
        "applications). Recommendation QUALITY therefore cannot be empirically "
        "validated yet; Precision@5/10, Recall@5/10 and NDCG@10 are implemented in "
        "src/recommendation_evaluation.py and will compute real values the moment a "
        "relevance source exists. Synthetic test profiles used in the test suite are "
        "for functional testing ONLY and are NOT evidence of recommendation accuracy."
    )
