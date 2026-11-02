"""TechPulse Phase 6 — Skill priority ranking.

Ranks missing skills into High / Medium / Low priority using ONLY
dataset-observed demand (no invented statistics).

Methodology (documented in the UI as well):
  demand_score = percentage of postings for the target role that require
                 the skill (0-100, from src/skill_demand.py).
  High   : demand_score >= 50
  Medium : 20 <= demand_score < 50
  Low    : demand_score < 20  (skill still appears in the role's postings)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.skill_demand import demand_score_for


@dataclass
class SkillPriority:
    """One missing skill with its dataset-derived priority."""

    skill: str
    demand_score: float      # pct of role postings requiring it
    priority: str            # "high" | "medium" | "low"

    ORDER = {"high": 0, "medium": 1, "low": 2}


def rank_missing_skills(missing_skills: list[str], role_demand) -> list[SkillPriority]:
    """Rank missing skills by dataset demand, then alphabetically as a tiebreak.

    missing_skills may be None / empty -> returns []. Never fabricates a skill.
    """
    if not missing_skills:
        return []
    demand = demand_score_for(list(dict.fromkeys(missing_skills)), role_demand)
    ranked = []
    for s in sorted(set(missing_skills)):
        d = demand.get(s, 0.0)
        if d >= 50:
            prio = "high"
        elif d >= 20:
            prio = "medium"
        else:
            prio = "low"
        ranked.append(SkillPriority(skill=s, demand_score=d, priority=prio))
    ranked.sort(key=lambda sp: (-sp.demand_score, sp.skill))
    return ranked


def priority_counts(ranked: list[SkillPriority]) -> dict[str, int]:
    """Counts per priority bucket for the dashboard."""
    counts = {"high": 0, "medium": 0, "low": 0}
    for sp in ranked:
        counts[sp.priority] += 1
    return counts


PRIORITY_METHOD = (
    "Priority is derived only from the cleaned dataset: each missing skill's "
    "demand score is the share of postings for the target role that list it. "
    "High ≥ 50% of postings, Medium ≥ 20%, Low < 20%. Ties are broken "
    "alphabetically. No external market data or opinions are used."
)
