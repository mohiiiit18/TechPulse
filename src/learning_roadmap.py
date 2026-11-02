"""TechPulse Phase 6 — Personalized learning roadmap.

Generates a staged, editable learning sequence from prioritized missing
skills. Constraints honored:
  - No course links, providers, or prices (none exist in the dataset).
  - No guaranteed learning durations — stages list skills only, and the
    user-chosen weekly time budget is shown as planning context, not a promise.
  - Skills and stages come from dataset-derived priorities only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.skill_prioritization import SkillPriority

STATUSES = ("Not Started", "In Progress", "Completed")


@dataclass
class RoadmapItem:
    """One skill in the roadmap with an editable status."""

    skill: str
    priority: str                 # high | medium | low
    demand_score: float
    status: str = "Not Started"   # one of STATUSES
    stage: int = 1


@dataclass
class LearningRoadmap:
    """Staged roadmap. Stage 1 = high priority, 2 = medium, 3 = low."""

    target_role: str
    items: list[RoadmapItem] = field(default_factory=list)
    weekly_hours: float | None = None   # user context; no duration derived from it

    @property
    def n_stages(self) -> int:
        return max([it.stage for it in self.items], default=0)

    def stage_items(self, stage: int) -> list[RoadmapItem]:
        return [it for it in self.items if it.stage == stage]

    def progress_summary(self) -> dict:
        """Counts + completion % for progress tracking. No time promises."""
        total = len(self.items)
        completed = sum(1 for it in self.items if it.status == "Completed")
        in_progress = sum(1 for it in self.items if it.status == "In Progress")
        return {
            "total": total,
            "not_started": total - completed - in_progress,
            "in_progress": in_progress,
            "completed": completed,
            "completion_pct": round(completed / total * 100, 1) if total else 0.0,
        }


def build_roadmap(ranked_priorities, target_role: str,
                  weekly_hours: float | None = None) -> LearningRoadmap:
    """Map ranked priorities into stages. Empty input -> empty roadmap.

    Stage assignment: high→1, medium→2, low→3 (dataset demand order).
    """
    stage_of = {"high": 1, "medium": 2, "low": 3}
    items = [
        RoadmapItem(skill=sp.skill, priority=sp.priority,
                    demand_score=sp.demand_score, stage=stage_of.get(sp.priority, 3))
        for sp in ranked_priorities
    ]
    # Within a stage keep the ranking order (demand desc, then alphabetical).
    items.sort(key=lambda it: (it.stage, -it.demand_score, it.skill))
    return LearningRoadmap(target_role=str(target_role), items=items,
                           weekly_hours=weekly_hours)


ROADMAP_NOTE = (
    "Stages group skills by dataset-derived priority (High → Stage 1, "
    "Medium → Stage 2, Low → Stage 3). TechPulse provides skills and demand "
    "evidence only — no course links, providers, prices, or guaranteed "
    "completion durations. Your weekly time budget is shown for planning "
    "context and does not change the roadmap."
)
