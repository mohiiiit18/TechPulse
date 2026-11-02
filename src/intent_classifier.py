"""
TechPulse Phase 8 — Intent classification for the AI career assistant.

Rule-based keyword scoring (no LLM dependency, fully offline, deterministic).
Returns the best-matching TechPulse intent or "general" with a confidence
score and the keyword evidence used, so the UI can show routing transparency.
"""
from __future__ import annotations

import re

from dataclasses import dataclass, field

# Intent → weighted keyword patterns. Order matters only via score magnitude.
INTENT_KEYWORDS: dict[str, list[str]] = {
    "resume_analysis": [
        "resume", "cv", "cover letter", "ats", "my resume", "resume analysis",
        "improve my resume", "resume review",
    ],
    "salary_guidance": [
        "salary", "compensation", "pay", "earn", "wage", "how much", "worth",
        "expect to make", "salary range", "usd", "$",
    ],
    "skill_gap": [
        "skill gap", "what skills", "which skills", "missing skills",
        "skills should i learn", "skills do i need", "learn to become",
        "become a", "become an", "gap analysis", "analyze my career gaps",
        "career gaps", "coverage",
    ],
    "job_recommendations": [
        "which jobs", "job match", "jobs match", "find jobs", "recommend jobs",
        "job for me", "openings", "suitable jobs", "match my profile",
        "companies hiring",
    ],
    "interview_prep": [
        "interview", "prepare for interview", "interview questions",
        "technical questions", "hr questions", "scenario question",
        "mock interview",
    ],
    "learning_roadmap": [
        "roadmap", "learning path", "learning plan", "study plan",
        "where to start", "how to become", "in what order",
    ],
    "career_guidance": [
        "career", "career path", "which career", "suitable for my skills",
        "best role for me", "career suitable", "what should i do",
        "fit for", "transition", "switch",
    ],
}

GENERAL_FALLBACK = "general"


@dataclass
class IntentResult:
    intent: str
    confidence: float           # 0-1 heuristic keyword score, normalized
    matched_keywords: list[str] = field(default_factory=list)
    scores: dict = field(default_factory=dict)


def classify_intent(message: str) -> IntentResult:
    """Score every intent by keyword hits; the max wins. Ties → earlier key.

    Matching uses word-boundary regex so 'earn' inside 'learn' or 'ats'
    inside 'cats' do not cause false positives.
    """
    text = (message or "").lower().strip()
    scores: dict[str, tuple[float, list[str]]] = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        hits, matched = 0.0, []
        for kw in keywords:
            pattern = r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])"
            if re.search(pattern, text):
                hits += 1 + (0.5 if " " in kw else 0.0)   # multi-word phrases weigh more
                matched.append(kw)
        scores[intent] = (hits, matched)

    best_intent, best = GENERAL_FALLBACK, (0.0, [])
    total = sum(h for h, _ in scores.values())
    for intent, (hits, matched) in scores.items():
        if hits > best[0]:
            best_intent, best = intent, (hits, matched)

    confidence = round(best[0] / total, 3) if total else 0.0
    return IntentResult(
        intent=best_intent if best[0] > 0 else GENERAL_FALLBACK,
        confidence=confidence if best[0] > 0 else 0.0,
        matched_keywords=best[1],
        scores={k: round(h, 2) for k, (h, _) in scores.items() if h > 0},
    )
