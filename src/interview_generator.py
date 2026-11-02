"""
TechPulse Phase 8 — Interview question generation.

Curated, role-relevant question banks built from the assistant's own
knowledge (NOT scraped from or attributed to real companies). Categories:
technical, python, sql, machine-learning, hr, scenario. Each item may carry a
model answer / explanation. The dataset decides which categories apply to a
role (via its top required skills) — nothing invented about actual employers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class InterviewQuestion:
    category: str
    question: str
    answer: str = ""       # guidance/model answer; empty → discussion prompt
    explanation: str = ""


# Question banks. Keys align with dataset skill tokens where sensible so the
# role→category mapping is data-driven.
BANKS: dict[str, list[dict]] = {
    "technical": [
        {"q": "Walk me through how you would design and deploy a data pipeline that must run reliably every day.",
         "a": "Cover: requirements → source/schema checks → idempotent transform → orchestration (e.g. Airflow) → monitoring/alerting → backfill strategy.",
         "e": "Tests end-to-end systems thinking, not just coding."},
        {"q": "How do you approach debugging a model that performs well offline but poorly in production?",
         "a": "Check training/serving skew, data drift, feature availability at inference time, label leakage, and evaluation-set representativeness.",
         "e": "Very common real-world failure mode."},
        {"q": "Describe how you would evaluate whether a new feature improves your model fairly.",
         "a": "Same split/seed, group-aware CV, compare distributions, check per-segment metrics, not just aggregate scores.",
         "e": "Probes experimental rigor."},
        {"q": "A stakeholder disagrees with your analysis. How do you handle it?",
         "a": "Restate their concern, show assumptions explicitly, offer to test their hypothesis with data, document trade-offs.",
         "e": "Communication and collaboration under disagreement."},
    ],
    "python": [
        {"q": "What is the difference between a list and a tuple in Python, and when do you use each?",
         "a": "List: mutable, dynamic; tuple: immutable, hashable (usable as dict key), slightly faster. Use tuples for fixed records.",
         "e": "Fundamentals question."},
        {"q": "Explain list comprehensions and give one case where a generator is better.",
         "a": "Comprehension materializes the whole list; a generator yields lazily — better for large/streaming data to bound memory.",
         "e": "Checks memory-awareness."},
        {"q": "What are common pitfalls of mutable default arguments?",
         "a": "def f(x=[]) shares one list across calls; use None sentinel and create a new list inside.",
         "e": "Classic gotcha."},
        {"q": "How do you handle exceptions in a data pipeline so one bad row doesn't kill the batch?",
         "a": "Catch narrow exceptions per record, log to a dead-letter store, aggregate failures, decide pass/fail thresholds up front.",
         "e": "Production robustness."},
    ],
    "sql": [
        {"q": "Write a query to return the second-highest salary per department.",
         "a": "SELECT department, MAX(salary) FROM (SELECT department, salary, DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) r FROM emp) t WHERE r = 2 GROUP BY department;",
         "e": "Tests window functions."},
        {"q": "When would you use a LEFT JOIN instead of an INNER JOIN?",
         "a": "When you must keep all rows from the left table even without a match (e.g. all users, with or without orders).",
         "e": "Join semantics."},
        {"q": "How do you find duplicate rows and keep only one copy?",
         "a": "GROUP BY the natural key HAVING COUNT(*) > 1 to inspect, then DELETE with ROW_NUMBER() keeping rn = 1.",
         "e": "Data-cleaning fluency."},
        {"q": "Explain the difference between WHERE and HAVING.",
         "a": "WHERE filters rows before grouping; HAVING filters aggregated groups after GROUP BY.",
         "e": "Fundamentals."},
    ],
    "machine-learning": [
        {"q": "Your classifier has 99% accuracy on an imbalanced dataset. What do you check first?",
         "a": "Class distribution (majority baseline), then precision/recall per class, PR-AUC, confusion matrix.",
         "e": "Imbalance literacy."},
        {"q": "How do you prevent data leakage in a group-aware train/test split?",
         "a": "Group rows that share an entity (user/company/skill-set) and split by groups, never by rows alone.",
         "e": "Mirrors TechPulse's own methodology."},
        {"q": "When would you prefer a simpler model (linear/logistic) over a boosted ensemble?",
         "a": "Small data, need interpretability, strict latency/infra limits, or when the signal is mostly linear — plus as a sanity baseline.",
         "e": "Model-selection judgement."},
        {"q": "Explain precision vs recall with a hiring-screening example.",
         "a": "Precision: of candidates you advance, how many were actually good (avoid wasted interviews). Recall: of all good candidates, how many you caught (avoid missing talent). Trade-off depends on cost of each error.",
         "e": "Communication of core trade-off."},
    ],
    "hr": [
        {"q": "Tell me about yourself.",
         "a": "60–90 seconds: present role → 2 relevant accomplishments → why this role/company. Keep it structured and honest.",
         "e": "Warm-up — practice a tight narrative."},
        {"q": "Why do you want to work here?",
         "a": "Tie your skills to the company's product/domain and show specific research; avoid generic praise.",
         "e": "Motivation fit."},
        {"q": "Describe a time you failed. What did you learn?",
         "a": "Use STAR: Situation, Task, Action, Result — pick a real failure with a concrete corrective action.",
         "e": "Behavioral, requires a true story — don't invent one."},
        {"q": "What are your salary expectations?",
         "a": "Give a researched range (e.g. from TechPulse's salary tools), state flexibility, ask about the band.",
         "e": "Use dataset evidence, not a made-up number."},
    ],
    "scenario": [
        {"q": "You have one day to deliver an analysis and you find a data-quality problem that changes the conclusion. What do you do?",
         "a": "Communicate early with the caveat, quantify the impact, present both views if possible, agree a plan to fix properly.",
         "e": "Integrity under deadline pressure."},
        {"q": "Two teams want incompatible metrics definitions for the same KPI. How do you resolve it?",
         "a": "Facilitate a single agreed definition document, show consequences of each variant, escalate only if unresolved.",
         "e": "Stakeholder alignment."},
        {"q": "Your manager asks you to present a result you believe is statistically weak. What do you do?",
         "a": "State the uncertainty honestly, show caveats and sample sizes, suggest what evidence would strengthen it. Never overclaim.",
         "e": "Ethics and communication."},
    ],
}

CATEGORY_LABELS = {
    "technical": "Technical / role-specific",
    "python": "Python",
    "sql": "SQL",
    "machine-learning": "Machine Learning",
    "hr": "HR / behavioral",
    "scenario": "Scenario-based",
}


def categories_for_role(df: pd.DataFrame, role: str, top_skills: int = 5) -> list[str]:
    """Data-driven: map a role's top required skills in the dataset to question
    categories. Falls back to the standard set for unknown roles."""
    standard = ["technical", "python", "sql", "machine-learning", "hr", "scenario"]
    if df is None or "job_role" not in df.columns:
        return standard
    rd = df[df["job_role"] == role]
    if rd.empty:
        return standard
    top = set(rd["parsed_skills"].explode().dropna().value_counts().head(top_skills).index)
    cats = ["technical"]
    if "python" in top or "pandas" in top or "numpy" in top:
        cats.append("python")
    if "sql" in top or "postgresql" in top or "mysql" in top:
        cats.append("sql")
    if any(s in top for s in ("machine-learning", "scikit-learn", "pytorch", "tensorflow", "llm")):
        cats.append("machine-learning")
    cats += ["hr", "scenario"]
    return [c for c in dict.fromkeys(cats) if c in BANKS]


def generate_questions(role: str, df: pd.DataFrame | None = None,
                       per_category: int = 3) -> list[InterviewQuestion]:
    """Generate curated questions for a role, category set decided by dataset
    evidence. Deterministic; no claim these are real company questions."""
    cats = categories_for_role(df, role) if df is not None else list(BANKS)
    out: list[InterviewQuestion] = []
    for cat in cats:
        for item in BANKS[cat][:per_category]:
            out.append(InterviewQuestion(category=CATEGORY_LABELS.get(cat, cat),
                                         question=item["q"],
                                         answer=item.get("a", ""),
                                         explanation=item.get("e", "")))
    return out


DISCLAIMER = (
    "These are curated practice questions relevant to the role's skills — NOT actual "
    "questions from any specific company (those are proprietary and not in the dataset)."
)
