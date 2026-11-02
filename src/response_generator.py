"""
TechPulse Phase 8 — Response generation for the AI career assistant.

Deterministic, template-based answers grounded in the existing TechPulse
modules (recommendation engine, skill gap analyzer, salary predictor, ATS
analyzer, learning roadmap, skill demand). All numbers come from those
modules / the dataset — nothing is invented. When an LLM is available the
grounded facts are handed to it as context (facts-only prompt); when not,
the template answer is returned unchanged.
"""
from __future__ import annotations

import pandas as pd

from src import llm_provider
from src.interview_generator import generate_questions, DISCLAIMER as INTERVIEW_DISCLAIMER
from src.knowledge_retrieval import (dataset_summary, most_demanded_skills,
                                     role_profile, search_jobs)
from src.learning_roadmap import build_roadmap
from src.skill_prioritization import rank_missing_skills
from src.skill_demand import role_skill_demand
from src.recommendation_engine import recommend, UserProfile
from src.skill_gap_analyzer import analyze_skill_gap
from src.salary_predictor import SalaryPredictor, LIMITATIONS as SALARY_LIMITATIONS

SUGGESTED_QUESTIONS = [
    "Which career is suitable for my skills?",
    "What skills should I learn to become a Data Scientist?",
    "Analyze my career gaps.",
    "Which jobs match my profile?",
    "How can I improve my resume?",
    "What salary can I expect?",
    "Prepare interview questions for a Python Developer.",
]

ASSISTANT_DISCLAIMER = (
    "I answer from TechPulse's cleaned dataset (historical job records up to Aug 2026) "
    "and its trained models — I don't have live market data and I never invent "
    "statistics, postings or your qualifications."
)


class CareerAssistant:
    """Grounded career chatbot. `profile` holds user-provided context only."""

    def __init__(self, df: pd.DataFrame, profile: dict | None = None):
        self.df = df
        self.profile: dict = profile or {}
        self._salary = None

    # ── profile helpers ────────────────────────────────────────────────────
    @property
    def user_skills(self) -> list[str]:
        return list(self.profile.get("skills") or [])

    @property
    def user_role(self) -> str | None:
        return self.profile.get("target_role") or None

    @property
    def has_profile(self) -> bool:
        return bool(self.user_skills or self.user_role
                    or self.profile.get("preferred_location")
                    or self.profile.get("salary_expectation") is not None)

    def _salary_predictor(self) -> SalaryPredictor:
        if self._salary is None:
            self._salary = SalaryPredictor()
        return self._salary

    # ── main entry ─────────────────────────────────────────────────────────
    def respond(self, message: str, intent: str,
                use_llm: bool = False, resume_consent: bool = False,
                resume_text: str = "") -> str:
        """Route an intent to the grounded answer. Never raises for content
        issues; returns an honest fallback message instead."""
        handlers = {
            "career_guidance": self.answer_career_guidance,
            "job_recommendations": self.answer_job_recommendations,
            "skill_gap": self.answer_skill_gap,
            "resume_analysis": self.answer_resume,
            "salary_guidance": self.answer_salary,
            "interview_prep": self.answer_interview,
            "learning_roadmap": self.answer_roadmap,
        }
        handler = handlers.get(intent)
        base = handler(message) if handler else self.answer_general(message)
        if use_llm:
            base = self._enhance(base, message, use_resume=resume_consent,
                                 resume_text=resume_text)
        return base

    # ── grounded answers ───────────────────────────────────────────────────
    def answer_career_guidance(self, message: str) -> str:
        if not self.user_skills:
            return ("To suggest which career fits you, I need your skills. "
                    "Add them in the sidebar (or paste them here) and I'll run "
                    "the Career Matcher's model over the dataset.")
        summary = dataset_summary(self.df)
        from utils.data_loader import predict_roles, load_model
        model, encoder = load_model()
        predicted_role, results_df, unknown = predict_roles(self.user_skills, model, encoder)
        if predicted_role is None:
            return ("Your skills didn't match the model's training vocabulary well "
                    f"enough to predict a role. The dataset covers: {', '.join(summary['roles'])}. "
                    "Try adding widely-used skills like Python, SQL or Machine Learning.")
        lines = [f"Based on your skills, my top role match is **{predicted_role}** "
                 f"(from {summary['n_jobs']:,} dataset records)."]
        for other in results_df["job_role"].head(2):
            if other != predicted_role:
                lines.append(f"- Alternatives: {other}")
        if unknown:
            lines.append(f"*Ignored (not in model vocabulary): {', '.join(unknown[:5])}*")
        gap = analyze_skill_gap(self.user_skills, predicted_role, self.df)
        if gap.is_known_role and gap.missing:
            lines.append(f"Top skills to add for {predicted_role}: {', '.join(gap.missing[:6])}.")
        return "\n".join(lines)

    def answer_job_recommendations(self, message: str) -> str:
        profile = UserProfile(
            skills=self.user_skills,
            preferred_role=self.user_role,
            preferred_country=self.profile.get("preferred_location"),
        )
        recs = recommend(self.df, profile, top_n=5)
        if recs.empty:
            return ("I couldn't find records matching those constraints in the "
                    "dataset. Try relaxing filters or adding skills.")
        lines = ["Here are the dataset records that fit best (scored by the "
                 "recommendation engine):"]
        for _, r in recs.iterrows():
            title = str(r.get("title_clean") or r.get("title") or "Untitled role")
            comp = str(r.get("company_name") or "Unknown company")
            loc = str(r.get("location_raw") or r.get("country") or "Location not disclosed")
            cov = float(r.get("skill_coverage") or 0)
            lines.append(f"1. **{title}** — {comp} ({loc}), {cov:.0f}% skill coverage")
        lines.append("Open the Job recommendations page for the full ranked list "
                     "with per-job explanations.")
        return "\n".join(lines)

    def answer_skill_gap(self, message: str) -> str:
        role = self.user_role
        if role:
            for known in self.df["job_role"].dropna().unique():
                if str(role).lower() in str(known).lower() or str(known).lower() in str(role).lower():
                    role = known
                    break
        if not role:
            return ("Which target role should I analyze? Tell me (e.g. 'become a "
                    f"Data Scientist'). The dataset covers: "
                    f"{', '.join(sorted(dataset_summary(self.df)['roles']))}.")
        if not self.user_skills:
            prof = role_profile(self.df, role)
            if not prof["known"]:
                return (f"I don't have postings for '{role}' in the dataset, so I can't "
                        "analyze that gap without inventing requirements. Try one of the "
                        f"roles I know: {', '.join(sorted(dataset_summary(self.df)['roles']))}.")
            skills = list(prof["skill_demand_pct"])[:8]
            return (f"Top skills for **{role}** in the dataset (share of postings "
                    f"requiring them): " + ", ".join(f"{s} ({p:.0f}%)" for s, p in
                                                     prof["skill_demand_pct"].items()))
        gap = analyze_skill_gap(self.user_skills, role, self.df)
        if not gap.is_known_role:
            return (f"I don't have dataset postings for '{role}', so I won't invent "
                    "requirements. Choose one of the known roles and I'll compute "
                    "your real coverage.")
        lines = [f"Skill coverage for **{role}**: {gap.coverage_pct:.0f}% "
                 f"({gap.matched_count}/{gap.required_count} required skills matched, "
                 f"across {gap.analyzed_postings} postings)."]
        if gap.matched:
            lines.append(f"✓ You have: {', '.join(gap.matched[:6])}")
        if gap.missing:
            lines.append(f"→ Missing: {', '.join(gap.missing[:8])}")
            lines.append("Note: coverage is a skill-overlap metric, not a hiring probability.")
        else:
            lines.append("You cover every skill this role's postings require — "
                         "per the dataset.")
        return "\n".join(lines)

    def answer_resume(self, message: str) -> str:
        if not self.profile.get("resume_skills"):
            return ("I can review your resume two ways: 1) upload it on the "
                    "**Resume analyzer** page for the full ATS-style scoring "
                    "(structure, skills, keywords), or 2) list your resume's "
                    "skills here and I'll compare them to dataset demand.")
        skills = self.profile["resume_skills"]
        demand = most_demanded_skills(self.df, top_n=20)
        hot = [s for s in demand["skill"] if s in set(skills)]
        missing = [s for s in demand["skill"].head(8) if s not in set(skills)]
        lines = [f"Your resume lists {len(skills)} skills; {len(hot)} of them are "
                 "among the most-demanded skills in the dataset."]
        if hot:
            lines.append(f"✓ In-demand skills you have: {', '.join(hot[:8])}")
        if missing:
            lines.append(f"→ Consider adding (high dataset demand): {', '.join(missing[:6])}")
        lines.append("For scoring against ATS heuristics (sections, keywords, "
                     "formatting), use the Resume analyzer page.")
        return "\n".join(lines)

    def answer_salary(self, message: str) -> str:
        role = self.user_role
        try:
            p = self._salary_predictor()
        except Exception:
            return "The salary model isn't available right now — try the Salary benchmark page."
        if not role:
            known = ", ".join(p.known_roles)
            return ("Which role should I estimate for? The model covers: "
                    f"{known}. Estimates are USD-annual and based on historical "
                    "postings, not offers.")
        country = self.profile.get("preferred_location") or "United States"
        pred, lo, hi, info = p.predict_salary(role, country, self.user_skills or [],
                                              is_remote=self.profile.get("is_remote", False),
                                              seniority=self.profile.get("seniority", "unspecified"))
        lines = [f"Model estimate for **{role}** in {country}: "
                 f"**${pred:,.0f}/year** (empirical range ${lo:,.0f}–${hi:,.0f}, "
                 f"{info['range_note'].lower()})."]
        lines.append("Key limitations: " + SALARY_LIMITATIONS[0] + " " + SALARY_LIMITATIONS[2])
        return "\n".join(lines)

    def answer_interview(self, message: str) -> str:
        role = self.user_role
        for known in self.df["job_role"].dropna().unique():
            if str(known).lower() in (message or "").lower():
                role = known
                break
        if not role:
            roles = ", ".join(sorted(dataset_summary(self.df)["roles"]))
            return (f"Which role should I prepare questions for? The dataset covers: {roles}. "
                    "Note: these are curated practice questions, not real company questions.")
        qs = generate_questions(role, self.df, per_category=2)
        lines = [f"Practice interview questions for **{role}**:", ""]
        seen = set()
        for q in qs:
            if q.category in seen:
                continue
            if q.category not in seen:
                seen.add(q.category)
                lines.append(f"**{q.category}**")
            lines.append(f"- {q.question}")
            if q.answer:
                lines.append(f"  - *Answer guidance:* {q.answer}")
        lines.append("")
        lines.append(f"*{INTERVIEW_DISCLAIMER}*")
        return "\n".join(lines)

    def answer_roadmap(self, message: str) -> str:
        role = self.user_role
        if not role or not self.user_skills:
            return ("A learning roadmap needs (a) your current skills and (b) a "
                    "target role from the dataset. Provide both and I'll build a "
                    "prioritized, staged plan from the skill-gap analyzer.")
        gap = analyze_skill_gap(self.user_skills, role, self.df)
        if not gap.is_known_role or not gap.missing:
            return (f"I couldn't compute a gap for '{role}' with your skills — "
                    "either the role isn't in the dataset or you already cover "
                    "its requirements.")
        demand = role_skill_demand(self.df, role, top_n=None)
        pct = dict(zip(demand["skill"], demand["pct_of_role_postings"]))
        ranked = rank_missing_skills(gap.missing, pct)
        roadmap = build_roadmap(ranked, role)
        lines = [f"Learning roadmap for **{role}** (dataset-prioritized):"]
        for it in roadmap.items[:8]:
            lines.append(f"  Stage {it.stage}: {it.skill} ({it.priority} priority, "
                         f"{it.demand_score:.0f}% of postings)")
        lines.append("No durations or courses are promised — the dataset doesn't "
                     "contain that information.")
        return "\n".join(lines)

    def answer_general(self, message: str) -> str:
        # Retrieval: try to find relevant dataset records.
        hits = search_jobs(self.df, message, top_n=3)
        if hits.empty:
            return ("I couldn't find that in the dataset, and I won't invent an "
                    "answer. I can help with: " + "; ".join(SUGGESTED_QUESTIONS))
        lines = ["Closest matches in the dataset for your question:"]
        for _, r in hits.iterrows():
            lines.append(f"- **{r.get('title_clean', r.get('title', 'Role'))}** — "
                         f"{r.get('company_name', 'Unknown')} ({r.get('country', 'n/a')})")
        lines.append("Ask me about roles, skills, salaries or interviews for more.")
        return "\n".join(lines)

    # ── optional LLM enhancement ───────────────────────────────────────────
    def _enhance(self, grounded: str, message: str,
                 use_resume: bool = False, resume_text: str = "") -> str:
        """Rephrase the grounded answer via the LLM, feeding it ONLY verified
        facts. Falls back to the grounded text on any failure."""
        facts = grounded
        if use_resume and resume_text:
            facts = f"User resume summary (user consented to sharing):\n{resume_text}\n\n{grounded}"
        system = ("You are TechPulse's career assistant. Use ONLY the facts given. "
                  "Do not invent statistics, salaries, job postings or user "
                  "qualifications. If a fact is missing, say so.")
        try:
            reply = llm_provider.chat(f"{facts}\n\nUser question: {message}", system=system)
            if reply:
                return reply + "\n\n*(Answer grounded in TechPulse dataset facts.)*"
        except llm_provider.LLMUnavailable:
            pass
        return grounded  # graceful degradation — always usable offline
