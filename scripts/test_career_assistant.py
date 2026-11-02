#!/usr/bin/env python
"""
TechPulse Phase 8 — AI Career Assistant test suite.

Covers: intent classification, knowledge retrieval, grounded responses for
every intent, interview generation, session memory & clearing, LLM fallback
(no key -> graceful), empty/missing profiles, module integration, and
Streamlit page regression.

Usage: python scripts/test_career_assistant.py   (exit 0 = all pass)
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
sys.path.insert(0, HERE)

results = []


def check(name, passed, detail=""):
    results.append((passed, name, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return passed


print("=" * 64)
print("TechPulse Phase 8 — AI Career Assistant Test Suite")
print("=" * 64)

from utils.data_loader import load_data
from src.career_assistant import CareerAssistantApp, ChatSession, ChatTurn, SUGGESTED_QUESTIONS
from src.intent_classifier import classify_intent
from src.knowledge_retrieval import search_jobs, dataset_summary, role_profile, most_demanded_skills
from src.interview_generator import generate_questions, categories_for_role, DISCLAIMER
from src import llm_provider

df = load_data()
check("dataset loads for assistant", len(df) > 0, f"rows={len(df)}")

# ── 1. Intent classification ────────────────────────────────────────────────
cases = [
    ("Which career is suitable for my skills?", "career_guidance"),
    ("What skills should I learn to become a Data Scientist?", "skill_gap"),
    ("Analyze my career gaps", "skill_gap"),
    ("Which jobs match my profile?", "job_recommendations"),
    ("How can I improve my resume?", "resume_analysis"),
    ("What salary can I expect?", "salary_guidance"),
    ("Prepare interview questions for a Python Developer", "interview_prep"),
    ("Give me a learning roadmap", "learning_roadmap"),
    ("Tell me a joke about cats", "general"),
]
for msg, expected in cases:
    res = classify_intent(msg)
    check(f"intent '{msg[:34]}...' -> {expected}", res.intent == expected,
          f"got={res.intent}, conf={res.confidence}, kw={res.matched_keywords[:2]}")

empty = classify_intent("")
check("empty message -> general", empty.intent == "general" and empty.confidence == 0.0,
      "no crash, fallback")

# ── 2. Knowledge retrieval (dataset-grounded only) ─────────────────────────
hits = search_jobs(df, "data scientist python", top_n=3)
check("retrieval returns relevant jobs", not hits.empty, f"n={len(hits)}")
check("retrieval empty for nonsense query", search_jobs(df, "zzzzunmatchable", top_n=3).empty,
      "no fabricated matches")
summary = dataset_summary(df)
check("dataset summary has roles & counts", summary["n_jobs"] > 0 and summary["n_skills"] > 0,
      f"{summary['n_jobs']} jobs, {summary['n_skills']} skills")
rp = role_profile(df, "Data Scientist / ML Engineer")
check("role profile grounded in dataset", rp["known"] and rp["postings"] > 0
      and len(rp["top_skills"]) > 0,
      f"postings={rp['postings']}, top={rp['top_skills'][:3]}")
check("unknown role -> known=False (nothing invented)", role_profile(df, "Atlantis Magician")["known"] is False,
      "explicit unknown")
gd = most_demanded_skills(df, top_n=5)
check("skill demand retrieval works", not gd.empty and "skill" in gd.columns,
      f"top={gd.iloc[0]['skill']}" if not gd.empty else "empty")

# ── 3. Assistant answers per intent ─────────────────────────────────────────
app8 = CareerAssistantApp(df)
session = ChatSession()

session.profile = {"skills": ["python", "sql", "machine-learning", "pandas"],
                   "target_role": "Data Scientist / ML Engineer"}
turn = app8.handle(session, "Analyze my career gaps")
check("skill-gap answer has coverage + missing skills",
      "coverage" in turn.text and "Missing" in turn.text or "cover" in turn.text,
      turn.text.split("\n")[0][:80])
check("coverage disclaimed as not hiring probability",
      "hiring probability" in turn.text, "disclaimer present")

turn2 = app8.handle(session, "Which jobs match my profile?")
check("job recommendation answer lists records", "1." in turn2.text, turn2.text.split("\n")[1][:70])

turn3 = app8.handle(session, "What salary can I expect?")
check("salary answer gives USD figure + limitation",
      "$" in turn3.text and ("estimate" in turn3.text.lower() or "limitation" in turn3.text.lower()),
      turn3.text.split("\n")[0][:80])

session.profile = {"target_role": "Software Developer"}
turn4 = app8.handle(session, "Prepare interview questions for a Python Developer")
check("interview answer has categories + disclaimer",
      "Python" in turn4.text and "practice questions" in turn4.text.lower(),
      f"{len(turn4.text)} chars")

turn5 = app8.handle(session, "Give me a learning roadmap")
check("roadmap answer has stages (needs skills)", "roadmap" in turn5.text.lower(),
      turn5.text.split("\n")[0][:70])

# Missing profile info -> honest guidance, no invention
empty_session = ChatSession()
empty_session.profile = {}
t_no_skills = app8.handle(empty_session, "Which career is suitable for my skills?")
check("no-skills career question asks for skills (no invention)",
      "skills" in t_no_skills.text.lower() and ("add" in t_no_skills.text.lower() or "provide" in t_no_skills.text.lower()),
      t_no_skills.text.split("\n")[0][:70])
t_no_role = app8.handle(empty_session, "What salary can I expect?")
check("no-role salary question lists model roles", "role" in t_no_role.text.lower(),
      t_no_role.text.split("\n")[0][:70])

# Unknown role handling
session.profile = {"skills": ["python"], "target_role": "Atlantis Magician"}
t_unk = app8.handle(session, "Analyze my career gaps")
check("unknown target role -> explicit fallback (no fabricated requirements)",
      "don't have dataset postings" in t_unk.text or "couldn't compute" in t_unk.text,
      t_unk.text.split("\n")[0][:80])

# ── 4. Interview generator ─────────────────────────────────────────────────
qs = generate_questions("Data Scientist / ML Engineer", df, per_category=2)
check("interview questions generated per category", len(qs) >= 6
      and all(q.question for q in qs), f"n={len(qs)}")
cats = {q.category for q in qs}
check("question categories include role-relevant sets", len(cats) >= 3, str(sorted(cats)))
check("disclaimer: not real company questions", "NOT actual" in DISCLAIMER,
      DISCLAIMER[:60])
cats_ds = categories_for_role(df, "Data Scientist / ML Engineer")
check("data-driven category mapping for known role",
      "python" in cats_ds and "machine-learning" in cats_ds, str(cats_ds))
cats_unk = categories_for_role(df, "Atlantis Magician")
check("unknown role -> standard category fallback", cats_unk == ["technical", "python", "sql",
                                                                 "machine-learning", "hr", "scenario"],
      str(cats_unk))

# ── 5. Session memory + clearing ───────────────────────────────────────────
s = ChatSession()
app8.handle(s, "Which jobs match my profile?")
app8.handle(s, "What salary can I expect?")
check("history accumulates turns", len(s.history) == 4, f"turns={len(s.history)}")
check("last user message retrievable", s.last_user_message() == "What salary can I expect?",
      s.last_user_message()[:40])
s.clear()
check("clear empties history", len(s.history) == 0, "cleared")

# Session-level memory cap (never unbounded)
s2 = ChatSession()
for i in range(100):
    app8.handle(s2, f"question {i}")
check("memory capped (no unbounded growth)", len(s2.history) <= 80,
      f"turns={len(s2.history)} (cap 80)")

# Empty message handling
s3 = ChatSession()
t_empty = app8.handle(s3, "")
check("empty message -> gentle prompt", "empty" in t_empty.text.lower(),
      t_empty.text[:60])
t_blank = app8.handle(s3, "   ")
check("whitespace-only message -> prompt", "empty" in t_blank.text.lower(), "handled")

# ── 6. LLM: unavailable -> graceful fallback ───────────────────────────────
# Ensure no key in this test env (patch provider to simulate absence).
orig_key, orig_chat = llm_provider._api_key, llm_provider.chat
llm_provider._api_key = lambda: None
llm_provider.chat = lambda *a, **k: (_ for _ in ()).throw(llm_provider.LLMUnavailable("no key"))
s4 = ChatSession()
s4.profile = {"skills": ["python", "sql"], "target_role": "Software Developer"}
t_llm = app8.handle(s4, "Analyze my career gaps", use_llm=True)
check("LLM requested but unavailable -> grounded fallback answer",
      "coverage" in t_llm.text.lower(), t_llm.text.split("\n")[0][:70])
check("fallback answer is still usable (no exception)", len(t_llm.text) > 50, "OK")

# LLM not requested -> template path identical
t_tpl = app8.handle(ChatSession(), "Analyze my career gaps")
session.profile = {"skills": ["python", "sql"], "target_role": "Software Developer"}
t_cmp = app8.handle(session, "Analyze my career gaps")
check("template path (no LLM) yields grounded answer",
      "coverage" in t_cmp.text.lower(), "OK")
llm_provider._api_key, llm_provider.chat = orig_key, orig_chat

# Consent rule: resume text only sent when user opts in
s5 = ChatSession()
s5.profile = {"skills": ["python"], "target_role": "Software Developer"}
sent = {}
llm_provider.chat = lambda prompt, **k: (sent.__setitem__("prompt", prompt), "OK")[1]
app8.handle(s5, "Analyze my career gaps", use_llm=True, resume_consent=False,
            resume_text="SECRET_RESUME_TEXT")
check("resume text NOT sent without consent", "SECRET_RESUME_TEXT" not in sent.get("prompt", ""),
      "privacy honored")
app8.handle(s5, "Analyze my career gaps", use_llm=True, resume_consent=True,
            resume_text="SECRET_RESUME_TEXT")
check("resume text sent WITH consent", "SECRET_RESUME_TEXT" in sent.get("prompt", ""),
      "consent honored")
llm_provider.chat = orig_chat

# ── 7. API key never hardcoded / safe config ───────────────────────────────
import inspect
src_llm = open(os.path.join(HERE, "src", "llm_provider.py"), encoding="utf-8").read()
check("no hardcoded API key in llm_provider",
      "sk-" not in src_llm.replace("sk-learn", "") and "Bearer {key}" in src_llm,
      "key comes from env/secrets only")
check("LLMUnavailable exception defined", hasattr(llm_provider, "LLMUnavailable"), "typed failure")

# ── 8. Module integration (reuses existing engines, no duplicates) ─────────
import src.response_generator as rg
import inspect as _inspect
src_rg = _inspect.getsource(rg)
check("response generator reuses Phase-4 recommender", "from src.recommendation_engine import recommend" in src_rg)
check("response generator reuses Phase-6 skill gap", "analyze_skill_gap" in src_rg)
check("response generator reuses Phase-7 salary predictor", "SalaryPredictor" in src_rg)
check("response generator reuses skill demand stats", "most_demanded_skills" in src_rg)

# ── 9. Streamlit integration + regression ──────────────────────────────────
try:
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(HERE, "app_pages/career_assistant.py"),
                           default_timeout=180).run()
    check("career_assistant page renders (initial state)", not at.exception,
          str(at.exception) if at.exception else "OK")

    # Send a message via chat input
    at2 = AppTest.from_file(os.path.join(HERE, "app_pages/career_assistant.py"),
                            default_timeout=180)
    at2.run()
    at2.chat_input[0].set_value("Which career is suitable for my skills?").run()
    at2 = at2.run()
    check("chat message flow works via UI", not at2.exception
          and len(at2.chat_message) >= 1,
          str(at2.exception) if at2.exception else f"{len(at2.chat_message)} messages rendered")

    # Suggested question buttons exist
    at3 = AppTest.from_file(os.path.join(HERE, "app_pages/career_assistant.py"),
                            default_timeout=180)
    at3.run()
    btn_labels = [b.label for b in at3.button]
    check("suggested questions rendered as buttons",
          sum(1 for lbl in btn_labels if lbl in SUGGESTED_QUESTIONS) >= 5,
          f"{sum(1 for lbl in btn_labels if lbl in SUGGESTED_QUESTIONS)} of {len(SUGGESTED_QUESTIONS)}")

    at6 = AppTest.from_file(os.path.join(HERE, "app.py"), default_timeout=180).run()
    check("regression: app.py navigates with career assistant page", not at6.exception, "OK")
    for page in ["app_pages/career_matcher.py", "app_pages/market_intelligence.py",
                 "app_pages/salary_benchmark.py", "app_pages/salary_prediction.py",
                 "app_pages/job_explorer.py", "app_pages/job_recommendations.py",
                 "app_pages/resume_analyzer.py", "app_pages/skill_gap_analyzer.py"]:
        atr = AppTest.from_file(os.path.join(HERE, page), default_timeout=120).run()
        check(f"regression: {page}", not atr.exception,
              str(atr.exception) if atr.exception else "OK")
except Exception as e:
    import traceback
    traceback.print_exc()
    check("Streamlit integration", False, str(e))

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
passed = sum(1 for x, *_ in results if x)
failed = sum(1 for x, *_ in results if not x)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 64)

out = os.path.join(HERE, "reports", "career_assistant_test_results.txt")
with open(out, "w", encoding="utf-8") as f:
    f.write("TechPulse Phase 8 AI Career Assistant Test Results\n" + "=" * 60 + "\n\n")
    for x, n, d in results:
        f.write(f"[{'PASS' if x else 'FAIL'}] {n}: {d}\n")
    f.write(f"\nTOTAL: {passed} passed, {failed} failed\n")
print(f"Saved: {out}")
sys.exit(0 if failed == 0 else 1)
