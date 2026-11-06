import streamlit as st
import pandas as pd
from utils.theme import apply_base_css, page_header, section_title
from utils.data_loader import load_data
from utils import auth
from src.resume_parser import parse_resume, validate_upload, ResumeParseError
from src.resume_matcher import match_resume_to_dataset_job, match_resume_to_description
from src.resume_suggestions import generate_suggestions

MAX_MB = 5

apply_base_css()
page_header(":material/document_scanner:", "AI Resume Analyzer",
            f"Upload a text-based PDF resume (max {MAX_MB} MB) to extract your information, get a "
            "transparent heuristic resume score, and match against dataset jobs or a pasted job "
            "description. Your resume is processed in memory only — it is **never stored** on the "
            "server. Scores are heuristic feedback, NOT a real ATS vendor score and NOT a guarantee "
            "of interview selection.")

df = load_data()

# ── 1. Upload ────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"],
    accept_multiple_files=False,
    help=f"PDF only, up to {MAX_MB} MB. Scanned/image-only PDFs cannot be parsed — upload a text-based PDF.",
)

resume = None
if uploaded is not None:
    err = validate_upload(uploaded.name, uploaded.size)
    if err:
        st.error(err, icon=":material/error:")
    else:
        with st.spinner("Extracting resume content (in memory)..."):
            try:
                resume = parse_resume(uploaded.getvalue())
            except ResumeParseError as e:
                st.error(str(e), icon=":material/picture_as_pdf:")
            except Exception as e:
                st.error(f"Unexpected parsing failure ({type(e).__name__}). "
                         "Please verify the PDF opens correctly and try again.", icon=":material/error:")

if resume is None:
    st.info("Upload a PDF resume to begin. Nothing is uploaded to any third party; "
            "parsing happens locally in this app's memory.", icon=":material/info:")
    st.stop()

# ── 2. Extracted information ─────────────────────────────────────────────────
st.space("small")
# Phase 6 integration: expose extracted skills for the Skill Gap Analyzer handoff.
st.session_state["resume_skills"] = list(resume.all_skills)

# Action row: handoff to Skill Gap Analyzer & explicit Save to History
col_hand, col_save_hist = st.columns(2)
with col_hand:
    if st.button("Send skills to Skill Gap Analyzer", icon=":material/swap_vert:", use_container_width=True):
        st.session_state["sga_skills"] = [
            s for s in resume.all_skills
        ]
        st.switch_page("app_pages/skill_gap_analyzer.py")

with col_save_hist:
    if auth.is_authenticated():
        if st.button("Save Resume Audit to History", icon=":material/save:", type="primary", use_container_width=True):
            ok, msg = auth.add_history(
                st.session_state["user"]["id"],
                "resume_analysis",
                {
                    "full_name": resume.full_name or "Candidate",
                    "skills_found": len(resume.all_skills),
                    "skills": list(resume.all_skills)[:25],
                    "pages": resume.n_pages,
                    "experience_entries": len(resume.experience),
                    "projects_entries": len(resume.projects)
                }
            )
            if ok:
                st.success("Resume audit saved to your profile history!")
            else:
                st.error(f"Unable to save: {msg}")
    else:
        st.caption("Sign in to save this resume analysis to your account history.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pages", resume.n_pages)
c2.metric("Skills found", len(resume.all_skills))
c3.metric("Experience entries", len(resume.experience))
c4.metric("Projects", len(resume.projects))

contact_cols = st.columns(3)
contact_cols[0].markdown(f"**Name:** {resume.full_name or '_not detected_'}")
contact_cols[1].markdown(f"**Email:** {resume.email or '_not detected_'}")
contact_cols[2].markdown(f"**Phone:** {resume.phone or '_not detected_'}")

with st.expander("Parsed sections", icon=":material/segment:"):
    for name in ["education", "experience", "projects", "certifications"]:
        items = getattr(resume, name)
        st.markdown(f"**{name.title()}** — {len(items)} item(s) detected")
        if items:
            for it in items[:8]:
                st.caption(f"• {it[:180]}")
        else:
            st.caption("_not detected in this resume (nothing invented)_")
        st.divider()

# ── 3. ATS-style score ──────────────────────────────────────────────────────
from src.ats_analyzer import analyze_resume

ats = analyze_resume(resume)
st.space("small")
st.markdown("**Resume score breakdown**")
score_cols = st.columns([1, 2])
with score_cols[0]:
    st.metric("Heuristic resume score", f"{ats.total:.0f}/100",
              delta="Good" if ats.total >= 70 else ("Fair" if ats.total >= 45 else "Needs work"),
              border=True)
with score_cols[1]:
    for c in ats.checks:
        st.progress(min(c.earned / c.maximum, 1.0) if c.maximum else 0.0,
                    text=f"{c.name}: {c.earned:.0f}/{c.maximum} — {c.detail}")
st.caption(ats.disclaimer)
with st.expander("How this score is calculated", icon=":material/calculate:"):
    st.markdown(
        "Deterministic points, all derived from your document only:\n\n"
        "| Component | Max | Earned for |\n|---|---|---|\n"
        "| Contact | 20 | email (10) + phone (7) + clear name line (3) |\n"
        "| Skills | 25 | labeled section (10) + breadth, 3/category (max 15) |\n"
        "| Experience | 25 | labeled section (10) + 5 per dated/bulleted entry (max 15) |\n"
        "| Education | 15 | labeled section (7) + 4 per degree line (max 8) |\n"
        "| Projects | 10 | section (5) + 2.5 per item |\n"
        "| Certifications | 5 | 2.5 per recognized certification keyword |\n\n"
        "No randomness, no ML, no vendor integration — the same resume always yields "
        "the same score."
    )

# ── 4. Skills breakdown ──────────────────────────────────────────────────────
st.space("small")
with st.container(border=True):
    st.markdown("**Extracted skills by category**")
    st.caption("Extracted by literal matching against TechPulse's reusable skill dictionary "
               "with alias normalization — unknown or ambiguous tokens are ignored, never guessed.")
    if resume.skills_by_category:
        cat_cols = st.columns(min(3, len(resume.skills_by_category)))
        for i, (cat, skills) in enumerate(resume.skills_by_category.items()):
            with cat_cols[i % len(cat_cols)]:
                st.markdown(f"**:material/category: {cat}**")
                for s in skills:
                    st.markdown(f":green-badge[{s}]")
    else:
        st.warning("No known technical skills were detected. If your resume lists skills with "
                   "unusual spellings, standardize them (e.g. 'Postgres' → 'PostgreSQL').",
                   icon=":material/warning:")

# ── 5. Job compatibility ─────────────────────────────────────────────────────
st.space("small")
with st.container(border=True):
    st.markdown("**Job compatibility**")
    match_mode = st.radio(
        "Match against", ["A job from the dataset", "A pasted job description"],
        horizontal=True, label_visibility="collapsed")

    match_result = None
    if match_mode == "A job from the dataset":
        titles = (df["title_clean"].fillna(df["title"]).astype(str) + " @ " + df["company_name"].astype(str))
        titles = sorted(titles.unique())
        choice = st.selectbox("Choose a job record", titles,
                              help="1,700+ historical records — availability not verified.")
        sel_mask = titles.index(choice)
        job_row = df.iloc[sel_mask]
        if st.button("Analyze compatibility", icon=":material/rule:"):
            st.session_state["run_match"] = True
        if st.session_state.get("run_match"):
            match_result = match_resume_to_dataset_job(
                resume.raw_text, resume.all_skills, job_row)
            match_result.job_title = choice
    else:
        desc = st.text_area("Paste the job description",
                            height=220,
                            placeholder="Paste the full job posting text here...")
        custom_title = st.text_input("Job title (optional)", placeholder="e.g. Senior Data Engineer")
        if st.button("Analyze compatibility", icon=":material/rule:") and desc.strip():
            st.session_state["run_match"] = True
        if st.session_state.get("run_match") and desc.strip():
            match_result = match_resume_to_description(
                resume.raw_text, resume.all_skills, desc, custom_title)
        elif st.session_state.get("run_match") and not desc.strip():
            st.warning("Paste a job description first.", icon=":material/warning:")

    if match_result is not None:
        if not resume.all_skills:
            st.warning("No skills were extracted from the resume, so the skill component is 0 — "
                       "the compatibility number below is unreliable until skills are detected.",
                       icon=":material/warning:")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Compatibility", f"{match_result.compatibility:.0f}/100", border=True)
        m2.metric("Skill coverage", f"{match_result.coverage_pct:.0f}%", border=True)
        m3.metric("TF-IDF similarity", f"{match_result.similarity * 100:.0f}%", border=True)
        m4.metric("Matched / required",
                  f"{len(match_result.matched)}/{len(match_result.required_skills) or '—'}", border=True)

        if match_result.matched:
            st.markdown("**:material/check: Matching skills**")
            st.markdown(" ".join(f":green-badge[{s}]" for s in match_result.matched[:15]))
        if match_result.missing:
            st.markdown("**:material/close: Missing skills**")
            st.markdown(" ".join(f":red-badge[{s}]" for s in match_result.missing[:15]))
        if not match_result.required_skills:
            st.caption("This source disclosed no recognizable required skills — compatibility "
                       "rests on text similarity alone.")
        st.caption("Compatibility = 60% skill coverage + 40% TF-IDF text similarity "
                   "(components counted once — no double counting). It is a relevance "
                   "indicator, not a hiring probability.")

        # ── 6. Improvement suggestions ──
        st.space("small")
        st.markdown("**Improvement suggestions**")
        for sug in generate_suggestions(resume, match_result, df):
            icon = {"high": "red", "medium": "yellow", "low": "gray"}[sug.priority]
            st.markdown(f":{icon}-badge[{sug.priority.upper()}] **{sug.category}** — {sug.text}")
