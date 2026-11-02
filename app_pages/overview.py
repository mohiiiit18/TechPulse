#!/usr/bin/env python
"""
TechPulse — Dashboard (Overview).

Personalized landing page. Every number shown is computed directly from
the dataset and production ML models — zero external database dependencies.
"""
import streamlit as st
import pandas as pd
import altair as alt

from utils.theme import (
    apply_base_css, hero, section_title, kpi_row,
)
from utils.data_loader import (
    load_data, load_model, get_dataset_date_range,
)

apply_base_css()
df = load_data()
model, encoder = load_model()
_latest = get_dataset_date_range()["latest"]
_latest_str = f"{_latest:%b %Y}" if _latest is not None else "August 2026"

hero("Welcome to TechPulse 👋",
     f"AI-Powered Career & Job Market Intelligence Platform — Historical Dataset through {_latest_str}.")

# ── Dataset-level KPIs ────────────────────────────────────────────────────────
skills_exploded = df["parsed_skills"].explode().dropna()
kpi_row([
    {"label": "Job records", "value": f"{len(df):,}"},
    {"label": "Companies", "value": f"{df['company_name'].nunique():,}"},
    {"label": "Remote roles", "value":
        f"{(df['remote_type'] == 'Remote').mean() * 100:.0f}%"},
    {"label": "Skills tracked", "value": f"{skills_exploded.nunique():,}"},
])

st.space("small")

# ── Market Snapshot ──────────────────────────────────────────────────────────
c_role, c_skill = st.columns(2, gap="medium")

with c_role:
    with st.container(border=True):
        section_title("Role distribution", ":material/pie_chart:")
        role_counts = df["job_role"].value_counts().reset_index()
        role_counts.columns = ["Role", "Postings"]
        role_chart = (
            alt.Chart(role_counts)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#E32636")
            .encode(
                x=alt.X("Postings:Q", title="Job Postings"),
                y=alt.Y("Role:N", sort="-x", title=None),
                tooltip=["Role", "Postings"]
            )
            .properties(height=240)
        )
        st.altair_chart(role_chart, width="stretch", alt="Job postings by role")

with c_skill:
    with st.container(border=True):
        section_title("Top demanded skills", ":material/trending_up:")
        top_skills = skills_exploded.value_counts().head(8).reset_index()
        top_skills.columns = ["Skill", "Postings"]
        skill_chart = (
            alt.Chart(top_skills)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#D4AF37")
            .encode(
                x=alt.X("Postings:Q", title="Postings"),
                y=alt.Y("Skill:N", sort="-x", title=None),
                tooltip=["Skill", "Postings"]
            )
            .properties(height=240)
        )
        st.altair_chart(skill_chart, width="stretch", alt="Top skills in demand")

# ── Quick access cards ───────────────────────────────────────────────────────
st.space("small")
section_title("Quick access", ":material/bolt:")
cards = [
    ("app_pages/career_matcher.py", ":material/psychology:", "Career Matcher",
     "Predict your best-fit role from your skills."),
    ("app_pages/job_recommendations.py", ":material/auto_awesome:", "Job Recommendations",
     "Personalized matches ranked by relevance."),
    ("app_pages/resume_analyzer.py", ":material/document_scanner:", "Resume Analyzer",
     "Score and match your resume against jobs."),
    ("app_pages/skill_gap_analyzer.py", ":material/swap_vert:", "Skill Gap Analyzer",
     "Find missing skills and a learning roadmap."),
    ("app_pages/market_intelligence.py", ":material/analytics:", "Market Intelligence",
     "Hiring trends across the dataset."),
    ("app_pages/salary_benchmark.py", ":material/payments:", "Salary Benchmark",
     "USD percentile benchmarks by role."),
    ("app_pages/salary_prediction.py", ":material/insights:", "Salary Prediction",
     "XGBoost machine learning salary regressor."),
    ("app_pages/job_explorer.py", ":material/manage_search:", "Job Explorer",
     "Search and filter all job records."),
]
ccols = st.columns(4)
for i, (path, icon, title, blurb) in enumerate(cards):
    with ccols[i % 4]:
        with st.container(border=True):
            st.markdown(f"#### {icon} {title}")
            st.caption(blurb)
            if st.button("Open", key=f"qa_{i}", use_container_width=True):
                st.switch_page(path)
