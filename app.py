import streamlit as st

st.set_page_config(
    page_title="TechPulse - Career & Job Market Intelligence",
    page_icon=":material/work:",
    layout="wide"
)

from utils.theme import apply_base_css

apply_base_css()

# Grouped sidebar navigation by domain
pages = {
    "DASHBOARD": [
        st.Page("app_pages/overview.py", title="Overview",
                icon=":material/dashboard:", default=True),
    ],
    "CAREER INTELLIGENCE": [
        st.Page("app_pages/career_matcher.py", title="Career Matcher",
                icon=":material/psychology:"),
        st.Page("app_pages/job_recommendations.py", title="Job Recommendations",
                icon=":material/auto_awesome:"),
        st.Page("app_pages/skill_gap_analyzer.py", title="Skill Gap Analyzer",
                icon=":material/swap_vert:"),
        st.Page("app_pages/resume_analyzer.py", title="Resume Analyzer",
                icon=":material/document_scanner:"),
    ],
    "JOB MARKET": [
        st.Page("app_pages/market_intelligence.py", title="Market Intelligence",
                icon=":material/analytics:"),
        st.Page("app_pages/job_explorer.py", title="Job Explorer",
                icon=":material/manage_search:"),
        st.Page("app_pages/salary_benchmark.py", title="Salary Benchmark",
                icon=":material/payments:"),
        st.Page("app_pages/salary_prediction.py", title="Salary Prediction",
                icon=":material/insights:"),
    ],
    "AI ASSISTANCE": [
        st.Page("app_pages/career_assistant.py", title="AI Career Assistant",
                icon=":material/forum:"),
    ],
}

# Sidebar Brand Header with TechPulse Red & Gold Badging
with st.sidebar:
    st.markdown("### <span style='color:#E32636'>:material/insights:</span> <span style='color:#FFD76A'>TechPulse</span>", unsafe_allow_html=True)
    st.caption("AI-powered job role recommendation & market intelligence engine.")
    with st.container(horizontal=True):
        st.badge("1,700+ Jobs", color="red")
        st.badge("ML v1.2", color="green")
    st.divider()
    st.markdown("### <span style='color:#D4AF37'>:material/filter_list: Navigation</span>", unsafe_allow_html=True)

nav = st.navigation(pages, position="sidebar")
nav.run()
