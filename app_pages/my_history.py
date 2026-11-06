import streamlit as st
from utils.theme import apply_base_css, page_header
from utils.data_loader import load_data
from utils.auth import (
    get_history,
    delete_history_item,
    get_saved_jobs_detailed,
    remove_saved_job,
    is_authenticated
)

apply_base_css()

page_header(":material/history:", "My History & Saved Jobs",
            "View all your saved career matches, resume audits, skill gap evaluations, and bookmarked jobs.")

if not is_authenticated():
    st.warning("Please log in from the Account page to view your saved history and bookmarked jobs.", icon=":material/lock:")
    st.stop()

user = st.session_state["user"]
user_id = user["id"]

# Load job dataset for metadata lookup
df = load_data()

tab_history, tab_saved_jobs = st.tabs(["Analysis History", "Saved Jobs"])

# ── Tab 1: Analysis History ──────────────────────────────────────────────────
with tab_history:
    history = get_history(user_id)

    if not history:
        st.info("No analysis history found yet. As you run Career Matcher, Resume Analyzer, and Skill Gap Analyzer, you can save your results here.", icon=":material/info:")
    else:
        st.markdown(f"**Total analysis records:** :green-badge[{len(history)}]")
        st.space("small")

        for h_idx, item in enumerate(history):
            created = item.get("created_at")
            date_str = created.strftime('%d %b %Y, %I:%M %p') if hasattr(created, 'strftime') else str(created or '')
            analysis_type = item.get("analysis_type", "analysis").replace('_', ' ').title()
            hist_id = item.get("id")

            with st.container(border=True):
                h_col_title, h_col_del = st.columns([5, 1])
                with h_col_title:
                    st.markdown(f"**{analysis_type}** • <span style='color:gray;'>{date_str}</span>", unsafe_allow_html=True)
                with h_col_del:
                    if hist_id and st.button("Delete", icon=":material/delete:", key=f"del_hist_{hist_id}_{h_idx}", use_container_width=True):
                        delete_history_item(user_id, hist_id)
                        st.rerun()

                res_val = item.get("result", {})
                if isinstance(res_val, dict):
                    # Show summary KPIs if present
                    if "predicted_role" in res_val:
                        st.caption(f"🎯 **Predicted Role:** {res_val.get('predicted_role')} (Score: {res_val.get('top_score', 0):.1f}%)")
                    elif "target_role" in res_val:
                        st.caption(f"🎯 **Target Role:** {res_val.get('target_role')} (Coverage: {res_val.get('coverage_pct', 0):.1f}%)")
                    elif "skills_found" in res_val:
                        st.caption(f"📄 **Resume Audit:** {res_val.get('skills_found', 0)} skills detected across {res_val.get('pages', 1)} page(s)")

                with st.expander("View Full JSON Payload", icon=":material/data_object:"):
                    st.json(res_val)

# ── Tab 2: Saved Jobs ────────────────────────────────────────────────────────
with tab_saved_jobs:
    saved_jobs = get_saved_jobs_detailed(user_id, df)

    if not saved_jobs:
        st.info("No saved jobs yet. Browse Job Recommendations and click 'Save Job' to bookmark opportunities here.", icon=":material/bookmark_border:")
    else:
        st.markdown(f"**Total bookmarked jobs:** :green-badge[{len(saved_jobs)}]")
        st.space("small")

        for s_idx, job in enumerate(saved_jobs):
            job_ident = job.get("job_id", "")
            saved_time = job.get("saved_at")
            time_str = saved_time.strftime('%d %b %Y, %I:%M %p') if hasattr(saved_time, 'strftime') else str(saved_time or '')

            with st.container(border=True):
                col_info, col_actions = st.columns([3, 1], gap="medium")

                with col_info:
                    st.markdown(f"### {job.get('title', 'Job Listing')}")
                    st.caption(f"🏢 **{job.get('company_name', 'Unknown Company')}** • 📍 {job.get('location', 'Global')}")
                    
                    # Salary Badge
                    sal = job.get("salary", "Not disclosed")
                    if "$" in sal or "USD" in sal:
                        st.markdown(f":green-badge[{sal}]")
                    elif sal != "Not disclosed":
                        st.markdown(f":red-badge[{sal}]")
                    else:
                        st.markdown(":gray-badge[Salary: Not disclosed]")

                    st.caption(f"Saved on {time_str}")

                with col_actions:
                    job_url = job.get("url", "")
                    if job_url and job_url.startswith("http"):
                        st.link_button("Open Job Posting", job_url, icon=":material/open_in_new:", use_container_width=True)
                    else:
                        st.button("No External URL", disabled=True, icon=":material/link_off:", key=f"no_ext_url_{s_idx}", use_container_width=True)

                    if st.button("Remove Bookmark", icon=":material/delete:", key=f"del_saved_job_{s_idx}", use_container_width=True):
                        remove_saved_job(user_id, job_ident)
                        st.rerun()
