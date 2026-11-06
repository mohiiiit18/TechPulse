import streamlit as st
import pandas as pd
from utils.theme import apply_base_css, page_header, empty_state, kpi_row
from utils.data_loader import load_data, get_dataset_date_range, SKILL_CATEGORIES
from src.recommendation_engine import UserProfile, recommend, explain_recommendation
from src.ranking import RankWeights
from src.recommendation_evaluation import no_ground_truth_available
from utils.auth import is_authenticated, save_job, is_job_saved, remove_saved_job, get_profile

df = load_data()
_latest = get_dataset_date_range()["latest"]
_latest_str = f"{_latest:%B %Y}" if _latest is not None else "unknown"

apply_base_css()
page_header(":material/auto_awesome:", "Job Recommendations",
            f"Personalized content-based recommendations ranked by skills, text similarity, "
            f"location, salary compatibility and work arrangement — historical dataset, latest records {_latest_str}. "
            "Recommendation scores are relevance indicators, NOT probabilities of getting hired.")

# ── User profile input (Step 2) ─────────────────────────────────────────────
available_skills = sorted({s for skills in df["parsed_skills"] for s in skills})

# Auto-load saved profile skills if user is authenticated
default_rec_skills = []
if is_authenticated():
    u_profile = get_profile(st.session_state["user"]["id"])
    if u_profile and u_profile.get("skills"):
        skill_map = {s.lower(): s for s in available_skills}
        default_rec_skills = [skill_map[s.lower()] for s in u_profile["skills"] if s.lower() in skill_map]

with st.container(border=True):
    st.markdown("**Your profile**")
    st.caption("All fields are optional — incomplete profiles are handled safely "
               "(unavailable components are excluded from scoring with renormalized weights).")

    profile_col, filter_col = st.columns([1, 1], gap="large")

    with profile_col:
        selected_skills = st.multiselect(
            "Technical skills",
            options=available_skills,
            default=default_rec_skills,
            placeholder="Type to search e.g. python, kubernetes, react, aws...",
        )
        roles = ["No preference"] + sorted(df["job_role"].dropna().unique().tolist())
        pref_role = st.selectbox("Preferred job role", roles)

        exp_levels = ["No preference", "Entry level (unspecified)", "Senior",
                      "Staff", "Lead", "Principal"]
        pref_exp = st.selectbox("Experience level", exp_levels)
        st.caption("Experience is recorded as profile context only — the dataset's "
                   "`seniority` column is 60% unspecified, so it is not used as a scoring component.")

    with filter_col:
        c1, c2 = st.columns(2)
        with c1:
            countries = ["No preference"] + sorted([c for c in df["country"].unique() if c != "Unknown"])
            pref_country = st.selectbox("Preferred country", countries)
        with c2:
            cities = ["No preference"] + sorted([c for c in df["city"].unique() if c != "Unknown"])
            pref_city = st.selectbox("Preferred city", cities)

        work_opts = ["No preference", "Remote", "On-site / Hybrid"]
        pref_work = st.segmented_control("Preferred work arrangement", work_opts)

        s1, s2 = st.columns(2)
        with s1:
            salary_min = st.number_input("Expected minimum salary (USD/yr)", 0, 1_000_000, 0, 5_000)
        with s2:
            salary_max = st.number_input("Expected maximum salary (USD/yr)", 0, 1_000_000, 0, 5_000)

    # Recommendation filters (Step 7)
    st.divider()
    st.markdown("**Filters & weights**")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        filter_role = st.selectbox("Filter: job category", ["All"] + sorted(df["job_role"].dropna().unique().tolist()))
    with f2:
        filter_country = st.selectbox("Filter: location", ["All"] + sorted(df["country"].unique()))
    with f3:
        filter_remote = st.selectbox("Filter: work style", ["All", "Remote", "On-site / Hybrid"])
    with f4:
        min_coverage = st.slider("Minimum skill coverage (%)", 0, 100, 0)

    f5, f6 = st.columns(2)
    with f5:
        filter_company = st.text_input("Filter: company name contains", placeholder="e.g. stripe")
    with f6:
        sort_by = st.selectbox("Sort by", ["Best match", "Highest salary (USD)", "Most recent posting"])

    with st.expander("Advanced: component weights", icon=":material/tune:"):
        st.caption("Default: skill 40% · text similarity 30% · location 15% · salary 10% · work 5%. "
                   "Weights are renormalized over the components that have data.")
        w1, w2, w3, w4, w5 = st.columns(5)
        w_skill = w1.number_input("Skill", 0.0, 1.0, 0.40, 0.05)
        w_text = w2.number_input("Text", 0.0, 1.0, 0.30, 0.05)
        w_loc = w3.number_input("Location", 0.0, 1.0, 0.15, 0.05)
        w_sal = w4.number_input("Salary", 0.0, 1.0, 0.10, 0.05)
        w_work = w5.number_input("Work", 0.0, 1.0, 0.05, 0.05)
        weights = RankWeights.from_dict({"skill": w_skill, "text": w_text,
                                         "location": w_loc, "salary": w_sal, "work": w_work})

profile = UserProfile(
    skills=selected_skills,
    preferred_role=None if pref_role == "No preference" else pref_role,
    experience_level=None if pref_exp == "No preference" else pref_exp,
    preferred_country=None if pref_country == "No preference" else pref_country,
    preferred_city=None if pref_city == "No preference" else pref_city,
    preferred_work_arrangement=None if pref_work in (None, "No preference") else pref_work,
    salary_min=float(salary_min) if salary_min > 0 else None,
    salary_max=float(salary_max) if salary_max > 0 else None,
)

filters = {
    "job_role": None if filter_role == "All" else filter_role,
    "country": None if filter_country == "All" else filter_country,
    "remote_type": None if filter_remote == "All" else filter_remote,
    "company": filter_company or None,
    "min_skill_coverage": float(min_coverage),
}

if st.button("Find matching jobs", icon=":material/auto_awesome:", type="primary"):
    st.session_state["recs_requested"] = True

if not st.session_state.get("recs_requested"):
    st.info("Fill in your profile (skills recommended) and click **Find matching jobs**.", icon=":material/info:")
    st.stop()

if profile.is_empty():
    st.warning("Your profile is empty — every component is unknown, so no meaningful ranking is possible. "
               "Add at least a few skills or a preferred role.", icon=":material/warning:")
    st.stop()

with st.spinner("Scoring 1,700+ job records..."):
    recs = recommend(df, profile, weights=weights, filters=filters, top_n=20)

# Sorting (Step 7)
if not recs.empty:
    if sort_by == "Highest salary (USD)" and "annual_salary_avg" in recs.columns:
        recs = recs.sort_values("annual_salary_avg", ascending=False, na_position="last")
    elif sort_by == "Most recent posting" and "posted_date" in recs.columns:
        recs = recs.sort_values("posted_date", ascending=False, na_position="last")

if recs.empty:
    empty_state(":material/search_off:", "No matches found",
                "No job records match your profile and filters. Try relaxing filters or adding more common skills.")
    st.stop()

st.success(f"Top {len(recs)} recommendations", icon=":material/check_circle:")

job_cols = st.columns(2)
for idx, (_, job) in enumerate(recs.iterrows()):
    with job_cols[idx % 2]:
        with st.container(border=True):
            st.markdown(f"**{str(job['title_clean'] or job['title']).title()}**")
            st.caption(f"🏢 **{job['company_name'].title()}** • 📍 {job['country']}"
                       + (f" / {job['city']}" if job['city'] != 'Unknown' else "")
                       + f" • {job['remote_type']}"
                       + (f" • 🕐 {job['posted_date']:%b %Y}" if pd.notna(job.get('posted_date')) else ""))

            # Salary — only dataset-disclosed values, never invented
            if pd.notna(job.get("annual_salary_avg")) and job.get("annual_salary_avg", 0) > 0:
                st.markdown(f":green-badge[Disclosed: ${int(job['annual_salary_min']):,}–{int(job['annual_salary_max']):,}/yr (USD)]")
            elif pd.notna(job.get("salary_display_currency")) and pd.notna(job.get("salary_display_min")):
                hi = int(job["salary_display_max"]) if pd.notna(job.get("salary_display_max")) else int(job["salary_display_min"])
                st.markdown(f":red-badge[{job['salary_display_currency']} {int(job['salary_display_min']):,}–{hi:,} / {job['salary_display_period']}]")
            else:
                st.markdown(":gray-badge[Salary: Not disclosed]")

            c_a, c_b, c_c = st.columns(3)
            c_a.metric("Match score", f"{job['recommendation_score']:.0f}", border=True)
            c_b.metric("Skill coverage", f"{job['skill_coverage']:.0f}%", border=True)
            c_c.metric("Text similarity", f"{job['similarity']*100:.0f}%", border=True)

            if job["matched_skills"]:
                st.markdown(" ".join(f":green-badge[{s}]" for s in job["matched_skills"][:10]))
            if job["missing_skills"]:
                st.caption("Missing: " + ", ".join(job["missing_skills"][:6]))

            st.caption(f"💡 {explain_recommendation(job, df)}")

            has_url = pd.notna(job["url"]) and str(job["url"]).startswith("http")
            job_url = str(job["url"]) if has_url else ""
            target_job_id = job_url if has_url else f"{job.get('title', 'job')}_{idx}"

            col_view, col_save = st.columns(2)

            with col_view:
                if has_url:
                    st.link_button(
                        "View Job",
                        job_url,
                        icon=":material/open_in_new:",
                        use_container_width=True
                    )
                else:
                    st.button(
                        "No Link",
                        disabled=True,
                        icon=":material/link_off:",
                        key=f"no_link_{idx}",
                        use_container_width=True
                    )

            with col_save:
                if is_authenticated():
                    user_id = st.session_state["user"]["id"]
                    already_saved = is_job_saved(user_id, target_job_id)

                    if already_saved:
                        if st.button(
                            "Saved",
                            icon=":material/bookmark_added:",
                            key=f"saved_job_{idx}",
                            use_container_width=True
                        ):
                            remove_saved_job(user_id, target_job_id)
                            st.rerun()
                    else:
                        if st.button(
                            "Save Job",
                            icon=":material/bookmark_add:",
                            key=f"save_job_{idx}",
                            use_container_width=True
                        ):
                            try:
                                save_job(user_id, target_job_id)
                                st.success("Job saved successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Unable to save job: {e}")
                else:
                    st.caption("Log in to save jobs.")

st.divider()
with st.expander("Why can't recommendation quality be validated yet?", icon=":material/help:"):
    st.write(no_ground_truth_available())
st.caption("Scores are relevance indicators computed from historical data only. "
           "Record availability is NOT verified — listings may be closed. ")
