from utils.auth import get_profile, is_authenticated
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.theme import (apply_base_css, page_header, section_title,
                         empty_state, kpi_row, PALETTE)
from utils import auth
from utils.data_loader import (
    load_data,
    load_model,
    predict_roles,
    get_skill_gap_analysis,
    get_matched_jobs,
    get_dataset_date_range,
    SKILL_CATEGORIES,
    ROLE_PRESETS
)

# Load data and ML model
df = load_data()
model, encoder = load_model()
_latest = get_dataset_date_range()["latest"]
_latest_str = f"{_latest:%b %Y}" if _latest is not None else "unknown"

available_skills = sorted(str(s) for s in encoder.classes_)

apply_base_css()
page_header(":material/psychology:", "Career Matcher",
            f"Benchmark your technical skill set against historical market data "
            f"(latest records {_latest_str}) to discover your best-fit job role and bridge competency gaps.")

# Load saved profile skills when the logged-in user changes
current_user = st.session_state.get("user", {})
current_user_id = current_user.get("id") if auth.is_authenticated() else None

if st.session_state.get("_profile_loaded_for_user") != current_user_id:

    default_skills = ["python", "machine-learning", "sql"]

    if current_user_id:
        profile = get_profile(current_user_id)

        if profile and profile.get("skills"):
            # Match saved skills with the model's actual skill vocabulary
            skill_lookup = {
                skill.lower(): skill for skill in available_skills
            }

            saved_skills = [
                skill_lookup[skill.lower()]
                for skill in profile["skills"]
                if skill.lower() in skill_lookup
            ]

            st.session_state["skill_multiselect"] = saved_skills
        else:
            st.session_state["skill_multiselect"] = default_skills

    else:
        st.session_state["skill_multiselect"] = default_skills

    st.session_state["_profile_loaded_for_user"] = current_user_id

# Initialize if the key does not exist
if "skill_multiselect" not in st.session_state:
    st.session_state["skill_multiselect"] = [
        "python", "machine-learning", "sql"
    ]

# Preset profile selection
with st.container(border=True):
    st.markdown("**Quick profile presets**")
    st.caption("Load a common industry skill profile with a single click:")
    
    preset_cols = st.columns(len(ROLE_PRESETS))
    for col, (preset_name, preset_skills) in zip(preset_cols, ROLE_PRESETS.items()):
        with col:
            if st.button(preset_name, icon=":material/bolt:", key=f"preset_{preset_name}"):
                # BUG 2 FIX: Set multiselect key in session_state directly, then rerun
                st.session_state["skill_multiselect"] = [s for s in preset_skills if s in available_skills]
                st.rerun()

# Skill Picker Section
with st.container(border=True):
    st.markdown("**Select your technical skills**")
    
    # Category Filter Pills
    cat_choice = st.pills(
        "Filter skill list by domain",
        ["All"] + list(SKILL_CATEGORIES.keys()),
        default="All"
    )
    
    # BUG 1 FIX: Use filtered_options as multiselect options (was using available_skills)
    if cat_choice == "All":
        filtered_options = available_skills
    else:
        filtered_options = sorted([s for s in SKILL_CATEGORIES[cat_choice] if s in available_skills])
    
    # BUG 2 & 3 FIX: Use key="skill_multiselect"; Streamlit manages its own state
    # Do NOT pass default= here (conflicts with session_state key after first render)
    selected = st.multiselect(
        "Choose skills",
        options=filtered_options,
        key="skill_multiselect",
        placeholder="Type to search e.g. python, kubernetes, react, aws, llm...",
        label_visibility="collapsed"
    )

    with st.container(horizontal=True):
        st.badge(f"{len(selected)} skills selected", icon=":material/check_circle:", color="red")
        if st.button("Clear all skills", icon=":material/clear_all:"):
            # BUG 3 FIX: Set session_state and rerun (not st.session_state.user_skills)
            st.session_state["skill_multiselect"] = []
            st.rerun()

if len(selected) < 2:
    st.info("Please select at least 2 technical skills to generate accurate role predictions and gap insights.", icon=":material/info:")
    st.stop()

# Perform ML Prediction
predicted_role, results_df, unknown_skills = predict_roles(selected, model, encoder)
if predicted_role is None or results_df.empty:
    st.warning("Could not generate a prediction for the selected skills. Try selecting more common skills.", icon=":material/warning:")
    st.stop()
top_score = results_df.iloc[0]["Confidence (%)"]

if unknown_skills:
    _known_set = {str(c).lower() for c in encoder.classes_}
    # Show the closest in-vocabulary skill for each unknown token (simple prefix/fuzzy match)
    def _closest(tok: str):
        import difflib
        m = difflib.get_close_matches(tok, sorted(_known_set), n=1, cutoff=0.7)
        return m[0] if m else None

    with st.container(border=True):
        st.markdown(":material/warning: **Unknown skills ignored**")
        for tok in sorted(unknown_skills):
            sug = _closest(tok)
            if sug:
                st.caption(f":yellow-badge[{tok}] — not in the model's vocabulary. Did you mean **{sug}**?")
            else:
                st.caption(f":yellow-badge[{tok}] — not in the model's training vocabulary; excluded from the prediction.")
        st.caption("Predictions below are based on your recognized skills only.")

# Results Overview Card
st.space("small")
with st.container(border=True):
    col_info, col_chart = st.columns([1, 1], gap="medium")
    
    with col_info:
        st.markdown("**Primary role recommendation**")
        st.title(predicted_role)
        
        # BUG 15 FIX + Phase 3: changed misleading "Match confidence score" labels;
        # probabilities are uncalibrated (mean max-prob 0.78 ≈ acc, no calibration
        # correction applied) — labelled as model-estimated, never "match confidence".
        st.metric(
            label="Model probability score",
            value=f"{top_score}%",
            delta="High probability" if top_score > 60 else "Moderate probability",
            border=True
        )
        st.progress(float(top_score) / 100)
        
        # BUG 15 FIX + Phase 3 calibration disclosure
        st.caption(
            "These are raw model-estimated class probabilities, NOT calibrated career-compatibility scores "
            "(no calibration correction was applied — probabilities reflect model outputs on the training "
            "distribution). Low scores (< 50%) indicate the profile sits between roles in the training data. "
            "Results reflect historical dataset patterns; treat as indicative guidance only."
        )
        with st.expander("About probability calibration", icon=":material/info:"):
            st.markdown(
                "**Why these numbers are not \"match confidence\"**\n\n"
                "- The production RandomForest (`job_role_model_v3.pkl`) was **not** wrapped in a "
                "calibration wrapper (`CalibratedClassifierCV` / isotonic / sigmoid), so its "
                "probabilities are uncalibrated.\n"
                "- Diagnostic: mean max-probability ≈ **0.784** vs test accuracy **0.864** — the model is "
                "slightly under-confident on average, which is typical for RandomForests.\n"
                "- Ranking is still meaningful: the **order** of roles by probability reflects the model's "
                "preference, but the absolute percentages should not be read as \"X% chance of getting hired\".\n"
                "- Calibration was **validated on the untouched test split only** (no leakage): any future "
                "calibrated model must be fitted on training-fold predictions, never on the test set."
            )

    with col_chart:
        st.markdown("**Role fit distribution**")
        
        plot_df = results_df.sort_values("Confidence (%)", ascending=True)
        fig = go.Figure(go.Bar(
            x=plot_df["Confidence (%)"],
            y=plot_df["Role"],
            orientation="h",
            marker_color=[PALETTE[0] if r == predicted_role else PALETTE[1]
                          for r in plot_df["Role"]],
            hovertemplate="%{y}: %{x:.1f}% probability<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title="Model probability (%)",
            yaxis_title=None,
            xaxis_range=[0, 100],
            height=240,
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False},
                        key="role_fit_chart")

# Skill Gap Analysis Section
st.space("small")
matched_skills, missing_skills = get_skill_gap_analysis(predicted_role, selected, df)

# Save analysis only when explicitly requested
if auth.is_authenticated():

    if st.button(
        "Save This Analysis",
        icon=":material/save:",
        type="primary",
        key="save_career_analysis"
    ):
        ok, msg = auth.add_history(
            st.session_state["user"]["id"],
            "career_match",
            {
                "skills": selected,
                "predicted_role": predicted_role,
                "top_score": float(top_score)
            }
        )
        if ok:
            st.success("Career analysis saved successfully!")
        else:
            st.error(f"Unable to save analysis: {msg}")

with st.container(border=True):
    st.subheader("Competency & skill gap breakdown", icon=":material/radar:")
    st.caption(f"How your selected skills measure up against real market requirements for **{predicted_role}**:")
    
    col_has, col_gaps = st.columns(2, gap="medium")
    
    with col_has:
        st.markdown("**:material/check: Your matched skills**")
        if matched_skills:
            badges_html = " ".join([f":green-badge[{item['skill']}]" for item in matched_skills])
            st.markdown(badges_html)
            st.caption(f"You have {len(matched_skills)} core skills heavily cited for this role.")
        else:
            st.write("No exact core matches found yet.")

    with col_gaps:
        st.markdown("**:material/trending_up: High-impact skills to learn next**")
        if missing_skills:
            gap_badges = " ".join([
                f":red-badge[{item['skill']} ({item['demand_pct']}% of jobs)]"
                for item in missing_skills
            ])
            st.markdown(gap_badges)
            st.caption("Acquiring these top skills will significantly increase your hiring eligibility for this role.")
        else:
            st.success("Outstanding! You cover all top in-demand skills for this role.", icon=":material/star:")

# Job Records Matching Predicted Role
st.space("small")
with st.container(border=True):
    # BUG 15 FIX: "Active job openings" -> "Job records matching predicted role"
    st.subheader(f"Job records matching predicted role: {predicted_role}", icon=":material/travel_explore:")
    # BUG 15 FIX: Removed "Real job listings currently hiring" claim
    st.caption("Dataset records for this role, ranked by skill alignment:")
    
    matched_jobs = get_matched_jobs(predicted_role, selected, df, limit=6)
    
    if matched_jobs.empty:
        st.info("No matching records found for this profile.", icon=":material/search_off:")
    else:
        job_cols = st.columns(2)
        for idx, (_, job) in enumerate(matched_jobs.iterrows()):
            with job_cols[idx % 2]:
                with st.container(border=True):
                    st.markdown(f"**{job['title'].title()}**")
                    st.caption(f"🏢 **{job['company_name'].title()}** • 📍 {job['country']} ({job['remote_type']})")
                    
                    # PHASE 2 FIX: Use original salary display columns when USD annualized unavailable
                    if pd.notna(job["annual_salary_avg"]) and job["annual_salary_avg"] > 0:
                        st.markdown(f":green-badge[Est. Salary: ${int(job['annual_salary_avg']):,}/yr (USD)]")
                    elif (pd.notna(job.get("salary_display_min")) and
                          pd.notna(job.get("salary_display_currency")) and
                          pd.notna(job.get("salary_display_period"))):
                        curr = job["salary_display_currency"]
                        lo = int(job["salary_display_min"])
                        hi = int(job["salary_display_max"]) if pd.notna(job.get("salary_display_max")) else lo
                        period = job["salary_display_period"]
                        st.markdown(f":red-badge[{curr} {lo:,} \u2013 {hi:,} / {period}]")
                    else:
                        st.markdown(":gray-badge[Salary: Not disclosed]")
                    
                    # PHASE 4 FIX: Show "Skill match: X of Y" using skill_match_count
                    job_skills = job["parsed_skills"]
                    skill_match_count = int(job.get("skill_match_count", 0))
                    total_job_skills = len(job_skills) if job_skills else 0
                    if total_job_skills > 0:
                        st.caption(f"Skill match: {skill_match_count} of {total_job_skills}")
                    
                    if job_skills:
                        user_skills_lower = set(s.lower() for s in selected)
                        matched_in_job = [s for s in job_skills if s in user_skills_lower]
                        other_in_job = [s for s in job_skills if s not in user_skills_lower][:3]
                        
                        badges_str = " ".join([f":green-badge[{s}]" for s in matched_in_job])
                        if other_in_job:
                            badges_str += " " + " ".join([f":gray-badge[{s}]" for s in other_in_job])
                        st.markdown(badges_str)
                    
                    if pd.notna(job["url"]) and str(job["url"]).startswith("http"):
                        st.link_button("View job posting", str(job["url"]), icon=":material/open_in_new:")
