import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.theme import (apply_base_css, page_header, section_title,
                         empty_state, kpi_row, PALETTE)
from utils.data_loader import load_data, get_dataset_date_range, SKILL_CATEGORIES, ROLE_PRESETS
from utils import auth
from src.skill_gap_analyzer import analyze_skill_gap, gap_table, COVERAGE_NOTE
from src.skill_demand import global_skill_demand, role_skill_demand, role_posting_counts
from src.skill_prioritization import rank_missing_skills, priority_counts, PRIORITY_METHOD
from src.learning_roadmap import build_roadmap, ROADMAP_NOTE, STATUSES

df = load_data()
_latest = get_dataset_date_range()["latest"]
_latest_str = f"{_latest:%b %Y}" if _latest is not None else "unknown"

apply_base_css()
page_header(":material/swap_vert:", "Skill Gap Analyzer",
            f"Compare your skills against the skills actually required by postings in the "
            f"cleaned dataset (latest records {_latest_str}) and get a staged learning "
            "roadmap. All demand statistics come from the dataset — nothing is invented.")

available_skills = sorted(
    {s for lst in df["parsed_skills"] for s in (lst or [])} | set(df["parsed_skills"].explode().dropna().astype(str))
)
roles = sorted(df["job_role"].dropna().unique()) if "job_role" in df.columns else []

# ── Step 2: User profile ─────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("**Your profile**")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        target_role = st.selectbox("Target job role", roles, key="sga_role")
        experience = st.selectbox(
            "Experience level (optional, informational only)",
            ["Not specified", "Entry", "Mid", "Senior"],
            key="sga_exp",
        )
        weekly_hours = st.slider("Learning time availability (hours/week)", 0, 40, 5, 1, key="sga_hours")
        st.caption("Time budget is planning context only — no duration is promised.")
    with pcol2:
        default_skills = st.session_state.get("skill_multiselect", ["python", "sql"])
        chosen = st.multiselect(
            "Current technical skills (multi-select)",
            options=available_skills,
            default=[s for s in default_skills if s in available_skills],
            key="sga_skills",
        )
        if st.button("Import skills from Career matcher", icon=":material/download:",
                     disabled=not st.session_state.get("skill_multiselect")):
            st.session_state["sga_skills"] = [
                s for s in st.session_state["skill_multiselect"] if s in available_skills
            ]
            st.rerun()
        if st.button("Import skills from Resume Analyzer", icon=":material/history_edu:",
                     disabled=not st.session_state.get("resume_skills")):
            st.session_state["sga_skills"] = [
                s for s in st.session_state.get("resume_skills", []) if s in available_skills
            ]
            st.rerun()

if not chosen:
    st.info("Select at least one skill and a target role to run the gap analysis.", icon=":material/info:")
    st.stop()

# ── Step 4: Skill gap analysis ───────────────────────────────────────────────
result = analyze_skill_gap(chosen, target_role, df)
if not result.is_known_role:
    st.warning(f"No postings found for role “{target_role}” in the dataset — nothing was invented. "
               "Pick another role.", icon=":material/warning:")
    st.stop()

st.divider()
section_title("Skill coverage", ":material/radar:")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Skill coverage", f"{result.coverage_pct:.0f}%", border=True)
c2.metric("Matched skills", f"{result.matched_count}/{result.required_count}", border=True)
c3.metric("Missing skills", str(len(result.missing)), border=True)
c4.metric("Postings analyzed", str(result.analyzed_postings), border=True)
st.progress(result.coverage_pct / 100.0 if result.coverage_pct else 0.0)
st.caption(COVERAGE_NOTE)

# Save analysis to user profile history when explicitly requested
if auth.is_authenticated():
    if st.button("Save Skill Gap Analysis", icon=":material/save:", type="primary", key="save_skill_gap"):
        ok, msg = auth.add_history(
            st.session_state["user"]["id"],
            "skill_gap",
            {
                "target_role": target_role,
                "coverage_pct": float(result.coverage_pct),
                "n_skills": len(chosen),
                "matched_skills": list(result.matched)[:20],
                "missing_skills": list(result.missing)[:20]
            }
        )
        if ok:
            st.success("Skill gap analysis saved to your history!")
        else:
            st.error(f"Unable to save: {msg}")

if result.matched:
    st.markdown("**:material/check: Matching skills**")
    st.markdown(" ".join(f":green-badge[{s}]" for s in result.matched[:30]))
if result.missing:
    st.markdown("**:material/close: Missing skills**")
    st.markdown(" ".join(f":red-badge[{s}]" for s in result.missing[:30]))

# ── Step 3: Skill demand chart ───────────────────────────────────────────────
st.divider()
section_title("Skill demand for this role", ":material/trending_up:")
demand = role_skill_demand(df, target_role, top_n=15)
if not demand.empty:
    plot_df = demand.sort_values("pct_of_role_postings", ascending=True)
    fig = go.Figure(go.Bar(
        x=plot_df["pct_of_role_postings"],
        y=plot_df["skill"],
        orientation="h",
        marker_color=[PALETTE[3] if s in result.matched else PALETTE[1]
                      for s in plot_df["skill"]],
        hovertemplate="%{y}: required in %{x:.0f}% of role postings<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="% of role postings requiring the skill",
        yaxis_title="Skill",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                    key="skill_demand_chart")
    st.caption(f"Based on {result.analyzed_postings} postings for {target_role} in the dataset. "
               "Green bars = skills you already have.")

# ── Step 5: Priority ranking ─────────────────────────────────────────────────
st.divider()
section_title("Missing-skill priority ranking", ":material/low_priority:")
ranked = rank_missing_skills(result.missing, role_skill_demand(df, target_role, top_n=None))
counts = priority_counts(ranked)
if ranked:
    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("High priority", counts["high"], border=True)
    pc2.metric("Medium priority", counts["medium"], border=True)
    pc3.metric("Low priority", counts["low"], border=True)
    prio_df = pd.DataFrame([{
        "skill": sp.skill, "priority": sp.priority.title(),
        "demand (% of postings)": sp.demand_score,
    } for sp in ranked])
    st.dataframe(prio_df, width="stretch", hide_index=True)
    st.caption(PRIORITY_METHOD)
else:
    st.success("No missing skills — you cover every skill listed across this role's postings "
               "in the dataset. Coverage is not a hiring guarantee.", icon=":material/celebration:")

# ── Step 6: Roadmap ──────────────────────────────────────────────────────────
st.divider()
section_title("Personalized learning roadmap", ":material/map:")
roadmap = build_roadmap(ranked, target_role, weekly_hours)
if roadmap.items:
    pr1, pr2 = st.columns(2)
    prog = roadmap.progress_summary()
    pr1.metric("Roadmap progress", f"{prog['completion_pct']:.0f}%", border=True)
    pr2.metric("Completed / total", f"{prog['completed']}/{prog['total']}", border=True)
    st.progress(prog["completion_pct"] / 100.0)

    for stage in range(1, roadmap.n_stages + 1):
        items = roadmap.stage_items(stage)
        label = {1: "Stage 1 — High priority", 2: "Stage 2 — Medium priority",
                 3: "Stage 3 — Low priority"}.get(stage, f"Stage {stage}")
        with st.expander(f"{label} ({len(items)} skills)", icon=":material/flag:"):
            for it in items:
                i1, i2 = st.columns([3, 1])
                i1.markdown(f"**{it.skill}** — demand {it.demand_score:.0f}% of postings")
                i2.selectbox(
                    "Status", STATUSES, key=f"roadmap_status_{it.skill}",
                    label_visibility="collapsed",
                )
    st.caption(ROADMAP_NOTE)
else:
    st.caption("Nothing to learn — your skill set covers all role requirements in the dataset.")

st.caption("Skill coverage is a curriculum-planning indicator, NOT a hiring probability. "
           "Demand statistics reflect historical postings only.")
