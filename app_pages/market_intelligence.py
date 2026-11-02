import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.theme import (apply_base_css, page_header, section_title,
                         empty_state, PALETTE)
from utils.data_loader import load_data, get_dataset_date_range

df = load_data()
_latest = get_dataset_date_range()["latest"]
_latest_str = f"{_latest:%B %Y}" if _latest is not None else "unknown"

apply_base_css()
page_header(":material/analytics:", "Market Intelligence",
            f"Aggregate insights from historical tech job posting data. Latest records from {_latest_str}.")

# ── Performance (Phase 10): cache base-frame copies + full-dataset skill counts
@st.cache_data(ttl="1h")
def _base_counts(_df_hash: int, skills_col: str) -> pd.Series:
    """Cached skill value counts over the full dataset (filter-independent).
    _df_hash only busts the cache when the underlying dataset changes."""
    return df[skills_col].explode().dropna().value_counts()

_FULL_SKILL_COUNTS = _base_counts(len(df), "parsed_skills")

# Sidebar / Filter Panel
with st.container(border=True):
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        all_roles = ["All roles"] + sorted(df["job_role"].dropna().unique().tolist())
        selected_role = st.selectbox("Filter by role", all_roles)
        
    with col_f2:
        remote_options = ["All work styles", "Remote", "On-site / Hybrid"]
        selected_remote = st.segmented_control("Work style", remote_options, default="All work styles")
        
    with col_f3:
        all_countries = ["All countries"] + sorted([c for c in df["country"].dropna().unique() if c != "Unknown"])
        selected_country = st.selectbox("Filter by country", all_countries)

# Apply filters
filtered_df = df.copy()

if selected_role != "All roles":
    filtered_df = filtered_df[filtered_df["job_role"] == selected_role]

if selected_remote and selected_remote != "All work styles":
    filtered_df = filtered_df[filtered_df["remote_type"] == selected_remote]

if selected_country != "All countries":
    filtered_df = filtered_df[filtered_df["country"] == selected_country]

if filtered_df.empty:
    st.warning("No job postings match the selected filters. Please adjust your criteria.", icon=":material/filter_alt_off:")
    st.stop()

# Responsive KPI Row
total_jobs = len(filtered_df)
unique_companies = filtered_df["company_name"].nunique()
remote_pct = round((filtered_df["remote_type"] == "Remote").mean() * 100, 1)

# Top skill computation
all_skills = filtered_df["parsed_skills"].explode().dropna()
top_skill = all_skills.mode()[0] if not all_skills.empty else "N/A"

with st.container(horizontal=True):
    # PHASE 5 FIX: "Active job postings" → "Job records"
    st.metric(
        label="Job records",
        value=f"{total_jobs:,}",
        border=True
    )
    # PHASE 5 FIX: "Hiring companies" → "Companies in dataset"
    st.metric(
        label="Companies in dataset",
        value=f"{unique_companies:,}",
        border=True
    )
    # PHASE 5 FIX: "Remote flexibility rate" → "Remote jobs (%)"
    st.metric(
        label="Remote jobs (%)",
        value=f"{remote_pct}%",
        border=True
    )
    st.metric(
        label="Top in-demand skill",
        value=top_skill.title(),
        border=True
    )

st.space("small")

# Visual Analytics Section
tab_skills, tab_roles, tab_geo, tab_companies = st.tabs([
    "Skills in demand",
    "Roles & employment",
    "Geographic distribution",
    "Top hiring companies"
])

with tab_skills:
    with st.container(border=True):
        st.markdown("**Top 20 most demanded skills**")
        st.caption("Frequencies of technical skills requested across job records in the dataset:")
        
        # Use the cached full-dataset counts when no filters narrow the view
        if filtered_df is df or len(filtered_df) == len(df):
            top_n_skills = _FULL_SKILL_COUNTS.head(20).reset_index()
        else:
            top_n_skills = all_skills.value_counts().head(20).reset_index()
        top_n_skills.columns = ["Skill", "Postings"]
        plot_skills = top_n_skills.sort_values("Postings", ascending=True)

        fig = go.Figure(go.Bar(
            x=plot_skills["Postings"], y=plot_skills["Skill"],
            orientation="h", marker_color=PALETTE[0],
            hovertemplate="%{y}: %{x:,} records<extra></extra>",
        ))
        fig.update_layout(xaxis_title="Number of records", yaxis_title=None,
                          height=460)
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False},
                        key="top_skills_chart")

with tab_roles:
    col_r1, col_r2 = st.columns(2, gap="medium")
    
    with col_r1:
        with st.container(border=True):
            st.markdown("**Postings by job role**")
            role_counts = filtered_df["job_role"].value_counts().reset_index()
            role_counts.columns = ["Role", "Postings"]
            plot_roles = role_counts.sort_values("Postings", ascending=True)

            fig = go.Figure(go.Bar(
                x=plot_roles["Postings"], y=plot_roles["Role"],
                orientation="h", marker_color=PALETTE[1],
                hovertemplate="%{y}: %{x:,} records<extra></extra>",
            ))
            fig.update_layout(xaxis_title="Job records", yaxis_title=None,
                              height=300)
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False},
                            key="role_counts_chart")
            
    with col_r2:
        with st.container(border=True):
            st.markdown("**Remote vs on-site breakdown**")
            remote_counts = filtered_df["remote_type"].value_counts().reset_index()
            remote_counts.columns = ["Work Style", "Count"]
            
            fig = go.Figure(go.Pie(
                labels=remote_counts["Work Style"], values=remote_counts["Count"],
                hole=0.55, marker_colors=PALETTE[:len(remote_counts)],
                hovertemplate="%{label}: %{value:,} records (%{percent})<extra></extra>",
            ))
            fig.update_layout(height=300, showlegend=True,
                              legend=dict(orientation="h", y=-0.1))
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False},
                            key="remote_donut_chart")

with tab_geo:
    with st.container(border=True):
        st.markdown("**Top countries by job volume**")
        country_counts = (
            filtered_df[filtered_df["country"] != "Unknown"]["country"]
            .value_counts().head(15)
            .reset_index()
        )
        
        if not country_counts.empty:
            country_counts.columns = ["Country", "Postings"]
            plot_geo = country_counts.sort_values("Postings", ascending=True)

            fig = go.Figure(go.Bar(
                x=plot_geo["Postings"], y=plot_geo["Country"],
                orientation="h", marker_color=PALETTE[2],
                hovertemplate="%{y}: %{x:,} records<extra></extra>",
            ))
            fig.update_layout(xaxis_title="Job records", yaxis_title=None,
                              height=380)
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False},
                            key="geo_chart")
        else:
            st.info("Country location data not specified for these filtered postings.")

with tab_companies:
    with st.container(border=True):
        st.markdown("**Top 15 hiring organizations**")
        comp_counts = filtered_df["company_name"].value_counts().head(15).reset_index()
        comp_counts.columns = ["Company", "Openings"]
        plot_comp = comp_counts.sort_values("Openings", ascending=True)

        fig = go.Figure(go.Bar(
            x=plot_comp["Openings"], y=plot_comp["Company"],
            orientation="h", marker_color=PALETTE[0],
            hovertemplate="%{y}: %{x:,} records<extra></extra>",
        ))
        fig.update_layout(xaxis_title="Job records", yaxis_title=None,
                          height=380)
        # BUG 5 FIX: single chart call only (was called twice)
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False},
                        key="top_companies_chart")
