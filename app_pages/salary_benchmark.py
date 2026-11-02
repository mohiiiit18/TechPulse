import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.theme import (apply_base_css, page_header, section_title,
                         empty_state, PALETTE, GOLD)
from utils.data_loader import load_data, get_dataset_date_range

df = load_data()
_latest = get_dataset_date_range()["latest"]
_latest_str = f"{_latest:%B %Y}" if _latest is not None else "unknown"

apply_base_css()
page_header(":material/payments:", "Salary Benchmark",
            "Comprehensive compensation insights, percentile breakdowns, and earnings estimator based on disclosed salary data.")

# PHASE 2 FIX: Prominent USD-only caption + dataset date
st.info(
    "**All salary figures are in USD only.** Non-USD salaries are excluded from these benchmarks. "
    f"Historical dataset, latest records from {_latest_str}.",
    icon=":material/info:"
)

# Clean salary subset (annualized USD equivalent)
salary_df = df[df["annual_salary_avg"].notna() & (df["annual_salary_avg"] >= 20000)].copy()

if salary_df.empty:
    st.info("No salary disclosures available.")
    st.stop()

# Overall KPI Metrics
median_salary = salary_df["annual_salary_avg"].median()
remote_median = salary_df[salary_df["remote_type"] == "Remote"]["annual_salary_avg"].median()
onsite_median = salary_df[salary_df["remote_type"] == "On-site / Hybrid"]["annual_salary_avg"].median()
p90_salary = salary_df["annual_salary_avg"].quantile(0.90)

with st.container(horizontal=True):
    st.metric(
        label="Median annual salary",
        value=f"${int(median_salary):,}",
        border=True
    )
    # BUG 6 FIX: Guard against NaN median when a group is empty
    if pd.notna(remote_median) and pd.notna(onsite_median):
        remote_delta = f"+${int(remote_median - onsite_median):,} vs On-site" if remote_median > onsite_median else f"-${int(onsite_median - remote_median):,} vs On-site"
        remote_value = f"${int(remote_median):,}"
    elif pd.notna(remote_median):
        remote_delta = None
        remote_value = f"${int(remote_median):,}"
    else:
        remote_delta = None
        remote_value = "N/A"

    st.metric(
        label="Remote median salary",
        value=remote_value,
        delta=remote_delta,
        border=True
    )
    # PHASE 5 FIX: "Top 10% staff/lead tier" → "Top decile (P90)"
    st.metric(
        label="Top decile (P90)",
        value=f"${int(p90_salary):,}",
        border=True
    )
    st.metric(
        label="Disclosed compensation records",
        value=f"{len(salary_df):,}",
        border=True
    )

st.space("small")

# Interactive Salary Estimator
with st.container(border=True):
    st.markdown("**Personalized compensation estimator**")
    # PHASE 5 FIX: Removed "experience tier" from caption
    st.caption("Estimate descriptive salary percentiles for a role and work style, based on disclosed USD salary data.")
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        target_role = st.selectbox(
            "Target role",
            sorted(salary_df["job_role"].unique())
        )
    with col_e2:
        target_work_style = st.segmented_control(
            "Target work style",
            ["All", "Remote", "On-site / Hybrid"],
            default="All"
        )
        
    est_subset = salary_df[salary_df["job_role"] == target_role]
    if target_work_style != "All":
        est_subset = est_subset[est_subset["remote_type"] == target_work_style]
        
    if not est_subset.empty:
        p25 = est_subset["annual_salary_avg"].quantile(0.25)
        p50 = est_subset["annual_salary_avg"].median()
        p75 = est_subset["annual_salary_avg"].quantile(0.75)
        p90 = est_subset["annual_salary_avg"].quantile(0.90)
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        # PHASE 5 FIX: Updated percentile metric labels
        col_m1.metric("P25 (lower quartile)", f"${int(p25):,}", border=True)
        col_m2.metric("Median (P50)", f"${int(p50):,}", border=True)
        col_m3.metric("P75 (upper quartile)", f"${int(p75):,}", border=True)
        col_m4.metric("P90 (top decile)", f"${int(p90):,}", border=True)
        
        # PHASE 5 FIX: Add descriptive statistics disclaimer
        st.caption("These are descriptive statistics from the dataset, not ML predictions or experience-level thresholds.")
    else:
        st.info("Insufficient data for this specific role and work style combination.")

st.space("small")

# Salary Charts Section
col_chart1, col_chart2 = st.columns(2, gap="medium")

with col_chart1:
    with st.container(border=True):
        st.markdown("**Salary distribution by role**")
        st.caption("Comparing median, 25th, and 75th percentiles (USD):")
        
        stats_list = []
        for role, grp in salary_df.groupby("job_role"):
            stats_list.append({
                "Role": role,
                "Lower (25%)": grp["annual_salary_avg"].quantile(0.25),
                "Median": grp["annual_salary_avg"].median(),
                "Upper (75%)": grp["annual_salary_avg"].quantile(0.75)
            })
        stats_df = pd.DataFrame(stats_list).sort_values("Median")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=stats_df["Upper (75%)"] - stats_df["Lower (25%)"],
            y=stats_df["Role"],
            orientation="h", base=stats_df["Lower (25%)"],
            marker_color="rgba(227,38,54,0.35)",
            marker_line=dict(color="#E32636", width=1.5),
            name="P25–P75 range",
            customdata=stats_df["Median"],
            hovertemplate="%{y}: P25 $%{base:,.0f} – P75 $%{x:,.0f}"
                          "<br>Median $%{customdata:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=stats_df["Median"], y=stats_df["Role"],
            mode="markers", marker_color=GOLD, marker_size=12,
            marker_line=dict(color="#FFD76A", width=1.5),
            name="Median",
            hovertemplate="%{y}: median $%{x:,.0f}<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title="Annual salary (USD)",
            yaxis_title=None,
            height=320,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False},
                        key="salary_range_chart")

with col_chart2:
    with st.container(border=True):
        st.markdown("**Remote vs on-site compensation comparison**")
        st.caption("Median annual salary across job roles and working styles:")

        comp_work = (
            salary_df.groupby(["job_role", "remote_type"])["annual_salary_avg"]
            .median().reset_index()
            .rename(columns={"job_role": "Role", "remote_type": "Work Style",
                             "annual_salary_avg": "Median Salary"})
        )

        fig = go.Figure()
        for i, style in enumerate(comp_work["Work Style"].unique()):
            sub = comp_work[comp_work["Work Style"] == style]
            fig.add_trace(go.Bar(
                x=sub["Median Salary"], y=sub["Role"],
                orientation="h", name=style,
                marker_color=PALETTE[i % len(PALETTE)],
                hovertemplate="%{y} (%{fullData.name}): median $%{x:,.0f}<extra></extra>",
            ))
        fig.update_layout(
            barmode="group",
            xaxis_title="Median annual salary (USD)",
            yaxis_title=None,
            height=320,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False},
                        key="salary_remote_chart")
