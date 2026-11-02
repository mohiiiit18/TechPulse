import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.theme import (apply_base_css, page_header, section_title,
                         empty_state, PALETTE)
from utils.data_loader import load_data, get_dataset_date_range
from src.salary_predictor import SalaryPredictor, LIMITATIONS

_latest = get_dataset_date_range()["latest"]
_latest_str = f"{_latest:%B %Y}" if _latest is not None else "unknown"

apply_base_css()
page_header(":material/insights:", "Salary Prediction (ML)",
            f"Estimate expected annual compensation (USD) from job role, seniority, country, "
            f"employment type, remote status and skills — historical dataset, latest records {_latest_str}. "
            "This is a machine-learning estimate, NOT an offer guarantee; see limitations below. "
            "The descriptive Salary Benchmark page remains separate.")

@st.cache_resource
def _get_predictor():
    return SalaryPredictor()

try:
    predictor = _get_predictor()
except FileNotFoundError:
    st.error("Salary model artifacts not found. Train the model first: `python src/train_salary_model.py`",
             icon=":material/error:")
    st.stop()

df = load_data()
meta = predictor.meta

# ── Inputs ───────────────────────────────────────────────────────────────────
with st.container(border=True):
    c1, c2 = st.columns(2)
    with c1:
        role = st.selectbox("Job role", predictor.known_roles)
        country = st.selectbox("Country", predictor.known_countries)
        employment = st.selectbox("Employment type", ["(unspecified)"] + predictor.known_employment_types)
    with c2:
        seniority_opts = [c.removeprefix("seniority_") for c in predictor.feature_names
                          if c.startswith("seniority_")]
        seniority = st.selectbox("Experience (parsed seniority)", seniority_opts,
                                 help="Parsed from job titles in the dataset — 'unspecified' is a real observed category.")
        is_remote = st.toggle("Remote position", value=False)
    skills = st.multiselect("Skills", options=sorted(predictor.known_skills),
                            placeholder="Type to search e.g. python, aws, kubernetes...")

    if st.button("Predict salary", icon=":material/payments:", type="primary"):
        st.session_state["sal_pred_requested"] = True

if not st.session_state.get("sal_pred_requested"):
    st.info("Fill in the job details and click **Predict salary**.", icon=":material/info:")
    st.stop()

pred, lo, hi, info = predictor.predict_salary(
    role, country, skills, is_remote=is_remote,
    employment_type=None if employment == "(unspecified)" else employment,
    seniority=seniority)

# ── Prediction display ───────────────────────────────────────────────────────
st.space("small")
m1, m2, m3 = st.columns(3)
m1.metric("Predicted salary", f"${pred:,.0f}", border=True)
m2.metric("Estimated range (10–90%)", f"${lo:,.0f} – ${hi:,.0f}", border=True)
m3.metric("Model", info["model"], border=True)
st.caption(f"Currency: USD · Period: {info['period']} · {info['range_note']}")

# ── Market comparison (descriptive, from the same dataset) ──────────────────
st.divider()
section_title("Market comparison", ":material/query_stats:")
st.caption("Descriptive statistics of disclosed USD salaries in the dataset.")
seg = df[(df["salary_norm_currency"] == "USD") & (df["job_role"] == role)]
if seg.empty:
    st.caption("No disclosed USD salaries for this role in the dataset.")
else:
    q = seg["salary_norm_avg"].quantile([0.25, 0.5, 0.75])
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Postings", len(seg), border=True)
    mc2.metric("P25", f"${q[0.25]:,.0f}", border=True)
    mc3.metric("Median", f"${q[0.5]:,.0f}", border=True)
    mc4.metric("P75", f"${q[0.75]:,.0f}", border=True)
    delta = pred - q[0.5]
    st.metric("Prediction vs role median", f"{'+' if delta >= 0 else ''}${delta:,.0f}",
              delta=f"{delta / q[0.5] * 100:+.0f}% vs median", border=True)

    # Prediction vs dataset distribution (histogram with the prediction marked)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=seg["salary_norm_avg"], nbinsx=30,
        marker_color=PALETTE[0],
        marker_line=dict(color="#3A2020", width=1),
        name="Disclosed salaries",
        hovertemplate="${x:,.0f}: %{y} records<extra></extra>",
    ))
    fig.add_vline(x=pred, line_color="#FFD76A", line_width=3,
                  annotation_text=f"ML prediction ${pred:,.0f}",
                  annotation_font=dict(color="#FFD76A", size=12),
                  annotation_position="top")
    fig.update_layout(
        xaxis_title="Annual salary (USD)", yaxis_title="Number of records",
        height=300, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False},
                    key="salary_pred_dist_chart")

# ── Limitations ──────────────────────────────────────────────────────────────
st.divider()
with st.expander("Model limitations", icon=":material/warning:"):
    for lim in LIMITATIONS:
        st.caption(f"• {lim}")
    st.caption(f"Trained {meta['training_date'][:10]} on {meta['n_train']} postings; "
               f"test MAE ${meta['metrics'][meta['selected']]['test_mae']:,.0f}, "
               f"R² {meta['metrics'][meta['selected']]['test_r2']:.2f}.")
