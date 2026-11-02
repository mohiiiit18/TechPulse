import streamlit as st
import pandas as pd
from utils.theme import apply_base_css, page_header, empty_state
from utils.data_loader import load_data, get_dataset_date_range

df = load_data()
_latest = get_dataset_date_range()["latest"]
_latest_str = f"{_latest:%B %Y}" if _latest is not None else "unknown"

apply_base_css()
page_header(":material/manage_search:", "Job Explorer",
            f"Browse, filter, and explore 1,700+ tech job records — historical dataset, latest records from {_latest_str}.")

# Search & Filters
with st.container(border=True):
    col_search, col_role = st.columns([3, 2])
    
    with col_search:
        search_query = st.text_input(
            "Search jobs",
            placeholder="Search by title, company, or skill (e.g. AI Engineer, Python, Airwallex, AWS)...",
            icon=":material/search:",
            label_visibility="collapsed"
        )
        
    with col_role:
        roles = ["All roles"] + sorted(df["job_role"].dropna().unique().tolist())
        selected_role = st.selectbox("Role", roles, label_visibility="collapsed")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        remote_filter = st.segmented_control(
            "Work style",
            ["All", "Remote", "On-site / Hybrid"],
            default="All"
        )
    with col_f2:
        countries = ["All countries"] + sorted([c for c in df["country"].dropna().unique() if c != "Unknown"])
        selected_country = st.selectbox("Country", countries)
    with col_f3:
        only_salary = st.toggle("Disclosed compensation only", value=False)

# Filter logic
view_df = df.copy()

if search_query:
    q = search_query.lower().strip()
    # BUG 4 FIX: Added regex=False to prevent crash on special characters like C++, (AWS), [React]
    view_df = view_df[
        view_df["title"].str.lower().str.contains(q, na=False, regex=False) |
        view_df["company_name"].str.lower().str.contains(q, na=False, regex=False) |
        view_df["skills"].str.lower().str.contains(q, na=False, regex=False)
    ]

if selected_role != "All roles":
    view_df = view_df[view_df["job_role"] == selected_role]

if remote_filter and remote_filter != "All":
    view_df = view_df[view_df["remote_type"] == remote_filter]

if selected_country != "All countries":
    view_df = view_df[view_df["country"] == selected_country]

if only_salary:
    view_df = view_df[view_df["annual_salary_avg"].notna() & (view_df["annual_salary_avg"] > 0)]

# Summary count
with st.container(horizontal=True):
    st.badge(f"Displaying {len(view_df):,} job records", icon=":material/work:", color="red")
    if only_salary:
        st.badge("Filter: Disclosed salary only", icon=":material/payments:", color="green")
    if remote_filter == "Remote":
        st.badge("Filter: Remote only", icon=":material/home_work:", color="red")

st.space("small")

# Helper to format salary display per record (original value + currency + period)
def format_salary_display(row):
    """Returns a formatted salary string using original values when USD annualized is unavailable."""
    if pd.notna(row.get("annual_salary_avg")) and row["annual_salary_avg"] > 0:
        return f"${int(row['annual_salary_avg']):,}"
    elif (pd.notna(row.get("salary_display_min")) and
          pd.notna(row.get("salary_display_currency")) and
          pd.notna(row.get("salary_display_period"))):
        curr = row["salary_display_currency"]
        lo = int(row["salary_display_min"])
        hi = int(row["salary_display_max"]) if pd.notna(row.get("salary_display_max")) else lo
        period = row["salary_display_period"]
        return f"{curr} {lo:,}–{hi:,}/{period}"
    elif pd.notna(row.get("salary_min")) and pd.notna(row.get("salary_currency")):
        curr = row["salary_currency"]
        lo = int(row["salary_min"])
        hi = int(row["salary_max"]) if pd.notna(row.get("salary_max")) else lo
        period = row.get("salary_period", "")
        return f"{curr} {lo:,}–{hi:,}/{period}" if period else f"{curr} {lo:,}–{hi:,}"
    return None

# Data table with column configuration
if view_df.empty:
    empty_state(":material/search_off:", "No matching records",
                "No matching job records found. Try broadening your search or resetting filters.")
else:
    # Prepare display columns
    display_cols = [
        "title",
        "company_name",
        "job_role",
        "remote_type",
        "country",
        "annual_salary_avg",
        "skills",
        "url"
    ]
    
    # Format skills for cleaner viewing
    table_df = view_df[display_cols].copy()
    table_df["title"] = table_df["title"].str.title()
    table_df["company_name"] = table_df["company_name"].str.title()
    table_df["skills"] = table_df["skills"].str.replace("|", ", ")
    
    st.dataframe(
        table_df,
        column_config={
            "title": st.column_config.TextColumn("Role Title", width="medium"),
            "company_name": st.column_config.TextColumn("Company", width="small"),
            "job_role": st.column_config.TextColumn("Domain Role", width="small"),
            "remote_type": st.column_config.TextColumn("Work Style", width="small"),
            "country": st.column_config.TextColumn("Location", width="small"),
            # PHASE 5 FIX: Annotate that column is USD-annualized only
            "annual_salary_avg": st.column_config.NumberColumn(
                "Est. Salary (USD annualized only)",
                format="$%d",
                width="small"
            ),
            "skills": st.column_config.TextColumn("Key Skills", width="large"),
            "url": st.column_config.LinkColumn("Action", display_text="Apply \u2197", width="small")
        },
        hide_index=True,
        alt="Directory of tech job records matching search and filter parameters"
    )

    # Job Detail Viewer
    st.space("small")
    with st.expander("Inspect a job in detail", icon=":material/visibility:"):
        job_titles = [f"{row['title']} @ {row['company_name']} ({row['country']})" for _, row in table_df.head(50).iterrows()]
        if job_titles:
            picked_job_idx = st.selectbox("Select job to view full specifications", range(len(job_titles)), format_func=lambda i: job_titles[i])
            chosen_row = view_df.iloc[picked_job_idx]
            
            c_left, c_right = st.columns([2, 1])
            with c_left:
                st.markdown(f"### {chosen_row['title'].title()}")
                st.caption(f"**Company:** {chosen_row['company_name'].title()} • **Location:** {chosen_row['location_raw']}")
                
                st.markdown("**Required technical competencies:**")
                badges_str = " ".join([f":red-badge[{s}]" for s in chosen_row["parsed_skills"]])
                st.markdown(badges_str if badges_str else "Not explicitly specified")
                
            with c_right:
                with st.container(border=True):
                    st.metric("Work arrangement", chosen_row["remote_type"])
                    # PHASE 2 FIX: Show original salary when USD annualized unavailable
                    salary_display = format_salary_display(chosen_row)
                    if salary_display:
                        if pd.notna(chosen_row.get("annual_salary_avg")) and chosen_row["annual_salary_avg"] > 0:
                            st.metric("Est. Annual Salary (USD)", salary_display)
                        else:
                            st.metric("Salary (original currency)", salary_display)
                    if pd.notna(chosen_row["url"]) and str(chosen_row["url"]).startswith("http"):
                        st.link_button("Go to application page", str(chosen_row["url"]), icon=":material/open_in_new:")
