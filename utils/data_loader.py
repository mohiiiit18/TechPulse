from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed dataset.csv"
MODELS_DIR = BASE_DIR / "models"

# Hourly → annual assumption (PHASE 2, documented): ~2,000 paid work hours per year.
# Monthly → annual uses ×12. These are conversion assumptions, not exchange rates.
ANNUALIZATION_FACTORS = {"year": 1.0, "hour": 2000.0, "month": 12.0}

@st.cache_data(ttl="1h")
def get_dataset_date_range() -> dict:
    """Latest collection date from the dataset, used to date-stamp every page
    (PHASE 5: replaces hardcoded 'collected up to August 2026' captions)."""
    df = pd.read_csv(DATA_PATH, usecols=["posted_at", "scraped_at"], low_memory=False)
    dates = pd.concat([
        pd.to_datetime(df.get("posted_at"), errors="coerce", utc=True),
        pd.to_datetime(df.get("scraped_at"), errors="coerce", utc=True),
    ]).dropna()
    if dates.empty:
        return {"latest": None}
    return {"latest": dates.max()}

MODEL_PATH = BASE_DIR / "models" / "job_role_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "skill_encoder.pkl"

# Skill category groupings for easy discovery
SKILL_CATEGORIES = {
    "AI & Data Science": [
        "python", "machine-learning", "deep-learning", "llm", "pytorch",
        "tensorflow", "scikit-learn", "nlp", "computer-vision", "data-science",
        "statistics", "numpy", "pandas", "data-engineering"
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "ci-cd", "devops", "linux", "ansible", "serverless", "security"
    ],
    "Data & Databases": [
        "sql", "nosql", "redis", "elasticsearch", "bigquery", "snowflake",
        "databricks", "dbt", "spark", "hadoop", "airflow", "kafka",
        "redshift", "etl", "power-bi", "tableau", "looker"
    ],
    "Web & Backend": [
        "javascript", "typescript", "react", "next.js", "vue", "angular",
        "node.js", "html-css", "django", "flask", "fastapi", "spring",
        "java", "c#", "c++", "go", "rust", "ruby", "php", "scala",
        "rest-api", "graphql", "grpc", "microservices"
    ],
    "Mobile & Others": [
        "android", "ios", "flutter", "react-native", "kotlin", "swift",
        "unity", "robotics", "embedded", "hardware", "gis", "blockchain",
        "ui-ux", "figma", "product-management", "project-management",
        "communication", "salesforce", "crm", "qa-testing"
    ]
}

ROLE_PRESETS = {
    "AI & Machine Learning": [
        "python", "machine-learning", "pytorch", "llm", "deep-learning", "sql"
    ],
    "Full-Stack Web Dev": [
        "javascript", "typescript", "react", "node.js", "html-css", "sql"
    ],
    "Cloud & DevOps / SRE": [
        "kubernetes", "docker", "aws", "terraform", "ci-cd", "linux"
    ],
    "Data Platform Engineer": [
        "python", "sql", "airflow", "spark", "snowflake", "data-engineering"
    ],
    "Product & Tech Leader": [
        "product-management", "project-management", "communication", "ui-ux"
    ]
}

@st.cache_data(ttl="1h")
def load_data() -> pd.DataFrame:
    """Load the app dataset.

    Prefers the Phase-2 cleaned dataset (data/processed/techpulse_cleaned.csv),
    which adds standardized skills, seniority, freshness and normalized salary
    columns, and preserves all original values. Falls back to the original
    snapshot and derives legacy columns on the fly if the cleaned file is
    missing, so older deployments keep working.
    """
    cleaned_path = BASE_DIR / "data" / "processed" / "techpulse_cleaned.csv"
    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path, low_memory=False)
        df = df.drop_duplicates(subset=["job_id"]).copy()

        # Legacy alias columns expected by existing pages/models.
        if "parsed_skills" not in df.columns:
            df["parsed_skills"] = df["parsed_skills_std"]
        # parsed_skills_std is a stringified Python list after CSV round-trip;
        # restore real lists with the Phase-3 safe parser (never fabricates).
        try:
            from src.preprocessing import parse_skill_list as _parse_list
        except ImportError:  # pragma: no cover - direct-script fallback
            from preprocessing import parse_skill_list as _parse_list
        if df["parsed_skills"].map(lambda v: isinstance(v, str)).any():
            df["parsed_skills"] = df["parsed_skills"].apply(_parse_list)

        # posted_date was serialized as a string in CSV; restore datetime.
        if "posted_date" in df.columns:
            df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce", utc=True)
        elif "posted_at" in df.columns:
            df["posted_date"] = pd.to_datetime(df["posted_at"], errors="coerce", utc=True)

        # Text columns may contain NaN after CSV round-trip (e.g. salary cols).
        for col in ["company_name", "title", "department", "location_raw", "country", "city"]:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str).str.strip()

        # Remote status formatting (same rule as legacy pipeline; preserves Unknown)
        if "is_remote" in df.columns:
            df["remote_type"] = df["is_remote"].map(
                {True: "Remote", False: "On-site / Hybrid"}
            ).fillna("Unknown")
        else:
            df["remote_type"] = "Unknown"

        # Annualized USD columns used by pages/recommendation ranking.
        if "annual_salary_min" not in df.columns:
            df["annual_salary_min"] = df.get("salary_norm_min", pd.Series(np.nan, index=df.index))
        if "annual_salary_max" not in df.columns:
            df["annual_salary_max"] = df.get("salary_norm_max", pd.Series(np.nan, index=df.index))

        # Explicit fallbacks for salary display if cleaned columns ever absent.
        if "salary_display_min" not in df.columns:
            df["salary_display_min"] = df.get("salary_min", pd.Series(np.nan, index=df.index))
            df["salary_display_max"] = df.get("salary_max", pd.Series(np.nan, index=df.index))
            df["salary_display_currency"] = df.get("salary_currency", pd.Series(np.nan, index=df.index))
            df["salary_display_period"] = df.get("salary_period", pd.Series(np.nan, index=df.index))

        if "annual_salary_avg" not in df.columns:
            df["annual_salary_avg"] = df.get("salary_norm_avg", np.nan)

        return df

    # ── Fallback: original pipeline on the raw snapshot ──
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.drop_duplicates(subset=["job_id"]).copy()

    # Clean text columns
    for col in ["company_name", "title", "department", "location_raw", "country", "city"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    # Parse skills
    def parse_skill_list(s):
        if pd.isna(s) or s == "nan":
            return []
        return [x.strip().lower() for x in str(s).split("|") if x.strip()]

    df["parsed_skills"] = df["skills"].apply(parse_skill_list)

    # Remote status formatting
    # BUG 7: is_remote has only True/False values; .fillna covers future data
    if "is_remote" in df.columns:
        df["remote_type"] = df["is_remote"].map({
            True: "Remote",
            False: "On-site / Hybrid"
        }).fillna("Unknown")
    else:
        df["remote_type"] = "Unknown"

    # Clean posted date
    if "posted_at" in df.columns:
        df["posted_date"] = pd.to_datetime(df["posted_at"], errors="coerce")
    else:
        df["posted_date"] = pd.NaT

    # -------------------------------------------------------------------------
    # PHASE 2 FIX (BUG 8-12):
    # annual_salary_avg = USD-only annualized salary (used for benchmarks/charts)
    # salary_display_* = original salary values from dataset (any currency)
    # -------------------------------------------------------------------------

    # Numeric coercion guards against malformed salary values (PHASE 2 #9)
    # Done BEFORE copying to salary_display_* so display columns are numeric too
    for _c in ("salary_min", "salary_max"):
        if _c in df.columns:
            df[_c] = pd.to_numeric(df[_c], errors="coerce")

    # Preserve original salary columns for display (any currency/period)
    # These are used in career_matcher.py and job_explorer.py for display purposes
    df["salary_display_min"] = df["salary_min"].copy() if "salary_min" in df.columns else np.nan
    df["salary_display_max"] = df["salary_max"].copy() if "salary_max" in df.columns else np.nan
    df["salary_display_currency"] = df["salary_currency"].copy() if "salary_currency" in df.columns else np.nan
    df["salary_display_period"] = df["salary_period"].copy() if "salary_period" in df.columns else np.nan

    # Compute USD-only annualized salary (used for salary_benchmark.py and match ranking)
    df["annual_salary_min"] = np.nan
    df["annual_salary_max"] = np.nan

    usd_mask = (df["salary_currency"] == "USD") if "salary_currency" in df.columns else pd.Series(False, index=df.index)

    # Annualize USD salaries per documented ANNUALIZATION_FACTORS above.
    for period, factor in ANNUALIZATION_FACTORS.items():
        pmask = (df["salary_period"] == period) if "salary_period" in df.columns else pd.Series(False, index=df.index)
        sel = usd_mask & pmask
        if factor != 1.0:
            df.loc[sel, "annual_salary_min"] = df.loc[sel, "salary_min"] * factor
            df.loc[sel, "annual_salary_max"] = df.loc[sel, "salary_max"] * factor
        else:
            df.loc[sel, "annual_salary_min"] = df.loc[sel, "salary_min"]
            df.loc[sel, "annual_salary_max"] = df.loc[sel, "salary_max"]

    # Average annual salary midpoint (USD only — non-USD rows remain NaN)
    df["annual_salary_avg"] = (df["annual_salary_min"] + df["annual_salary_max"]) / 2

    return df

@st.cache_resource
def load_model():
    """Load the production ML model + encoder.

    Preference order (all validated before promotion, never overwrite):
      1. v3 — Phase-3 pipeline (models/job_role_model_v3.pkl, leakage-aware
         stratified-group split, macro-F1 selected, sklearn-matched pickle).
      2. v2 — Phase-1 retrained RandomForest.
      3. Original LogisticRegression pickle.
    """
    candidates = [
        (MODELS_DIR / "job_role_model_v3.pkl", MODELS_DIR / "skill_encoder_v3.pkl"),
        (MODELS_DIR / "job_role_model_v2.pkl", MODELS_DIR / "skill_encoder_v2.pkl"),
        (MODEL_PATH, ENCODER_PATH),
    ]
    for mp, ep in candidates:
        if mp.exists() and ep.exists():
            return joblib.load(mp), joblib.load(ep)
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)

def predict_roles(selected_skills: list, model, encoder):
    """Predict role(s) from a user skill list.

    Handles: empty input (returns None), unknown skills (silently ignored by
    the encoder transform; reported back), and missing predict_proba.
    Returns (predicted_role|None, results_df, unknown_skills:set).
    """
    if not selected_skills:
        return None, pd.DataFrame(columns=["Role", "Confidence (%)"]), set()

    user_set = {s.strip().lower() for s in selected_skills}
    known = {str(c).lower() for c in getattr(encoder, "classes_", [])}
    unknown_skills = user_set - known

    try:
        # SkillEncoder.transform returns a (X, unknown) tuple; plain MLB returns X.
        result = encoder.transform([list(user_set)])
        features = result[0] if isinstance(result, tuple) else result
    except Exception:
        # Some encoders raise on unknown labels — filter to known first.
        filtered = [s for s in user_set if s in known]
        result = encoder.transform([filtered]) if filtered else encoder.transform([[]])
        features = result[0] if isinstance(result, tuple) else result

    predicted_role = model.predict(features)[0]
    results = pd.DataFrame(columns=["Role", "Confidence (%)"], dtype=object)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        top_indices = np.argsort(probabilities)[::-1]
        results = pd.DataFrame({
            "Role": model.classes_[top_indices],
            "Confidence (%)": (probabilities[top_indices] * 100).round(1),
        })
    else:
        results = pd.DataFrame({"Role": [predicted_role], "Confidence (%)": [100.0]})

    return predicted_role, results, unknown_skills

def get_skill_gap_analysis(predicted_role: str, user_skills: list, df: pd.DataFrame, top_n: int = 8):
    role_df = df[df["job_role"] == predicted_role]
    if role_df.empty:
        return [], []
    
    all_role_skills = (
        role_df["parsed_skills"].explode().dropna()
        .value_counts()
    )
    
    user_skills_set = set(s.lower() for s in user_skills)
    
    # Skills the user has that match the role
    matched_skills = [
        {"skill": skill, "count": int(count)}
        for skill, count in all_role_skills.items()
        if skill in user_skills_set
    ]
    
    # High-impact missing skills
    missing_skills = [
        {
            "skill": skill,
            "count": int(count),
            "demand_pct": round((count / len(role_df)) * 100, 1)
        }
        for skill, count in all_role_skills.items()
        if skill not in user_skills_set
    ][:top_n]
    
    return matched_skills, missing_skills

def get_matched_jobs(predicted_role: str, user_skills: list, df: pd.DataFrame, limit: int = 6):
    """
    Rank jobs by a composite score:
    - 60%: Skill coverage % (matched skills / job's total skills)
    - 30%: Recency (normalized days since posting, more recent = higher)
    - 10%: Has salary disclosure

    Only returns jobs with at least 1 matching skill.
    Zero-skill matches are excluded entirely.
    """
    role_df = df[df["job_role"] == predicted_role].copy()
    if role_df.empty or not user_skills:
        return pd.DataFrame()

    user_skills_set = set(s.lower() for s in user_skills)

    def skill_coverage(skills_list):
        if not skills_list:
            return 0.0
        matched = len(user_skills_set.intersection(set(skills_list)))
        return matched / len(skills_list)  # coverage %

    def matched_count(skills_list):
        return len(user_skills_set.intersection(set(skills_list))) if skills_list else 0

    role_df["skill_match_count"] = role_df["parsed_skills"].apply(matched_count)
    role_df["skill_coverage"] = role_df["parsed_skills"].apply(skill_coverage)

    # Filter out zero-match jobs
    role_df = role_df[role_df["skill_match_count"] > 0]
    if role_df.empty:
        return pd.DataFrame()

    # Recency score (0-1): more recent = higher
    if "posted_date" in role_df.columns and role_df["posted_date"].notna().any():
        max_date = role_df["posted_date"].max()
        min_date = role_df["posted_date"].min()
        date_range = (max_date - min_date).total_seconds()
        if date_range > 0:
            role_df["recency_score"] = role_df["posted_date"].apply(
                lambda d: (d - min_date).total_seconds() / date_range if pd.notna(d) else 0.0
            )
        else:
            role_df["recency_score"] = 1.0
    else:
        role_df["recency_score"] = 0.0

    # Salary disclosure bonus
    role_df["salary_score"] = role_df["annual_salary_avg"].apply(
        lambda x: 1.0 if pd.notna(x) and x > 0 else 0.0
    )

    # Composite score
    role_df["match_score"] = (
        0.60 * role_df["skill_coverage"] +
        0.30 * role_df["recency_score"] +
        0.10 * role_df["salary_score"]
    )

    return role_df.sort_values("match_score", ascending=False).head(limit)
