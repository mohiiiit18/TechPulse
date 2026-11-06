"""
TechPulse Central Authentication, Profiles, History, Saved Jobs, Chat & Feedback Module.

This is the canonical database operations module for user-specific features.
Uses SQLAlchemy 2.x parameterized queries, Psycopg 3 native types, bcrypt security,
and robust JSONB serialization for non-native types (NumPy/Pandas).
"""

import re
import json
import logging
from typing import Optional, List, Dict, Any, Tuple, Union

import bcrypt
import streamlit as st
from sqlalchemy import text

from utils.db import get_session, get_db_session, is_database_available

logger = logging.getLogger("techpulse.auth")

# Lazy schema initialization flag
_SCHEMA_INITIALIZED = False


class TechPulseJSONEncoder(json.JSONEncoder):
    """JSON Encoder that handles NumPy, Pandas, datetime, and set/tuple objects safely."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
        except ImportError:
            pass

        try:
            import pandas as pd
            if isinstance(obj, (pd.Timestamp,)):
                return obj.isoformat()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
        except ImportError:
            pass

        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        elif isinstance(obj, (set, tuple)):
            return list(obj)
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()

        return super().default(obj)


def serialize_for_jsonb(data: Any) -> str:
    """Safely serialize structured data into a valid JSON string for PostgreSQL JSONB."""
    return json.dumps(data, cls=TechPulseJSONEncoder)


def ensure_database_initialized():
    """Lazily run database migrations once per application lifecycle."""
    global _SCHEMA_INITIALIZED
    if _SCHEMA_INITIALIZED:
        return
    try:
        from database.migrate import run_migrations
        ok, msg = run_migrations(force=False)
        if ok:
            _SCHEMA_INITIALIZED = True
        else:
            logger.warning(f"Database migration notice: {msg}")
    except Exception as e:
        logger.warning(f"Database lazy initialization notice: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. USER AUTHENTICATION & ACCESS CONTROL
# ─────────────────────────────────────────────────────────────────────────────

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def validate_email(email: str) -> bool:
    """Validate email format."""
    if not email or not isinstance(email, str):
        return False
    return bool(EMAIL_REGEX.match(email.strip()))


def register_user(name: str, email: str, password: str) -> Tuple[bool, str]:
    """
    Register a new user in the database.
    
    Validates input, normalizes email, hashes password with bcrypt,
    and prevents duplicate registrations.
    """
    ensure_database_initialized()

    name = (name or "").strip()
    email = (email or "").strip().lower()
    password = password or ""

    if not name:
        return False, "Please provide your full name."
    if not validate_email(email):
        return False, "Please enter a valid email address."
    if len(password) < 8 or password.isspace():
        return False, "Password must be at least 8 characters long and not contain only spaces."

    # Hash password securely
    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(rounds=12)
    ).decode("utf-8")

    try:
        with get_db_session() as db:
            # Check for existing user
            existing = db.execute(
                text("SELECT id FROM users WHERE email = :email"),
                {"email": email}
            ).first()
            if existing:
                return False, "Registration failed: An account with this email already exists."

            db.execute(
                text("""
                    INSERT INTO users (name, email, password_hash)
                    VALUES (:name, :email, :password_hash)
                """),
                {
                    "name": name,
                    "email": email,
                    "password_hash": password_hash
                }
            )
            return True, "Account created successfully! You can now log in."

    except Exception as e:
        err_msg = str(e)
        logger.error(f"Registration exception: {err_msg}")
        if "unique" in err_msg.lower() or "duplicate" in err_msg.lower():
            return False, "Registration failed: An account with this email already exists."
        return False, "Registration failed due to a database error. Please try again later."


def login_user(email: str, password: str) -> bool:
    """
    Authenticate a user by email and password.
    
    On success, stores user details in st.session_state["user"].
    """
    ensure_database_initialized()

    email = (email or "").strip().lower()
    password = password or ""

    if not email or not password:
        return False

    try:
        with get_db_session() as db:
            user = db.execute(
                text("SELECT id, name, email, password_hash FROM users WHERE email = :email"),
                {"email": email}
            ).mappings().first()

        if not user:
            return False

        if not bcrypt.checkpw(
            password.encode("utf-8"),
            user["password_hash"].encode("utf-8")
        ):
            return False

        # Session state assignment
        st.session_state["user"] = {
            "id": int(user["id"]),
            "name": str(user["name"]),
            "email": str(user["email"])
        }
        return True

    except Exception as e:
        logger.error(f"Login error: {e}")
        return False


def logout_user():
    """Clear authenticated session from Streamlit session state."""
    st.session_state.pop("user", None)
    st.session_state.pop("_profile_loaded_for_user", None)


def is_authenticated() -> bool:
    """Check if the current session has an authenticated user."""
    user = st.session_state.get("user")
    return bool(user and isinstance(user, dict) and user.get("id"))


def get_current_user() -> Optional[Dict[str, Any]]:
    """Return the authenticated user dictionary, or None if anonymous."""
    if is_authenticated():
        return st.session_state["user"]
    return None


def get_current_user_id() -> Optional[int]:
    """Return the authenticated user's ID, or None if anonymous."""
    user = get_current_user()
    return user["id"] if user else None


# ─────────────────────────────────────────────────────────────────────────────
# 2. USER PROFILE INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def save_profile(
    user_id: int,
    skills: Union[List[str], str],
    target_role: Optional[str] = None,
    location: Optional[str] = None,
    experience: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Save or update the user's career profile (UPSERT).
    
    Prevents overwriting existing profile data with empty values.
    """
    ensure_database_initialized()

    # Verify authorization
    current_uid = get_current_user_id()
    if current_uid is not None and user_id != current_uid:
        return False, "Unauthorized: Cannot modify another user's profile."

    # Parse and clean skills into a Python list for PostgreSQL TEXT[]
    if isinstance(skills, (list, tuple, set)):
        clean_skills = [str(s).strip() for s in skills if str(s).strip()]
    elif isinstance(skills, str):
        try:
            parsed = json.loads(skills)
            if isinstance(parsed, list):
                clean_skills = [str(s).strip() for s in parsed if str(s).strip()]
            else:
                clean_skills = [skills.strip()]
        except Exception:
            clean_skills = [s.strip() for s in skills.split(",") if s.strip()]
    else:
        clean_skills = []

    target_role = (target_role or "").strip() or None
    location = (location or "").strip() or None
    experience = (experience or "").strip() or None

    try:
        with get_db_session() as db:
            # Check existing profile to avoid destructive blank overwrites
            existing = db.execute(
                text("SELECT skills, target_role, preferred_location, experience_level FROM user_profiles WHERE user_id = :user_id"),
                {"user_id": user_id}
            ).mappings().first()

            if existing:
                # Merge existing fields if omitted in new input
                if not clean_skills and existing["skills"]:
                    clean_skills = existing["skills"]
                if not target_role:
                    target_role = existing["target_role"]
                if not location:
                    location = existing["preferred_location"]
                if not experience:
                    experience = existing["experience_level"]

            is_postgres = "postgres" in db.bind.dialect.name.lower()
            skills_payload = clean_skills if is_postgres else json.dumps(clean_skills)

            db.execute(
                text("""
                    INSERT INTO user_profiles
                    (user_id, skills, target_role, preferred_location, experience_level, updated_at)
                    VALUES (:user_id, :skills, :target_role, :location, :experience, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id)
                    DO UPDATE SET
                        skills = EXCLUDED.skills,
                        target_role = EXCLUDED.target_role,
                        preferred_location = EXCLUDED.preferred_location,
                        experience_level = EXCLUDED.experience_level,
                        updated_at = CURRENT_TIMESTAMP
                """),
                {
                    "user_id": user_id,
                    "skills": skills_payload,
                    "target_role": target_role,
                    "location": location,
                    "experience": experience
                }
            )
        return True, "Profile saved successfully."

    except Exception as e:
        logger.error(f"save_profile error: {e}")
        return False, f"Failed to save profile: {str(e)}"


def get_profile(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve user profile by user_id.
    
    Guarantees 'skills' is returned as a clean Python list of strings.
    """
    ensure_database_initialized()

    try:
        with get_db_session() as db:
            result = db.execute(
                text("""
                    SELECT user_id, skills, target_role, preferred_location,
                           experience_level, updated_at
                    FROM user_profiles
                    WHERE user_id = :user_id
                """),
                {"user_id": user_id}
            ).mappings().first()

            if not result:
                return None

            profile = dict(result)
            raw_skills = profile.get("skills")

            # Normalize skills to list
            if isinstance(raw_skills, list):
                profile["skills"] = [str(s) for s in raw_skills]
            elif isinstance(raw_skills, str):
                try:
                    parsed = json.loads(raw_skills)
                    profile["skills"] = [str(s) for s in parsed] if isinstance(parsed, list) else [raw_skills]
                except Exception:
                    profile["skills"] = [s.strip() for s in raw_skills.split(",") if s.strip()]
            else:
                profile["skills"] = []

            return profile

    except Exception as e:
        logger.error(f"get_profile error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. ANALYSIS HISTORY INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def add_history(user_id: int, analysis_type: str, result: Any) -> Tuple[bool, str]:
    """
    Insert a structured analysis result into the analysis_history table.
    
    Serializes result as valid PostgreSQL JSONB, handling non-native types.
    """
    ensure_database_initialized()

    current_uid = get_current_user_id()
    if current_uid is not None and user_id != current_uid:
        return False, "Unauthorized: Cannot add history for another user."

    payload_json = serialize_for_jsonb(result)

    try:
        with get_db_session() as db:
            # Check for PostgreSQL jsonb cast vs SQLite text
            is_postgres = "postgres" in db.bind.dialect.name.lower()
            sql_clause = "CAST(:result AS jsonb)" if is_postgres else ":result"

            db.execute(
                text(f"""
                    INSERT INTO analysis_history
                    (user_id, analysis_type, result, created_at)
                    VALUES (:user_id, :analysis_type, {sql_clause}, CURRENT_TIMESTAMP)
                """),
                {
                    "user_id": user_id,
                    "analysis_type": str(analysis_type).strip(),
                    "result": payload_json
                }
            )
        return True, "Analysis history saved successfully."

    except Exception as e:
        logger.error(f"add_history error: {e}")
        return False, f"Failed to save analysis history: {str(e)}"


def get_history(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieve analysis history records for user_id ordered from newest to oldest.
    
    Returns deserialized JSON results.
    """
    ensure_database_initialized()

    try:
        with get_db_session() as db:
            rows = db.execute(
                text("""
                    SELECT id, analysis_type, result, created_at
                    FROM analysis_history
                    WHERE user_id = :user_id
                    ORDER BY created_at DESC
                    LIMIT :limit
                """),
                {"user_id": user_id, "limit": limit}
            ).mappings().all()

            results = []
            for row in rows:
                item = dict(row)
                res_val = item.get("result")
                if isinstance(res_val, str):
                    try:
                        item["result"] = json.loads(res_val)
                    except Exception:
                        pass
                results.append(item)
            return results

    except Exception as e:
        logger.error(f"get_history error: {e}")
        return []


def get_user_history(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """Alias for get_history for backwards compatibility."""
    return get_history(user_id, limit=limit)


def delete_history_item(user_id: int, history_id: int) -> bool:
    """Safely delete an analysis history record belonging to the given user."""
    ensure_database_initialized()

    current_uid = get_current_user_id()
    if current_uid is not None and user_id != current_uid:
        return False

    try:
        with get_db_session() as db:
            result = db.execute(
                text("""
                    DELETE FROM analysis_history
                    WHERE id = :history_id AND user_id = :user_id
                """),
                {"history_id": history_id, "user_id": user_id}
            )
            return result.rowcount > 0
    except Exception as e:
        logger.error(f"delete_history_item error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAVED JOBS FUNCTIONALITY
# ─────────────────────────────────────────────────────────────────────────────

def save_job(user_id: int, job_id: str) -> Tuple[bool, str]:
    """
    Bookmark a job for the authenticated user.
    
    Prevents duplicate entries via ON CONFLICT DO NOTHING.
    """
    ensure_database_initialized()

    current_uid = get_current_user_id()
    if current_uid is not None and user_id != current_uid:
        return False, "Unauthorized."

    job_id_str = str(job_id).strip()
    if not job_id_str:
        return False, "Invalid job identifier."

    try:
        with get_db_session() as db:
            db.execute(
                text("""
                    INSERT INTO saved_jobs (user_id, job_id, saved_at)
                    VALUES (:user_id, :job_id, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id, job_id) DO NOTHING
                """),
                {
                    "user_id": user_id,
                    "job_id": job_id_str
                }
            )
        return True, "Job bookmarked successfully."

    except Exception as e:
        logger.error(f"save_job error: {e}")
        return False, f"Failed to save job: {str(e)}"


def remove_saved_job(user_id: int, job_id: str) -> bool:
    """Remove a bookmarked job for the authenticated user."""
    ensure_database_initialized()

    current_uid = get_current_user_id()
    if current_uid is not None and user_id != current_uid:
        return False

    try:
        with get_db_session() as db:
            result = db.execute(
                text("""
                    DELETE FROM saved_jobs
                    WHERE user_id = :user_id AND job_id = :job_id
                """),
                {
                    "user_id": user_id,
                    "job_id": str(job_id).strip()
                }
            )
            return result.rowcount > 0
    except Exception as e:
        logger.error(f"remove_saved_job error: {e}")
        return False


def is_job_saved(user_id: int, job_id: str) -> bool:
    """Check if a specific job is already saved by the user."""
    ensure_database_initialized()

    try:
        with get_db_session() as db:
            row = db.execute(
                text("""
                    SELECT 1 FROM saved_jobs
                    WHERE user_id = :user_id AND job_id = :job_id
                    LIMIT 1
                """),
                {"user_id": user_id, "job_id": str(job_id).strip()}
            ).first()
            return row is not None
    except Exception:
        return False


def get_saved_jobs(user_id: int) -> List[Dict[str, Any]]:
    """Retrieve raw saved jobs for user_id."""
    ensure_database_initialized()

    try:
        with get_db_session() as db:
            rows = db.execute(
                text("""
                    SELECT id, job_id, saved_at
                    FROM saved_jobs
                    WHERE user_id = :user_id
                    ORDER BY saved_at DESC
                """),
                {"user_id": user_id}
            ).mappings().all()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"get_saved_jobs error: {e}")
        return []


def get_saved_jobs_detailed(user_id: int, df_jobs: Any) -> List[Dict[str, Any]]:
    """
    Retrieve saved jobs enriched with metadata from the recommendation dataset.
    
    If a job is no longer present in the dataset, returns a graceful fallback
    record rather than crashing. Never invents job information.
    """
    raw_saved = get_saved_jobs(user_id)
    if not raw_saved:
        return []

    import pandas as pd

    enriched = []
    for item in raw_saved:
        ident = item["job_id"]
        saved_at = item["saved_at"]
        matched_row = None

        if df_jobs is not None and not df_jobs.empty:
            # Try matching by job_id or url
            mask = pd.Series([False] * len(df_jobs))
            if "job_id" in df_jobs.columns:
                mask = mask | (df_jobs["job_id"] == ident)
            if "url" in df_jobs.columns:
                mask = mask | (df_jobs["url"] == ident)

            subset = df_jobs[mask]
            if not subset.empty:
                matched_row = subset.iloc[0]

        if matched_row is not None:
            # Salary formatting
            salary_str = "Not disclosed"
            if pd.notna(matched_row.get("annual_salary_avg")) and matched_row.get("annual_salary_avg", 0) > 0:
                salary_str = f"${int(matched_row['annual_salary_avg']):,}/yr (USD)"
            elif pd.notna(matched_row.get("salary_display_min")) and pd.notna(matched_row.get("salary_display_currency")):
                curr = matched_row["salary_display_currency"]
                lo = int(matched_row["salary_display_min"])
                hi = int(matched_row["salary_display_max"]) if pd.notna(matched_row.get("salary_display_max")) else lo
                period = matched_row.get("salary_display_period", "yr")
                salary_str = f"{curr} {lo:,}–{hi:,} / {period}"

            enriched.append({
                "saved_id": item["id"],
                "job_id": ident,
                "saved_at": saved_at,
                "in_dataset": True,
                "title": str(matched_row.get("title_clean") or matched_row.get("title", "Job Posting")).title(),
                "company_name": str(matched_row.get("company_name", "Unknown Company")).title(),
                "location": f"{matched_row.get('country', '')}" + (f" ({matched_row.get('remote_type', '')})" if pd.notna(matched_row.get("remote_type")) else ""),
                "salary": salary_str,
                "url": str(matched_row.get("url", "")) if pd.notna(matched_row.get("url")) else ident
            })
        else:
            # Graceful fallback when job is not in dataset
            is_url = str(ident).startswith("http")
            enriched.append({
                "saved_id": item["id"],
                "job_id": ident,
                "saved_at": saved_at,
                "in_dataset": False,
                "title": "Saved Listing (Archived/External)",
                "company_name": "Historical Dataset Record",
                "location": "Location info archived",
                "salary": "Not disclosed",
                "url": ident if is_url else ""
            })

    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# 5. AI CAREER ASSISTANT CHAT MESSAGES PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_chat_message(
    user_id: int,
    session_id: str,
    role: str,
    message: str
) -> bool:
    """
    Save a user or assistant message to chat_messages.
    
    Roles allowed: 'user', 'assistant', 'system'.
    """
    ensure_database_initialized()

    role = str(role).strip().lower()
    if role not in ("user", "assistant", "system"):
        return False
    if not message or not str(message).strip():
        return False
    session_id = str(session_id).strip()
    if not session_id:
        session_id = "default"

    try:
        with get_db_session() as db:
            db.execute(
                text("""
                    INSERT INTO chat_messages
                    (user_id, session_id, role, message, created_at)
                    VALUES (:user_id, :session_id, :role, :message, CURRENT_TIMESTAMP)
                """),
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "role": role,
                    "message": str(message)
                }
            )
        return True
    except Exception as e:
        logger.error(f"save_chat_message error: {e}")
        return False


def get_chat_history(
    user_id: int,
    session_id: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Retrieve conversation messages in chronological order (oldest first).
    
    Strictly isolated to user_id.
    """
    ensure_database_initialized()

    try:
        with get_db_session() as db:
            if session_id:
                rows = db.execute(
                    text("""
                        SELECT id, session_id, role, message, created_at
                        FROM chat_messages
                        WHERE user_id = :user_id AND session_id = :session_id
                        ORDER BY created_at ASC
                        LIMIT :limit
                    """),
                    {"user_id": user_id, "session_id": str(session_id), "limit": limit}
                ).mappings().all()
            else:
                rows = db.execute(
                    text("""
                        SELECT id, session_id, role, message, created_at
                        FROM chat_messages
                        WHERE user_id = :user_id
                        ORDER BY created_at ASC
                        LIMIT :limit
                    """),
                    {"user_id": user_id, "limit": limit}
                ).mappings().all()

            return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"get_chat_history error: {e}")
        return []


def get_user_chat_sessions(user_id: int) -> List[Dict[str, Any]]:
    """Retrieve distinct conversation session IDs with latest timestamps."""
    ensure_database_initialized()

    try:
        with get_db_session() as db:
            rows = db.execute(
                text("""
                    SELECT session_id,
                           COUNT(*) as message_count,
                           MAX(created_at) as last_activity
                    FROM chat_messages
                    WHERE user_id = :user_id
                    GROUP BY session_id
                    ORDER BY last_activity DESC
                """),
                {"user_id": user_id}
            ).mappings().all()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"get_user_chat_sessions error: {e}")
        return []


def clear_chat_session(user_id: int, session_id: str) -> bool:
    """Clear messages for a specific session."""
    ensure_database_initialized()

    try:
        with get_db_session() as db:
            db.execute(
                text("""
                    DELETE FROM chat_messages
                    WHERE user_id = :user_id AND session_id = :session_id
                """),
                {"user_id": user_id, "session_id": str(session_id)}
            )
        return True
    except Exception as e:
        logger.error(f"clear_chat_session error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 6. FEEDBACK INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def submit_feedback(
    user_id: Optional[int],
    rating: int,
    comments: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Submit user feedback and rating (1-5).
    
    Accepts nullable user_id for anonymous feedback.
    """
    ensure_database_initialized()

    try:
        rating_int = int(rating)
    except (ValueError, TypeError):
        return False, "Rating must be an integer."

    if not (1 <= rating_int <= 5):
        return False, "Rating must be between 1 and 5 stars."

    comments_str = (comments or "").strip() or None

    try:
        with get_db_session() as db:
            db.execute(
                text("""
                    INSERT INTO feedback (user_id, rating, comments, created_at)
                    VALUES (:user_id, :rating, :comments, CURRENT_TIMESTAMP)
                """),
                {
                    "user_id": user_id,
                    "rating": rating_int,
                    "comments": comments_str
                }
            )
        return True, "Thank you! Your feedback has been submitted successfully."

    except Exception as e:
        logger.error(f"submit_feedback error: {e}")
        return False, "Unable to record feedback right now. Please try again later."


def get_all_feedback(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve submitted feedback records."""
    ensure_database_initialized()

    try:
        with get_db_session() as db:
            rows = db.execute(
                text("""
                    SELECT f.id, f.user_id, u.name as user_name, f.rating, f.comments, f.created_at
                    FROM feedback f
                    LEFT JOIN users u ON f.user_id = u.id
                    ORDER BY f.created_at DESC
                    LIMIT :limit
                """),
                {"limit": limit}
            ).mappings().all()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"get_all_feedback error: {e}")
        return []


def init_auth_tables():
    """Backwards compatibility alias for ensure_database_initialized."""
    ensure_database_initialized()
