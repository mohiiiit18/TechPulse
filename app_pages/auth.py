"""
Authentication module proxy for backward compatibility.
Canonical location: utils/auth.py
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.auth import (
    register_user,
    login_user,
    logout_user,
    is_authenticated,
    get_current_user,
    get_current_user_id,
    add_history,
    get_user_history,
    get_history,
    delete_history_item,
    save_profile,
    get_profile,
    save_job,
    remove_saved_job,
    is_job_saved,
    get_saved_jobs,
    get_saved_jobs_detailed,
    save_chat_message,
    get_chat_history,
    get_user_chat_sessions,
    clear_chat_session,
    submit_feedback,
    get_all_feedback,
    init_auth_tables,
    ensure_database_initialized,
)

__all__ = [
    "register_user",
    "login_user",
    "logout_user",
    "is_authenticated",
    "get_current_user",
    "get_current_user_id",
    "add_history",
    "get_user_history",
    "get_history",
    "delete_history_item",
    "save_profile",
    "get_profile",
    "save_job",
    "remove_saved_job",
    "is_job_saved",
    "get_saved_jobs",
    "get_saved_jobs_detailed",
    "save_chat_message",
    "get_chat_history",
    "get_user_chat_sessions",
    "clear_chat_session",
    "submit_feedback",
    "get_all_feedback",
    "init_auth_tables",
    "ensure_database_initialized",
]
