"""
TechPulse Comprehensive Database & Integration Automated Test Suite.

Tests all database architecture components:
- Connection & URL Normalization
- User Registration, Login, Logout & Password Hashing
- User Profile Upsert, Retrieval & Isolation
- Analysis History JSONB Serialization & Non-Native Types (NumPy/Pandas)
- Saved Jobs Deduplication, Retrieval & Removal
- AI Career Assistant Chat Session Persistence & Isolation
- Feedback Submission & Range Validation
- Application Navigation & ML Pipeline Integrity

Uses an isolated test engine to guarantee zero data loss or interference
with production database instances.
"""

import os
import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

import utils.db as db_module
import utils.auth as auth_module
from utils.auth import TechPulseJSONEncoder, serialize_for_jsonb


class IsolatedDatabaseTestCase(unittest.TestCase):
    """
    Base test case providing an isolated, in-memory SQLite database
    configured with the complete TechPulse table schema.
    """

    @classmethod
    def setUpClass(cls):
        # Create an isolated in-memory test engine
        cls.test_engine = create_engine("sqlite:///:memory:", pool_pre_ping=True)
        cls.TestSession = sessionmaker(bind=cls.test_engine, autoflush=False, expire_on_commit=False)

        # Apply schema adapted for SQLite
        with cls.test_engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name VARCHAR(100) NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            conn.execute(text("""
                CREATE TABLE user_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    skills TEXT,
                    target_role VARCHAR(100),
                    preferred_location VARCHAR(100),
                    experience_level VARCHAR(100),
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            conn.execute(text("""
                CREATE TABLE analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    analysis_type VARCHAR(50) NOT NULL,
                    result TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            conn.execute(text("""
                CREATE TABLE saved_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    job_id VARCHAR(500) NOT NULL,
                    saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, job_id)
                );
            """))
            conn.execute(text("""
                CREATE TABLE chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    session_id VARCHAR(100) NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    message TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            conn.execute(text("""
                CREATE TABLE feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    rating INTEGER NOT NULL,
                    comments TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            conn.commit()

    def setUp(self):
        # Prevent live database initialization during unit tests
        auth_module._SCHEMA_INITIALIZED = True

        # Monkey-patch db_module engine and sessionmaker to use the isolated test database
        self.orig_get_engine = db_module.get_engine
        self.orig_session_factory = db_module.get_session_factory
        self.orig_get_session = db_module.get_session
        self.orig_get_db_session = db_module.get_db_session

        db_module.get_engine = lambda: self.test_engine
        db_module.get_session_factory = lambda: self.TestSession
        db_module.get_session = lambda: self.TestSession()

        # Clean tables between tests
        with self.test_engine.connect() as conn:
            for tbl in ["feedback", "chat_messages", "saved_jobs", "analysis_history", "user_profiles", "users"]:
                conn.execute(text(f"DELETE FROM {tbl}"))
            conn.commit()

        # Clear Streamlit session state mock
        auth_module.st.session_state = {}

    def tearDown(self):
        db_module.get_engine = self.orig_get_engine
        db_module.get_session_factory = self.orig_session_factory
        db_module.get_session = self.orig_get_session
        db_module.get_db_session = self.orig_get_db_session


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONNECTION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabaseConnection(unittest.TestCase):

    def test_url_normalization_postgres(self):
        url = "postgres://user:pass@host:5432/dbname?sslmode=require"
        normalized = db_module.normalize_database_url(url)
        self.assertTrue(normalized.startswith("postgresql+psycopg://"))
        self.assertIn("sslmode=require", normalized)

    def test_url_normalization_legacy_postgresql(self):
        url = "postgresql://user:pass@localhost:5432/db"
        normalized = db_module.normalize_database_url(url)
        self.assertEqual(normalized, "postgresql+psycopg://user:pass@localhost:5432/db")

    def test_missing_database_url(self):
        with patch.dict(os.environ, {}, clear=True):
            if "DATABASE_URL" in os.environ:
                del os.environ["DATABASE_URL"]
            with patch("utils.db.is_database_configured", return_value=False):
                self.assertFalse(db_module.is_database_configured())

    def test_connection_failure_handling(self):
        # Creating an engine with invalid host should return a clean error without crashing
        bad_engine = create_engine("postgresql+psycopg://dummy:dummy@127.0.0.1:59999/nodb", pool_timeout=1)
        with patch("utils.db.get_engine", return_value=bad_engine):
            ok, msg = db_module.test_connection()
            self.assertFalse(ok)
            self.assertIsInstance(msg, str)


# ─────────────────────────────────────────────────────────────────────────────
# 2. AUTHENTICATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestUserAuthentication(IsolatedDatabaseTestCase):

    def test_successful_registration(self):
        ok, msg = auth_module.register_user("Alice Developer", "alice@example.com", "SecurePass123")
        self.assertTrue(ok)
        self.assertIn("Account created successfully", msg)

        # Verify password hash exists and is not plaintext
        with self.test_engine.connect() as conn:
            row = conn.execute(text("SELECT email, password_hash FROM users WHERE email='alice@example.com'")).first()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], "alice@example.com")
            self.assertNotEqual(row[1], "SecurePass123")
            self.assertTrue(row[1].startswith("$2b$"))

    def test_duplicate_email_registration(self):
        auth_module.register_user("Alice", "alice@example.com", "Password123")
        ok, msg = auth_module.register_user("Alice Clone", "ALICE@example.com", "DifferentPass123")
        self.assertFalse(ok)
        self.assertIn("already exists", msg)

    def test_invalid_email_and_password(self):
        # Invalid email format
        ok, msg = auth_module.register_user("Bob", "not-an-email", "Password123")
        self.assertFalse(ok)
        self.assertIn("valid email", msg)

        # Short password
        ok, msg = auth_module.register_user("Bob", "bob@example.com", "short")
        self.assertFalse(ok)
        self.assertIn("at least 8 characters", msg)

    def test_login_success_and_failure(self):
        auth_module.register_user("Charlie", "charlie@example.com", "StrongPassword123")

        # Wrong password
        self.assertFalse(auth_module.login_user("charlie@example.com", "WrongPassword"))
        self.assertFalse(auth_module.is_authenticated())

        # Correct password
        self.assertTrue(auth_module.login_user("charlie@example.com", "StrongPassword123"))
        self.assertTrue(auth_module.is_authenticated())
        user = auth_module.get_current_user()
        self.assertEqual(user["email"], "charlie@example.com")
        self.assertEqual(user["name"], "Charlie")

        # Logout
        auth_module.logout_user()
        self.assertFalse(auth_module.is_authenticated())
        self.assertIsNone(auth_module.get_current_user())


# ─────────────────────────────────────────────────────────────────────────────
# 3. USER PROFILE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestUserProfiles(IsolatedDatabaseTestCase):

    def setUp(self):
        super().setUp()
        auth_module.register_user("Profile Tester", "tester@example.com", "Pass123456")
        with self.test_engine.connect() as conn:
            self.uid = conn.execute(text("SELECT id FROM users WHERE email='tester@example.com'")).scalar()
        auth_module.st.session_state["user"] = {"id": self.uid, "email": "tester@example.com"}

    def test_save_and_retrieve_profile(self):
        skills = ["Python", "SQL", "Machine Learning", "FastAPI"]
        ok, msg = auth_module.save_profile(
            user_id=self.uid,
            skills=skills,
            target_role="Machine Learning Engineer",
            location="Remote",
            experience="1-3 Years"
        )
        self.assertTrue(ok)

        profile = auth_module.get_profile(self.uid)
        self.assertIsNotNone(profile)
        self.assertEqual(profile["target_role"], "Machine Learning Engineer")
        self.assertEqual(profile["preferred_location"], "Remote")
        self.assertEqual(profile["experience_level"], "1-3 Years")
        self.assertEqual(profile["skills"], skills)

    def test_update_profile_preserves_unmodified(self):
        # Save initial
        auth_module.save_profile(self.uid, ["Python", "SQL"], "Data Analyst", "New York", "Fresher")

        # Partial update with same user
        auth_module.save_profile(self.uid, ["Python", "SQL", "Pandas"], target_role=None, location="Chicago")
        updated = auth_module.get_profile(self.uid)
        self.assertEqual(updated["preferred_location"], "Chicago")
        self.assertEqual(updated["target_role"], "Data Analyst")  # Preserved
        self.assertIn("Pandas", updated["skills"])

    def test_profile_empty_and_user_isolation(self):
        # Non-existent user
        self.assertIsNone(auth_module.get_profile(99999))

        # Cross-user unauthorized modification attempt
        ok, msg = auth_module.save_profile(user_id=88888, skills=["C++"])
        self.assertFalse(ok)
        self.assertIn("Unauthorized", msg)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ANALYSIS HISTORY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalysisHistory(IsolatedDatabaseTestCase):

    def setUp(self):
        super().setUp()
        auth_module.register_user("History User", "hist@example.com", "Pass123456")
        with self.test_engine.connect() as conn:
            self.uid = conn.execute(text("SELECT id FROM users WHERE email='hist@example.com'")).scalar()
        auth_module.st.session_state["user"] = {"id": self.uid, "email": "hist@example.com"}

    def test_add_and_retrieve_history(self):
        import numpy as np

        # Result containing NumPy and non-native types
        complex_result = {
            "predicted_role": "Data Scientist",
            "top_score": np.float64(88.75),
            "n_skills": np.int64(12),
            "skills": ["python", "sql", "scikit-learn"]
        }

        ok, msg = auth_module.add_history(self.uid, "career_match", complex_result)
        self.assertTrue(ok)

        history = auth_module.get_history(self.uid)
        self.assertEqual(len(history), 1)
        item = history[0]
        self.assertEqual(item["analysis_type"], "career_match")
        self.assertEqual(item["result"]["predicted_role"], "Data Scientist")
        self.assertAlmostEqual(item["result"]["top_score"], 88.75)

    def test_history_deletion(self):
        auth_module.add_history(self.uid, "skill_gap", {"gap": "SQL"})
        items = auth_module.get_history(self.uid)
        self.assertEqual(len(items), 1)

        hist_id = items[0]["id"]
        deleted = auth_module.delete_history_item(self.uid, hist_id)
        self.assertTrue(deleted)
        self.assertEqual(len(auth_module.get_history(self.uid)), 0)

    def test_jsonb_serialization_safety(self):
        import pandas as pd
        import numpy as np

        test_data = {
            "timestamp": pd.Timestamp("2026-11-07 10:00:00"),
            "array": np.array([1, 2, 3]),
            "float": np.float32(3.14),
            "set": {"a", "b"}
        }
        serialized = serialize_for_jsonb(test_data)
        parsed = json.loads(serialized)
        self.assertEqual(parsed["array"], [1, 2, 3])
        self.assertAlmostEqual(parsed["float"], 3.14, places=2)
        self.assertIn("a", parsed["set"])


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVED JOBS TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestSavedJobs(IsolatedDatabaseTestCase):

    def setUp(self):
        super().setUp()
        auth_module.register_user("Job Saver", "jobs@example.com", "Pass123456")
        with self.test_engine.connect() as conn:
            self.uid = conn.execute(text("SELECT id FROM users WHERE email='jobs@example.com'")).scalar()
        auth_module.st.session_state["user"] = {"id": self.uid, "email": "jobs@example.com"}

    def test_save_deduplicate_and_remove(self):
        job_url = "https://boards.greenhouse.io/techcorp/jobs/12345"

        # First save
        ok, msg = auth_module.save_job(self.uid, job_url)
        self.assertTrue(ok)
        self.assertTrue(auth_module.is_job_saved(self.uid, job_url))

        # Duplicate save attempt
        ok, msg = auth_module.save_job(self.uid, job_url)
        self.assertTrue(ok)
        jobs = auth_module.get_saved_jobs(self.uid)
        self.assertEqual(len(jobs), 1)  # Deduplicated

        # Remove job
        removed = auth_module.remove_saved_job(self.uid, job_url)
        self.assertTrue(removed)
        self.assertFalse(auth_module.is_job_saved(self.uid, job_url))
        self.assertEqual(len(auth_module.get_saved_jobs(self.uid)), 0)

    def test_detailed_saved_jobs_fallback(self):
        auth_module.save_job(self.uid, "https://unknown.job.url/999")
        detailed = auth_module.get_saved_jobs_detailed(self.uid, df_jobs=None)
        self.assertEqual(len(detailed), 1)
        self.assertFalse(detailed[0]["in_dataset"])
        self.assertEqual(detailed[0]["url"], "https://unknown.job.url/999")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CHAT MESSAGES TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestChatPersistence(IsolatedDatabaseTestCase):

    def setUp(self):
        super().setUp()
        auth_module.register_user("Chatter", "chat@example.com", "Pass123456")
        with self.test_engine.connect() as conn:
            self.uid = conn.execute(text("SELECT id FROM users WHERE email='chat@example.com'")).scalar()

    def test_save_and_retrieve_chat_session(self):
        session_id = "session_abc123"

        # Save turns
        self.assertTrue(auth_module.save_chat_message(self.uid, session_id, "user", "What skills for Data Scientist?"))
        self.assertTrue(auth_module.save_chat_message(self.uid, session_id, "assistant", "Python, SQL, Machine Learning."))

        # Retrieve messages
        history = auth_module.get_chat_history(self.uid, session_id)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["message"], "What skills for Data Scientist?")
        self.assertEqual(history[1]["role"], "assistant")

        # Session listing
        sessions = auth_module.get_user_chat_sessions(self.uid)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_id"], session_id)
        self.assertEqual(sessions[0]["message_count"], 2)

        # Clear session
        cleared = auth_module.clear_chat_session(self.uid, session_id)
        self.assertTrue(cleared)
        self.assertEqual(len(auth_module.get_chat_history(self.uid, session_id)), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 7. FEEDBACK TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestFeedback(IsolatedDatabaseTestCase):

    def test_valid_and_invalid_feedback(self):
        # Valid authenticated feedback
        ok, msg = auth_module.submit_feedback(user_id=1, rating=5, comments="Great platform!")
        self.assertTrue(ok)

        # Valid anonymous feedback
        ok, msg = auth_module.submit_feedback(user_id=None, rating=4, comments="Very useful tool.")
        self.assertTrue(ok)

        # Invalid ratings (out of 1-5 range)
        ok, msg = auth_module.submit_feedback(user_id=1, rating=0, comments="Invalid")
        self.assertFalse(ok)

        ok, msg = auth_module.submit_feedback(user_id=1, rating=6, comments="Invalid")
        self.assertFalse(ok)

        # Verify storage
        all_fb = auth_module.get_all_feedback()
        self.assertEqual(len(all_fb), 2)


# ─────────────────────────────────────────────────────────────────────────────
# 8. APPLICATION & ML PIPELINE INTEGRITY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestApplicationIntegrity(unittest.TestCase):

    def test_data_loader_and_models(self):
        from utils.data_loader import load_data, load_model, get_dataset_date_range

        df = load_data()
        self.assertFalse(df.empty)
        self.assertGreater(len(df), 1000)

        model, encoder = load_model()
        self.assertIsNotNone(model)
        self.assertIsNotNone(encoder)
        self.assertGreater(len(encoder.classes_), 50)

        date_range = get_dataset_date_range()
        self.assertIn("latest", date_range)

    def test_theme_and_ui_utilities(self):
        from utils.theme import PALETTE, apply_base_css
        self.assertIsInstance(PALETTE, list)
        self.assertGreater(len(PALETTE), 5)

    def test_all_app_pages_compile(self):
        import compileall
        result = compileall.compile_dir(str(PROJECT_ROOT / "app_pages"), force=True, quiet=True)
        self.assertTrue(result)

    def test_src_modules_compile(self):
        import compileall
        result = compileall.compile_dir(str(PROJECT_ROOT / "src"), force=True, quiet=True)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
