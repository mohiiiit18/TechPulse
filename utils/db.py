"""
TechPulse Database Engine & Session Management Module.

Provides centralized SQLAlchemy 2.x engine management, connection pooling,
safe session scoping, URL normalization for Psycopg 3, and health checking.
Supports both local PostgreSQL and cloud providers (Neon, Supabase, RDS).
"""

import os
import re
import logging
from contextlib import contextmanager
from typing import Optional, Tuple, Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, text, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool

# Load environment variables from .env
load_dotenv()

logger = logging.getLogger("techpulse.db")

# Global engine singleton for non-Streamlit environments (tests, scripts, CLI)
_GLOBAL_ENGINE: Optional[Engine] = None
_GLOBAL_SESSION_FACTORY: Optional[sessionmaker] = None


def normalize_database_url(url: str) -> str:
    """
    Normalize PostgreSQL database URL for SQLAlchemy 2.x + Psycopg 3.
    
    Converts:
      - postgres://... -> postgresql+psycopg://...
      - postgresql://... -> postgresql+psycopg://...
      - postgresql+psycopg2://... -> postgresql+psycopg://...
    Preserves query parameters (e.g. ?sslmode=require).
    """
    if not url:
        return ""
    
    url = url.strip().strip("'\"")
    
    # Replace legacy dialect prefixes with modern psycopg 3 dialect
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    elif url.startswith("postgresql+psycopg2://"):
        url = "postgresql+psycopg://" + url[len("postgresql+psycopg2://"):]
        
    return url


def get_database_url() -> str:
    """
    Retrieve and normalize the database connection URL.
    
    Resolution order:
      1. os.environ["DATABASE_URL"]
      2. streamlit.secrets["DATABASE_URL"] (if inside Streamlit)
    """
    url = os.getenv("DATABASE_URL")

    # Check Streamlit secrets if not found in env
    if not url:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
                url = st.secrets["DATABASE_URL"]
        except Exception:
            pass

    if not url:
        raise ValueError(
            "DATABASE_URL is not configured. Set DATABASE_URL in your .env file "
            "or in Streamlit secrets (e.g. postgresql+psycopg://user:pass@localhost:5432/dbname)."
        )

    return normalize_database_url(url)


def is_database_configured() -> bool:
    """Check whether a DATABASE_URL has been defined in environment or secrets."""
    try:
        url = os.getenv("DATABASE_URL")
        if url:
            return True
        import streamlit as st
        if hasattr(st, "secrets") and "DATABASE_URL" in st.secrets:
            return True
    except Exception:
        pass
    return False


def _create_configured_engine(url: str) -> Engine:
    """Helper to create and configure a SQLAlchemy 2.x Engine with connection pooling."""
    is_sqlite = url.startswith("sqlite")
    
    if is_sqlite:
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True
        )
    
    # PostgreSQL engine with robust production pooling parameters
    return create_engine(
        url,
        poolclass=QueuePool,
        pool_pre_ping=True,       # Auto-reconnect stale sockets
        pool_size=5,              # Steady state pool connections
        max_overflow=10,          # Peak burst connections
        pool_recycle=1800,        # Recycle connections after 30 minutes
        pool_timeout=15           # 15s timeout before raising PoolTimeout
    )


def get_engine() -> Engine:
    """
    Return the centralized SQLAlchemy engine singleton.
    
    Uses Streamlit @st.cache_resource when running inside Streamlit,
    otherwise falls back cleanly to a module-level singleton for tests/scripts.
    """
    global _GLOBAL_ENGINE

    # Try Streamlit-cached engine if runtime context is present
    try:
        import streamlit as st
        # Only use st.cache_resource if inside an active script run context
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            @st.cache_resource(show_spinner=False)
            def _get_st_engine(db_url: str) -> Engine:
                return _create_configured_engine(db_url)
            
            return _get_st_engine(get_database_url())
    except Exception:
        pass

    # Fallback to module singleton (CLI, tests, offline scripts)
    if _GLOBAL_ENGINE is None:
        db_url = get_database_url()
        _GLOBAL_ENGINE = _create_configured_engine(db_url)
        
    return _GLOBAL_ENGINE


def get_session_factory() -> sessionmaker:
    """Return the centralized sessionmaker bound to the active engine."""
    global _GLOBAL_SESSION_FACTORY
    if _GLOBAL_SESSION_FACTORY is None:
        _GLOBAL_SESSION_FACTORY = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            expire_on_commit=False
        )
    return _GLOBAL_SESSION_FACTORY


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions with automatic commit and rollback.
    
    Usage:
        with get_db_session() as db:
            db.execute(...)
            # Automatically committed upon exit; rolled back upon exception
    """
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database transaction error: {e}")
        raise
    finally:
        session.close()


def get_session() -> Session:
    """
    Return a database session for backwards compatibility.
    
    Note: When using directly, prefer calling session.commit() or use get_db_session().
    """
    factory = get_session_factory()
    return factory()


def test_connection() -> Tuple[bool, str]:
    """
    Test connectivity to the configured database.
    
    Returns:
        (True, version_string) on success,
        (False, error_description) on failure.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            if "sqlite" in engine.dialect.name.lower():
                result = conn.execute(text("SELECT sqlite_version()")).scalar()
                return True, f"SQLite {result}"
            else:
                result = conn.execute(text("SELECT version()")).scalar()
                return True, str(result or "Connected successfully.")
    except Exception as e:
        return False, str(e)


def is_database_available() -> bool:
    """Quick boolean check if database is online and reachable."""
    if not is_database_configured():
        return False
    ok, _ = test_connection()
    return ok
