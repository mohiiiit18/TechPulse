"""
TechPulse Phase 8 — Optional LLM provider.

Design goals:
- Entirely OPTIONAL: the assistant is fully usable offline (rule-based +
  retrieval) with no API key at all.
- Key comes from Streamlit secrets (st.secrets) or environment variables —
  never hardcoded, never logged.
- Any API failure raises LLMUnavailable and the caller degrades gracefully.
- Resume text is only included in prompts when the user has explicitly
  consented (consent flag checked by the caller, not here).
"""
from __future__ import annotations

import json
import os
import urllib.request


class LLMUnavailable(Exception):
    """No provider configured, or the API call failed."""


def _api_key() -> str | None:
    key = os.environ.get("TECHPULSE_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip() or None
    try:
        key = st_secrets_key()
    except Exception:
        key = None
    return key


def st_secrets_key() -> str | None:
    import streamlit as st
    try:
        if "LLM_API_KEY" in st.secrets:
            return str(st.secrets["LLM_API_KEY"]).strip() or None
    except Exception:
        pass
    return None


def provider_name() -> str:
    return "OpenAI (chat-completions, TECHPULSE_LLM_API_KEY or OPENAI_API_KEY or st.secrets['LLM_API_KEY'])"


def llm_available() -> bool:
    return _api_key() is not None


def chat(prompt: str, system: str = "", timeout: int = 30) -> str:
    """One-shot completion. Raises LLMUnavailable on any failure.

    The consent check for private resume text is the CALLER's responsibility:
    this function must never receive resume content unless the user opted in.
    """
    key = _api_key()
    if not key:
        raise LLMUnavailable("No LLM API key configured.")
    body = json.dumps({
        "model": os.environ.get("TECHPULSE_LLM_MODEL", "gpt-4o-mini"),
        "messages": ([{"role": "system", "content": system}] if system else [])
        + [{"role": "user", "content": prompt}],
        "max_tokens": 600,
        "temperature": 0.4,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise LLMUnavailable(f"LLM API call failed: {e.__class__.__name__}") from e
