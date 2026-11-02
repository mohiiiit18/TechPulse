"""TechPulse Phase 4 — Text similarity module (TF-IDF + cosine similarity).

Content-based signal for the recommendation engine: builds a TF-IDF matrix
over a job "content document" (cleaned title + skills + cleaned dataset text)
once, then transforms a user profile into a query vector with the *same*
fitted vectorizer — the matrix and vectorizer are cached (functools.lru_cache)
so nothing is refit per user request.

Dataset note: the cleaned dataset has NO free-text job description column
(inspected: no `description`/`description_text` column exists). The content
document therefore uses the available text: title_clean, title, skills,
department and company_name. Empty descriptions/titles are handled by
substituting the skills text alone; a job with no text at all gets an empty
document and similarity 0.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Tokens that carry no topical signal in this dataset's titles.
_STOPWORDS = {
    "and", "the", "for", "with", "of", "to", "in", "a", "an", "at", "on",
    "our", "your", "we", "you", "are", "is", "be", "will", "job", "role",
    "position", "hiring", "looking", "apply", "now", "new", "part", "full",
    "time", "remote", "hybrid", "onsite", "on-site", "contract", "temporary",
    "f", "m", "x", "ii", "iii", "iv", "sr", "jr",
}

_TOKEN_RE = re.compile(r"[a-z0-9+#.\-]+")

# Duplicate/boilerplate fragments observed in raw titles (kept out of the doc).
_BOILERPLATE = [
    r"\bapply now\b", r"\burgent(ly)? hiring\b", r"\bimmediate(ly)?\b",
    r"\b\d+[- ]?(hrs?|hours?)[ /]*(per|/)\s*week\b", r"\bfreelance\b",
]


def clean_text(text) -> str:
    """Clean one piece of dataset text: lowercase, strip boilerplate, tokenize,
    drop stopwords. Returns "" for NaN/empty input."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).lower()
    for pattern in _BOILERPLATE:
        s = re.sub(pattern, " ", s)
    tokens = [t for t in _TOKEN_RE.findall(s) if t not in _STOPWORDS]
    return " ".join(tokens)


def build_content_document(row) -> str:
    """Assemble one job's content document from available dataset columns.

    Missing fields are skipped (never fabricated). If the row has no text at
    all, returns "" (caller assigns similarity 0).
    """
    parts: list[str] = []
    for col in ("title_clean", "title"):
        val = row.get(col)
        if val is not None and pd.notna(val) and str(val).strip():
            parts.append(clean_text(val))
    # Skills are the strongest signal: include twice to up-weight them.
    skills = row.get("parsed_skills")
    if skills is not None and len(skills) > 0:
        skill_text = clean_text(" ".join(str(s).replace("-", " ") for s in skills))
        parts.extend([skill_text, skill_text])
    for col in ("department", "job_role"):
        val = row.get(col)
        if val is not None and pd.notna(val) and str(val).strip() and str(val) != "Unknown":
            parts.append(clean_text(val))
    return " ".join(p for p in parts if p)


def build_user_query(user_skills, preferred_role: str | None = None) -> str:
    """Build the user-profile query document for the fitted vectorizer."""
    parts: list[str] = []
    if user_skills:
        skill_text = clean_text(" ".join(str(s).replace("-", " ") for s in user_skills))
        parts.extend([skill_text, skill_text])  # up-weight skills, mirroring jobs
    if preferred_role and str(preferred_role).strip() and preferred_role != "None":
        parts.append(clean_text(preferred_role))
    return " ".join(p for p in parts if p)


@lru_cache(maxsize=8)
def _fit_tfidf(documents_tuple: tuple[str, ...], min_df: int):
    """Fit TfidfVectorizer once per corpus snapshot (cached on content hash).

    `documents_tuple` must be a hashable tuple so lru_cache can memoize.
    Returns (vectorizer, tfidf_matrix).
    """
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=min_df,
        sublinear_tf=True,
        # stop words handled by clean_text(); passing the set here triggers a
        # spurious sklearn inconsistency warning ('on-site' tokenizes to 'site').
        stop_words=None,
    )
    matrix = vectorizer.fit_transform(list(documents_tuple))
    return vectorizer, matrix


def compute_similarities(df: pd.DataFrame, user_skills, preferred_role: str | None = None,
                         min_df: int = 2) -> pd.Series:
    """Return cosine similarity (0-1) of the user profile vs every job row.

    The TF-IDF matrix is fitted once and cached (see _fit_tfidf); per-request
    work is only the query transform + one cosine similarity call.
    Missing descriptions → empty document → similarity 0 (handled, not NaN).
    """
    n = len(df)
    if n == 0:
        return pd.Series(dtype=float)

    documents = [build_content_document(row) for row in df.to_dict("records")]
    if not any(documents):
        return pd.Series(0.0, index=df.index)

    min_df_eff = min(min_df, max(1, sum(1 for d in documents if d) - 1))
    try:
        vectorizer, matrix = _fit_tfidf(tuple(documents), min_df_eff)
    except ValueError:
        # Corpus too small/degenerate for min_df — retry with min_df=1.
        vectorizer, matrix = _fit_tfidf(tuple(documents), 1)

    query = build_user_query(user_skills, preferred_role)
    if not query:
        return pd.Series(0.0, index=df.index)

    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, matrix).ravel()
    return pd.Series(sims, index=df.index)
