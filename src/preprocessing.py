"""
TechPulse ML — reusable preprocessing utilities.

Handles skill normalization, MultiLabelBinarizer encoding (with safe unknown
handling) and group-key construction for leakage-aware splitting.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Skill alias map — mirrors scripts/clean_dataset.py SKILL_ALIASES (kept in sync).
SKILL_ALIASES = {
    "ml": "machine-learning",
    "machine learning": "machine-learning",
    "natural language processing": "nlp",
    "js": "javascript",
    "nodejs": "node.js",
    "node": "node.js",
    "reactjs": "react",
    "react.js": "react",
    "nextjs": "next.js",
    "postgres": "postgresql",
    "k8s": "kubernetes",
    "ci/cd": "ci-cd",
    "tf": "terraform",
    "tf-learning": "tensorflow",
    "tensor flow": "tensorflow",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "html/css": "html-css",
    "html css": "html-css",
    "ui/ux": "ui-ux",
    "qa": "qa-testing",
}

_VALID_TOKEN = re.compile(r"^[a-z0-9.#+-]+$")


def normalize_skill(token: str) -> str | None:
    """Normalize a single skill token; returns None for invalid tokens."""
    if token is None:
        return None
    tok = str(token).strip().lower()
    if not tok or not _VALID_TOKEN.match(tok):
        return None
    return SKILL_ALIASES.get(tok, tok)


def parse_skill_list(skills) -> list[str]:
    """Parse a skill field into a normalized, de-duplicated list.

    Accepts pipe-separated strings ("python|sql") AND stringified Python
    lists ("['python', 'sql']", as produced by CSV round-trips of list
    columns). NaN / empty / malformed input -> [] (missing info preserved,
    never fabricated).
    """
    if skills is None or (isinstance(skills, float) and np.isnan(skills)):
        return []
    s = str(skills).strip()
    if s == "" or s.lower() == "nan" or s == "[]":
        return []
    # Stringified-list form (from CSV round-trip of a list column)
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, set)):
                seen: set[str] = set()
                out: list[str] = []
                for tok in parsed:
                    norm = normalize_skill(tok)
                    if norm and norm not in seen:
                        seen.add(norm)
                        out.append(norm)
                return out
        except (ValueError, SyntaxError):
            pass  # fall through to pipe-separated handling
    seen: set[str] = set()
    out: list[str] = []
    for tok in s.split("|"):
        norm = normalize_skill(tok)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def group_key(skills_std: str) -> str:
    """Leakage-group key: rows with identical skill sets belong to one group.

    Near-duplicate postings (identical `skills_std`) share a group key so a
    group-aware split keeps them on the same side of the train/test boundary.
    """
    return str(skills_std)


def build_group_array(skills_std: pd.Series) -> np.ndarray:
    """Map group keys to integer group ids for GroupShuffleSplit/GroupKFold."""
    keys = skills_std.map(group_key)
    codes = pd.factorize(keys)[0]
    return codes.astype(np.int64)


class SkillEncoder:
    """Thin wrapper around MultiLabelBinarizer with safe unknown-skill handling.

    `transform` silently ignores skills unseen during training (they map to an
    all-zero row) instead of raising, so prediction never crashes on unknown
    or emerging skills.
    """

    def __init__(self):
        self._mlb = MultiLabelBinarizer()
        self._fitted = False

    @property
    def classes_(self) -> np.ndarray:
        return self._mlb.classes_

    @property
    def known_skills(self) -> list[str]:
        return list(self._mlb.classes_) if self._fitted else []

    def fit(self, skill_lists: list[list[str]]) -> "SkillEncoder":
        self._mlb.fit(skill_lists)
        self._fitted = True
        return self

    def transform(self, skill_lists: list[list[str]]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("SkillEncoder must be fitted before transform.")
        known = set(self._mlb.classes_)
        filtered = [[s for s in lst if s in known] for lst in skill_lists]
        unknown = {s for lst in skill_lists for s in lst if s not in known}
        X = self._mlb.transform(filtered)
        return X, unknown

    def transform_single(self, skills: list[str]):
        """Encode one skill list -> (X 1-row, unknown-skills set)."""
        return self.transform([skills])
