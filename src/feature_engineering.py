"""
TechPulse ML — feature engineering pipeline.

Converts the cleaned dataset into (X, y, groups) using the reusable
SkillEncoder. The target column is NEVER included in features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import SkillEncoder, parse_skill_list

TARGET_COL = "job_role"
SKILL_COL = "parsed_skills_std"


def prepare_features(df: pd.DataFrame, encoder: SkillEncoder | None = None):
    """Build (X, y, groups, encoder, stats).

    Parameters
    ----------
    df : cleaned dataset (requires TARGET_COL and SKILL_COL columns).
    encoder : optional pre-fitted SkillEncoder (used at prediction time).

    Returns
    -------
    X        : np.ndarray multi-hot matrix (skill features only)
    y        : np.ndarray target labels
    groups   : np.ndarray leakage-group ids
    encoder  : fitted SkillEncoder
    stats    : dict with vocabulary size, unknown counts, empty rows, etc.
    """
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' missing from dataset.")
    if SKILL_COL not in df.columns:
        raise KeyError(f"Skill column '{SKILL_COL}' missing from dataset.")

    df = df[df[TARGET_COL].notna()].copy()
    skill_lists = df[SKILL_COL].apply(parse_skill_list).tolist()
    empty_rows = sum(1 for lst in skill_lists if not lst)
    # Rows with genuinely no skills carry no signal; keep them but record it.
    non_empty_mask = [bool(lst) for lst in skill_lists]

    y = df[TARGET_COL].astype(str).to_numpy()
    groups_raw = df[SKILL_COL].astype(str).to_numpy()

    if encoder is None:
        encoder = SkillEncoder().fit([lst for lst, keep in zip(skill_lists, non_empty_mask) if keep])

    X, _ = encoder.transform(skill_lists)

    # Group ids computed on full skill strings (near-duplicate detection)
    import numpy as _np
    _, inv = _np.unique(groups_raw, return_inverse=True)
    groups = inv.astype(np.int64)

    stats = {
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "vocabulary_size": len(encoder.classes_),
        "n_target_classes": int(len(np.unique(y))),
        "empty_skill_rows": empty_rows,
        "density": float(X.mean()) if X.size else 0.0,
        "target_included_in_features": False,  # guaranteed: X built from skills only
    }
    return X, y, groups, encoder, stats
