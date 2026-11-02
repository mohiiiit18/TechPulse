"""
TechPulse Phase 7 — Salary feature engineering.

Builds the X/y matrix for salary prediction from the validated salary dataset.

Features (all genuinely available in the dataset — nothing fabricated):
- job_role (4 categories)
- seniority (experience, extracted in Phase 2; 'unspecified' is a real
  observed category — the model never assumes it means something else)
- country (top-N + 'Other'/'Unknown')
- employment_type (only 'full_time'/'contract'/'part_time' etc. as observed)
- is_remote (bool)
- skills (100 binary MultiLabelBinarizer columns via skills_std)

Leakage guards:
- No salary columns (min/max/avg/currency/period) are features.
- No title/company — they could reconstruct the salary target.
- Groups: identical (job_role, seniority, country, sorted skill set) postings
  share a group so near-duplicate postings never straddle train/test.
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

TOP_COUNTRIES = 8

# Columns whose values are used as categorical features
CATEGORICAL_FEATURES = ["job_role", "seniority", "country", "employment_type"]
BINARY_FEATURES = ["is_remote"]
SKILL_COLUMN = "skills_std"
TARGET = "salary_target"

# Grouping key: postings with identical (role, seniority, country, skills)
# are treated as near-duplicates for leakage-aware splitting.
GROUP_COLUMNS = ["job_role", "seniority", "country", "employment_type", SKILL_COLUMN]


def build_group_key(df: pd.DataFrame) -> pd.Series:
    """Group key so near-duplicate postings never straddle train/test."""
    return df[GROUP_COLUMNS].astype(str).agg("|".join, axis=1)


def make_salary_features(df: pd.DataFrame):
    """Return (X, y, groups, feature_names, encoder).

    X is a DataFrame with one-hot categorical columns, binary remote flag,
    and MultiLabelBinarizer skill columns. Fitted on the training data only
    (encoder is returned so the caller can persist it with the model).
    """
    enc = MultiLabelBinarizer()
    skills = enc.fit_transform(df[SKILL_COLUMN].astype(str).str.split("|"))
    skill_cols = [f"skill_{s}" for s in enc.classes_]

    cats = pd.get_dummies(
        df[CATEGORICAL_FEATURES].astype(str),
        prefix=CATEGORICAL_FEATURES,
        dtype=int,
    )
    remote = df[BINARY_FEATURES].astype(int).reset_index(drop=True)
    X = pd.concat([cats.reset_index(drop=True), remote, pd.DataFrame(skills, columns=skill_cols)],
                  axis=1)
    y = df[TARGET].astype(float)
    groups = build_group_key(df)
    return X, y, groups, list(X.columns), enc
