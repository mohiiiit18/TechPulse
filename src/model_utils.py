"""
TechPulse ML — model utilities: split, training, tuning, evaluation, saving.
"""

from __future__ import annotations

import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, StratifiedGroupKFold, cross_val_score

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


# ─────────────────────────────────────────────────────────────────────────────
# Split (leakage-aware)
# ─────────────────────────────────────────────────────────────────────────────
def make_split(X, y, groups):
    """Stratified + group-aware 80/20 split.

    StratifiedGroupKFold preserves both class ratios AND keeps rows sharing
    the same skill-set group on one side of the split, preventing
    near-duplicate leakage between train and test.
    """
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = sgkf.split(X, y, groups)
    train_idx, test_idx = next(splits)
    return train_idx, test_idx


# ─────────────────────────────────────────────────────────────────────────────
# Model zoo
# ─────────────────────────────────────────────────────────────────────────────
def build_models(n_classes: int) -> dict:
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced",
            n_jobs=1,
        ),
        "XGBoost": None,  # filled below if available
    }
    try:
        # XGBoost requires integer class labels 0..K-1; wrap with a LabelEncoder
        # so the same pipeline API works for string targets.
        from sklearn.preprocessing import LabelEncoder

        class _XGBWrapper:
            """Wraps XGBClassifier with label encoding for string targets."""

            def __init__(self, **kw):
                from sklearn.base import BaseEstimator, ClassifierMixin

                class _Inner(BaseEstimator, ClassifierMixin):
                    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.1,
                                 objective="multi:softprob", random_state=42, n_jobs=1,
                                 eval_metric="mlogloss", tree_method="hist",
                                 subsample=1.0, colsample_bytree=1.0):
                        self.n_estimators = n_estimators
                        self.max_depth = max_depth
                        self.learning_rate = learning_rate
                        self.objective = objective
                        self.random_state = random_state
                        self.n_jobs = n_jobs
                        self.eval_metric = eval_metric
                        self.tree_method = tree_method
                        self.subsample = subsample
                        self.colsample_bytree = colsample_bytree
                        self._le = LabelEncoder()
                        self._clf = None

                    def fit(self, X, y):
                        import xgboost as _xgb
                        y_enc = self._le.fit_transform(y)
                        self._clf = _xgb.XGBClassifier(
                            n_estimators=self.n_estimators, max_depth=self.max_depth,
                            learning_rate=self.learning_rate, objective=self.objective,
                            random_state=self.random_state, n_jobs=self.n_jobs,
                            eval_metric=self.eval_metric, tree_method=self.tree_method,
                            subsample=self.subsample, colsample_bytree=self.colsample_bytree,
                        )
                        self._clf.fit(X, y_enc)
                        return self

                    def predict(self, X):
                        return self._le.inverse_transform(self._clf.predict(X).astype(int))

                    def predict_proba(self, X):
                        return self._clf.predict_proba(X)

                    @property
                    def classes_(self):
                        return self._le.classes_

                self._impl = _Inner(**kw)

            def get_params(self, deep=True):
                return self._impl.get_params(deep=deep)

            def set_params(self, **params):
                self._impl.set_params(**params)
                return self

            def fit(self, X, y):
                self._impl.fit(X, y)
                return self

            def predict(self, X):
                return self._impl.predict(X)

            def predict_proba(self, X):
                return self._impl.predict_proba(X)

            @property
            def classes_(self):
                return self._impl.classes_

            def __sklearn_tags__(self):
                return self._impl.__sklearn_tags__()

            def __getattr__(self, name):
                return getattr(self._impl, name)

        kw = dict(n_estimators=300, max_depth=6, learning_rate=0.1,
                  objective="multi:softprob", random_state=RANDOM_STATE,
                  n_jobs=1, eval_metric="mlogloss", tree_method="hist")
        # XGBoost handles imbalance internally via its loss; per-class weights
        # can be added through sample_weight if needed (kept simple here).
        models["XGBoost"] = _XGBWrapper(**kw)
    except ImportError:
        models.pop("XGBoost")
    return models


def param_distributions() -> dict:
    """Compact randomized-search spaces (avoid unnecessarily expensive search)."""
    return {
        "LogisticRegression": {
            "C": [0.1, 0.5, 1.0, 2.0, 5.0],
            "solver": ["lbfgs", "liblinear"],
        },
        "RandomForest": {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "XGBoost": {
            "n_estimators": [200, 300, 500],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation (training data only)
# ─────────────────────────────────────────────────────────────────────────────
def cross_validate(model, X_train, y_train, groups_train, cv_folds: int = 3):
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, groups=groups_train,
                             cv=cv, scoring="f1_macro", n_jobs=1)
    return {"f1_macro_mean": round(float(scores.mean()), 4),
            "f1_macro_std": round(float(scores.std()), 4),
            "folds": [round(float(s), 4) for s in scores]}


# ─────────────────────────────────────────────────────────────────────────────
# Tuning
# ─────────────────────────────────────────────────────────────────────────────
def tune(model_name: str, base_model, X_train, y_train, groups_train, n_iter: int = 15):
    spaces = param_distributions().get(model_name)
    if not spaces:
        return base_model, {}
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base_model, spaces, n_iter=n_iter, scoring="f1_macro",
        cv=cv, random_state=RANDOM_STATE, n_jobs=1, refit=True,
    )
    search.fit(X_train, y_train, groups=groups_train)
    return search.best_estimator_, {
        "best_params": {k: (v if v is not None else "None") for k, v in search.best_params_.items()},
        "best_cv_f1_macro": round(float(search.best_score_), 4),
        "n_iter": n_iter,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(clf, X_train, y_train, X_test, y_test) -> dict:
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    res = {
        "train_accuracy": round(float(train_acc), 4),
        "test_accuracy": round(float(test_acc), 4),
        "precision_macro": round(float(precision_score(y_test, y_pred_test, average="macro", zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(y_test, y_pred_test, average="weighted", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred_test, average="macro", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_test, y_pred_test, average="weighted", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred_test, average="macro")), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred_test, average="weighted")), 4),
        "overfit_gap": round(float(train_acc - test_acc), 4),
        "overfit_warning": bool(train_acc - test_acc > 0.15),
        "classification_report": classification_report(y_test, y_pred_test, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
        "classes": [str(c) for c in clf.classes_] if hasattr(clf, "classes_") else sorted(set(y_test)),
    }
    return res


def format_confusion_png(cm: np.ndarray, classes: list[str], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    short = [c.split(" / ")[0] for c in classes]
    ax.set_xticklabels(short, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Persistence (v3 slot — never overwrite validated production artifacts)
# ─────────────────────────────────────────────────────────────────────────────
def save_model(clf, encoder, metadata: dict):
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "job_role_model_v3.pkl"
    enc_path = MODELS_DIR / "skill_encoder_v3.pkl"
    meta_path = MODELS_DIR / "model_metadata.json"
    joblib.dump(clf, model_path)
    joblib.dump(encoder, enc_path)
    metadata = {
        **metadata,
        "model_file": model_path.name,
        "encoder_file": enc_path.name,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return model_path, enc_path, meta_path
