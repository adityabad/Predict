"""Failure-probability model: gradient boosting on history features, heuristic fallback on cold start."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .history import FEATURES, build_dataset
from .store import Run

MIN_FAILURES_TO_TRAIN = 15
MAX_TRAIN_ROWS = 400_000


def heuristic_scores(X: np.ndarray) -> np.ndarray:
    """Hand-weighted score used until there is enough failure history to train."""
    f = {name: X[:, i] for i, name in enumerate(FEATURES)}
    s = (
        0.40 * f["last_failed"]
        + 0.25 * f["fail_rate_20"]
        + 0.60 * np.clip(f["assoc_lift"], 0, 1)
        + 0.30 * f["test_file_changed"]
        + 0.10 * f["path_similarity"]
        + 0.10 * f["is_new"]
    )
    return np.clip(s, 0.0, 1.0)


class FailureModel:
    def __init__(self, clf=None, negative_rate: float = 1.0, meta: Optional[Dict] = None):
        self.clf = clf
        self.negative_rate = negative_rate
        self.meta = meta or {}
        self.features = list(FEATURES)

    @property
    def kind(self) -> str:
        return "gradient-boosting" if self.clf is not None else "heuristic"

    @classmethod
    def train(cls, runs: Sequence[Run], seed: int = 7) -> "FailureModel":
        total_rows = sum(1 for run in runs for r in run.results if r.status != "skipped")
        negative_rate = min(1.0, MAX_TRAIN_ROWS / max(total_rows, 1))
        X, y, _ = build_dataset(runs, negative_rate=negative_rate, seed=seed)
        n_fail = int(sum(y))
        meta = {"runs": len(runs), "rows": len(y), "failures": n_fail}
        if n_fail < MIN_FAILURES_TO_TRAIN or n_fail == len(y):
            meta["note"] = f"only {n_fail} failures in history; using heuristic scoring"
            return cls(None, 1.0, meta)

        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=seed,
        )
        clf.fit(np.asarray(X, dtype=np.float32), np.asarray(y))
        return cls(clf, negative_rate, meta)

    def predict(self, X: List[List[float]]) -> np.ndarray:
        if not X:
            return np.zeros(0)
        arr = np.asarray(X, dtype=np.float32)
        if self.clf is None:
            return heuristic_scores(arr)
        p = self.clf.predict_proba(arr)[:, 1]
        if self.negative_rate < 1.0:
            # Undo negative subsampling: true odds = model odds * sampling rate.
            odds = p / np.clip(1 - p, 1e-9, None) * self.negative_rate
            p = odds / (1 + odds)
        return p

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "FailureModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        if model.features != FEATURES:
            raise ValueError("model was trained with a different feature set; re-run `tpredict train`")
        return model


def load_or_heuristic(path: Optional[str]) -> FailureModel:
    if path and Path(path).exists():
        return FailureModel.load(path)
    return FailureModel(None, 1.0, {"note": "no trained model found; using heuristic scoring"})
