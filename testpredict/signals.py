"""External risk signals that raise a test's failure probability beyond what history shows.

This is the extension point for the planned RAG component: a PRD/release-notes analyzer
returns {test pattern -> risk} for an upcoming release, and the planner folds it in with
    p_final = 1 - (1 - p_model) * (1 - risk)

Today a signal can be a hand-written JSON file, e.g. risk.json:
    {"tests.payments.*": 0.4, "*checkout*": 0.2}
"""
from __future__ import annotations

import fnmatch
import json
import os
from typing import Dict, Iterable, List, Protocol, Sequence


class RiskSignal(Protocol):
    name: str

    def scores(self, test_ids: Sequence[str], changed_files: Sequence[str]) -> Dict[str, float]:
        """Return extra risk in [0, 1] for any subset of test_ids."""
        ...


class PatternRiskSignal:
    """Glob patterns over test ids -> risk. The RAG layer can emit the same structure."""

    def __init__(self, patterns: Dict[str, float], name: str = "patterns"):
        self.patterns = {p: max(0.0, min(1.0, float(v))) for p, v in patterns.items()}
        self.name = name

    @classmethod
    def from_file(cls, path: str) -> "PatternRiskSignal":
        with open(path) as f:
            return cls(json.load(f), name=os.path.basename(path))

    def scores(self, test_ids: Sequence[str], changed_files: Sequence[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for t in test_ids:
            risk = max((v for p, v in self.patterns.items() if fnmatch.fnmatch(t, p)), default=0.0)
            if risk:
                out[t] = risk
        return out


def combine(probs: Dict[str, float], signals: Iterable[RiskSignal],
            changed_files: Sequence[str]) -> Dict[str, List[str]]:
    """Mutates `probs` in place; returns test_id -> names of signals that raised it."""
    sources: Dict[str, List[str]] = {}
    ids = list(probs)
    for sig in signals:
        for t, risk in sig.scores(ids, changed_files).items():
            if t in probs and risk > 0:
                probs[t] = 1 - (1 - probs[t]) * (1 - risk)
                sources.setdefault(t, []).append(sig.name)
    return sources
