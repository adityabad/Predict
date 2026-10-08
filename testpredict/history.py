"""Chronological replay of test history and per-test feature extraction.

Features for run N are computed only from runs 0..N-1, so training never sees the future.
"""
from __future__ import annotations

import math
import os
import random
import re
from collections import Counter, deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from .store import Run

FEATURES = [
    "log_n_runs",        # how much history the test has
    "fail_rate",         # lifetime failure rate
    "fail_rate_5",       # failure rate over last 5 executions
    "fail_rate_20",      # failure rate over last 20 executions
    "last_failed",       # failed in its most recent execution
    "fail_streak",       # consecutive failures up to now (capped, scaled)
    "runs_since_fail",   # executions since last failure (capped, scaled; 1.0 = never failed)
    "flip_rate",         # pass<->fail transitions per execution
    "retry_rate",        # share of executions that only passed on retry
    "log_duration",      # mean duration (log seconds)
    "test_file_changed", # the test's own file is in the change set
    "path_similarity",   # token overlap between test id/path and changed paths
    "assoc_max",         # P(test fails | one of the changed files/dirs changed), smoothed
    "assoc_lift",        # assoc_max minus the test's base failure rate
    "log_n_changed",     # size of the change set
    "is_new",            # never executed before
]

_GENERIC_TOKENS = {
    "test", "tests", "spec", "specs", "py", "java", "js", "ts", "tsx", "jsx", "kt", "rb", "go",
    "src", "main", "lib", "com", "org", "io", "app", "e2e", "it", "unit", "integration",
    "index", "init", "the", "and", "should", "when", "given",
}
_TOKEN_RE = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])|\d+")
_ASSOC_K = 2.0  # smoothing strength toward the test's base failure rate
_ASSOC_MIN_FAILS = 2  # a single coincidental co-failure is not evidence of a link


def tokens(text: str) -> set:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")} - _GENERIC_TOKENS


def test_path_stem(test_id: str, file: Optional[str]) -> str:
    """Best-effort repo path (without extension) of the file that defines a test."""
    if file:
        return os.path.splitext(file.replace("\\", "/"))[0]
    classname = test_id.split("::", 1)[0]
    return classname.replace(".", "/")


def change_keys(path: str) -> List[str]:
    """A changed path contributes its own key plus its parent directories (for generalization).
    Top-level directories (src/, tests/) are too broad to carry signal and are skipped."""
    path = path.replace("\\", "/")
    keys = [path]
    parts = path.split("/")[:-1]
    for i in range(len(parts), 1, -1):
        keys.append("dir:" + "/".join(parts[:i]))
    return keys


class TestState:
    __test__ = False  # not a pytest test class

    __slots__ = ("n", "fails", "retries", "flips", "last_outcome", "recent", "streak",
                 "last_fail_exec", "durations", "fail_with", "commits", "messages",
                 "last_seen_run", "file", "tokens")

    def __init__(self, test_id: str, file: Optional[str]):
        self.n = 0
        self.fails = 0
        self.retries = 0
        self.flips = 0
        self.last_outcome: Optional[int] = None
        self.recent: Deque[int] = deque(maxlen=20)
        self.streak = 0
        self.last_fail_exec: Optional[int] = None
        self.durations: Deque[float] = deque(maxlen=20)
        self.fail_with: Counter = Counter()   # change key -> failures while it was changed
        self.commits: Dict[str, int] = {}     # sha -> bitmask (1 = passed, 2 = failed, 4 = retried)
        self.messages: Deque[str] = deque(maxlen=20)
        self.last_seen_run = -1
        self.file = file
        self.tokens = tokens(test_id) | tokens(file or "")

    @property
    def fail_rate(self) -> float:
        return self.fails / self.n if self.n else 0.0

    @property
    def mean_duration(self) -> float:
        return sum(self.durations) / len(self.durations) if self.durations else 0.0


class History:
    """Replays runs oldest-first and keeps the per-test state needed for features."""

    def __init__(self) -> None:
        self.tests: Dict[str, TestState] = {}
        self.key_changed_runs: Counter = Counter()
        self.run_index = 0

    @classmethod
    def from_runs(cls, runs: Iterable[Run]) -> "History":
        h = cls()
        for run in runs:
            h.update(run)
        return h

    def state(self, test_id: str, file: Optional[str] = None) -> TestState:
        st = self.tests.get(test_id)
        if st is None:
            st = self.tests[test_id] = TestState(test_id, file)
        elif file and not st.file:
            st.file = file
        return st

    # ---- features ---------------------------------------------------------------------------

    def change_context(self, changed_files: Sequence[str]) -> dict:
        keys = set()
        stems = set()
        toks = []
        for f in changed_files:
            keys.update(change_keys(f))
            stems.add(os.path.splitext(f.replace("\\", "/"))[0])
            t = tokens(f)
            if t:
                toks.append(t)
        return {"keys": keys, "stems": stems, "tokens": toks, "n": len(changed_files)}

    def features(self, test_id: str, file: Optional[str], ctx: dict) -> List[float]:
        st = self.tests.get(test_id)
        if st is None:
            st = TestState(test_id, file)
        n = st.n
        base = (st.fails + 0.5) / (n + 1.0)
        recent = list(st.recent)
        r5 = recent[-5:]

        stem = test_path_stem(test_id, file or st.file)
        file_changed = 1.0 if any(stem == s or stem.endswith("/" + s) or s.endswith("/" + stem)
                                  for s in ctx["stems"]) else 0.0

        sim = 0.0
        if st.tokens:
            for t in ctx["tokens"]:
                sim = max(sim, len(st.tokens & t) / len(t))

        assoc = base
        for key in ctx["keys"]:
            hits = st.fail_with.get(key, 0)
            if hits >= _ASSOC_MIN_FAILS:
                seen = self.key_changed_runs[key]
                assoc = max(assoc, (hits + _ASSOC_K * base) / (seen + _ASSOC_K))

        since = (n - st.last_fail_exec) if st.last_fail_exec is not None else None
        return [
            math.log1p(n),
            st.fail_rate,
            sum(r5) / len(r5) if r5 else 0.0,
            sum(recent) / len(recent) if recent else 0.0,
            float(st.last_outcome == 1),
            min(st.streak, 10) / 10.0,
            min(since, 50) / 50.0 if since is not None else 1.0,
            st.flips / max(n - 1, 1),
            st.retries / n if n else 0.0,
            math.log1p(st.mean_duration),
            file_changed,
            sim,
            assoc,
            assoc - base,
            math.log1p(ctx["n"]),
            float(n == 0),
        ]

    # ---- state update -------------------------------------------------------------------------

    def update(self, run: Run) -> None:
        keys = set()
        for f in run.changed_files:
            keys.update(change_keys(f))
        for r in run.results:
            if r.status == "skipped":
                continue
            st = self.state(r.test_id, r.file)
            outcome = 1 if r.status == "failed" else 0
            if st.last_outcome is not None and outcome != st.last_outcome:
                st.flips += 1
            st.last_outcome = outcome
            st.recent.append(outcome)
            st.n += 1
            st.durations.append(r.duration)
            st.last_seen_run = self.run_index
            if r.status == "flaky":
                st.retries += 1
            if outcome:
                st.fails += 1
                st.streak += 1
                st.last_fail_exec = st.n
                for k in keys:
                    st.fail_with[k] += 1
            else:
                st.streak = 0
            if r.status in ("failed", "flaky") and r.message:
                st.messages.append(r.message)
            if run.commit_sha:
                bit = {"passed": 1, "failed": 2, "flaky": 4}[r.status]
                st.commits[run.commit_sha] = st.commits.get(run.commit_sha, 0) | bit
        for k in keys:
            self.key_changed_runs[k] += 1
        self.run_index += 1


def build_dataset(
    runs: Iterable[Run],
    negative_rate: float = 1.0,
    seed: int = 7,
) -> Tuple[List[List[float]], List[int], History]:
    """Feature rows + labels (1 = test failed) for every executed test in every run.

    `negative_rate` < 1 subsamples passing rows (failures are rare) to bound memory on big suites;
    the model corrects its probabilities for this.
    """
    rng = random.Random(seed)
    hist = History()
    X: List[List[float]] = []
    y: List[int] = []
    for run in runs:
        ctx = hist.change_context(run.changed_files)
        for r in run.results:
            if r.status == "skipped":
                continue
            label = 1 if r.status == "failed" else 0
            if label == 0 and negative_rate < 1.0 and rng.random() > negative_rate:
                continue
            X.append(hist.features(r.test_id, r.file, ctx))
            y.append(label)
        hist.update(run)
    return X, y, hist
