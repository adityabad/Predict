"""Flaky-test detection from replayed history.

Evidence, strongest first:
  1. passed only on retry within the same run
  2. both passed and failed on the *same commit* (identical code, different outcome)
  3. frequent pass<->fail flipping
  4. failure messages that look like infrastructure/timing problems
Tests that fail consistently are classified as "broken", not flaky.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import List

from .history import History, TestState

INFRA_PATTERNS = re.compile(
    r"timeout|timed out|stale ?element|element ?click ?intercepted|not ?interactable|"
    r"connection (reset|refused|aborted)|econn|socket|503|502|504|service unavailable|"
    r"deadlock|lock wait|too many requests|429|net::err|target closed|browser has been closed|"
    r"no such window|session not created|resource temporarily unavailable",
    re.IGNORECASE,
)


@dataclass
class FlakyReport:
    test_id: str
    category: str          # flaky | broken | stable | insufficient-data
    score: float
    runs: int
    fail_rate: float
    retry_passes: int
    mixed_commits: int
    flip_rate: float
    infra_ratio: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def assess(test_id: str, st: TestState, min_runs: int = 5, threshold: float = 0.5) -> FlakyReport:
    mixed = sum(1 for m in st.commits.values() if (m & 2 and m & 1) or m & 4)
    fail_commits = sum(1 for m in st.commits.values() if m & 2 or m & 4)
    flip_rate = st.flips / max(st.n - 1, 1)
    infra = sum(1 for m in st.messages if INFRA_PATTERNS.search(m))
    infra_ratio = infra / len(st.messages) if st.messages else 0.0

    evidence = []
    reasons = []
    if st.retries:
        evidence.append(min(1.0, st.retries / 3) * 0.9)
        reasons.append(f"passed on retry {st.retries}x")
    if mixed:
        # One mixed commit can be a single infra blip; repeated ones are strong evidence.
        evidence.append(min(1.0, mixed / 3) * mixed / max(fail_commits, 1) * 0.8)
        reasons.append(f"pass+fail on same commit {mixed}x")
    if st.flips >= 2:
        evidence.append(min(1.0, flip_rate * 3) * 0.6)
        reasons.append(f"flip rate {flip_rate:.0%}")
    if infra_ratio >= 0.5 and st.messages:
        evidence.append(0.3)
        reasons.append(f"{infra_ratio:.0%} infra/timing-style errors")

    score = 1.0
    for e in evidence:
        score *= 1 - e
    score = 1 - score

    if st.streak >= 3:
        reasons.insert(0, f"failing {st.streak} runs in a row")
    if score >= threshold and (st.n >= min_runs or st.retries >= 2):
        category = "flaky"
    elif st.streak >= 3:
        category = "broken"
    elif st.n < min_runs:
        category = "insufficient-data"
    else:
        category = "stable"

    return FlakyReport(test_id, category, round(score, 3), st.n, round(st.fail_rate, 3),
                       st.retries, mixed, round(flip_rate, 3), round(infra_ratio, 3), reasons)


def detect(history: History, min_runs: int = 5, threshold: float = 0.5) -> List[FlakyReport]:
    reports = [assess(t, st, min_runs, threshold) for t, st in history.tests.items()]
    order = {"broken": 0, "flaky": 1, "stable": 2, "insufficient-data": 3}
    reports.sort(key=lambda r: (order[r.category], -r.score, r.test_id))
    return reports
