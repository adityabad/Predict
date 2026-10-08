"""Turn failure probabilities + flakiness into a CI execution plan.

Buckets:
  selected    - blocking run, ordered so likely failures surface first (fail fast)
  quarantined - known-flaky tests unrelated to this change; run non-blocking so they can't redden the build
  deferred    - low-risk tests dropped to meet the time budget; run them in the nightly/full run
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .flaky import FlakyReport
from .history import FEATURES, History
from .model import FailureModel
from .signals import RiskSignal, combine

_IDX = {name: i for i, name in enumerate(FEATURES)}
DEFAULT_DURATION = 5.0  # seconds, for tests with no timing history


@dataclass
class PlannedTest:
    test_id: str
    file: Optional[str]
    p_fail: float
    duration: float
    flaky_score: float = 0.0
    bucket: str = "selected"
    reasons: List[str] = field(default_factory=list)

    @property
    def priority(self) -> float:
        # Expected failures found per second of execution.
        return self.p_fail / max(self.duration, 0.5)


@dataclass
class Plan:
    selected: List[PlannedTest]
    quarantined: List[PlannedTest]
    deferred: List[PlannedTest]
    changed_files: List[str]
    model_kind: str
    budget_seconds: Optional[float]

    def summary(self) -> Dict[str, float]:
        def secs(ts):
            return sum(t.duration for t in ts)

        def exp(ts):
            return sum(t.p_fail for t in ts)

        blocking_all = self.selected + self.deferred
        total = secs(blocking_all + self.quarantined)
        return {
            "tests_total": len(blocking_all) + len(self.quarantined),
            "tests_selected": len(self.selected),
            "tests_quarantined": len(self.quarantined),
            "tests_deferred": len(self.deferred),
            "seconds_full_suite": round(total, 1),
            "seconds_selected": round(secs(self.selected), 1),
            "time_saved_pct": round(100 * (1 - secs(self.selected) / total), 1) if total else 0.0,
            "expected_failures_covered_pct": round(100 * exp(self.selected) / exp(blocking_all), 1)
            if exp(blocking_all) else 100.0,
        }

    def to_dict(self) -> dict:
        return {
            "model": self.model_kind,
            "budget_seconds": self.budget_seconds,
            "changed_files": self.changed_files,
            "summary": self.summary(),
            "selected": [asdict(t) for t in self.selected],
            "quarantined": [asdict(t) for t in self.quarantined],
            "deferred": [asdict(t) for t in self.deferred],
        }


def select(
    candidates: List[PlannedTest],
    budget_seconds: Optional[float],
    must_run_p: float,
) -> Tuple[List[PlannedTest], List[PlannedTest]]:
    """Greedy risk-per-second selection. Tests flagged must-run are always kept."""
    ranked = sorted(candidates, key=lambda t: (-t.priority, -t.p_fail, t.test_id))
    if budget_seconds is None:
        return ranked, []
    must = [t for t in ranked if t.p_fail >= must_run_p or "must-run" in t.reasons]
    must_ids = {t.test_id for t in must}
    used = sum(t.duration for t in must)
    chosen = list(must)
    deferred = []
    for t in ranked:
        if t.test_id in must_ids:
            continue
        if used + t.duration <= budget_seconds:
            chosen.append(t)
            used += t.duration
        else:
            deferred.append(t)
    chosen.sort(key=lambda t: (-t.priority, -t.p_fail, t.test_id))
    return chosen, deferred


def build_plan(
    history: History,
    model: FailureModel,
    changed_files: Sequence[str],
    flaky_reports: Iterable[FlakyReport] = (),
    budget_seconds: Optional[float] = None,
    budget_fraction: Optional[float] = None,
    must_run_p: float = 0.2,
    stale_runs: int = 20,
    universe_runs: int = 30,
    quarantine_flaky: bool = True,
    signals: Iterable[RiskSignal] = (),
    extra_tests: Iterable[str] = (),
) -> Plan:
    """`extra_tests`: test ids present in the repo but absent from history (brand-new tests)."""
    ctx = history.change_context(changed_files)
    cutoff = history.run_index - universe_runs
    ids = [t for t, st in history.tests.items() if st.last_seen_run >= cutoff]
    ids += [t for t in extra_tests if t not in history.tests]

    rows, files = [], {}
    for t in ids:
        st = history.tests.get(t)
        files[t] = st.file if st else None
        rows.append(history.features(t, files[t], ctx))
    probs = dict(zip(ids, (float(p) for p in model.predict(rows))))
    raised = combine(probs, signals, changed_files)
    flaky = {r.test_id: r for r in flaky_reports}

    candidates, quarantined = [], []
    for t, row in zip(ids, rows):
        st = history.tests.get(t)
        dur = st.mean_duration if st and st.durations else DEFAULT_DURATION
        pt = PlannedTest(t, files[t], round(probs[t], 4), round(dur, 2))
        if row[_IDX["is_new"]]:
            pt.reasons += ["must-run", "new test"]
        if row[_IDX["test_file_changed"]]:
            pt.reasons += ["must-run", "test file changed"]
        if st and history.run_index - 1 - st.last_seen_run >= stale_runs:
            pt.reasons += ["must-run", "not run recently"]
        if row[_IDX["last_failed"]]:
            pt.reasons.append("failed last run")
        if row[_IDX["assoc_lift"]] >= 0.1:
            pt.reasons.append("historically fails with these changes")
        if t in raised:
            pt.reasons.append("risk signal: " + ", ".join(raised[t]))

        rep = flaky.get(t)
        if rep and rep.category == "flaky":
            pt.flaky_score = rep.score
            related = row[_IDX["test_file_changed"]] or row[_IDX["assoc_lift"]] >= 0.1
            if quarantine_flaky and not related:
                pt.bucket = "quarantined"
                pt.reasons.append("flaky: " + "; ".join(rep.reasons))
                quarantined.append(pt)
                continue
            pt.reasons.append("flaky (kept: related to change)")
        candidates.append(pt)

    if budget_seconds is None and budget_fraction is not None:
        budget_seconds = budget_fraction * sum(t.duration for t in candidates)
    selected, deferred = select(candidates, budget_seconds, must_run_p)
    for t in deferred:
        t.bucket = "deferred"
    quarantined.sort(key=lambda t: -t.flaky_score)
    return Plan(selected, quarantined, deferred, list(changed_files), model.kind, budget_seconds)


# ---- output formats ----------------------------------------------------------------------------

def format_test(t: PlannedTest, fmt: str) -> str:
    """Render a test id for a runner. JUnit ids are `classname::name`."""
    if fmt == "ids" or "::" not in t.test_id:
        return t.test_id
    classname, name = t.test_id.split("::", 1)
    if fmt == "pytest":
        if t.file:
            module = t.file[:-3].replace("/", ".") if t.file.endswith(".py") else ""
            cls = classname[len(module) + 1:] if module and classname.startswith(module + ".") else ""
            return "::".join(p for p in (t.file, cls, name) if p)
        parts = classname.split(".")
        if len(parts) > 1 and parts[-1][:1].isupper():
            return "/".join(parts[:-1]) + ".py::" + parts[-1] + "::" + name
        return "/".join(parts) + ".py::" + name
    if fmt == "maven":
        return classname.rsplit(".", 1)[-1] + "#" + name.split("(")[0]
    raise ValueError(f"unknown format {fmt!r}")


def write_plan(plan: Plan, out_dir: str, fmt: str = "ids") -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for bucket in ("selected", "quarantined", "deferred"):
        p = out / f"{bucket}.txt"
        p.write_text("".join(format_test(t, fmt) + "\n" for t in getattr(plan, bucket)))
        paths[bucket] = str(p)
    p = out / "plan.json"
    p.write_text(json.dumps(plan.to_dict(), indent=2))
    paths["json"] = str(p)
    return paths
