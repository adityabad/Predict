"""Markdown rendering (works in terminals, PR comments and $GITHUB_STEP_SUMMARY)."""
from __future__ import annotations

from typing import Dict, List

from .flaky import FlakyReport
from .planner import Plan, PlannedTest


def _table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(out)


def flaky_markdown(reports: List[FlakyReport], limit: int = 30) -> str:
    counts: Dict[str, int] = {}
    for r in reports:
        counts[r.category] = counts.get(r.category, 0) + 1
    lines = ["## Test health", "",
             " · ".join(f"**{counts.get(c, 0)}** {c}" for c in ("broken", "flaky", "stable", "insufficient-data")),
             ""]
    for cat, title in (("broken", "Broken (consistently failing — fix or ticket)"),
                       ("flaky", "Flaky (non-deterministic — quarantine and fix)")):
        rows = [r for r in reports if r.category == cat][:limit]
        if rows:
            lines += [f"### {title}", "", _table(
                ["test", "score", "runs", "fail rate", "why"],
                [[f"`{r.test_id}`", f"{r.score:.2f}", r.runs, f"{r.fail_rate:.0%}", "; ".join(r.reasons)]
                 for r in rows]), ""]
    return "\n".join(lines)


def _plan_rows(tests: List[PlannedTest], limit: int) -> List[List[str]]:
    return [[f"`{t.test_id}`", f"{t.p_fail:.0%}", f"{t.duration:.1f}s",
             ", ".join(r for r in t.reasons if r != "must-run")] for t in tests[:limit]]


def plan_markdown(plan: Plan, limit: int = 15) -> str:
    s = plan.summary()
    lines = [
        "## Predictive test plan", "",
        f"Model: `{plan.model_kind}` · changed files: {len(plan.changed_files)}", "",
        f"- **Run:** {s['tests_selected']} tests, ~{s['seconds_selected'] / 60:.1f} min "
        f"(full suite ~{s['seconds_full_suite'] / 60:.1f} min, **{s['time_saved_pct']}% saved**)",
        f"- **Quarantined flaky (non-blocking):** {s['tests_quarantined']}",
        f"- **Deferred to nightly:** {s['tests_deferred']}",
        f"- Expected failures covered by the blocking run: **{s['expected_failures_covered_pct']}%**", "",
        "### Highest-risk tests (run first)", "",
        _table(["test", "P(fail)", "est. time", "why"], _plan_rows(plan.selected, limit)), "",
    ]
    if plan.quarantined:
        lines += ["### Quarantined", "",
                  _table(["test", "P(fail)", "est. time", "why"], _plan_rows(plan.quarantined, limit)), ""]
    return "\n".join(lines)


def evaluation_markdown(m: Dict) -> str:
    lines = ["## Backtest", "",
             f"Model `{m['model']}` trained on {m['train_runs']} runs, evaluated on {m['eval_runs']} later runs "
             f"({m['eval_runs_with_failures']} with failures).", ""]
    if "roc_auc" in m:
        lines += [f"- ROC AUC: **{m['roc_auc']}** · average precision: **{m['average_precision']}** "
                  f"(random ≈ {m['failure_rate']})", ""]
    if "time_to_first_failure_pct" in m:
        t = m["time_to_first_failure_pct"]
        lines += ["**Time until the first failure is reported** (% of full run time, lower is better):", "",
                  _table(["original order", "random order", "predicted order"],
                         [[f"{t['original']}%", f"{t['random']}%", f"{t['predicted']}%"]]), "",
                  "**Trimmed runs** (time budget as % of full suite):", "",
                  _table(["budget", "failing tests caught", "red builds still red", "actual time used"],
                         [[b, f"{v['failing_tests_caught_pct']}%", f"{v['red_builds_still_detected_pct']}%",
                           f"{v['actual_time_used_pct']}%"] for b, v in m["budgets"].items()]), ""]
    return "\n".join(lines)
