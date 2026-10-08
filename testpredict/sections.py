"""Built-in HTML report sections. Each accepts per-section options from the report config."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Dict

from .htmlreport import (ReportContext, Raw, bar_chart, code, details, esc, html_table, line_chart, note,
                         section, stat_tiles)


def _short_date(run) -> str:
    return run.started_at[5:10] if len(run.started_at) >= 10 else run.run_id


def _point_label(run) -> str:
    return f"{run.run_id} · {run.started_at[:16].replace('T', ' ')}"


def _limit(ctx: ReportContext, opts: Dict[str, Any]) -> int:
    return int(opts.get("limit", ctx.config["top_n"]))


@section("summary", "Summary", "Where the suite stands right now.")
def summary(ctx: ReportContext, opts: Dict[str, Any]) -> str:
    runs = ctx.runs
    if not runs:
        return note("No runs in history yet. Ingest JUnit reports with `tpredict ingest`.")
    last = runs[-1]
    executed = [r for r in last.results if r.status != "skipped"]
    failed = sum(r.status == "failed" for r in executed)
    retried = sum(r.status == "flaky" for r in executed)
    pass_rate = 100 * (len(executed) - failed) / len(executed) if executed else 0
    recent = ctx.recent_runs(20)
    minutes = mean(sum(r.duration for r in run.results) for run in recent) / 60
    green = 100 * mean(all(r.status != "failed" for r in run.results) for run in ctx.recent_runs())
    cats = Counter(r.category for r in ctx.flaky_reports)
    return stat_tiles([
        {"label": "Last run pass rate", "value": f"{pass_rate:.1f}%",
         "note": f"{failed} failed · {retried} passed on retry · {last.run_id}",
         **({"status": "critical", "status_label": "red build"} if failed else
            {"status": "good", "status_label": "green build"})},
        {"label": "Green builds", "value": f"{green:.0f}%", "note": f"last {len(ctx.recent_runs())} runs"},
        {"label": "Avg test time per run", "value": f"{minutes:.1f} min", "note": "sum of test durations, last 20 runs"},
        {"label": "Broken tests", "value": cats.get("broken", 0), "note": "failing 3+ runs in a row",
         **({"status": "critical", "status_label": "needs a fix"} if cats.get("broken") else {})},
        {"label": "Flaky tests", "value": cats.get("flaky", 0), "note": f"of {len(ctx.history.tests)} tracked",
         **({"status": "warning", "status_label": "quarantine"} if cats.get("flaky") else {})},
        {"label": "History", "value": f"{len(runs)} runs", "note": f"{runs[0].started_at[:10]} → {last.started_at[:10]}"},
    ])


@section("trend", "Failure trend", "Hard failures vs. tests that only passed on retry, per run.")
def trend(ctx: ReportContext, opts: Dict[str, Any]) -> str:
    runs = ctx.recent_runs(opts.get("runs"))
    failed = [sum(r.status == "failed" for r in run.results) for run in runs]
    retried = [sum(r.status == "flaky" for r in run.results) for run in runs]
    return line_chart([("Failed", failed), ("Passed on retry", retried)],
                      [_short_date(r) for r in runs], f"Tests per run, last {len(runs)} runs",
                      point_labels=[_point_label(r) for r in runs])


@section("duration", "Execution time", "Total test time per run. Rising lines mean the suite is getting slower.")
def duration(ctx: ReportContext, opts: Dict[str, Any]) -> str:
    runs = ctx.recent_runs(opts.get("runs"))
    mins = [sum(r.duration for r in run.results) / 60 for run in runs]
    return line_chart([("Test time", mins)], [_short_date(r) for r in runs],
                      f"Minutes of test time per run, last {len(runs)} runs", unit=" min",
                      point_labels=[_point_label(r) for r in runs])


@section("health", "Flaky & broken tests",
         "Broken = consistently failing (fix or ticket). Flaky = non-deterministic (quarantine and fix).")
def health(ctx: ReportContext, opts: Dict[str, Any]) -> str:
    limit = _limit(ctx, opts)
    out = []
    for cat, heading in (("broken", "Broken"), ("flaky", "Flaky")):
        rows = [r for r in ctx.flaky_reports if r.category == cat][:limit]
        out.append(f"<h3>{heading} ({sum(r.category == cat for r in ctx.flaky_reports)})</h3>")
        out.append(html_table(
            ["test", "score", "runs", "fail rate", "retry passes", "why"],
            [[code(r.test_id), r.score, r.runs, (f"{r.fail_rate:.1%}", r.fail_rate), r.retry_passes,
              "; ".join(r.reasons)] for r in rows],
            empty=f"No {heading.lower()} tests."))
    return "".join(out)


@section("top-failing", "Most-failing tests", "Tests with the most hard failures in recent runs.")
def top_failing(ctx: ReportContext, opts: Dict[str, Any]) -> str:
    runs = ctx.recent_runs(opts.get("runs"))
    fails, execs, last_msg = Counter(), Counter(), {}
    for run in runs:
        for r in run.results:
            if r.status == "skipped":
                continue
            execs[r.test_id] += 1
            if r.status == "failed":
                fails[r.test_id] += 1
                last_msg[r.test_id] = r.message or ""
    cats = {r.test_id: r.category for r in ctx.flaky_reports}
    rows = [[code(t), n, execs[t], (f"{n / execs[t]:.0%}", n / execs[t]), cats.get(t, ""), last_msg[t][:160]]
            for t, n in fails.most_common(_limit(ctx, opts))]
    return html_table(["test", "failures", "runs", "rate", "health", "last error"], rows,
                      empty=f"No failures in the last {len(runs)} runs.")


_SIG_SUBS = [
    (re.compile(r"0x[0-9a-fA-F]+|\b[0-9a-f]{7,40}\b"), "<hex>"),
    (re.compile(r"'[^']*'|\"[^\"]*\""), "'…'"),
    (re.compile(r"\d+(\.\d+)?"), "<n>"),
    (re.compile(r"\s+"), " "),
]


def signature(message: str) -> str:
    """Normalize an error message so the same root cause groups together."""
    s = (message or "(no message)").strip()
    for pat, rep in _SIG_SUBS:
        s = pat.sub(rep, s)
    return s[:140]


@section("failure-clusters", "Recurring errors",
         "Failures grouped by normalized error message: one row is usually one root cause.")
def failure_clusters(ctx: ReportContext, opts: Dict[str, Any]) -> str:
    runs = ctx.recent_runs(opts.get("runs"))
    count, tests, run_ids, last = Counter(), defaultdict(set), defaultdict(set), {}
    for run in runs:
        for r in run.results:
            if r.status in ("failed", "flaky"):
                sig = signature(r.message or "")
                count[sig] += 1
                tests[sig].add(r.test_id)
                run_ids[sig].add(run.run_id)
                last[sig] = run.started_at[:16].replace("T", " ")
    rows = []
    for sig, n in count.most_common(_limit(ctx, opts)):
        sample = sorted(tests[sig])
        examples = Raw("<br>".join(f"<code>{esc(t)}</code>" for t in sample[:3])
                       + (f"<br>+{len(sample) - 3} more" if len(sample) > 3 else ""))
        rows.append([code(sig), n, len(sample), len(run_ids[sig]), last[sig], examples])
    return html_table(["error signature", "occurrences", "tests", "runs", "last seen", "example tests"], rows,
                      empty=f"No failures in the last {len(runs)} runs.")


@section("slowest", "Slowest tests", "Where execution time goes. Good candidates for optimization or parallelization.")
def slowest(ctx: ReportContext, opts: Dict[str, Any]) -> str:
    tests = [(t, st.mean_duration, st) for t, st in ctx.history.tests.items() if st.durations]
    total = sum(d for _, d, _ in tests) or 1
    tests.sort(key=lambda x: -x[1])
    top = tests[:_limit(ctx, opts)]
    widest = top[0][1] if top else 1
    rows = [[code(t), (f"{d:.1f}s", d),
             (Raw(f'<span class="meter" style="width:{60 * d / widest:.0f}px"></span>{100 * d / total:.1f}%'),
              d / total), st.n]
            for t, d, st in top]
    share = 100 * sum(d for _, d, _ in top) / total
    return (f'<p class="desc">The {len(top)} slowest tests take {share:.0f}% of total test time.</p>'
            + html_table(["test", "avg duration", "share of suite", "runs"], rows))


def _plan_rows(tests, limit, numbered=False):
    rows = []
    for i, t in enumerate(tests[:limit], 1):
        reasons = ", ".join(r for r in t.reasons if r != "must-run")
        row = [code(t.test_id), (f"{t.p_fail:.0%}", t.p_fail), (f"{t.duration:.1f}s", t.duration), reasons]
        rows.append(([i] if numbered else []) + row)
    return rows


@section("plan", "Predictive test plan",
         "Blocking tests in fail-fast order; flaky tests quarantined; low-risk tests deferred to the full run.")
def plan(ctx: ReportContext, opts: Dict[str, Any]) -> str:
    p = ctx.plan
    s = p.summary()
    limit = _limit(ctx, opts)
    changed = (Raw(", ".join(f"<code>{esc(f)}</code>" for f in p.changed_files[:12])
                   + (f" +{len(p.changed_files) - 12} more" if len(p.changed_files) > 12 else ""))
               if p.changed_files else "none given (history-only ranking)")
    budget = (f"{p.budget_seconds / 60:.1f} min budget" if p.budget_seconds else "no time budget (reorder only)")
    head = (f'<p class="desc">Changed files: {changed} · {esc(budget)} · model <code>{esc(p.model_kind)}</code></p>')
    tiles = stat_tiles([
        {"label": "Tests to run", "value": s["tests_selected"], "note": f"of {s['tests_total']}"},
        {"label": "Blocking run time", "value": f"{s['seconds_selected'] / 60:.1f} min",
         "note": f"full suite {s['seconds_full_suite'] / 60:.1f} min"},
        {"label": "Time saved", "value": f"{s['time_saved_pct']}%"},
        {"label": "Expected failures covered", "value": f"{s['expected_failures_covered_pct']}%",
         "note": "model estimate"},
        {"label": "Quarantined", "value": s["tests_quarantined"], "note": "flaky, non-blocking"},
        {"label": "Deferred", "value": s["tests_deferred"], "note": "run in nightly"},
    ])
    selected = html_table(["#", "test", "P(fail)", "est. time", "why"], _plan_rows(p.selected, limit, True))
    more = (f'<p class="desc">Showing the first {limit} of {len(p.selected)}; the full list is in plan.json.</p>'
            if len(p.selected) > limit else "")
    return (head + tiles + "<h3>Run first</h3>" + selected + more
            + details(f"Quarantined ({len(p.quarantined)})",
                      html_table(["test", "P(fail)", "est. time", "why"], _plan_rows(p.quarantined, 1000)))
            + details(f"Deferred ({len(p.deferred)})",
                      html_table(["test", "P(fail)", "est. time", "why"], _plan_rows(p.deferred, 1000))))


@section("backtest", "Prediction quality",
         "Walk-forward backtest: trained on older runs, replayed on newer ones the model never saw.")
def backtest(ctx: ReportContext, opts: Dict[str, Any]) -> str:
    m = ctx.metrics
    if m is None:
        return note(f"Backtest unavailable: {ctx.metrics_error}")
    out = [f'<p class="desc">Model <code>{esc(m["model"])}</code> trained on {m["train_runs"]} runs, '
           f'evaluated on {m["eval_runs"]} later runs ({m["eval_runs_with_failures"]} with failures).</p>']
    tiles = []
    if "roc_auc" in m:
        tiles += [{"label": "ROC AUC", "value": m["roc_auc"], "note": "0.5 = random, 1.0 = perfect ranking"},
                  {"label": "Average precision", "value": m["average_precision"],
                   "note": f"random ≈ {m['failure_rate']}"}]
    if "time_to_first_failure_pct" in m:
        t = m["time_to_first_failure_pct"]
        tiles.append({"label": "First failure reported after", "value": f"{t['predicted']}%",
                      "note": f"of run time (was {t['original']}%)"})
    if tiles:
        out.append(stat_tiles(tiles))
    if "time_to_first_failure_pct" in m:
        t = m["time_to_first_failure_pct"]
        out.append(bar_chart([("Original order", t["original"]), ("Random order", t["random"]),
                              ("Predicted order", t["predicted"])],
                             "Time until first failure, % of full run (lower is better)", unit="%",
                             highlight="Predicted order"))
        out.append("<h3>Trimmed runs</h3>")
        out.append(html_table(
            ["time budget", "failing tests caught", "red builds still red", "actual time used"],
            [[b, (f"{v['failing_tests_caught_pct']}%", v["failing_tests_caught_pct"]),
              (f"{v['red_builds_still_detected_pct']}%", v["red_builds_still_detected_pct"]),
              (f"{v['actual_time_used_pct']}%", v["actual_time_used_pct"])] for b, v in m["budgets"].items()],
            filterable=False))
    return "".join(out)
