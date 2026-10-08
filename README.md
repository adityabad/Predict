# testpredict

Predict which automated tests will fail, flag flaky ones, and run the risky tests first. The goal is
shorter CI runs, failures surfaced in the first minutes, and green/red results you can trust.

It learns from the **JUnit XML** your framework already produces (pytest, TestNG/JUnit/Surefire,
Playwright, Cypress, Jest...) plus the list of files changed in each run.

| Capability | What you get |
|---|---|
| **Failure prediction** | P(fail) per test for a given change, from a gradient-boosting model on test history + change set |
| **Fail-fast ordering** | Tests ordered by expected failures per second, so a red build turns red early |
| **Test selection** | Optional time budget: low-risk tests are deferred to the nightly full run |
| **Flaky detection** | Retry-passes, pass+fail on the same commit, flip rate and infra-style errors → `flaky` vs `broken` |
| **Quarantine** | Known-flaky tests unrelated to the change run in a non-blocking job, so they can't redden the build |
| **HTML reports** | Self-contained, offline HTML (charts, sortable tables, dark mode) with pluggable report types |
| **Risk signals** | Hook for external risk (manual hot-spots today, RAG over PRDs later; see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)) |

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
```

See the whole pipeline on synthetic history (400 runs, 200 tests, ~12 seconds):

```bash
.venv/bin/tpredict demo --out demo
```

Results on the synthetic data (backtest: trained on the first 70% of runs, replayed the remaining 30%):

| Metric | Result |
|---|---|
| Time until first failure (original order → predicted order) | 35% → **3%** of run time |
| Failing tests caught with a 35% time budget | **94%** |
| Red builds still detected with a 20% time budget | **98%** |
| Flaky tests found (vs. ground truth) | 16/16, no false positives |

These numbers come from synthetic data. Run `tpredict evaluate` on your own history to get real ones.

## Using it with your suite

```bash
# 1. After every CI run: record results (changed files from git, metadata from GitHub/GitLab env vars)
tpredict ingest reports/ --git-base origin/main

# 2. Periodically (or every main build): retrain
tpredict train

# 3. Before running tests: plan for this change
tpredict plan --git-base origin/main --budget-fraction 0.4 --format pytest --out-dir plan
#    -> plan/selected.txt (run, in this order), plan/quarantined.txt (non-blocking),
#       plan/deferred.txt (nightly), plan/plan.json, plan/plan.md

# Health report
tpredict flaky

# How good are the predictions on your history?
tpredict evaluate
```

## HTML reports

```bash
tpredict report --type health --out report.html        # health | plan | backtest | executive | full
tpredict report --type plan --git-base origin/main --budget-fraction 0.4 --out plan.html
tpredict report --sections summary,trend,slowest       # pick sections ad hoc
tpredict report --list                                  # list report types and sections
```

Each report is a single HTML file with no external dependencies, so it can be uploaded as a CI
artifact, emailed, or opened offline. The demo writes `demo/report.html`.

| Section | Shows |
|---|---|
| `summary` | Last run pass rate, green-build %, test time, broken/flaky counts |
| `trend` / `duration` | Failures and retry-passes per run; total test time per run (hover for values) |
| `health` | Broken and flaky tests with the evidence behind each |
| `top-failing` | Tests with the most hard failures recently |
| `failure-clusters` | Failures grouped by normalized error message (one row ≈ one root cause) |
| `slowest` | Where execution time goes |
| `plan` | Predictive plan for a change: run order, quarantined, deferred |
| `backtest` | Prediction quality on your own history |

### Customizing: your own report types and sections

A JSON config ([examples/report-config.json](examples/report-config.json)) sets branding, defines
new report types, and passes options to individual sections:

```json
{
  "title": "Checkout squad — QA health", "brand": "ACME QA", "accent": "#7b2cbf",
  "plugins": ["report_plugins/module_breakdown.py"],
  "report_types": {
    "standup": {"title": "Daily standup", "sections": ["summary", "trend", "module-breakdown", "health"]}
  },
  "section_options": {"health": {"limit": 10, "title": "Tests to fix this sprint"}, "trend": {"runs": 30}}
}
```

```bash
tpredict report --config examples/report-config.json --type standup --out standup.html
```

**New sections** are plain Python functions registered with a decorator. They get lazy access to
`ctx.runs`, `ctx.history`, `ctx.flaky_reports`, `ctx.plan` and `ctx.metrics`, plus the same helpers the
built-ins use: `html_table`, `line_chart`, `bar_chart`, `stat_tiles`, `note`, `details`. See
[examples/report_plugins/module_breakdown.py](examples/report_plugins/module_breakdown.py):

```python
from testpredict.htmlreport import section, html_table

@section("owners", "Failures by owner")
def owners(ctx, opts):
    rows = ...  # compute from ctx.runs / ctx.flaky_reports
    return html_table(["owner", "failures"], rows)
```

Load plugins via the config's `plugins` list or `--plugin path/or.module`. Registering an existing id
overrides that built-in section. A section that raises an error shows an error note in its card
instead of breaking the whole report.

Ready-made CI templates: [ci/github-actions.yml](ci/github-actions.yml) and [ci/gitlab-ci.yml](ci/gitlab-ci.yml).

**Bootstrapping:** with no history, `plan` falls back to a heuristic. Back-fill by ingesting old
JUnit artifacts with `--run-id`, `--started-at`, `--commit` and `--changed-files`, or by
ingesting each run as CI produces it. The model trains automatically once there are 15 or more
failures in history. Expect useful ordering after about 50 runs.

### Key options for `plan`

| Option | Meaning |
|---|---|
| `--budget-minutes N` / `--budget-fraction F` | Cap the blocking run. Without either, everything runs, just reordered |
| `--must-run-p 0.2` | Tests at or above this P(fail) are never deferred |
| `--risk-file risk.json` | `{"tests.payments.*": 0.4}`: raise risk for areas you know are changing |
| `--all-tests ids.txt` | Current test ids, so brand-new tests (absent from history) are always run |
| `--no-quarantine` | Keep flaky tests in the blocking run |
| `--format ids\|pytest\|maven` | How test ids are written to the `.txt` files |

Safety rails: new tests, tests whose own file changed, and tests not executed in the last 20 runs
are always selected. Flaky tests are only quarantined when the change is unrelated to them.

## Project layout

```
testpredict/
  ingest.py    JUnit XML parsing (incl. Surefire flakyFailure/rerun tags, duplicate retry entries), CI metadata
  store.py     SQLite history (runs, results)
  history.py   chronological replay + 16 per-test features (no future leakage)
  model.py     HistGradientBoosting failure model, heuristic fallback
  flaky.py     flaky / broken classification with reasons
  planner.py   selection, ordering, quarantine; output formats
  signals.py   external risk-signal interface (RAG plugs in here)
  htmlreport.py  HTML report engine: section registry, report types, config/plugins, chart+table helpers
  sections.py  built-in report sections
  evaluate.py  walk-forward backtest
  synth.py     synthetic CI history generator
ci/            GitHub Actions and GitLab CI templates
examples/      report config + example report plugin
tests/         unit + end-to-end tests (pytest)
```
