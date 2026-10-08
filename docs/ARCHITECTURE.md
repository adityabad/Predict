# Architecture and roadmap

## Data flow

```
CI run ──JUnit XML + changed files──▶ ingest ──▶ SQLite history
                                                   │
                         ┌─────────────────────────┼──────────────────────────┐
                         ▼                         ▼                          ▼
                  History replay            train (GBM)                flaky detection
               (features per test)              │                           │
                         └──────────┬───────────┘                           │
  next change (git diff) ──────────▶ planner ◀── risk signals (RAG later)    │
                                    │        ◀───────────────────────────────┘
                     selected (ordered) / quarantined / deferred
```

## Failure prediction

Each (test, run) row is labelled `1` if the test's final status was `failed`. Features are computed
from **earlier runs only** (see `history.py`):

- **Test history:** lifetime / last-5 / last-20 failure rate, last outcome, failure streak, runs since last failure, flip rate, retry rate, duration
- **Change relevance:** the test's own file changed, token similarity between the test path and changed paths, and a learned *co-failure association*: how often this test failed when this file, or its directory, changed. A link needs at least 2 co-failures, so one coincidence doesn't count.
- **Context:** change-set size, new test

Model: scikit-learn `HistGradientBoostingClassifier`. Passing rows are subsampled on large suites
and the probabilities are corrected afterwards. With fewer than 15 failures in history, a
transparent heuristic is used instead.

**Ordering** sorts by `P(fail) / duration` (expected failures found per second), which minimizes
the time to the first failure. **Selection** keeps must-run tests, then fills the time budget greedily.

## Flaky vs. broken

| Evidence | Weight |
|---|---|
| Passed on retry in the same run | strongest |
| Passed and failed on the **same commit** (repeated) | strong |
| Pass↔fail flip rate | medium |
| Infra/timing error messages (timeouts, stale elements, 5xx, connection resets) | weak |

A test failing three or more runs in a row with no flaky evidence is `broken`: a real
regression that needs a fix or a ticket, not a quarantine.

## Roadmap

### Phase 2: operationalize
- Move history from a cached SQLite file to Postgres when several pipelines write at once
- Failure **clustering**: group failures by normalized stack trace/message so one root cause = one ticket
- Auto-create or update tickets for `broken` tests and long-lived flaky ones; track time-to-fix
- Dashboard: flakiness trend, time saved per week, escaped failures (failures that only showed up in nightly)
- Code-coverage mapping (test → source files) as an extra, more precise change-relevance feature

### Phase 3: RAG over product documents (predict risk for upcoming releases)
The planner already accepts `RiskSignal`s (`signals.py`): anything that returns
`{test id pattern: risk 0..1}` is folded into P(fail) as `1 - (1 - p)(1 - risk)`.

Proposed RAG signal:
1. **Index:** chunk PRDs, release notes, design docs and Jira epics, and embed them in a vector store (pgvector/Chroma). Also index a *test catalog*: test id, docstring, file, module and recent failure clusters.
2. **Map features to tests:** for a release, retrieve the PRD sections that describe the change. An LLM (Claude) maps each requirement to impacted modules and tests from the catalog and returns structured JSON with a risk level and a rationale.
3. **Ground it in history:** combine the LLM's mapping with each area's historical failure rate, so areas that broke before when touched get higher risk.
4. **Gaps:** requirements with no matching tests are reported as *coverage gaps* (candidate new tests). This is often the most valuable output.
5. **Feedback loop:** after the release, compare predicted vs. actual failures per area and use that to weight future RAG signals.

Implementation sketch: a `PrdRiskSignal(RiskSignal)` class in `testpredict/rag/`, run with
`tpredict plan --risk-from-prd docs/prd/feature-x.md`, plus a `tpredict release-risk` report for the
whole release before code lands.
