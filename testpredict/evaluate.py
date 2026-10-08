"""Walk-forward backtest: train on older runs, replay newer runs one at a time and measure
how early failures surface and how many are caught under a time budget."""
from __future__ import annotations

import random
from statistics import mean
from typing import Dict, List, Sequence

from .history import FEATURES, History
from .model import FailureModel
from .planner import PlannedTest, select
from .store import Run

_IDX = {name: i for i, name in enumerate(FEATURES)}


def _time_to_first_failure(order: List[int], failed: List[bool], dur: List[float]) -> float:
    """Fraction of the run's total time spent before the first failure finishes."""
    total = sum(dur) or 1.0
    elapsed = 0.0
    for i in order:
        elapsed += dur[i]
        if failed[i]:
            return elapsed / total
    return 1.0


def backtest(
    runs: Sequence[Run],
    train_frac: float = 0.7,
    budgets: Sequence[float] = (0.2, 0.35, 0.5),
    must_run_p: float = 0.2,
    seed: int = 7,
) -> Dict:
    runs = list(runs)
    split = max(1, int(len(runs) * train_frac))
    if split >= len(runs):
        raise ValueError("need more runs than the training split to backtest")
    model = FailureModel.train(runs[:split], seed=seed)
    hist = History.from_runs(runs[:split])
    rng = random.Random(seed)

    all_p: List[float] = []
    all_y: List[int] = []
    ttff = {"original": [], "random": [], "predicted": []}
    budget_stats = {b: {"test_recall": [], "run_detected": [], "time_used": []} for b in budgets}
    eval_runs = failing_runs = 0

    for run in runs[split:]:
        ctx = hist.change_context(run.changed_files)
        executed = [r for r in run.results if r.status != "skipped"]
        if not executed:
            hist.update(run)
            continue
        eval_runs += 1
        rows = [hist.features(r.test_id, r.file, ctx) for r in executed]
        probs = model.predict(rows)
        failed = [r.status == "failed" for r in executed]
        dur = [r.duration for r in executed]
        all_p.extend(float(p) for p in probs)
        all_y.extend(int(f) for f in failed)

        if any(failed):
            failing_runs += 1
            cands = []
            for i, (r, row) in enumerate(zip(executed, rows)):
                st = hist.tests.get(r.test_id)
                est = st.mean_duration if st and st.durations else 5.0
                pt = PlannedTest(str(i), r.file, float(probs[i]), est)
                if row[_IDX["is_new"]] or row[_IDX["test_file_changed"]]:
                    pt.reasons.append("must-run")
                cands.append(pt)

            ranked, _ = select(cands, None, must_run_p)
            ttff["predicted"].append(_time_to_first_failure([int(t.test_id) for t in ranked], failed, dur))
            original = sorted(range(len(executed)), key=lambda i: executed[i].position)
            ttff["original"].append(_time_to_first_failure(original, failed, dur))
            shuffles = []
            for _ in range(10):
                idx = list(range(len(executed)))
                rng.shuffle(idx)
                shuffles.append(_time_to_first_failure(idx, failed, dur))
            ttff["random"].append(mean(shuffles))

            est_total = sum(t.duration for t in cands)
            n_failed = sum(failed)
            for b in budgets:
                chosen, _ = select(cands, b * est_total, must_run_p)
                hit = sum(failed[int(t.test_id)] for t in chosen)
                s = budget_stats[b]
                s["test_recall"].append(hit / n_failed)
                s["run_detected"].append(float(hit > 0))
                s["time_used"].append(sum(dur[int(t.test_id)] for t in chosen) / (sum(dur) or 1))

        hist.update(run)

    result: Dict = {
        "model": model.kind,
        "model_meta": model.meta,
        "train_runs": split,
        "eval_runs": eval_runs,
        "eval_runs_with_failures": failing_runs,
        "failure_rate": round(mean(all_y), 4) if all_y else 0.0,
    }
    if all_y and 0 < sum(all_y) < len(all_y):
        from sklearn.metrics import average_precision_score, roc_auc_score
        result["roc_auc"] = round(float(roc_auc_score(all_y, all_p)), 3)
        result["average_precision"] = round(float(average_precision_score(all_y, all_p)), 3)
    if failing_runs:
        result["time_to_first_failure_pct"] = {
            k: round(100 * mean(v), 1) for k, v in ttff.items()
        }
        result["budgets"] = {
            f"{int(b * 100)}%": {
                "failing_tests_caught_pct": round(100 * mean(s["test_recall"]), 1),
                "red_builds_still_detected_pct": round(100 * mean(s["run_detected"]), 1),
                "actual_time_used_pct": round(100 * mean(s["time_used"]), 1),
            }
            for b, s in budget_stats.items()
        }
    return result
