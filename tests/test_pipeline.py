import json

import pytest

from testpredict import flaky, planner, synth
from testpredict.evaluate import backtest
from testpredict.history import History, build_dataset
from testpredict.ingest import build_run
from testpredict.model import FailureModel
from testpredict.signals import PatternRiskSignal
from testpredict.store import Run, Store, TestResult


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    truth = synth.generate(str(out), n_runs=150, seed=3)
    with Store(out / "h.db") as store:
        for d in sorted((out / "runs").iterdir()):
            meta = json.loads((d / "meta.json").read_text())
            store.save_run(build_run([str(d / "junit.xml")], **meta))
        return store.runs(), truth


def _run(i, results, changed=(), sha=None):
    return Run(f"r{i}", f"2026-01-01T00:00:{i:02d}Z", commit_sha=sha or f"s{i}", changed_files=list(changed),
               results=[TestResult(t, s, duration=1.0) for t, s in results])


def test_features_do_not_see_current_run():
    X, y, _ = build_dataset([_run(0, [("a::t", "failed")]), _run(1, [("a::t", "passed")])])
    assert y == [1, 0]
    assert X[0][-1] == 1.0          # first time seen -> is_new
    assert X[1][4] == 1.0           # second row knows only that it failed last time


def test_change_association_learned():
    runs = [_run(i, [("pay::t", "failed" if i % 2 else "passed"), ("cart::t", "passed")],
                 changed=["src/pay/api.py"] if i % 2 else ["src/cart/x.py"]) for i in range(20)]
    h = History.from_runs(runs)
    pay = h.features("pay::t", None, h.change_context(["src/pay/api.py"]))
    other = h.features("pay::t", None, h.change_context(["src/cart/x.py"]))
    assert pay[13] > 0.3 > other[13]


def test_flaky_vs_broken():
    flaky_runs = [_run(i, [("f::t", "flaky" if i % 4 == 0 else "passed")]) for i in range(20)]
    broken_runs = [_run(i, [("b::t", "failed" if i > 15 else "passed")]) for i in range(20)]
    h = History.from_runs(flaky_runs + broken_runs)
    rep = {r.test_id: r for r in flaky.detect(h)}
    assert rep["f::t"].category == "flaky"
    assert rep["b::t"].category == "broken"


def test_end_to_end(runs):
    runs, truth = runs
    model = FailureModel.train(runs)
    assert model.kind == "gradient-boosting"
    hist = History.from_runs(runs)
    reports = flaky.detect(hist)
    found = {r.test_id for r in reports if r.category == "flaky"}
    assert len(found & set(truth["flaky_tests"])) >= 0.7 * len(truth["flaky_tests"])

    plan = planner.build_plan(hist, model, ["src/payments/service.py"], reports, budget_fraction=0.3)
    s = plan.summary()
    assert s["tests_selected"] + s["tests_deferred"] + s["tests_quarantined"] == s["tests_total"]
    assert s["time_saved_pct"] > 50
    pri = [t.priority for t in plan.selected]
    assert pri == sorted(pri, reverse=True)
    assert all(t.bucket == "quarantined" for t in plan.quarantined)


def test_risk_signal_raises_probability(runs):
    runs, _ = runs
    hist = History.from_runs(runs)
    model = FailureModel(None)
    sig = PatternRiskSignal({"tests.search.*": 0.9})
    plan = planner.build_plan(hist, model, [], signals=[sig], quarantine_flaky=False)
    search = [t for t in plan.selected if t.test_id.startswith("tests.search.")]
    assert search and all(t.p_fail >= 0.9 for t in search)


def test_backtest_beats_original_order(runs):
    runs, _ = runs
    m = backtest(runs)
    t = m["time_to_first_failure_pct"]
    assert t["predicted"] < t["original"]
    assert m["roc_auc"] > 0.75


def test_format_test():
    t = planner.PlannedTest("tests.cart.test_cart.TestCart::test_add", "tests/cart/test_cart.py", 0.1, 1)
    assert planner.format_test(t, "pytest") == "tests/cart/test_cart.py::TestCart::test_add"
    t = planner.PlannedTest("com.acme.CartTest::addsItem", None, 0.1, 1)
    assert planner.format_test(t, "maven") == "CartTest#addsItem"
