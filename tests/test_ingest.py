from testpredict.ingest import build_run, parse_reports

JUNIT = """<?xml version="1.0"?>
<testsuites>
  <testsuite name="s">
    <testcase classname="tests.cart.test_cart" name="test_ok" file="tests/cart/test_cart.py" time="1.5"/>
    <testcase classname="tests.cart.test_cart" name="test_bad" time="2">
      <failure message="AssertionError: 1 != 2">trace</failure>
    </testcase>
    <testcase classname="com.acme.CartIT" name="retried" time="3">
      <flakyFailure message="TimeoutException"/>
    </testcase>
    <testcase classname="tests.cart.test_cart" name="test_skip"><skipped/></testcase>
    <testcase classname="tests.cart.test_cart" name="test_err" time="0.1"><error message="boom"/></testcase>
  </testsuite>
</testsuites>
"""

RETRY_DUP = """<testsuite>
  <testcase classname="e2e.login" name="logs in" time="4"><failure message="timeout"/></testcase>
  <testcase classname="e2e.login" name="logs in" time="3"/>
</testsuite>"""


def test_statuses(tmp_path):
    p = tmp_path / "r.xml"
    p.write_text(JUNIT)
    res = {r.test_id: r for r in parse_reports([str(p)])}
    assert res["tests.cart.test_cart::test_ok"].status == "passed"
    assert res["tests.cart.test_cart::test_ok"].file == "tests/cart/test_cart.py"
    assert res["tests.cart.test_cart::test_bad"].status == "failed"
    assert res["tests.cart.test_cart::test_bad"].message == "AssertionError: 1 != 2"
    assert res["com.acme.CartIT::retried"].status == "flaky"
    assert res["com.acme.CartIT::retried"].attempts == 2
    assert res["tests.cart.test_cart::test_skip"].status == "skipped"
    assert res["tests.cart.test_cart::test_err"].status == "failed"
    assert [r.position for r in res.values()] == [0, 1, 2, 3, 4]


def test_duplicate_attempts_merge_to_flaky(tmp_path):
    p = tmp_path / "r.xml"
    p.write_text(RETRY_DUP)
    (r,) = parse_reports([str(tmp_path)])
    assert r.status == "flaky"
    assert r.attempts == 2
    assert r.duration == 7


def test_build_run_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_RUN_ID", "99")
    monkeypatch.setenv("GITHUB_SHA", "abc")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    p = tmp_path / "r.xml"
    p.write_text(JUNIT)
    run = build_run([str(p)], changed_files=["src/cart.py"])
    assert run.run_id == "gh-99-1"
    assert run.commit_sha == "abc" and run.branch == "main"
    assert run.changed_files == ["src/cart.py"]
