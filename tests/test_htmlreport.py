import json
from pathlib import Path

import pytest

from testpredict import htmlreport, synth
from testpredict.cli import main
from testpredict.htmlreport import REPORT_TYPES, SECTIONS, ReportContext, render_report
from testpredict.ingest import build_run
from testpredict.store import Run, Store, TestResult

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    synth.generate(str(out), n_runs=80, seed=5)
    path = out / "h.db"
    with Store(path) as store:
        for d in sorted((out / "runs").iterdir()):
            store.save_run(build_run([str(d / "junit.xml")], **json.loads((d / "meta.json").read_text())))
    return str(path)


@pytest.mark.parametrize("report_type", sorted(REPORT_TYPES))
def test_builtin_types_render(db, report_type):
    page = render_report(ReportContext(db=db), report_type)
    assert page.startswith("<!doctype html>")
    for sid in REPORT_TYPES[report_type]["sections"]:
        assert f'id="{sid}"' in page
    assert "failed to render" not in page


def test_untrusted_text_is_escaped():
    evil = '<script>alert(1)</script>'
    runs = [Run(f"r{i}", f"2026-01-01T00:00:{i:02d}Z", commit_sha="c",
                results=[TestResult(f"x::{evil}", "failed", 1.0, message=evil)]) for i in range(4)]
    page = render_report(ReportContext(runs=runs), "health")
    assert evil not in page
    assert "&lt;script&gt;" in page


def test_custom_type_plugin_and_options(db, tmp_path):
    cfg = htmlreport.load_config(str(EXAMPLES / "report-config.json"))
    htmlreport.load_plugins(cfg["plugins"])
    assert "module-breakdown" in SECTIONS
    page = render_report(ReportContext(db=db, config=cfg), "standup")
    assert "Checkout squad" in page                    # config title overrides type title
    assert 'id="module-breakdown"' in page
    assert "Tests to fix this sprint" in page          # per-section title option
    assert "#7b2cbf" in page                           # accent


def test_section_list_override_and_errors(db):
    page = render_report(ReportContext(db=db), sections=["summary", "slowest"])
    assert 'id="slowest"' in page and 'id="trend"' not in page
    with pytest.raises(ValueError, match="unknown section"):
        render_report(ReportContext(db=db), sections=["nope"])
    with pytest.raises(ValueError, match="unknown report type"):
        render_report(ReportContext(db=db), "nope")


def test_failing_plugin_section_does_not_break_report(db):
    @htmlreport.section("boom", "Boom")
    def boom(ctx, opts):
        raise RuntimeError("kaput")
    try:
        page = render_report(ReportContext(db=db), sections=["summary", "boom"])
        assert "failed to render" in page and 'id="summary"' in page
    finally:
        del SECTIONS["boom"]


def test_cli_report(db, tmp_path, capsys):
    out = tmp_path / "r.html"
    assert main(["report", "--db", db, "--type", "plan", "--budget-fraction", "0.5", "--out", str(out)]) == 0
    assert 'id="plan"' in out.read_text()
    assert main(["report", "--list", "--plugin", str(EXAMPLES / "report_plugins/module_breakdown.py")]) == 0
    assert "module-breakdown" in capsys.readouterr().out
