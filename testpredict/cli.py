"""tpredict command line."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import evaluate, flaky, ingest, planner, report
from .history import History
from .model import FailureModel, load_or_heuristic
from .signals import PatternRiskSignal
from .store import Store

DEFAULT_DB = ".testpredict/history.db"
DEFAULT_MODEL = ".testpredict/model.pkl"


def _changed_files(args) -> List[str]:
    files: List[str] = []
    if getattr(args, "changed_files", None):
        files += ingest.read_changed_files(args.changed_files)
    if getattr(args, "git_base", None):
        files += ingest.changed_files_from_git(args.git_base)
    return sorted(set(files))


def _emit(text: str, out: Optional[str]) -> None:
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n")
    print(text)


def cmd_ingest(args) -> int:
    meta = {}
    if args.meta:
        meta = json.loads(Path(args.meta).read_text())
    run = ingest.build_run(
        args.reports,
        run_id=args.run_id or meta.get("run_id"),
        started_at=args.started_at or meta.get("started_at"),
        branch=args.branch or meta.get("branch"),
        commit_sha=args.commit or meta.get("commit_sha"),
        changed_files=_changed_files(args) or meta.get("changed_files"),
    )
    with Store(args.db) as store:
        store.save_run(run)
        total = store.run_count()
    counts = {}
    for r in run.results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print(f"ingested {run.run_id}: {len(run.results)} tests {counts}; history now {total} runs")
    return 0


def cmd_train(args) -> int:
    with Store(args.db) as store:
        runs = store.runs(args.branch)
    if not runs:
        print("no runs in history; ingest some reports first", file=sys.stderr)
        return 1
    model = FailureModel.train(runs)
    model.save(args.model)
    print(f"trained {model.kind} model on {model.meta} -> {args.model}")
    return 0


def cmd_flaky(args) -> int:
    with Store(args.db) as store:
        hist = History.from_runs(store.iter_runs(args.branch))
    reports = flaky.detect(hist, min_runs=args.min_runs, threshold=args.threshold)
    if args.json:
        _emit(json.dumps([r.to_dict() for r in reports if r.category in ("flaky", "broken")], indent=2), args.out)
    else:
        _emit(report.flaky_markdown(reports), args.out)
    return 0


def cmd_plan(args) -> int:
    with Store(args.db) as store:
        hist = History.from_runs(store.iter_runs(args.branch))
    if not hist.tests and not args.all_tests:
        print("no history; nothing to plan (pass --all-tests to seed the test list)", file=sys.stderr)
        return 1
    model = load_or_heuristic(args.model)
    reports = flaky.detect(hist, threshold=args.flaky_threshold)
    signals = [PatternRiskSignal.from_file(p) for p in args.risk_file or []]
    extra = ingest.read_changed_files(args.all_tests) if args.all_tests else []
    budget = args.budget_minutes * 60 if args.budget_minutes else None
    plan = planner.build_plan(
        hist, model, _changed_files(args), reports,
        budget_seconds=budget, budget_fraction=args.budget_fraction,
        must_run_p=args.must_run_p, quarantine_flaky=not args.no_quarantine,
        signals=signals, extra_tests=extra,
    )
    paths = planner.write_plan(plan, args.out_dir, args.format)
    md = report.plan_markdown(plan)
    (Path(args.out_dir) / "plan.md").write_text(md + "\n")
    print(md)
    print(f"\nwrote {', '.join(paths.values())}")
    return 0


def cmd_evaluate(args) -> int:
    with Store(args.db) as store:
        runs = store.runs(args.branch)
    metrics = evaluate.backtest(runs, train_frac=args.train_frac)
    _emit(json.dumps(metrics, indent=2) if args.json else report.evaluation_markdown(metrics), args.out)
    return 0


def cmd_report(args) -> int:
    from . import htmlreport

    cfg = htmlreport.load_config(args.config)
    htmlreport.load_plugins(list(cfg["plugins"]) + (args.plugin or []))
    from . import sections  # noqa: F401  (register built-ins before listing)

    if args.list:
        print("Report types:")
        for name, spec in sorted(htmlreport.report_types(cfg).items()):
            print(f"  {name:<12} {', '.join(spec['sections'])}")
        print("\nSections:")
        for sid, sec in sorted(htmlreport.SECTIONS.items()):
            print(f"  {sid:<18} {sec.title}")
        return 0

    ctx = htmlreport.ReportContext(
        db=args.db, branch=args.branch, model_path=args.model, changed_files=_changed_files(args),
        budget_seconds=args.budget_minutes * 60 if args.budget_minutes else None,
        budget_fraction=args.budget_fraction, config=cfg,
    )
    if not ctx.runs:
        print("no runs in history; ingest some reports first", file=sys.stderr)
        return 1
    sections_override = [s.strip() for s in args.sections.split(",") if s.strip()] if args.sections else None
    page = htmlreport.render_report(ctx, args.type, sections_override)
    htmlreport.write_report(args.out, page)
    print(f"wrote {args.out}")
    return 0


def cmd_demo(args) -> int:
    from . import synth

    out = Path(args.out)
    db = out / "history.db"
    if db.exists():
        db.unlink()
    print(f"1/5 generating {args.runs} synthetic CI runs in {out}/runs ...")
    truth = synth.generate(str(out), n_runs=args.runs, seed=args.seed)

    print("2/5 ingesting JUnit reports ...")
    with Store(db) as store:
        for run_dir in sorted((out / "runs").iterdir()):
            meta = json.loads((run_dir / "meta.json").read_text())
            run = ingest.build_run([str(run_dir / "junit.xml")], **meta)
            store.save_run(run)
        runs = store.runs()

    print("3/5 backtesting (train on first 70%, replay the rest) ...")
    metrics = evaluate.backtest(runs)

    print("4/5 training on full history + flaky detection ...")
    model = FailureModel.train(runs)
    model.save(out / "model.pkl")
    hist = History.from_runs(runs)
    reports = flaky.detect(hist)
    found = {r.test_id for r in reports if r.category == "flaky"}
    actual = set(truth["flaky_tests"])
    tp = len(found & actual)

    print("5/5 planning for a sample change ...\n")
    change = ["src/payments/service.py", "src/checkout/api.py"]
    plan = planner.build_plan(hist, model, change, reports, budget_fraction=0.35)
    planner.write_plan(plan, str(out / "plan"), "pytest")

    md = "\n\n".join([
        report.evaluation_markdown(metrics),
        "## Flaky detection vs. ground truth\n\n"
        f"Flagged {len(found)} flaky tests; {tp} of {len(actual)} truly flaky tests found "
        f"(precision {tp / max(len(found), 1):.0%}, recall {tp / max(len(actual), 1):.0%}).",
        report.flaky_markdown(reports, limit=10),
        f"Sample change: `{'`, `'.join(change)}` with a 35% time budget\n\n" + report.plan_markdown(plan, limit=10),
    ])
    (out / "report.md").write_text(md + "\n")
    print(md)

    from . import htmlreport
    ctx = htmlreport.ReportContext(runs=runs, changed_files=change, budget_fraction=0.35, plan=plan,
                                   metrics=metrics, history=hist, flaky_reports=reports, model=model)
    htmlreport.write_report(str(out / "report.html"), htmlreport.render_report(ctx, "full"))
    print(f"\nreports: {out / 'report.md'}, {out / 'report.html'} · plan files: {out / 'plan'}/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tpredict", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, model=False):
        sp.add_argument("--db", default=DEFAULT_DB, help=f"history database (default {DEFAULT_DB})")
        sp.add_argument("--branch", help="only use history from this branch")
        if model:
            sp.add_argument("--model", default=DEFAULT_MODEL, help=f"model file (default {DEFAULT_MODEL})")

    def change_args(sp):
        sp.add_argument("--changed-files", help="file listing changed paths, one per line")
        sp.add_argument("--git-base", help="compute changed files with `git diff BASE...HEAD`")

    sp = sub.add_parser("ingest", help="store JUnit XML results of one CI run")
    common(sp)
    change_args(sp)
    sp.add_argument("reports", nargs="+", help="JUnit XML files, directories or globs")
    sp.add_argument("--meta", help="JSON with run_id/started_at/branch/commit_sha/changed_files")
    sp.add_argument("--run-id")
    sp.add_argument("--started-at", help="ISO-8601 timestamp (default: now)")
    sp.add_argument("--commit")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("train", help="train the failure-prediction model")
    common(sp, model=True)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("flaky", help="report flaky and broken tests")
    common(sp)
    sp.add_argument("--threshold", type=float, default=0.5)
    sp.add_argument("--min-runs", type=int, default=5)
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--out", help="also write the report to this file")
    sp.set_defaults(func=cmd_flaky)

    sp = sub.add_parser("plan", help="prioritize / trim tests for a change")
    common(sp, model=True)
    change_args(sp)
    sp.add_argument("--budget-minutes", type=float, help="max blocking-run time")
    sp.add_argument("--budget-fraction", type=float, help="max blocking-run time as a fraction of the suite")
    sp.add_argument("--must-run-p", type=float, default=0.2, help="always run tests at or above this P(fail)")
    sp.add_argument("--flaky-threshold", type=float, default=0.5)
    sp.add_argument("--no-quarantine", action="store_true", help="keep flaky tests in the blocking run")
    sp.add_argument("--risk-file", action="append", help="JSON {test glob: risk} (repeatable)")
    sp.add_argument("--all-tests", help="file listing every current test id, to include brand-new tests")
    sp.add_argument("--format", choices=["ids", "pytest", "maven"], default="ids")
    sp.add_argument("--out-dir", default=".testpredict/plan")
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("evaluate", help="walk-forward backtest on history")
    common(sp)
    sp.add_argument("--train-frac", type=float, default=0.7)
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--out")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("report", help="generate a self-contained HTML report")
    common(sp, model=True)
    change_args(sp)
    sp.add_argument("--type", default="health",
                    help="report type: health, plan, backtest, executive, full, or one defined in --config")
    sp.add_argument("--sections", help="comma-separated section ids (overrides --type)")
    sp.add_argument("--config", help="JSON report config (title, branding, custom types, section options, plugins)")
    sp.add_argument("--plugin", action="append", help="module or .py file that registers extra sections (repeatable)")
    sp.add_argument("--budget-minutes", type=float, help="time budget for the plan section")
    sp.add_argument("--budget-fraction", type=float, help="time budget for the plan section, as a fraction")
    sp.add_argument("--out", default=".testpredict/report.html")
    sp.add_argument("--list", action="store_true", help="list report types and sections, then exit")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("demo", help="generate synthetic history and run the whole pipeline")
    sp.add_argument("--out", default="demo")
    sp.add_argument("--runs", type=int, default=400)
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_demo)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
