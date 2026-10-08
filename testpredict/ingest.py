"""Parse JUnit XML reports (pytest, Surefire/TestNG, Playwright, Cypress, Jest...) into a Run."""
from __future__ import annotations

import glob
import os
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .store import Run, TestResult

# Surefire/Failsafe emit these when a test failed and then passed on rerun.
RETRY_TAGS = {"flakyFailure", "flakyError", "rerunFailure", "rerunError"}
FAIL_TAGS = {"failure", "error"}
MAX_MESSAGE = 500


def expand_paths(paths: Iterable[str]) -> List[Path]:
    """Accept files, directories (searched recursively for *.xml) and glob patterns."""
    out: List[Path] = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(sorted(Path(p).rglob("*.xml")))
        else:
            matches = glob.glob(p, recursive=True)
            out.extend(Path(m) for m in sorted(matches) if m.endswith(".xml"))
    return out


def _message(el: ET.Element) -> str:
    msg = el.get("message") or (el.text or "")
    return " ".join(msg.split())[:MAX_MESSAGE]


def _test_id(case: ET.Element) -> str:
    classname = case.get("classname") or ""
    name = case.get("name") or ""
    return f"{classname}::{name}" if classname else name


def parse_testcase(case: ET.Element) -> TestResult:
    tags = [child.tag for child in case]
    fails = [c for c in case if c.tag in FAIL_TAGS]
    retries = [c for c in case if c.tag in RETRY_TAGS]
    if fails:
        status, message = "failed", _message(fails[0])
    elif retries:
        status, message = "flaky", _message(retries[0])
    elif "skipped" in tags:
        status, message = "skipped", None
    else:
        status, message = "passed", None
    try:
        duration = float(case.get("time") or 0)
    except ValueError:
        duration = 0.0
    return TestResult(
        test_id=_test_id(case),
        status=status,
        duration=duration,
        file=case.get("file"),
        attempts=1 + len(retries),
        message=message,
    )


def _merge(a: TestResult, b: TestResult) -> TestResult:
    """Same test reported twice in one run (e.g. a framework that lists every retry attempt)."""
    statuses = {a.status, b.status}
    if "passed" in statuses and ("failed" in statuses or "flaky" in statuses):
        status = "flaky"
    elif "failed" in statuses:
        status = "failed"
    elif "flaky" in statuses:
        status = "flaky"
    else:
        status = b.status
    return TestResult(
        test_id=a.test_id, status=status, duration=a.duration + b.duration,
        file=a.file or b.file, attempts=a.attempts + b.attempts,
        message=a.message or b.message, position=a.position,
    )


def parse_reports(paths: Iterable[str]) -> List[TestResult]:
    results: Dict[str, TestResult] = {}
    position = 0
    for path in expand_paths(paths):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as e:
            raise ValueError(f"{path}: not valid XML ({e})") from e
        for case in root.iter("testcase"):
            r = parse_testcase(case)
            if r.test_id in results:
                results[r.test_id] = _merge(results[r.test_id], r)
            else:
                r.position = position
                position += 1
                results[r.test_id] = r
    return list(results.values())


def ci_metadata() -> Dict[str, Optional[str]]:
    """Read run metadata from GitHub Actions or GitLab CI environment variables."""
    env = os.environ
    if env.get("GITHUB_ACTIONS"):
        return {
            "run_id": f"gh-{env.get('GITHUB_RUN_ID')}-{env.get('GITHUB_RUN_ATTEMPT', '1')}",
            "branch": env.get("GITHUB_HEAD_REF") or env.get("GITHUB_REF_NAME"),
            "commit_sha": env.get("GITHUB_SHA"),
            "pipeline": env.get("GITHUB_WORKFLOW"),
        }
    if env.get("GITLAB_CI"):
        return {
            "run_id": f"gl-{env.get('CI_PIPELINE_ID')}-{env.get('CI_JOB_ID')}",
            "branch": env.get("CI_COMMIT_REF_NAME"),
            "commit_sha": env.get("CI_COMMIT_SHA"),
            "pipeline": env.get("CI_PROJECT_PATH"),
        }
    return {}


def changed_files_from_git(base: str, cwd: Optional[str] = None) -> List[str]:
    """Files changed between `base` and HEAD (merge-base diff, like a PR diff)."""
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        cwd=cwd, capture_output=True, text=True, check=True,
    ).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def read_changed_files(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_run(
    report_paths: Iterable[str],
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
    branch: Optional[str] = None,
    commit_sha: Optional[str] = None,
    pipeline: Optional[str] = None,
    changed_files: Optional[List[str]] = None,
) -> Run:
    meta = ci_metadata()
    results = parse_reports(report_paths)
    if not results:
        raise ValueError("no <testcase> elements found in the given reports")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return Run(
        run_id=run_id or meta.get("run_id") or f"local-{now}",
        started_at=started_at or now,
        branch=branch or meta.get("branch"),
        commit_sha=commit_sha or meta.get("commit_sha"),
        pipeline=pipeline or meta.get("pipeline"),
        changed_files=changed_files or [],
        results=results,
    )
