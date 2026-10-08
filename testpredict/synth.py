"""Synthetic CI history generator for demos and tests.

Models a repo with modules (src/<module>/*.py) and tests that depend on some of those files.
Failures come from: regressions introduced by changing a file a test depends on (persisting
until fixed), bad edits to test files, intrinsically flaky tests (timing/infra errors, often
passing on retry), reruns of the same commit, and rare background noise.
"""
from __future__ import annotations

import json
import math
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Set

MODULES = ["auth", "cart", "checkout", "payments", "search", "profile",
           "inventory", "notifications", "admin", "reports"]
SRC_FILES = ["service", "api", "models", "validators"]
TEST_AREAS = ["flow", "edge_cases", "api_contract", "ui"]
CROSS_DEPS = {"checkout": ["payments", "cart", "inventory"], "cart": ["inventory"],
              "reports": ["admin"], "notifications": ["profile"], "admin": ["auth"]}
INFRA_ERRORS = [
    "TimeoutError: page.click: Timeout 30000ms exceeded waiting for selector",
    "StaleElementReferenceException: stale element reference: element is not attached",
    "ConnectionResetError: [Errno 104] Connection reset by peer",
    "HTTPError: 503 Service Unavailable from staging gateway",
    "ElementClickInterceptedException: element click intercepted",
]


@dataclass
class SynthTest:
    module: str
    area: str
    name: str
    deps: List[str]
    duration: float
    flake_p: float = 0.0
    born: int = 0

    @property
    def file(self) -> str:
        return f"tests/{self.module}/test_{self.area}.py"

    @property
    def classname(self) -> str:
        return f"tests.{self.module}.test_{self.area}"

    @property
    def test_id(self) -> str:
        return f"{self.classname}::{self.name}"


@dataclass
class SynthRepo:
    tests: List[SynthTest] = field(default_factory=list)
    flaky_ids: Set[str] = field(default_factory=set)


def make_repo(rng: random.Random, tests_per_area: int = 5, n_runs: int = 400) -> SynthRepo:
    repo = SynthRepo()
    for m in MODULES:
        own = [f"src/{m}/{f}.py" for f in SRC_FILES]
        for area in TEST_AREAS:
            for i in range(tests_per_area):
                deps = rng.sample(own, k=rng.randint(1, 3))
                for other in CROSS_DEPS.get(m, []):
                    if rng.random() < 0.3:
                        deps.append(f"src/{other}/{rng.choice(SRC_FILES)}.py")
                base = 20.0 if area == "ui" else 3.0
                t = SynthTest(m, area, f"test_{area}_{i:02d}", deps,
                              duration=round(rng.lognormvariate(math.log(base), 0.6), 2))
                if rng.random() < 0.07:
                    t.flake_p = rng.uniform(0.04, 0.25)
                    repo.flaky_ids.add(t.test_id)
                if rng.random() < 0.12:
                    t.born = rng.randint(n_runs // 5, n_runs - 20)
                repo.tests.append(t)
    return repo


def generate(out_dir: str, n_runs: int = 400, seed: int = 42) -> Dict:
    """Write runs/<run_id>/junit.xml + meta.json under out_dir. Returns ground truth."""
    rng = random.Random(seed)
    repo = make_repo(rng, n_runs=n_runs)
    all_src = [f"src/{m}/{f}.py" for m in MODULES for f in SRC_FILES]
    out = Path(out_dir)
    (out / "runs").mkdir(parents=True, exist_ok=True)

    open_bugs: Dict[str, Dict] = {}   # test_id -> {"file": src file, "left": runs remaining}
    t0 = datetime(2026, 6, 1, 8, 0, tzinfo=timezone.utc)
    sha = None
    changed: List[str] = []

    for k in range(n_runs):
        rerun = sha is not None and rng.random() < 0.1
        if not rerun:
            sha = "%040x" % rng.getrandbits(160)
            hot = rng.sample(MODULES, k=rng.choice([1, 1, 1, 2]))
            pool = [f for f in all_src if f.split("/")[1] in hot]
            changed = rng.sample(pool, k=min(len(pool), rng.randint(1, 4)))
            # Bugs are fixed by a new commit that touches the file that introduced them.
            for tid in [t for t, b in open_bugs.items() if b["left"] <= 0]:
                changed.append(open_bugs.pop(tid)["file"])
            alive = [t for t in repo.tests if t.born <= k]
            for t in alive:
                if t.born == k and k > 0:
                    changed.append(t.file)                       # new test added in this change
                elif rng.random() < 0.004:
                    changed.append(t.file)                       # edit to an existing test file
            changed = sorted(set(changed))
            # Each changed source file may introduce a regression in tests depending on it.
            for f in changed:
                if f.startswith("src/") and rng.random() < 0.1:
                    victims = [t for t in alive if f in t.deps]
                    for t in rng.sample(victims, k=min(len(victims), rng.randint(1, 4))):
                        open_bugs[t.test_id] = {"file": f, "left": rng.choice([1, 1, 2, 3])}

        run_id = f"run-{k:04d}"
        started = t0 + timedelta(hours=6 * k + rng.randint(0, 5))
        suite = ET.Element("testsuite", name="synthetic")
        for t in repo.tests:
            if t.born > k:
                continue
            case = ET.SubElement(suite, "testcase", classname=t.classname, name=t.name, file=t.file)
            dur = round(t.duration * rng.uniform(0.8, 1.25), 2)
            bug = open_bugs.get(t.test_id)
            if bug:
                ET.SubElement(case, "failure", message=f"AssertionError: expected order total to match "
                                                        f"(regression in {bug['file']})")
            elif t.file in changed and not rerun and t.born != k and rng.random() < 0.15:
                ET.SubElement(case, "failure", message="AssertionError: updated fixture mismatch")
            elif t.flake_p and rng.random() < t.flake_p:
                msg = rng.choice(INFRA_ERRORS)
                if rng.random() < 0.85:
                    ET.SubElement(case, "flakyFailure", message=msg)   # passed on retry
                    dur *= 2
                else:
                    ET.SubElement(case, "failure", message=msg)
            elif rng.random() < 0.001:
                ET.SubElement(case, "failure", message="AssertionError: unexpected value")
            case.set("time", f"{dur:.2f}")
        run_dir = out / "runs" / run_id
        run_dir.mkdir(exist_ok=True)
        root = ET.Element("testsuites", name=run_id)
        root.append(suite)
        ET.ElementTree(root).write(run_dir / "junit.xml", encoding="utf-8", xml_declaration=True)
        (run_dir / "meta.json").write_text(json.dumps({
            "run_id": run_id, "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "branch": "main", "commit_sha": sha, "pipeline": "synthetic",
            "changed_files": changed,
        }))
        if rerun:
            continue  # same code, so the same bugs are still there
        for bug in open_bugs.values():
            bug["left"] -= 1

    truth = {"flaky_tests": sorted(repo.flaky_ids), "n_runs": n_runs, "n_tests": len(repo.tests)}
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    return truth
