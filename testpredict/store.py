"""SQLite history store. One row per CI run, one row per (run, test) result."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    started_at    TEXT NOT NULL,          -- ISO-8601, used for chronological ordering
    branch        TEXT,
    commit_sha    TEXT,
    pipeline      TEXT,
    changed_files TEXT NOT NULL DEFAULT '[]'  -- JSON list of repo paths changed vs. base
);
CREATE TABLE IF NOT EXISTS results (
    run_id    TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    test_id   TEXT NOT NULL,
    file      TEXT,
    status    TEXT NOT NULL,              -- passed | failed | skipped | flaky (failed, then passed on retry)
    duration  REAL NOT NULL DEFAULT 0,
    attempts  INTEGER NOT NULL DEFAULT 1,
    message   TEXT,
    position  INTEGER NOT NULL DEFAULT 0, -- order the test appeared in the report
    PRIMARY KEY (run_id, test_id)
);
CREATE INDEX IF NOT EXISTS idx_results_test ON results(test_id);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);
"""

STATUSES = ("passed", "failed", "skipped", "flaky")


@dataclass
class TestResult:
    __test__ = False  # not a pytest test class

    test_id: str
    status: str
    duration: float = 0.0
    file: Optional[str] = None
    attempts: int = 1
    message: Optional[str] = None
    position: int = 0


@dataclass
class Run:
    run_id: str
    started_at: str
    branch: Optional[str] = None
    commit_sha: Optional[str] = None
    pipeline: Optional[str] = None
    changed_files: List[str] = field(default_factory=list)
    results: List[TestResult] = field(default_factory=list)


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def save_run(self, run: Run) -> None:
        """Insert or replace a run and all of its results."""
        with self.conn:
            self.conn.execute("DELETE FROM runs WHERE run_id = ?", (run.run_id,))
            self.conn.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                (run.run_id, run.started_at, run.branch, run.commit_sha, run.pipeline,
                 json.dumps(sorted(set(run.changed_files)))),
            )
            self.conn.executemany(
                "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(run.run_id, r.test_id, r.file, r.status, r.duration, r.attempts, r.message, r.position)
                 for r in run.results],
            )

    def run_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

    def iter_runs(self, branch: Optional[str] = None) -> Iterator[Run]:
        """Yield runs oldest-first with their results attached."""
        sql = "SELECT run_id, started_at, branch, commit_sha, pipeline, changed_files FROM runs"
        args: tuple = ()
        if branch:
            sql += " WHERE branch = ?"
            args = (branch,)
        sql += " ORDER BY started_at, run_id"
        for row in self.conn.execute(sql, args).fetchall():
            run = Run(row[0], row[1], row[2], row[3], row[4], json.loads(row[5] or "[]"))
            run.results = [
                TestResult(test_id=r[0], file=r[1], status=r[2], duration=r[3], attempts=r[4],
                           message=r[5], position=r[6])
                for r in self.conn.execute(
                    "SELECT test_id, file, status, duration, attempts, message, position "
                    "FROM results WHERE run_id = ? ORDER BY position", (run.run_id,))
            ]
            yield run

    def runs(self, branch: Optional[str] = None) -> List[Run]:
        return list(self.iter_runs(branch))


def save_runs(store: Store, runs: Iterable[Run]) -> int:
    n = 0
    for run in runs:
        store.save_run(run)
        n += 1
    return n
