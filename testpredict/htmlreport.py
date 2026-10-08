"""Self-contained HTML reports built from pluggable sections.

A *section* is a function `(ctx, options) -> HTML fragment` registered under an id.
A *report type* is a named, ordered list of section ids.

Customize without touching this package:
  - a JSON config (`tpredict report --config report.json`) can add or override report types,
    set the title/branding, and pass per-section options;
  - plugins (Python modules or .py files listed in the config or via `--plugin`) can register
    new sections with the `@section(...)` decorator and reuse the helpers below.

Example plugin (my_sections.py):

    from testpredict.htmlreport import section, html_table

    @section("owners", "Failures by owner")
    def owners(ctx, opts):
        rows = [...]  # anything computed from ctx.runs / ctx.history / ctx.flaky_reports
        return html_table(["owner", "failures"], rows)
"""
from __future__ import annotations

import copy
import html
import importlib
import importlib.util
import json
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .history import History
from .store import Run, Store

# ---- registry ----------------------------------------------------------------------------------

SectionFn = Callable[["ReportContext", Dict[str, Any]], str]


@dataclass
class Section:
    id: str
    title: str
    render: SectionFn
    description: str = ""


SECTIONS: Dict[str, Section] = {}


def section(id: str, title: str, description: str = "") -> Callable[[SectionFn], SectionFn]:
    """Register a report section. Re-registering an id replaces it (lets plugins override built-ins)."""
    def deco(fn: SectionFn) -> SectionFn:
        SECTIONS[id] = Section(id, title, fn, description)
        return fn
    return deco


REPORT_TYPES: Dict[str, Dict[str, Any]] = {
    "health": {
        "title": "Test health report",
        "description": "Flaky and broken tests, failure trends, recurring errors and slow tests.",
        "sections": ["summary", "trend", "duration", "health", "top-failing", "failure-clusters", "slowest"],
    },
    "plan": {
        "title": "Predictive test plan",
        "description": "Which tests to run, in what order, and what was quarantined or deferred.",
        "sections": ["summary", "plan"],
    },
    "backtest": {
        "title": "Prediction quality",
        "description": "How well the model would have done on recent history.",
        "sections": ["backtest"],
    },
    "executive": {
        "title": "Test automation overview",
        "description": "Headline numbers and trends for stakeholders.",
        "sections": ["summary", "trend", "backtest"],
    },
    "full": {
        "title": "Test intelligence report",
        "description": "Everything: health, trends, plan and prediction quality.",
        "sections": ["summary", "trend", "duration", "health", "top-failing", "failure-clusters",
                     "slowest", "plan", "backtest"],
    },
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "title": None,            # overrides the report type's title
    "subtitle": None,
    "brand": "testpredict",   # small label in the header
    "accent": None,           # CSS color for header rule and links, e.g. "#7b2cbf"
    "trend_runs": 60,         # runs shown in trend charts and recent-window tables
    "top_n": 20,              # default row limit for tables
    "budget_fraction": None,  # plan section budget when not given on the command line
    "plugins": [],
    "report_types": {},       # {"name": {"title": ..., "sections": [...]}}
    "section_options": {},    # {"section-id": {...}} passed to that section
}


def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path) as f:
            user = json.load(f)
        unknown = set(user) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
        cfg.update(user)
        # Relative plugin paths are relative to the config file.
        base = os.path.dirname(os.path.abspath(path))
        cfg["plugins"] = [os.path.join(base, p) if p.endswith(".py") and not os.path.isabs(p) else p
                          for p in cfg["plugins"]]
    return cfg


def load_plugins(plugins: Sequence[str]) -> None:
    for p in plugins:
        if p.endswith(".py") or os.path.exists(p):
            name = "testpredict_plugin_" + Path(p).stem
            spec = importlib.util.spec_from_file_location(name, p)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load plugin file {p}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        else:
            importlib.import_module(p)


def report_types(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    types = copy.deepcopy(REPORT_TYPES)
    for name, spec in cfg.get("report_types", {}).items():
        if "sections" not in spec:
            raise ValueError(f"report type {name!r} needs a 'sections' list")
        types[name] = {"title": spec.get("title", name), "description": spec.get("description", ""),
                       "sections": list(spec["sections"])}
    return types


# ---- data context ------------------------------------------------------------------------------

class ReportContext:
    """Lazily computed data shared by sections, so each expensive step runs at most once."""

    def __init__(
        self,
        db: Optional[str] = None,
        runs: Optional[List[Run]] = None,
        branch: Optional[str] = None,
        model_path: Optional[str] = None,
        changed_files: Optional[List[str]] = None,
        budget_seconds: Optional[float] = None,
        budget_fraction: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
        plan: Any = None,
        metrics: Optional[Dict] = None,
        history: Optional[History] = None,
        flaky_reports: Any = None,
        model: Any = None,
    ):
        """Anything already computed (history, model, plan, metrics...) can be passed in to skip recomputation."""
        self.db = db
        self.branch = branch
        self.model_path = model_path
        self.changed_files = changed_files or []
        self.budget_seconds = budget_seconds
        self.config = config or copy.deepcopy(DEFAULT_CONFIG)
        self.budget_fraction = budget_fraction if budget_fraction is not None else self.config.get("budget_fraction")
        self._runs = runs
        self._history = history
        self._flaky = flaky_reports
        self._model = model
        self._plan = plan
        self._metrics = metrics
        self._metrics_error: Optional[str] = None

    @property
    def runs(self) -> List[Run]:
        if self._runs is None:
            if not self.db:
                raise ValueError("ReportContext needs either runs or a db path")
            with Store(self.db) as store:
                self._runs = store.runs(self.branch)
        return self._runs

    def recent_runs(self, n: Optional[int] = None) -> List[Run]:
        n = n or self.config["trend_runs"]
        return self.runs[-n:]

    @property
    def history(self) -> History:
        if self._history is None:
            self._history = History.from_runs(self.runs)
        return self._history

    @property
    def flaky_reports(self):
        if self._flaky is None:
            from . import flaky
            self._flaky = flaky.detect(self.history)
        return self._flaky

    @property
    def model(self):
        if self._model is None:
            from .model import load_or_heuristic
            self._model = load_or_heuristic(self.model_path)
        return self._model

    @property
    def plan(self):
        if self._plan is None:
            from .planner import build_plan
            self._plan = build_plan(self.history, self.model, self.changed_files, self.flaky_reports,
                                    budget_seconds=self.budget_seconds, budget_fraction=self.budget_fraction)
        return self._plan

    @property
    def metrics(self) -> Optional[Dict]:
        if self._metrics is None and self._metrics_error is None:
            from .evaluate import backtest
            try:
                self._metrics = backtest(self.runs)
            except ValueError as e:
                self._metrics_error = str(e)
        return self._metrics

    @property
    def metrics_error(self) -> Optional[str]:
        return self._metrics_error


# ---- HTML helpers (public: use them in plugins) ---------------------------------------------------

def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


class Raw(str):
    """A table cell / fragment that is already HTML and must not be escaped."""


Cell = Union[str, int, float, Raw, Tuple[Any, Any], None]


def _cell(c: Cell) -> str:
    if isinstance(c, tuple):              # (display, sort value)
        display, sort = c
        inner = display if isinstance(display, Raw) else esc(display)
        num = ' class="num"' if isinstance(sort, (int, float)) and not isinstance(display, Raw) else ""
        return f'<td{num} data-v="{esc(sort)}">{inner}</td>'
    if isinstance(c, Raw):
        return f"<td>{c}</td>"
    if isinstance(c, (int, float)) and not isinstance(c, bool):
        return f'<td class="num" data-v="{c}">{esc(c)}</td>'
    return f"<td>{esc(c)}</td>"


def html_table(headers: Sequence[str], rows: Sequence[Sequence[Cell]], filterable: Optional[bool] = None,
               empty: str = "Nothing to show.", caption: str = "") -> str:
    """Sortable table (click a header). Rows over 10 get a filter box unless filterable=False."""
    if not rows:
        return note(empty)
    if filterable is None:
        filterable = len(rows) > 10
    head = "".join(f'<th scope="col" tabindex="0">{esc(h)}</th>' for h in headers)
    body = "".join("<tr>" + "".join(_cell(c) for c in r) + "</tr>" for r in rows)
    flt = ('<input class="filter" type="search" placeholder="Filter rows…" aria-label="Filter rows">'
           if filterable else "")
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return (f'<div class="table-wrap">{flt}<div class="scroll"><table class="dt">{cap}'
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></div>")


def note(text: str, kind: str = "info") -> str:
    return f'<p class="note note-{esc(kind)}">{esc(text)}</p>'


def code(text: Any) -> Raw:
    return Raw(f"<code>{esc(text)}</code>")


STATUS_ICONS = {"good": "✓", "warning": "!", "serious": "▲", "critical": "✕"}


def stat_tiles(tiles: Sequence[Dict[str, Any]]) -> str:
    """tiles: [{"label", "value", "note"?, "status"? (good|warning|serious|critical)}]"""
    out = []
    for t in tiles:
        status = t.get("status")
        badge = (f'<span class="status status-{esc(status)}"><span aria-hidden="true">'
                 f'{STATUS_ICONS.get(status, "")}</span> {esc(t.get("status_label", status))}</span>'
                 if status else "")
        sub = f'<div class="tile-note">{esc(t["note"])}</div>' if t.get("note") else ""
        out.append(f'<div class="tile"><div class="tile-label">{esc(t["label"])}</div>'
                   f'<div class="tile-value">{esc(t["value"])}</div>{badge}{sub}</div>')
    return f'<div class="tiles">{"".join(out)}</div>'


def details(summary: str, body: str, open: bool = False) -> str:
    return f'<details{" open" if open else ""}><summary>{esc(summary)}</summary>{body}</details>'


def _nice_max(v: float) -> Tuple[float, float]:
    """A round axis maximum and tick step for values up to v."""
    if v <= 0:
        return 1.0, 0.25
    import math
    raw = v / 4
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if step >= raw:
            break
    return step * math.ceil(v / step), step


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:,.1f}"


def _json_attr(data: Any) -> str:
    return esc(json.dumps(data, separators=(",", ":")))


def line_chart(series: Sequence[Tuple[str, Sequence[float]]], x_labels: Sequence[str],
               title: str, unit: str = "", height: int = 220,
               point_labels: Optional[Sequence[str]] = None) -> str:
    """Up to 4 series on one axis, with legend, direct end labels, crosshair tooltip and a data table.
    `point_labels` (default: x_labels) names each point in the tooltip and data table."""
    series = list(series)[:4]
    n = len(x_labels)
    if n == 0 or not series:
        return note("Not enough runs to chart yet.")
    W, H = 760, height
    ml, mr, mt, mb = 44, 96, 12, 26
    pw, ph = W - ml - mr, H - mt - mb
    ymax, step = _nice_max(max((max(v) for _, v in series if len(v)), default=0))

    def x(i: int) -> float:
        return ml + (pw * i / (n - 1) if n > 1 else pw / 2)

    def y(v: float) -> float:
        return mt + ph - ph * v / ymax

    parts = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">']
    t = 0.0
    while t <= ymax + 1e-9:
        parts.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
                     f'<text class="tick" x="{ml - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{_fmt(t)}</text>')
        t += step
    parts.append(f'<line class="axis" x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}"/>')
    for i in sorted({0, n - 1, *range(0, n, max(1, n // 6))}):
        if i < n - 1 and n - 1 - i < max(1, n // 12):
            continue  # avoid overlapping the last label
        parts.append(f'<text class="tick" x="{x(i):.1f}" y="{H - 6}" text-anchor="middle">{esc(x_labels[i])}</text>')

    ends = []
    for k, (name, vals) in enumerate(series):
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
        parts.append(f'<polyline class="line s{k + 1}" points="{pts}"/>')
        ends.append([y(vals[-1]), name, vals[-1]])
    # Direct end labels, nudged apart so they never collide.
    ends.sort()
    for j in range(1, len(ends)):
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + 13)
    for ly, name, v in ends:
        parts.append(f'<text class="end-label" x="{ml + pw + 8}" y="{ly + 4:.1f}">{esc(name)} {_fmt(v)}{esc(unit)}</text>')

    parts.append(f'<line class="xh" x1="0" x2="0" y1="{mt}" y2="{mt + ph}" visibility="hidden"/>')
    for k in range(len(series)):
        parts.append(f'<circle class="dot s{k + 1}" r="4" cx="0" cy="0" visibility="hidden"/>')
    for i in range(n):
        x0 = (x(i - 1) + x(i)) / 2 if i else ml
        x1 = (x(i) + x(i + 1)) / 2 if i < n - 1 else ml + pw
        parts.append(f'<rect class="hit" x="{x0:.1f}" y="{mt}" width="{max(x1 - x0, 1):.1f}" height="{ph}" data-i="{i}"/>')
    parts.append("</svg>")

    point_labels = list(point_labels or x_labels)
    data = {"x": point_labels, "unit": unit,
            "series": [{"name": nm, "values": [round(float(v), 3) for v in vals],
                        "y": [round(y(v), 1) for v in vals]} for nm, vals in series],
            "px": [round(x(i), 1) for i in range(n)]}
    legend = ""
    if len(series) > 1:
        legend = '<div class="legend">' + "".join(
            f'<span class="key"><span class="key-line s{k + 1}"></span>{esc(nm)}</span>'
            for k, (nm, _) in enumerate(series)) + "</div>"
    table = html_table(["run"] + [nm for nm, _ in series],
                       [[point_labels[i]] + [round(float(v[i]), 2) for _, v in series] for i in range(n)],
                       filterable=False)
    return (f'<figure class="viz" data-chart="{_json_attr(data)}"><figcaption>{esc(title)}</figcaption>'
            f'{legend}<div class="plot"><div class="plot-scroll">{"".join(parts)}</div><div class="tip" role="status" hidden></div></div>'
            f'{details("Show data table", table)}</figure>')


def bar_chart(bars: Sequence[Tuple[str, float]], title: str, unit: str = "", highlight: Optional[str] = None) -> str:
    """Horizontal bars, one series. `highlight` names the bar to emphasize (the rest are muted)."""
    bars = list(bars)
    if not bars:
        return note("Nothing to chart.")
    W, row, gap = 760, 22, 10
    ml, mr = 150, 70
    H = len(bars) * (row + gap) + gap
    vmax = max(v for _, v in bars) or 1
    pw = W - ml - mr
    parts = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">',
             f'<line class="axis" x1="{ml}" x2="{ml}" y1="0" y2="{H}"/>']
    for i, (label, v) in enumerate(bars):
        top = gap + i * (row + gap)
        w = max(pw * v / vmax, 2)
        r = min(4, w / 2)
        cls = "bar s1" if highlight is None or label == highlight else "bar muted"
        d = (f"M{ml},{top} H{ml + w - r} Q{ml + w},{top} {ml + w},{top + r} V{top + row - r} "
             f"Q{ml + w},{top + row} {ml + w - r},{top + row} H{ml} Z")
        tip = f"{label}: {_fmt(v)}{unit}"
        parts.append(f'<text class="cat" x="{ml - 8}" y="{top + row / 2 + 4}" text-anchor="end">{esc(label)}</text>'
                     f'<path class="{cls}" d="{d}"/>'
                     f'<text class="val" x="{ml + w + 6}" y="{top + row / 2 + 4}">{_fmt(v)}{esc(unit)}</text>'
                     f'<rect class="hit bar-hit" x="{ml}" y="{top - gap / 2}" width="{pw + mr}" height="{row + gap}" '
                     f'data-tip="{esc(tip)}" tabindex="0"/>')
    parts.append("</svg>")
    return (f'<figure class="viz"><figcaption>{esc(title)}</figcaption>'
            f'<div class="plot"><div class="plot-scroll">{"".join(parts)}</div><div class="tip" role="status" hidden></div></div></figure>')


# ---- page ----------------------------------------------------------------------------------------

_CSS = """
:root{color-scheme:light;
 --page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;
 --grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--wash:rgba(11,11,11,.04);
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;
 --good:#0ca30c;--good-ink:#006300;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b;
 --accent:ACCENT_LIGHT}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;
 --page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink-2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--wash:rgba(255,255,255,.05);
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--good-ink:#0ca30c;--accent:ACCENT_DARK}}
:root[data-theme="dark"]{color-scheme:dark;
 --page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink-2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--wash:rgba(255,255,255,.05);
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--good-ink:#0ca30c;--accent:ACCENT_DARK}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--accent)}
header.top{border-top:4px solid var(--accent);background:var(--surface);box-shadow:0 1px 0 var(--ring)}
.wrap{max-width:1120px;margin:0 auto;padding:0 16px}
header .wrap{padding-top:20px;padding-bottom:16px;display:flex;gap:16px;align-items:flex-start;justify-content:space-between}
.brand{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2)}
h1{font-size:24px;margin:2px 0 4px;line-height:1.25}
.meta{color:var(--ink-2);font-size:13px}
button.theme{font:inherit;font-size:13px;color:var(--ink-2);background:transparent;border:1px solid var(--ring);border-radius:6px;padding:4px 10px;cursor:pointer;white-space:nowrap}
nav.toc{position:sticky;top:0;z-index:5;background:var(--page);border-bottom:1px solid var(--ring)}
nav.toc .wrap{display:flex;gap:4px;overflow-x:auto;padding-top:8px;padding-bottom:8px}
nav.toc a{color:var(--ink-2);text-decoration:none;font-size:13px;padding:4px 10px;border-radius:999px;white-space:nowrap}
nav.toc a:hover{background:var(--wash);color:var(--ink)}
main .wrap{padding-top:20px;padding-bottom:48px}
section.card{background:var(--surface);border-radius:10px;box-shadow:0 0 0 1px var(--ring);padding:20px;margin-bottom:20px;scroll-margin-top:56px}
section.card>h2{font-size:17px;margin:0 0 2px}
.desc{color:var(--ink-2);margin:0 0 14px;font-size:13px}
h3{font-size:14px;margin:18px 0 8px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:4px 0 8px}
.tile{padding:12px 14px;border-radius:8px;box-shadow:0 0 0 1px var(--ring)}
.tile-label{font-size:12px;color:var(--ink-2)}
.tile-value{font-size:26px;font-weight:600;line-height:1.2;margin-top:2px}
.tile-note{font-size:12px;color:var(--muted);margin-top:2px}
.status{display:inline-flex;align-items:center;gap:4px;font-size:12px;color:var(--ink-2);margin-top:4px}
.status>span{display:inline-grid;place-items:center;width:16px;height:16px;border-radius:50%;font-size:10px;font-weight:700;color:#fff}
.status-good>span{background:var(--good)}.status-warning>span{background:var(--warning);color:#0b0b0b}
.status-serious>span{background:var(--serious);color:#0b0b0b}.status-critical>span{background:var(--critical)}
.note{color:var(--ink-2);background:var(--wash);border-radius:6px;padding:8px 12px;margin:8px 0}
.note-error{box-shadow:inset 3px 0 0 var(--critical)}
code{font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;overflow-wrap:anywhere}
.table-wrap{margin:8px 0}
.scroll{overflow-x:auto}
input.filter{font:inherit;font-size:13px;width:100%;max-width:320px;padding:6px 10px;margin-bottom:8px;border-radius:6px;border:1px solid var(--ring);background:var(--page);color:var(--ink)}
table.dt{border-collapse:collapse;width:100%;font-size:13px}
table.dt th{text-align:left;font-weight:600;color:var(--ink-2);border-bottom:1px solid var(--axis);padding:6px 10px;cursor:pointer;user-select:none;white-space:nowrap}
table.dt th[aria-sort=ascending]::after{content:" ↑"}table.dt th[aria-sort=descending]::after{content:" ↓"}
table.dt td{border-bottom:1px solid var(--grid);padding:6px 10px;vertical-align:top}
table.dt td.num{font-variant-numeric:tabular-nums;text-align:right}
table.dt tbody tr:hover{background:var(--wash)}
.meter{display:inline-block;height:6px;border-radius:0 3px 3px 0;background:var(--s1);vertical-align:middle;margin-right:6px}
details{margin-top:8px}summary{cursor:pointer;color:var(--ink-2);font-size:13px}
figure.viz{margin:8px 0 4px}
figcaption{font-weight:600;font-size:13px;margin-bottom:6px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--ink-2);margin-bottom:4px}
.key{display:inline-flex;align-items:center;gap:6px}
.key-line{display:inline-block;width:14px;height:2px;border-radius:1px}
.key-line.s1{background:var(--s1)}.key-line.s2{background:var(--s2)}.key-line.s3{background:var(--s3)}.key-line.s4{background:var(--s4)}
.plot{position:relative}
.plot-scroll{overflow-x:auto}
svg.chart{width:100%;min-width:560px;height:auto;display:block;overflow:visible}
svg .grid{stroke:var(--grid);stroke-width:1}svg .axis{stroke:var(--axis);stroke-width:1}
svg .tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
svg .end-label,svg .cat{fill:var(--ink-2);font-size:11px}svg .val{fill:var(--ink);font-size:11px;font-variant-numeric:tabular-nums}
svg .line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
svg .line.s1{stroke:var(--s1)}svg .line.s2{stroke:var(--s2)}svg .line.s3{stroke:var(--s3)}svg .line.s4{stroke:var(--s4)}
svg .dot{stroke:var(--surface);stroke-width:2}
svg .dot.s1{fill:var(--s1)}svg .dot.s2{fill:var(--s2)}svg .dot.s3{fill:var(--s3)}svg .dot.s4{fill:var(--s4)}
svg .xh{stroke:var(--axis);stroke-width:1}
svg .bar.s1{fill:var(--s1)}svg .bar.muted{fill:var(--axis)}
svg .hit{fill:transparent}svg .bar-hit:hover+*,svg .bar-hit:focus{outline:none}
svg .bar-hit:hover,svg .bar-hit:focus{fill:var(--wash)}
.tip{position:absolute;pointer-events:none;background:var(--surface);color:var(--ink);box-shadow:0 0 0 1px var(--ring),0 4px 16px rgba(0,0,0,.12);border-radius:6px;padding:6px 10px;font-size:12px;white-space:nowrap;z-index:3}
.tip .t-head{color:var(--ink-2);margin-bottom:2px}
.tip .t-row{display:flex;align-items:center;gap:6px}
.tip .t-row strong{font-variant-numeric:tabular-nums}
.tip .t-name{color:var(--ink-2)}
footer{color:var(--muted);font-size:12px;text-align:center;padding:0 16px 32px}
@media (max-width:600px){h1{font-size:20px}section.card{padding:14px}.tile-value{font-size:22px}}
@media print{nav.toc,button.theme,input.filter{display:none}section.card{break-inside:avoid;box-shadow:none;border:1px solid #ccc}}
"""

_JS = r"""
(function(){
  var root=document.documentElement;
  var btn=document.querySelector('button.theme');
  function cur(){return root.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');}
  function label(){btn.textContent=cur()==='dark'?'Light mode':'Dark mode';}
  try{var saved=localStorage.getItem('tp-theme');if(saved)root.setAttribute('data-theme',saved);}catch(e){}
  if(btn){label();btn.addEventListener('click',function(){var t=cur()==='dark'?'light':'dark';root.setAttribute('data-theme',t);try{localStorage.setItem('tp-theme',t);}catch(e){}label();});}

  function fmt(v){return Math.abs(v)>=100||v%1===0?Math.round(v).toLocaleString():v.toFixed(1);}
  function place(tip,plot,px,py){
    var r=plot.getBoundingClientRect();tip.hidden=false;
    var w=tip.offsetWidth,h=tip.offsetHeight,x=px+14,y=py-h-10;
    if(x+w>r.width)x=px-w-14; if(x<0)x=0; if(y<0)y=py+14;
    tip.style.left=x+'px';tip.style.top=y+'px';
  }
  function row(tip,value,name,cls){
    var d=document.createElement('div');d.className='t-row';
    if(cls){var k=document.createElement('span');k.className='key-line '+cls;d.appendChild(k);}
    var s=document.createElement('strong');s.textContent=value;d.appendChild(s);
    if(name){var n=document.createElement('span');n.className='t-name';n.textContent=name;d.appendChild(n);}
    tip.appendChild(d);
  }
  document.querySelectorAll('figure.viz').forEach(function(fig){
    var plot=fig.querySelector('.plot'),tip=fig.querySelector('.tip'),svg=fig.querySelector('svg');
    if(!plot||!svg)return;
    var data=fig.getAttribute('data-chart')?JSON.parse(fig.getAttribute('data-chart')):null;
    var xh=svg.querySelector('.xh'),dots=svg.querySelectorAll('.dot');
    function scale(){var vb=svg.viewBox.baseVal;return svg.getBoundingClientRect().width/vb.width;}
    function show(ev,el){
      tip.textContent='';
      var rect=plot.getBoundingClientRect(),px,py;
      if(ev&&ev.clientX!==undefined){px=ev.clientX-rect.left;py=ev.clientY-rect.top;}
      else{var b=el.getBoundingClientRect();px=b.left-rect.left+b.width/2;py=b.top-rect.top+b.height/2;}
      if(data&&el.hasAttribute('data-i')){
        var i=+el.getAttribute('data-i'),s=scale();
        var head=document.createElement('div');head.className='t-head';head.textContent=data.x[i];tip.appendChild(head);
        data.series.forEach(function(se,k){row(tip,fmt(se.values[i])+(data.unit||''),se.name,'s'+(k+1));
          dots[k].setAttribute('cx',data.px[i]);dots[k].setAttribute('cy',se.y[i]);dots[k].setAttribute('visibility','visible');});
        xh.setAttribute('x1',data.px[i]);xh.setAttribute('x2',data.px[i]);xh.setAttribute('visibility','visible');
        px=data.px[i]*s+(svg.getBoundingClientRect().left-rect.left);
      }else{row(tip,el.getAttribute('data-tip'));}
      place(tip,plot,px,py);
    }
    function hide(){tip.hidden=true;if(xh)xh.setAttribute('visibility','hidden');dots.forEach(function(d){d.setAttribute('visibility','hidden');});}
    svg.querySelectorAll('.hit').forEach(function(h){
      h.addEventListener('pointermove',function(e){show(e,h);});
      h.addEventListener('focus',function(){show(null,h);});
      h.addEventListener('blur',hide);
    });
    svg.addEventListener('pointerleave',hide);
  });

  document.querySelectorAll('table.dt').forEach(function(t){
    t.querySelectorAll('th').forEach(function(th,col){
      function sort(){
        var asc=th.getAttribute('aria-sort')!=='ascending';
        t.querySelectorAll('th').forEach(function(o){o.removeAttribute('aria-sort');});
        th.setAttribute('aria-sort',asc?'ascending':'descending');
        var body=t.tBodies[0],rows=Array.prototype.slice.call(body.rows);
        function key(r){var c=r.cells[col];if(!c)return '';var v=c.getAttribute('data-v');
          if(v!==null&&v!==''&&!isNaN(+v))return +v; return (v!==null?v:c.textContent).toLowerCase();}
        rows.sort(function(a,b){var x=key(a),y=key(b);if(x<y)return asc?-1:1;if(x>y)return asc?1:-1;return 0;});
        rows.forEach(function(r){body.appendChild(r);});
      }
      th.addEventListener('click',sort);
      th.addEventListener('keydown',function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();sort();}});
    });
  });
  document.querySelectorAll('input.filter').forEach(function(inp){
    var t=inp.parentNode.querySelector('table.dt');
    inp.addEventListener('input',function(){var q=inp.value.toLowerCase();
      Array.prototype.forEach.call(t.tBodies[0].rows,function(r){r.hidden=q&&r.textContent.toLowerCase().indexOf(q)<0;});});
  });
})();
"""


def _page(title: str, subtitle: str, brand: str, accent: Optional[str], sections_html: List[Tuple[str, str]],
          body: str) -> str:
    css = _CSS.replace("ACCENT_LIGHT", accent or "#2a78d6").replace("ACCENT_DARK", accent or "#3987e5")
    toc = "".join(f'<a href="#{esc(sid)}">{esc(t)}</a>' for sid, t in sections_html)
    nav = f'<nav class="toc" aria-label="Sections"><div class="wrap">{toc}</div></nav>' if len(sections_html) > 1 else ""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>{css}</style></head>
<body><header class="top"><div class="wrap"><div><div class="brand">{esc(brand)}</div><h1>{esc(title)}</h1>
<div class="meta">{esc(subtitle)} · generated {generated}</div></div>
<button class="theme" type="button">Dark mode</button></div></header>
{nav}<main><div class="wrap">{body}</div></main>
<footer>Generated by testpredict</footer><script>{_JS}</script></body></html>
"""


def render_report(ctx: ReportContext, report_type: str = "health",
                  sections: Optional[Sequence[str]] = None) -> str:
    from . import sections as _builtin  # noqa: F401  (registers built-in sections)

    cfg = ctx.config
    types = report_types(cfg)
    if sections:
        spec = {"title": cfg.get("title") or "Custom report", "description": "", "sections": list(sections)}
    else:
        if report_type not in types:
            raise ValueError(f"unknown report type {report_type!r}; available: {', '.join(sorted(types))}")
        spec = types[report_type]
    missing = [s for s in spec["sections"] if s not in SECTIONS]
    if missing:
        raise ValueError(f"unknown section(s) {missing}; available: {', '.join(sorted(SECTIONS))}")

    cards, toc = [], []
    for sid in spec["sections"]:
        sec = SECTIONS[sid]
        opts = dict(cfg.get("section_options", {}).get(sid, {}))
        try:
            inner = sec.render(ctx, opts)
        except Exception as e:  # one broken (plugin) section must not sink the whole report
            traceback.print_exc()
            inner = note(f"This section failed to render: {type(e).__name__}: {e}", "error")
        title = opts.get("title", sec.title)
        desc = f'<p class="desc">{esc(sec.description)}</p>' if sec.description else ""
        cards.append(f'<section class="card" id="{esc(sid)}"><h2>{esc(title)}</h2>{desc}{inner}</section>')
        toc.append((sid, title))

    runs = ctx.runs
    span = f"{len(runs)} runs" + (f" · {runs[0].started_at[:10]} → {runs[-1].started_at[:10]}" if runs else "")
    subtitle = cfg.get("subtitle") or (spec.get("description") or span)
    if cfg.get("subtitle") is None and spec.get("description"):
        subtitle = f"{spec['description']} {span}"
    return _page(cfg.get("title") or spec["title"], subtitle, cfg.get("brand") or "testpredict",
                 cfg.get("accent"), toc, "".join(cards))


def write_report(path: str, html_text: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html_text, encoding="utf-8")
    return path
