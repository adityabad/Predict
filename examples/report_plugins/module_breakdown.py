"""Example report plugin: failures broken down by module.

Load it with `--plugin examples/report_plugins/module_breakdown.py` or list it under
"plugins" in a report config. Any section id registered here can then be used in
`--sections` or in a custom report type.
"""
from collections import Counter

from testpredict.htmlreport import bar_chart, html_table, section


def module_of(test_id: str, depth: int) -> str:
    """'tests.checkout.test_flow::test_x' -> 'checkout' (depth=2 keeps the 2nd dotted part)."""
    parts = test_id.split("::", 1)[0].split(".")
    return parts[min(depth, len(parts)) - 1] if parts else test_id


@section("module-breakdown", "Failures by module", "Which areas of the product break most often.")
def module_breakdown(ctx, opts):
    depth = int(opts.get("depth", 2))          # configurable via section_options
    runs = ctx.recent_runs(opts.get("runs"))
    fails, retries = Counter(), Counter()
    for run in runs:
        for r in run.results:
            mod = module_of(r.test_id, depth)
            if r.status == "failed":
                fails[mod] += 1
            elif r.status == "flaky":
                retries[mod] += 1
    top = fails.most_common(int(opts.get("limit", 10)))
    chart = bar_chart(top, f"Hard failures by module, last {len(runs)} runs",
                      highlight=top[0][0] if top else None)
    table = html_table(["module", "hard failures", "passed on retry"],
                       [[m, n, retries[m]] for m, n in top], filterable=False)
    return chart + table
