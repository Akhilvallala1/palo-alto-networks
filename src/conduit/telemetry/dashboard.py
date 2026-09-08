"""A deliberately plain HTML view of the same aggregates `/metrics` returns.

One server-rendered string: no JavaScript, no build step, no assets to serve.
The point is that a reviewer can read the routing and cost story in one
screenshot, and that the page cannot show a number `/metrics` would not.
"""

from html import escape
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # `metrics` imports `render_dashboard` from here at runtime.
    from conduit.telemetry.metrics import Aggregate, MetricsSnapshot

__all__ = ["render_dashboard"]

_STYLE = """
:root { color-scheme: light dark; }
body { font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; margin: 2rem auto;
       max-width: 60rem; padding: 0 1rem; }
h1 { font-size: 1.25rem; margin-bottom: 0.25rem; }
p.meta { color: #666; margin-top: 0; }
table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }
caption { text-align: left; font-weight: 600; padding: 0.5rem 0; }
th, td { border-bottom: 1px solid #ccc; padding: 0.35rem 0.6rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
thead th { border-bottom: 2px solid #888; }
tr.total td { font-weight: 600; }
td.empty { text-align: left; color: #666; font-style: italic; }
"""

_COLUMNS = (
    ("requests", "requests"),
    ("prompt_tokens", "prompt tok"),
    ("completion_tokens", "completion tok"),
    ("total_tokens", "total tok"),
    ("cost_usd", "cost"),
    ("p50_latency_ms", "p50 ms"),
    ("p95_latency_ms", "p95 ms"),
    ("errors", "errors"),
)


def _usd(value: float) -> str:
    return f"${value:,.4f}"


def _cell(aggregate: "Aggregate", field: str) -> str:
    value = getattr(aggregate, field)
    if field == "cost_usd":
        return _usd(float(value))
    return f"{int(value):,}"


def _row(aggregate: "Aggregate", *, css_class: str = "") -> str:
    cells = "".join(f"<td>{_cell(aggregate, field)}</td>" for field, _ in _COLUMNS)
    attr = f' class="{css_class}"' if css_class else ""
    return f"<tr{attr}><td>{escape(aggregate.key)}</td>{cells}</tr>"


def _table(title: str, label: str, rows: "list[Aggregate]") -> str:
    header = "".join(f"<th>{escape(heading)}</th>" for _, heading in _COLUMNS)
    if rows:
        body = "\n".join(_row(row) for row in rows)
    else:
        body = f'<tr><td class="empty" colspan="{len(_COLUMNS) + 1}">no requests recorded</td></tr>'
    return (
        f"<table><caption>{escape(title)}</caption>"
        f"<thead><tr><th>{escape(label)}</th>{header}</tr></thead>"
        f"<tbody>\n{body}\n</tbody></table>"
    )


def _filter_summary(snapshot: "MetricsSnapshot") -> str:
    active = snapshot.filters.model_dump(exclude_none=True, mode="json")
    return ", ".join(f"{key}={value}" for key, value in sorted(active.items())) or "no filters"


def render_dashboard(snapshot: "MetricsSnapshot") -> str:
    """Render a whole page from one `/metrics` snapshot."""
    tables = "\n".join(
        (
            _table("Overall", "scope", [snapshot.overall]),
            _table("By provider", "provider", snapshot.by_provider),
            _table("By team", "team", snapshot.by_team),
            _table("By tier", "tier", snapshot.by_tier),
        )
    )
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Conduit telemetry</title>"
        f"<style>{_STYLE}</style></head><body>"
        "<h1>Conduit telemetry</h1>"
        f'<p class="meta">generated {escape(snapshot.generated_at.isoformat())} &middot; '
        f"{escape(_filter_summary(snapshot))}</p>"
        f"{tables}"
        "</body></html>\n"
    )
