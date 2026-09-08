"""`GET /metrics` and `GET /dashboard`, computed from stored rows.

Aggregation is split deliberately: SQLite does the counting and summing, this
module does the percentile math. SQLite has no `percentile_cont`, and the
alternatives (a UDF, or bucketed histograms) are harder to hand-check than the
one thing a reviewer actually wants to verify — that p95 is the number they
would get by sorting the latencies themselves.

Percentiles use the **nearest-rank** method: p is the sample at index
`ceil(p/100 * n)` in the ascending sample list. It always returns a latency that
really happened, and it is reproducible by hand from the JSONL log.
"""

import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from conduit.telemetry.dashboard import render_dashboard
from conduit.telemetry.store import (
    UNSLICED_KEY,
    SliceAggregate,
    TelemetryQuery,
    TelemetryStore,
)

__all__ = [
    "Aggregate",
    "MetricsSnapshot",
    "collect",
    "create_metrics_router",
    "percentile",
    "slice_by",
]


def percentile(samples: Sequence[int], p: float) -> int:
    """Nearest-rank percentile over ascending `samples`. Empty set is 0.

    `p` is a percentage: `percentile(xs, 95)`.
    """
    if not samples:
        return 0
    if not 0 < p <= 100:
        raise ValueError(f"percentile must be in (0, 100], got {p}")
    ordered = sorted(samples)
    rank = math.ceil(p / 100 * len(ordered))
    return ordered[max(1, rank) - 1]


class Aggregate(BaseModel):
    """One slice of traffic: what it cost, how much it moved, how slow it was."""

    key: str
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    errors: int = 0
    p50_latency_ms: int = 0
    p95_latency_ms: int = 0


class MetricsSnapshot(BaseModel):
    """What `/metrics` returns: overall plus the three slices AC-12 names."""

    generated_at: datetime
    filters: TelemetryQuery
    overall: Aggregate
    by_provider: list[Aggregate]
    by_team: list[Aggregate]
    by_tier: list[Aggregate]


def _to_aggregate(row: SliceAggregate, latencies: Sequence[int]) -> Aggregate:
    return Aggregate(
        key=row.key,
        requests=row.requests,
        prompt_tokens=row.prompt_tokens,
        completion_tokens=row.completion_tokens,
        cache_read_tokens=row.cache_read_tokens,
        cache_write_tokens=row.cache_write_tokens,
        total_tokens=(
            row.prompt_tokens
            + row.completion_tokens
            + row.cache_read_tokens
            + row.cache_write_tokens
        ),
        cost_usd=row.cost_usd,
        errors=row.errors,
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
    )


def slice_by(
    store: TelemetryStore, column: str, query: TelemetryQuery | None = None
) -> list[Aggregate]:
    """Aggregates grouped by one column, most expensive slice first."""
    latencies = store.latencies(query, group_by=column)
    rows = store.aggregate(query, column)
    return [_to_aggregate(row, latencies.get(row.key, [])) for row in rows]


def collect(store: TelemetryStore, query: TelemetryQuery | None = None) -> MetricsSnapshot:
    """Build the full `/metrics` payload for a window."""
    query = query or TelemetryQuery()
    rows = store.aggregate(query, None)
    latencies = store.latencies(query, None).get(UNSLICED_KEY, [])
    overall = _to_aggregate(rows[0], latencies) if rows else Aggregate(key=UNSLICED_KEY)
    return MetricsSnapshot(
        generated_at=datetime.now(UTC),
        filters=query,
        overall=overall,
        by_provider=slice_by(store, "provider", query),
        by_team=slice_by(store, "team", query),
        by_tier=slice_by(store, "tier", query),
    )


def create_metrics_router(store: TelemetryStore) -> APIRouter:
    """Routes the gateway mounts: `GET /metrics` and `GET /dashboard`.

    Telemetry owns the read side of its own data, so #6 mounts this router
    rather than reimplementing aggregation against the store.
    """
    router = APIRouter(tags=["telemetry"])

    def _query(
        window_minutes: int | None,
        provider: str | None,
        team: str | None,
        tier: str | None,
    ) -> TelemetryQuery:
        since = (
            datetime.now(UTC) - timedelta(minutes=window_minutes)
            if window_minutes is not None
            else None
        )
        return TelemetryQuery(since=since, provider=provider, team=team, tier=tier)

    @router.get(
        "/metrics",
        response_model=MetricsSnapshot,
        summary="Traffic, tokens, cost and latency percentiles",
    )
    def metrics(
        window_minutes: int | None = Query(default=None, gt=0),
        provider: str | None = None,
        team: str | None = None,
        tier: str | None = None,
    ) -> MetricsSnapshot:
        return collect(store, _query(window_minutes, provider, team, tier))

    @router.get("/dashboard", response_class=HTMLResponse, summary="The same numbers, in HTML")
    def dashboard(
        window_minutes: int | None = Query(default=None, gt=0),
        provider: str | None = None,
        team: str | None = None,
        tier: str | None = None,
    ) -> HTMLResponse:
        snapshot = collect(store, _query(window_minutes, provider, team, tier))
        return HTMLResponse(render_dashboard(snapshot))

    return router
