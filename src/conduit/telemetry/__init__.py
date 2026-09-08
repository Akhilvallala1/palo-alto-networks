"""Telemetry: token, cost, latency and verdict recording plus `/metrics`.

The front door for the gateway (#6) is two calls::

    recorder = build_recorder()                       # config-driven wiring
    app.include_router(create_metrics_router(recorder.store))

    with recorder.record(req) as trace:               # one row per request
        ...
        trace.set_response(response)
"""

from pathlib import Path

from conduit.config import load_models
from conduit.telemetry.dashboard import render_dashboard
from conduit.telemetry.metrics import (
    Aggregate,
    MetricsSnapshot,
    collect,
    create_metrics_router,
    percentile,
    slice_by,
)
from conduit.telemetry.recorder import (
    Recorder,
    RecordStatus,
    RequestTrace,
    TelemetryRecord,
    price_usage,
)
from conduit.telemetry.settings import TelemetrySettings, load_telemetry
from conduit.telemetry.store import (
    JsonlSqliteStore,
    SliceAggregate,
    TelemetryQuery,
    TelemetryStore,
    read_jsonl,
)

__all__ = [
    "Aggregate",
    "JsonlSqliteStore",
    "MetricsSnapshot",
    "RecordStatus",
    "Recorder",
    "RequestTrace",
    "SliceAggregate",
    "TelemetryQuery",
    "TelemetryRecord",
    "TelemetrySettings",
    "TelemetryStore",
    "build_recorder",
    "collect",
    "create_metrics_router",
    "load_telemetry",
    "open_store",
    "percentile",
    "price_usage",
    "read_jsonl",
    "render_dashboard",
    "slice_by",
]


def open_store(settings: TelemetrySettings) -> JsonlSqliteStore:
    """Open the configured JSONL + SQLite store, creating its directory."""
    return JsonlSqliteStore(settings.jsonl_path, settings.sqlite_path)


def build_recorder(
    config_dir: Path | None = None,
    *,
    settings: TelemetrySettings | None = None,
    store: TelemetryStore | None = None,
) -> Recorder:
    """Wire a recorder from `config/`: store paths from telemetry.yaml, prices
    from models.yaml.

    Prices come from the registry rather than from the provider's own
    `usage.cost_usd` so the cost column is reproducible from config alone.
    """
    settings = settings or load_telemetry(config_dir)
    return Recorder(
        store if store is not None else open_store(settings),
        models=load_models(config_dir).models,
        settings=settings,
    )
