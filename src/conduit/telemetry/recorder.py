"""One telemetry row per request, priced from the model registry.

The unit of telemetry is the *request*, not the provider call. A request that
fails over across three models is one row whose `fallback_from` names the two
that failed — otherwise every cost and latency number in `docs/BENCHMARKS.md`
would double-count exactly the traffic that is most interesting.

That invariant is enforced here rather than trusted: a `RequestTrace` writes at
most once, so a caller that both uses the context manager and calls `finish()`
still produces a single row.

Recording never fails the user's request. Anything the store raises is caught
and logged; the caller gets its answer either way.
"""

import logging
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from types import TracebackType
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from conduit.config import ModelSpec
from conduit.contracts import (
    CompletionRequest,
    CompletionResponse,
    Complexity,
    GuardVerdict,
    Usage,
)
from conduit.telemetry.settings import TelemetrySettings

if TYPE_CHECKING:  # `store` imports `TelemetryRecord` from here at runtime.
    from conduit.telemetry.store import TelemetryStore

__all__ = [
    "CACHE_READ_MULTIPLIER",
    "CACHE_WRITE_MULTIPLIER",
    "UNATTRIBUTED",
    "RecordStatus",
    "Recorder",
    "RequestTrace",
    "TelemetryRecord",
    "price_usage",
]

log = logging.getLogger(__name__)

#: Metadata keys are free-form strings on `CompletionRequest`; traffic that
#: arrives without them still has to land somewhere countable.
UNATTRIBUTED = "unknown"

#: From the frozen contract's `Usage` comments: cache reads bill at 0.1x the
#: input price, cache writes at 1.25x (5m TTL).
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25

RecordStatus = Literal["ok", "error", "blocked"]


class TelemetryRecord(BaseModel):
    """One request, as persisted. Append-only: nothing rewrites a row."""

    record_id: str
    trace_id: str
    ts: datetime
    team: str = UNATTRIBUTED
    workflow: str = UNATTRIBUTED

    tier: Complexity | None = None
    provider: str | None = None
    model: str | None = None

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    #: End-to-end, measured by the recorder — what a caller actually waited.
    latency_ms: int = 0
    #: What the provider reported for its own call, when there was one.
    provider_latency_ms: int | None = None

    guard_categories: list[str] = Field(default_factory=list)
    fallback_from: list[str] = Field(default_factory=list)

    status: RecordStatus = "ok"
    error: str | None = None

    #: None unless `persist_text` is enabled. Off in the shipped config.
    prompt_text: str | None = None
    completion_text: str | None = None

    @property
    def total_tokens(self) -> int:
        return (
            self.prompt_tokens
            + self.completion_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


def price_usage(usage: Usage, model: str, models: Mapping[str, ModelSpec]) -> float | None:
    """Price reported token usage from `config/models.yaml`.

    Returns None for a model the registry does not know, so the caller can fall
    back to the provider's own figure rather than silently report $0.
    """
    spec = models.get(model)
    if spec is None:
        return None
    input_rate = spec.input_per_1m_usd / 1_000_000
    output_rate = spec.output_per_1m_usd / 1_000_000
    return (
        usage.prompt_tokens * input_rate
        + usage.cache_read_tokens * input_rate * CACHE_READ_MULTIPLIER
        + usage.cache_write_tokens * input_rate * CACHE_WRITE_MULTIPLIER
        + usage.completion_tokens * output_rate
    )


class RequestTrace:
    """Mutable accumulator for one in-flight request.

    The gateway fills this in as the request moves through guards, router and
    providers; `Recorder.finish` freezes it into a `TelemetryRecord`.
    """

    def __init__(
        self,
        *,
        trace_id: str,
        team: str,
        workflow: str,
        started_at: datetime,
        started_monotonic: float,
        prompt_text: str | None,
    ) -> None:
        self.trace_id = trace_id
        self.team = team
        self.workflow = workflow
        self.started_at = started_at
        self.started_monotonic = started_monotonic
        self.prompt_text = prompt_text
        self.completion_text: str | None = None

        self.tier: Complexity | None = None
        self.provider: str | None = None
        self.model: str | None = None
        self.usage: Usage | None = None
        self.provider_latency_ms: int | None = None
        self.fallback_from: list[str] = []
        self.guard_categories: list[str] = []
        self.status: RecordStatus = "ok"
        self.error: str | None = None

        self.written = False

    # -- population ------------------------------------------------------- #

    def observe_guard(self, verdict: GuardVerdict) -> None:
        """Fold a guard verdict in. A block is recorded, not raised."""
        self.add_categories(verdict.categories)
        if not verdict.allowed:
            self.status = "blocked"

    def add_categories(self, categories: Iterable[str]) -> None:
        for category in categories:
            if category not in self.guard_categories:
                self.guard_categories.append(category)

    def set_tier(self, tier: Complexity) -> None:
        self.tier = tier

    def set_attempt(self, provider: str, model: str) -> None:
        """Note the model currently being tried, so a failure still names one."""
        self.provider = provider
        self.model = model

    def record_failover(self, model: str) -> None:
        """Append a model that was tried and failed, in order."""
        self.fallback_from.append(model)

    def set_response(self, response: CompletionResponse) -> None:
        self.tier = response.routed_tier
        self.provider = response.provider
        self.model = response.model
        self.usage = response.usage
        self.provider_latency_ms = response.latency_ms
        self.completion_text = response.text
        # The failover layer owns `fallback_from`; trust it over anything the
        # caller accumulated by hand, but do not lose hops if it is empty.
        if response.fallback_from:
            self.fallback_from = list(response.fallback_from)
        self.status = "ok"
        self.error = None

    def fail(self, error: BaseException | str) -> None:
        self.status = "error"
        if isinstance(error, BaseException):
            self.error = f"{type(error).__name__}: {error}"
        else:
            self.error = error


class Recorder:
    """Builds and persists exactly one `TelemetryRecord` per request."""

    def __init__(
        self,
        store: "TelemetryStore",
        *,
        models: Mapping[str, ModelSpec] | None = None,
        settings: TelemetrySettings | None = None,
        now: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._store = store
        self._models = dict(models or {})
        self._settings = settings or TelemetrySettings()
        self._now = now or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic

    @property
    def store(self) -> "TelemetryStore":
        """The store rows land in — what `/metrics` reads back."""
        return self._store

    # -- lifecycle -------------------------------------------------------- #

    def begin(self, req: CompletionRequest) -> RequestTrace:
        return RequestTrace(
            trace_id=req.metadata.get("trace_id") or uuid.uuid4().hex,
            team=req.metadata.get("team") or UNATTRIBUTED,
            workflow=req.metadata.get("workflow") or UNATTRIBUTED,
            started_at=self._now(),
            started_monotonic=self._monotonic(),
            prompt_text=self._capture("\n\n".join(m.content for m in req.messages)),
        )

    def finish(self, trace: RequestTrace) -> TelemetryRecord | None:
        """Write the row. A second call for the same trace is a no-op."""
        if trace.written:
            log.warning("telemetry: trace %s already recorded; not rewriting", trace.trace_id)
            return None
        trace.written = True

        record = self.build(trace)
        if not self._settings.enabled:
            return record
        try:
            self._store.write(record)
        except Exception:
            # Deliberately broad: a telemetry store failure is never allowed to
            # become the user's problem. The row is lost, the request is not.
            log.exception("telemetry: store write failed, dropping trace %s", trace.trace_id)
        return record

    @contextmanager
    def record(self, req: CompletionRequest) -> Iterator[RequestTrace]:
        """Trace a request. Exactly one row on exit, however the block leaves.

        An exception escaping the block is recorded as a failed request and then
        re-raised: the gateway decides what the caller sees, telemetry does not.
        """
        trace = self.begin(req)
        try:
            yield trace
        except BaseException as exc:
            trace.fail(exc)
            raise
        finally:
            self.finish(trace)

    # -- construction ----------------------------------------------------- #

    def build(self, trace: RequestTrace) -> TelemetryRecord:
        usage = trace.usage
        cost = 0.0
        if usage is not None:
            priced = price_usage(usage, trace.model, self._models) if trace.model else None
            if priced is None:
                # Unknown model: the provider's own figure is all we have.
                log.debug("telemetry: no registry price for model %r", trace.model)
                cost = usage.cost_usd
            else:
                cost = priced

        return TelemetryRecord(
            record_id=uuid.uuid4().hex,
            trace_id=trace.trace_id,
            ts=trace.started_at,
            team=trace.team,
            workflow=trace.workflow,
            tier=trace.tier,
            provider=trace.provider,
            model=trace.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            cache_read_tokens=usage.cache_read_tokens if usage else 0,
            cache_write_tokens=usage.cache_write_tokens if usage else 0,
            cost_usd=cost,
            latency_ms=max(0, round((self._monotonic() - trace.started_monotonic) * 1000)),
            provider_latency_ms=trace.provider_latency_ms,
            guard_categories=list(trace.guard_categories),
            fallback_from=list(trace.fallback_from),
            status=trace.status,
            error=trace.error,
            prompt_text=trace.prompt_text,
            completion_text=self._capture(trace.completion_text),
        )

    def _capture(self, text: str | None) -> str | None:
        """Return text only when `persist_text` is on. Off in the shipped config."""
        if text is None or not self._settings.persist_text:
            return None
        return text[: self._settings.max_text_chars]

    # -- convenience ------------------------------------------------------ #

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._store.close()
