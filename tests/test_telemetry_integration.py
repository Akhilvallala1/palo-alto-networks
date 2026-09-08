"""Integration tests: one row per request under failover, and graceful degradation.

The gateway (#6) is not built here. What is exercised is the contract telemetry
offers it: a request that fails over across three models is still one row, and a
store that cannot write never reaches the caller.
"""

from collections.abc import Sequence
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from conduit.config import load_models
from conduit.contracts import CompletionRequest, CompletionResponse, Complexity
from conduit.telemetry import build_recorder
from conduit.telemetry.metrics import collect, create_metrics_router
from conduit.telemetry.recorder import Recorder, TelemetryRecord
from conduit.telemetry.settings import TelemetrySettings
from conduit.telemetry.store import JsonlSqliteStore, read_jsonl
from telemetry_fixtures import HAIKU, REPO_CONFIG, StubClock, make_request, make_response

MODELS = load_models(REPO_CONFIG, env={}).models


class StubProvider:
    """A `Provider` that answers or raises, with no network anywhere."""

    def __init__(self, name: str, model: str, *, fails: int = 0) -> None:
        self.name = name
        self.model = model
        self.remaining_failures = fails
        self.calls = 0

    def supports(self, model: str) -> bool:
        return model == self.model

    async def complete(self, req: CompletionRequest, model: str) -> CompletionResponse:
        self.calls += 1
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise TimeoutError(f"{self.name} timed out")
        return make_response(model=model, provider=self.name, tier=Complexity.STANDARD)

    async def health(self) -> bool:
        return self.remaining_failures == 0


class ExplodingStore:
    """A store whose disk is full, or whose path is not writable."""

    def __init__(self) -> None:
        self.attempts = 0

    def write(self, record: TelemetryRecord) -> None:
        self.attempts += 1
        raise OSError("no space left on device")

    def close(self) -> None:
        return None


async def serve(
    recorder: Recorder,
    req: CompletionRequest,
    chain: Sequence[tuple[StubProvider, str]],
    *,
    attempts_per_model: int = 1,
) -> CompletionResponse:
    """A stand-in for the gateway's request path: guards, failover, one row.

    Deliberately written the way #6 will use the recorder — one `record` block
    wrapping the *whole* request, not one per provider call.
    """
    with recorder.record(req) as trace:
        trace.set_tier(Complexity.STANDARD)
        for provider, model in chain:
            trace.set_attempt(provider.name, model)
            for _ in range(attempts_per_model):
                try:
                    response = await provider.complete(req, model)
                except (TimeoutError, OSError):
                    continue
                # The failover layer owns `fallback_from`; mirror that here.
                response.fallback_from = list(trace.fallback_from)
                trace.set_response(response)
                return response
            trace.record_failover(model)
        raise RuntimeError("every provider in the chain failed")


def recorder_for(tmp_path: Path, store: object | None = None) -> Recorder:
    settings = TelemetrySettings(dir=tmp_path)
    return build_recorder(
        REPO_CONFIG,
        settings=settings,
        store=store,  # type: ignore[arg-type]
    )


async def test_a_failover_chain_writes_exactly_one_row(tmp_path: Path) -> None:
    recorder = recorder_for(tmp_path)
    store = recorder.store
    assert isinstance(store, JsonlSqliteStore)

    chain = [
        (StubProvider("anthropic", "claude-opus-5", fails=1), "claude-opus-5"),
        (StubProvider("anthropic", "claude-sonnet-5", fails=1), "claude-sonnet-5"),
        (StubProvider("anthropic", HAIKU), HAIKU),
    ]
    response = await serve(recorder, make_request(), chain)

    assert response.model == HAIKU
    rows = store.records()
    assert len(rows) == 1
    assert len(list(read_jsonl(store.jsonl_path))) == 1

    (row,) = rows
    assert row.status == "ok"
    assert row.model == HAIKU
    assert row.fallback_from == ["claude-opus-5", "claude-sonnet-5"]
    # Priced off the model that actually answered, not the two that failed.
    assert row.cost_usd == pytest.approx(0.0035)
    store.close()


async def test_retries_within_one_request_do_not_double_count(tmp_path: Path) -> None:
    recorder = recorder_for(tmp_path)
    store = recorder.store
    assert isinstance(store, JsonlSqliteStore)

    provider = StubProvider("anthropic", HAIKU, fails=2)
    await serve(recorder, make_request(), [(provider, HAIKU)], attempts_per_model=3)

    assert provider.calls == 3
    assert len(store.records()) == 1
    assert collect(store).overall.requests == 1
    store.close()


async def test_a_request_that_exhausts_the_chain_is_one_error_row(tmp_path: Path) -> None:
    recorder = recorder_for(tmp_path)
    store = recorder.store
    assert isinstance(store, JsonlSqliteStore)

    chain = [(StubProvider("anthropic", "claude-opus-5", fails=1), "claude-opus-5")]
    with pytest.raises(RuntimeError, match="every provider in the chain failed"):
        await serve(recorder, make_request(), chain)

    (row,) = store.records()
    assert row.status == "error"
    assert row.fallback_from == ["claude-opus-5"]
    assert row.cost_usd == 0.0
    store.close()


async def test_a_failing_store_never_fails_the_request(tmp_path: Path) -> None:
    store = ExplodingStore()
    recorder = recorder_for(tmp_path, store=store)

    chain = [(StubProvider("mock", "mock:echo"), "mock:echo")]
    response = await serve(recorder, make_request(), chain)

    assert response.text  # the caller got their answer
    assert store.attempts == 1  # and telemetry tried, and swallowed the failure


def test_a_failing_store_does_not_500_the_route(tmp_path: Path) -> None:
    """The same failure, seen through an HTTP handler that records a request."""
    store = ExplodingStore()
    recorder = recorder_for(tmp_path, store=store)
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions() -> dict[str, str]:
        response = await serve(
            recorder, make_request(), [(StubProvider("mock", "mock:echo"), "mock:echo")]
        )
        return {"text": response.text}

    result = TestClient(app).post("/v1/chat/completions")
    assert result.status_code == 200
    assert result.json()["text"]
    assert store.attempts == 1


async def test_metrics_reflect_rows_written_by_the_request_path(tmp_path: Path) -> None:
    recorder = Recorder(
        JsonlSqliteStore(tmp_path / "events.jsonl", tmp_path / "events.db"),
        models=MODELS,
        settings=TelemetrySettings(dir=tmp_path),
        monotonic=StubClock(step=0.1),
    )
    store = recorder.store
    assert isinstance(store, JsonlSqliteStore)

    for index in range(3):
        await serve(
            recorder,
            make_request(trace_id=f"trace-{index}"),
            [(StubProvider("anthropic", HAIKU), HAIKU)],
        )

    app = FastAPI()
    app.include_router(create_metrics_router(store))
    payload = TestClient(app).get("/metrics").json()

    assert payload["overall"]["requests"] == 3
    assert payload["overall"]["cost_usd"] == pytest.approx(3 * 0.0035)
    assert payload["overall"]["p50_latency_ms"] == 100  # StubClock: 0.1s per request
    assert payload["by_provider"][0]["key"] == "anthropic"
    assert payload["by_team"][0]["key"] == "gtm"
    store.close()
