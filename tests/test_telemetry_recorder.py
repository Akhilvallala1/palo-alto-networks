"""Unit tests for row construction, pricing and redaction-by-default."""

from typing import Any

import pytest

from conduit.config import load_models
from conduit.contracts import Complexity, GuardVerdict, Usage
from conduit.telemetry.recorder import UNATTRIBUTED, Recorder, TelemetryRecord, price_usage
from conduit.telemetry.settings import TelemetrySettings, load_telemetry
from telemetry_fixtures import (
    EPOCH,
    HAIKU,
    REPO_CONFIG,
    StubClock,
    make_request,
    make_response,
)

MODELS = load_models(REPO_CONFIG, env={}).models


class CollectingStore:
    """A store that keeps rows in memory. Satisfies the write side of the protocol."""

    def __init__(self) -> None:
        self.rows: list[TelemetryRecord] = []

    def write(self, record: TelemetryRecord) -> None:
        self.rows.append(record)

    def close(self) -> None:
        return None


def build_recorder(**kwargs: Any) -> tuple[Recorder, CollectingStore]:
    store = CollectingStore()
    settings = kwargs.pop("settings", TelemetrySettings())
    recorder = Recorder(
        store,  # type: ignore[arg-type]
        models=MODELS,
        settings=settings,
        now=lambda: EPOCH,
        monotonic=StubClock(),
        **kwargs,
    )
    return recorder, store


# --------------------------------------------------------------------------- #
# Row construction
# --------------------------------------------------------------------------- #


def test_row_carries_request_attribution_and_response_shape() -> None:
    recorder, store = build_recorder()
    with recorder.record(make_request()) as trace:
        trace.set_response(make_response())

    (row,) = store.rows
    assert row.trace_id == "trace-1"
    assert row.team == "gtm"
    assert row.workflow == "quote_approval"
    assert row.tier is Complexity.TRIVIAL
    assert row.provider == "anthropic"
    assert row.model == HAIKU
    assert (row.prompt_tokens, row.completion_tokens) == (1000, 500)
    assert row.status == "ok"
    assert row.ts == EPOCH


def test_unattributed_traffic_still_lands_somewhere_countable() -> None:
    recorder, store = build_recorder()
    with recorder.record(make_request(team=None, workflow=None)) as trace:
        trace.set_response(make_response())

    (row,) = store.rows
    assert row.team == UNATTRIBUTED
    assert row.workflow == UNATTRIBUTED


def test_missing_trace_id_gets_one_generated() -> None:
    recorder, store = build_recorder()
    request = make_request()
    del request.metadata["trace_id"]
    with recorder.record(request) as trace:
        trace.set_response(make_response())

    (row,) = store.rows
    assert row.trace_id


def test_latency_is_measured_end_to_end_not_taken_from_the_provider() -> None:
    recorder, store = build_recorder()
    with recorder.record(make_request()) as trace:
        trace.set_response(make_response(latency_ms=42))

    (row,) = store.rows
    assert row.latency_ms == 250  # StubClock: one 0.25s step between begin and build
    assert row.provider_latency_ms == 42


def test_failover_hops_are_recorded_on_the_single_row() -> None:
    recorder, store = build_recorder()
    with recorder.record(make_request()) as trace:
        trace.set_response(make_response(fallback_from=["claude-opus-5", "claude-sonnet-5"]))

    (row,) = store.rows
    assert row.fallback_from == ["claude-opus-5", "claude-sonnet-5"]


def test_guard_verdicts_fold_in_without_duplicates_and_a_block_sets_status() -> None:
    recorder, store = build_recorder()
    with recorder.record(make_request()) as trace:
        trace.observe_guard(GuardVerdict(allowed=True, risk_score=0.1, categories=["pii:email"]))
        trace.observe_guard(
            GuardVerdict(
                allowed=False,
                risk_score=0.9,
                categories=["pii:email", "injection:instruction_override"],
            )
        )

    (row,) = store.rows
    assert row.guard_categories == ["pii:email", "injection:instruction_override"]
    assert row.status == "blocked"


def test_an_exception_is_recorded_as_a_failed_request_and_re_raised() -> None:
    recorder, store = build_recorder()
    with pytest.raises(RuntimeError, match="upstream exploded"):  # noqa: SIM117
        with recorder.record(make_request()) as trace:
            trace.set_attempt("anthropic", HAIKU)
            raise RuntimeError("upstream exploded")

    (row,) = store.rows
    assert row.status == "error"
    assert row.error == "RuntimeError: upstream exploded"
    assert row.model == HAIKU  # we still know what was being tried


def test_finishing_twice_writes_one_row() -> None:
    recorder, store = build_recorder()
    trace = recorder.begin(make_request())
    trace.set_response(make_response())
    recorder.finish(trace)
    assert recorder.finish(trace) is None
    assert len(store.rows) == 1


def test_disabled_telemetry_builds_the_row_but_writes_nothing() -> None:
    recorder, store = build_recorder(settings=TelemetrySettings(enabled=False))
    trace = recorder.begin(make_request())
    trace.set_response(make_response())
    record = recorder.finish(trace)
    assert record is not None
    assert store.rows == []


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #


def test_cost_is_priced_from_the_registry_not_from_the_provider() -> None:
    recorder, store = build_recorder()
    with recorder.record(make_request()) as trace:
        # Haiku: $1.00/1M in, $5.00/1M out. 1000 in + 500 out = 0.001 + 0.0025.
        trace.set_response(make_response(prompt_tokens=1000, completion_tokens=500, cost_usd=99.0))

    (row,) = store.rows
    assert row.cost_usd == pytest.approx(0.0035)


def test_cache_tokens_price_at_their_contract_multipliers() -> None:
    recorder, store = build_recorder()
    with recorder.record(make_request()) as trace:
        trace.set_response(
            make_response(
                prompt_tokens=0,
                completion_tokens=0,
                cache_read_tokens=2000,  # 0.1x input
                cache_write_tokens=400,  # 1.25x input
                cost_usd=0.0,
            )
        )

    (row,) = store.rows
    assert row.cost_usd == pytest.approx(2000 * 1e-6 * 0.1 + 400 * 1e-6 * 1.25)
    assert row.cache_read_tokens == 2000
    assert row.cache_write_tokens == 400


def test_a_free_model_costs_zero_even_if_the_provider_claims_otherwise() -> None:
    recorder, store = build_recorder()
    with recorder.record(make_request()) as trace:
        trace.set_response(make_response(model="mock:echo", provider="mock", cost_usd=1.23))

    (row,) = store.rows
    assert row.cost_usd == 0.0


def test_unknown_model_falls_back_to_the_provider_reported_cost() -> None:
    recorder, store = build_recorder()
    with recorder.record(make_request()) as trace:
        trace.set_response(make_response(model="who:knows", provider="who", cost_usd=0.42))

    (row,) = store.rows
    assert row.cost_usd == 0.42


def test_price_usage_is_pure_and_declines_unknown_models() -> None:
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=0, cost_usd=0.0)
    assert price_usage(usage, HAIKU, MODELS) == pytest.approx(1.00)
    assert price_usage(usage, "not-a-model", MODELS) is None


def test_a_request_that_never_reached_a_provider_costs_nothing() -> None:
    recorder, store = build_recorder()
    with recorder.record(make_request()) as trace:
        trace.observe_guard(
            GuardVerdict(allowed=False, risk_score=1.0, categories=["injection:jailbreak"])
        )

    (row,) = store.rows
    assert row.cost_usd == 0.0
    assert (row.prompt_tokens, row.completion_tokens) == (0, 0)
    assert row.provider is None


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #


def test_prompt_and_completion_text_are_not_persisted_by_default() -> None:
    recorder, store = build_recorder()
    with recorder.record(make_request(text="my SSN is 123-45-6789")) as trace:
        trace.set_response(make_response(text="I cannot help with that."))

    (row,) = store.rows
    assert row.prompt_text is None
    assert row.completion_text is None
    assert "123-45-6789" not in row.model_dump_json()


def test_text_is_persisted_and_truncated_only_when_the_flag_is_on() -> None:
    settings = TelemetrySettings(persist_text=True, max_text_chars=10)
    recorder, store = build_recorder(settings=settings)
    with recorder.record(make_request(text="0123456789abcdef")) as trace:
        trace.set_response(make_response(text="abcdefghijklmnop"))

    (row,) = store.rows
    assert row.prompt_text == "0123456789"
    assert row.completion_text == "abcdefghij"


def test_the_shipped_config_keeps_text_persistence_off() -> None:
    assert load_telemetry(REPO_CONFIG, env={}).persist_text is False


def test_text_persistence_can_be_enabled_by_env_override() -> None:
    settings = load_telemetry(REPO_CONFIG, env={"CONDUIT_TELEMETRY__PERSIST_TEXT": "true"})
    assert settings.persist_text is True
