"""Integration tests for the failover chain with injected 500 / 429 / timeout.

These run the real registry, the real adapters and the real retry policy; only
the HTTP transport and the breaker's clock are fakes, so a failure here is a
failure of the failover logic and not of a stub (AC-8).
"""

import httpx
import pytest

from conduit.contracts import Complexity
from conduit.providers.base import ProviderError, RawCompletion
from conduit.providers.failover import (
    AllProvidersFailedError,
    BreakerState,
    CircuitBreakers,
    FailoverChain,
    FailoverEvent,
)
from conduit.providers.mock import MockProvider
from conduit.providers.registry import ModelRegistry, build_registry
from provider_fixtures import (
    ECHO,
    HAIKU,
    INSTANT_RETRY,
    LLAMA,
    REPO_CONFIG,
    FakeApi,
    anthropic_body,
    ollama_body,
    repo_models,
    request,
    timeout,
)
from test_circuit_breaker import Clock

LIVE_ENV = {"ANTHROPIC_API_KEY": "test-key"}
TRIVIAL_CHAIN = [HAIKU, LLAMA, ECHO]


def wired(api: FakeApi) -> ModelRegistry:
    """The real registry, with every HTTP adapter pointed at the fake API."""
    return build_registry(
        config_dir=REPO_CONFIG, env=LIVE_ENV, transport=api.transport, retry=INSTANT_RETRY
    )


def chain(api: FakeApi, clock: Clock | None = None) -> FailoverChain:
    breakers = CircuitBreakers(clock=clock) if clock is not None else CircuitBreakers()
    return FailoverChain(wired(api), breakers=breakers)


# --------------------------------------------------------------------------- #
# AC-3: injected failures fall over
# --------------------------------------------------------------------------- #


async def test_a_500_on_the_primary_succeeds_from_the_next_provider() -> None:
    """AC-3."""
    api = FakeApi().fail("/v1/messages", 500).json("/api/chat", ollama_body("local answer"))
    resp = await chain(api).complete(request(), TRIVIAL_CHAIN, Complexity.TRIVIAL)

    assert resp.provider == "ollama"
    assert resp.model == LLAMA
    assert resp.text == "local answer"
    assert resp.fallback_from == [HAIKU]
    assert api.count("/v1/messages") == 3  # retried three times before giving up


async def test_a_429_that_outlives_the_retry_budget_falls_over() -> None:
    api = FakeApi().fail("/v1/messages", 429).json("/api/chat", ollama_body())
    resp = await chain(api).complete(request(), TRIVIAL_CHAIN, Complexity.TRIVIAL)

    assert resp.provider == "ollama"
    assert resp.fallback_from == [HAIKU]
    assert api.count("/v1/messages") == 3  # AC-6: a 429 is retried, then failed over


async def test_a_timeout_falls_over() -> None:
    api = FakeApi().on("/v1/messages", timeout).json("/api/chat", ollama_body())
    resp = await chain(api).complete(request(), TRIVIAL_CHAIN, Complexity.TRIVIAL)
    assert resp.provider == "ollama"
    assert resp.fallback_from == [HAIKU]


async def test_a_non_retryable_401_still_fails_over_to_the_next_provider() -> None:
    """Not retryable is not the same as not worth failing over from."""
    api = FakeApi().fail("/v1/messages", 401).json("/api/chat", ollama_body())
    resp = await chain(api).complete(request(), TRIVIAL_CHAIN, Complexity.TRIVIAL)
    assert resp.provider == "ollama"
    assert api.count("/v1/messages") == 1  # AC-6: refused once, never retried


async def test_multi_hop_failover_records_every_model_it_tried_in_order() -> None:
    api = FakeApi().fail("/v1/messages", 503).fail("/api/chat", 500)
    resp = await chain(api).complete(request(), TRIVIAL_CHAIN, Complexity.TRIVIAL)

    assert resp.provider == "mock"
    assert resp.fallback_from == [HAIKU, LLAMA]


async def test_a_first_hop_success_leaves_fallback_from_empty() -> None:
    api = FakeApi().json("/v1/messages", anthropic_body("primary answered"))
    resp = await chain(api).complete(request(), TRIVIAL_CHAIN, Complexity.TRIVIAL)

    assert resp.provider == "anthropic"
    assert resp.fallback_from == []
    assert api.count("/api/chat") == 0


async def test_the_chains_tier_overrides_the_providers_own_guess() -> None:
    """A provider cannot classify complexity; the router's verdict wins."""
    api = FakeApi().json("/v1/messages", anthropic_body())
    resp = await chain(api).complete(request(), [HAIKU], Complexity.COMPLEX)
    assert resp.routed_tier is Complexity.COMPLEX


async def test_exhausting_the_chain_reports_everything_that_was_tried() -> None:
    api = FakeApi().fail("/v1/messages", 500).fail("/api/chat", 500)
    with pytest.raises(AllProvidersFailedError) as caught:
        await chain(api).complete(request(), [HAIKU, LLAMA], Complexity.TRIVIAL)

    assert caught.value.attempted == (HAIKU, LLAMA)
    assert isinstance(caught.value.last_error, ProviderError)
    assert HAIKU in str(caught.value)


async def test_a_dormant_model_in_the_chain_is_dropped_rather_than_attempted() -> None:
    api = FakeApi().json("/api/chat", ollama_body())
    registry = build_registry(config_dir=REPO_CONFIG, env={}, transport=api.transport)
    resp = await FailoverChain(registry).complete(request(), TRIVIAL_CHAIN, Complexity.TRIVIAL)

    assert resp.provider == "ollama"
    assert resp.fallback_from == []  # haiku was never in the registry to try
    assert api.count("/v1/messages") == 0


async def test_a_chain_with_nothing_usable_fails_immediately() -> None:
    api = FakeApi()
    registry = build_registry(config_dir=REPO_CONFIG, env={}, transport=api.transport)
    with pytest.raises(AllProvidersFailedError, match="no usable models"):
        await FailoverChain(registry).complete(request(), [HAIKU], Complexity.TRIVIAL)
    assert api.count() == 0


# --------------------------------------------------------------------------- #
# AC-4 / AC-5: the breaker, in the chain
# --------------------------------------------------------------------------- #


async def test_the_breaker_opens_after_five_failed_requests_and_then_skips_the_call() -> None:
    """AC-4: the sixth request does not reach the dead provider at all."""
    api = FakeApi().fail("/v1/messages", 500).json("/api/chat", ollama_body())
    clock = Clock()
    failover = chain(api, clock)

    for _ in range(5):
        await failover.complete(request(), TRIVIAL_CHAIN, Complexity.TRIVIAL)
    assert failover.breakers.state("anthropic") is BreakerState.OPEN
    calls_while_closed = api.count("/v1/messages")
    assert calls_while_closed == 15  # 5 requests x 3 attempts

    resp = await failover.complete(request(), TRIVIAL_CHAIN, Complexity.TRIVIAL)
    assert api.count("/v1/messages") == calls_while_closed  # rejected without a call
    assert resp.provider == "ollama"
    assert resp.fallback_from == [HAIKU]  # still visible as a hop that did not answer


async def test_a_half_open_probe_recovers_the_primary() -> None:
    """AC-5, end to end: the primary comes back and traffic returns to it."""
    down = {"value": True}

    def flaky(_: httpx.Request) -> httpx.Response:
        if down["value"]:
            return httpx.Response(500, json={"error": "down"})
        return httpx.Response(200, json=anthropic_body("primary is back"))

    api = FakeApi().on("/v1/messages", flaky).json("/api/chat", ollama_body())
    clock = Clock()
    failover = chain(api, clock)

    for _ in range(5):
        await failover.complete(request(), TRIVIAL_CHAIN, Complexity.TRIVIAL)
    assert failover.breakers.state("anthropic") is BreakerState.OPEN

    down["value"] = False
    clock.advance(30.0)
    resp = await failover.complete(request(), TRIVIAL_CHAIN, Complexity.TRIVIAL)

    assert resp.provider == "anthropic"
    assert resp.text == "primary is back"
    assert resp.fallback_from == []
    assert failover.breakers.state("anthropic") is BreakerState.CLOSED


async def test_a_failed_half_open_probe_reopens_the_breaker() -> None:
    """AC-5: still down after 30s means another 30s of fast rejection."""
    api = FakeApi().fail("/v1/messages", 500).json("/api/chat", ollama_body())
    clock = Clock()
    failover = chain(api, clock)

    for _ in range(5):
        await failover.complete(request(), TRIVIAL_CHAIN, Complexity.TRIVIAL)
    clock.advance(30.0)

    before = api.count("/v1/messages")
    await failover.complete(request(), TRIVIAL_CHAIN, Complexity.TRIVIAL)
    assert api.count("/v1/messages") > before  # the probe went out
    assert failover.breakers.state("anthropic") is BreakerState.OPEN

    probe_calls = api.count("/v1/messages")
    await failover.complete(request(), TRIVIAL_CHAIN, Complexity.TRIVIAL)
    assert api.count("/v1/messages") == probe_calls  # shut again


# --------------------------------------------------------------------------- #
# Observability and containment
# --------------------------------------------------------------------------- #


async def test_every_hop_emits_an_event_for_the_telemetry_layer() -> None:
    api = FakeApi().fail("/v1/messages", 500).json("/api/chat", ollama_body())
    events: list[FailoverEvent] = []
    failover = FailoverChain(wired(api), on_event=events.append)

    await failover.complete(request(), TRIVIAL_CHAIN, Complexity.TRIVIAL)

    assert [(e.provider, e.outcome) for e in events] == [
        ("anthropic", "failure"),
        ("ollama", "success"),
    ]
    assert events[0].model == HAIKU
    assert events[0].error is not None


async def test_a_circuit_open_skip_is_reported_as_its_own_outcome() -> None:
    api = FakeApi().fail("/v1/messages", 500).json("/api/chat", ollama_body())
    events: list[FailoverEvent] = []
    clock = Clock()
    failover = FailoverChain(
        wired(api), breakers=CircuitBreakers(clock=clock), on_event=events.append
    )

    for _ in range(5):
        await failover.complete(request(), TRIVIAL_CHAIN, Complexity.TRIVIAL)
    events.clear()
    await failover.complete(request(), TRIVIAL_CHAIN, Complexity.TRIVIAL)

    assert events[0].outcome == "circuit_open"
    assert events[0].breaker_state is BreakerState.OPEN


async def test_an_adapter_that_breaks_its_contract_is_contained_not_propagated() -> None:
    """A provider raising something other than ProviderError still fails over."""

    class Rogue(MockProvider):
        name = "mock"

        async def _invoke(self, req: object, model: str) -> RawCompletion:  # type: ignore[override]
            raise ZeroDivisionError("adapter bug")

    models = repo_models().models
    registry = ModelRegistry(
        {ECHO: models[ECHO], LLAMA: models[LLAMA]},
        [Rogue(models={ECHO: models[ECHO]}), _ok_ollama()],
    )
    resp = await FailoverChain(registry).complete(request(), [ECHO, LLAMA], Complexity.TRIVIAL)
    assert resp.provider == "ollama"
    assert resp.fallback_from == [ECHO]


def _ok_ollama() -> object:
    from conduit.providers.ollama import OllamaProvider

    api = FakeApi().json("/api/chat", ollama_body("fallback answer"))
    return OllamaProvider(
        models={LLAMA: repo_models().models[LLAMA]},
        transport=api.transport,
        retry=INSTANT_RETRY,
    )
