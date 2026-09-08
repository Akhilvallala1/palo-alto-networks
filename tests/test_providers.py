"""Adapter tests: request shaping, reply parsing, retry behaviour per vendor.

Every HTTP adapter is driven through `httpx.MockTransport`, so these exercise
the real client, the real headers and the real status-code handling without a
key or a socket (AC-8).
"""

import json
from collections.abc import Callable

import httpx
import pytest

from conduit.contracts import CompletionResponse, Complexity, Provider
from conduit.providers.anthropic import API_VERSION, AnthropicProvider
from conduit.providers.base import (
    BaseProvider,
    ProviderRefusedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitedError,
)
from conduit.providers.gemini import GeminiProvider
from conduit.providers.mock import MockProvider, digest_request
from conduit.providers.ollama import OllamaProvider
from conduit.providers.openai import OpenAIProvider
from provider_fixtures import (
    ECHO,
    FLASH,
    GPT,
    HAIKU,
    INSTANT_RETRY,
    LLAMA,
    NO_RETRY,
    FakeApi,
    anthropic_body,
    boom,
    gemini_body,
    no_sleep,
    ollama_body,
    openai_body,
    repo_models,
    request,
    timeout,
)

MODELS = repo_models().models


def anthropic(api: FakeApi) -> AnthropicProvider:
    return AnthropicProvider(
        models={m: s for m, s in MODELS.items() if s.provider == "anthropic"},
        api_key="test-key",
        base_url="https://api.anthropic.test",
        transport=api.transport,
        sleep=no_sleep,
        retry=INSTANT_RETRY,
    )


def ollama(api: FakeApi) -> OllamaProvider:
    return OllamaProvider(
        models={LLAMA: MODELS[LLAMA]},
        base_url="http://ollama.test:11434",
        transport=api.transport,
        sleep=no_sleep,
        retry=INSTANT_RETRY,
    )


def mock() -> MockProvider:
    return MockProvider(models={ECHO: MODELS[ECHO]})


# --------------------------------------------------------------------------- #
# AC-1: one request, three backends
# --------------------------------------------------------------------------- #


async def test_the_same_request_works_on_anthropic_ollama_and_mock() -> None:
    """AC-1: identical request, valid response everywhere, same text and tokens.

    The two HTTP fakes are pinned to whatever `mock` deterministically produces,
    so the only fields that may differ are the ones that describe *who* served
    the request — plus `cost_usd`, which differs precisely because the price
    table says Anthropic charges and the other two do not.
    """
    req = request()
    reference = await mock().complete(req, ECHO)
    text, prompt, completion = (
        reference.text,
        reference.usage.prompt_tokens,
        reference.usage.completion_tokens,
    )

    anthropic_api = FakeApi().json(
        "/v1/messages",
        anthropic_body(text, input_tokens=prompt, output_tokens=completion),
    )
    ollama_api = FakeApi().json(
        "/api/chat", ollama_body(text, prompt_tokens=prompt, eval_tokens=completion)
    )

    responses = [
        await anthropic(anthropic_api).complete(req, HAIKU),
        await ollama(ollama_api).complete(req, LLAMA),
        reference,
    ]

    for resp in responses:
        assert isinstance(resp, CompletionResponse)
        assert resp.text == text
        assert resp.usage.prompt_tokens == prompt
        assert resp.usage.completion_tokens == completion
        assert resp.fallback_from == []
        assert resp.latency_ms >= 0

    assert [r.provider for r in responses] == ["anthropic", "ollama", "mock"]
    assert [r.model for r in responses] == [HAIKU, LLAMA, ECHO]
    assert {r.routed_tier for r in responses} == {Complexity.TRIVIAL}


@pytest.mark.parametrize(
    ("expected_name", "build"),
    [
        ("anthropic", lambda: AnthropicProvider(models={}, api_key="k")),
        ("ollama", lambda: OllamaProvider(models={})),
        ("mock", lambda: MockProvider(models={})),
        ("openai", lambda: OpenAIProvider(models={}, api_key="k")),
        ("gemini", lambda: GeminiProvider(models={}, api_key="k")),
    ],
    ids=["anthropic", "ollama", "mock", "openai", "gemini"],
)
async def test_every_adapter_satisfies_the_provider_protocol(
    expected_name: str, build: Callable[[], BaseProvider]
) -> None:
    provider = build()
    try:
        assert isinstance(provider, Provider)
        assert provider.name == expected_name == expected_name.lower()
        assert provider.supports("anything") is False
    finally:
        await provider.aclose()


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #


async def test_anthropic_sends_the_messages_api_shape() -> None:
    api = FakeApi().json("/v1/messages", anthropic_body("drafted"))
    req = request("Quote Acme", system="You are terse.", max_tokens=256)
    resp = await anthropic(api).complete(req, HAIKU)

    sent = json.loads(api.last("/v1/messages").content)
    assert sent == {
        "model": HAIKU,
        "max_tokens": 256,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "Quote Acme"}],
        "system": "You are terse.",
    }
    assert api.last().headers["anthropic-version"] == API_VERSION
    assert api.last().headers["x-api-key"] == "test-key"
    assert resp.text == "drafted"
    assert resp.provider == "anthropic"


async def test_anthropic_prices_the_reply_from_the_yaml() -> None:
    """AC-2: 1M in + 200k out on haiku is $1.00 + $1.00."""
    api = FakeApi().json(
        "/v1/messages", anthropic_body(input_tokens=1_000_000, output_tokens=200_000)
    )
    resp = await anthropic(api).complete(request(), HAIKU)
    assert resp.usage.cost_usd == pytest.approx(2.00)


async def test_anthropic_reports_cache_tokens_separately() -> None:
    api = FakeApi().json(
        "/v1/messages",
        anthropic_body(input_tokens=100, output_tokens=10, cache_read=2_000, cache_write=500),
    )
    resp = await anthropic(api).complete(request(), HAIKU)
    assert resp.usage.prompt_tokens == 100
    assert resp.usage.cache_read_tokens == 2_000
    assert resp.usage.cache_write_tokens == 500
    expected = (100 + 2_000 * 0.1 + 500 * 1.25) * 1e-6 + 10 * 5e-6
    assert resp.usage.cost_usd == pytest.approx(expected)


async def test_anthropic_concatenates_multiple_text_blocks() -> None:
    body = anthropic_body()
    body["content"] = [
        {"type": "text", "text": "one "},
        {"type": "thinking", "thinking": "ignored"},
        {"type": "text", "text": "two"},
    ]
    api = FakeApi().json("/v1/messages", body)
    assert (await anthropic(api).complete(request(), HAIKU)).text == "one two"


async def test_anthropic_rejects_a_request_with_only_system_turns() -> None:
    api = FakeApi().json("/v1/messages", anthropic_body())
    req = request(system="only system")
    req.messages = [m for m in req.messages if m.role == "system"]
    with pytest.raises(ProviderRefusedError, match="no user or assistant turns"):
        await anthropic(api).complete(req, HAIKU)
    assert api.count() == 0


async def test_anthropic_retries_a_429_then_succeeds() -> None:
    """AC-6, at the wire level."""
    api = FakeApi().on(
        "/v1/messages",
        httpx.Response(429, json={"error": "slow down"}),
        httpx.Response(200, json=anthropic_body("recovered")),
    )
    resp = await anthropic(api).complete(request(), HAIKU)
    assert resp.text == "recovered"
    assert api.count("/v1/messages") == 2


@pytest.mark.parametrize("status", [400, 401, 403])
async def test_anthropic_does_not_retry_a_client_error(status: int) -> None:
    """AC-6: exactly one call, then give up."""
    api = FakeApi().fail("/v1/messages", status)
    with pytest.raises(ProviderRefusedError) as caught:
        await anthropic(api).complete(request(), HAIKU)
    assert caught.value.status_code == status
    assert api.count("/v1/messages") == 1


async def test_anthropic_retries_a_500_up_to_the_attempt_limit() -> None:
    api = FakeApi().fail("/v1/messages", 500)
    with pytest.raises(ProviderUnavailableError):
        await anthropic(api).complete(request(), HAIKU)
    assert api.count("/v1/messages") == 3


async def test_anthropic_maps_a_dropped_connection_to_unavailable() -> None:
    api = FakeApi().on("/v1/messages", boom)
    with pytest.raises(ProviderUnavailableError):
        await anthropic(api).complete(request(), HAIKU)


async def test_anthropic_maps_a_read_timeout_to_provider_timeout() -> None:
    api = FakeApi().on("/v1/messages", timeout)
    with pytest.raises(ProviderTimeoutError):
        await anthropic(api).complete(request(), HAIKU)


async def test_anthropic_health_is_true_when_the_models_endpoint_answers() -> None:
    api = FakeApi().json("/v1/models", {"data": []})
    assert await anthropic(api).health() is True


async def test_anthropic_health_is_false_on_a_bad_key_and_never_raises() -> None:
    api = FakeApi().fail("/v1/models", 401)
    assert await anthropic(api).health() is False


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #


async def test_ollama_strips_the_registry_prefix_on_the_wire() -> None:
    api = FakeApi().json("/api/chat", ollama_body("local answer"))
    resp = await ollama(api).complete(request("hello"), LLAMA)

    sent = json.loads(api.last("/api/chat").content)
    assert sent["model"] == "llama3.2"
    assert sent["stream"] is False
    assert sent["options"] == {"temperature": 0.0, "num_predict": 1024}
    assert resp.model == LLAMA  # the registry id, not the wire tag
    assert resp.text == "local answer"


def test_ollama_wire_model_leaves_an_unprefixed_tag_alone() -> None:
    assert OllamaProvider.wire_model("ollama:llama3.2") == "llama3.2"
    assert OllamaProvider.wire_model("llama3.2") == "llama3.2"


async def test_ollama_keeps_system_turns_inline() -> None:
    api = FakeApi().json("/api/chat", ollama_body())
    await ollama(api).complete(request("hi", system="be terse"), LLAMA)
    roles = [m["role"] for m in json.loads(api.last().content)["messages"]]
    assert roles == ["system", "user"]


async def test_ollama_maps_its_token_counters_and_is_free() -> None:
    api = FakeApi().json("/api/chat", ollama_body(prompt_tokens=900, eval_tokens=77))
    resp = await ollama(api).complete(request(), LLAMA)
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens) == (900, 77)
    assert resp.usage.cost_usd == 0.0


async def test_ollama_health_probes_the_tags_endpoint() -> None:
    api = FakeApi().json("/api/tags", {"models": []})
    assert await ollama(api).health() is True


async def test_ollama_health_is_false_when_nothing_is_listening() -> None:
    api = FakeApi().on("/api/tags", boom)
    assert await ollama(api).health() is False


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #


async def test_mock_is_deterministic_across_calls() -> None:
    req = request("same input")
    first = await mock().complete(req, ECHO)
    second = await mock().complete(request("same input"), ECHO)
    assert first.text == second.text
    assert first.usage.model_dump() == second.usage.model_dump()


async def test_mock_changes_its_answer_when_the_prompt_changes() -> None:
    a = await mock().complete(request("alpha"), ECHO)
    b = await mock().complete(request("beta"), ECHO)
    assert a.text != b.text


async def test_mock_ignores_metadata_so_a_new_trace_id_is_not_a_new_answer() -> None:
    a = await mock().complete(request("x", metadata={"trace_id": "one"}), ECHO)
    b = await mock().complete(request("x", metadata={"trace_id": "two"}), ECHO)
    assert a.text == b.text


def test_mock_digest_covers_the_fields_that_should_change_the_answer() -> None:
    base = request("x")
    assert digest_request(base, ECHO) != digest_request(request("x", max_tokens=8), ECHO)
    assert digest_request(base, ECHO) != digest_request(request("x", temperature=1.0), ECHO)
    assert digest_request(base, ECHO) != digest_request(base, "mock:other")


async def test_mock_echoes_the_last_turn_and_costs_nothing() -> None:
    resp = await mock().complete(request("Draft a quote"), ECHO)
    assert resp.text.endswith("Draft a quote")
    assert resp.usage.cost_usd == 0.0
    assert resp.provider == "mock"


async def test_mock_caps_its_completion_at_max_tokens() -> None:
    resp = await mock().complete(request("a" * 4000, max_tokens=5), ECHO)
    assert resp.usage.completion_tokens == 5


async def test_mock_is_always_healthy_and_needs_no_transport() -> None:
    assert await mock().health() is True


async def test_mock_refuses_an_empty_request() -> None:
    req = request()
    req.messages = []
    with pytest.raises(ProviderRefusedError):
        await mock().complete(req, ECHO)


# --------------------------------------------------------------------------- #
# Dormant adapters
# --------------------------------------------------------------------------- #


async def test_openai_subtracts_cached_tokens_from_the_prompt_count() -> None:
    api = FakeApi().json(
        "/v1/chat/completions",
        openai_body("hi", prompt_tokens=1000, completion_tokens=10, cached_tokens=400),
    )
    provider = OpenAIProvider(
        models={GPT: MODELS[GPT]},
        api_key="k",
        base_url="https://openai.test",
        transport=api.transport,
        sleep=no_sleep,
        retry=NO_RETRY,
    )
    resp = await provider.complete(request(), GPT)
    assert resp.usage.prompt_tokens == 600
    assert resp.usage.cache_read_tokens == 400
    assert api.last().headers["authorization"] == "Bearer k"


async def test_openai_reports_a_reply_with_no_choices_as_unavailable() -> None:
    api = FakeApi().json("/v1/chat/completions", {"choices": []})
    provider = OpenAIProvider(
        models={GPT: MODELS[GPT]},
        api_key="k",
        base_url="https://openai.test",
        transport=api.transport,
        sleep=no_sleep,
        retry=NO_RETRY,
    )
    with pytest.raises(ProviderUnavailableError, match="no choices"):
        await provider.complete(request(), GPT)


async def test_gemini_translates_roles_and_pulls_the_system_prompt_out() -> None:
    api = FakeApi().json(":generateContent", gemini_body("gen"))
    provider = GeminiProvider(
        models={FLASH: MODELS[FLASH]},
        api_key="k",
        base_url="https://gemini.test",
        transport=api.transport,
        sleep=no_sleep,
        retry=NO_RETRY,
    )
    resp = await provider.complete(request("hi", system="be terse"), FLASH)

    sent = json.loads(api.last().content)
    assert sent["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert sent["systemInstruction"] == {"parts": [{"text": "be terse"}]}
    assert api.last().headers["x-goog-api-key"] == "k"
    assert resp.text == "gen"
    assert resp.model == FLASH


async def test_gemini_prices_with_the_yaml_rates() -> None:
    api = FakeApi().json(
        ":generateContent", gemini_body(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    )
    provider = GeminiProvider(
        models={FLASH: MODELS[FLASH]},
        api_key="k",
        base_url="https://gemini.test",
        transport=api.transport,
        retry=NO_RETRY,
    )
    resp = await provider.complete(request(), FLASH)
    assert resp.usage.cost_usd == pytest.approx(0.30 + 2.50)


async def test_a_rate_limited_dormant_provider_still_reports_a_rate_limit() -> None:
    api = FakeApi().fail("/v1/chat/completions", 429)
    provider = OpenAIProvider(
        models={GPT: MODELS[GPT]},
        api_key="k",
        base_url="https://openai.test",
        transport=api.transport,
        sleep=no_sleep,
        retry=NO_RETRY,
    )
    with pytest.raises(RateLimitedError):
        await provider.complete(request(), GPT)
