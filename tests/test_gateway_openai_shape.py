"""Unit tests for the OpenAI <-> `contracts` translation.

AC-1's bar is an *unmodified* SDK client, so these assert on the fields that
SDK's response model requires, not on a shape that merely looks familiar.
"""

import pytest

from conduit.contracts import CompletionResponse, Complexity, Usage
from conduit.gateway.errors import BadRequestError, StreamingUnsupportedError
from conduit.gateway.openai_compat import (
    ChatCompletionRequest,
    to_completion_request,
    to_openai_response,
)


def payload(**overrides: object) -> ChatCompletionRequest:
    body: dict[str, object] = {
        "model": "mock:echo",
        "messages": [{"role": "user", "content": "hello"}],
    }
    body.update(overrides)
    return ChatCompletionRequest.model_validate(body)


def response(**overrides: object) -> CompletionResponse:
    body: dict[str, object] = {
        "text": "hi there",
        "model": "mock:echo",
        "provider": "mock",
        "usage": Usage(prompt_tokens=10, completion_tokens=4, cost_usd=0.5),
        "latency_ms": 12,
        "routed_tier": Complexity.TRIVIAL,
    }
    body.update(overrides)
    return CompletionResponse.model_validate(body)


# -- request ---------------------------------------------------------------- #


def test_a_minimal_openai_request_becomes_a_completion_request() -> None:
    req = to_completion_request(payload())
    assert [(m.role, m.content) for m in req.messages] == [("user", "hello")]
    assert req.model == "mock:echo"


def test_unset_fields_fall_back_to_the_contract_defaults() -> None:
    """Not to the gateway's own numbers — the contract owns these."""
    req = to_completion_request(payload())
    assert req.max_tokens == 1024
    assert req.temperature == 0.0


def test_max_completion_tokens_is_accepted_as_max_tokens() -> None:
    assert to_completion_request(payload(max_completion_tokens=64)).max_tokens == 64


def test_unknown_sdk_parameters_are_tolerated_not_rejected() -> None:
    """A real SDK sends `top_p` and friends; 400-ing would fail the drop-in claim."""
    req = to_completion_request(payload(top_p=0.9, presence_penalty=0.1, n=1))
    assert req.max_tokens == 1024


def test_the_developer_role_maps_onto_system() -> None:
    body = payload(messages=[{"role": "developer", "content": "be terse"}])
    assert to_completion_request(body).messages[0].role == "system"


def test_an_unsupported_role_is_a_named_400() -> None:
    body = payload(messages=[{"role": "tool", "content": "{}"}])
    with pytest.raises(BadRequestError, match="Unsupported message role"):
        to_completion_request(body)


def test_text_content_parts_are_flattened() -> None:
    body = payload(
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            }
        ]
    )
    assert to_completion_request(body).messages[0].content == "ab"


def test_a_non_text_content_part_is_refused_rather_than_dropped() -> None:
    body = payload(
        messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
    )
    with pytest.raises(BadRequestError, match="Unsupported content part"):
        to_completion_request(body)


def test_stream_true_is_a_400_naming_the_limitation() -> None:
    with pytest.raises(StreamingUnsupportedError) as exc:
        to_completion_request(payload(stream=True))
    assert exc.value.status_code == 400
    assert "Streaming" in exc.value.message
    assert "stream: false" in exc.value.message


def test_stream_false_is_simply_a_non_streaming_request() -> None:
    assert to_completion_request(payload(stream=False)).messages


def test_an_empty_message_list_is_a_400() -> None:
    with pytest.raises(BadRequestError, match="at least one message"):
        to_completion_request(payload(messages=[]))


def test_the_openai_user_field_becomes_the_workflow_tag() -> None:
    """It is the nearest thing Conduit has, and telemetry slices on it."""
    req = to_completion_request(payload(user="quote-agent"))
    assert req.metadata["workflow"] == "quote-agent"


def test_explicit_metadata_wins_over_the_user_field() -> None:
    req = to_completion_request(payload(user="a", metadata={"workflow": "b"}))
    assert req.metadata["workflow"] == "b"


# -- response --------------------------------------------------------------- #


def test_the_response_carries_every_field_the_sdk_requires() -> None:
    out = to_openai_response(response(), created=1_700_000_000)
    assert out.object == "chat.completion"
    assert out.id.startswith("chatcmpl-")
    assert out.created == 1_700_000_000
    assert out.model == "mock:echo"
    assert out.choices[0].index == 0
    assert out.choices[0].message.role == "assistant"
    assert out.choices[0].message.content == "hi there"
    assert out.choices[0].finish_reason == "stop"


def test_usage_totals_include_cached_tokens() -> None:
    """`Usage.prompt_tokens` is the uncached remainder, so a naive copy undercounts."""
    usage = Usage(
        prompt_tokens=10,
        completion_tokens=4,
        cost_usd=0.0,
        cache_read_tokens=5,
        cache_write_tokens=2,
    )
    out = to_openai_response(response(usage=usage))
    assert out.usage.prompt_tokens == 17
    assert out.usage.completion_tokens == 4
    assert out.usage.total_tokens == 21


def test_finish_reason_is_length_when_the_budget_was_consumed() -> None:
    out = to_openai_response(response(), max_tokens=4)
    assert out.choices[0].finish_reason == "length"


def test_routing_detail_rides_in_the_x_conduit_extension() -> None:
    out = to_openai_response(
        response(routed_tier=Complexity.COMPLEX, fallback_from=["a", "b"]),
        trace_id="abc123",
    )
    assert out.x_conduit["provider"] == "mock"
    assert out.x_conduit["routed_tier"] == "complex"
    assert out.x_conduit["fallback_from"] == ["a", "b"]
    assert out.x_conduit["trace_id"] == "abc123"
