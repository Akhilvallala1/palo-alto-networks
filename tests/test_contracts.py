"""Schema stability tests for the frozen interface contract.

These are deliberately blunt: they pin the exact field names of every model so
that a concurrent agent adding, renaming or reordering a field fails CI instead
of silently forking the seam. If a test here fails, the fix is to update the
epic, not to loosen the test.
"""

import inspect
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from conduit.contracts import (
    CompletionRequest,
    CompletionResponse,
    Complexity,
    Guard,
    GuardVerdict,
    JudgeScore,
    Message,
    Provider,
    Usage,
)

# model -> (all field names, required field names)
EXPECTED_FIELDS: dict[type[BaseModel], tuple[set[str], set[str]]] = {
    Message: ({"role", "content"}, {"role", "content"}),
    CompletionRequest: (
        {"messages", "model", "max_tokens", "temperature", "metadata"},
        {"messages"},
    ),
    Usage: (
        {
            "prompt_tokens",
            "completion_tokens",
            "cost_usd",
            "cache_read_tokens",
            "cache_write_tokens",
        },
        {"prompt_tokens", "completion_tokens", "cost_usd"},
    ),
    CompletionResponse: (
        {
            "text",
            "model",
            "provider",
            "usage",
            "latency_ms",
            "routed_tier",
            "fallback_from",
        },
        {"text", "model", "provider", "usage", "latency_ms", "routed_tier"},
    ),
    GuardVerdict: (
        {"allowed", "risk_score", "categories", "redacted_text", "entity_map"},
        {"allowed", "risk_score", "categories"},
    ),
    JudgeScore: (
        {"rubric", "score", "reasoning", "passed"},
        {"rubric", "score", "reasoning", "passed"},
    ),
}

MESSAGE = Message(role="user", content="Draft a quote for Acme.")
USAGE = Usage(prompt_tokens=120, completion_tokens=40, cost_usd=0.00042)

INSTANCES: list[BaseModel] = [
    MESSAGE,
    Message(role="system", content="You are a pricing analyst."),
    CompletionRequest(messages=[MESSAGE]),
    CompletionRequest(
        messages=[MESSAGE],
        model="claude-sonnet-5",
        max_tokens=256,
        temperature=0.7,
        metadata={"team": "gtm", "workflow": "quote", "trace_id": "abc123"},
    ),
    USAGE,
    CompletionResponse(
        text="Quote drafted.",
        model="claude-haiku-4-5-20251001",
        provider="anthropic",
        usage=USAGE,
        latency_ms=412,
        routed_tier=Complexity.TRIVIAL,
    ),
    CompletionResponse(
        text="Quote drafted.",
        model="ollama:llama3.2",
        provider="ollama",
        usage=USAGE,
        latency_ms=980,
        routed_tier=Complexity.STANDARD,
        fallback_from=["anthropic"],
    ),
    GuardVerdict(allowed=True, risk_score=0.0, categories=[]),
    GuardVerdict(
        allowed=False,
        risk_score=0.91,
        categories=["pii:email", "injection:instruction_override"],
        redacted_text="Email <EMAIL_1> about the renewal.",
        entity_map={"<EMAIL_1>": "jane@acme.com"},
    ),
    JudgeScore(rubric="policy_citation", score=0.82, reasoning="Cites §4.2.", passed=True),
]


@pytest.mark.parametrize("instance", INSTANCES, ids=lambda i: type(i).__name__)
def test_round_trips_through_json(instance: BaseModel) -> None:
    """Every model survives serialize -> deserialize unchanged."""
    restored = type(instance).model_validate_json(instance.model_dump_json())
    assert restored == instance


@pytest.mark.parametrize("instance", INSTANCES, ids=lambda i: type(i).__name__)
def test_round_trips_through_dict(instance: BaseModel) -> None:
    """Every model survives model_dump -> model_validate unchanged."""
    restored = type(instance).model_validate(instance.model_dump())
    assert restored == instance


@pytest.mark.parametrize("model", list(EXPECTED_FIELDS), ids=lambda m: m.__name__)
def test_field_names_are_frozen(model: type[BaseModel]) -> None:
    expected, _ = EXPECTED_FIELDS[model]
    assert set(model.model_fields) == expected


@pytest.mark.parametrize("model", list(EXPECTED_FIELDS), ids=lambda m: m.__name__)
def test_required_fields_are_frozen(model: type[BaseModel]) -> None:
    _, expected_required = EXPECTED_FIELDS[model]
    required = {name for name, f in model.model_fields.items() if f.is_required()}
    assert required == expected_required


@pytest.mark.parametrize("model", list(EXPECTED_FIELDS), ids=lambda m: m.__name__)
def test_omitting_every_required_field_is_rejected(model: type[BaseModel]) -> None:
    _, expected_required = EXPECTED_FIELDS[model]
    if not expected_required:
        pytest.skip("no required fields")
    with pytest.raises(ValidationError):
        model.model_validate({})


def test_complexity_values() -> None:
    assert [c.value for c in Complexity] == ["trivial", "standard", "complex"]
    assert Complexity("trivial") is Complexity.TRIVIAL
    assert Complexity.COMPLEX == "complex"


def test_completion_request_defaults() -> None:
    req = CompletionRequest(messages=[MESSAGE])
    assert req.model is None  # None => router decides
    assert req.max_tokens == 1024
    assert req.temperature == 0.0
    assert req.metadata == {}


def test_completion_request_metadata_is_not_shared_between_instances() -> None:
    first = CompletionRequest(messages=[MESSAGE])
    first.metadata["team"] = "gtm"
    assert CompletionRequest(messages=[MESSAGE]).metadata == {}


def test_completion_response_defaults_have_no_fallback() -> None:
    resp = CompletionResponse(
        text="ok",
        model="mock:echo",
        provider="mock",
        usage=USAGE,
        latency_ms=1,
        routed_tier=Complexity.TRIVIAL,
    )
    assert resp.fallback_from == []


def test_guard_verdict_defaults() -> None:
    verdict = GuardVerdict(allowed=True, risk_score=0.0, categories=[])
    assert verdict.redacted_text is None
    assert verdict.entity_map == {}


def test_message_role_is_constrained() -> None:
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "tool", "content": "x"})


def test_routed_tier_accepts_the_wire_value() -> None:
    resp = CompletionResponse.model_validate(
        {
            "text": "ok",
            "model": "claude-opus-5",
            "provider": "anthropic",
            "usage": USAGE.model_dump(),
            "latency_ms": 7,
            "routed_tier": "complex",
        }
    )
    assert resp.routed_tier is Complexity.COMPLEX


def test_nested_usage_is_parsed_from_a_plain_dict() -> None:
    resp = CompletionResponse.model_validate(
        {
            "text": "ok",
            "model": "mock:echo",
            "provider": "mock",
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "cost_usd": 0.0},
            "latency_ms": 3,
            "routed_tier": "trivial",
        }
    )
    assert resp.usage == Usage(prompt_tokens=1, completion_tokens=2, cost_usd=0.0)


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("protocol", [Provider, Guard], ids=["Provider", "Guard"])
def test_protocol_documents_its_invariant(protocol: type[Any]) -> None:
    """AC-5: every Protocol states the invariant an implementer must uphold."""
    doc = inspect.getdoc(protocol)
    assert doc is not None
    assert "Invariant" in doc


def _protocol_members(protocol: type[Any]) -> set[str]:
    """Public members of a Protocol: annotated attributes plus declared methods."""
    annotated = set(getattr(protocol, "__annotations__", {}))
    declared = {name for name in vars(protocol) if not name.startswith("_")}
    return annotated | declared


def test_provider_protocol_members() -> None:
    assert _protocol_members(Provider) == {"name", "supports", "complete", "health"}


def test_guard_protocol_members() -> None:
    assert _protocol_members(Guard) == {"name", "inspect"}


class _StubProvider:
    name = "mock"

    def supports(self, model: str) -> bool:
        return model == "mock:echo"

    async def complete(self, req: CompletionRequest, model: str) -> CompletionResponse:
        return CompletionResponse(
            text=req.messages[-1].content,
            model=model,
            provider=self.name,
            usage=Usage(prompt_tokens=0, completion_tokens=0, cost_usd=0.0),
            latency_ms=0,
            routed_tier=Complexity.TRIVIAL,
        )

    async def health(self) -> bool:
        return True


class _StubGuard:
    name = "stub"

    async def inspect(self, text: str) -> GuardVerdict:
        return GuardVerdict(allowed=True, risk_score=0.0, categories=[])


def test_a_structural_implementation_satisfies_provider() -> None:
    provider: Provider = _StubProvider()
    assert isinstance(provider, Provider)
    assert provider.supports("mock:echo")


def test_a_structural_implementation_satisfies_guard() -> None:
    guard: Guard = _StubGuard()
    assert isinstance(guard, Guard)


async def test_guard_inspect_returns_a_verdict() -> None:
    assert (await _StubGuard().inspect("hello")).allowed


async def test_provider_complete_returns_a_completion_response() -> None:
    resp = await _StubProvider().complete(CompletionRequest(messages=[MESSAGE]), "mock:echo")
    assert resp.provider == "mock"
    assert resp.model == "mock:echo"
    assert resp.fallback_from == []
