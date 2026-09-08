"""Frozen interface contract for Conduit.

Every component in the system imports from this module and nothing else crosses
module boundaries. Providers, router, guards, gateway, telemetry and the eval
plane are built concurrently against these shapes, so any change here is an API
break, not a refactor: it requires updating every child issue in the epic.
"""

from enum import Enum
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

__all__ = [
    "CompletionRequest",
    "CompletionResponse",
    "Complexity",
    "Guard",
    "GuardVerdict",
    "JudgeScore",
    "Message",
    "Provider",
    "Usage",
]


class Complexity(str, Enum):
    TRIVIAL = "trivial"  # classification, extraction, formatting
    STANDARD = "standard"  # summarization, drafting, single-hop RAG
    COMPLEX = "complex"  # multi-hop reasoning, planning, code, math


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class CompletionRequest(BaseModel):
    messages: list[Message]
    model: str | None = None  # None => router decides
    max_tokens: int = 1024
    temperature: float = 0.0
    metadata: dict[str, str] = Field(default_factory=dict)  # team, workflow, trace_id


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class CompletionResponse(BaseModel):
    text: str
    model: str  # concrete model actually used
    provider: str
    usage: Usage
    latency_ms: int
    routed_tier: Complexity
    fallback_from: str | None = None  # set when primary provider failed


@runtime_checkable
class Provider(Protocol):
    """A model backend that can answer a `CompletionRequest`.

    Invariants an implementer must uphold:

    - `name` is a stable, lowercase identifier unique across providers
      (e.g. "anthropic", "ollama", "mock"). Telemetry and the failover chain key
      on it, so it must not change between calls or across process restarts.
    - `supports(model)` is pure and side-effect free: no network calls, no
      mutation. It must return True for exactly the model ids this provider can
      serve, and the router will only pass a `model` to `complete` for which
      `supports(model)` already returned True.
    - `complete(req, model)` returns a `CompletionResponse` whose `provider`
      equals `self.name` and whose `model` is the concrete model actually used —
      never an alias, never None. It must populate `usage` (token counts and the
      priced `cost_usd`) and a measured `latency_ms`. It must not set
      `fallback_from`; only the failover layer that reroutes a failed request
      may set that field.
    - `complete` raises on failure rather than returning a degraded response, so
      the failover layer can distinguish a retryable error from a real answer.
    - `health()` never raises. It reports reachability only and returns False on
      any error, so callers can use it as a circuit-breaker probe.
    """

    name: str

    def supports(self, model: str) -> bool: ...
    async def complete(self, req: CompletionRequest, model: str) -> CompletionResponse: ...
    async def health(self) -> bool: ...


class GuardVerdict(BaseModel):
    allowed: bool
    risk_score: float  # 0.0-1.0
    categories: list[str]  # ["pii:email", "injection:instruction_override"]
    redacted_text: str | None = None
    entity_map: dict[str, str] = Field(default_factory=dict)  # placeholder -> original


@runtime_checkable
class Guard(Protocol):
    """A synchronous safety check applied to text on ingress and egress.

    Invariants an implementer must uphold:

    - `name` is a stable, lowercase identifier unique across guards
      (e.g. "pii", "injection"). Verdicts are logged under it.
    - `inspect(text)` is pure: it must not mutate its argument or any shared
      state, and calling it twice with the same text must return an equivalent
      verdict. It is called on the hot request path and must not perform
      unbounded work.
    - `inspect` never raises. A guard that cannot reach a dependency must return
      a verdict rather than propagate an exception; fail-closed policy is
      expressed as `allowed=False`, not as an error.
    - `risk_score` is within 0.0-1.0 inclusive, and `categories` entries are
      `"<family>:<subtype>"` strings. `allowed=False` implies at least one
      category explaining the block.
    - If `redacted_text` is set, it is a full replacement for the inspected text
      and `entity_map` maps every placeholder appearing in it back to the exact
      original substring, so egress rehydration is lossless. If `redacted_text`
      is None, `entity_map` is empty.
    """

    name: str

    def inspect(self, text: str) -> GuardVerdict: ...


class JudgeScore(BaseModel):
    rubric: str
    score: float  # 0.0-1.0
    reasoning: str
    passed: bool
