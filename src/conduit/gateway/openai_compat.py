"""Translation between the OpenAI chat-completions shape and `contracts`.

AC-1 is "an unmodified OpenAI SDK client gets a valid completion", which sets a
higher bar than "accepts similar JSON": the SDK validates the response against
its own models, so anything it requires must be present and correctly typed.
That is why `id`, `object`, `created`, `choices[].index`, `finish_reason` and
the full `usage` block are all populated rather than approximated.

Extra request fields are accepted and ignored rather than rejected — real SDK
callers send `top_p`, `presence_penalty`, `n` and friends, and 400-ing on a
parameter Conduit simply does not vary would fail the drop-in claim on the first
request. `stream` is the one exception: it changes the response *shape*, so
ignoring it would hand the caller a body it cannot parse.

Routing detail the OpenAI schema has no room for — the tier, the provider, the
failover hops — rides in a non-standard `x_conduit` object. SDK response models
tolerate unknown fields, so it is visible to `curl` and invisible to the client.
"""

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from conduit.contracts import CompletionRequest, CompletionResponse, Message
from conduit.gateway.errors import BadRequestError, StreamingUnsupportedError

__all__ = [
    "ChatCompletionChoice",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "OpenAIUsage",
    "to_completion_request",
    "to_openai_response",
]

#: Roles `contracts.Message` accepts. The SDK can send others (`tool`,
#: `function`, `developer`); they are mapped or refused explicitly below.
_ROLE_ALIASES = {"developer": "system"}


class ChatMessage(BaseModel):
    """One OpenAI chat turn. `content` may arrive as content parts."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    """The subset of the OpenAI request Conduit acts on, plus tolerated extras."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, gt=0)
    # `max_completion_tokens` is the SDK's newer spelling of the same thing.
    max_completion_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    stream: bool | None = None
    user: str | None = None
    metadata: dict[str, str] | None = None


class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: OpenAIUsage
    #: Non-standard. Where the routing decision goes, since OpenAI has no field
    #: for "which tier did you pick and what failed first".
    x_conduit: dict[str, Any] = Field(default_factory=dict)


def _flatten_content(content: str | list[dict[str, Any]] | None) -> str:
    """Collapse content parts to text. Non-text parts are refused, not dropped.

    Silently discarding an image part would answer a question the caller did not
    ask; `contracts.Message.content` is a string, so multimodal input is a real
    limitation and is reported as one.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    pieces: list[str] = []
    for part in content:
        kind = part.get("type", "text")
        if kind != "text":
            raise BadRequestError(
                f"Unsupported content part {kind!r}. This gateway accepts text content only.",
                code="unsupported_content",
            )
        pieces.append(str(part.get("text", "")))
    return "".join(pieces)


def to_completion_request(payload: ChatCompletionRequest) -> CompletionRequest:
    """OpenAI request -> `contracts.CompletionRequest`."""
    if payload.stream:
        raise StreamingUnsupportedError()
    if not payload.messages:
        raise BadRequestError("`messages` must contain at least one message.")

    messages: list[Message] = []
    for turn in payload.messages:
        role = _ROLE_ALIASES.get(turn.role, turn.role)
        if role not in ("system", "user", "assistant"):
            raise BadRequestError(
                f"Unsupported message role {turn.role!r}. "
                "This gateway accepts system, user and assistant turns.",
                code="unsupported_role",
            )
        messages.append(Message(role=role, content=_flatten_content(turn.content)))

    metadata = dict(payload.metadata or {})
    if payload.user:
        # OpenAI's `user` is an end-user id for abuse tracing. Conduit's nearest
        # equivalent is the workflow tag telemetry slices on.
        metadata.setdefault("workflow", payload.user)

    defaults = CompletionRequest.model_fields
    max_tokens = payload.max_tokens or payload.max_completion_tokens
    return CompletionRequest(
        messages=messages,
        model=payload.model,
        max_tokens=max_tokens if max_tokens is not None else defaults["max_tokens"].default,
        temperature=(
            payload.temperature
            if payload.temperature is not None
            else defaults["temperature"].default
        ),
        metadata=metadata,
    )


def to_openai_response(
    response: CompletionResponse,
    *,
    created: int | None = None,
    trace_id: str | None = None,
    max_tokens: int | None = None,
) -> ChatCompletionResponse:
    """`contracts.CompletionResponse` -> the OpenAI response the SDK parses.

    `finish_reason` is inferred rather than invented: the contract carries no
    stop reason, so a completion that consumed its entire token budget reports
    "length" and everything else reports "stop". That is the one distinction a
    caller can act on, and it is derivable from what we have.
    """
    usage = response.usage
    finish: Literal["stop", "length"] = (
        "length" if max_tokens is not None and usage.completion_tokens >= max_tokens else "stop"
    )
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=created if created is not None else int(time.time()),
        model=response.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response.text),
                finish_reason=finish,
            )
        ],
        usage=OpenAIUsage(
            prompt_tokens=usage.prompt_tokens + usage.cache_read_tokens + usage.cache_write_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=(
                usage.prompt_tokens
                + usage.cache_read_tokens
                + usage.cache_write_tokens
                + usage.completion_tokens
            ),
        ),
        x_conduit={
            "provider": response.provider,
            "routed_tier": response.routed_tier.value,
            "fallback_from": list(response.fallback_from),
            "cost_usd": usage.cost_usd,
            "cache_read_tokens": usage.cache_read_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            **({"trace_id": trace_id} if trace_id else {}),
        },
    )
