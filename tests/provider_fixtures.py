"""Fake vendor APIs for the provider tests.

Every adapter takes an injectable `httpx` transport, so these fixtures give the
real request shaping and the real status-code handling with no socket and no
key. Nothing in `tests/` may open a network connection.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx

from conduit.config import ModelsConfig, load_models
from conduit.contracts import CompletionRequest, Message
from conduit.providers.base import RetryPolicy

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config"

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5"
OPUS = "claude-opus-5"
LLAMA = "ollama:llama3.2"
ECHO = "mock:echo"
GPT = "gpt-4.1"
FLASH = "gemini-2.5-flash"

#: Retries stay on (the policy is under test) but cost no wall-clock time.
INSTANT_RETRY = RetryPolicy(attempts=3, base_delay_s=0.0, max_delay_s=0.0)
NO_RETRY = RetryPolicy(attempts=1)


def repo_models() -> ModelsConfig:
    return load_models(REPO_CONFIG, env={})


def request(
    content: str = "Draft a renewal quote for Acme.",
    *,
    system: str | None = None,
    **kwargs: Any,
) -> CompletionRequest:
    messages = [Message(role="user", content=content)]
    if system is not None:
        messages.insert(0, Message(role="system", content=system))
    return CompletionRequest(messages=messages, **kwargs)


async def no_sleep(_: float) -> None:
    """A `Sleeper` that never yields wall-clock time."""


# --------------------------------------------------------------------------- #
# Canned vendor payloads
# --------------------------------------------------------------------------- #


def anthropic_body(
    text: str = "ok",
    *,
    model: str = HAIKU,
    input_tokens: int = 120,
    output_tokens: int = 40,
    cache_read: int = 0,
    cache_write: int = 0,
) -> dict[str, Any]:
    usage: dict[str, Any] = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    if cache_read:
        usage["cache_read_input_tokens"] = cache_read
    if cache_write:
        usage["cache_creation_input_tokens"] = cache_write
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": usage,
    }


def ollama_body(
    text: str = "ok", *, model: str = "llama3.2", prompt_tokens: int = 120, eval_tokens: int = 40
) -> dict[str, Any]:
    return {
        "model": model,
        "message": {"role": "assistant", "content": text},
        "done": True,
        "prompt_eval_count": prompt_tokens,
        "eval_count": eval_tokens,
    }


def openai_body(
    text: str = "ok",
    *,
    model: str = GPT,
    prompt_tokens: int = 120,
    completion_tokens: int = 40,
    cached_tokens: int = 0,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        },
    }


def gemini_body(
    text: str = "ok",
    *,
    prompt_tokens: int = 120,
    completion_tokens: int = 40,
    cached_tokens: int = 0,
) -> dict[str, Any]:
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": completion_tokens,
            "cachedContentTokenCount": cached_tokens,
        },
    }


# --------------------------------------------------------------------------- #
# Fake transport
# --------------------------------------------------------------------------- #

Responder = Callable[[httpx.Request], httpx.Response]


class FakeApi:
    """Routes by URL-path substring, records every request, counts calls.

    Register several responses for one route and they are consumed in order;
    the last one repeats forever. That is how "fail twice, then succeed" and
    "always 500" are both expressed without a stateful closure per test.
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._routes: list[tuple[str, list[Responder]]] = []

    def on(self, path_fragment: str, *responses: Responder | httpx.Response) -> "FakeApi":
        queue: list[Responder] = [
            (lambda _r, resp=r: resp) if isinstance(r, httpx.Response) else r  # type: ignore[misc]
            for r in responses
        ]
        self._routes.append((path_fragment, queue))
        return self

    def json(self, path_fragment: str, body: Mapping[str, Any], status: int = 200) -> "FakeApi":
        return self.on(path_fragment, httpx.Response(status, json=dict(body)))

    def fail(self, path_fragment: str, status: int, times: int | None = None) -> "FakeApi":
        error = httpx.Response(status, json={"error": {"message": f"injected {status}"}})
        if times is None:
            return self.on(path_fragment, error)
        return self.on(path_fragment, *([error] * times))

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def count(self, path_fragment: str = "") -> int:
        return sum(1 for r in self.requests if path_fragment in r.url.path)

    def last(self, path_fragment: str = "") -> httpx.Request:
        matches = [r for r in self.requests if path_fragment in r.url.path]
        if not matches:
            raise AssertionError(f"no request matched {path_fragment!r}")
        return matches[-1]

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        for fragment, queue in self._routes:
            if fragment in request.url.path:
                responder = queue[0] if len(queue) == 1 else queue.pop(0)
                return responder(request)
        raise AssertionError(f"unrouted request: {request.method} {request.url}")


def timeout(_: httpx.Request) -> httpx.Response:
    """A responder that behaves like a read timeout."""
    raise httpx.ReadTimeout("injected timeout")


def boom(_: httpx.Request) -> httpx.Response:
    """A responder that behaves like a dropped connection."""
    raise httpx.ConnectError("injected connection failure")
