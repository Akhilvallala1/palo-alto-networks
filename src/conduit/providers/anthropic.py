"""Anthropic adapter — the live, paid backend.

Talks to the Messages API over HTTP rather than through the `anthropic` SDK.
The SDK stays an optional extra: the epic forbids vendor SDKs as core
dependencies (AC-16), and a plain httpx client means CI can exercise the real
request shaping and the real status-code handling through
`httpx.MockTransport`, with no key and no socket.

Anthropic reports cache tokens separately from `input_tokens`, which lines up
exactly with `contracts.Usage`, where `prompt_tokens` is the uncached
remainder.
"""

import os
from collections.abc import Mapping
from typing import Any

from conduit.contracts import CompletionRequest

from .base import (
    HttpProvider,
    ProviderRefusedError,
    ProviderUnavailableError,
    RawCompletion,
    as_int,
    as_mapping,
    as_str,
    split_system,
)

__all__ = ["API_KEY_ENV", "API_VERSION", "DEFAULT_BASE_URL", "AnthropicProvider"]

DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"
API_KEY_ENV = "ANTHROPIC_API_KEY"
BASE_URL_ENV = "ANTHROPIC_BASE_URL"


class AnthropicProvider(HttpProvider):
    """Serves `claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5-20251001`.

    Which ids exactly is not decided here: the registry passes in the rows of
    `config/models.yaml` whose provider is `anthropic`, so adding a model is a
    config edit, not a code edit.
    """

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        env: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        source = os.environ if env is None else env
        key = api_key if api_key is not None else source.get(API_KEY_ENV, "")
        super().__init__(
            base_url=base_url or source.get(BASE_URL_ENV) or DEFAULT_BASE_URL,
            headers={
                "x-api-key": key,
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            },
            **kwargs,
        )

    async def _invoke(self, req: CompletionRequest, model: str) -> RawCompletion:
        system, turns = split_system(req.messages)
        if not turns:
            raise ProviderRefusedError(self.name, "request has no user or assistant turns")
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "messages": [{"role": m.role, "content": m.content} for m in turns],
        }
        if system is not None:
            payload["system"] = system

        body = await self._post("/v1/messages", payload)
        blocks = body.get("content")
        if not isinstance(blocks, list):
            raise ProviderUnavailableError(self.name, "messages reply has no content array")
        text = "".join(
            as_str(as_mapping(block).get("text"))
            for block in blocks
            if as_mapping(block).get("type") == "text"
        )
        usage = as_mapping(body.get("usage"))
        return RawCompletion(
            text=text,
            model=as_str(body.get("model")) or model,
            prompt_tokens=as_int(usage.get("input_tokens")),
            completion_tokens=as_int(usage.get("output_tokens")),
            cache_read_tokens=as_int(usage.get("cache_read_input_tokens")),
            cache_write_tokens=as_int(usage.get("cache_creation_input_tokens")),
        )

    async def _probe(self) -> bool:
        await self._get("/v1/models", params={"limit": "1"})
        return True
