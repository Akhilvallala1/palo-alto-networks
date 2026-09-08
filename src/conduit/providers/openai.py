"""OpenAI adapter — written, dormant.

Dormant means the code is real and tested but the registry only instantiates it
when `OPENAI_API_KEY` is set. That is the honest form of "model-vendor
independence": a second hosted vendor that is one env var away from live, not a
TODO comment claiming it would be easy.

One shape mismatch worth naming: OpenAI's `usage.prompt_tokens` *includes*
cached tokens, while `contracts.Usage.prompt_tokens` is the uncached remainder.
We subtract, so cost math stays comparable across vendors.
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
)

__all__ = ["API_KEY_ENV", "DEFAULT_BASE_URL", "OpenAIProvider"]

DEFAULT_BASE_URL = "https://api.openai.com"
API_KEY_ENV = "OPENAI_API_KEY"
BASE_URL_ENV = "OPENAI_BASE_URL"


class OpenAIProvider(HttpProvider):
    """Chat Completions. Registered only when `OPENAI_API_KEY` is present."""

    name = "openai"

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
                "authorization": f"Bearer {key}",
                "content-type": "application/json",
            },
            **kwargs,
        )

    async def _invoke(self, req: CompletionRequest, model: str) -> RawCompletion:
        if not req.messages:
            raise ProviderRefusedError(self.name, "request has no messages")
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "max_completion_tokens": req.max_tokens,
            "temperature": req.temperature,
        }
        body = await self._post("/v1/chat/completions", payload)
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderUnavailableError(self.name, "chat completion reply has no choices")
        message = as_mapping(as_mapping(choices[0]).get("message"))

        usage = as_mapping(body.get("usage"))
        cached = as_int(as_mapping(usage.get("prompt_tokens_details")).get("cached_tokens"))
        return RawCompletion(
            text=as_str(message.get("content")),
            model=as_str(body.get("model")) or model,
            prompt_tokens=max(0, as_int(usage.get("prompt_tokens")) - cached),
            completion_tokens=as_int(usage.get("completion_tokens")),
            cache_read_tokens=cached,
        )

    async def _probe(self) -> bool:
        await self._get("/v1/models")
        return True
