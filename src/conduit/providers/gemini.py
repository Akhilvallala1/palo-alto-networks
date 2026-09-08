"""Gemini adapter — written, dormant.

Registered only when `GOOGLE_API_KEY` is set. Gemini's wire shape diverges more
than the others: turns are `contents` with `parts`, the assistant role is
called `model`, and system prompts go in `systemInstruction`. All of that is
translated here so nothing above `providers/` ever learns about it.

As with OpenAI, `promptTokenCount` includes cached tokens and
`contracts.Usage.prompt_tokens` does not, so we subtract.
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

__all__ = ["API_KEY_ENV", "DEFAULT_BASE_URL", "GeminiProvider"]

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
API_KEY_ENV = "GOOGLE_API_KEY"
BASE_URL_ENV = "GOOGLE_BASE_URL"

_ROLES = {"user": "user", "assistant": "model"}


class GeminiProvider(HttpProvider):
    """`generateContent`. Registered only when `GOOGLE_API_KEY` is present."""

    name = "gemini"

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
            headers={"x-goog-api-key": key, "content-type": "application/json"},
            **kwargs,
        )

    async def _invoke(self, req: CompletionRequest, model: str) -> RawCompletion:
        system, turns = split_system(req.messages)
        if not turns:
            raise ProviderRefusedError(self.name, "request has no user or assistant turns")
        payload: dict[str, Any] = {
            "contents": [{"role": _ROLES[m.role], "parts": [{"text": m.content}]} for m in turns],
            "generationConfig": {
                "temperature": req.temperature,
                "maxOutputTokens": req.max_tokens,
            },
        }
        if system is not None:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        body = await self._post(f"/v1beta/models/{model}:generateContent", payload)
        candidates = body.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ProviderUnavailableError(self.name, "generateContent reply has no candidates")
        parts = as_mapping(as_mapping(candidates[0]).get("content")).get("parts")
        text = (
            "".join(as_str(as_mapping(part).get("text")) for part in parts)
            if isinstance(parts, list)
            else ""
        )

        usage = as_mapping(body.get("usageMetadata"))
        cached = as_int(usage.get("cachedContentTokenCount"))
        return RawCompletion(
            text=text,
            model=model,
            prompt_tokens=max(0, as_int(usage.get("promptTokenCount")) - cached),
            completion_tokens=as_int(usage.get("candidatesTokenCount")),
            cache_read_tokens=cached,
        )

    async def _probe(self) -> bool:
        await self._get("/v1beta/models")
        return True
