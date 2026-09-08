"""Ollama adapter — the zero-key local backend.

This is what makes the repo runnable by a reviewer who has no API keys, and
what gives failover somewhere real to fall to when Anthropic is down. Model ids
are namespaced `ollama:<tag>` in the registry so they cannot collide with a
hosted model of the same name; the prefix is stripped on the wire.
"""

import os
from collections.abc import Mapping
from typing import Any

from conduit.contracts import CompletionRequest

from .base import (
    HttpProvider,
    ProviderRefusedError,
    RawCompletion,
    as_int,
    as_mapping,
    as_str,
)

__all__ = ["DEFAULT_BASE_URL", "MODEL_PREFIX", "OllamaProvider"]

DEFAULT_BASE_URL = "http://localhost:11434"
BASE_URL_ENV = "OLLAMA_HOST"
MODEL_PREFIX = "ollama:"


class OllamaProvider(HttpProvider):
    """Local `/api/chat`. Free, so every priced row for it is 0.00."""

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        env: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        source = os.environ if env is None else env
        super().__init__(
            base_url=base_url or source.get(BASE_URL_ENV) or DEFAULT_BASE_URL,
            headers={"content-type": "application/json"},
            **kwargs,
        )

    @staticmethod
    def wire_model(model: str) -> str:
        """`ollama:llama3.2` -> `llama3.2`, the tag Ollama itself knows."""
        return model[len(MODEL_PREFIX) :] if model.startswith(MODEL_PREFIX) else model

    async def _invoke(self, req: CompletionRequest, model: str) -> RawCompletion:
        if not req.messages:
            raise ProviderRefusedError(self.name, "request has no messages")
        payload: dict[str, Any] = {
            "model": self.wire_model(model),
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "stream": False,
            "options": {"temperature": req.temperature, "num_predict": req.max_tokens},
        }
        body = await self._post("/api/chat", payload)
        message = as_mapping(body.get("message"))
        return RawCompletion(
            text=as_str(message.get("content")),
            model=model,
            prompt_tokens=as_int(body.get("prompt_eval_count")),
            completion_tokens=as_int(body.get("eval_count")),
        )

    async def _probe(self) -> bool:
        await self._get("/api/tags")
        return True
