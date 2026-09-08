"""Mock adapter — deterministic, offline, the backend CI actually runs against.

Determinism is the whole point: the same `CompletionRequest` must produce a
byte-identical answer on every machine and every run, so an eval suite or a
failover test can assert on the text instead of on its shape.

`metadata` is deliberately excluded from the digest. It carries `trace_id`,
which is different on every request; folding it in would make two logically
identical prompts produce different answers and quietly destroy the property
this provider exists to give.
"""

import hashlib
import json
from typing import Any

from conduit.contracts import CompletionRequest

from .base import BaseProvider, ProviderRefusedError, RawCompletion

__all__ = ["DEFAULT_MODEL", "MODEL_PREFIX", "MockProvider", "digest_request"]

DEFAULT_MODEL = "mock:echo"
MODEL_PREFIX = "mock:"

#: Rough characters-per-token, used only to make token counts a stable function
#: of the input. It is not a real tokenizer and does not claim to be.
CHARS_PER_TOKEN = 4


def digest_request(req: CompletionRequest, model: str) -> str:
    """A stable fingerprint of everything that should change the answer."""
    canonical = json.dumps(
        {
            "model": model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN) if text else 0


class MockProvider(BaseProvider):
    """Echoes the last turn with a fingerprint. Never touches the network."""

    name = "mock"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    async def _invoke(self, req: CompletionRequest, model: str) -> RawCompletion:
        if not req.messages:
            raise ProviderRefusedError(self.name, "request has no messages")
        fingerprint = digest_request(req, model)[:12]
        prompt = "\n".join(m.content for m in req.messages)
        last = req.messages[-1].content
        text = f"[{model} {fingerprint}] {last}"
        completion_tokens = min(req.max_tokens, _tokens(text))
        return RawCompletion(
            text=text,
            model=model,
            prompt_tokens=_tokens(prompt),
            completion_tokens=completion_tokens,
        )

    async def _probe(self) -> bool:
        return True
