"""Test doubles for the router.

Providers are a Protocol to the router, so the tests never import a concrete
provider and never touch the network: the transport is this file.
"""

from conduit.contracts import CompletionRequest, CompletionResponse, Complexity, Usage


class FakeProvider:
    """A `Provider` that returns scripted text and counts every call."""

    name = "fake"

    def __init__(self, *replies: str, fail: bool = False) -> None:
        self.replies = list(replies) or ["trivial"]
        self.fail = fail
        self.calls: list[tuple[CompletionRequest, str]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def supports(self, model: str) -> bool:
        return True

    async def complete(self, req: CompletionRequest, model: str) -> CompletionResponse:
        self.calls.append((req, model))
        if self.fail:
            raise RuntimeError("provider is down")
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return CompletionResponse(
            text=reply,
            model=model,
            provider=self.name,
            usage=Usage(prompt_tokens=10, completion_tokens=1, cost_usd=0.0),
            latency_ms=1,
            routed_tier=Complexity.TRIVIAL,
        )

    async def health(self) -> bool:
        return True
