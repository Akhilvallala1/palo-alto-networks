"""Fixtures for the L2C suite: a stub Conduit client and request builders.

The stub replaces only the HTTP hop to the gateway. Retrieval, the arithmetic
and the approval logic stay real in every unit test, because those are the parts
the acceptance criteria are about — stubbing them would test the stub.

`StubClient` records what each node asked for, so a test can assert on the
`workflow` tag a node sent (which is what routes it) without a gateway.
"""

from collections.abc import Sequence

from apps.l2c.models import QuoteRequest
from conduit.contracts import CompletionResponse, Complexity, Usage

__all__ = ["StubClient", "make_request"]

#: Workflow tag -> the tier `conduit.router.settings.DEFAULT_WORKFLOW_HINTS`
#: assigns it. Mirrored here so a unit test can hand back a plausible
#: `routed_tier` without standing up a router.
STUB_TIERS: dict[str, Complexity] = {
    "intake": Complexity.TRIVIAL,
    "policy_rag": Complexity.STANDARD,
    "discount_analyst": Complexity.COMPLEX,
    "explainer": Complexity.STANDARD,
}


class StubClient:
    """A `ConduitClient` stand-in that answers from a script.

    Duck-typed rather than a subclass: the nodes only ever call `complete`, and
    a subclass would drag an `httpx.AsyncClient` into every unit test.
    """

    def __init__(self, replies: Sequence[str] | str = "stub reply") -> None:
        self._replies = [replies] if isinstance(replies, str) else list(replies)
        self.calls: list[dict[str, object]] = []

    @property
    def workflows(self) -> list[str]:
        return [str(call["workflow"]) for call in self.calls]

    async def complete(
        self,
        *,
        workflow: str,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> CompletionResponse:
        self.calls.append(
            {
                "workflow": workflow,
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        index = min(len(self.calls) - 1, len(self._replies) - 1)
        return CompletionResponse(
            text=self._replies[index],
            model="mock:echo",
            provider="mock",
            usage=Usage(prompt_tokens=10, completion_tokens=5, cost_usd=0.0),
            latency_ms=1,
            routed_tier=STUB_TIERS.get(workflow, Complexity.STANDARD),
        )

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> "StubClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


def make_request(
    *,
    discount_pct: float = 26.0,
    list_amount_usd: float = 840_000.0,
    partner_involved: bool = True,
    justification: str = "competitive displacement",
    term_months: int = 24,
    support_tier: str = "premium",
) -> QuoteRequest:
    """A synthetic quote request. Defaults land in the escalation band."""
    return QuoteRequest(
        request_id="REQ-30007",
        account_id="ACC-10003",
        account_name="Northwind Traders",
        opportunity_id="OPP-80007",
        quote_id="Q-2026-00107",
        list_amount_usd=list_amount_usd,
        discount_pct=discount_pct,
        term_months=term_months,
        support_tier=support_tier,  # type: ignore[arg-type]
        partner_involved=partner_involved,
        justification=justification,
        requested_by="Jordan Ellis",
        requested_by_email="jellis@northwind.example",
    )
