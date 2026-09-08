"""Lead-to-Cash quote-approval workflow.

A LangGraph multi-agent graph over a synthetic GTM corpus. Each node calls
Conduit with a `workflow` metadata tag, so the router picks a different tier per
node and telemetry can show per-node cost for a single quote.

    from apps.l2c import ConduitClient, QuoteRequest, run_quote

    async with ConduitClient(app=create_app()) as client:
        decision = await run_quote(request, client=client)
        print(decision.outcome, decision.citation)
"""

from .client import ConduitClient, ConduitClientError
from .graph import build_graph, run_quote
from .models import Decision, DiscountAnalysis, PolicyEvidence, QuoteRequest
from .nodes import NodeContext, QuoteState
from .policies import DISCOUNT_BANDS, POLICY_SECTIONS, band_for
from .rag import HashingEmbedder, PolicyIndex, build_index

__all__ = [
    "DISCOUNT_BANDS",
    "POLICY_SECTIONS",
    "ConduitClient",
    "ConduitClientError",
    "Decision",
    "DiscountAnalysis",
    "HashingEmbedder",
    "NodeContext",
    "PolicyEvidence",
    "PolicyIndex",
    "QuoteRequest",
    "QuoteState",
    "band_for",
    "build_graph",
    "build_index",
    "run_quote",
]
