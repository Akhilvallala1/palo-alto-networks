"""Shapes passed between the L2C nodes.

These are the workflow's own types. Nothing here touches
`conduit.contracts` — that module describes the gateway's wire protocol, and
the graph talks to the gateway through `ConduitClient`, so the two vocabularies
stay separate and neither drifts into the other.

Every amount is USD and every percentage is a whole number out of 100 (``22.5``
means 22.5%), which is how the seeded quotes and the policy bands both express
them. Mixing the two conventions is the obvious way to get a silently wrong
approval, so it is stated once here and never converted anywhere else.
"""

from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "Decision",
    "DiscountAnalysis",
    "IntakeResult",
    "NodeCall",
    "PolicyEvidence",
    "QuoteRequest",
]

Outcome = Literal["approve", "escalate"]
SupportTier = Literal["standard", "premium"]


class QuoteRequest(BaseModel):
    """A quote-approval request as it arrives from the CLI or a seeded file."""

    request_id: str
    account_id: str
    account_name: str
    opportunity_id: str
    quote_id: str
    list_amount_usd: float = Field(gt=0)
    discount_pct: float = Field(ge=0, le=100)
    term_months: int = Field(gt=0)
    support_tier: SupportTier = "standard"
    partner_involved: bool = False
    justification: str = ""
    #: Synthetic contact details. These are the fields that exercise the
    #: gateway's PII guard: the vendor sees placeholders, the caller sees names.
    requested_by: str = ""
    requested_by_email: str = ""

    @property
    def net_amount_usd(self) -> float:
        """List less the requested discount. The number the bands are read against."""
        return round(self.list_amount_usd * (1.0 - self.discount_pct / 100.0), 2)


class NodeCall(BaseModel):
    """What one node's trip through Conduit cost and how it was routed.

    Collected per node so the CLI and the tests can show tier-per-node without
    re-reading the telemetry store, and so a node that never calls a model is
    visibly absent rather than silently zero.
    """

    node: str
    workflow: str
    tier: str
    model: str
    provider: str
    cost_usd: float
    latency_ms: int
    trace_id: str = ""
    fallback_from: list[str] = Field(default_factory=list)


class IntakeResult(BaseModel):
    """The normalised request fields the rest of the graph reads."""

    quote_id: str
    opportunity_id: str
    account_name: str
    discount_pct: float
    list_amount_usd: float
    term_months: int
    support_tier: SupportTier
    #: True when the model's echo of the payload agreed with the parsed request.
    model_confirmed: bool = False
    summary: str = ""


class PolicyEvidence(BaseModel):
    """One retrieved policy chunk, kept with the score that surfaced it."""

    section_id: str
    doc_id: str
    title: str
    text: str
    score: float


class DiscountAnalysis(BaseModel):
    """The arithmetic, done in Python, plus the model's narrative reading of it."""

    list_amount_usd: float
    discount_pct: float
    net_amount_usd: float
    discount_value_usd: float
    partner_fee_usd: float
    net_after_partner_fee_usd: float
    #: The band the discount falls in, by `section_id`.
    band_section_id: str
    required_approver: str
    auto_approvable: bool
    narrative: str = ""


class Decision(BaseModel):
    """The workflow's verdict, and the clause that produced it.

    `citation` is a `section_id` that must resolve to a real chunk in the
    indexed corpus. `PolicyIndex.resolve` is what makes that checkable rather
    than decorative.
    """

    outcome: Outcome
    approver: str
    citation: str
    citation_title: str
    citation_quote: str
    reasons: list[str] = Field(default_factory=list)
    analysis: DiscountAnalysis
    explanation: str = ""
    evidence: list[PolicyEvidence] = Field(default_factory=list)
    calls: list[NodeCall] = Field(default_factory=list)

    @property
    def escalated(self) -> bool:
        return self.outcome == "escalate"
