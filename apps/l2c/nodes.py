"""The five node implementations.

**What a model is allowed to decide.** Nothing that moves money. Every number
and every branch in the approval path is computed in Python from the request and
the policy bands; the model is asked for extraction and for prose. A quote is
approved or escalated by `approval_router`, which does arithmetic against
`policies.DISCOUNT_BANDS` and does not call a model at all.

That is a deliberate design choice rather than a limitation of the mock
provider. An approval that depends on a sampled token is not an approval —
re-running it could flip the answer, and no policy citation would be
reproducible. So the model does the parts where being approximately right is
fine (normalising a payload, narrating an analysis, explaining a decision to a
human) and the deterministic core does the part where it is not.

The consequence for the demo is the good kind: the workflow produces the same
decision and the same citation against the mock provider as it would against a
frontier model, so a reviewer can run it with zero keys and still be looking at
the real control flow.

Every node that does call a model goes through `ConduitClient`, tagged with a
`workflow` value that the router turns into a tier and telemetry turns into a
column. The tags and their tiers are fixed in `NODE_WORKFLOWS` below.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from conduit.contracts import CompletionResponse

from .client import ConduitClient
from .models import (
    Decision,
    DiscountAnalysis,
    IntakeResult,
    NodeCall,
    PolicyEvidence,
    QuoteRequest,
)
from .policies import PARTNER_FEE_PCT, band_for, section_by_id
from .rag import PolicyIndex

__all__ = [
    "NODE_WORKFLOWS",
    "NodeContext",
    "QuoteState",
    "approval_router_node",
    "discount_analyst_node",
    "explainer_node",
    "intake_node",
    "policy_rag_node",
]

#: node name -> the `workflow` metadata tag it sends to Conduit. These strings
#: are not free: `conduit.router.settings.DEFAULT_WORKFLOW_HINTS` maps them to
#: tiers, so renaming one here silently un-tiers that node.
NODE_WORKFLOWS: dict[str, str] = {
    "intake": "intake",
    "policy_rag": "policy_rag",
    "discount_analyst": "discount_analyst",
    "explainer": "explainer",
}

#: How many clauses the RAG node retrieves.
RETRIEVAL_K = 4


@dataclass
class NodeContext:
    """Collaborators the nodes need that are not part of the graph's state."""

    client: ConduitClient
    index: PolicyIndex
    retrieval_k: int = RETRIEVAL_K


@dataclass
class QuoteState:
    """The graph's state. Nodes return a partial update of these fields."""

    request: QuoteRequest
    intake: IntakeResult | None = None
    evidence: list[PolicyEvidence] = field(default_factory=list)
    analysis: DiscountAnalysis | None = None
    decision: Decision | None = None
    calls: list[NodeCall] = field(default_factory=list)


def _record(node: str, response: CompletionResponse, workflow: str) -> NodeCall:
    """Fold a `CompletionResponse` into the per-node routing record."""
    return NodeCall(
        node=node,
        workflow=workflow,
        tier=response.routed_tier.value,
        model=response.model,
        provider=response.provider,
        cost_usd=response.usage.cost_usd,
        latency_ms=response.latency_ms,
        fallback_from=list(response.fallback_from),
    )


def _extract_json(text: str) -> dict[str, object] | None:
    """First balanced JSON object in a model response, or None.

    Model output arrives wrapped in prose, prefixes and code fences depending on
    the provider, so scanning for a balanced brace span is more robust than
    `json.loads` on the whole string. Strings are tracked so a brace inside a
    quoted value does not end the span early.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None


# --------------------------------------------------------------------------- #
# 1. Intake
# --------------------------------------------------------------------------- #

_INTAKE_SYSTEM = (
    "You normalise quote-approval requests. Extract the quote id, opportunity id, "
    "account name, discount percentage, annual list amount, term in months and "
    "support tier. Return only a JSON object with those keys and no explanation."
)


async def intake_node(state: QuoteState, ctx: NodeContext) -> dict[str, object]:
    """Normalise the request. Trivial tier: extraction, no reasoning.

    The parsed request stays authoritative for every number. The model's answer
    is used to set `model_confirmed` — a cheap agreement check that shows the
    extraction round-tripped — and never to overwrite a field, because a
    hallucinated discount would silently change the approval band.
    """
    request = state.request
    payload = {
        "quote_id": request.quote_id,
        "opportunity_id": request.opportunity_id,
        "account_name": request.account_name,
        "discount_pct": request.discount_pct,
        "list_amount_usd": request.list_amount_usd,
        "term_months": request.term_months,
        "support_tier": request.support_tier,
    }
    workflow = NODE_WORKFLOWS["intake"]
    response = await ctx.client.complete(
        workflow=workflow,
        system=_INTAKE_SYSTEM,
        user=f"Normalise this quote request:\n{json.dumps(payload, sort_keys=True)}",
        max_tokens=256,
    )
    echoed = _extract_json(response.text) or {}
    confirmed = str(echoed.get("quote_id", "")) == request.quote_id

    intake = IntakeResult(
        quote_id=request.quote_id,
        opportunity_id=request.opportunity_id,
        account_name=request.account_name,
        discount_pct=request.discount_pct,
        list_amount_usd=request.list_amount_usd,
        term_months=request.term_months,
        support_tier=request.support_tier,
        model_confirmed=confirmed,
        summary=(
            f"{request.account_name} requests {request.discount_pct:g}% off "
            f"${request.list_amount_usd:,.0f} on a {request.term_months}-month term "
            f"with {request.support_tier} support."
        ),
    )
    return {"intake": intake, "calls": [*state.calls, _record("intake", response, workflow)]}


# --------------------------------------------------------------------------- #
# 2. Policy RAG
# --------------------------------------------------------------------------- #

_RAG_SYSTEM = (
    "You summarise the governing clause of a discount policy for a sales operations "
    "reader. Quote the threshold and name the approver. Two sentences at most."
)


async def policy_rag_node(state: QuoteState, ctx: NodeContext) -> dict[str, object]:
    """Retrieve the clauses that bear on this request, then condense them.

    Retrieval is local and happens before the model call, so what the model sees
    is the corpus rather than its own recollection of one. The retrieved chunks
    travel on in state; the model's summary is narrative only.
    """
    intake = state.intake
    if intake is None:  # pragma: no cover - graph order guarantees this
        raise RuntimeError("policy_rag ran before intake")

    query = (
        f"discount of {intake.discount_pct:g}% of list price approval required "
        f"{intake.term_months}-month term {intake.support_tier} support "
        "written approval threshold band"
    )
    evidence = ctx.index.search(query, k=ctx.retrieval_k)

    corpus = "\n\n".join(f"[{item.section_id}] {item.title}: {item.text}" for item in evidence)
    workflow = NODE_WORKFLOWS["policy_rag"]
    response = await ctx.client.complete(
        workflow=workflow,
        system=_RAG_SYSTEM,
        user=(
            f"Request: {intake.summary}\n\nRetrieved policy clauses:\n{corpus}\n\n"
            "Which clause governs this discount, and what does it require?"
        ),
        max_tokens=384,
    )
    return {
        "evidence": evidence,
        "calls": [*state.calls, _record("policy_rag", response, workflow)],
    }


# --------------------------------------------------------------------------- #
# 3. Discount analyst
# --------------------------------------------------------------------------- #

_ANALYST_SYSTEM = (
    "You are a deal desk analyst. Given a quote's arithmetic and the governing "
    "policy band, explain the margin impact and whether the request is defensible. "
    "Be concise and do not restate the numbers you were given."
)


def compute_analysis(request: QuoteRequest) -> DiscountAnalysis:
    """The arithmetic, in one place, with no model in the loop.

    Separated from the node so the numbers can be unit-tested directly and so
    the approval router can be re-derived from a request alone.
    """
    net = request.net_amount_usd
    discount_value = round(request.list_amount_usd - net, 2)
    partner_fee = round(net * PARTNER_FEE_PCT / 100.0, 2) if request.partner_involved else 0.0
    band = band_for(request.discount_pct)
    return DiscountAnalysis(
        list_amount_usd=request.list_amount_usd,
        discount_pct=request.discount_pct,
        net_amount_usd=net,
        discount_value_usd=discount_value,
        partner_fee_usd=partner_fee,
        net_after_partner_fee_usd=round(net - partner_fee, 2),
        band_section_id=band.section_id,
        required_approver=band.approver,
        auto_approvable=band.auto_approvable,
    )


async def discount_analyst_node(state: QuoteState, ctx: NodeContext) -> dict[str, object]:
    """Price the request and read it against the band. Complex tier."""
    request = state.request
    analysis = compute_analysis(request)
    band_section = section_by_id(analysis.band_section_id)
    band_text = band_section.text if band_section else ""

    workflow = NODE_WORKFLOWS["discount_analyst"]
    response = await ctx.client.complete(
        workflow=workflow,
        system=_ANALYST_SYSTEM,
        user=(
            f"Quote {request.quote_id} for {request.account_name}.\n"
            f"List ${analysis.list_amount_usd:,.2f}; discount {analysis.discount_pct:g}% "
            f"(${analysis.discount_value_usd:,.2f}); net ${analysis.net_amount_usd:,.2f}; "
            f"partner fee ${analysis.partner_fee_usd:,.2f}; "
            f"net after fee ${analysis.net_after_partner_fee_usd:,.2f}.\n"
            f"Term {request.term_months} months. "
            f"Justification: {request.justification or 'none stated'}.\n"
            f"Governing clause [{analysis.band_section_id}]: {band_text}\n\n"
            "Analyse the margin impact and weigh whether this discount is justified."
        ),
        max_tokens=512,
    )
    analysis = analysis.model_copy(update={"narrative": response.text})
    return {
        "analysis": analysis,
        "calls": [*state.calls, _record("discount_analyst", response, workflow)],
    }


# --------------------------------------------------------------------------- #
# 4. Approval router  (deterministic — no model call)
# --------------------------------------------------------------------------- #


def _ensure_cited(evidence: Sequence[PolicyEvidence], section_id: str) -> list[PolicyEvidence]:
    """Guarantee the cited clause is among the evidence.

    Retrieval ranks by lexical similarity and the citation comes from the band
    table, so the two can disagree — most easily when a request's wording looks
    like a different clause than its arithmetic falls under. The citation is the
    authority, so it is inserted at the front when retrieval missed it rather
    than being quietly dropped from the evidence a reader sees.
    """
    if any(item.section_id == section_id for item in evidence):
        return list(evidence)
    section = section_by_id(section_id)
    if section is None:  # pragma: no cover - bands reference authored sections
        return list(evidence)
    cited = PolicyEvidence(
        section_id=section.section_id,
        doc_id=section.doc_id,
        title=section.title,
        text=section.text,
        score=0.0,
    )
    return [cited, *evidence]


def decide(request: QuoteRequest, analysis: DiscountAnalysis) -> tuple[str, str, list[str]]:
    """Apply the policy. Returns `(outcome, approver, reasons)`.

    Escalation has three independent triggers, and any one of them is enough:

    * the discount falls outside the account executive's band (DISC-4.1);
    * the request states no justification while asking for an exception, which
      DISC-4.6 sends back rather than forward;
    * the term is missing, which DISC-4.7 requires on every exception request.
    """
    reasons: list[str] = []
    band = band_for(analysis.discount_pct)

    if band.auto_approvable:
        reasons.append(
            f"{analysis.discount_pct:g}% is within the account executive band "
            f"(up to {band.high_pct:g}% of list) under {band.section_id}."
        )
        return "approve", band.approver, reasons

    reasons.append(
        f"{analysis.discount_pct:g}% exceeds {band.low_pct:g}% of list, which "
        f"{band.section_id} places in the {band.label.lower()} band."
    )
    if band.requires_written_approval:
        reasons.append(
            f"{band.section_id} requires written approval from the "
            f"{band.approver.replace('_', ' ')} before the quote is issued."
        )
    if not request.justification.strip():
        reasons.append(
            "DISC-4.6 requires a named commercial justification on an exception "
            "request; none was stated, so the request returns to the account executive."
        )
    if request.term_months <= 0:  # pragma: no cover - QuoteRequest validates term > 0
        reasons.append("DISC-4.7 requires the contract term to be stated on every exception.")
    return "escalate", band.approver, reasons


async def approval_router_node(state: QuoteState, ctx: NodeContext) -> dict[str, object]:
    """Decide, and bind the decision to a clause that actually exists.

    This node makes no model call: see the module docstring. It is also the only
    place a citation is minted, and it refuses to mint one the index cannot
    resolve — an unresolvable citation is a bug in the corpus or the band table,
    and failing here is far better than shipping a decision that points at
    nothing.
    """
    request = state.request
    analysis = state.analysis
    if analysis is None:  # pragma: no cover - graph order guarantees this
        raise RuntimeError("approval_router ran before discount_analyst")

    outcome, approver, reasons = decide(request, analysis)
    citation = analysis.band_section_id
    section = ctx.index.resolve(citation)
    if section is None:
        raise RuntimeError(
            f"decision cited {citation!r}, which does not resolve to a chunk in the "
            f"policy index (indexed: {', '.join(ctx.index.section_ids)})"
        )

    decision = Decision(
        outcome=outcome,
        approver=approver,
        citation=section.section_id,
        citation_title=section.title,
        citation_quote=section.text,
        reasons=reasons,
        analysis=analysis,
        evidence=_ensure_cited(state.evidence, section.section_id),
    )
    return {"decision": decision}


# --------------------------------------------------------------------------- #
# 5. Explainer
# --------------------------------------------------------------------------- #

_EXPLAINER_SYSTEM = (
    "You explain a quote-approval decision to the account executive who requested it. "
    "State the outcome, name the approver, and quote the policy section that drove it. "
    "Three sentences at most, plain and specific."
)


async def explainer_node(state: QuoteState, ctx: NodeContext) -> dict[str, object]:
    """Narrate the decision that has already been made.

    The explainer never revisits the outcome; it renders one. `explanation` is
    the model's prose, while `reasons` and `citation` remain the machine-checked
    record of why.
    """
    decision = state.decision
    if decision is None:  # pragma: no cover - graph order guarantees this
        raise RuntimeError("explainer ran before approval_router")

    workflow = NODE_WORKFLOWS["explainer"]
    response = await ctx.client.complete(
        workflow=workflow,
        system=_EXPLAINER_SYSTEM,
        user=(
            f"Decision: {decision.outcome}. Approver: "
            f"{decision.approver.replace('_', ' ')}.\n"
            f"Policy [{decision.citation}] {decision.citation_title}: "
            f"{decision.citation_quote}\n"
            f"Reasons: {' '.join(decision.reasons)}\n\n"
            "Explain this decision to the account executive."
        ),
        max_tokens=384,
    )
    calls = [*state.calls, _record("explainer", response, workflow)]
    explained = decision.model_copy(update={"explanation": response.text, "calls": calls})
    return {"decision": explained, "calls": calls}
