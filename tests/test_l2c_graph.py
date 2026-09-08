"""Integration tests: the whole graph, on seeded data, with retrieval real.

Only the HTTP hop to the gateway is stubbed here; `tests/test_l2c_e2e.py` takes
that away too and drives the real gateway app.
"""

import pytest

from apps.l2c.graph import NODE_SEQUENCE, build_graph, evidence_section_ids, run_quote
from apps.l2c.nodes import NodeContext, QuoteState
from apps.l2c.rag import build_index
from tests.l2c_fixtures import StubClient, make_request


@pytest.fixture(scope="module")
def index():
    idx = build_index()
    yield idx
    idx.close()


async def test_full_graph_escalates_and_cites_a_resolvable_clause(index) -> None:
    """AC-1 and AC-3: an end-to-end run produces a decision, above threshold."""
    decision = await run_quote(make_request(discount_pct=26.0), client=StubClient(), index=index)

    assert decision.outcome == "escalate"
    assert decision.approver == "regional_vice_president"
    assert decision.citation == "DISC-4.3"
    assert index.resolve(decision.citation) is not None
    assert decision.reasons
    assert decision.explanation
    # The cited clause is among the evidence a reader is shown.
    assert decision.citation in evidence_section_ids(decision.evidence)


async def test_full_graph_approves_a_small_discount(index) -> None:
    decision = await run_quote(
        make_request(discount_pct=8.0, partner_involved=False), client=StubClient(), index=index
    )

    assert decision.outcome == "approve"
    assert decision.approver == "account_executive"
    assert decision.citation == "DISC-4.1"
    assert index.resolve(decision.citation) is not None
    assert decision.analysis.partner_fee_usd == 0.0


async def test_the_four_model_calls_land_on_distinct_tiers(index) -> None:
    """AC-4: intake and the analyst must not share a tier."""
    client = StubClient()
    decision = await run_quote(make_request(), client=client, index=index)

    by_node = {call.node: call.tier for call in decision.calls}
    assert by_node == {
        "intake": "trivial",
        "policy_rag": "standard",
        "discount_analyst": "complex",
        "explainer": "standard",
    }
    assert by_node["intake"] != by_node["discount_analyst"]
    # The approval router is deliberately absent: it calls no model.
    assert "approval_router" not in by_node
    assert len(client.calls) == 4


@pytest.mark.parametrize("discount_pct", [0.0, 5.0, 10.0, 10.5, 20.0, 22.0, 30.5, 41.0, 99.0])
async def test_every_decision_cites_a_chunk_that_resolves(index, discount_pct) -> None:
    """AC-2 swept across the bands, not asserted on one happy path."""
    decision = await run_quote(
        make_request(discount_pct=discount_pct), client=StubClient(), index=index
    )

    section = index.resolve(decision.citation)
    assert section is not None, f"{decision.citation} does not resolve"
    assert decision.citation_quote == section.text
    assert decision.escalated is (discount_pct > 10.0)


async def test_the_graph_topology_is_the_five_named_nodes_in_order(index) -> None:
    graph = build_graph(NodeContext(client=StubClient(), index=index))

    names = [name for name, _ in NODE_SEQUENCE]
    assert names == [
        "intake",
        "policy_rag",
        "discount_analyst",
        "approval_router",
        "explainer",
    ]
    # The compiled graph really contains them, not just our sequence constant.
    assert set(names) <= set(graph.get_graph().nodes)


async def test_state_accumulates_rather_than_overwrites(index) -> None:
    """Each node appends its call; a clobbering merge would lose the tier table."""
    ctx = NodeContext(client=StubClient(), index=index)
    graph = build_graph(ctx)

    final = await graph.ainvoke(QuoteState(request=make_request()))

    assert len(final["calls"]) == 4
    assert len(final["evidence"]) == ctx.retrieval_k
