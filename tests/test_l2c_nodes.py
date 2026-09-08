"""Unit tests for the five L2C nodes, the arithmetic and the policy index.

Each node runs in isolation against a stubbed Conduit client. Retrieval is real
throughout — the index is cheap to build and the point of several of these tests
is that a citation resolves to an actual chunk (issue #9 AC-2), which a stubbed
retriever would make vacuous.
"""

import json

import pytest

from apps.l2c.models import PolicyEvidence
from apps.l2c.nodes import (
    NODE_WORKFLOWS,
    NodeContext,
    QuoteState,
    _ensure_cited,
    _extract_json,
    approval_router_node,
    compute_analysis,
    decide,
    discount_analyst_node,
    explainer_node,
    intake_node,
    policy_rag_node,
)
from apps.l2c.policies import (
    AUTO_APPROVE_MAX_PCT,
    DISCOUNT_BANDS,
    POLICY_SECTIONS,
    band_for,
    section_by_id,
)
from apps.l2c.rag import DEFAULT_DIMENSIONS, MAX_DIMENSIONS, HashingEmbedder, build_index
from tests.l2c_fixtures import StubClient, make_request


@pytest.fixture(scope="module")
def index():
    idx = build_index()
    yield idx
    idx.close()


@pytest.fixture
def ctx(index):
    return NodeContext(client=StubClient(), index=index)


# --------------------------------------------------------------------------- #
# Intake
# --------------------------------------------------------------------------- #


async def test_intake_keeps_the_parsed_request_authoritative(index) -> None:
    """A model that reports a different discount must not change the band."""
    lying = json.dumps({"quote_id": "Q-WRONG", "discount_pct": 5.0, "list_amount_usd": 1.0})
    ctx = NodeContext(client=StubClient(lying), index=index)
    request = make_request(discount_pct=26.0, list_amount_usd=840_000.0)

    update = await intake_node(QuoteState(request=request), ctx)
    intake = update["intake"]

    assert intake.discount_pct == 26.0
    assert intake.list_amount_usd == 840_000.0
    assert intake.quote_id == "Q-2026-00107"
    # The disagreement is reported, not silently absorbed.
    assert intake.model_confirmed is False


async def test_intake_confirms_when_the_model_echoes_the_quote_id(index) -> None:
    echo = "Sure! ```json\n" + json.dumps({"quote_id": "Q-2026-00107"}) + "\n```"
    ctx = NodeContext(client=StubClient(echo), index=index)

    update = await intake_node(QuoteState(request=make_request()), ctx)

    assert update["intake"].model_confirmed is True


async def test_intake_tags_the_trivial_workflow_and_records_the_call(ctx) -> None:
    update = await intake_node(QuoteState(request=make_request()), ctx)

    assert ctx.client.workflows == ["intake"]
    (call,) = update["calls"]
    assert call.node == "intake"
    assert call.workflow == NODE_WORKFLOWS["intake"]
    assert call.tier == "trivial"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('prose before {"a": {"b": 2}} prose after', {"a": {"b": 2}}),
        # A brace inside a string must not close the span early.
        ('{"a": "} not the end", "b": 3}', {"a": "} not the end", "b": 3}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        # Broken first candidate, valid second.
        ('{oops} then {"a": 1}', {"a": 1}),
        ("no json here", None),
        ("[1, 2, 3]", None),
    ],
)
def test_extract_json_scans_for_a_balanced_object(text, expected) -> None:
    assert _extract_json(text) == expected


# --------------------------------------------------------------------------- #
# Policy RAG
# --------------------------------------------------------------------------- #


async def test_policy_rag_retrieves_k_chunks_and_shows_them_to_the_model(index) -> None:
    ctx = NodeContext(client=StubClient(), index=index, retrieval_k=3)
    state = QuoteState(request=make_request())
    state.intake = (await intake_node(state, ctx))["intake"]

    update = await policy_rag_node(state, ctx)
    evidence = update["evidence"]

    assert len(evidence) == 3
    assert all(isinstance(item, PolicyEvidence) for item in evidence)
    # Every retrieved chunk is a real section, and the model saw its text.
    prompt = str(ctx.client.calls[-1]["user"])
    for item in evidence:
        assert index.resolve(item.section_id) is not None
        assert item.section_id in prompt
    assert ctx.client.workflows[-1] == "policy_rag"


async def test_policy_rag_scores_are_cosine_similarities(index) -> None:
    """`vec0` is built with distance_metric=cosine, so 1-distance is in -1..1."""
    evidence = index.search("regional vice president written approval threshold", k=4)

    assert evidence, "retrieval returned nothing"
    assert all(-1.0001 <= item.score <= 1.0001 for item in evidence)
    # Best first.
    assert [item.score for item in evidence] == sorted(
        (item.score for item in evidence), reverse=True
    )


# --------------------------------------------------------------------------- #
# Arithmetic and bands
# --------------------------------------------------------------------------- #


def test_compute_analysis_arithmetic() -> None:
    analysis = compute_analysis(
        make_request(discount_pct=26.0, list_amount_usd=840_000.0, partner_involved=True)
    )

    assert analysis.net_amount_usd == 621_600.0
    assert analysis.discount_value_usd == 218_400.0
    # 8% channel partner fee on net.
    assert analysis.partner_fee_usd == 49_728.0
    assert analysis.net_after_partner_fee_usd == 571_872.0
    assert analysis.auto_approvable is False


def test_compute_analysis_charges_no_partner_fee_without_a_partner() -> None:
    analysis = compute_analysis(make_request(discount_pct=8.0, partner_involved=False))

    assert analysis.partner_fee_usd == 0.0
    assert analysis.net_after_partner_fee_usd == analysis.net_amount_usd
    assert analysis.auto_approvable is True


@pytest.mark.parametrize(
    ("pct", "section_id"),
    [
        (0.0, "DISC-4.1"),
        (10.0, "DISC-4.1"),  # inclusive upper edge of the AE band
        (10.01, "DISC-4.2"),
        (20.0, "DISC-4.2"),
        (26.0, "DISC-4.3"),
        (30.0, "DISC-4.3"),
        (40.0, "DISC-4.4"),
        (55.0, "DISC-4.5"),
        (100.0, "DISC-4.5"),
    ],
)
def test_band_for_picks_the_narrowest_containing_band(pct, section_id) -> None:
    """DISC-4.0 says the narrowest band containing the discount is the one that applies."""
    assert band_for(pct).section_id == section_id


def test_every_band_cites_a_section_that_exists() -> None:
    for band in DISCOUNT_BANDS:
        assert section_by_id(band.section_id) is not None, band.section_id


def test_only_the_ae_band_is_auto_approvable() -> None:
    auto = [band for band in DISCOUNT_BANDS if band.auto_approvable]

    assert [band.section_id for band in auto] == ["DISC-4.1"]
    assert auto[0].high_pct == AUTO_APPROVE_MAX_PCT


# --------------------------------------------------------------------------- #
# Discount analyst
# --------------------------------------------------------------------------- #


async def test_discount_analyst_runs_on_the_complex_tier(ctx) -> None:
    update = await discount_analyst_node(QuoteState(request=make_request()), ctx)

    (call,) = update["calls"]
    assert call.workflow == NODE_WORKFLOWS["discount_analyst"] == "discount_analyst"
    assert call.tier == "complex"
    assert update["analysis"].narrative == "stub reply"


# --------------------------------------------------------------------------- #
# Approval router  (AC-2, AC-3)
# --------------------------------------------------------------------------- #


async def test_router_escalates_above_the_threshold_and_cites_a_real_chunk(ctx) -> None:
    request = make_request(discount_pct=26.0)
    state = QuoteState(request=request, analysis=compute_analysis(request))

    decision = (await approval_router_node(state, ctx))["decision"]

    assert decision.outcome == "escalate"
    assert decision.approver == "regional_vice_president"
    assert decision.citation == "DISC-4.3"
    # Asserted, not eyeballed: the citation resolves to an indexed chunk and the
    # quote the decision carries is that chunk's text.
    section = ctx.index.resolve(decision.citation)
    assert section is not None
    assert decision.citation_quote == section.text
    assert decision.citation_title == section.title


async def test_router_approves_below_the_threshold(ctx) -> None:
    request = make_request(discount_pct=8.0, partner_involved=False)
    state = QuoteState(request=request, analysis=compute_analysis(request))

    decision = (await approval_router_node(state, ctx))["decision"]

    assert decision.outcome == "approve"
    assert decision.escalated is False
    assert decision.citation == "DISC-4.1"
    assert ctx.index.resolve(decision.citation) is not None


async def test_router_makes_no_model_call(ctx) -> None:
    """The approval branch must not depend on a sampled token."""
    request = make_request()
    state = QuoteState(request=request, analysis=compute_analysis(request))

    await approval_router_node(state, ctx)

    assert ctx.client.calls == []


async def test_router_refuses_a_citation_the_index_cannot_resolve(index) -> None:
    """A corpus missing the band's clause is a bug, and must fail loudly."""
    without_rvp = [s for s in POLICY_SECTIONS if s.section_id != "DISC-4.3"]
    partial = build_index(without_rvp)
    try:
        ctx = NodeContext(client=StubClient(), index=partial)
        request = make_request(discount_pct=26.0)
        state = QuoteState(request=request, analysis=compute_analysis(request))

        with pytest.raises(RuntimeError, match="does not resolve"):
            await approval_router_node(state, ctx)
    finally:
        partial.close()


def test_decide_escalates_an_unjustified_exception() -> None:
    request = make_request(discount_pct=26.0, justification="   ")
    outcome, approver, reasons = decide(request, compute_analysis(request))

    assert outcome == "escalate"
    assert approver == "regional_vice_president"
    assert any("DISC-4.6" in reason for reason in reasons)


def test_ensure_cited_inserts_a_clause_retrieval_missed() -> None:
    unrelated = PolicyEvidence(section_id="PAY-2.1", doc_id="d", title="t", text="x", score=0.5)

    result = _ensure_cited([unrelated], "DISC-4.3")

    assert [item.section_id for item in result] == ["DISC-4.3", "PAY-2.1"]


def test_ensure_cited_leaves_evidence_alone_when_the_clause_is_present() -> None:
    present = PolicyEvidence(section_id="DISC-4.3", doc_id="d", title="t", text="x", score=0.5)

    assert _ensure_cited([present], "DISC-4.3") == [present]


# --------------------------------------------------------------------------- #
# Explainer
# --------------------------------------------------------------------------- #


async def test_explainer_narrates_the_decision_on_the_standard_tier(ctx) -> None:
    request = make_request()
    state = QuoteState(request=request, analysis=compute_analysis(request))
    state.decision = (await approval_router_node(state, ctx))["decision"]

    update = await explainer_node(state, ctx)

    (call,) = update["calls"]
    assert call.workflow == "explainer"
    assert call.tier == "standard"
    assert update["decision"].explanation
    # Explaining must not change the verdict or the citation.
    assert update["decision"].outcome == state.decision.outcome
    assert update["decision"].citation == state.decision.citation


# --------------------------------------------------------------------------- #
# Embedder and index
# --------------------------------------------------------------------------- #


def test_embedder_is_deterministic_across_instances() -> None:
    """BLAKE2b, not hash(): stable across processes and PYTHONHASHSEED."""
    text = "discounts exceeding twenty percent require written approval"

    assert HashingEmbedder().embed(text) == HashingEmbedder().embed(text)


def test_embedder_returns_unit_vectors_and_handles_empty_text() -> None:
    vector = HashingEmbedder().embed("regional vice president approval")

    assert len(vector) == DEFAULT_DIMENSIONS
    assert sum(value * value for value in vector) == pytest.approx(1.0)
    assert HashingEmbedder().embed("") == [0.0] * DEFAULT_DIMENSIONS


def test_embedder_rejects_a_width_vec0_would_refuse() -> None:
    with pytest.raises(ValueError, match="must be <="):
        HashingEmbedder(MAX_DIMENSIONS + 1)
    with pytest.raises(ValueError, match="positive"):
        HashingEmbedder(0)


def test_index_resolves_known_sections_and_rejects_unknown(index) -> None:
    assert index.resolve("DISC-4.3").section_id == "DISC-4.3"
    assert index.resolve("NOPE-9.9") is None
    assert len(index) == len(POLICY_SECTIONS)


def test_index_search_degenerate_inputs(index) -> None:
    assert index.search("anything", k=0) == []
    assert index.search("", k=4) == []
