"""The LangGraph: Intake -> Policy RAG -> Discount Analyst -> Approval Router -> Explainer.

The topology is linear, and deliberately so. A conditional edge here would be
decoration: the escalate/approve branch is a *value* the approval router
computes, not a fork in the control flow, and the explainer has to narrate
either outcome. Adding a branch that reconverges one node later would make the
diagram look busier without changing what runs.

LangGraph earns its place by owning state merging and giving the workflow a
compiled, inspectable graph object — `graph.get_graph().draw_ascii()` prints the
node topology, and each node's partial update is applied to the state for the
next one rather than by hand.

Nodes are bound to their `NodeContext` with closures rather than LangGraph's
config plumbing, so the collaborators stay statically typed and a node can be
called directly from a unit test with no graph at all.
"""

import itertools
from collections.abc import Awaitable, Callable, Sequence

from langgraph.graph import END, START, StateGraph

from .client import ConduitClient
from .models import Decision, PolicyEvidence, QuoteRequest
from .nodes import (
    NodeContext,
    QuoteState,
    approval_router_node,
    discount_analyst_node,
    explainer_node,
    intake_node,
    policy_rag_node,
)
from .rag import PolicyIndex, build_index

__all__ = ["NODE_SEQUENCE", "build_graph", "run_quote"]

NodeFn = Callable[[QuoteState, NodeContext], Awaitable[dict[str, object]]]

#: The graph's only path, in order. Tests assert the compiled graph matches it.
NODE_SEQUENCE: tuple[tuple[str, NodeFn], ...] = (
    ("intake", intake_node),
    ("policy_rag", policy_rag_node),
    ("discount_analyst", discount_analyst_node),
    ("approval_router", approval_router_node),
    ("explainer", explainer_node),
)


def build_graph(ctx: NodeContext) -> object:
    """Compile the quote-approval graph with `ctx` bound into every node."""
    builder: StateGraph = StateGraph(QuoteState)

    def bind(fn: NodeFn) -> Callable[[QuoteState], Awaitable[dict[str, object]]]:
        async def run(state: QuoteState) -> dict[str, object]:
            return await fn(state, ctx)

        return run

    for name, fn in NODE_SEQUENCE:
        builder.add_node(name, bind(fn))

    builder.add_edge(START, NODE_SEQUENCE[0][0])
    for (source, _), (target, _) in itertools.pairwise(NODE_SEQUENCE):
        builder.add_edge(source, target)
    builder.add_edge(NODE_SEQUENCE[-1][0], END)

    return builder.compile()


async def run_quote(
    request: QuoteRequest,
    *,
    client: ConduitClient,
    index: PolicyIndex | None = None,
    retrieval_k: int | None = None,
) -> Decision:
    """Run one quote request end to end and return the decision.

    Builds the policy index if one is not supplied, so the common case is a
    single call. Tests that want to assert on retrieval pass their own index.
    """
    owned_index = index is None
    resolved_index = index if index is not None else build_index()
    try:
        ctx = NodeContext(client=client, index=resolved_index)
        if retrieval_k is not None:
            ctx.retrieval_k = retrieval_k
        graph = build_graph(ctx)
        final = await graph.ainvoke(QuoteState(request=request))  # type: ignore[attr-defined]
        decision = _decision_from(final)
        if decision is None:  # pragma: no cover - the explainer always sets one
            raise RuntimeError("the graph finished without producing a decision")
        return decision
    finally:
        if owned_index:
            resolved_index.close()


def _decision_from(final: object) -> Decision | None:
    """Read the decision out of whatever shape LangGraph hands back.

    `ainvoke` returns a dict of the state fields for a dataclass schema, but a
    future version returning the state object itself would still work here.
    """
    value = final.get("decision") if isinstance(final, dict) else getattr(final, "decision", None)
    return value if isinstance(value, Decision) else None


def evidence_section_ids(evidence: Sequence[PolicyEvidence]) -> tuple[str, ...]:
    """Citation keys of retrieved evidence, in rank order."""
    return tuple(item.section_id for item in evidence)
