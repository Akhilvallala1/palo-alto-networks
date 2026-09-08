"""Eval suite: routing accuracy against `evals/golden/routing.jsonl`.

This is the CI-enforceable form of epic AC-4 (>=85% tier agreement), AC-6
(>=80% resolved with zero LLM calls) and AC-5 (>=40% cheaper than routing
everything to `complex`). The full eval plane — judges, baselines, the gate —
is issue #8; this suite is the router's own regression floor.
"""

import pytest

from conduit.config import load_config
from conduit.contracts import Complexity
from conduit.router.benchmark import BenchmarkResult, load_golden, run_benchmark
from conduit.router.classifier import Classifier
from conduit.router.policy import RoutingPolicy
from router_fakes import FakeProvider

ACCURACY_FLOOR = 0.85
HEURISTIC_FLOOR = 0.80
SAVINGS_FLOOR = 40.0


@pytest.fixture(scope="module")
def result() -> BenchmarkResult:
    config = load_config()
    records = load_golden()
    policy = RoutingPolicy(config.routing, config.models)
    return run_benchmark(policy, config.models, records)


def test_golden_set_is_well_formed_and_covers_every_tier() -> None:
    records = load_golden()
    assert len(records) >= 50
    assert len({record.id for record in records}) == len(records)
    covered = {record.expected for record in records}
    assert covered == set(Complexity)
    # No tier may dominate the set, or the cost number would be an artefact of
    # the sample rather than of the routing.
    for tier in Complexity:
        share = sum(1 for record in records if record.expected is tier) / len(records)
        assert 0.2 <= share <= 0.5


@pytest.mark.xfail(
    strict=True,
    reason=(
        "AC-4 is unmet on independently labelled data. Issue #8 re-authored "
        "evals/golden/routing.jsonl: the 60 inherited inputs were relabelled from the "
        "tier definitions in docs/SPEC.md (all 60 labels agreed with #4's) and 45 "
        "independently written cases were added whose surface form does not telegraph "
        "the tier. Agreement is 82.9% overall — 98.3% on the inherited slice, 62.2% on "
        "the new one. The floor stays at 85% rather than being lowered to the measured "
        "value; see docs/EVAL.md. Remove this marker when the classifier clears it."
    ),
)
def test_tier_assignment_agrees_with_the_golden_labels(result: BenchmarkResult) -> None:
    assert result.accuracy >= ACCURACY_FLOOR, f"{result.accuracy:.1%} < {ACCURACY_FLOOR:.0%}"


def test_tier_assignment_still_clears_the_floor_on_the_inherited_slice() -> None:
    """Localises the regression above: the drop is in the new cases, not the old.

    Without this the xfail would be indistinguishable from the classifier having
    broken outright.
    """
    config = load_config()
    records = [r for r in load_golden() if r.metadata.get("slice") == "canonical"]
    assert len(records) == 60
    result = run_benchmark(RoutingPolicy(config.routing, config.models), config.models, records)
    assert result.accuracy >= ACCURACY_FLOOR, f"{result.accuracy:.1%} < {ACCURACY_FLOOR:.0%}"


def test_heuristic_resolves_the_golden_set_without_llm_calls(result: BenchmarkResult) -> None:
    assert result.heuristic_rate >= HEURISTIC_FLOOR
    assert result.tiebreaks == result.total - result.heuristic_resolved


async def test_the_heuristic_resolves_most_cases_without_a_provider_call() -> None:
    """AC-6: >=80% of requests resolve with zero LLM tie-break calls.

    This asserted `call_count == 0` until issue #8 re-authored the golden set.
    On #4's own corpus every case cleared the confidence threshold, so "80% need
    no call" and "no call is ever made" happened to coincide and the stricter
    reading went unnoticed. The epic's criterion is the first one, so that is
    what is asserted: escalations are permitted, and are bounded by the floor.
    """
    provider = FakeProvider("complex")
    classifier = Classifier(provider=provider, tiebreak_model="mock:echo")
    records = load_golden()
    for record in records:
        await classifier.classify(record.to_request())
    assert classifier.stats.heuristic_rate >= HEURISTIC_FLOOR
    assert provider.call_count == len(records) - classifier.stats.heuristic_resolved


def test_routing_is_cheaper_than_sending_everything_to_complex(result: BenchmarkResult) -> None:
    assert result.baseline_cost_per_1k_usd > result.routed_cost_per_1k_usd
    assert result.savings_pct >= SAVINGS_FLOOR, f"{result.savings_pct:.1f}% < {SAVINGS_FLOOR}%"


def test_benchmark_report_states_its_numbers(result: BenchmarkResult) -> None:
    from conduit.router.benchmark import default_golden_path, render_markdown

    report = render_markdown(result, default_golden_path())
    assert f"{result.accuracy * 100:.1f}%" in report
    assert f"${result.baseline_cost_per_1k_usd:.2f}" in report
    assert f"{result.savings_pct:.1f}%" in report
