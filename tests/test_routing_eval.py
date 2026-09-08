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


def test_tier_assignment_agrees_with_the_golden_labels(result: BenchmarkResult) -> None:
    assert result.accuracy >= ACCURACY_FLOOR, f"{result.accuracy:.1%} < {ACCURACY_FLOOR:.0%}"


def test_heuristic_resolves_the_golden_set_without_llm_calls(result: BenchmarkResult) -> None:
    assert result.heuristic_rate >= HEURISTIC_FLOOR
    assert result.tiebreaks == result.total - result.heuristic_resolved


async def test_the_eval_run_makes_zero_provider_calls() -> None:
    """AC-6 is only meaningful if nothing escalated behind our back."""
    provider = FakeProvider("complex")
    classifier = Classifier(provider=provider, tiebreak_model="mock:echo")
    for record in load_golden():
        await classifier.classify(record.to_request())
    assert provider.call_count == 0
    assert classifier.stats.heuristic_rate >= HEURISTIC_FLOOR


def test_routing_is_cheaper_than_sending_everything_to_complex(result: BenchmarkResult) -> None:
    assert result.baseline_cost_per_1k_usd > result.routed_cost_per_1k_usd
    assert result.savings_pct >= SAVINGS_FLOOR, f"{result.savings_pct:.1f}% < {SAVINGS_FLOOR}%"


def test_benchmark_report_states_its_numbers(result: BenchmarkResult) -> None:
    from conduit.router.benchmark import default_golden_path, render_markdown

    report = render_markdown(result, default_golden_path())
    assert f"{result.accuracy * 100:.1f}%" in report
    assert f"${result.baseline_cost_per_1k_usd:.2f}" in report
    assert f"{result.savings_pct:.1f}%" in report
