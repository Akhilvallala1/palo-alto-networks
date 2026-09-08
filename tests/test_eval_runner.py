"""Score aggregation, concurrency capping, and error handling in the runner."""

import asyncio

import pytest

from conduit.eval.dataset import EvalCase
from conduit.eval.runner import CaseResult, SuiteResult, run_suite


def case(index: int, *, slice_name: str = "default") -> EvalCase:
    return EvalCase(
        id=f"c{index}",
        input=f"input {index}",
        expected="x",
        rubric="contains:x",
        metadata={"slice": slice_name},
    )


class StubTarget:
    """Grades case `n` according to a supplied plan."""

    def __init__(self, plan: dict[str, CaseResult], delay: float = 0.0) -> None:
        self.plan = plan
        self.delay = delay
        self.in_flight = 0
        self.peak = 0

    async def evaluate(self, evaluated: EvalCase) -> CaseResult:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            return self.plan[evaluated.id]
        finally:
            self.in_flight -= 1


def result(case_id: str, *, score: float, passed: bool, **kwargs: object) -> CaseResult:
    return CaseResult(
        case_id=case_id, suite="s", expected="x", score=score, passed=passed, **kwargs
    )


async def run(target: StubTarget, cases: list[EvalCase], concurrency: int = 4) -> SuiteResult:
    return await run_suite(
        "s",
        cases,
        target,
        concurrency=concurrency,
        model_under_test="mock:echo",
        judge_model="offline-judge:rubric",
    )


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


async def test_pass_rate_and_mean_score_are_computed_over_the_whole_suite() -> None:
    cases = [case(i) for i in range(4)]
    plan = {
        "c0": result("c0", score=1.0, passed=True),
        "c1": result("c1", score=1.0, passed=True),
        "c2": result("c2", score=0.4, passed=False),
        "c3": result("c3", score=0.0, passed=False),
    }
    suite = await run(StubTarget(plan), cases)
    assert suite.total == 4
    assert suite.passed == 2
    assert suite.pass_rate == pytest.approx(0.5)
    assert suite.mean_score == pytest.approx(0.6)


async def test_an_errored_case_counts_against_the_pass_rate_but_is_not_scored() -> None:
    """Epic AC-4: a case the judge could not read is an error, never a zero.

    Averaging in a fabricated zero would move the gate for a reason that is not
    quality, so errors are excluded from the mean and counted in the pass rate.
    """
    cases = [case(i) for i in range(2)]
    plan = {
        "c0": result("c0", score=1.0, passed=True),
        "c1": result("c1", score=0.0, passed=False, error="judge returned prose"),
    }
    suite = await run(StubTarget(plan), cases)
    assert suite.errors == 1
    assert suite.pass_rate == pytest.approx(0.5)
    assert suite.mean_score == pytest.approx(1.0)


async def test_cost_and_latency_percentiles_come_from_the_cases() -> None:
    cases = [case(i) for i in range(4)]
    plan = {
        f"c{i}": result(f"c{i}", score=1.0, passed=True, cost_usd=0.01, latency_ms=(i + 1) * 100)
        for i in range(4)
    }
    suite = await run(StubTarget(plan), cases)
    assert suite.total_cost_usd == pytest.approx(0.04)
    assert suite.p50_latency_ms == 200
    assert suite.p95_latency_ms == 400


async def test_slices_are_reported_separately() -> None:
    cases = [case(0, slice_name="a"), case(1, slice_name="a"), case(2, slice_name="b")]
    plan = {
        "c0": result("c0", score=1.0, passed=True),
        "c1": result("c1", score=0.0, passed=False),
        "c2": result("c2", score=1.0, passed=True),
    }
    suite = await run(StubTarget(plan), cases)
    assert {s.name: s.pass_rate for s in suite.slices()} == {"a": 0.5, "b": 1.0}


async def test_an_empty_suite_reports_zero_rather_than_dividing_by_zero() -> None:
    suite = SuiteResult(suite="s", model_under_test="m", judge_model="j")
    assert suite.pass_rate == 0.0
    assert suite.mean_score == 0.0
    assert suite.p95_latency_ms == 0


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


async def test_concurrency_is_capped_by_the_setting_not_the_dataset_size() -> None:
    cases = [case(i) for i in range(12)]
    plan = {f"c{i}": result(f"c{i}", score=1.0, passed=True) for i in range(12)}
    target = StubTarget(plan, delay=0.01)
    await run(target, cases, concurrency=3)
    assert target.peak <= 3


async def test_results_come_back_in_dataset_order_regardless_of_scheduling() -> None:
    cases = [case(i) for i in range(6)]
    plan = {f"c{i}": result(f"c{i}", score=1.0, passed=True) for i in range(6)}
    suite = await run(StubTarget(plan, delay=0.001), cases, concurrency=6)
    assert [c.case_id for c in suite.cases] == [c.id for c in cases]


async def test_a_target_that_raises_loses_one_case_not_the_whole_run() -> None:
    class Exploding:
        async def evaluate(self, evaluated: EvalCase) -> CaseResult:
            if evaluated.id == "c1":
                raise RuntimeError("boom")
            return result(evaluated.id, score=1.0, passed=True)

    suite = await run_suite(
        "s",
        [case(i) for i in range(3)],
        Exploding(),
        concurrency=2,
        model_under_test="m",
        judge_model="j",
    )
    assert suite.total == 3
    assert suite.errors == 1
    failed = next(c for c in suite.cases if c.errored)
    assert failed.error is not None and "RuntimeError: boom" in failed.error


async def test_concurrency_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        await run_suite(
            "s", [case(0)], StubTarget({}), concurrency=0, model_under_test="m", judge_model="j"
        )
