"""Regression math and the baseline lifecycle."""

import json
from pathlib import Path

import pytest

from conduit.eval.gate import (
    Baseline,
    BaselineMissingError,
    baseline_from,
    compare,
    gate,
    load_baseline,
    write_baseline,
)
from conduit.eval.runner import CaseResult, SuiteResult

THRESHOLD = 0.05


def suite(pass_rate: float, *, name: str = "demo", cases: int = 200) -> SuiteResult:
    """A synthetic result with exactly the requested pass rate.

    200 cases so a rate like 0.855 is representable: rounding to a coarser grid
    would test the rounding, not the threshold.
    """
    passing = round(pass_rate * cases)
    return SuiteResult(
        suite=name,
        model_under_test="mock:echo",
        judge_model="offline-judge:rubric",
        cases=[
            CaseResult(
                case_id=f"c{index}",
                suite=name,
                expected="x",
                score=1.0 if index < passing else 0.0,
                passed=index < passing,
            )
            for index in range(cases)
        ],
    )


def baseline(pass_rate: float, *, name: str = "demo") -> Baseline:
    return Baseline(suite=name, pass_rate=pass_rate, mean_score=pass_rate, cases=20)


# --------------------------------------------------------------------------- #
# Regression arithmetic
# --------------------------------------------------------------------------- #


def test_a_drop_of_exactly_five_percent_does_not_regress() -> None:
    """ ">5%" is strict: the boundary passes, so the gate is not off by one."""
    outcomes = compare(suite(0.855), baseline(0.90), threshold=THRESHOLD)
    rate = next(o for o in outcomes if o.metric == "pass_rate")
    assert rate.relative_drop == pytest.approx(0.05)
    assert rate.regressed is False


def test_a_drop_past_five_percent_regresses() -> None:
    outcomes = compare(suite(0.80), baseline(0.90), threshold=THRESHOLD)
    rate = next(o for o in outcomes if o.metric == "pass_rate")
    assert rate.relative_drop == pytest.approx(0.1111, abs=1e-4)
    assert rate.regressed is True


def test_the_drop_is_relative_to_the_baseline_not_absolute_points() -> None:
    """0.90 -> 0.86 is 4.4% and passes; 0.40 -> 0.36 is 10% and fails.

    Both lost four points. Absolute points would let a weak suite rot while
    holding a strong one to a stricter standard than it was set at.
    """
    strong = next(
        o
        for o in compare(suite(0.86), baseline(0.90), threshold=THRESHOLD)
        if o.metric == "pass_rate"
    )
    weak = next(
        o
        for o in compare(suite(0.36), baseline(0.40), threshold=THRESHOLD)
        if o.metric == "pass_rate"
    )
    assert strong.regressed is False
    assert weak.regressed is True


def test_an_improvement_reports_a_negative_drop_and_never_regresses() -> None:
    outcome = next(
        o
        for o in compare(suite(0.95), baseline(0.80), threshold=THRESHOLD)
        if o.metric == "pass_rate"
    )
    assert outcome.relative_drop < 0
    assert outcome.regressed is False
    assert "ok" in outcome.describe()


def test_a_zero_baseline_cannot_be_regressed_against() -> None:
    """Nothing is below zero, so the drop is zero rather than an infinity."""
    outcome = next(
        o
        for o in compare(suite(0.0), baseline(0.0), threshold=THRESHOLD)
        if o.metric == "pass_rate"
    )
    assert outcome.relative_drop == 0.0
    assert outcome.regressed is False


def test_mean_score_is_gated_as_well_as_pass_rate() -> None:
    """A suite can rot on score while its binary pass rate holds."""
    result = suite(0.90)
    for case in result.cases:
        case.score = 0.5 if case.passed else 0.0
    outcomes = compare(
        result,
        Baseline(suite="demo", pass_rate=0.90, mean_score=0.90, cases=20),
        threshold=THRESHOLD,
    )
    assert next(o for o in outcomes if o.metric == "pass_rate").regressed is False
    assert next(o for o in outcomes if o.metric == "mean_score").regressed is True


# --------------------------------------------------------------------------- #
# Baseline lifecycle
# --------------------------------------------------------------------------- #


def test_a_baseline_round_trips_through_disk(tmp_path: Path) -> None:
    written = write_baseline(
        tmp_path / "demo.json", baseline_from(suite(0.75, cases=20), note="hello")
    )
    loaded = load_baseline(written)
    assert loaded.pass_rate == pytest.approx(0.75)
    assert loaded.cases == 20
    assert loaded.model_under_test == "mock:echo"
    assert loaded.note == "hello"
    # Committed baselines are diffable: sorted keys, indented, trailing newline.
    text = written.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert list(json.loads(text)) == sorted(json.loads(text))


def test_a_missing_baseline_explains_how_to_record_one(tmp_path: Path) -> None:
    with pytest.raises(BaselineMissingError, match="--update-baseline"):
        load_baseline(tmp_path / "absent.json")


def test_a_missing_baseline_fails_the_gate_rather_than_passing_by_default(
    tmp_path: Path,
) -> None:
    """An ungated suite that looks gated is worse than no gate at all."""
    report = gate([suite(1.0)], {"demo": tmp_path / "absent.json"}, threshold=THRESHOLD)
    assert report.missing_baselines == ["demo"]
    assert report.passed is False
    assert report.exit_code == 1


def test_the_gate_aggregates_across_suites_and_fails_on_any_one(tmp_path: Path) -> None:
    write_baseline(tmp_path / "good.json", baseline(0.90, name="good"))
    write_baseline(tmp_path / "bad.json", baseline(0.90, name="bad"))
    report = gate(
        [suite(0.92, name="good"), suite(0.70, name="bad")],
        {"good": tmp_path / "good.json", "bad": tmp_path / "bad.json"},
        threshold=THRESHOLD,
    )
    assert report.exit_code == 1
    assert {o.suite for o in report.regressions} == {"bad"}


def test_the_gate_passes_when_every_suite_holds(tmp_path: Path) -> None:
    write_baseline(tmp_path / "demo.json", baseline(0.90))
    report = gate([suite(0.90)], {"demo": tmp_path / "demo.json"}, threshold=THRESHOLD)
    assert report.passed is True
    assert report.exit_code == 0
    assert report.regressions == []
