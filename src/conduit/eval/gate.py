"""The CI regression gate: compare a run to its committed baseline.

The gate is the whole point of the eval plane. A notebook someone ran once
produces a number; this produces a non-zero exit code, which is the only form
of quality signal a pull request can actually be blocked on.

Two decisions worth stating plainly.

*Regression is relative, not absolute.* A suite that drops from 0.90 to 0.86 has
lost 4.4% of its baseline and passes; one that drops from 0.40 to 0.37 has lost
7.5% and fails. Absolute points would let a weak suite rot unnoticed while
holding a strong one to a stricter standard than it was set at.

*Baselines are never written by a run.* `--update-baseline` is an explicit act
that shows up as a committed diff a reviewer can question (epic AC-7). A gate
that records its own baseline on green cannot fail twice for the same
regression, which makes it decorative.
"""

import json
import math
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .runner import SuiteResult

__all__ = [
    "Baseline",
    "BaselineMissingError",
    "GateOutcome",
    "GateReport",
    "compare",
    "gate",
    "load_baseline",
    "write_baseline",
]


class BaselineMissingError(FileNotFoundError):
    """Raised when a gated suite has no committed baseline."""


class Baseline(BaseModel):
    """A committed reference point for one suite."""

    model_config = ConfigDict(protected_namespaces=())

    suite: str
    pass_rate: float = Field(ge=0.0, le=1.0)
    mean_score: float = Field(ge=0.0, le=1.0)
    cases: int = Field(ge=0)
    model_under_test: str = ""
    judge_model: str = ""
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    note: str = ""


class GateOutcome(BaseModel):
    """One suite's verdict against its baseline."""

    suite: str
    metric: str
    baseline: float
    current: float
    #: Fraction of the baseline lost. Negative means the suite improved.
    relative_drop: float
    threshold: float
    regressed: bool

    def describe(self) -> str:
        direction = "regressed" if self.regressed else "ok"
        return (
            f"{self.suite}/{self.metric}: {self.baseline:.3f} -> {self.current:.3f} "
            f"({self.relative_drop:+.1%} of baseline) [{direction}]"
        )


class GateReport(BaseModel):
    outcomes: list[GateOutcome] = Field(default_factory=list)
    missing_baselines: list[str] = Field(default_factory=list)

    @property
    def regressions(self) -> list[GateOutcome]:
        return [outcome for outcome in self.outcomes if outcome.regressed]

    @property
    def passed(self) -> bool:
        return not self.regressions and not self.missing_baselines

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1


def baseline_from(result: SuiteResult, *, note: str = "") -> Baseline:
    return Baseline(
        suite=result.suite,
        pass_rate=result.pass_rate,
        mean_score=result.mean_score,
        cases=result.total,
        model_under_test=result.model_under_test,
        judge_model=result.judge_model,
        note=note,
    )


def load_baseline(path: Path) -> Baseline:
    target = Path(path)
    if not target.is_file():
        raise BaselineMissingError(
            f"no baseline at {target}. Record one with "
            "`conduit-eval run --suite <name> --update-baseline`."
        )
    return Baseline.model_validate(json.loads(target.read_text(encoding="utf-8")))


def write_baseline(path: Path, baseline: Baseline) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(baseline.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def _relative_drop(baseline: float, current: float) -> float:
    """Fraction of the baseline lost.

    A zero baseline cannot be regressed against — nothing is below it — so the
    drop is zero rather than an infinity that would fail the gate forever.
    """
    if baseline <= 0.0:
        return 0.0
    return (baseline - current) / baseline


def compare(result: SuiteResult, baseline: Baseline, *, threshold: float) -> list[GateOutcome]:
    """Both headline metrics are gated: a suite can rot on either one."""
    pairs = (
        ("pass_rate", baseline.pass_rate, result.pass_rate),
        ("mean_score", baseline.mean_score, result.mean_score),
    )
    outcomes: list[GateOutcome] = []
    for metric, before, after in pairs:
        drop = _relative_drop(before, after)
        outcomes.append(
            GateOutcome(
                suite=result.suite,
                metric=metric,
                baseline=before,
                current=after,
                relative_drop=drop,
                threshold=threshold,
                # ">5%" is strict, and a drop of exactly the threshold is a
                # common deliberate landing spot, so binary representation error
                # must not be what fails a build.
                regressed=drop > threshold and not math.isclose(drop, threshold, rel_tol=1e-9),
            )
        )
    return outcomes


def gate(
    results: list[SuiteResult],
    baseline_paths: dict[str, Path],
    *,
    threshold: float,
) -> GateReport:
    """Compare every result to its baseline. A missing baseline fails the gate."""
    report = GateReport()
    for result in results:
        path = baseline_paths.get(result.suite)
        if path is None:
            report.missing_baselines.append(result.suite)
            continue
        try:
            baseline = load_baseline(path)
        except BaselineMissingError:
            report.missing_baselines.append(result.suite)
            continue
        report.outcomes.extend(compare(result, baseline, threshold=threshold))
    return report
