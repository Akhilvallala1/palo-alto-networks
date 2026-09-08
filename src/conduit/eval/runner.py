"""Suite execution: cases in, per-case results and suite statistics out.

Every suite runs through the same shape — produce an answer, score it — so the
report and the gate do arithmetic over one record type rather than three. What
differs per suite is the *target* (what produces the answer: the router, the
guard chain, or a model) and the *scorer* (what the answer is compared to).

Three deliberate choices:

- Concurrency is capped by a semaphore, not by the size of the dataset. An eval
  fans out across a shared gateway; unbounded `gather` over a hundred cases is
  a load test that happens to print a score.
- An unscoreable case is an error, not a zero. Errors count against the pass
  rate (they are not successes) but are excluded from the mean score, because
  averaging in a fabricated zero moves a gate for a reason that is not quality.
- Ordering is stable. Results come back in dataset order regardless of how the
  scheduler interleaved them, so two runs diff cleanly.
"""

import asyncio
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from conduit.config import ConduitConfig, load_config
from conduit.contracts import CompletionRequest, Complexity, Message, Provider
from conduit.guards import GuardChain, build_chain
from conduit.providers import ModelRegistry, build_registry
from conduit.router.classifier import Classifier
from conduit.router.settings import load_classifier_settings
from conduit.telemetry import percentile

from .dataset import EvalCase, load_cases
from .judge import Judge, JudgeError
from .offline import OFFLINE_JUDGE_MODEL, OfflineJudgeProvider
from .rubric import RubricError, parse_rubric
from .settings import EvalSettings, SuiteSpec

__all__ = [
    "CaseResult",
    "EvalHarness",
    "SliceStats",
    "SuiteResult",
    "Target",
    "run_suite",
]


class CaseResult(BaseModel):
    """One graded case. `error` and a score are mutually exclusive by design."""

    case_id: str
    suite: str
    slice: str = "default"
    expected: str
    actual: str = ""
    score: float = 0.0
    passed: bool = False
    reasoning: str = ""
    latency_ms: int = 0
    cost_usd: float = 0.0
    error: str | None = None

    @property
    def errored(self) -> bool:
        return self.error is not None


class SliceStats(BaseModel):
    name: str
    total: int
    passed: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


class SuiteResult(BaseModel):
    """Everything the report and the gate need about one suite run."""

    suite: str
    model_under_test: str
    judge_model: str
    ran_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: int = 0
    cases: list[CaseResult] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for case in self.cases if case.passed)

    @property
    def errors(self) -> int:
        return sum(1 for case in self.cases if case.errored)

    @property
    def pass_rate(self) -> float:
        """Errors are failures here: a suite that cannot run has not passed."""
        return self.passed / self.total if self.total else 0.0

    @property
    def mean_score(self) -> float:
        """Mean over scoreable cases only. Errors are counted, never scored."""
        scored = [case.score for case in self.cases if not case.errored]
        return sum(scored) / len(scored) if scored else 0.0

    @property
    def total_cost_usd(self) -> float:
        return sum(case.cost_usd for case in self.cases)

    def latency_percentile(self, p: float) -> int:
        return percentile(sorted(case.latency_ms for case in self.cases), p)

    @property
    def p50_latency_ms(self) -> int:
        return self.latency_percentile(50)

    @property
    def p95_latency_ms(self) -> int:
        return self.latency_percentile(95)

    def slices(self) -> list[SliceStats]:
        names = sorted({case.slice for case in self.cases})
        return [
            SliceStats(
                name=name,
                total=sum(1 for case in self.cases if case.slice == name),
                passed=sum(1 for case in self.cases if case.slice == name and case.passed),
            )
            for name in names
        ]

    def failures(self) -> list[CaseResult]:
        return [case for case in self.cases if not case.passed]


class Target(Protocol):
    """Runs one case and returns its graded result."""

    async def evaluate(self, case: EvalCase) -> CaseResult: ...


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #


class RoutingTarget:
    """Tier agreement. Calls the classifier only, so it is free and offline."""

    def __init__(self, suite: str, classifier: Classifier) -> None:
        self.suite = suite
        self.classifier = classifier

    async def evaluate(self, case: EvalCase) -> CaseResult:
        started = time.perf_counter()
        try:
            expected = Complexity(case.expected)
        except ValueError:
            return CaseResult(
                case_id=case.id,
                suite=self.suite,
                slice=case.slice,
                expected=case.expected,
                error=f"{case.expected!r} is not a complexity tier",
            )
        classification = await self.classifier.classify(case.to_request())
        actual = classification.complexity.value
        agreed = classification.complexity is expected
        return CaseResult(
            case_id=case.id,
            suite=self.suite,
            slice=case.slice,
            expected=case.expected,
            actual=actual,
            score=1.0 if agreed else 0.0,
            passed=agreed,
            reasoning=(
                f"classified {actual} via {classification.source} "
                f"(confidence {classification.confidence:.2f})"
            ),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


class GuardTarget:
    """Guard-chain decision accuracy: block, allow, or allow-but-redacted."""

    DECISIONS = ("block", "allow", "redact")

    def __init__(self, suite: str, chain: GuardChain) -> None:
        self.suite = suite
        self.chain = chain

    async def evaluate(self, case: EvalCase) -> CaseResult:
        started = time.perf_counter()
        if case.expected not in self.DECISIONS:
            return CaseResult(
                case_id=case.id,
                suite=self.suite,
                slice=case.slice,
                expected=case.expected,
                error=f"{case.expected!r} is not one of {list(self.DECISIONS)}",
            )
        verdict = await self.chain.inspect(case.input)
        redacted = verdict.redacted_text is not None and bool(verdict.entity_map)
        if not verdict.allowed:
            actual = "block"
        elif redacted:
            actual = "redact"
        else:
            actual = "allow"
        # A `redact` case is satisfied by any outcome that let the request
        # through with the PII replaced; an `allow` case is satisfied by any
        # outcome that let it through at all, redacted or not, because
        # redacting a benign name is not a false block.
        if case.expected == "redact":
            correct = verdict.allowed and redacted
        elif case.expected == "allow":
            correct = verdict.allowed
        else:
            correct = not verdict.allowed
        return CaseResult(
            case_id=case.id,
            suite=self.suite,
            slice=case.slice,
            expected=case.expected,
            actual=actual,
            score=1.0 if correct else 0.0,
            passed=correct,
            reasoning=(
                f"risk={verdict.risk_score:.2f} categories={verdict.categories or ['none']}"
            ),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


class JudgedTarget:
    """Answer quality: call the model under test, then grade with the judge."""

    def __init__(
        self,
        suite: str,
        provider: Provider,
        model: str,
        judge: Judge,
        *,
        pass_threshold: float,
        max_tokens: int = 512,
    ) -> None:
        self.suite = suite
        self.provider = provider
        self.model = model
        self.judge = judge
        self.pass_threshold = pass_threshold
        self.max_tokens = max_tokens

    async def evaluate(self, case: EvalCase) -> CaseResult:
        started = time.perf_counter()
        base = CaseResult(
            case_id=case.id, suite=self.suite, slice=case.slice, expected=case.expected
        )
        try:
            parse_rubric(case.rubric)
        except RubricError as exc:
            return base.model_copy(update={"error": f"bad rubric: {exc}"})
        request = CompletionRequest(
            messages=[Message(role="user", content=case.input)],
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
        )
        try:
            response = await self.provider.complete(request, self.model)
        except Exception as exc:
            return base.model_copy(
                update={
                    "error": f"model under test failed: {exc}",
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                }
            )
        try:
            verdict = await self.judge.score(
                case, response.text, pass_threshold=self.pass_threshold
            )
        except (JudgeError, RubricError) as exc:
            return base.model_copy(
                update={
                    "actual": response.text,
                    "cost_usd": response.usage.cost_usd,
                    "error": str(exc),
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                }
            )
        return base.model_copy(
            update={
                "actual": response.text,
                "score": verdict.score.score,
                "passed": verdict.score.passed,
                "reasoning": verdict.score.reasoning,
                "cost_usd": response.usage.cost_usd + verdict.cost_usd,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
        )


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


async def run_suite(
    name: str,
    cases: Sequence[EvalCase],
    target: Target,
    *,
    concurrency: int,
    model_under_test: str,
    judge_model: str,
) -> SuiteResult:
    """Run every case through `target`, at most `concurrency` at a time."""
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")
    semaphore = asyncio.Semaphore(concurrency)
    started = time.perf_counter()

    async def one(case: EvalCase) -> CaseResult:
        async with semaphore:
            try:
                result = await target.evaluate(case)
                # The case owns its slice, not the target: a target that forgot
                # to copy it would collapse the per-slice report into one number,
                # which is exactly where the provenance signal lives.
                return result.model_copy(update={"slice": case.slice})
            except Exception as exc:  # a target bug must not lose the whole run
                return CaseResult(
                    case_id=case.id,
                    suite=name,
                    slice=case.slice,
                    expected=case.expected,
                    error=f"{type(exc).__name__}: {exc}",
                )

    results = await asyncio.gather(*(one(case) for case in cases))
    return SuiteResult(
        suite=name,
        model_under_test=model_under_test,
        judge_model=judge_model,
        duration_ms=int((time.perf_counter() - started) * 1000),
        cases=list(results),
    )


class EvalHarness:
    """Builds the target a suite needs and runs it.

    Wiring lives here rather than in the CLI so a test can construct the
    harness with an injected provider and get the same code path CI runs.
    """

    def __init__(
        self,
        settings: EvalSettings,
        *,
        config: ConduitConfig | None = None,
        registry: ModelRegistry | None = None,
        judge_provider: Provider | None = None,
        root: Path | None = None,
    ) -> None:
        self.settings = settings
        self.config = config if config is not None else load_config()
        self._registry = registry
        self._judge_provider = judge_provider
        self.root = root

    @property
    def registry(self) -> ModelRegistry:
        if self._registry is None:
            self._registry = build_registry(models=self.config.models)
        return self._registry

    def judge_provider(self) -> Provider:
        if self._judge_provider is not None:
            return self._judge_provider
        if self.settings.judge_model == OFFLINE_JUDGE_MODEL:
            self._judge_provider = OfflineJudgeProvider()
        else:
            self._judge_provider = self.registry.provider_for(self.settings.judge_model)
        return self._judge_provider

    def build_target(self, name: str, spec: SuiteSpec) -> Target:
        if spec.scorer == "tier":
            return RoutingTarget(name, Classifier(load_classifier_settings()))
        if spec.scorer == "guard":
            return GuardTarget(name, build_chain(self.config.guards))
        judge = Judge(
            self.judge_provider(),
            self.settings.judge_model,
            retries=self.settings.judge_retries,
        )
        return JudgedTarget(
            name,
            self.registry.provider_for(self.settings.model_under_test),
            self.settings.model_under_test,
            judge,
            pass_threshold=spec.pass_threshold,
        )

    async def run(self, name: str) -> SuiteResult:
        spec = self.settings.suite(name)
        cases = load_cases(self.settings.resolve_path(spec, self.root))
        return await run_suite(
            name,
            cases,
            self.build_target(name, spec),
            concurrency=self.settings.concurrency,
            model_under_test=self.settings.model_under_test,
            judge_model=self.settings.judge_model,
        )

    async def run_all(self, names: Sequence[str] | None = None) -> list[SuiteResult]:
        selected = list(names) if names else sorted(self.settings.suites)
        return [await self.run(name) for name in selected]
