"""Measurement script behind epic AC-4/AC-5/AC-6.

Runs the heuristic classifier over `evals/golden/routing.jsonl`, then prices
the routed traffic against the "everything goes to the complex tier" baseline
using `config/models.yaml`. Nothing here calls a network: the point is that the
cost claim is reproducible on a laptop with no keys set.

    python -m conduit.router.benchmark                     # print the table
    python -m conduit.router.benchmark --write docs/BENCHMARKS.md

Token assumptions are explicit inputs (`--completion-tokens`), because a cost
number without its assumptions is a slide, not a measurement.
"""

import argparse
import contextlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from conduit.config import ConfigError, ModelsConfig, default_config_dir, load_config
from conduit.contracts import CompletionRequest, Complexity, Message, Usage

from .classifier import Classifier, estimate_tokens
from .policy import RoutingPolicy, price_usd
from .settings import ClassifierSettings, load_classifier_settings

__all__ = [
    "BenchmarkResult",
    "GoldenRecord",
    "TierRow",
    "default_golden_path",
    "load_golden",
    "main",
    "render_markdown",
    "run_benchmark",
]

DEFAULT_COMPLETION_TOKENS = 400


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_golden_path() -> Path:
    return _repo_root() / "evals" / "golden" / "routing.jsonl"


class GoldenRecord(BaseModel):
    """One labelled routing example. Schema matches the epic's golden format."""

    id: str
    input: str
    expected: Complexity
    tier: Complexity | None = None
    rubric: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)

    def to_request(self) -> CompletionRequest:
        return CompletionRequest(
            messages=[Message(role="user", content=self.input)],
            metadata=dict(self.metadata),
        )


def load_golden(path: Path | None = None) -> list[GoldenRecord]:
    """Read the golden JSONL, one record per non-empty line."""
    target = default_golden_path() if path is None else Path(path)
    if not target.is_file():
        raise ConfigError(f"missing golden dataset: {target}")
    records: list[GoldenRecord] = []
    for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{target}:{number}: invalid JSON: {exc}") from exc
        records.append(GoldenRecord.model_validate(payload))
    if not records:
        raise ConfigError(f"{target} contains no records")
    return records


class TierRow(BaseModel):
    """Per-tier routing volume and price, as routed (not as labelled)."""

    tier: Complexity
    model: str
    requests: int
    share: float
    cost_per_1k_usd: float


class BenchmarkResult(BaseModel):
    """Accuracy, heuristic coverage and the cost delta, in one object."""

    total: int
    correct: int
    heuristic_resolved: int
    tiebreaks: int
    completion_tokens: int
    rows: list[TierRow]
    routed_cost_per_1k_usd: float
    baseline_model: str
    baseline_cost_per_1k_usd: float
    confusion: dict[str, int]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def heuristic_rate(self) -> float:
        return self.heuristic_resolved / self.total if self.total else 0.0

    @property
    def savings_pct(self) -> float:
        if self.baseline_cost_per_1k_usd <= 0:
            return 0.0
        delta = self.baseline_cost_per_1k_usd - self.routed_cost_per_1k_usd
        return 100.0 * delta / self.baseline_cost_per_1k_usd


def run_benchmark(
    policy: RoutingPolicy,
    models: ModelsConfig,
    records: Sequence[GoldenRecord],
    *,
    settings: ClassifierSettings | None = None,
    completion_tokens: int = DEFAULT_COMPLETION_TOKENS,
) -> BenchmarkResult:
    """Classify every record heuristically and price the resulting mix.

    Stage 2 is deliberately not wired here: the benchmark measures what the
    zero-cost path achieves, and counts how often it would have escalated.
    """
    classifier = Classifier(settings or ClassifierSettings())
    threshold = classifier.settings.confidence_threshold

    correct = 0
    resolved = 0
    per_tier: dict[Complexity, list[int]] = {tier: [] for tier in Complexity}
    confusion: dict[str, int] = {}

    for record in records:
        verdict = classifier.heuristic(record.to_request())
        if verdict.complexity == record.expected:
            correct += 1
        if verdict.confidence >= threshold:
            resolved += 1
        confusion[f"{record.expected.value}->{verdict.complexity.value}"] = (
            confusion.get(f"{record.expected.value}->{verdict.complexity.value}", 0) + 1
        )
        per_tier[verdict.complexity].append(estimate_tokens(record.input))

    total = len(records)
    rows: list[TierRow] = []
    routed_total = 0.0
    baseline_model = policy.primary_for(Complexity.COMPLEX)
    baseline_total = 0.0

    for tier in Complexity:
        prompt_tokens = per_tier[tier]
        model = policy.primary_for(tier)
        cost = sum(
            price_usd(
                models.models[model],
                Usage(prompt_tokens=tokens, completion_tokens=completion_tokens, cost_usd=0.0),
            )
            for tokens in prompt_tokens
        )
        routed_total += cost
        rows.append(
            TierRow(
                tier=tier,
                model=model,
                requests=len(prompt_tokens),
                share=len(prompt_tokens) / total if total else 0.0,
                cost_per_1k_usd=1000 * cost / len(prompt_tokens) if prompt_tokens else 0.0,
            )
        )

    for record in records:
        baseline_total += price_usd(
            models.models[baseline_model],
            Usage(
                prompt_tokens=estimate_tokens(record.input),
                completion_tokens=completion_tokens,
                cost_usd=0.0,
            ),
        )

    return BenchmarkResult(
        total=total,
        correct=correct,
        heuristic_resolved=resolved,
        tiebreaks=total - resolved,
        completion_tokens=completion_tokens,
        rows=rows,
        routed_cost_per_1k_usd=1000 * routed_total / total if total else 0.0,
        baseline_model=baseline_model,
        baseline_cost_per_1k_usd=1000 * baseline_total / total if total else 0.0,
        confusion=confusion,
    )


def render_markdown(result: BenchmarkResult, golden: Path) -> str:
    """The table that lands in docs/BENCHMARKS.md."""
    with contextlib.suppress(ValueError):  # already relative, or outside the repo
        golden = golden.resolve().relative_to(_repo_root())
    lines = [
        "# Conduit Benchmarks",
        "",
        "> Generated by `python -m conduit.router.benchmark --write docs/BENCHMARKS.md`.",
        "> No network calls: the router's stage-1 classifier is a pure function and prices",
        "> come from `config/models.yaml`.",
        "",
        "## Routing accuracy and cost (issue #4)",
        "",
        f"- Dataset: `{golden.as_posix()}` — {result.total} labelled requests",
        f"- Assumed completion length: {result.completion_tokens} tokens; prompt tokens "
        "estimated at 4 characters per token",
        f"- Tier agreement with the golden labels: **{result.accuracy * 100:.1f}%** "
        "(epic AC-4 target: >=85%)",
        f"- Resolved by the heuristic alone, zero LLM tie-break calls: "
        f"**{result.heuristic_rate * 100:.1f}%** (epic AC-6 target: >=80%)",
        "",
        "### Cost per 1,000 requests",
        "",
        "| Tier | Primary model | Requests | Share | $/1k requests at this tier |",
        "|---|---|---:|---:|---:|",
    ]
    for row in result.rows:
        lines.append(
            f"| {row.tier.value} | `{row.model}` | {row.requests} | "
            f"{row.share * 100:.1f}% | ${row.cost_per_1k_usd:.2f} |"
        )
    lines += [
        "",
        "| Strategy | $/1k requests |",
        "|---|---:|",
        f"| All traffic to `complex` (`{result.baseline_model}`) | "
        f"${result.baseline_cost_per_1k_usd:.2f} |",
        f"| Conduit complexity routing | ${result.routed_cost_per_1k_usd:.2f} |",
        f"| **Reduction** | **{result.savings_pct:.1f}%** (epic AC-5 target: >=40%) |",
        "",
        "### Where the classifier disagrees with the labels",
        "",
        "| Labelled | Routed | Count |",
        "|---|---|---:|",
    ]
    for pair, count in sorted(result.confusion.items()):
        expected, routed = pair.split("->")
        marker = "" if expected == routed else " ⚠"
        lines.append(f"| {expected} | {routed}{marker} | {count} |")
    lines += [
        "",
        "### Method",
        "",
        "1. Every golden prompt is classified by `router.classifier.Classifier.heuristic`,",
        "   which performs no I/O.",
        "2. Each routed tier is priced at its chain primary from `config/routing.yaml`,",
        "   using the per-1M rates in `config/models.yaml`.",
        "3. The baseline prices the identical traffic at the `complex` tier primary.",
        "4. Latency and quality are not modelled here; the eval plane (#8) owns quality.",
        "",
        "### Caveats",
        "",
        "- The golden set was re-authored under issue #8 without reading the classifier's",
        "  lexicons, and extended with 45 cases whose surface form is uncorrelated with",
        "  tier, so agreement is now measured rather than self-graded. It fell from 100%",
        "  to 82.9% when the corpus stopped confirming its author; see `docs/EVAL.md`.",
        f"- Stage 2 escalated {result.tiebreaks} of {result.total} prompts here, so the cost",
        "  above is very nearly the pure stage-1 cost. Escalating traffic adds one",
        "  cheapest-tier call per distinct prompt hash, cached thereafter.",
        "- Prices are list rates from `config/models.yaml` with no prompt caching applied;",
        "  cache reads and writes are priced by `router.policy.price_usd` when they occur.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure routing accuracy and cost savings.")
    parser.add_argument("--golden", type=Path, default=None, help="path to routing.jsonl")
    parser.add_argument("--config-dir", type=Path, default=None, help="path to config/")
    parser.add_argument("--completion-tokens", type=int, default=DEFAULT_COMPLETION_TOKENS)
    parser.add_argument("--write", type=Path, default=None, help="write markdown to this path")
    parser.add_argument("--json", action="store_true", help="print the raw result as JSON")
    args = parser.parse_args(argv)

    config_dir = args.config_dir or default_config_dir()
    config = load_config(config_dir)
    settings = load_classifier_settings(config_dir)
    policy = RoutingPolicy(config.routing, config.models)
    golden_path = args.golden or default_golden_path()
    records = load_golden(golden_path)

    result = run_benchmark(
        policy,
        config.models,
        records,
        settings=settings,
        completion_tokens=args.completion_tokens,
    )

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(render_markdown(result, golden_path))

    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(render_markdown(result, golden_path), encoding="utf-8")
        print(f"\nwrote {args.write}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
