"""Reports: markdown for a human reading a PR, JSON for anything downstream.

Both carry total cost. An eval plane that measures the spend of the system
under test while hiding its own is an unbudgeted line item, and the one running
in CI on every pull request is exactly the one that grows quietly (epic AC-6).

Per-case pass/fail is in both formats, not just a summary line: "routing: 82.9%"
tells a reviewer nothing about which cases moved.
"""

import json
from datetime import UTC, datetime
from typing import Any

from .gate import GateReport
from .runner import SuiteResult

__all__ = ["render_json", "render_markdown", "summarise"]

_MAX_LISTED_FAILURES = 20


def summarise(result: SuiteResult) -> dict[str, Any]:
    """The per-suite headline numbers, in one place for both renderers."""
    return {
        "suite": result.suite,
        "model_under_test": result.model_under_test,
        "judge_model": result.judge_model,
        "cases": result.total,
        "passed": result.passed,
        "failed": result.total - result.passed,
        "errors": result.errors,
        "pass_rate": round(result.pass_rate, 4),
        "mean_score": round(result.mean_score, 4),
        "total_cost_usd": round(result.total_cost_usd, 6),
        "p50_latency_ms": result.p50_latency_ms,
        "p95_latency_ms": result.p95_latency_ms,
        "duration_ms": result.duration_ms,
        "slices": {stat.name: round(stat.pass_rate, 4) for stat in result.slices()},
    }


def render_json(results: list[SuiteResult], gate_report: GateReport | None = None) -> str:
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_cost_usd": round(sum(result.total_cost_usd for result in results), 6),
        "suites": [
            {
                **summarise(result),
                "cases_detail": [case.model_dump(mode="json") for case in result.cases],
            }
            for result in results
        ],
    }
    if gate_report is not None:
        payload["gate"] = {
            "passed": gate_report.passed,
            "exit_code": gate_report.exit_code,
            "missing_baselines": gate_report.missing_baselines,
            "outcomes": [outcome.model_dump(mode="json") for outcome in gate_report.outcomes],
        }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _escape(text: str, limit: int = 90) -> str:
    flat = " ".join(text.split())
    if len(flat) > limit:
        flat = flat[: limit - 1] + "…"
    return flat.replace("|", "\\|")


def render_markdown(results: list[SuiteResult], gate_report: GateReport | None = None) -> str:
    lines: list[str] = ["# Conduit eval report", ""]
    lines.append(f"Generated {datetime.now(UTC).isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("| Suite | Cases | Pass rate | Mean score | Errors | Cost | p50 | p95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        lines.append(
            f"| {result.suite} | {result.total} | {result.pass_rate:.1%} | "
            f"{result.mean_score:.3f} | {result.errors} | "
            f"${result.total_cost_usd:.4f} | {result.p50_latency_ms} ms | "
            f"{result.p95_latency_ms} ms |"
        )
    total_cost = sum(result.total_cost_usd for result in results)
    lines.append("")
    lines.append(f"**Total eval spend: ${total_cost:.4f}**")

    for result in results:
        lines.append("")
        lines.append(f"## {result.suite}")
        lines.append("")
        lines.append(
            f"Model under test `{result.model_under_test}`, judge `{result.judge_model}`, "
            f"{result.duration_ms} ms wall clock."
        )
        slices = result.slices()
        if len(slices) > 1:
            lines.append("")
            lines.append("| Slice | Cases | Pass rate |")
            lines.append("|---|---:|---:|")
            for stat in slices:
                lines.append(f"| {stat.name} | {stat.total} | {stat.pass_rate:.1%} |")
        failures = result.failures()
        lines.append("")
        if not failures:
            lines.append("All cases passed.")
            continue
        lines.append(f"{len(failures)} of {result.total} cases did not pass.")
        lines.append("")
        lines.append("| Case | Expected | Actual | Score | Note |")
        lines.append("|---|---|---|---:|---|")
        for case in failures[:_MAX_LISTED_FAILURES]:
            note = case.error or case.reasoning if case.errored else case.reasoning
            lines.append(
                f"| {case.case_id} | {_escape(case.expected, 40)} | "
                f"{_escape(case.actual, 40)} | {case.score:.2f} | {_escape(note)} |"
            )
        if len(failures) > _MAX_LISTED_FAILURES:
            lines.append("")
            lines.append(f"…and {len(failures) - _MAX_LISTED_FAILURES} more; see the JSON report.")

    if gate_report is not None:
        lines.append("")
        lines.append("## Gate")
        lines.append("")
        if gate_report.missing_baselines:
            missing = ", ".join(gate_report.missing_baselines)
            lines.append(f"No committed baseline for: {missing}. **Gate fails.**")
            lines.append("")
        lines.append("| Suite | Metric | Baseline | Current | Change | Verdict |")
        lines.append("|---|---|---:|---:|---:|---|")
        for outcome in gate_report.outcomes:
            verdict = "**REGRESSED**" if outcome.regressed else "ok"
            lines.append(
                f"| {outcome.suite} | {outcome.metric} | {outcome.baseline:.3f} | "
                f"{outcome.current:.3f} | {-outcome.relative_drop:+.1%} | {verdict} |"
            )
        lines.append("")
        lines.append("Gate **passed**." if gate_report.passed else "Gate **failed** (exit code 1).")
    return "\n".join(lines) + "\n"
